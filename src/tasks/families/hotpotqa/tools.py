"""Wikipedia tools for the HotPotQA family.

Three tools close over a shared :class:`WikipediaCache`. On cache hit
the tools return the stored response without touching the network;
on cache miss they issue a live HTTP request, record the response,
and store it. Rate limiting and exponential backoff apply to live
calls only — cache hits are free.

The tools are intentionally defensive: Wikipedia's REST API returns
404s for pages that have been moved or renamed, and the
``error_recovery`` condition depends on that 404 being surfaced to
the agent as a structured ``{"error": "not_found", ...}`` payload
rather than raised as an exception.
"""

from __future__ import annotations

import random
import time
from typing import Any

import requests

from ....agent.tools import ToolSpec
from .env import WikipediaCache


# Minimum gap (seconds) between real HTTP requests.
_MIN_INTERVAL_S = 0.1

# Exponential backoff parameters for 429 / 5xx responses.
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 0.5

# Truncation ceiling for page bodies — keeps prompt context reasonable.
_MAX_TEXT_CHARS = 2000
_SEARCH_RESULTS = 5

_WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

_USER_AGENT = (
    "SpeculativeDecodingAgent/0.1 (research; kyrylldekanenko@gmail.com)"
)


class RateLimiter:
    """Simple monotonic-clock rate limiter.

    Attributes:
        min_interval: Minimum seconds between successive
            :meth:`wait` returns.
        sleeper: Callable used to sleep. Publicly accessible so HTTP
            retry code can share the same fake clock in tests.
    """

    def __init__(
        self,
        min_interval: float = _MIN_INTERVAL_S,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        """Initialise the limiter.

        Args:
            min_interval: Minimum gap between calls, in seconds.
            clock: Callable returning a monotonic timestamp.
            sleeper: Callable that sleeps for the given duration.
        """
        self.min_interval = min_interval
        self.sleeper = sleeper
        self._clock = clock
        self._last_call: float | None = None

    def wait(self) -> None:
        """Block until enough time has elapsed since the last call."""
        now = self._clock()
        if self._last_call is not None:
            elapsed = now - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                self.sleeper(remaining)
                now = self._clock()
        self._last_call = now


def _backoff_delay(attempt: int) -> float:
    """Return the backoff delay for a given retry attempt.

    Args:
        attempt: Zero-based retry index.

    Returns:
        Exponential delay with jitter, in seconds.
    """
    jitter = random.uniform(0.0, 0.1)
    return _BACKOFF_BASE_S * (2**attempt) + jitter


def _http_get_with_retry(
    url: str,
    params: dict[str, Any] | None,
    limiter: RateLimiter,
) -> requests.Response:
    """Issue a rate-limited GET with exponential backoff.

    Args:
        url: Request URL.
        params: Query-string parameters, or ``None``.
        limiter: Rate limiter controlling request cadence. Its
            ``sleeper`` attribute is also used for backoff sleeps so
            tests can share a single fake clock.

    Returns:
        The final :class:`requests.Response` object. 404 is returned
        directly (not retried); 429 / 5xx are retried up to
        ``_MAX_RETRIES`` times before the last response is returned.
    """
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    last: requests.Response | None = None
    for attempt in range(_MAX_RETRIES + 1):
        limiter.wait()
        response = requests.get(url, params=params, headers=headers, timeout=30)
        last = response
        if response.status_code == 404:
            return response
        if response.status_code < 400:
            return response
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt < _MAX_RETRIES:
                limiter.sleeper(_backoff_delay(attempt))
                continue
        return response
    assert last is not None
    return last


def _search_live(query: str, limiter: RateLimiter) -> dict[str, Any]:
    """Fetch search results from Wikipedia's opensearch endpoint.

    Args:
        query: Free-text search query.
        limiter: Shared rate limiter.

    Returns:
        ``{"results": [{"title": str, "snippet": str}, ...]}``. On
        failure, returns ``{"results": [], "error": "<tag>"}``.
    """
    params = {
        "action": "opensearch",
        "search": query,
        "limit": _SEARCH_RESULTS,
        "namespace": 0,
        "format": "json",
    }
    response = _http_get_with_retry(_WIKI_API_URL, params, limiter)
    if response.status_code >= 400:
        return {"results": [], "error": f"http_{response.status_code}"}
    try:
        payload = response.json()
    except ValueError:
        return {"results": [], "error": "invalid_json"}
    # opensearch shape: [query, [titles], [snippets], [urls]]
    titles = payload[1] if len(payload) > 1 else []
    snippets = payload[2] if len(payload) > 2 else []
    results = []
    for i, title in enumerate(titles):
        snippet = snippets[i] if i < len(snippets) else ""
        results.append({"title": title, "snippet": snippet})
    return {"results": results}


def _page_live(title: str, limiter: RateLimiter) -> dict[str, Any]:
    """Fetch a Wikipedia page summary + extract.

    Args:
        title: Page title.
        limiter: Shared rate limiter.

    Returns:
        ``{"title", "summary", "text"}`` on success,
        ``{"error": "not_found", "title": ...}`` on 404.
    """
    summary_url = _WIKI_SUMMARY_URL + requests.utils.quote(title, safe="")
    response = _http_get_with_retry(summary_url, None, limiter)
    if response.status_code == 404:
        return {"error": "not_found", "title": title}
    if response.status_code >= 400:
        return {
            "error": f"http_{response.status_code}",
            "title": title,
        }
    try:
        summary_payload = response.json()
    except ValueError:
        return {"error": "invalid_json", "title": title}
    summary = summary_payload.get("extract", "") or ""

    # Fetch the longer plaintext extract for body context.
    extract_params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    extract_response = _http_get_with_retry(_WIKI_API_URL, extract_params, limiter)
    text = ""
    if extract_response.status_code < 400:
        try:
            data = extract_response.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if "extract" in page:
                    text = page["extract"] or ""
                    break
        except ValueError:
            text = ""
    text = text[:_MAX_TEXT_CHARS]
    return {
        "title": summary_payload.get("title", title),
        "summary": summary,
        "text": text,
    }


def build_wikipedia_tools(
    cache: WikipediaCache,
    limiter: RateLimiter | None = None,
) -> list[ToolSpec]:
    """Build the HotPotQA tool set closed over a cache and limiter.

    Args:
        cache: The per-task :class:`WikipediaCache`.
        limiter: Optional rate limiter. When ``None`` a default
            in-process limiter is created. Shared by ``search`` and
            ``get_page`` so they throttle each other.

    Returns:
        A list of three :class:`ToolSpec` instances:
        ``search_wikipedia``, ``get_wiki_page``, ``finish``.
    """
    rl = limiter if limiter is not None else RateLimiter()

    def _search(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"results": [], "error": "empty_query"}
        cached = cache.get("search_wikipedia", {"query": query})
        if cached is not None:
            return cached
        result = _search_live(query, rl)
        cache.set("search_wikipedia", {"query": query}, result)
        return result

    def _page(args: dict[str, Any]) -> dict[str, Any]:
        title = str(args.get("title", "")).strip()
        if not title:
            return {"error": "empty_title"}
        cached = cache.get("get_wiki_page", {"title": title})
        if cached is not None:
            return cached
        result = _page_live(title, rl)
        cache.set("get_wiki_page", {"title": title}, result)
        return result

    def _finish(args: dict[str, Any]) -> dict[str, Any]:
        answer = str(args.get("answer", "")).strip()
        return {"done": True, "answer": answer}

    return [
        ToolSpec(
            name="search_wikipedia",
            description=(
                "Search Wikipedia for page titles matching a query. "
                "Returns up to 5 {title, snippet} results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query.",
                    },
                },
                "required": ["query"],
            },
            fn=_search,
        ),
        ToolSpec(
            name="get_wiki_page",
            description=(
                "Fetch a Wikipedia page by exact title. Returns "
                "{title, summary, text}. If the page does not exist, "
                "returns {error: 'not_found', title}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact Wikipedia page title.",
                    },
                },
                "required": ["title"],
            },
            fn=_page,
        ),
        ToolSpec(
            name="finish",
            description=(
                "Submit the final short answer to the user's question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Concise final answer.",
                    },
                },
                "required": ["answer"],
            },
            fn=_finish,
        ),
    ]
