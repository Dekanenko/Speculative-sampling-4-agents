"""Tests for the HotPotQA family.

All HTTP is mocked via ``responses``. Nothing in this file touches
the live Wikipedia API.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
import responses

from src.tasks.families.base import EvaluationResult
from src.tasks.families.hotpotqa import (
    HotpotqaFamily,
    RateLimiter,
    WikipediaCache,
    build_wikipedia_tools,
    exact_match,
    extract_predicted_answer,
    token_f1,
)
from src.tasks.families.hotpotqa.evaluator import normalise_answer
from src.tasks.families.hotpotqa.loader import split_dir
from src.tasks.families.hotpotqa.tools import (
    _WIKI_API_URL,
    _WIKI_SUMMARY_URL,
)
from src.tasks.schema import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_step(
    role: str = "assistant",
    text: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
) -> types.SimpleNamespace:
    """Build a duck-typed step satisfying what the evaluator reads."""
    return types.SimpleNamespace(role=role, text=text, tool_calls=tool_calls)


def _fake_trajectory(steps: list[types.SimpleNamespace]) -> types.SimpleNamespace:
    """Wrap steps into a minimal trajectory-shaped object."""
    return types.SimpleNamespace(steps=steps)


def _make_task(task_id: str = "t1", gold_answer: str = "Paris") -> Task:
    """Build a :class:`Task` pointed at the HotPotQA family."""
    return Task(
        task_id=task_id,
        condition="simple",
        system_prompt="sys",
        user_prompt="usr",
        allowed_tools=["search_wikipedia", "get_wiki_page", "finish"],
        family="hotpotqa",
        expected={"answer": gold_answer, "supporting_titles": []},
    )


# ---------------------------------------------------------------------------
# WikipediaCache
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "t1.json"
    cache = WikipediaCache(path)
    assert cache.get("search_wikipedia", {"query": "x"}) is None
    cache.set("search_wikipedia", {"query": "x"}, {"results": [{"title": "X"}]})
    cache.save()

    loaded = WikipediaCache(path)
    got = loaded.get("search_wikipedia", {"query": "x"})
    assert got == {"results": [{"title": "X"}]}

    # Mutating the returned object must not affect the stored entry.
    got["results"].append({"title": "Y"})
    assert loaded.get("search_wikipedia", {"query": "x"}) == {
        "results": [{"title": "X"}]
    }


def test_cache_refresh_bypasses_lookup(tmp_path: Path) -> None:
    path = tmp_path / "t2.json"
    cache = WikipediaCache(path)
    cache.set("get_wiki_page", {"title": "A"}, {"text": "body"})
    cache.save()

    fresh = WikipediaCache(path, refresh=True)
    assert fresh.get("get_wiki_page", {"title": "A"}) is None
    assert fresh.has("get_wiki_page", {"title": "A"}) is True


def test_cache_save_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "t3.json"
    cache = WikipediaCache(path)
    cache.set("search_wikipedia", {"query": "q"}, {"results": []})
    cache.save()
    first_mtime = path.stat().st_mtime_ns

    # Saving again without modifications should not touch the file.
    cache.save()
    assert path.stat().st_mtime_ns == first_mtime

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert list(stored.keys()) == ['search_wikipedia::{"query": "q"}']


# ---------------------------------------------------------------------------
# Tools — HTTP-mocked end-to-end
# ---------------------------------------------------------------------------


def _mk_opensearch_response(query: str, titles: list[str]) -> list[Any]:
    """Build a fake opensearch payload."""
    snippets = [f"snip for {t}" for t in titles]
    urls = [f"https://en.wikipedia.org/wiki/{t}" for t in titles]
    return [query, titles, snippets, urls]


@responses.activate
def test_search_wikipedia_hits_cache_on_second_call(tmp_path: Path) -> None:
    responses.add(
        responses.GET,
        _WIKI_API_URL,
        json=_mk_opensearch_response("python", ["Python (programming language)"]),
        status=200,
    )
    cache = WikipediaCache(tmp_path / "c.json")
    tools = {t.name: t for t in build_wikipedia_tools(cache, RateLimiter(min_interval=0))}
    search = tools["search_wikipedia"].fn

    first = search({"query": "python"})
    assert first["results"][0]["title"] == "Python (programming language)"

    # Second call: no new HTTP request should fire.
    second = search({"query": "python"})
    assert second == first
    assert len(responses.calls) == 1


@responses.activate
def test_get_wiki_page_404_returns_not_found(tmp_path: Path) -> None:
    title = "Totally_Nonexistent_xyzq"
    responses.add(
        responses.GET,
        _WIKI_SUMMARY_URL + title,
        json={"title": title, "type": "https://mediawiki.org/wiki/HyperSwitch/errors/not_found"},
        status=404,
    )
    cache = WikipediaCache(tmp_path / "c.json")
    tools = {t.name: t for t in build_wikipedia_tools(cache, RateLimiter(min_interval=0))}
    result = tools["get_wiki_page"].fn({"title": title})
    assert result == {"error": "not_found", "title": title}

    # Negative results must be cached too so replay is deterministic.
    cache.save()
    reloaded = WikipediaCache(tmp_path / "c.json")
    assert reloaded.get("get_wiki_page", {"title": title}) == result


@responses.activate
def test_get_wiki_page_success_returns_title_summary_text(tmp_path: Path) -> None:
    title = "Albert_Einstein"
    responses.add(
        responses.GET,
        _WIKI_SUMMARY_URL + title,
        json={"title": "Albert Einstein", "extract": "Physicist (1879-1955)."},
        status=200,
    )
    long_text = "x" * 4000
    responses.add(
        responses.GET,
        _WIKI_API_URL,
        json={"query": {"pages": {"42": {"extract": long_text}}}},
        status=200,
    )
    cache = WikipediaCache(tmp_path / "c.json")
    tools = {t.name: t for t in build_wikipedia_tools(cache, RateLimiter(min_interval=0))}
    result = tools["get_wiki_page"].fn({"title": title})
    assert result["title"] == "Albert Einstein"
    assert "Physicist" in result["summary"]
    # Text truncated to <=2000 chars.
    assert len(result["text"]) == 2000


def test_finish_tool_is_pure() -> None:
    cache = WikipediaCache(Path("/tmp/_unused_cache_path.json"))
    tools = {t.name: t for t in build_wikipedia_tools(cache)}
    out = tools["finish"].fn({"answer": "  Paris  "})
    assert out == {"done": True, "answer": "Paris"}


# ---------------------------------------------------------------------------
# Rate limiter + 429 backoff
# ---------------------------------------------------------------------------


class _FakeClock:
    """Monotonic fake clock controlled by ``sleep`` calls."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_waits_between_calls() -> None:
    clock = _FakeClock()
    limiter = RateLimiter(min_interval=0.1, clock=clock.time, sleeper=clock.sleep)
    limiter.wait()
    assert clock.sleeps == []  # first call: no waiting
    clock.now += 0.02  # 20 ms elapsed
    limiter.wait()
    # Should have slept ~0.08 seconds to enforce the 100 ms gap.
    assert clock.sleeps and abs(clock.sleeps[0] - 0.08) < 1e-9


@responses.activate
def test_search_backs_off_on_429(tmp_path: Path) -> None:
    # First call 429, second call 200.
    responses.add(
        responses.GET,
        _WIKI_API_URL,
        json={"error": "rate limited"},
        status=429,
    )
    responses.add(
        responses.GET,
        _WIKI_API_URL,
        json=_mk_opensearch_response("cats", ["Cat"]),
        status=200,
    )
    clock = _FakeClock()
    limiter = RateLimiter(min_interval=0, clock=clock.time, sleeper=clock.sleep)
    cache = WikipediaCache(tmp_path / "c.json")
    tools = {t.name: t for t in build_wikipedia_tools(cache, limiter)}
    out = tools["search_wikipedia"].fn({"query": "cats"})
    assert out["results"][0]["title"] == "Cat"
    assert len(responses.calls) == 2
    # At least one backoff sleep must have happened between the two HTTP calls.
    assert any(s > 0 for s in clock.sleeps)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def test_normalise_answer_strips_articles_and_punctuation() -> None:
    assert normalise_answer("The United States.") == "united states"


def test_exact_match_and_f1_on_perfect_answer() -> None:
    assert exact_match("Paris", "paris") == 1.0
    assert token_f1("Paris", "paris") == 1.0


def test_f1_partial_overlap() -> None:
    # "Barack Hussein Obama" vs "Barack Obama": overlap 2/3 predicted tokens,
    # 2/2 gold tokens → F1 = 2 * (2/3 * 1) / (2/3 + 1) = 0.8.
    f1 = token_f1("Barack Hussein Obama", "Barack Obama")
    assert 0.0 < f1 < 1.0


def test_evaluate_perfect_finish() -> None:
    fam = HotpotqaFamily()
    task = _make_task(gold_answer="Paris")
    traj = _fake_trajectory(
        [
            _fake_step(
                role="assistant",
                tool_calls=[{"name": "finish", "arguments": {"answer": "Paris"}}],
            )
        ]
    )
    result = fam.evaluate(task, traj, env=None)
    assert isinstance(result, EvaluationResult)
    assert result.success is True
    assert result.score == 1.0
    assert result.details["em"] == 1.0
    assert result.details["f1"] == 1.0
    assert result.details["predicted_answer"] == "Paris"
    assert result.details["gold_answer"] == "Paris"


def test_evaluate_partial_overlap_no_exact() -> None:
    fam = HotpotqaFamily()
    task = _make_task(gold_answer="Barack Obama")
    traj = _fake_trajectory(
        [
            _fake_step(
                role="assistant",
                tool_calls=[
                    {"name": "finish", "arguments": {"answer": "President Obama"}}
                ],
            )
        ]
    )
    result = fam.evaluate(task, traj, env=None)
    assert result.details["em"] == 0.0
    assert 0.0 < result.details["f1"] < 1.0
    assert result.success is False


def test_evaluate_falls_back_to_final_assistant_text() -> None:
    fam = HotpotqaFamily()
    task = _make_task(gold_answer="Paris")
    traj = _fake_trajectory(
        [
            _fake_step(
                role="assistant",
                text="Based on the context, the answer is Paris",
                tool_calls=None,
            )
        ]
    )
    result = fam.evaluate(task, traj, env=None)
    # Contains "paris" among many tokens → f1 < 1, em=0.
    assert result.details["em"] == 0.0
    assert result.details["f1"] > 0.0


def test_extract_predicted_answer_returns_empty_on_empty_trajectory() -> None:
    assert extract_predicted_answer(None) == ""
    assert extract_predicted_answer(_fake_trajectory([])) == ""


def test_extract_predicted_answer_prefers_last_finish_call() -> None:
    traj = _fake_trajectory(
        [
            _fake_step(
                role="assistant",
                tool_calls=[{"name": "finish", "arguments": {"answer": "wrong"}}],
            ),
            _fake_step(
                role="assistant",
                tool_calls=[{"name": "finish", "arguments": {"answer": "right"}}],
            ),
        ]
    )
    assert extract_predicted_answer(traj) == "right"


# ---------------------------------------------------------------------------
# Family integration: build_env, build_tools, load_tasks
# ---------------------------------------------------------------------------


def test_build_env_returns_cache_with_expected_path(tmp_path: Path) -> None:
    fam = HotpotqaFamily()
    task = _make_task(task_id="hotpotqa_comparison_0001")
    env = fam.build_env(task)
    assert isinstance(env, WikipediaCache)
    assert env.path.name == "hotpotqa_comparison_0001.json"
    assert env.path.parent.name == "_cache"


def test_build_tools_names() -> None:
    fam = HotpotqaFamily()
    env = WikipediaCache(Path("/tmp/_unused_build_tools.json"))
    tools = fam.build_tools(env)
    assert {t.name for t in tools} == {
        "search_wikipedia",
        "get_wiki_page",
        "finish",
    }


def test_build_tools_rejects_non_cache_env() -> None:
    fam = HotpotqaFamily()
    with pytest.raises(TypeError):
        fam.build_tools(env=object())


def test_load_tasks_rejects_unknown_split() -> None:
    fam = HotpotqaFamily()
    with pytest.raises(FileNotFoundError):
        fam.load_tasks("no-such-split")


def test_load_tasks_reads_yamls_if_prepared() -> None:
    """If prepare has been run, dev_sample50 should load without issue."""
    fam = HotpotqaFamily()
    directory = split_dir()
    if not directory.is_dir() or not list(directory.glob("*.yaml")):
        pytest.skip(
            "HotPotQA benchmark not prepared yet; run scripts/prepare_hotpotqa.py"
        )
    tasks = fam.load_tasks("dev_sample50")
    assert len(tasks) == 50
    assert all(t.family == "hotpotqa" for t in tasks)
    condition_counts: dict[str, int] = {}
    for t in tasks:
        condition_counts[t.condition] = condition_counts.get(t.condition, 0) + 1
    assert condition_counts.get("simple", 0) == 20
    assert condition_counts.get("multi_step", 0) == 20
    assert condition_counts.get("error_recovery", 0) == 10


def test_teardown_env_persists_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fam = HotpotqaFamily()
    path = tmp_path / "tear.json"
    env = WikipediaCache(path)
    env.set("search_wikipedia", {"query": "q"}, {"results": []})
    fam.teardown_env(env)
    assert path.exists()
    reloaded = WikipediaCache(path)
    assert reloaded.get("search_wikipedia", {"query": "q"}) == {"results": []}
