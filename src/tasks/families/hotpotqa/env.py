"""Wikipedia record-and-replay cache.

The HotPotQA family hits the live Wikipedia API exactly once, during
``prepare.py``. After that, every trajectory run replays from a
per-task JSON cache under
``src/tasks/benchmarks/hotpotqa-v0/_cache/{task_id}.json``. This
guarantees that:

* Trajectories are reproducible — the same task always sees the same
  tool outputs, even if Wikipedia changes underneath us.
* Tests and CI can run offline.
* The one-shot live pull is visibly separated from measurement.

The cache is a plain dict keyed on ``(tool_name, stable_json(args))``.
``refresh=True`` bypasses the cache on lookups, which lets the prepare
step force a re-pull of specific entries without wiping the whole
file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _stable_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a deterministic cache key for a tool call.

    Args:
        tool_name: Name of the tool (``"search_wikipedia"``, ...).
        arguments: Argument dict as passed to the tool.

    Returns:
        A stable, JSON-friendly string key combining the tool name and
        canonicalised arguments.
    """
    args_blob = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return f"{tool_name}::{args_blob}"


class WikipediaCache:
    """Per-task Wikipedia response cache.

    Attributes:
        path: JSON file the cache is persisted to.
        refresh: If ``True``, :meth:`get` returns ``None`` even for
            stored entries, forcing a re-pull on the next live call.
    """

    def __init__(self, path: str | Path, refresh: bool = False) -> None:
        """Initialise the cache from disk.

        Args:
            path: Destination JSON file. If missing, starts empty.
            refresh: If ``True``, lookups bypass existing entries.
        """
        self.path = Path(path)
        self.refresh = refresh
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as fh:
                self._entries = json.load(fh)

    def get(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return the cached response for a tool call, or ``None``.

        Args:
            tool_name: Name of the tool.
            arguments: Argument dict.

        Returns:
            The cached response dict, or ``None`` if missing or if
            ``refresh=True`` was set at construction time.
        """
        if self.refresh:
            return None
        key = _stable_key(tool_name, arguments)
        entry = self._entries.get(key)
        if entry is None:
            return None
        # Return a deep-ish copy so callers can't mutate cached state.
        return json.loads(json.dumps(entry))

    def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        """Store a response for a tool call.

        Args:
            tool_name: Name of the tool.
            arguments: Argument dict.
            response: Response dict to cache.
        """
        key = _stable_key(tool_name, arguments)
        self._entries[key] = json.loads(json.dumps(response))
        self._dirty = True

    def has(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Return ``True`` if an entry exists (ignoring ``refresh``).

        Args:
            tool_name: Name of the tool.
            arguments: Argument dict.

        Returns:
            Whether the underlying store contains the key.
        """
        return _stable_key(tool_name, arguments) in self._entries

    def save(self) -> None:
        """Persist the current entries to :attr:`path` if dirty."""
        if not self._dirty and self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._entries, fh, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.replace(self.path)
        self._dirty = False

    def __len__(self) -> int:
        """Return the number of stored entries."""
        return len(self._entries)
