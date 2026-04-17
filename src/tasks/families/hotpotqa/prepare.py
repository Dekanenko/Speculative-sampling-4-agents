"""One-shot live-data pull for the HotPotQA family.

Running :func:`prepare` hits the live Wikipedia API to pre-warm the
per-task caches and writes 50 YAML task files under
``src/tasks/benchmarks/hotpotqa-v0/``. Everything downstream — the
agent run, unit tests, CI — replays from those caches.

Task selection is deterministic under ``random.Random(42)``:
  * 20 ``type == 'comparison'`` tasks    → condition ``simple``
  * 20 ``type == 'bridge'`` tasks        → condition ``multi_step``
  * 10 additional ``type == 'bridge'`` tasks, with one gold
    supporting title corrupted, → condition ``error_recovery``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .env import WikipediaCache
from .loader import cache_path_for, split_dir
from .tools import RateLimiter, build_wikipedia_tools


_RANDOM_SEED = 42
_N_COMPARISON = 20
_N_BRIDGE = 20
_N_ERROR_RECOVERY = 10
_PERTURB_SUFFIX = "_xyzq_nonexistent"
_MAX_STEPS = 15

_SYSTEM_PROMPT = (
    "You are a research assistant. Use the wikipedia tools to answer "
    "the user's question. When you have the answer, call "
    "finish(answer=...)."
)


@dataclass(frozen=True)
class _SelectedTask:
    """Internal record describing one chosen HotPotQA example.

    Attributes:
        task_id: Generated benchmark id.
        condition: Experimental condition label.
        question_type: Original HotPotQA ``type`` field.
        question: The natural-language question.
        gold_answer: Short gold answer.
        supporting_titles: De-duplicated list of gold supporting titles.
        hotpot_qa_id: Original ``_id`` from the HF dataset.
        difficulty: Level if present in dataset, else ``"unknown"``.
        perturbed_title: Corrupted title for ``error_recovery``, else None.
    """

    task_id: str
    condition: str
    question_type: str
    question: str
    gold_answer: str
    supporting_titles: list[str]
    hotpot_qa_id: str
    difficulty: str
    perturbed_title: str | None


def _iter_dataset_rows() -> Iterable[dict[str, Any]]:
    """Yield HotPotQA distractor dev-split rows as plain dicts.

    Returns:
        Iterator over HotPotQA records. Downloads via Hugging Face
        ``datasets`` on first run; cached locally afterwards.
    """
    from datasets import load_dataset  # Local import keeps test imports light.

    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    for row in ds:
        yield dict(row)


def _extract_supporting_titles(row: dict[str, Any]) -> list[str]:
    """Pull the unique list of gold supporting-fact titles.

    Args:
        row: Raw HotPotQA row.

    Returns:
        Order-preserving de-duplicated list of supporting titles.
    """
    sf = row.get("supporting_facts") or {}
    titles = sf.get("title") if isinstance(sf, dict) else None
    if titles is None and isinstance(sf, list):
        titles = [item[0] for item in sf if isinstance(item, (list, tuple)) and item]
    titles = list(titles or [])
    seen: set[str] = set()
    out: list[str] = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _select_tasks(rows: list[dict[str, Any]]) -> list[_SelectedTask]:
    """Pick 50 tasks (20 comparison + 20 bridge + 10 error_recovery).

    Args:
        rows: All dev-split rows.

    Returns:
        The 50 chosen tasks with benchmark-style ids, deterministically
        shuffled under ``random.Random(42)``.
    """
    rng = random.Random(_RANDOM_SEED)
    comparison = [r for r in rows if r.get("type") == "comparison"]
    bridge = [r for r in rows if r.get("type") == "bridge"]
    rng.shuffle(comparison)
    rng.shuffle(bridge)

    picked_comparison = comparison[:_N_COMPARISON]
    picked_bridge = bridge[:_N_BRIDGE]
    picked_error = bridge[_N_BRIDGE : _N_BRIDGE + _N_ERROR_RECOVERY]
    if len(picked_comparison) < _N_COMPARISON or len(picked_bridge) < _N_BRIDGE:
        raise RuntimeError(
            "HotPotQA dev split does not contain enough comparison / bridge "
            "rows to assemble the 50-task sample."
        )
    if len(picked_error) < _N_ERROR_RECOVERY:
        raise RuntimeError(
            "HotPotQA dev split does not contain enough additional bridge "
            "rows for the error_recovery condition."
        )

    selected: list[_SelectedTask] = []
    for i, row in enumerate(picked_comparison, start=1):
        selected.append(
            _build_selected(
                row,
                task_id=f"hotpotqa_comparison_{i:04d}",
                condition="simple",
                perturbed_title=None,
            )
        )
    for i, row in enumerate(picked_bridge, start=1):
        selected.append(
            _build_selected(
                row,
                task_id=f"hotpotqa_bridge_{i:04d}",
                condition="multi_step",
                perturbed_title=None,
            )
        )
    # Assign error_recovery task_ids from a separate counter so that
    # rows with empty supporting_facts (skipped below) do not create
    # gaps in the numbering (0001, 0002, 0004, ...).
    error_recovery_idx = 0
    for row in picked_error:
        supporting = _extract_supporting_titles(row)
        if not supporting:
            continue
        error_recovery_idx += 1
        perturbed = supporting[0] + _PERTURB_SUFFIX
        selected.append(
            _build_selected(
                row,
                task_id=f"hotpotqa_error_recovery_{error_recovery_idx:04d}",
                condition="error_recovery",
                perturbed_title=perturbed,
            )
        )
    selected.sort(key=lambda t: t.task_id)
    return selected


def _build_selected(
    row: dict[str, Any],
    *,
    task_id: str,
    condition: str,
    perturbed_title: str | None,
) -> _SelectedTask:
    """Convert a HotPotQA row into a :class:`_SelectedTask`.

    Args:
        row: Raw HotPotQA record.
        task_id: Benchmark id to assign.
        condition: Experimental condition label.
        perturbed_title: Corrupted title for error_recovery, else None.

    Returns:
        A populated :class:`_SelectedTask`.
    """
    return _SelectedTask(
        task_id=task_id,
        condition=condition,
        question_type=str(row.get("type", "")),
        question=str(row.get("question", "")).strip(),
        gold_answer=str(row.get("answer", "")).strip(),
        supporting_titles=_extract_supporting_titles(row),
        hotpot_qa_id=str(row.get("id", row.get("_id", ""))),
        difficulty=str(row.get("level") or "unknown"),
        perturbed_title=perturbed_title,
    )


def _write_task_yaml(task: _SelectedTask, directory: Path) -> Path:
    """Render one :class:`_SelectedTask` to YAML on disk.

    Args:
        task: Task to serialise.
        directory: Output directory.

    Returns:
        The path the YAML file was written to.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task.task_id}.yaml"
    payload: dict[str, Any] = {
        "task_id": task.task_id,
        "condition": task.condition,
        "family": "hotpotqa",
        "system_prompt": _SYSTEM_PROMPT + "\n",
        "user_prompt": task.question,
        "allowed_tools": ["search_wikipedia", "get_wiki_page", "finish"],
        "max_steps": _MAX_STEPS,
        "expected": {
            "answer": task.gold_answer,
            "supporting_titles": list(task.supporting_titles),
            "question_type": task.question_type,
        },
        "metadata": {
            "hotpot_qa_id": task.hotpot_qa_id,
            "difficulty": task.difficulty,
            "perturbed_title": task.perturbed_title,
        },
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
    return path


def _warm_cache_for_task(task: _SelectedTask, refresh: bool) -> None:
    """Pre-populate the per-task Wikipedia cache.

    For every gold supporting title (and the perturbed title, if set)
    this invokes ``search_wikipedia`` and ``get_wiki_page`` via the
    real tool implementations, which write into the cache. The cache
    is persisted to disk before returning.

    Args:
        task: Selected task whose cache should be warmed.
        refresh: If ``True``, re-pull even entries that are already
            cached.
    """
    cache = WikipediaCache(cache_path_for(task.task_id), refresh=refresh)
    tools = {t.name: t for t in build_wikipedia_tools(cache, RateLimiter())}
    titles = list(task.supporting_titles)
    if task.perturbed_title is not None:
        titles.append(task.perturbed_title)
    for title in titles:
        tools["search_wikipedia"].fn({"query": title})
        tools["get_wiki_page"].fn({"title": title})
    cache.save()


def prepare(split: str, refresh_cache: bool = False) -> None:
    """Build the HotPotQA-v0 benchmark from live data.

    Args:
        split: Split label. Only ``"dev_sample50"`` is currently
            supported.
        refresh_cache: If ``True``, bypass existing cache entries and
            re-pull from Wikipedia even when present.

    Raises:
        ValueError: If ``split`` is not ``"dev_sample50"``.
    """
    if split != "dev_sample50":
        raise ValueError(
            f"Unknown HotPotQA split {split!r}; only 'dev_sample50' supported."
        )
    directory = split_dir()
    rows = list(_iter_dataset_rows())
    tasks = _select_tasks(rows)
    for task in tasks:
        _write_task_yaml(task, directory)
    for task in tasks:
        _warm_cache_for_task(task, refresh=refresh_cache)
