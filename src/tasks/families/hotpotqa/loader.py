"""HotPotQA task family: loader and :class:`TaskFamily` implementation.

The family targets HotPotQA's ``distractor`` dev split. Task YAMLs and
their paired Wikipedia caches live under
``src/tasks/benchmarks/hotpotqa-v0/`` and are produced by
``prepare.py`` (the only code path that touches the live Wikipedia
API).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....agent.tools import ToolSpec
from ...registry import load_task_set
from ...schema import Task
from ..base import EvaluationResult, Env, TaskFamily
from .env import WikipediaCache
from .evaluator import exact_match, extract_predicted_answer, token_f1
from .tools import build_wikipedia_tools


BENCHMARKS_ROOT = Path(__file__).resolve().parents[3] / "tasks" / "benchmarks"
"""Root containing split sub-directories for the HotPotQA family."""

SPLIT_NAME = "hotpotqa-v0"
"""Split sub-directory name. Only ``"dev_sample50"`` is recognised as
a logical split argument today."""

_CACHE_SUBDIR = "_cache"


def cache_path_for(task_id: str) -> Path:
    """Return the on-disk cache path for a given task id.

    Args:
        task_id: HotPotQA task id (e.g. ``hotpotqa_bridge_0003``).

    Returns:
        Absolute path to the task's cache JSON file.
    """
    return BENCHMARKS_ROOT / SPLIT_NAME / _CACHE_SUBDIR / f"{task_id}.json"


def split_dir() -> Path:
    """Return the directory holding the YAML task files for the split."""
    return BENCHMARKS_ROOT / SPLIT_NAME


class HotpotqaFamily(TaskFamily):
    """HotPotQA family with Wikipedia-backed tools and F1 scoring.

    Attributes:
        name: ``"hotpotqa"``.
    """

    name = "hotpotqa"

    def load_tasks(self, split: str) -> list[Task]:
        """Load the 50-task dev sample.

        Args:
            split: Only ``"dev_sample50"`` is supported currently.

        Returns:
            List of ``Task`` instances tagged with ``family="hotpotqa"``,
            sorted by ``task_id``.

        Raises:
            FileNotFoundError: If ``split`` is not ``"dev_sample50"``
                or the benchmark directory does not exist.
        """
        if split != "dev_sample50":
            raise FileNotFoundError(
                f"HotPotQA family has no split named {split!r}; "
                "only 'dev_sample50' is available."
            )
        directory = split_dir()
        if not directory.is_dir():
            raise FileNotFoundError(
                f"HotPotQA split directory not found: {directory}. "
                "Run scripts/prepare_hotpotqa.py first."
            )
        return load_task_set(directory)

    def build_env(self, task: Task) -> Env | None:
        """Return a :class:`WikipediaCache` for this task.

        Args:
            task: The task being prepared for a run.

        Returns:
            A cache pointed at ``_cache/{task_id}.json``. Reads from
            disk if the file exists; otherwise starts empty.
        """
        return WikipediaCache(cache_path_for(task.task_id))

    def build_tools(self, env: Env | None) -> list[ToolSpec]:
        """Return the three Wikipedia tools closed over ``env``.

        Args:
            env: A :class:`WikipediaCache` from :meth:`build_env`.

        Returns:
            ``[search_wikipedia, get_wiki_page, finish]``.

        Raises:
            TypeError: If ``env`` is not a :class:`WikipediaCache`.
        """
        if not isinstance(env, WikipediaCache):
            raise TypeError(
                "HotpotqaFamily.build_tools requires a WikipediaCache env; "
                f"got {type(env).__name__}"
            )
        return build_wikipedia_tools(env)

    def evaluate(
        self,
        task: Task,
        trajectory: Any,
        env: Env | None,
    ) -> EvaluationResult:
        """Score a trajectory with EM + token-level F1.

        Args:
            task: The task that was run.
            trajectory: The ``Trajectory`` the agent produced.
            env: The cache env (unused by this evaluator).

        Returns:
            An :class:`EvaluationResult` with ``score = f1`` and
            ``success = (em == 1.0)``.
        """
        gold = ""
        if task.expected is not None:
            gold = str(task.expected.get("answer", "") or "")
        predicted = extract_predicted_answer(trajectory)
        em = exact_match(predicted, gold)
        f1 = token_f1(predicted, gold)
        return EvaluationResult(
            task_id=task.task_id,
            success=em == 1.0,
            score=f1,
            details={
                "em": em,
                "f1": f1,
                "predicted_answer": predicted,
                "gold_answer": gold,
            },
        )

    def teardown_env(self, env: Env | None) -> None:
        """Persist any new cache entries the run collected.

        Args:
            env: The cache env returned by :meth:`build_env`, or ``None``.
        """
        if isinstance(env, WikipediaCache):
            env.save()
