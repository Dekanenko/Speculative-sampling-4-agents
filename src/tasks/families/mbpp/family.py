"""MBPP family implementation plugging into :class:`TaskFamily`.

Loads YAMLs via :mod:`.loader`, builds a fresh
:class:`TempDirSandbox` per task seeded with ``solution.py`` (stub)
and ``test_solution.py`` (hidden tests), exposes the four coding
tools, and scores trajectories with :func:`evaluate_mbpp`.
"""

from __future__ import annotations

from typing import Any

from ....agent.tools import ToolSpec
from ...schema import Task
from ..base import EvaluationResult, Env, TaskFamily
from .env import TempDirSandbox
from .evaluator import evaluate_mbpp
from .loader import load_mbpp_split
from .tools import build_coding_tools


# Keys expected inside ``task.metadata``. Centralised so typos fail at
# one place rather than raising ``KeyError`` mid-run.
METADATA_STUB_KEY: str = "stub_code"
METADATA_TEST_KEY: str = "test_code"


class MbppFamily(TaskFamily):
    """The MBPP coding family.

    Attributes:
        name: ``"mbpp"``.
    """

    name = "mbpp"

    def load_tasks(self, split: str) -> list[Task]:
        """Load all MBPP tasks in a split.

        Args:
            split: Subdirectory name under ``src/tasks/benchmarks/``
                (e.g. ``"mbpp-v0"``).

        Returns:
            Tasks sorted by ``task_id``.

        Raises:
            FileNotFoundError: If the split directory is missing.
        """
        return load_mbpp_split(split)

    def build_env(self, task: Task) -> Env | None:
        """Create a fresh sandbox and seed it with the task files.

        Writes ``solution.py`` (stub the agent will edit) and
        ``test_solution.py`` (hidden pytest file) into the sandbox
        from the task's ``metadata`` dict.

        Args:
            task: The task to build the env for. ``task.metadata`` must
                contain ``stub_code`` and ``test_code`` keys.

        Returns:
            A :class:`TempDirSandbox` ready to receive tool calls.

        Raises:
            KeyError: If ``stub_code`` or ``test_code`` is missing
                from ``task.metadata``.
        """
        metadata = task.metadata or {}
        if METADATA_STUB_KEY not in metadata:
            raise KeyError(
                f"Task {task.task_id!r} is missing metadata.{METADATA_STUB_KEY}"
            )
        if METADATA_TEST_KEY not in metadata:
            raise KeyError(
                f"Task {task.task_id!r} is missing metadata.{METADATA_TEST_KEY}"
            )

        sandbox = TempDirSandbox()
        (sandbox.root / "solution.py").write_text(
            metadata[METADATA_STUB_KEY], encoding="utf-8"
        )
        sandbox.hidden_test_file.write_text(
            metadata[METADATA_TEST_KEY], encoding="utf-8"
        )
        return sandbox

    def build_tools(self, env: Env | None) -> list[ToolSpec]:
        """Return the four coding tools closed over the task sandbox.

        Args:
            env: A :class:`TempDirSandbox`. Must be non-``None`` for
                MBPP — the tools cannot operate without one.

        Returns:
            The four-tool list: read_file, write_file, run_tests, finish.

        Raises:
            ValueError: If ``env`` is ``None``.
        """
        if env is None:
            raise ValueError(
                "MbppFamily.build_tools requires a TempDirSandbox env"
            )
        return build_coding_tools(env)

    def evaluate(
        self,
        task: Task,
        trajectory: Any,
        env: Env | None,
    ) -> EvaluationResult:
        """Score one trajectory using :func:`evaluate_mbpp`.

        Note that this must run *before* :meth:`teardown_env`, so the
        evaluator can read ``solution.py`` from the sandbox.

        Args:
            task: The task that was run.
            trajectory: The ``Trajectory`` the agent produced.
            env: The sandbox (must still be live on disk).

        Returns:
            An ``EvaluationResult`` describing pass/fail state.
        """
        return evaluate_mbpp(task.task_id, trajectory, env)

    def teardown_env(self, env: Env | None) -> None:
        """Remove the sandbox temp directory.

        Args:
            env: The sandbox to tear down. No-op if ``None``.
        """
        if env is None:
            return
        env.teardown()
