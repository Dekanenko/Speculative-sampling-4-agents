"""Core abstractions for task families.

A ``TaskFamily`` is the plug-in unit that lets the runner work with a
new dataset without touching the Agent. It bundles four
responsibilities:

1. **Task loading**: turn a logical split name (``"dev_sample50"``)
   into a list of :class:`~src.tasks.schema.Task` instances.
2. **Per-task env construction**: some families hold state per task —
   a temp directory for coding, a Wikipedia cache handle for
   retrieval. Stateless families (mocks) return ``None``.
3. **Per-task tool construction**: tools may close over the env, so
   they are built per task, not once at Agent init.
4. **Evaluation**: after the Agent produces a trajectory, the family
   scores it with family-specific logic and returns an
   :class:`EvaluationResult`.

``Env`` is intentionally typed as ``Any`` — each family decides what
its env object looks like. Tools receive the env via closure, the
evaluator receives it by argument.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from ...agent.tools import ToolSpec
from ..schema import Task


Env = Any


@dataclass
class EvaluationResult:
    """Outcome of evaluating one trajectory against its task.

    Attributes:
        task_id: ID of the task this evaluation belongs to.
        success: Whether the trajectory satisfied the family's
            success criterion.
        score: Numeric score in ``[0, 1]``; family defines the scale.
            ``1.0`` means perfect. ``success`` is typically
            ``score == 1.0`` but each family may choose.
        details: Family-specific breakdown (e.g. pass/fail counts,
            F1, supporting-fact overlap).
    """

    task_id: str
    success: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return asdict(self)


class TaskFamily(ABC):
    """Abstract base class for a pluggable task family.

    Subclasses must define a class-level ``name`` attribute and
    implement ``load_tasks``, ``build_tools``, and ``evaluate``.
    ``build_env`` defaults to returning ``None`` — override for
    stateful families.

    Attributes:
        name: Short family identifier (e.g. ``"hotpotqa"``).
    """

    name: str

    @abstractmethod
    def load_tasks(self, split: str) -> list[Task]:
        """Load all tasks in a split.

        Args:
            split: Logical split identifier (family-specific).

        Returns:
            A list of ``Task`` instances, ordered by ``task_id``.
        """

    def build_env(self, task: Task) -> Env | None:
        """Build a per-task environment handle.

        Default implementation returns ``None`` — families with no
        per-task state override this (file sandboxes, API caches,
        game engines, etc.).

        Args:
            task: The task this env will serve.

        Returns:
            A family-specific env object, or ``None``.
        """
        return None

    @abstractmethod
    def build_tools(self, env: Env | None) -> list[ToolSpec]:
        """Build the tool list for a single task run.

        Args:
            env: The env returned by ``build_env`` for this task, or
                ``None`` for stateless families.

        Returns:
            A list of ``ToolSpec`` — tools close over the env.
        """

    @abstractmethod
    def evaluate(
        self,
        task: Task,
        trajectory: Any,
        env: Env | None,
    ) -> EvaluationResult:
        """Score a trajectory against its task.

        Args:
            task: The task that was run.
            trajectory: The ``Trajectory`` the Agent produced.
            env: The env used during the run, or ``None``.

        Returns:
            An ``EvaluationResult`` the runner writes as a
            ``.eval.json`` sidecar to the trajectory file.
        """

    def teardown_env(self, env: Env | None) -> None:
        """Release any resources held by the env.

        Default implementation is a no-op. Families with temp dirs,
        open files, or running processes override this to clean up
        after the run completes.

        Args:
            env: The env returned by ``build_env``, or ``None``.
        """
        return None
