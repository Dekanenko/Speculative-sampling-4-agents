"""Task definition data class.

Tasks are loaded from YAML files under ``src/tasks/benchmarks/``. Each
file defines one task. Families (``mocks``, ``hotpotqa``, ``mbpp``,
...) share this schema and plug in their own loaders, tools, and
evaluators via :class:`~src.tasks.families.base.TaskFamily`.

Three optional fields keep the schema extensible without breaking old
Phase 1 YAMLs: ``family`` (defaults to ``"mocks"``), ``expected``
(gold answers / labels, family-defined shape), and ``metadata`` (any
family-specific scalars that don't deserve top-level fields).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import Condition


@dataclass(frozen=True)
class Task:
    """A single agent task definition.

    Attributes:
        task_id: Unique identifier (also used as the trajectory filename).
        condition: Experimental condition label.
        system_prompt: System message given to the agent.
        user_prompt: Initial user message.
        allowed_tools: Tool names this task is allowed to invoke. Empty
            list means "all tools the family provides".
        max_steps: Per-task override for the agent's max step count.
        family: Name of the ``TaskFamily`` that owns this task.
            Defaults to ``"mocks"`` so Phase 1 YAMLs continue to work
            without modification.
        expected: Optional gold answer / label payload the family's
            evaluator reads. Shape is family-specific (HotPotQA uses
            ``{"answer": str, "supporting_facts": [...]}``, MBPP uses
            ``{"tests": [str]}``, etc.).
        metadata: Family-specific scalars — difficulty tier, source
            dataset IDs, env init hints. Opaque to the Agent and the
            scorer.
    """

    task_id: str
    condition: Condition
    system_prompt: str
    user_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    max_steps: int | None = None
    family: str = "mocks"
    expected: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
