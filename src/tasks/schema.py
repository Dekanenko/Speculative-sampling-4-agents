"""Task definition data class.

Tasks are loaded from YAML files under ``src/tasks/benchmarks/``. Each
file defines one task: a system prompt, a user prompt, the condition
label, and optionally a list of allowed tool names (if omitted, the
full mock tool list is available).
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
            list means "all mock tools".
        max_steps: Per-task override for the agent's max step count.
    """

    task_id: str
    condition: Condition
    system_prompt: str
    user_prompt: str
    allowed_tools: list[str] = field(default_factory=list)
    max_steps: int | None = None
