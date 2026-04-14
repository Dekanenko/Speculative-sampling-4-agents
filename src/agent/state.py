"""Per-run Agent state.

``AgentState`` holds the mutable conversation and step counter for a
single task run. It is intentionally a plain dataclass so that tests
can construct and inspect it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    """Mutable state for a single task run.

    Attributes:
        messages: Chat messages accumulated so far. Each message is a
            dict with at least ``role`` and ``content`` keys.
        step_index: Next step index to assign.
        finished: True once the agent has produced a tool-free response
            or hit ``max_steps``.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    step_index: int = 0
    finished: bool = False

    def append_message(self, role: str, content: Any) -> None:
        """Append a chat message.

        Args:
            role: Chat role (``"system"``, ``"user"``, ``"assistant"``,
                ``"tool"``).
            content: Message content — a string or a structured dict,
                depending on what the chat template expects.
        """
        self.messages.append({"role": role, "content": content})
