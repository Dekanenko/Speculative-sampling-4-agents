"""Model-family profile dataclasses.

A ``ModelProfile`` captures everything that differs between model
families (Qwen2.5, Qwen3, Llama3): delimiter token IDs, the tool call
parser, stop tokens, and whether the family supports reasoning tokens.

All delimiter token IDs are resolved lazily against the actual
tokenizer at profile construction time. The registry keeps factory
functions, not frozen profiles, so the same profile can be built
against different tokenizer instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal


TokenType = Literal["tool_call", "reasoning", "response"]


@dataclass(frozen=True)
class ParsedToolCall:
    """A single parsed tool call extracted from generated text.

    Attributes:
        name: Tool name.
        arguments: JSON-decoded argument dict.
        raw: Raw text slice the call was parsed from (for debugging).
    """

    name: str
    arguments: dict
    raw: str


@dataclass(frozen=True)
class DelimiterSet:
    """Token ID sequences that open and close labelled regions.

    Each field is a list of token IDs because a single delimiter string
    may span multiple tokens depending on the tokenizer (e.g., ``<``,
    ``tool_call``, ``>``).

    Attributes:
        tool_call_open: Token ID sequence for the tool call opener.
        tool_call_close: Token ID sequence for the tool call closer.
        think_open: Token ID sequence for the thinking opener, or None.
        think_close: Token ID sequence for the thinking closer, or None.
    """

    tool_call_open: list[int]
    tool_call_close: list[int]
    think_open: list[int] | None = None
    think_close: list[int] | None = None

    def has_reasoning(self) -> bool:
        """Return True if this delimiter set defines thinking tokens."""
        return self.think_open is not None and self.think_close is not None


@dataclass(frozen=True)
class ModelProfile:
    """Everything the Agent needs to know about a model family.

    Attributes:
        name: Short human-readable family name (e.g., ``"qwen2.5"``).
        delimiters: Resolved delimiter token ID sequences.
        tool_call_parser: Function that extracts tool calls from text.
        tool_schema_formatter: Function that renders a raw
            ``(name, description, parameters)`` tuple into the exact
            dict shape the family's chat template expects inside its
            ``<tools>`` block. Qwen wants the flat form, Llama wants a
            different shape, etc. — so this is per-family.
        stop_token_ids: Extra stop token IDs beyond the tokenizer default.
        supports_reasoning: Whether the family emits ``<think>`` tokens.
    """

    name: str
    delimiters: DelimiterSet
    tool_call_parser: Callable[[str], list[ParsedToolCall]]
    tool_schema_formatter: Callable[[str, str, dict[str, Any]], dict[str, Any]]
    stop_token_ids: list[int] = field(default_factory=list)
    supports_reasoning: bool = False
