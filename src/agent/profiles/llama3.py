"""Llama 3 profile factory (stub).

Llama 3.1 Instruct supports tool calling via a different chat template
format than Qwen (Python-style function calls inside
``<|python_tag|>...<|eom_id|>``). This profile is a placeholder for
Phase 1 wrap-up — delimiter resolution and parser will be filled in
once we start running the Llama pair.
"""

from __future__ import annotations

from typing import Any

from .base import DelimiterSet, ModelProfile, ParsedToolCall


def _llama_tool_parser(text: str) -> list[ParsedToolCall]:
    """Placeholder Llama tool call parser.

    Not yet implemented. Returns an empty list so the rest of the
    pipeline can be exercised with Llama profiles wired up.
    """
    return []


def build(tokenizer: Any) -> ModelProfile:
    """Build a Llama 3 profile (stub).

    Args:
        tokenizer: Loaded Llama 3 tokenizer.

    Returns:
        A ``ModelProfile`` whose parser and delimiters are placeholders.

    Raises:
        NotImplementedError: Always — delimiter resolution is TODO.
    """
    raise NotImplementedError(
        "Llama 3 profile is a stub; delimiter IDs and tool call parser "
        "will be implemented once Phase 1 validates the Qwen profiles."
    )
