"""Qwen3 profile factory.

Qwen3 uses the same ``<tool_call>`` JSON format as Qwen2.5 but also
emits explicit ``<think>...</think>`` blocks before the final answer.
Both are needed for the reasoning-token acceptance analysis.
"""

from __future__ import annotations

from typing import Any

from ._common import encode_delimiter, parse_xml_tool_calls, qwen_flat_tool_schema
from .base import DelimiterSet, ModelProfile


def build(tokenizer: Any) -> ModelProfile:
    """Build a Qwen3 profile against a concrete tokenizer.

    Args:
        tokenizer: Loaded Qwen3 tokenizer.

    Returns:
        A resolved ``ModelProfile`` ready to hand to the Agent.
    """
    delimiters = DelimiterSet(
        tool_call_open=encode_delimiter(tokenizer, "<tool_call>"),
        tool_call_close=encode_delimiter(tokenizer, "</tool_call>"),
        think_open=encode_delimiter(tokenizer, "<think>"),
        think_close=encode_delimiter(tokenizer, "</think>"),
    )
    return ModelProfile(
        name="qwen3",
        delimiters=delimiters,
        tool_call_parser=parse_xml_tool_calls,
        tool_schema_formatter=qwen_flat_tool_schema,
        stop_token_ids=[],
        supports_reasoning=True,
    )
