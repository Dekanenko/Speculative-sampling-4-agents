"""Shared helpers for building model profiles."""

from __future__ import annotations

import json
import re
from typing import Any

from .base import ParsedToolCall


def encode_delimiter(tokenizer: Any, text: str) -> list[int]:
    """Encode a delimiter string to a list of token IDs.

    Resolves the delimiter against the *actual* tokenizer so that
    profiles never hardcode token IDs. Special tokens are disabled so
    that, e.g., ``<tool_call>`` is not swallowed by a special-token
    registration.

    Args:
        tokenizer: A Hugging Face tokenizer instance with ``encode``.
        text: The delimiter string to resolve.

    Returns:
        The list of token IDs the tokenizer assigns to ``text``.

    Raises:
        ValueError: If ``text`` encodes to zero tokens.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Delimiter {text!r} encodes to no tokens")
    return list(ids)


def qwen_flat_tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Render a tool as the flat schema Qwen chat templates expect.

    Qwen2.5 and Qwen3 were trained on a ``<tools>`` block containing
    flat ``{"name", "description", "parameters"}`` dicts — passing the
    OpenAI ``{"type": "function", "function": {...}}`` wrapper causes
    the wrapper to leak into the rendered prompt verbatim, which is
    off-distribution for the model.

    Args:
        name: Tool name.
        description: Human-readable tool description.
        parameters: JSON schema for the tool's arguments.

    Returns:
        A dict in the format Qwen's template expects.
    """
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
    re.DOTALL,
)


def parse_xml_tool_calls(text: str) -> list[ParsedToolCall]:
    """Parse ``<tool_call>{...}</tool_call>`` blocks from generated text.

    This is the format used by Qwen2.5 and Qwen3 chat templates. Each
    block contains a JSON object with ``name`` and ``arguments`` keys.

    Malformed blocks (bad JSON, missing fields) are skipped silently
    here; the caller is responsible for surfacing them as step errors.

    Args:
        text: Decoded assistant turn text.

    Returns:
        List of successfully parsed tool calls in order of appearance.
    """
    calls: list[ParsedToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        body = match.group("body")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        args = payload.get("arguments", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            continue
        calls.append(ParsedToolCall(name=name, arguments=args, raw=match.group(0)))
    return calls
