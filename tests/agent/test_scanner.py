"""Tests for the delimiter scanner."""

from __future__ import annotations

from src.agent.profiles.base import DelimiterSet
from src.agent.scanner import label_tokens


def _delims_single() -> DelimiterSet:
    return DelimiterSet(
        tool_call_open=[100],
        tool_call_close=[101],
        think_open=[200],
        think_close=[201],
    )


def _delims_multi() -> DelimiterSet:
    return DelimiterSet(
        tool_call_open=[10, 11, 12],
        tool_call_close=[13, 14, 15],
        think_open=None,
        think_close=None,
    )


def test_all_response_when_no_delimiters_seen() -> None:
    ids = [1, 2, 3, 4]
    result = label_tokens(ids, _delims_single())
    assert result.labels == ["response"] * 4
    assert result.unterminated is None


def test_single_token_tool_call_region() -> None:
    ids = [1, 100, 2, 3, 101, 4]
    result = label_tokens(ids, _delims_single())
    assert result.labels == [
        "response",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "response",
    ]
    assert result.unterminated is None


def test_multi_token_delimiter_sequences_match() -> None:
    ids = [1, 10, 11, 12, 99, 98, 13, 14, 15, 2]
    result = label_tokens(ids, _delims_multi())
    assert result.labels == [
        "response",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "response",
    ]
    assert result.unterminated is None


def test_reasoning_region_detected_when_supported() -> None:
    ids = [1, 200, 42, 43, 201, 2]
    result = label_tokens(ids, _delims_single())
    assert result.labels == [
        "response",
        "reasoning",
        "reasoning",
        "reasoning",
        "reasoning",
        "response",
    ]
    assert result.unterminated is None


def test_unterminated_tool_call_flagged() -> None:
    ids = [1, 100, 2, 3]
    result = label_tokens(ids, _delims_single())
    assert result.labels == ["response", "tool_call", "tool_call", "tool_call"]
    assert result.unterminated == "tool_call"


def test_stray_close_delimiter_is_ignored() -> None:
    ids = [1, 101, 2]
    result = label_tokens(ids, _delims_single())
    assert result.labels == ["response", "response", "response"]
    assert result.unterminated is None


def test_think_inside_response_then_tool_call_sequence() -> None:
    ids = [200, 5, 201, 6, 100, 7, 101, 8]
    result = label_tokens(ids, _delims_single())
    assert result.labels == [
        "reasoning",
        "reasoning",
        "reasoning",
        "response",
        "tool_call",
        "tool_call",
        "tool_call",
        "response",
    ]
    assert result.unterminated is None


def test_tool_call_opener_inside_reasoning_stays_reasoning() -> None:
    # <think> 5 <tool_call> 6 </tool_call> 7 </think> 8
    # The apparent <tool_call> and </tool_call> must be treated as
    # literal reasoning content, not a nested region.
    ids = [200, 5, 100, 6, 101, 7, 201, 8]
    result = label_tokens(ids, _delims_single())
    assert result.labels == [
        "reasoning",
        "reasoning",
        "reasoning",
        "reasoning",
        "reasoning",
        "reasoning",
        "reasoning",
        "response",
    ]
    assert result.unterminated is None


def test_think_opener_inside_tool_call_stays_tool_call() -> None:
    # <tool_call> 5 <think> 6 </think> 7 </tool_call> 8
    # Symmetric: a <think> appearing inside a tool_call is literal.
    ids = [100, 5, 200, 6, 201, 7, 101, 8]
    result = label_tokens(ids, _delims_single())
    assert result.labels == [
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "tool_call",
        "response",
    ]
    assert result.unterminated is None
