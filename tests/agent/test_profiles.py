"""Tests for profile factories using a fake tokenizer.

These tests avoid downloading real model weights. They use a minimal
fake tokenizer that encodes delimiter strings to stable sequences of
integers.
"""

from __future__ import annotations

import pytest

from src.agent.profiles.qwen25 import build as build_qwen25
from src.agent.profiles.qwen3 import build as build_qwen3
from src.agent.profiles.registry import build_profile, list_profiles
from src.agent.profiles._common import parse_xml_tool_calls


class FakeTokenizer:
    """Encodes any string to a deterministic list of character code points.

    Good enough for unit testing delimiter resolution: distinct input
    strings map to distinct token ID sequences.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]


def test_qwen25_profile_resolves_tool_call_delimiters() -> None:
    profile = build_qwen25(FakeTokenizer())
    assert profile.name == "qwen2.5"
    assert profile.supports_reasoning is False
    assert profile.delimiters.tool_call_open == [ord(c) for c in "<tool_call>"]
    assert profile.delimiters.tool_call_close == [ord(c) for c in "</tool_call>"]
    assert profile.delimiters.think_open is None
    assert profile.delimiters.think_close is None


def test_qwen3_profile_resolves_think_delimiters() -> None:
    profile = build_qwen3(FakeTokenizer())
    assert profile.name == "qwen3"
    assert profile.supports_reasoning is True
    assert profile.delimiters.has_reasoning()
    assert profile.delimiters.think_open == [ord(c) for c in "<think>"]
    assert profile.delimiters.think_close == [ord(c) for c in "</think>"]


def test_registry_builds_known_profiles() -> None:
    assert "qwen2.5" in list_profiles()
    assert "qwen3" in list_profiles()
    profile = build_profile("qwen2.5", FakeTokenizer())
    assert profile.name == "qwen2.5"


def test_registry_raises_on_unknown_profile() -> None:
    with pytest.raises(KeyError):
        build_profile("does-not-exist", FakeTokenizer())


def test_parse_xml_tool_calls_extracts_one_call() -> None:
    text = (
        'Some reasoning.\n'
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Berlin"}}\n</tool_call>'
    )
    calls = parse_xml_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "Berlin"}


def test_parse_xml_tool_calls_skips_malformed_json() -> None:
    text = "<tool_call>not-json</tool_call>"
    assert parse_xml_tool_calls(text) == []


def test_parse_xml_tool_calls_extracts_multiple_calls_in_order() -> None:
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    calls = parse_xml_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[1].arguments == {"x": 1}
