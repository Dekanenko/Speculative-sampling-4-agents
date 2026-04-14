"""Tests for ToolSpec / ToolRegistry."""

from __future__ import annotations

import pytest

from src.agent.tools import ToolRegistry, ToolSpec


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"echo {name}",
        parameters={"type": "object", "properties": {}},
        fn=lambda args: {"echo": args, "tool": name},
    )


def test_registry_from_list_dispatches_by_name() -> None:
    registry = ToolRegistry.from_list([_spec("a"), _spec("b")])
    result = registry.call("a", {"x": 1})
    assert result == {"echo": {"x": 1}, "tool": "a"}


def test_registry_rejects_duplicate_tool_names() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry.from_list([_spec("a"), _spec("a")])


def test_registry_unknown_tool_raises_key_error() -> None:
    registry = ToolRegistry.from_list([_spec("a")])
    with pytest.raises(KeyError):
        registry.call("b", {})


def test_schemas_use_profile_provided_formatter() -> None:
    from src.agent.profiles._common import qwen_flat_tool_schema

    registry = ToolRegistry.from_list([_spec("a")])
    schemas = registry.schemas(qwen_flat_tool_schema)
    assert set(schemas[0].keys()) == {"name", "description", "parameters"}
    assert schemas[0]["name"] == "a"
    assert "type" not in schemas[0]


def test_schemas_accept_custom_formatter() -> None:
    def wrap(name: str, desc: str, params: dict) -> dict:
        return {"type": "function", "function": {"name": name}}

    registry = ToolRegistry.from_list([_spec("a")])
    schemas = registry.schemas(wrap)
    assert schemas[0] == {"type": "function", "function": {"name": "a"}}
