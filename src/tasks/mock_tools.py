"""Deterministic mock tools for Phase 1 experiments.

These tools return fixed, in-process results so that trajectories are
reproducible without network access or external APIs. Each tool's
output is a pure function of its arguments.
"""

from __future__ import annotations

from typing import Any

from ..agent.tools import ToolSpec


_WEATHER_TABLE: dict[str, dict[str, Any]] = {
    "berlin": {"temp_c": 12, "conditions": "cloudy"},
    "paris": {"temp_c": 15, "conditions": "sunny"},
    "tokyo": {"temp_c": 20, "conditions": "rain"},
    "new york": {"temp_c": 8, "conditions": "windy"},
}


def _get_weather(args: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic weather reading for a city."""
    city = str(args.get("city", "")).strip().lower()
    if city not in _WEATHER_TABLE:
        return {"error": f"unknown city: {city!r}"}
    return {"city": city, **_WEATHER_TABLE[city]}


def _calculator(args: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a simple arithmetic expression over two numbers."""
    op = args.get("op")
    a = args.get("a")
    b = args.get("b")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return {"error": "a and b must be numbers"}
    if op == "add":
        return {"result": a + b}
    if op == "sub":
        return {"result": a - b}
    if op == "mul":
        return {"result": a * b}
    if op == "div":
        if b == 0:
            return {"error": "division by zero"}
        return {"result": a / b}
    return {"error": f"unknown op: {op!r}"}


def _search(args: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic search stub keyed on the query string."""
    query = str(args.get("query", "")).strip().lower()
    if not query:
        return {"results": []}
    return {
        "results": [
            {"title": f"Result 1 for {query}", "snippet": "mock snippet A"},
            {"title": f"Result 2 for {query}", "snippet": "mock snippet B"},
        ]
    }


def build_mock_tools() -> list[ToolSpec]:
    """Return the standard Phase 1 mock tool list.

    Returns:
        A list of ``ToolSpec`` covering weather, calculator, and search.
    """
    return [
        ToolSpec(
            name="get_weather",
            description="Get the current weather for a known city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
            fn=_get_weather,
        ),
        ToolSpec(
            name="calculator",
            description="Evaluate a simple arithmetic operation on two numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["add", "sub", "mul", "div"],
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["op", "a", "b"],
            },
            fn=_calculator,
        ),
        ToolSpec(
            name="search",
            description="Return mock search results for a query string.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
            fn=_search,
        ),
    ]
