"""Tests for the deterministic mock tools."""

from __future__ import annotations

from src.tasks.mock_tools import build_mock_tools


def _by_name() -> dict:
    return {spec.name: spec for spec in build_mock_tools()}


def test_get_weather_returns_known_city() -> None:
    tool = _by_name()["get_weather"]
    result = tool.fn({"city": "Berlin"})
    assert result["temp_c"] == 12
    assert result["conditions"] == "cloudy"


def test_get_weather_unknown_city_returns_error() -> None:
    tool = _by_name()["get_weather"]
    result = tool.fn({"city": "Atlantis"})
    assert "error" in result


def test_calculator_add_and_divide_by_zero() -> None:
    tool = _by_name()["calculator"]
    assert tool.fn({"op": "add", "a": 2, "b": 3}) == {"result": 5}
    assert "error" in tool.fn({"op": "div", "a": 1, "b": 0})


def test_search_returns_two_results_for_query() -> None:
    tool = _by_name()["search"]
    result = tool.fn({"query": "qwen"})
    assert len(result["results"]) == 2
