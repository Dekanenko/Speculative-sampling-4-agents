"""Mocks family: deterministic toy tools for Phase 1 / pipeline validation.

This is the original Phase 1 mock tool set (weather, calculator,
search) wrapped behind the :class:`TaskFamily` interface. The tools
are stateless pure functions, so ``build_env`` returns ``None`` and
``build_tools`` ignores the env argument.

``evaluate`` always reports ``success=True, score=1.0`` because the
mocks have no intrinsic success criterion — they exist to exercise
the measurement pipeline, not to be correct about anything. The
runner still writes an ``.eval.json`` sidecar for uniformity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....agent.tools import ToolSpec
from ...registry import load_task_set
from ...schema import Task
from ..base import EvaluationResult, Env, TaskFamily


_BENCHMARKS_ROOT = Path(__file__).resolve().parents[3] / "tasks" / "benchmarks"


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

    Kept at module level so external callers (older scripts, tests)
    can import it without going through the family abstraction.

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


class MocksFamily(TaskFamily):
    """The Phase 1 pipeline-validation family.

    Attributes:
        name: ``"mocks"``.
    """

    name = "mocks"

    def load_tasks(self, split: str) -> list[Task]:
        """Load all mock tasks in a split.

        Args:
            split: Subdirectory under ``src/tasks/benchmarks/``
                (e.g. ``"phase1-v0"``).

        Returns:
            All tasks from that directory, tagged with ``family="mocks"``.

        Raises:
            FileNotFoundError: If the split directory does not exist.
        """
        split_dir = _BENCHMARKS_ROOT / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Mocks split directory not found: {split_dir}"
            )
        return load_task_set(split_dir)

    def build_env(self, task: Task) -> Env | None:
        """Mocks are stateless — no env."""
        return None

    def build_tools(self, env: Env | None) -> list[ToolSpec]:
        """Return the standard mock tool list, ignoring ``env``."""
        return build_mock_tools()

    def evaluate(
        self,
        task: Task,
        trajectory: Any,
        env: Env | None,
    ) -> EvaluationResult:
        """Always report success — mocks have no correctness criterion."""
        return EvaluationResult(
            task_id=task.task_id,
            success=True,
            score=1.0,
            details={"note": "mocks family has no intrinsic success criterion"},
        )
