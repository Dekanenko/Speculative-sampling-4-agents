"""Tests for the MBPP family.

Pure local tests: no network, no HF dataset download, no model
loading. The ``run_tests`` tests spawn a short-lived ``python -m
pytest`` subprocess inside a temp dir — CI needs a working Python
and pytest, which are already required for the test suite itself.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from src.tasks.families.base import EvaluationResult
from src.tasks.families.mbpp import MbppFamily, TempDirSandbox, build_coding_tools
from src.tasks.families.mbpp.env import HIDDEN_TEST_FILENAME
from src.tasks.families.mbpp.evaluator import evaluate_mbpp
from src.tasks.families.mbpp.loader import BENCHMARKS_ROOT
from src.tasks.families.mbpp.tools import _parse_counts
from src.tasks.schema import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tools_by_name(sandbox: TempDirSandbox) -> dict[str, Any]:
    """Return ``{name: fn}`` for all coding tools built on ``sandbox``."""
    return {t.name: t.fn for t in build_coding_tools(sandbox)}


def _fake_step(
    tool_calls: list[dict[str, Any]] | None,
    tool_results: list[dict[str, Any]] | None,
) -> types.SimpleNamespace:
    """Build a duck-typed step with just the fields the evaluator reads."""
    return types.SimpleNamespace(
        tool_calls=tool_calls,
        tool_results=tool_results,
    )


def _fake_trajectory(steps: list[types.SimpleNamespace]) -> types.SimpleNamespace:
    """Wrap steps into a minimal trajectory-shaped object."""
    return types.SimpleNamespace(steps=steps)


def _write_pytest_file(sandbox: TempDirSandbox, body: str) -> None:
    """Write ``body`` to the sandbox's hidden pytest file."""
    sandbox.hidden_test_file.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


def test_sandbox_create_and_teardown() -> None:
    sandbox = TempDirSandbox()
    root = sandbox.root
    assert root.is_dir()
    assert sandbox.hidden_test_file.name == HIDDEN_TEST_FILENAME
    sandbox.teardown()
    assert not root.exists()


def test_sandbox_teardown_is_idempotent() -> None:
    sandbox = TempDirSandbox()
    sandbox.teardown()
    sandbox.teardown()  # should not raise


def test_sandbox_rejects_dotdot_paths() -> None:
    sandbox = TempDirSandbox()
    try:
        with pytest.raises(ValueError):
            sandbox.resolve("../escape.txt")
        with pytest.raises(ValueError):
            sandbox.resolve("subdir/../../escape.txt")
    finally:
        sandbox.teardown()


def test_sandbox_rejects_absolute_paths() -> None:
    sandbox = TempDirSandbox()
    try:
        with pytest.raises(ValueError):
            sandbox.resolve("/etc/passwd")
        with pytest.raises(ValueError):
            sandbox.resolve(str(sandbox.root / "x.txt"))
    finally:
        sandbox.teardown()


def test_sandbox_rejects_empty_path() -> None:
    sandbox = TempDirSandbox()
    try:
        with pytest.raises(ValueError):
            sandbox.resolve("")
    finally:
        sandbox.teardown()


# ---------------------------------------------------------------------------
# read_file / write_file tools
# ---------------------------------------------------------------------------


def test_write_then_read_round_trip() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        w = tools["write_file"]({"path": "solution.py", "contents": "x = 1\n"})
        assert w["success"] is True
        assert w["bytes_written"] == len("x = 1\n".encode("utf-8"))
        r = tools["read_file"]({"path": "solution.py"})
        assert r == {"contents": "x = 1\n"}
    finally:
        sandbox.teardown()


def test_read_file_not_found_returns_error_dict() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        result = tools["read_file"]({"path": "no_such_file.py"})
        assert result == {"error": "not_found"}
    finally:
        sandbox.teardown()


def test_write_file_rejects_path_escape() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        result = tools["write_file"](
            {"path": "../evil.py", "contents": "oops"}
        )
        assert result == {"error": "path_escape"}
    finally:
        sandbox.teardown()


def test_read_file_rejects_absolute_path() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        result = tools["read_file"]({"path": "/etc/passwd"})
        assert result == {"error": "path_escape"}
    finally:
        sandbox.teardown()


def test_write_file_rejects_non_string_contents() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        result = tools["write_file"]({"path": "x.py", "contents": 42})
        assert "error" in result
    finally:
        sandbox.teardown()


def test_write_file_creates_parent_directories() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        result = tools["write_file"](
            {"path": "a/b/c/file.py", "contents": "# nested\n"}
        )
        assert result["success"] is True
        assert (sandbox.root / "a" / "b" / "c" / "file.py").is_file()
    finally:
        sandbox.teardown()


# ---------------------------------------------------------------------------
# run_tests tool
# ---------------------------------------------------------------------------


def test_parse_counts() -> None:
    assert _parse_counts("1 passed in 0.01s") == (1, 0)
    assert _parse_counts("2 failed, 3 passed in 0.05s") == (3, 2)
    assert _parse_counts("no match here") == (0, 0)


def test_run_tests_reports_pass() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        tools["write_file"](
            {"path": "solution.py", "contents": "def add(a, b):\n    return a + b\n"}
        )
        _write_pytest_file(
            sandbox,
            "from solution import add\n\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
            "    assert add(-1, 1) == 0\n",
        )
        result = tools["run_tests"]({})
        assert result["timed_out"] is False
        assert result["passed"] >= 1
        assert result["failed"] == 0
    finally:
        sandbox.teardown()


def test_run_tests_reports_fail() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        tools["write_file"](
            {"path": "solution.py", "contents": "def add(a, b):\n    return a - b\n"}
        )
        _write_pytest_file(
            sandbox,
            "from solution import add\n\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n",
        )
        result = tools["run_tests"]({})
        assert result["timed_out"] is False
        assert result["failed"] >= 1
        assert result["passed"] == 0
    finally:
        sandbox.teardown()


def test_run_tests_times_out_on_infinite_loop() -> None:
    sandbox = TempDirSandbox()
    try:
        # Use a short timeout so the test stays snappy.
        tools_list = build_coding_tools(sandbox, test_timeout_s=2.0)
        tools = {t.name: t.fn for t in tools_list}
        tools["write_file"](
            {
                "path": "solution.py",
                "contents": "def spin():\n    while True:\n        pass\n",
            }
        )
        _write_pytest_file(
            sandbox,
            "from solution import spin\n\n\n"
            "def test_spin():\n"
            "    spin()\n",
        )
        result = tools["run_tests"]({})
        assert result["timed_out"] is True
        assert result["passed"] == 0
        assert result["failed"] == 0
    finally:
        sandbox.teardown()


def test_finish_tool_returns_done() -> None:
    sandbox = TempDirSandbox()
    try:
        tools = _tools_by_name(sandbox)
        assert tools["finish"]({}) == {"done": True}
    finally:
        sandbox.teardown()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _task(task_id: str = "mbpp_simple_0001") -> Task:
    return Task(
        task_id=task_id,
        condition="simple",
        system_prompt="sys",
        user_prompt="usr",
        family="mbpp",
        metadata={
            "stub_code": "def solve():\n    raise NotImplementedError\n",
            "test_code": "from solution import solve\n\n\n"
                         "def test_solve():\n    assert solve() == 1\n",
        },
    )


def test_evaluate_success_when_last_run_passes() -> None:
    trajectory = _fake_trajectory(
        [
            _fake_step(
                tool_calls=[{"name": "run_tests", "arguments": {}}],
                tool_results=[
                    {
                        "passed": 0,
                        "failed": 1,
                        "output": "first attempt failed",
                        "timed_out": False,
                    }
                ],
            ),
            _fake_step(
                tool_calls=[{"name": "run_tests", "arguments": {}}],
                tool_results=[
                    {
                        "passed": 3,
                        "failed": 0,
                        "output": "3 passed",
                        "timed_out": False,
                    }
                ],
            ),
        ]
    )
    result = evaluate_mbpp("t1", trajectory, env=None)
    assert isinstance(result, EvaluationResult)
    assert result.success is True
    assert result.score == 1.0
    assert result.details["passed"] == 3
    assert result.details["failed"] == 0
    assert result.details["n_tool_calls_total"] == 2


def test_evaluate_partial_when_tests_fail() -> None:
    trajectory = _fake_trajectory(
        [
            _fake_step(
                tool_calls=[{"name": "run_tests", "arguments": {}}],
                tool_results=[
                    {
                        "passed": 2,
                        "failed": 1,
                        "output": "mixed",
                        "timed_out": False,
                    }
                ],
            ),
        ]
    )
    result = evaluate_mbpp("t1", trajectory, env=None)
    assert result.success is False
    assert result.score == pytest.approx(2 / 3)
    assert result.details["passed"] == 2
    assert result.details["failed"] == 1


def test_evaluate_zero_when_tests_never_ran() -> None:
    trajectory = _fake_trajectory(
        [
            _fake_step(
                tool_calls=[{"name": "write_file", "arguments": {}}],
                tool_results=[{"success": True, "bytes_written": 10}],
            ),
        ]
    )
    result = evaluate_mbpp("t1", trajectory, env=None)
    assert result.success is False
    assert result.score == 0.0
    assert result.details["n_tool_calls_total"] == 1
    assert result.details["passed"] == 0


def test_evaluate_reads_final_solution_from_env() -> None:
    sandbox = TempDirSandbox()
    try:
        (sandbox.root / "solution.py").write_text(
            "def f():\n    return 42\n", encoding="utf-8"
        )
        trajectory = _fake_trajectory([])
        result = evaluate_mbpp("t1", trajectory, env=sandbox)
        assert result.details["final_solution"] == "def f():\n    return 42\n"
    finally:
        sandbox.teardown()


# ---------------------------------------------------------------------------
# MbppFamily end-to-end (no network)
# ---------------------------------------------------------------------------


def test_build_env_writes_stub_and_test_files() -> None:
    family = MbppFamily()
    task = _task()
    env = family.build_env(task)
    try:
        assert (env.root / "solution.py").read_text(encoding="utf-8") == (
            "def solve():\n    raise NotImplementedError\n"
        )
        assert env.hidden_test_file.read_text(encoding="utf-8").startswith(
            "from solution import solve"
        )
    finally:
        family.teardown_env(env)
        assert not env.root.exists()


def test_build_env_raises_on_missing_metadata() -> None:
    family = MbppFamily()
    task = Task(
        task_id="bad",
        condition="simple",
        system_prompt="s",
        user_prompt="u",
        family="mbpp",
        metadata={},
    )
    with pytest.raises(KeyError):
        family.build_env(task)


def test_build_tools_requires_env() -> None:
    family = MbppFamily()
    with pytest.raises(ValueError):
        family.build_tools(None)


def test_build_tools_returns_four_specs() -> None:
    family = MbppFamily()
    task = _task()
    env = family.build_env(task)
    try:
        specs = family.build_tools(env)
        names = {s.name for s in specs}
        assert names == {"read_file", "write_file", "run_tests", "finish"}
    finally:
        family.teardown_env(env)


def test_load_tasks_reads_generated_yamls() -> None:
    split_dir = BENCHMARKS_ROOT / "mbpp-v0"
    if not split_dir.is_dir() or not any(split_dir.glob("*.yaml")):
        pytest.skip(
            "mbpp-v0 split not generated yet; "
            "run scripts/prepare_mbpp.py first"
        )
    family = MbppFamily()
    tasks = family.load_tasks("mbpp-v0")
    assert len(tasks) > 0
    assert all(t.family == "mbpp" for t in tasks)
    for task in tasks:
        assert "stub_code" in task.metadata
        assert "test_code" in task.metadata
