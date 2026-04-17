"""Trajectory evaluator for the MBPP family.

Scoring rule:

1. If any ``run_tests`` tool result in the trajectory reports
   ``failed == 0`` and ``passed > 0`` and ``timed_out is False``, the
   task succeeded: ``score=1.0, success=True``.
2. Otherwise, if at least one ``run_tests`` result exists with
   ``failed > 0``, the score is the pass rate of the *last* such
   result: ``passed / (passed + failed)``, ``success=False``.
3. Otherwise (no tests ever ran), ``score=0.0, success=False``.

Details returned include the final solution file contents, which must
be captured before the sandbox is torn down.
"""

from __future__ import annotations

from typing import Any

from ..base import EvaluationResult
from .env import TempDirSandbox
from .tools import OUTPUT_TRUNCATE_CHARS


# Maximum characters of solution.py to embed in the evaluation details.
# Big solutions can pollute logs; cap to the same limit as pytest output.
SOLUTION_TRUNCATE_CHARS: int = OUTPUT_TRUNCATE_CHARS


def _iter_run_tests_results(trajectory: Any) -> list[dict[str, Any]]:
    """Flatten all ``run_tests`` tool results out of a trajectory.

    Args:
        trajectory: A ``Trajectory`` (or object with ``.steps``).

    Returns:
        List of result dicts, in step order. Steps without tool
        activity or with no ``run_tests`` calls are skipped.
    """
    results: list[dict[str, Any]] = []
    steps = getattr(trajectory, "steps", None) or []
    for step in steps:
        calls = getattr(step, "tool_calls", None) or []
        step_results = getattr(step, "tool_results", None) or []
        for call, result in zip(calls, step_results):
            name = call.get("name") if isinstance(call, dict) else None
            if name == "run_tests" and isinstance(result, dict):
                results.append(result)
    return results


def _count_tool_calls(trajectory: Any) -> int:
    """Count every tool call across every step in the trajectory."""
    total = 0
    steps = getattr(trajectory, "steps", None) or []
    for step in steps:
        calls = getattr(step, "tool_calls", None) or []
        total += len(calls)
    return total


def _read_final_solution(env: TempDirSandbox | None) -> str | None:
    """Read ``solution.py`` from the sandbox if still available.

    Args:
        env: The per-task sandbox, or ``None`` if already torn down.

    Returns:
        File contents (truncated) or ``None`` if the file is missing
        or the sandbox is gone.
    """
    if env is None:
        return None
    solution_path = env.root / "solution.py"
    if not solution_path.is_file():
        return None
    try:
        text = solution_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if len(text) > SOLUTION_TRUNCATE_CHARS:
        return text[:SOLUTION_TRUNCATE_CHARS] + "\n...<truncated>..."
    return text


def evaluate_mbpp(
    task_id: str,
    trajectory: Any,
    env: TempDirSandbox | None,
) -> EvaluationResult:
    """Score one MBPP trajectory.

    Args:
        task_id: ID of the task that was run.
        trajectory: The ``Trajectory`` the agent produced.
        env: The sandbox used for the run. Must still exist on disk
            (``teardown_env`` has not yet been called) if the
            evaluator is to capture ``final_solution``.

    Returns:
        An ``EvaluationResult`` with ``details`` populated.
    """
    test_results = _iter_run_tests_results(trajectory)
    n_tool_calls = _count_tool_calls(trajectory)
    final_solution = _read_final_solution(env)

    # Find the last successful run.
    last_success: dict[str, Any] | None = None
    for result in test_results:
        if (
            result.get("failed", 0) == 0
            and result.get("passed", 0) > 0
            and not result.get("timed_out", False)
        ):
            last_success = result

    if last_success is not None:
        details: dict[str, Any] = {
            "passed": last_success.get("passed", 0),
            "failed": last_success.get("failed", 0),
            "timed_out": False,
            "last_run_tests_output": last_success.get("output", ""),
            "n_tool_calls_total": n_tool_calls,
            "final_solution": final_solution,
        }
        return EvaluationResult(
            task_id=task_id,
            success=True,
            score=1.0,
            details=details,
        )

    # No success. Fall back to the last failing result if any.
    last_failing: dict[str, Any] | None = None
    for result in test_results:
        if result.get("failed", 0) > 0:
            last_failing = result

    if last_failing is not None:
        passed = last_failing.get("passed", 0)
        failed = last_failing.get("failed", 0)
        denom = passed + failed
        score = (passed / denom) if denom > 0 else 0.0
        details = {
            "passed": passed,
            "failed": failed,
            "timed_out": bool(last_failing.get("timed_out", False)),
            "last_run_tests_output": last_failing.get("output", ""),
            "n_tool_calls_total": n_tool_calls,
            "final_solution": final_solution,
        }
        return EvaluationResult(
            task_id=task_id,
            success=False,
            score=score,
            details=details,
        )

    # No run_tests ever invoked.
    details = {
        "passed": 0,
        "failed": 0,
        "timed_out": False,
        "last_run_tests_output": "",
        "n_tool_calls_total": n_tool_calls,
        "final_solution": final_solution,
    }
    return EvaluationResult(
        task_id=task_id,
        success=False,
        score=0.0,
        details=details,
    )
