"""Coding tools exposed to the agent during an MBPP task.

Four tools close over a :class:`TempDirSandbox`:

- ``read_file`` / ``write_file`` — sandboxed text file IO.
- ``run_tests``                 — invoke pytest on the hidden test file.
- ``finish``                    — signal the agent is done.

Every tool returns a JSON-serialisable dict and never raises — errors
are reported as ``{"error": "..."}`` so the agent can see them and
react.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from ....agent.tools import ToolSpec
from .env import TempDirSandbox


# Upper bound on pytest output returned to the agent. Real pytest
# output can explode on long tracebacks; truncation keeps the context
# window well-behaved.
OUTPUT_TRUNCATE_CHARS: int = 4000

# Per-invocation wall-clock ceiling for pytest. Any test suite that
# does not finish in this many seconds is considered hung.
DEFAULT_TEST_TIMEOUT_S: float = 10.0


# Regex: matches pytest summary lines like "3 passed, 1 failed in 0.01s".
_PASSED_RE = re.compile(r"(\d+)\s+passed", re.IGNORECASE)
_FAILED_RE = re.compile(r"(\d+)\s+failed", re.IGNORECASE)


def _parse_counts(output: str) -> tuple[int, int]:
    """Parse pytest pass/fail counts from captured output.

    Args:
        output: Combined stdout+stderr captured from pytest.

    Returns:
        ``(passed, failed)`` integers. Best-effort — missing values
        default to ``0``.
    """
    passed_match = _PASSED_RE.search(output)
    failed_match = _FAILED_RE.search(output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    return passed, failed


def _truncate(text: str, limit: int = OUTPUT_TRUNCATE_CHARS) -> str:
    """Truncate a string to ``limit`` characters, marking the cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def _make_read_file(sandbox: TempDirSandbox) -> Any:
    """Build a ``read_file`` handler closed over ``sandbox``."""

    def read_file(args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path", "")
        try:
            resolved = sandbox.resolve(str(path))
        except ValueError:
            return {"error": "path_escape"}
        if not resolved.is_file():
            return {"error": "not_found"}
        try:
            contents = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            return {"error": f"read_failed: {exc}"}
        return {"contents": contents}

    return read_file


def _make_write_file(sandbox: TempDirSandbox) -> Any:
    """Build a ``write_file`` handler closed over ``sandbox``."""

    def write_file(args: dict[str, Any]) -> dict[str, Any]:
        path = args.get("path", "")
        contents = args.get("contents", "")
        if not isinstance(contents, str):
            return {"error": "contents must be a string"}
        try:
            resolved = sandbox.resolve(str(path))
        except ValueError:
            return {"error": "path_escape"}
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(contents, encoding="utf-8")
        except OSError as exc:
            return {"error": f"write_failed: {exc}"}
        return {"success": True, "bytes_written": len(contents.encode("utf-8"))}

    return write_file


def _make_run_tests(
    sandbox: TempDirSandbox,
    timeout_s: float = DEFAULT_TEST_TIMEOUT_S,
) -> Any:
    """Build a ``run_tests`` handler closed over ``sandbox``.

    Args:
        sandbox: The per-task sandbox whose hidden test file to run.
        timeout_s: Wall-clock timeout passed to ``subprocess.run``.
    """

    def run_tests(args: dict[str, Any]) -> dict[str, Any]:
        del args  # unused — run_tests takes no parameters
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(sandbox.hidden_test_file),
            "--no-header",
            "-q",
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(sandbox.root),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": 0,
                "failed": 0,
                "output": "<truncated>",
                "duration_ms": int(timeout_s * 1000),
                "timed_out": True,
            }
        except OSError as exc:
            return {
                "passed": 0,
                "failed": 0,
                "output": f"pytest_invocation_failed: {exc}",
                "duration_ms": 0,
                "timed_out": False,
            }
        combined = (completed.stdout or "") + (completed.stderr or "")
        passed, failed = _parse_counts(combined)
        return {
            "passed": passed,
            "failed": failed,
            "output": _truncate(combined),
            "duration_ms": 0,
            "timed_out": False,
        }

    return run_tests


def _make_finish() -> Any:
    """Build a stateless ``finish`` handler."""

    def finish(args: dict[str, Any]) -> dict[str, Any]:
        del args
        return {"done": True}

    return finish


def build_coding_tools(
    sandbox: TempDirSandbox,
    *,
    test_timeout_s: float = DEFAULT_TEST_TIMEOUT_S,
) -> list[ToolSpec]:
    """Return the four MBPP coding tools, each closed over ``sandbox``.

    Args:
        sandbox: The per-task sandbox the tools should operate on.
        test_timeout_s: Timeout for the ``run_tests`` subprocess.

    Returns:
        A list of four ``ToolSpec``s: read_file, write_file, run_tests,
        finish.
    """
    return [
        ToolSpec(
            name="read_file",
            description=(
                "Read a text file from the sandbox. Returns {contents} "
                "on success or {error} on failure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path inside the sandbox."
                        ),
                    },
                },
                "required": ["path"],
            },
            fn=_make_read_file(sandbox),
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Write a text file inside the sandbox, creating parent "
                "directories as needed. Overwrites existing files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative path inside the sandbox."
                        ),
                    },
                    "contents": {
                        "type": "string",
                        "description": "Full text content to write.",
                    },
                },
                "required": ["path", "contents"],
            },
            fn=_make_write_file(sandbox),
        ),
        ToolSpec(
            name="run_tests",
            description=(
                "Run the hidden pytest file against the current "
                "solution.py. Returns pass/fail counts and truncated "
                "output."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
            fn=_make_run_tests(sandbox, timeout_s=test_timeout_s),
        ),
        ToolSpec(
            name="finish",
            description=(
                "Signal that the task is complete. Call only after "
                "all tests pass."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
            fn=_make_finish(),
        ),
    ]
