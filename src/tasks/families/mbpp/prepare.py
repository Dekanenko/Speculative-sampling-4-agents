"""One-shot MBPP benchmark preparation.

Given a split name (e.g. ``"mbpp-v0"``), pulls the Hugging Face
``mbpp`` ``full`` train split, selects 50 tasks stratified by
reference-solution length, and writes one YAML file per task under
``src/tasks/benchmarks/<split>/``.

Stratification buckets:

- ``simple``        — ref solution ≤ 3 non-empty body lines (20 tasks)
- ``multi_step``    — 4-8 body lines                       (20 tasks)
- ``long_context``  — ≥ 9 body lines                       (10 tasks)

Selection is deterministic under ``random.Random(42)``; outputs are
sorted by task_id so the on-disk listing is stable.
"""

from __future__ import annotations

import ast
import random
import re
from pathlib import Path
from typing import Any

import yaml

from .loader import BENCHMARKS_ROOT


# How many tasks we pull out per bucket. Totals to 50.
BUCKET_COUNTS: dict[str, int] = {
    "simple": 20,
    "multi_step": 20,
    "long_context": 10,
}

# Deterministic seed for the sampler.
SELECTION_SEED: int = 42

# Per-task step budget for the agent. Agents that need more than this
# to pass a basic MBPP problem are not measuring what we care about.
TASK_MAX_STEPS: int = 20


SYSTEM_PROMPT: str = (
    "You are a Python coding assistant. Implement the requested "
    "function in solution.py. After writing the code, call "
    "run_tests() to check it passes. If it fails, read the output, "
    "fix your code, and run tests again. When all tests pass, call "
    "finish().\n"
)


USER_PROMPT_TEMPLATE: str = (
    "{problem}\n\n"
    "The function signature should be implementable as described. "
    "Write your solution to `solution.py`. The tests are in "
    "`test_solution.py` and you cannot read them. Use read_file / "
    "write_file to work with solution.py, and run_tests() to check "
    "your work.\n"
)


# Regex that pulls the tested function's name out of an assert line.
_TESTED_FN_RE = re.compile(r"assert\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(")

# Python builtins we never want to treat as the function under test.
_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "int", "float", "str", "bool", "list", "dict", "set", "tuple",
        "len", "abs", "max", "min", "sum", "sorted", "reversed",
        "print", "range", "map", "filter", "any", "all", "round",
        "type", "isinstance", "getattr", "setattr", "hasattr",
    }
)


def _body_line_count(code: str) -> int:
    """Count non-empty lines in ``code`` excluding ``def`` signature lines.

    Multi-line signatures (Python allows wrapping) count as a single
    signature line — we approximate by stripping any line that begins
    with ``def``.

    Args:
        code: Python source for the reference solution.

    Returns:
        The body-line count used for bucket stratification.
    """
    count = 0
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("def "):
            continue
        count += 1
    return count


def _bucket_for(length: int) -> str | None:
    """Map a body-line count to a difficulty bucket.

    Args:
        length: Reference-solution body line count.

    Returns:
        Bucket name, or ``None`` if the length is non-positive.
    """
    if length <= 0:
        return None
    if length <= 3:
        return "simple"
    if length <= 8:
        return "multi_step"
    return "long_context"


def _extract_tested_fn_name(test_list: list[str]) -> str | None:
    """Find the function name being called by the first assert.

    Args:
        test_list: MBPP ``test_list`` — list of assert strings.

    Returns:
        The function name, or ``None`` if none of the usual patterns
        match or if the extracted name is a Python builtin.
    """
    for test in test_list:
        match = _TESTED_FN_RE.search(test)
        if match:
            name = match.group(1)
            if name in _BUILTIN_NAMES:
                return None
            return name
    return None


def _find_function_def(code: str, name: str) -> ast.FunctionDef | None:
    """Locate a top-level or nested ``def`` with the given name.

    Args:
        code: Python source.
        name: Function name to locate.

    Returns:
        The matching :class:`ast.FunctionDef` node, or ``None``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _render_signature(fn: ast.FunctionDef) -> str:
    """Render a function signature line with ``raise NotImplementedError``.

    Positional, keyword, and vararg / varkwarg parameters are all
    emitted with their names only — MBPP sources rarely annotate
    types, so we don't try to carry annotations through.

    Args:
        fn: The :class:`ast.FunctionDef` node to stub.

    Returns:
        A two-line source block: ``def ...(args):\n    raise
        NotImplementedError``.
    """
    parts: list[str] = []
    args = fn.args
    # Positional-only (rare in MBPP, but cheap to handle).
    posonly = getattr(args, "posonlyargs", []) or []
    for a in posonly:
        parts.append(a.arg)
    if posonly:
        parts.append("/")
    for a in args.args:
        parts.append(a.arg)
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for a in args.kwonlyargs:
        parts.append(a.arg)
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    sig = ", ".join(parts)
    return f"def {fn.name}({sig}):\n    raise NotImplementedError\n"


def _build_stub_and_test(
    row: dict[str, Any],
) -> tuple[str, str, str] | None:
    """Assemble ``(fn_name, stub_code, test_code)`` for one MBPP row.

    Args:
        row: A raw MBPP dataset row.

    Returns:
        ``(fn_name, stub_code, test_code)`` triple, or ``None`` if the
        row cannot be turned into a self-contained sandbox task (no
        matching function definition, tests that need setup, etc.).
    """
    if row.get("test_setup_code", "").strip():
        return None
    test_list: list[str] = list(row.get("test_list") or [])
    if not test_list:
        return None
    fn_name = _extract_tested_fn_name(test_list)
    if fn_name is None:
        return None
    fn_def = _find_function_def(row["code"], fn_name)
    if fn_def is None:
        return None

    stub_code = _render_signature(fn_def)

    indented_asserts = "\n".join(f"    {line}" for line in test_list)
    test_code = (
        f"from solution import {fn_name}\n\n\n"
        f"def test_{fn_name}():\n"
        f"{indented_asserts}\n"
    )
    return fn_name, stub_code, test_code


def _pick_stratified(
    rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select tasks into the three difficulty buckets.

    Args:
        rows: Candidate MBPP rows, each enriched with ``_bucket`` and
            ``_ref_len`` keys.
        rng: Random generator used for sampling.

    Returns:
        The selected rows (``sum(BUCKET_COUNTS.values())`` entries)
        with ``_bucket`` and ``_ref_len`` preserved.

    Raises:
        ValueError: If any bucket has fewer candidates than the
            requested count.
    """
    selected: list[dict[str, Any]] = []
    for bucket, count in BUCKET_COUNTS.items():
        pool = [r for r in rows if r["_bucket"] == bucket]
        if len(pool) < count:
            raise ValueError(
                f"Only {len(pool)} candidates for bucket {bucket!r}, "
                f"need {count}"
            )
        picked = rng.sample(pool, count)
        selected.extend(picked)
    return selected


def _prepare_rows() -> list[dict[str, Any]]:
    """Load MBPP train, filter, and tag each row with bucket + ref_len.

    Returns:
        A list of enriched dataset rows suitable for ``_pick_stratified``.
    """
    from datasets import load_dataset  # local import: heavy dep
    ds = load_dataset("mbpp", "full", split="train")

    enriched: list[dict[str, Any]] = []
    for row in ds:
        built = _build_stub_and_test(row)
        if built is None:
            continue
        fn_name, stub_code, test_code = built
        ref_len = _body_line_count(row["code"])
        bucket = _bucket_for(ref_len)
        if bucket is None:
            continue
        enriched.append(
            {
                "mbpp_task_id": int(row["task_id"]),
                "text": row["text"],
                "code": row["code"],
                "fn_name": fn_name,
                "stub_code": stub_code,
                "test_code": test_code,
                "_ref_len": ref_len,
                "_bucket": bucket,
            }
        )
    return enriched


def _task_id_for(bucket: str, ordinal: int) -> str:
    """Format a zero-padded task_id like ``mbpp_simple_0001``."""
    return f"mbpp_{bucket}_{ordinal:04d}"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Dump a task dict to YAML with deterministic key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, width=120)


def prepare(split: str) -> None:
    """Generate the ``split`` benchmark directory from Hugging Face MBPP.

    Args:
        split: Subdirectory name to write under
            ``src/tasks/benchmarks/`` (e.g. ``"mbpp-v0"``).
    """
    out_dir = BENCHMARKS_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SELECTION_SEED)
    enriched = _prepare_rows()
    picked = _pick_stratified(enriched, rng)

    # Sort deterministically by (bucket order, mbpp_task_id).
    bucket_order = {b: i for i, b in enumerate(BUCKET_COUNTS)}
    picked.sort(key=lambda r: (bucket_order[r["_bucket"]], r["mbpp_task_id"]))

    # Assign per-bucket ordinals so task_ids are contiguous within
    # each bucket.
    ordinal_by_bucket: dict[str, int] = {b: 0 for b in BUCKET_COUNTS}
    for row in picked:
        ordinal_by_bucket[row["_bucket"]] += 1
        ordinal = ordinal_by_bucket[row["_bucket"]]
        task_id = _task_id_for(row["_bucket"], ordinal)

        payload: dict[str, Any] = {
            "task_id": task_id,
            "condition": row["_bucket"],
            "family": "mbpp",
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT_TEMPLATE.format(problem=row["text"]),
            "allowed_tools": ["read_file", "write_file", "run_tests", "finish"],
            "max_steps": TASK_MAX_STEPS,
            "expected": {
                "task_id": row["mbpp_task_id"],
                "ref_solution_length": row["_ref_len"],
            },
            "metadata": {
                "mbpp_task_id": row["mbpp_task_id"],
                "difficulty_bucket": row["_bucket"],
                "ref_solution_length": row["_ref_len"],
                "fn_name": row["fn_name"],
                "test_code": row["test_code"],
                "stub_code": row["stub_code"],
            },
        }
        _write_yaml(out_dir / f"{task_id}.yaml", payload)
