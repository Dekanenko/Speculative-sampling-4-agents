#!/usr/bin/env python
"""Thin CLI wrapper around
:func:`src.tasks.families.mbpp.prepare.prepare`.

Usage::

    PYTHONPATH=. python scripts/prepare_mbpp.py --split mbpp-v0

Loads the Hugging Face ``mbpp`` ``full`` train split, selects 50
stratified coding tasks, and writes one YAML per task under
``src/tasks/benchmarks/<split>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tasks.families.mbpp.prepare import prepare  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Select 50 stratified MBPP tasks and write task YAMLs."
        )
    )
    parser.add_argument(
        "--split",
        default="mbpp-v0",
        help="Split label (subdirectory under src/tasks/benchmarks/).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point.

    Returns:
        Process exit code (``0`` on success).
    """
    args = _parse_args()
    prepare(args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
