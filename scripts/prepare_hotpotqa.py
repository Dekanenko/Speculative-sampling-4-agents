#!/usr/bin/env python
"""Thin CLI wrapper around
:func:`src.tasks.families.hotpotqa.prepare.prepare`.

Usage::

    PYTHONPATH=. python scripts/prepare_hotpotqa.py \\
        --split dev_sample50 [--refresh-cache]

Hits the live Wikipedia API to warm the per-task caches and write
50 task YAML files under ``src/tasks/benchmarks/hotpotqa-v0/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is importable when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tasks.families.hotpotqa.prepare import prepare  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Pull HotPotQA distractor dev-split rows, pick 50, write task "
            "YAMLs, and pre-warm the Wikipedia caches."
        )
    )
    parser.add_argument(
        "--split",
        default="dev_sample50",
        help="Split label (currently only 'dev_sample50' is supported).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-pull from Wikipedia even for already-cached entries.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point.

    Returns:
        Process exit code (``0`` on success).
    """
    args = _parse_args()
    prepare(args.split, refresh_cache=args.refresh_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
