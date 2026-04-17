"""YAML loader for the MBPP family benchmark splits.

Benchmarks are stored as one YAML file per task under
``src/tasks/benchmarks/<split>/``. The loader reuses the shared YAML
schema in :mod:`src.tasks.registry` — MBPP-specific fields ride along
in the ``metadata`` dict (``test_code``, ``stub_code``,
``mbpp_task_id``, ``difficulty_bucket``, ``ref_solution_length``).
"""

from __future__ import annotations

from pathlib import Path

from ...registry import load_task_set
from ...schema import Task


# Anchor the benchmarks root at the repo-level ``src/tasks/benchmarks``
# directory. ``__file__`` lives at ``src/tasks/families/mbpp/loader.py``,
# so three ``parents`` hops reach ``src/tasks/``.
BENCHMARKS_ROOT: Path = (
    Path(__file__).resolve().parents[2] / "benchmarks"
)


def load_mbpp_split(split: str) -> list[Task]:
    """Load every MBPP YAML in a split directory.

    Args:
        split: Subdirectory name under ``src/tasks/benchmarks/``
            (e.g. ``"mbpp-v0"``).

    Returns:
        Tasks sorted by ``task_id``. Each task's ``family`` field must
        be ``"mbpp"`` — no check is enforced here; the family's loader
        is the one that validates this.

    Raises:
        FileNotFoundError: If ``split`` does not exist on disk.
    """
    split_dir = BENCHMARKS_ROOT / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"MBPP split directory not found: {split_dir}"
        )
    return load_task_set(split_dir)
