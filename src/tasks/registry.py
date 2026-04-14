"""YAML-backed task loader.

Tasks live under ``src/tasks/benchmarks/<phase>/<task>.yaml``. Each
YAML file maps 1:1 to a ``Task`` dataclass. Loading is deliberately
simple — no schema validation library, just dict unpacking with
explicit error messages on missing fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schema import Task


_REQUIRED_FIELDS = ("task_id", "condition", "system_prompt", "user_prompt")


def load_task(path: str | Path) -> Task:
    """Load a single task from a YAML file.

    Args:
        path: Path to a YAML file describing one task.

    Returns:
        A ``Task`` dataclass.

    Raises:
        ValueError: If a required field is missing.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    for field_name in _REQUIRED_FIELDS:
        if field_name not in raw:
            raise ValueError(f"Task file {path} is missing field {field_name!r}")

    return Task(
        task_id=raw["task_id"],
        condition=raw["condition"],
        system_prompt=raw["system_prompt"],
        user_prompt=raw["user_prompt"],
        allowed_tools=list(raw.get("allowed_tools", [])),
        max_steps=raw.get("max_steps"),
    )


def load_task_set(directory: str | Path) -> list[Task]:
    """Load every ``*.yaml`` file under ``directory`` as a Task.

    Args:
        directory: Directory to scan (non-recursive).

    Returns:
        List of tasks sorted by ``task_id``.
    """
    directory = Path(directory)
    tasks = [load_task(p) for p in sorted(directory.glob("*.yaml"))]
    return sorted(tasks, key=lambda t: t.task_id)
