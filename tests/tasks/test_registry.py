"""Tests for the YAML task loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tasks.registry import load_task, load_task_set


_TASK_YAML = """\
task_id: test_t1
condition: simple
system_prompt: sys
user_prompt: hello
allowed_tools: [get_weather]
"""


def test_load_task_parses_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(_TASK_YAML)
    task = load_task(path)
    assert task.task_id == "test_t1"
    assert task.condition == "simple"
    assert task.allowed_tools == ["get_weather"]


def test_load_task_missing_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text("task_id: x\ncondition: simple\nsystem_prompt: s\n")
    with pytest.raises(ValueError, match="user_prompt"):
        load_task(path)


def test_load_task_set_returns_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.yaml").write_text(
        _TASK_YAML.replace("test_t1", "btask")
    )
    (tmp_path / "a.yaml").write_text(
        _TASK_YAML.replace("test_t1", "atask")
    )
    tasks = load_task_set(tmp_path)
    assert [t.task_id for t in tasks] == ["atask", "btask"]


def test_phase1_benchmark_tasks_load() -> None:
    root = Path(__file__).parents[2] / "src" / "tasks" / "benchmarks" / "phase1"
    tasks = load_task_set(root)
    assert len(tasks) >= 3
    conditions = {t.condition for t in tasks}
    assert "simple" in conditions
    assert "multi_step" in conditions
    assert "error_recovery" in conditions
