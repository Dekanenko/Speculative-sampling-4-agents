"""Tests for the TaskFamily abstraction and registry."""

from __future__ import annotations

import pytest

from src.tasks.families import (
    EvaluationResult,
    TaskFamily,
    get_family,
    list_families,
    register_family,
)
from src.tasks.families.mocks import MocksFamily
from src.tasks.schema import Task


def _task(task_id: str = "t1", family: str = "mocks") -> Task:
    return Task(
        task_id=task_id,
        condition="simple",
        system_prompt="sys",
        user_prompt="usr",
        family=family,
    )


def test_evaluation_result_is_json_serialisable() -> None:
    result = EvaluationResult(task_id="t1", success=True, score=1.0, details={"k": 1})
    payload = result.to_dict()
    assert payload["task_id"] == "t1"
    assert payload["success"] is True
    assert payload["score"] == 1.0
    assert payload["details"] == {"k": 1}


def test_mocks_family_registered_by_default() -> None:
    assert "mocks" in list_families()
    assert isinstance(get_family("mocks"), MocksFamily)


def test_get_family_raises_on_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_family("does_not_exist")


def test_register_family_rejects_duplicate() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_family(MocksFamily())


def test_mocks_family_tools_are_stateless() -> None:
    fam = get_family("mocks")
    env = fam.build_env(_task())
    assert env is None
    tools = fam.build_tools(env)
    tool_names = {t.name for t in tools}
    assert tool_names == {"get_weather", "calculator", "search"}


def test_mocks_family_loads_phase1_v0_split() -> None:
    fam = get_family("mocks")
    tasks = fam.load_tasks("phase1-v0")
    assert len(tasks) >= 3
    assert all(t.family == "mocks" for t in tasks)


def test_mocks_family_load_unknown_split_raises() -> None:
    fam = get_family("mocks")
    with pytest.raises(FileNotFoundError):
        fam.load_tasks("no-such-split")


def test_mocks_family_evaluate_always_succeeds() -> None:
    fam = get_family("mocks")
    task = _task()
    # Trajectory content is irrelevant for the mocks evaluator
    result = fam.evaluate(task, trajectory=None, env=None)
    assert result.task_id == "t1"
    assert result.success is True
    assert result.score == 1.0


def test_task_defaults_apply_when_family_omitted() -> None:
    task = Task(
        task_id="x",
        condition="simple",
        system_prompt="s",
        user_prompt="u",
    )
    assert task.family == "mocks"
    assert task.expected is None
    assert task.metadata == {}
