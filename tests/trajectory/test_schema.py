"""Tests for trajectory dataclasses."""

from __future__ import annotations

import pytest

from src.trajectory.schema import Trajectory, TrajectoryMetadata, TrajectoryStep


def _metadata() -> TrajectoryMetadata:
    return TrajectoryMetadata(
        task_id="t1",
        condition="simple",
        model_target="fake/model",
        profile_name="qwen2.5",
        seed=7,
        generation_kwargs={"max_new_tokens": 8},
        dataset_version="phase1-v0",
        system_prompt="sys",
        user_prompt="hello",
        tool_schemas=[{"name": "mock", "description": "d", "parameters": {}}],
    )


def _step(idx: int) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=idx,
        task_id="t1",
        condition="simple",
        role="assistant",
        text="ok",
        token_ids=[1, 2],
        token_strings=["a", "b"],
        token_types=["response", "response"],
        target_logprobs=[-0.1, -0.2],
        draft_logprobs=None,
        acceptance_proxy=None,
        tool_calls=None,
        tool_results=None,
        error=None,
        wall_time_ms=1.0,
        model_target="fake/model",
        model_draft=None,
        seed=7,
    )


def test_trajectory_append_asserts_step_index() -> None:
    traj = Trajectory(metadata=_metadata())
    traj.append(_step(0))
    traj.append(_step(1))
    with pytest.raises(AssertionError):
        traj.append(_step(5))


def test_step_to_dict_is_json_friendly() -> None:
    step = _step(0)
    payload = step.to_dict()
    assert payload["step_index"] == 0
    assert payload["token_ids"] == [1, 2]
    assert payload["draft_logprobs"] is None
