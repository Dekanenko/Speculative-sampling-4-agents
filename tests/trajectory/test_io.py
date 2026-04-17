"""Tests for trajectory JSONL IO."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.trajectory.io import read_trajectory, write_trajectory
from src.trajectory.schema import Trajectory, TrajectoryMetadata, TrajectoryStep


def _trajectory() -> Trajectory:
    meta = TrajectoryMetadata(
        task_id="t1",
        condition="simple",
        model_target="fake/model",
        profile_name="qwen2.5",
        seed=7,
        generation_kwargs={"max_new_tokens": 8},
        family="mocks",
        dataset_split="phase1-v0",
        system_prompt="sys",
        user_prompt="hello",
        tool_schemas=[{"name": "mock", "description": "d", "parameters": {}}],
    )
    traj = Trajectory(metadata=meta)
    traj.append(
        TrajectoryStep(
            step_index=0,
            task_id="t1",
            condition="simple",
            role="assistant",
            text="hi",
            token_ids=[1, 2, 3],
            token_strings=["a", "b", "c"],
            token_types=["response", "response", "response"],
            target_logprobs=[-0.1, -0.2, -0.3],
            draft_logprobs=None,
            acceptance_proxy=None,
            tool_calls=None,
            tool_results=None,
            error=None,
            wall_time_ms=2.5,
            model_target="fake/model",
            model_draft=None,
            seed=7,
        )
    )
    return traj


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "traj.jsonl"
    traj = _trajectory()
    write_trajectory(path, traj)
    restored = read_trajectory(path)
    assert restored.metadata == traj.metadata
    assert len(restored.steps) == len(traj.steps)
    assert restored.steps[0] == traj.steps[0]


def test_read_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_trajectory(path)


def test_read_missing_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "nohead.jsonl"
    path.write_text('{"__kind__": "step"}\n')
    with pytest.raises(ValueError, match="header"):
        read_trajectory(path)
