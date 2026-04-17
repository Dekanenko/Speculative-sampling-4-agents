"""Unit tests for the draft-scoring primitives.

These cover the pure functions (replay invariant check, logprob
gather, acceptance proxy). The full :func:`score_trajectory`
orchestrator is exercised end-to-end on the EC2 dev server against
a real draft model — there is no unit test that loads a model.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.scoring.draft import (
    compute_acceptance_proxy,
    gather_draft_logprobs,
    verify_replay_invariant,
)
from src.trajectory.schema import Trajectory, TrajectoryMetadata, TrajectoryStep


def _metadata() -> TrajectoryMetadata:
    return TrajectoryMetadata(
        task_id="t1",
        condition="simple",
        model_target="fake/target",
        profile_name="qwen2.5",
        seed=7,
        generation_kwargs={"max_new_tokens": 8},
        family="mocks",
        dataset_split="phase1-v0",
        system_prompt="sys",
        user_prompt="usr",
        tool_schemas=[],
    )


def _step(step_index: int, token_ids: list[int], tlps: list[float]) -> TrajectoryStep:
    return TrajectoryStep(
        step_index=step_index,
        task_id="t1",
        condition="simple",
        role="assistant",
        text="",
        token_ids=token_ids,
        token_strings=[f"tok_{i}" for i in token_ids],
        token_types=["response"] * len(token_ids),
        target_logprobs=tlps,
        draft_logprobs=None,
        acceptance_proxy=None,
        tool_calls=None,
        tool_results=None,
        error=None,
        wall_time_ms=0.0,
        model_target="fake/target",
        model_draft=None,
        seed=7,
    )


def test_replay_invariant_accepts_aligned_offsets() -> None:
    traj = Trajectory(
        metadata=_metadata(),
        steps=[
            _step(0, [10, 11, 12], [-0.1, -0.2, -0.3]),
            _step(1, [20, 21], [-0.4, -0.5]),
        ],
    )
    full_ids = [1, 2, 3, 10, 11, 12, 4, 5, 20, 21]
    prefix_lens = [3, 8]
    verify_replay_invariant(full_ids, traj, prefix_lens)


def test_replay_invariant_rejects_misaligned_step() -> None:
    traj = Trajectory(
        metadata=_metadata(),
        steps=[_step(0, [10, 11, 12], [-0.1, -0.2, -0.3])],
    )
    # Intentionally place tokens at the wrong offset
    full_ids = [1, 2, 3, 99, 11, 12]
    with pytest.raises(RuntimeError, match="Replay invariant violated"):
        verify_replay_invariant(full_ids, traj, [3])


def test_gather_draft_logprobs_applies_causal_shift() -> None:
    # Build a hand-crafted log_probs tensor of shape (L=5, V=4).
    # The scorer gathers log_probs[start - 1 + j, token_ids[j]].
    # For start=3 and token_ids=[2, 0], we should read
    # log_probs[2, 2] and log_probs[3, 0].
    log_probs = torch.tensor(
        [
            [-0.1, -0.2, -0.3, -0.4],  # pos 0
            [-1.1, -1.2, -1.3, -1.4],  # pos 1
            [-2.1, -2.2, -2.3, -2.4],  # pos 2  -> want log_probs[2, 2] = -2.3
            [-3.1, -3.2, -3.3, -3.4],  # pos 3  -> want log_probs[3, 0] = -3.1
            [-4.1, -4.2, -4.3, -4.4],  # pos 4
        ]
    )
    result = gather_draft_logprobs(log_probs, token_ids=[2, 0], start=3)
    assert result == pytest.approx([-2.3, -3.1])


def test_gather_draft_logprobs_rejects_start_zero() -> None:
    log_probs = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        gather_draft_logprobs(log_probs, token_ids=[0], start=0)


def test_acceptance_proxy_matches_formula() -> None:
    target = [-0.1, -2.0, -5.0]
    draft = [-0.1, -1.0, -4.5]
    expected = [
        min(1.0, math.exp(-0.1 - (-0.1))),   # 1.0
        min(1.0, math.exp(-2.0 - (-1.0))),   # exp(-1) ~= 0.368
        min(1.0, math.exp(-5.0 - (-4.5))),   # exp(-0.5) ~= 0.607
    ]
    result = compute_acceptance_proxy(target, draft)
    assert result == pytest.approx(expected)


def test_acceptance_proxy_caps_at_one() -> None:
    # Target more confident than draft — ratio > 1, should cap at 1.0.
    result = compute_acceptance_proxy([-1.0], [-5.0])
    assert result == [1.0]


def test_acceptance_proxy_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_acceptance_proxy([-0.1, -0.2], [-0.1])
