"""Trajectory data classes.

A ``Trajectory`` is the record of a single agent run on a single task.
It is the unit that gets serialised to JSONL and later re-scored by
draft models offline.

The design is append-only: steps are produced one at a time by the
Agent and are never mutated. Draft logprob fields are left as ``None``
here; an offline scoring pass fills them in on copies of the step and
writes a new trajectory file rather than editing in place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..agent.profiles.base import TokenType
from ..config import Condition


@dataclass
class TrajectoryStep:
    """One agent generation + optional tool invocation.

    Attributes:
        step_index: Zero-based index of this step within its trajectory.
        task_id: ID of the task this trajectory belongs to.
        condition: Experimental condition label.
        role: Chat role of this step.
        text: Decoded text of the generated assistant turn (or tool result).
        token_ids: Raw token IDs — the single source of truth.
        token_strings: Debug-only stringified tokens, aligned with ``token_ids``.
        token_types: Per-token type labels, aligned with ``token_ids``.
        target_logprobs: Target model logprob per generated token.
        draft_logprobs: Draft model teacher-forced logprob per token, filled
            in by an offline scoring pass.
        acceptance_proxy: Per-token acceptance proxy values, filled in after
            draft scoring.
        tool_calls: List of parsed tool call dicts produced by this step.
            Qwen may emit multiple ``<tool_call>`` blocks in a single
            assistant turn, so this is always a list (possibly empty /
            None if the step is a tool-free final response).
        tool_results: List of tool execution result dicts, one per call
            in ``tool_calls`` and aligned by index.
        error: Human-readable error tag for malformed output, if any.
        wall_time_ms: Wall-clock time spent on this step's target generation.
        model_target: HF model id of the target model that produced the tokens.
        model_draft: HF model id of the draft model used for scoring, or None.
        seed: The seed under which this step was produced.
    """

    step_index: int
    task_id: str
    condition: Condition
    role: Literal["assistant", "tool", "user", "system"]
    text: str
    token_ids: list[int]
    token_strings: list[str]
    token_types: list[TokenType]
    target_logprobs: list[float]
    draft_logprobs: list[float] | None
    acceptance_proxy: list[float] | None
    tool_calls: list[dict[str, Any]] | None
    tool_results: list[dict[str, Any]] | None
    error: str | None
    wall_time_ms: float
    model_target: str
    model_draft: str | None
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return asdict(self)


@dataclass
class TrajectoryMetadata:
    """Metadata header written as the first line of a trajectory JSONL file.

    Attributes:
        task_id: ID of the task this trajectory belongs to.
        condition: Experimental condition label.
        model_target: HF model id of the target model.
        profile_name: Name of the ``ModelProfile`` used.
        seed: Master seed.
        generation_kwargs: Full generation kwargs dict used for ``generate``.
        dataset_version: Version tag of the task set this task came from.
        git_commit_sha: Git commit SHA at run time, or None if unavailable.
        docker_image_tag: Docker image tag, or None if not running in Docker.
    """

    task_id: str
    condition: Condition
    model_target: str
    profile_name: str
    seed: int
    generation_kwargs: dict[str, Any]
    dataset_version: str
    git_commit_sha: str | None = None
    docker_image_tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return asdict(self)


@dataclass
class Trajectory:
    """A complete agent run on a single task.

    Attributes:
        metadata: Header describing the run configuration.
        steps: Ordered list of steps produced by the Agent.
    """

    metadata: TrajectoryMetadata
    steps: list[TrajectoryStep] = field(default_factory=list)

    def append(self, step: TrajectoryStep) -> None:
        """Append a step, asserting its index is correct."""
        assert step.step_index == len(self.steps), (
            f"step_index {step.step_index} does not match current length "
            f"{len(self.steps)}"
        )
        self.steps.append(step)
