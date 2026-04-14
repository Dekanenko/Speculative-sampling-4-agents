"""Offline draft-model scoring pass.

Reads target-model trajectories, loads a same-family draft model,
rebuilds the prompt prefix each step saw, runs one teacher-forced
forward pass per trajectory, and fills in ``draft_logprobs`` and
``acceptance_proxy`` on a fresh copy of each step.
"""

from .draft import (
    compute_acceptance_proxy,
    gather_draft_logprobs,
    reconstruct_messages,
    score_trajectory,
    verify_replay_invariant,
)

__all__ = [
    "compute_acceptance_proxy",
    "gather_draft_logprobs",
    "reconstruct_messages",
    "score_trajectory",
    "verify_replay_invariant",
]
