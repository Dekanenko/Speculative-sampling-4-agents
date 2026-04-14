"""Pure draft-scoring primitives and the trajectory-level orchestrator.

The scorer reconstructs the **exact** multi-turn conversation the
target model saw during an agent run, tokenises it with the draft
tokenizer, runs a single forward pass on the draft model, and
gathers the draft's logprob at every token position the target
generated. The acceptance proxy is then computed per token as
``min(1, exp(target_lp - draft_lp))``.

Correctness depends on three invariants, each enforced at runtime:

1. **Replay invariant** — when the full conversation is rendered via
   the chat template, each step's stored ``token_ids`` must appear
   byte-exactly at the offset corresponding to its ``prefix_len``.
   Checked by :func:`verify_replay_invariant` and raised loudly.
2. **Causal shift** — in a causal LM, ``logits[pos]`` predicts the
   token at ``pos + 1``. So for a token at position ``p`` we read
   logits at position ``p - 1``. Encoded in
   :func:`gather_draft_logprobs`.
3. **Precision parity** — the target side casts bf16 logits to fp32
   before ``log_softmax``; the scorer does the same, so target and
   draft logprobs are directly comparable.

The orchestrator :func:`score_trajectory` is deliberately thin: it
wires the primitives together and handles model I/O. All the
non-trivial logic lives in pure helpers that can be unit-tested
without a model.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from ..trajectory.schema import Trajectory, TrajectoryStep


def reconstruct_messages(
    traj: Trajectory,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Rebuild the conversation and per-step prefix lengths.

    Walks the stored steps in order, decoding each step's tokens
    with ``skip_special_tokens=True`` (matching what the Agent fed
    back into the chat template at runtime) and appending aligned
    tool responses.

    Args:
        traj: The trajectory to replay.
        tokenizer: A tokenizer from the same family as the target
            model — it must produce byte-identical chat template
            output to the one used at trajectory time.

    Returns:
        A tuple ``(messages, prefix_lens)``:

        - ``messages``: the full conversation list, ready to render
          as a single sequence (including every assistant turn and
          every tool response).
        - ``prefix_lens``: for each step ``k``, the number of tokens
          in the prefix the Agent passed to ``generate`` at step
          ``k`` (i.e., the length of the rendered conversation
          **before** step ``k`` with ``add_generation_prompt=True``).
    """
    meta = traj.metadata
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": meta.system_prompt},
        {"role": "user", "content": meta.user_prompt},
    ]
    prefix_lens: list[int] = []
    for step in traj.steps:
        enc = tokenizer.apply_chat_template(
            messages,
            tools=meta.tool_schemas,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        prefix_lens.append(int(enc["input_ids"].shape[1]))

        messages.append(
            {
                "role": "assistant",
                "content": tokenizer.decode(
                    step.token_ids, skip_special_tokens=True
                ),
            }
        )
        if step.tool_results:
            for result in step.tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
    return messages, prefix_lens


def verify_replay_invariant(
    full_ids: list[int],
    traj: Trajectory,
    prefix_lens: list[int],
) -> None:
    """Abort if stored per-step tokens don't sit at the expected offsets.

    This is the single most important correctness check in the
    scorer. If it fails, the downstream gather would silently pull
    logprobs for the wrong tokens, and every acceptance-rate number
    we produce would be garbage.

    Args:
        full_ids: Flat list of token IDs from the full-conversation
            render.
        traj: The trajectory whose steps we're about to score.
        prefix_lens: Per-step prefix lengths from
            :func:`reconstruct_messages`.

    Raises:
        RuntimeError: If any step's ``token_ids`` do not appear at
            the expected offset inside ``full_ids``.
    """
    for step_idx, (step, start) in enumerate(zip(traj.steps, prefix_lens)):
        n = len(step.token_ids)
        actual = full_ids[start : start + n]
        if actual != step.token_ids:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(actual, step.token_ids)) if a != b),
                min(len(actual), n),
            )
            raise RuntimeError(
                f"Replay invariant violated on step {step_idx} of task "
                f"{traj.metadata.task_id!r}: stored token_ids do not match "
                f"full render at positions [{start}, {start + n}). "
                f"First diff at offset {first_diff}: "
                f"actual={actual[first_diff] if first_diff < len(actual) else None}, "
                f"stored={step.token_ids[first_diff] if first_diff < n else None}"
            )


def gather_draft_logprobs(
    log_probs: torch.Tensor,
    token_ids: list[int],
    start: int,
) -> list[float]:
    """Gather the draft model's logprob at each emitted token position.

    Uses the causal shift-by-one: in a causal LM, ``log_probs[p]``
    is the distribution over the token at position ``p + 1``
    conditioned on tokens ``0..p``. So to score the token at
    position ``start + j`` we look at ``log_probs[start - 1 + j]``.

    Args:
        log_probs: A 2D tensor of shape ``(L, V)`` containing the
            full-sequence log-softmax output of the draft model
            already cast to float32.
        token_ids: The step's stored token IDs.
        start: The absolute offset of ``token_ids[0]`` inside the
            full sequence (i.e., the step's prefix length).

    Returns:
        A list of Python floats, one per token in ``token_ids``.

    Raises:
        ValueError: If ``start`` is 0 (the very first token can't be
            scored because there is no position ``-1``). This
            shouldn't happen in practice because every trajectory
            has at least a system message in its prefix.
    """
    if start < 1:
        raise ValueError(
            f"start must be >= 1 for causal shift; got {start}"
        )
    draft_lps: list[float] = []
    for j, token_id in enumerate(token_ids):
        pos = start - 1 + j
        draft_lps.append(float(log_probs[pos, token_id].item()))
    return draft_lps


def compute_acceptance_proxy(
    target_logprobs: list[float],
    draft_logprobs: list[float],
) -> list[float]:
    """Compute the per-token acceptance proxy.

    ``acceptance_proxy[i] = min(1, exp(target_lp[i] - draft_lp[i]))``

    This mirrors the Leviathan et al. speculative-decoding acceptance
    rule without running a real verification loop at generation
    time. It is the quantity the entire research project is built
    around.

    Args:
        target_logprobs: Per-token target-model logprobs from the
            stored trajectory.
        draft_logprobs: Per-token draft-model logprobs from
            :func:`gather_draft_logprobs`.

    Returns:
        A list of floats in ``[0, 1]``, one per token.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(target_logprobs) != len(draft_logprobs):
        raise ValueError(
            f"length mismatch: target={len(target_logprobs)}, "
            f"draft={len(draft_logprobs)}"
        )
    return [
        min(1.0, math.exp(t - d))
        for t, d in zip(target_logprobs, draft_logprobs)
    ]


def score_trajectory(
    traj: Trajectory,
    draft_tokenizer: Any,
    draft_model: Any,
    draft_model_name: str,
    device: str,
) -> Trajectory:
    """Score a trajectory with a draft model in a single forward pass.

    Args:
        traj: A target-produced trajectory with ``target_logprobs``
            populated and ``draft_logprobs`` / ``acceptance_proxy``
            ``None``.
        draft_tokenizer: A tokenizer from the same family as the
            draft model. Must produce the same chat-template output
            as the target (verified via the replay invariant).
        draft_model: A loaded causal LM in eval mode, same family as
            the target.
        draft_model_name: HF model id; recorded into each step's
            ``model_draft`` field.
        device: Torch device string.

    Returns:
        A new ``Trajectory`` with fully-populated ``draft_logprobs``,
        ``acceptance_proxy``, and ``model_draft`` on every step.
    """
    messages, prefix_lens = reconstruct_messages(traj, draft_tokenizer)

    enc_full = draft_tokenizer.apply_chat_template(
        messages,
        tools=traj.metadata.tool_schemas,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    full_ids_tensor: torch.Tensor = enc_full["input_ids"]
    full_ids_list: list[int] = full_ids_tensor[0].tolist()

    verify_replay_invariant(full_ids_list, traj, prefix_lens)

    with torch.no_grad():
        logits = draft_model(full_ids_tensor.to(device)).logits  # (1, L, V)
    # Precision parity with the target side: cast to fp32 before log_softmax.
    log_probs_all = F.log_softmax(logits[0].float(), dim=-1).cpu()  # (L, V)

    new_steps: list[TrajectoryStep] = []
    for step, start in zip(traj.steps, prefix_lens):
        draft_lps = gather_draft_logprobs(log_probs_all, step.token_ids, start)
        accept = compute_acceptance_proxy(step.target_logprobs, draft_lps)
        new_steps.append(
            replace(
                step,
                draft_logprobs=draft_lps,
                acceptance_proxy=accept,
                model_draft=draft_model_name,
            )
        )

    return Trajectory(metadata=traj.metadata, steps=new_steps)
