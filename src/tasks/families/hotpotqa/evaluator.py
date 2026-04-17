"""Evaluation logic for the HotPotQA family.

Implements the standard HotPotQA answer-level metrics — exact match
(EM, case- and punctuation-insensitive after normalisation) and
token-level F1 — against the gold short answer. Supporting-fact
scoring is intentionally **not** implemented here; it requires
step-level overlap comparison that belongs in a separate analysis
pass.

The agent's predicted answer is extracted with a two-tier fallback:
first, the ``answer`` argument of the last ``finish(...)`` tool call
in the trajectory; if no finish call is present, the ``text`` field
of the final assistant step, stripped of any Qwen-style special
token markers.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")
_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>.*?</tool_call>", re.IGNORECASE | re.DOTALL
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def normalise_answer(text: str) -> str:
    """Apply HotPotQA answer normalisation.

    Lowercases, strips punctuation, removes articles, and collapses
    whitespace — matching the reference implementation in the
    HotPotQA repo.

    Args:
        text: Raw answer text.

    Returns:
        Normalised text.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLE_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def exact_match(predicted: str, gold: str) -> float:
    """Return 1.0 if normalised predicted == gold, else 0.0.

    Args:
        predicted: Model's predicted answer.
        gold: Gold reference answer.

    Returns:
        ``1.0`` or ``0.0``.
    """
    return 1.0 if normalise_answer(predicted) == normalise_answer(gold) else 0.0


def token_f1(predicted: str, gold: str) -> float:
    """Compute token-level F1 between predicted and gold answers.

    Args:
        predicted: Model's predicted answer.
        gold: Gold reference answer.

    Returns:
        Token-level F1 in ``[0, 1]``. Returns 1.0 when both are empty
        post-normalisation, 0.0 when only one is empty.
    """
    pred_tokens = normalise_answer(predicted).split()
    gold_tokens = normalise_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _strip_special_tokens(text: str) -> str:
    """Remove chat template markers and think/tool-call blocks.

    Args:
        text: Raw assistant text.

    Returns:
        Cleaned text suitable for comparison against gold answers.
    """
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _TOOL_CALL_BLOCK_RE.sub(" ", text)
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    return text.strip()


def extract_predicted_answer(trajectory: Any) -> str:
    """Pull the agent's final answer out of a trajectory.

    Looks for the last ``finish(...)`` tool call in the trajectory;
    if present, returns its ``answer`` argument. Otherwise falls
    back to the cleaned text of the final assistant step.

    Args:
        trajectory: A ``Trajectory`` instance (duck-typed — only
            ``steps`` is accessed).

    Returns:
        The predicted answer string. Empty string when no assistant
        text is available at all.
    """
    if trajectory is None or not getattr(trajectory, "steps", None):
        return ""
    steps = trajectory.steps
    # Last finish(...) call wins.
    for step in reversed(steps):
        tool_calls = getattr(step, "tool_calls", None) or []
        for call in reversed(tool_calls):
            if call.get("name") == "finish":
                arguments = call.get("arguments", {}) or {}
                answer = arguments.get("answer")
                if answer is not None:
                    return str(answer).strip()
    # Fallback: last assistant step's cleaned text.
    for step in reversed(steps):
        if getattr(step, "role", None) == "assistant":
            return _strip_special_tokens(getattr(step, "text", "") or "")
    return ""
