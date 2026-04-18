"""Token type labelling via delimiter subsequence matching.

The scanner walks a sequence of token IDs and assigns each token a
``TokenType`` label (``"tool_call"``, ``"reasoning"``, ``"response"``)
based on where it falls relative to the profile's delimiter tokens.

Labels are computed from token IDs directly, never from decoded text.
BPE token boundaries and text character boundaries disagree, so regex
over decoded text produces off-by-one errors.

Design notes:
- A single delimiter string may be split across multiple tokens. The
  scanner matches delimiters as *subsequences* of token IDs.
- A stack tracks nested regions so that, e.g., a ``<think>`` inside
  nothing opens a reasoning region and the matching ``</think>``
  closes it.
- Unterminated regions are labelled with the current mode up to the
  end of the sequence, and ``ScanResult.unterminated`` is set to the
  leftover mode so the caller can record an error.
"""

from __future__ import annotations

from dataclasses import dataclass

from .profiles.base import DelimiterSet, TokenType


@dataclass(frozen=True)
class ScanResult:
    """The output of a delimiter scan.

    Attributes:
        labels: Per-token type labels, aligned 1:1 with the input IDs.
        unterminated: The innermost mode still open at end-of-input,
            or ``None`` if the scan ended cleanly in ``"response"``.
    """

    labels: list[TokenType]
    unterminated: TokenType | None


def label_tokens(
    token_ids: list[int],
    delimiters: DelimiterSet,
) -> ScanResult:
    """Label every token with its ``TokenType``.

    Scanning rule: only the outermost (``"response"``) mode will
    open a nested region. Once we are inside ``"tool_call"`` or
    ``"reasoning"``, only the matching close is recognised — any
    apparent opener is treated as literal content of the outer
    region. This prevents a model that describes a tool call
    inside a ``<think>`` block from having those tokens labelled
    as ``tool_call`` (they should be ``reasoning``) and mirrors the
    defence the tool-call parser uses to reject example calls
    buried in reasoning.

    Args:
        token_ids: Flat list of token IDs to label.
        delimiters: Resolved delimiter set from a ``ModelProfile``.

    Returns:
        A ``ScanResult`` with one label per input token.
    """
    labels: list[TokenType] = ["response"] * len(token_ids)
    mode_stack: list[TokenType] = ["response"]

    openers: list[tuple[list[int], TokenType]] = [
        (delimiters.tool_call_open, "tool_call"),
    ]
    if delimiters.has_reasoning():
        assert delimiters.think_open is not None
        openers.append((delimiters.think_open, "reasoning"))

    closers_by_mode: dict[TokenType, list[int]] = {
        "tool_call": delimiters.tool_call_close,
    }
    if delimiters.has_reasoning():
        assert delimiters.think_close is not None
        closers_by_mode["reasoning"] = delimiters.think_close

    i = 0
    n = len(token_ids)
    while i < n:
        current = mode_stack[-1]

        if current == "response":
            # In the outer mode we may open a new region.
            matched = False
            for open_seq, open_mode in openers:
                if _matches_at(token_ids, i, open_seq):
                    for j in range(len(open_seq)):
                        labels[i + j] = open_mode
                    mode_stack.append(open_mode)
                    i += len(open_seq)
                    matched = True
                    break
            if matched:
                continue
        else:
            # Inside tool_call or reasoning — only the matching close
            # exits this region. Any apparent opener is literal content.
            close_seq = closers_by_mode[current]
            if _matches_at(token_ids, i, close_seq):
                for j in range(len(close_seq)):
                    labels[i + j] = current
                mode_stack.pop()
                i += len(close_seq)
                continue

        labels[i] = current
        i += 1

    unterminated: TokenType | None = None
    if len(mode_stack) > 1:
        unterminated = mode_stack[-1]

    return ScanResult(labels=labels, unterminated=unterminated)


def _matches_at(token_ids: list[int], offset: int, pattern: list[int]) -> bool:
    """Return True if ``pattern`` appears in ``token_ids`` starting at ``offset``."""
    if offset + len(pattern) > len(token_ids):
        return False
    for k, pid in enumerate(pattern):
        if token_ids[offset + k] != pid:
            return False
    return True
