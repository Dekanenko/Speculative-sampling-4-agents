"""Central configuration constants.

All magic numbers, default hyperparameters, and model identifiers live
here so they can be adjusted in one place and logged as a unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Condition = Literal["simple", "multi_step", "error_recovery", "long_context"]
"""Experimental condition label shared by tasks and trajectories."""


DEFAULT_SEED: int = 42
DEFAULT_MAX_STEPS: int = 10
DEFAULT_MAX_NEW_TOKENS: int = 2048
"""Per-step generation ceiling.

Chosen to accommodate Qwen3-style ``<think>...</think>`` blocks
plus a tool call or final answer in a single step. Qwen2.5 tasks
rarely need more than ~150 tokens per step, so the higher cap is
unused for them; Qwen3 reasoning on HotPotQA bridge or MBPP
multi-step can easily exceed 1000 tokens before the tool call.
"""
DEFAULT_MAX_PREFIX_TOKENS: int = 28_672
"""Hard ceiling on prompt-prefix length, below Qwen2.5's 32 768-token
context window to leave headroom for generated tokens."""


PRIMARY_TARGET_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
PRIMARY_DRAFT_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"

SECONDARY_TARGET_MODEL: str = "Qwen/Qwen3-8B"
SECONDARY_DRAFT_MODEL: str = "Qwen/Qwen3-1.7B"

LLAMA_TARGET_MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_DRAFT_MODEL: str = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass(frozen=True)
class GenerationKwargs:
    """Default generation kwargs for the target model.

    Attributes:
        do_sample: Whether to sample or use greedy decoding.
        temperature: Sampling temperature (used only when sampling).
        top_p: Nucleus sampling parameter (used only when sampling).
        max_new_tokens: Maximum number of new tokens per generation call.
    """

    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    def as_dict(self) -> dict[str, Any]:
        """Return a generation-kwargs dict with sampling fields omitted in greedy mode.

        Transformers warns when ``temperature`` or ``top_p`` are passed
        alongside ``do_sample=False`` because they are ignored. Omitting
        them silences the warning and makes the intent explicit.

        Returns:
            A dict suitable for unpacking into ``model.generate``.
        """
        payload: dict[str, Any] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.do_sample:
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p
        return payload
