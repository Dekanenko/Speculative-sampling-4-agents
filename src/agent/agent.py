"""The Agent class: a custom Hugging Face wrapper.

The Agent drives a single target model through a tool-calling loop
and emits fully-reconstructable ``Trajectory`` records. It does not
touch the draft model — draft scoring happens offline, in a separate
pass, from stored trajectories.

Design constraints:
- Token IDs come exclusively from ``generate()`` — never from
  re-tokenising decoded text.
- The prefix/generated split is asserted explicitly to prevent
  off-by-one errors in logprob alignment.
- Target logprobs are computed from the **raw** per-step logits
  (``output.logits``), not the post-processor scores, so they
  reflect what the model actually believed rather than the
  temperature/top-p-warped sampling distribution.
- All floats returned to the trajectory are plain Python floats (no
  ``torch.Tensor`` or numpy types) so JSON serialisation is trivial.
"""

from __future__ import annotations

import json
import random
import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import (
    Condition,
    DEFAULT_MAX_PREFIX_TOKENS,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    GenerationKwargs,
)
from ..trajectory.schema import (
    Trajectory,
    TrajectoryMetadata,
    TrajectoryStep,
)
from .profiles.base import ModelProfile
from .profiles.registry import build_profile
from .scanner import label_tokens
from .state import AgentState
from .tools import ToolRegistry, ToolSpec


class Agent:
    """A custom HF-model wrapper that records full trajectories.

    The Agent owns a target model and tokenizer, a tool registry, and
    a ``ModelProfile`` describing how to parse and label the model's
    output. Each call to :meth:`run` produces one ``Trajectory``.

    Attributes:
        model_name: HF id of the target model.
        profile_name: Name of the ``ModelProfile`` in use.
        seed: Master seed.
        generation_kwargs: Generation kwargs passed to ``model.generate``.
        max_steps: Hard cap on agent loop iterations per task.
        max_prefix_tokens: Soft cap on prompt-prefix length.
        device: Torch device string.
    """

    def __init__(
        self,
        model_name: str,
        profile_name: str,
        tools: list[ToolSpec],
        seed: int = DEFAULT_SEED,
        max_steps: int = DEFAULT_MAX_STEPS,
        generation_kwargs: GenerationKwargs | None = None,
        device: str = "cpu",
        torch_dtype: str | torch.dtype | None = None,
        trust_remote_code: bool = False,
        max_prefix_tokens: int = DEFAULT_MAX_PREFIX_TOKENS,
    ) -> None:
        """Load the target model and resolve the profile.

        Args:
            model_name: HF id of the target model.
            profile_name: Registered profile name (e.g., ``"qwen2.5"``).
            tools: Tool specs exposed to the agent.
            seed: Master seed for torch/numpy/random.
            max_steps: Hard cap on agent loop iterations per task.
            generation_kwargs: Generation kwargs override. Defaults to
                greedy decoding via :class:`GenerationKwargs`.
            device: Torch device string.
            torch_dtype: dtype to load model weights in. Pass
                ``"bfloat16"`` / ``torch.bfloat16`` for 7B+ models that
                will not fit in float32. ``None`` keeps the HF default.
            trust_remote_code: Whether to allow custom modelling code
                from the HF hub (required for some model families).
            max_prefix_tokens: Soft cap on prompt-prefix length. A step
                whose prefix exceeds this ceiling raises ``RuntimeError``
                rather than silently truncating.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.profile_name = profile_name
        self.seed = seed
        self.max_steps = max_steps
        self.generation_kwargs = generation_kwargs or GenerationKwargs()
        self.device = device
        self.max_prefix_tokens = max_prefix_tokens

        self._set_seed(seed)

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        # transformers >=5 renamed `torch_dtype` to `dtype`; the old
        # kwarg still works but emits a deprecation warning, so pass
        # the new name explicitly.
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self._model.eval()

        self._profile: ModelProfile = build_profile(profile_name, self._tokenizer)
        self._tools = ToolRegistry.from_list(tools)

        self._eos_token_ids: list[int] = self._resolve_eos_ids()
        self._pad_token_id: int = self._resolve_pad_id()

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Seed torch, numpy, and random with the given value."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _resolve_eos_ids(self) -> list[int]:
        """Return the list of EOS token IDs to pass to ``generate``.

        We pass both the chat turn terminator (``<|im_end|>``) and the
        absolute end-of-text token so that generation stops on either.
        Profile stop tokens are merged in as well.

        Returns:
            Sorted unique list of EOS token IDs.
        """
        ids: set[int] = set()
        if self._tokenizer.eos_token_id is not None:
            ids.add(int(self._tokenizer.eos_token_id))
        for name in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"):
            tok_id = self._tokenizer.convert_tokens_to_ids(name)
            if isinstance(tok_id, int) and tok_id >= 0:
                ids.add(tok_id)
        for tok_id in self._profile.stop_token_ids:
            ids.add(int(tok_id))
        return sorted(ids)

    def _resolve_pad_id(self) -> int:
        """Return a sane pad token ID for ``generate``.

        Falls back to the first EOS token if the tokenizer has no pad
        token configured (common for decoder-only chat models).
        """
        pad = self._tokenizer.pad_token_id
        if pad is None:
            return self._eos_token_ids[0]
        return int(pad)

    @property
    def profile(self) -> ModelProfile:
        """Return the resolved model profile."""
        return self._profile

    @property
    def tokenizer(self) -> Any:
        """Return the target tokenizer."""
        return self._tokenizer

    def run(
        self,
        task_id: str,
        condition: Condition,
        system_prompt: str,
        user_prompt: str,
        dataset_version: str = "phase1-v0",
    ) -> Trajectory:
        """Execute the full agent loop for a single task.

        Args:
            task_id: Unique task identifier.
            condition: Experimental condition label.
            system_prompt: System message.
            user_prompt: Initial user message.
            dataset_version: Version tag of the task set.

        Returns:
            A populated ``Trajectory``. The last step's ``error`` field
            is set to ``"max_steps_reached"`` if the loop exited without
            a tool-free final response.
        """
        metadata = TrajectoryMetadata(
            task_id=task_id,
            condition=condition,
            model_target=self.model_name,
            profile_name=self.profile_name,
            seed=self.seed,
            generation_kwargs=self.generation_kwargs.as_dict(),
            dataset_version=dataset_version,
        )
        trajectory = Trajectory(metadata=metadata)

        self._set_seed(self.seed)
        state = AgentState()
        state.append_message("system", system_prompt)
        state.append_message("user", user_prompt)

        while not state.finished and state.step_index < self.max_steps:
            step = self._step(state, task_id, condition)
            trajectory.append(step)

        if not state.finished and trajectory.steps:
            last = trajectory.steps[-1]
            last.error = _merge_error(last.error, "max_steps_reached")

        return trajectory

    def _step(
        self,
        state: AgentState,
        task_id: str,
        condition: Condition,
    ) -> TrajectoryStep:
        """Execute one generation + optional tool call.

        Args:
            state: The current agent state (will be mutated).
            task_id: Task ID for the trajectory step record.
            condition: Condition label for the trajectory step record.

        Returns:
            A fully-populated ``TrajectoryStep`` with target logprobs.
        """
        input_ids, attention_mask = self._render_prefix(state.messages)
        prefix_len = int(input_ids.shape[1])
        if prefix_len > self.max_prefix_tokens:
            raise RuntimeError(
                f"Prefix length {prefix_len} exceeds max_prefix_tokens "
                f"{self.max_prefix_tokens} on task {task_id!r}"
            )

        t0 = time.perf_counter()
        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                output_logits=True,
                return_dict_in_generate=True,
                eos_token_id=self._eos_token_ids,
                pad_token_id=self._pad_token_id,
                **self.generation_kwargs.as_dict(),
            )
        wall_time_ms = (time.perf_counter() - t0) * 1000.0

        new_token_ids, target_logprobs = self._extract_new_tokens(
            output, prefix_len
        )
        # Decode twice: once with specials kept (debug-visible trajectory
        # record), once with specials stripped (clean content to feed
        # back into the next turn's chat template).
        # `skip_special_tokens=True` strips sentinels such as <|im_end|>
        # but preserves <tool_call>/</tool_call> because Qwen marks them
        # with special=False in its added-tokens table — verified live
        # against Qwen2.5-1.5B-Instruct.
        text = self._tokenizer.decode(new_token_ids, skip_special_tokens=False)
        message_text = self._tokenizer.decode(
            new_token_ids, skip_special_tokens=True
        )
        token_strings = self._tokenizer.convert_ids_to_tokens(new_token_ids)
        scan = label_tokens(new_token_ids, self._profile.delimiters)

        error: str | None = None
        if len(new_token_ids) == 0:
            error = _merge_error(error, "empty_generation")
        if scan.unterminated is not None:
            error = _merge_error(error, f"unterminated_{scan.unterminated}")

        parsed_calls = self._profile.tool_call_parser(message_text)
        tool_calls_out: list[dict[str, Any]] | None = None
        tool_results_out: list[dict[str, Any]] | None = None
        if parsed_calls:
            tool_calls_out = []
            tool_results_out = []
            for call in parsed_calls:
                tool_calls_out.append(
                    {"name": call.name, "arguments": call.arguments}
                )
                try:
                    result = self._tools.call(call.name, call.arguments)
                except KeyError as exc:
                    result = {"error": f"unknown_tool: {exc}"}
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                tool_results_out.append(result)

        step = TrajectoryStep(
            step_index=state.step_index,
            task_id=task_id,
            condition=condition,
            role="assistant",
            text=text,
            token_ids=new_token_ids,
            token_strings=list(token_strings),
            token_types=list(scan.labels),
            target_logprobs=target_logprobs,
            draft_logprobs=None,
            acceptance_proxy=None,
            tool_calls=tool_calls_out,
            tool_results=tool_results_out,
            error=error,
            wall_time_ms=wall_time_ms,
            model_target=self.model_name,
            model_draft=None,
            seed=self.seed,
        )

        state.append_message("assistant", message_text)
        state.step_index += 1

        if tool_results_out is not None:
            for result in tool_results_out:
                state.append_message("tool", _safe_json_dumps(result))
            state.finished = False
        else:
            state.finished = True

        return step

    def _render_prefix(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the chat template and return ``(input_ids, attention_mask)``.

        In transformers >=5, ``apply_chat_template`` with
        ``return_tensors="pt"`` returns a ``BatchEncoding`` (dict-like)
        rather than a raw tensor, so we request ``return_dict=True``
        explicitly and pull ``input_ids`` / ``attention_mask`` out.

        Args:
            messages: The current conversation messages.

        Returns:
            A tuple of 2D LongTensors, both shape ``(1, prefix_len)``.
        """
        encoded = self._tokenizer.apply_chat_template(
            messages,
            tools=self._tools.schemas(self._profile.tool_schema_formatter),
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        return input_ids, attention_mask

    def _extract_new_tokens(
        self,
        output: Any,
        prefix_len: int,
    ) -> tuple[list[int], list[float]]:
        """Pull generated token IDs and target logprobs out of ``generate`` output.

        The target logprob for token ``i`` is the log-softmax value
        assigned to the sampled token ID at generation step ``i``,
        computed from the **raw** per-step logits so that sampling
        temperature and top-p do not distort the measurement.

        Args:
            output: The return value of ``model.generate`` with
                ``output_logits=True, return_dict_in_generate=True``.
            prefix_len: Number of input (prefix) tokens.

        Returns:
            A tuple ``(new_token_ids, target_logprobs)``.

        Raises:
            AssertionError: If the number of per-step logits does not
                match the number of newly generated tokens.
        """
        full_sequence = output.sequences[0]
        new_ids_tensor = full_sequence[prefix_len:]
        new_token_ids: list[int] = new_ids_tensor.tolist()

        per_step = output.logits
        assert len(per_step) == len(new_token_ids), (
            f"logits length {len(per_step)} != new tokens length "
            f"{len(new_token_ids)}"
        )

        logprobs: list[float] = []
        for step_idx, token_id in enumerate(new_token_ids):
            step_logits = (
                per_step[step_idx][0].detach().to("cpu", dtype=torch.float32)
            )
            step_logprobs = F.log_softmax(step_logits, dim=-1)
            logprobs.append(float(step_logprobs[token_id].item()))

        return new_token_ids, logprobs


def _merge_error(existing: str | None, addition: str) -> str:
    """Combine error tags into a single semicolon-separated string."""
    if existing is None or existing == "":
        return addition
    return f"{existing};{addition}"


def _safe_json_dumps(payload: Any) -> str:
    """Serialise a tool result to JSON, coercing unknown types to strings."""
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(payload, ensure_ascii=False, default=str)
