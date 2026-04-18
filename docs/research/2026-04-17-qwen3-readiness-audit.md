# 2026-04-17 — Qwen3 reasoning readiness audit

## Context

After the first full sweep on Qwen2.5 (same-family pair, see
`2026-04-17-first-full-sweep.md`), the user opted to skip
cross-family experiments and instead run the **Qwen3 pair**
(`Qwen3-8B` target, `Qwen3-1.7B` draft) to measure whether
reasoning tokens (`<think>...</think>`) behave differently from
`response` and `tool_call` tokens on agentic workloads.

This is a pre-flight compatibility audit before burning EC2 time:
does the existing pipeline work with Qwen3, or do we need code
changes first?

## What got built

No code written — this is a read-only audit against the current
main. Live probing done with the real `Qwen/Qwen3-1.7B` tokenizer
on Mac.

## Findings

### 1. Delimiters resolve as single special tokens

| delimiter | Qwen3-1.7B token ID |
|---|---|
| `<think>` | 151667 |
| `</think>` | 151668 |
| `<tool_call>` | 151657 |
| `</tool_call>` | 151658 |
| `<\|im_end\|>` | 151645 |

All four scanner delimiters resolve to exactly one token each.
The existing `src/agent/profiles/qwen3.py` factory encodes them
correctly via `encode_delimiter`.

### 2. Scanner labels reasoning tokens correctly

On a realistic assistant turn `<think>...</think><tool_call>...</tool_call>`:
the scanner produces `11 reasoning + 19 tool_call + 2 response`
(the inter-block `\n` + the trailing `<|im_end|>`). Nested region
handling, stack pushes/pops, and `has_reasoning` gating all fire
as designed.

### 3. Tool call parser handles reasoning prefix

`parse_xml_tool_calls` operates on decoded text and searches for
`<tool_call>` blocks; it does not get confused by a preceding
`<think>` block. Parser finds `get_weather({"city": "Berlin"})`
from the full reasoning + tool call string.

### 4. `skip_special_tokens=True` preserves `<think>` / `<tool_call>`

Only `<|im_end|>` is marked `special=True` in the Qwen3 added-tokens
table; `<think>`, `</think>`, `<tool_call>`, `</tool_call>` are all
`special=False`. So the agent's decode-to-content-text pass keeps
all four tags intact while stripping the sentinel. The scorer's
`reconstruct_messages` path works unchanged.

### 5. Chat template **strips `<think>` from prior assistant turns**

This is the most important Qwen3-specific behavior. A full rendered
multi-turn conversation contains **zero** `<think>` tags in prior
assistant turns — the template only retains the current turn's
thinking. Verified live:

```python
msgs = [system, user, asst("<think>...</think>answer"), user, asst("<think>...</think>answer"), user]
rendered = tokenizer.apply_chat_template(msgs, ...)
# rendered.count("<think>") == 0
```

**Why it doesn't break scoring**: the agent and the scorer both
call `apply_chat_template` on the same messages. Both go through
the same stripping logic. Each step's prefix tokens match between
agent and scorer. Per-token target and draft logprobs are
computed against the same prefix.

**What it means for data**: only the CURRENT step's reasoning
tokens make it into the model's context at any given step. This
is Qwen3's intended design (thinking is "consumed" once the turn
produces an answer), so we don't fight it.

### 6. Default `enable_thinking` behavior is permissive

Qwen3's chat template takes an `enable_thinking` kwarg:

- `enable_thinking=True` (or omitted): prefix ends with
  `<|im_start|>assistant\n` — model is *free to emit* `<think>`
  or start directly with the answer.
- `enable_thinking=False`: prefix ends with
  `<|im_start|>assistant\n<think>\n\n</think>\n\n` — explicitly
  closes an empty thinking block, forcing no-thinking mode.

The current default (no kwarg) = `enable_thinking=True`. So we're
already in "thinking optional" mode. Whether the model actually
emits reasoning depends on task prompting.

**No code change needed** to enable reasoning — but if we see on
EC2 that the model rarely emits `<think>`, we'll want to adjust
system prompts to encourage it. That's a prompt-engineering
question, not a code question.

### 7. Full-pipeline end-to-end dry run passes

Built a mocked Agent with the `qwen3` profile and the real Qwen3
tokenizer, scripted a two-step assistant sequence that uses
reasoning + tool call + final answer, and verified:

- Agent.run produced 2 steps with correct label counts:
  - step 0: 11 reasoning + 19 tool_call + 2 response
  - step 1: 15 reasoning + 13 response
- Scorer (with a mocked draft model returning random logits)
  populated `draft_logprobs` and `acceptance_proxy` on every
  reasoning/tool_call/response position with length alignment
  preserved.
- `traj.metadata.family` / `profile_name` / `dataset_split`
  carry through correctly.

No code changes were made to get this working — the existing
profile + scanner + parser + runner + scorer already supports
Qwen3.

## Interpretation

**The pipeline is ready for Qwen3 out of the box.** The three
non-obvious points future sessions should remember:

1. Chat template strips prior-turn `<think>` blocks — this is
   by design, consistent across agent and scorer, and does not
   break replay.
2. Default rendering does not *force* thinking; model chooses
   whether to emit `<think>` blocks.
3. All `<think>` / `<tool_call>` tag tokens are `special=False`,
   so `skip_special_tokens=True` preserves them (needed for
   the message roundtrip fix from the Qwen2.5 session).

Model sizes are compatible with the L4 budget:
- Qwen3-1.7B ≈ 3.4 GB VRAM in bf16 (comparable to Qwen2.5-1.5B's
  3.09 GB). Fits easily.
- Qwen3-8B ≈ 16–18 GB VRAM in bf16 (estimate — slightly larger
  than Qwen2.5-7B's 15.28 GB). Tight but fits on 24 GB L4
  with ~4–6 GB headroom for KV cache at 8k context.

## Post-audit code change

One default bumped to avoid truncating reasoning blocks:

- `DEFAULT_MAX_NEW_TOKENS` in `src/config.py` — 512 → 2048
- `--max-new-tokens` CLI default in `scripts/run_family.py` — 512 → 2048

Qwen2.5 steps typically use < 200 tokens each so the raise is
unused on the prior sweep; Qwen3 reasoning blocks alone can be
1000–2000 tokens, so 512 would have produced
`unterminated_reasoning` errors on most tasks.

## Adversarial-think defence (bug found and fixed in audit)

Red-team probe with a model that describes a tool call inside its
`<think>` block:

```text
<think>I might use <tool_call>{"name": "fake", ...}</tool_call>.</think>
<tool_call>{"name": "real", ...}</tool_call>
```

**Before the fix**: scanner labelled the inner `<tool_call>` tokens
as `tool_call` (should be `reasoning`), and the parser extracted
both the `fake` and `real` calls. The agent would have dispatched
the fake tool from reasoning. Unterminated `<think>` (the model
hit `max_new_tokens` mid-thought) leaked `<tool_call>` matches the
same way.

**Fix**:

1. `src/agent/scanner.py` — only the outermost (`response`) mode
   may open a new region. Once inside `tool_call` or `reasoning`,
   only the matching close is recognised. An apparent `<tool_call>`
   inside `<think>` is treated as literal reasoning content.
2. `src/agent/profiles/_common.py` — `parse_xml_tool_calls` now
   pre-strips complete `<think>...</think>` blocks and drops
   everything after an unterminated `<think>` before running the
   `<tool_call>` regex.

Five new tests under `tests/agent/test_scanner.py` and
`tests/agent/test_profiles.py` lock this in. 101/101 suite still
green.

Downstream evaluators were already robust: HotPotQA's
`_strip_special_tokens` removes `<think>` blocks from fallback
answer extraction, and MBPP's evaluator reads from
`step.tool_calls` which is now built by the fixed parser. No
change needed there.

## Is the stripping OK for divergence measurement?

**Yes — and it is in fact required.** The argument:

- Per-token acceptance requires target and draft logprobs to be
  computed under the **same prefix**. Otherwise the comparison
  is contaminated.
- Qwen3 was trained with the stripping convention. Its
  target-time logprobs at inference are conditioned on
  stripped prefixes by default.
- Our scorer applies the same template, which strips identically.
  Both target and draft see the same context.
- If we retained prior `<think>` blocks we would be giving the
  models context they were never trained to see at inference
  time; the logprobs would be out-of-distribution and the
  acceptance comparison would be noisy in an uninterpretable
  way.

**Caveat for interpretation**: stripping means each step's
reasoning is "fresh" — the model lacks the context of its own
prior chain of thought in a multi-step agent loop. So the
reasoning blocks we measure are shorter and less dependent than
what appears in pure reasoning benchmarks (e.g. AIME single-turn).
This is fine for the project's research question ("is spec
decoding viable on agentic workloads with Qwen3-style thinking")
but would not generalise to claims about long-form reasoning.

If we later want to stress long-form reasoning specifically, the
right move is a single-turn reasoning benchmark (math problems
with no tools) rather than fighting the chat template.

## Next steps

- Run `scripts/run_family.py --model Qwen/Qwen3-8B --profile qwen3`
  against at least the three families on an EC2 g6.xlarge.
- Score with `Qwen/Qwen3-1.7B` draft.
- Extend `scripts/aggregate.py` tables to include the `reasoning`
  token type row in every cell.
- Compare per-(condition × token_type) acceptance distributions
  to the Qwen2.5 sweep; the research question is whether
  `reasoning` behaves like `response`, like `tool_call`, or as
  its own population.
- If the model rarely emits `<think>` on our existing tasks,
  iterate on system prompts to encourage it (or switch to tasks
  where reasoning is clearly required — MBPP and HotPotQA
  bridge are the obvious candidates).
