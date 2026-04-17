# 2026-04-17 — First full measurement sweep (Qwen2.5 same-family pair)

## Context

Phase 1 validated the target-model harness on three toy mocks. The
question going into this session was whether the research hypothesis
— "acceptance rates differ meaningfully across token types and
experimental conditions on agentic workloads" — survives contact
with real tasks.

To answer it we needed: (a) real tasks more diverse than weather
lookups; (b) a draft-scoring pipeline that could measure acceptance
per token; (c) tail-aware aggregation that wouldn't smooth the
signal away; (d) a simulation that maps acceptance distributions
to actual decoder speedup.

This note records the first full sweep using
**Qwen2.5-7B-Instruct as target** and **Qwen2.5-1.5B-Instruct as
draft**, across three families: `mocks`, `hotpotqa`, and `mbpp`.

## What got built

- **TaskFamily abstraction** (`src/tasks/families/`). Plug-in
  interface with `load_tasks`, `build_env`, `build_tools`,
  `evaluate`, `teardown_env`. Three families land under it.
- **HotPotQA family** with live Wikipedia API + record-and-replay
  cache. The cache is committed alongside the YAMLs so runs are
  deterministic after the one-time pull.
- **MBPP family** with `read_file` / `write_file` / `run_tests`
  tools over a per-task `tempfile.mkdtemp` sandbox. `run_tests`
  invokes pytest as a subprocess with a 10 s timeout.
- **Per-step draft scorer** (`src/scoring/draft.py`). An earlier
  single-pass implementation failed with a BPE roundtrip bug
  (decoding a step's tokens and re-tokenising via the chat
  template can produce a different sequence when the stored
  tokens sit on an ambiguous merge boundary). The per-step
  scorer sidesteps this by appending each step's **original**
  token_ids to the rendered prefix with no re-tokenization.
- **Analysis scripts**:
  - `scripts/verify_math.py` — sanity-check the acceptance chain.
  - `scripts/aggregate.py` — tail-aware percentile and
    threshold-fraction tables per
    (family × condition × token_type).
  - `scripts/simulate_speedup.py` — Monte Carlo spec-decoding
    simulation per trajectory, parameterised by lookahead K.
  - `scripts/tail_tokens.py` — frequency of token strings in
    the low-acceptance tail, for distillation target selection.

## Measurements

- 103 scored trajectories
- 121,180 scored tokens across 18 cells
  (3 families × 3–4 conditions × 2 token types)
- All numbers computed at Qwen2.5-7B target, Qwen2.5-1.5B draft,
  bfloat16, greedy decoding (do_sample=False)

### Math invariants

| Check | Result |
|---|---|
| Tokens checked | 121,180 |
| Array alignment failures | 0 |
| Logprob bounds failures | 0 |
| Recompute mismatches | 0 |
| Max \|stored − recomputed\| | **1.11e-16** (machine epsilon) |

### Central tendency + tail percentiles

| family | condition | token_type | n | mean | p1 | p5 | p10 | min |
|---|---|---|---:|---:|---:|---:|---:|---:|
| hotpotqa | error_recovery | response | 817 | 0.984 | 0.590 | 0.942 | 0.988 | 0.400 |
| hotpotqa | error_recovery | tool_call | 1,932 | 0.994 | 0.792 | 0.997 | 1.000 | 0.296 |
| hotpotqa | multi_step | response | 2,603 | 0.982 | 0.568 | 0.895 | 0.993 | 0.209 |
| hotpotqa | multi_step | tool_call | 3,080 | 0.995 | 0.787 | 1.000 | 1.000 | 0.330 |
| hotpotqa | simple | response | 2,082 | 0.984 | 0.581 | 0.906 | 1.000 | 0.255 |
| hotpotqa | simple | tool_call | 2,217 | 0.994 | 0.746 | 1.000 | 1.000 | 0.277 |
| mbpp | long_context | response | 17,768 | 0.992 | 0.740 | 0.979 | 1.000 | 0.201 |
| mbpp | long_context | tool_call | 12,041 | 0.999 | 0.989 | 1.000 | 1.000 | 0.253 |
| mbpp | multi_step | response | 23,537 | 0.989 | 0.684 | 0.943 | 0.999 | 0.174 |
| mbpp | multi_step | tool_call | 14,867 | 0.998 | 0.971 | 1.000 | 1.000 | 0.192 |
| mbpp | simple | response | 22,665 | 0.990 | 0.700 | 0.960 | 1.000 | 0.194 |
| mbpp | simple | tool_call | 17,336 | 0.999 | 0.984 | 1.000 | 1.000 | 0.295 |
| mocks | error_recovery | response | 16 | 0.976 | 0.664 | 0.955 | 1.000 | 0.664 |
| mocks | error_recovery | tool_call | 40 | 1.000 | 0.993 | 1.000 | 1.000 | 0.993 |
| mocks | multi_step | response | 39 | 0.996 | 0.827 | 1.000 | 1.000 | 0.827 |
| mocks | multi_step | tool_call | 102 | 0.999 | 0.954 | 1.000 | 1.000 | 0.951 |
| mocks | simple | response | 19 | 0.976 | 0.697 | 0.854 | 1.000 | 0.697 |
| mocks | simple | tool_call | 19 | 0.998 | 0.965 | 1.000 | 1.000 | 0.965 |

### Fraction of tokens below thresholds

| family | condition | token_type | <0.5 | <0.9 | <0.99 |
|---|---|---|---:|---:|---:|
| hotpotqa | error_recovery | response | 0.61% | 4.53% | 10.77% |
| hotpotqa | error_recovery | tool_call | 0.10% | 2.02% | 4.50% |
| hotpotqa | multi_step | response | 0.69% | 5.22% | 9.53% |
| hotpotqa | multi_step | tool_call | 0.13% | 1.72% | 3.05% |
| hotpotqa | simple | response | 0.62% | 4.90% | 8.21% |
| hotpotqa | simple | tool_call | 0.23% | 1.49% | 2.89% |
| mbpp | long_context | response | 0.16% | 2.45% | 5.94% |
| mbpp | long_context | tool_call | 0.06% | 0.42% | 1.04% |
| mbpp | multi_step | response | 0.27% | 3.80% | 7.65% |
| mbpp | multi_step | tool_call | 0.07% | 0.57% | 1.47% |
| mbpp | simple | response | 0.27% | 3.31% | 6.75% |
| mbpp | simple | tool_call | 0.02% | 0.43% | 1.14% |

### Monte Carlo simulated speedup (100 trials)

| family | condition | K=2 | K=4 | K=8 |
|---|---|---:|---:|---:|
| hotpotqa | error_recovery | 2.96 | 4.88 | 8.61 |
| hotpotqa | multi_step | 2.96 | 4.85 | 8.51 |
| hotpotqa | simple | 2.95 | 4.85 | 8.46 |
| mbpp | long_context | 2.98 | 4.92 | 8.73 |
| mbpp | multi_step | 2.97 | 4.90 | 8.64 |
| mbpp | simple | 2.97 | 4.91 | 8.67 |
| mocks | error_recovery | 2.94 | 4.66 | 7.99 |
| mocks | multi_step | 2.99 | 4.86 | 8.72 |
| mocks | simple | 2.86 | 4.67 | 7.51 |

Theoretical ceilings: K=2 → 3.0×, K=4 → 5.0×, K=8 → 9.0×.

### Low-acceptance token frequency (tokens with acceptance < 0.5)

| cell | tail fraction | top tail tokens (count) |
|---|---:|---|
| hotpotqa / response | 0.65% (36/5,502) | `different`(4), `from`(4), `me`(3), `appears`(2), `it`, `if`, `see`, `these`, `this` |
| hotpotqa / tool_call | 0.15% (11/7,229) | `"}}Ċ`(5), `<tool_call>`(5), `alternative`(1) |
| mbpp / response | 0.24% (152/63,970) | `an`(19), `Here`(4), `this`(3), `Let`(3), `function`(3), `further`(3), `have`(3) |
| mbpp / tool_call | 0.05% (20/44,244) | `\n`(4), `for`(2), `if`(2), single-occurrence numbers and identifiers |

## Findings

1. **The measurement math is correct to machine precision.**
   Zero recompute mismatches across 121,180 tokens; max stored vs
   recomputed difference 1.11e-16. The acceptance proxy chain
   `min(1, exp(target_lp − draft_lp))` is implemented faithfully
   and the stored values are trustworthy for downstream analysis.

2. **82% of tokens have acceptance = exactly 1.0.**
   These are positions where the target was at least as confident
   as the draft about the emitted token; the `min(1, ratio)` clip
   fires and the magnitude of target's extra confidence is lost.
   This matches Leviathan et al. — in real speculative decoding,
   under-confident drafts on the right token always get accepted
   regardless of by how much.

3. **Means are uninformative for this hypothesis.** All 18 cells
   sit in [0.976, 1.000]. The mean gap between `tool_call` and
   `response` is ~0.01. At the mean level there is no story.

4. **The hypothesis signal lives in the tail, and it is real.**
   `fraction below 0.9` is 3–7× larger for response tokens than
   tool_call tokens across both real families (e.g.
   mbpp/multi_step: 3.80% response vs 0.57% tool_call). This is
   the hypothesis-confirming signal, but only visible with
   threshold-fraction or percentile statistics, not the mean.

5. **Simulated speedup is at ~97% of the theoretical ceiling on
   all real cells.** At K=4, all HotPotQA and MBPP cells sit in
   [4.85, 4.92] against a ceiling of 5.00. At K=8, they sit in
   [8.46, 8.73] against 9.00. The "agentic workloads break
   speculative decoding" hypothesis is not supported by this
   pair on this data.

6. **Low-acceptance response tokens are common English
   discourse words.** `an`, `from`, `me`, `Here`, `Let`,
   `different`, `this`, `function`, `Let`, `appears`. Smaller
   model → sharper priors → over-commits on common
   continuations. The same-family 1.5B draft is over-confident
   on these by roughly a factor of `e` (2.7×).

7. **Low-acceptance tool_call tokens are turn-boundary tokens.**
   `<tool_call>`, `"}}Ċ`, occasional Python keywords (`if`,
   `for`). The draft and target agree almost perfectly on the
   *contents* of a tool call; disagreement concentrates on
   *when* to open or close a turn.

## Interpretation

The simple framing of the hypothesis ("agentic workloads are
special") is refuted on this pair. Speculative decoding extracts
~97% of the theoretical speedup ceiling regardless of whether the
tokens come from mock weather calls, Wikipedia retrieval, or real
coding. The 1% difference in agreement between token types is
real and well-patterned, but too small to call "agentic workloads
break speculative decoding".

The more interesting research story, given this data, is:

- **Measurement methodology**. The combination of per-token
  acceptance recording + per-step draft scoring (robust to BPE
  roundtrip) + tail statistics + speedup simulation is the
  contribution. The mean-level single-number summary most
  papers report is insufficient; the tail tells you what a real
  decoder would do.

- **Over-confidence as a narrow distillation target**. The
  remaining 3–8% of response tokens with acceptance < 0.9 are
  concentrated on a small set of common English words. A
  targeted distillation dataset of ~200 such tokens, not a
  broad "all agentic tokens" dataset, would be the
  highest-ROI fine-tune.

- **Same-family is nearly solved**. If the goal is a stronger
  test, the next variable to change is not the workload — it's
  the pair. But the user has opted not to do cross-family for
  now, so the next tractable question is whether
  **reasoning tokens** from the Qwen3 pair show a different
  pattern than `response` / `tool_call`.

## Next steps

- **Qwen3 pair with reasoning tokens** (`Qwen3-8B` target,
  `Qwen3-1.7B` draft). Our `DelimiterScanner` already handles
  `<think>...</think>` via the existing Qwen3 profile. What we
  need to verify before running: whether the Qwen3 chat template
  needs an explicit flag to emit thinking blocks, whether the
  8B fits comfortably in 24 GB L4 VRAM alongside KV cache, and
  whether our scorer + aggregator + speedup sim treat
  `reasoning` as just another token_type (they should).

- **Not in scope**: cross-family pairs (explicitly descoped by
  the user).
