# 2026-04-18 — Qwen3 reasoning-token sweep (Qwen3-8B + Qwen3-1.7B)

## Context

The Qwen2.5 same-family sweep
(`2026-04-17-first-full-sweep.md`) gave us per-token acceptance
rates at ~97% of the theoretical ceiling across all three families.
That result did not support the simple hypothesis "speculative
decoding breaks on agentic workloads". We concluded that the more
interesting variable was **reasoning tokens**, introduced by the
Qwen3 pair, which the Qwen2.5 models did not emit. The research
question for this session was:

> Do reasoning tokens (`<think>...</think>` blocks) behave like
> response or tool_call tokens on the acceptance-rate distribution,
> or as their own population?

This note records the first full sweep on the Qwen3 pair.

## What got built

No new code — the run used the same runner and scorer as before.
Two pre-flight fixes from the Qwen3 readiness audit
(`2026-04-17-qwen3-readiness-audit.md`) were in place:

- Scanner does not open `tool_call` regions inside `<think>` blocks.
- `parse_xml_tool_calls` strips complete and unterminated `<think>`
  blocks before extracting tool calls.

The run itself was a single `nohup`'d pipeline on a `g6.xlarge`:

- `scripts/run_family.py --family mocks --split phase1-v0 --model Qwen/Qwen3-8B --profile qwen3 --dtype bfloat16`
- `scripts/run_family.py --family hotpotqa --split dev_sample50 ...`
- `scripts/run_family.py --family mbpp --split mbpp-v0 ...`
- Three `scripts/score_draft.py` runs pairing the 8B with
  `Qwen/Qwen3-1.7B` as draft.

Total wall time ~2h49m. Final artefacts:

- `runs/Qwen_Qwen3-8B/…` — 103 raw trajectories
- `runs/scored/Qwen_Qwen3-8B__Qwen_Qwen3-1.7B/…` — 103 scored
  trajectories + `summary.csv`

## Measurements

**Target** Qwen3-8B, **Draft** Qwen3-1.7B, bfloat16, greedy decoding
(`do_sample=False`), `max_new_tokens=2048` per step.

### Token counts

| family | condition | trajectories | total tokens |
|---|---|---:|---:|
| mocks | all | 3 | 1,088 |
| hotpotqa | all | 50 | 51,472 |
| mbpp | all | 50 | 170,342 |
| **grand total** | | **103** | **222,902** |

Reasoning tokens are the dominant type across everything:

| family | condition | reasoning | response | tool_call |
|---|---|---:|---:|---:|
| mocks | simple | 243 | 18 | 19 |
| mocks | multi_step | 275 | 35 | 102 |
| mocks | error_recovery | 338 | 18 | 40 |
| hotpotqa | simple | 13,122 | 864 | 1,695 |
| hotpotqa | multi_step | 18,317 | 753 | 1,973 |
| hotpotqa | error_recovery | 13,024 | 304 | 1,420 |
| mbpp | simple | 32,869 | 18 | 534 |
| mbpp | multi_step | 31,928 | 287 | 1,009 |
| mbpp | long_context | 19,567 | 4 | 224 |

### Central tendency + tail percentiles

Compact summary — `mean`, `p1`, `p5`, `p10` of the per-token
acceptance proxy:

| family | condition | token_type | n | mean | p1 | p5 | p10 |
|---|---|---|---:|---:|---:|---:|---:|
| mbpp | simple | **reasoning** | 32,869 | 0.9651 | **0.556** | **0.742** | **0.874** |
| mbpp | multi_step | **reasoning** | 31,928 | 0.9697 | **0.554** | **0.769** | **0.901** |
| mbpp | long_context | **reasoning** | 19,567 | 0.9715 | 0.590 | 0.789 | 0.904 |
| hotpotqa | multi_step | **reasoning** | 18,317 | 0.9777 | 0.592 | 0.822 | 0.951 |
| hotpotqa | simple | **reasoning** | 13,122 | 0.9808 | 0.632 | 0.846 | 0.970 |
| hotpotqa | error_recovery | **reasoning** | 13,024 | 0.9817 | 0.644 | 0.857 | 0.976 |
| mocks | error_recovery | reasoning | 338 | 0.9913 | 0.735 | 0.953 | 0.999 |
| mocks | multi_step | reasoning | 275 | 0.9898 | 0.631 | 0.959 | 0.999 |
| mocks | simple | reasoning | 243 | 0.9933 | 0.779 | 0.997 | 1.000 |
| mbpp | simple | response | 18 | 1.000 | 1.000 | 1.000 | 1.000 |
| mbpp | multi_step | response | 287 | 0.9889 | 0.724 | 0.942 | 1.000 |
| mbpp | long_context | response | 4 | ... | ... | ... | ... |
| hotpotqa | multi_step | response | 753 | ... | ... | ... | ... |
| hotpotqa | simple | response | 864 | ... | ... | ... | ... |
| mbpp | simple | tool_call | 534 | 0.9981 | 0.967 | 1.000 | 1.000 |
| mbpp | multi_step | tool_call | 1,009 | 0.9995 | 0.999 | 1.000 | 1.000 |
| hotpotqa | simple | tool_call | 1,695 | 0.9937 | 0.816 | 1.000 | 1.000 |

Full numbers in `runs/scored/Qwen_Qwen3-8B__Qwen_Qwen3-1.7B/summary.csv`.

### Fraction of tokens below thresholds

The headline view of the distribution tails:

| family | condition | token_type | <0.5 | <0.9 | <0.99 |
|---|---|---|---:|---:|---:|
| **mbpp** | **simple** | **reasoning** | **0.49%** | **11.50%** | **22.63%** |
| mbpp | multi_step | reasoning | 0.49% | 9.93% | 18.86% |
| mbpp | long_context | reasoning | 0.38% | 9.81% | 18.89% |
| hotpotqa | multi_step | reasoning | 0.22% | 5.93% | 11.26% |
| hotpotqa | simple | reasoning | 0.27% | 4.69% | 8.76% |
| hotpotqa | error_recovery | reasoning | 0.28% | 5.08% | 9.91% |
| mocks | error_recovery | reasoning | 0.00% | 2.96% | 6.80% |
| mocks | multi_step | reasoning | 0.00% | 2.91% | 6.91% |
| mbpp | multi_step | response | 0.35% | 3.48% | 7.32% |
| hotpotqa | multi_step | response | 0.13% | 1.59% | 2.92% |
| hotpotqa | simple | response | 0.12% | 1.50% | 3.82% |
| mbpp | simple | tool_call | 0.00% | 0.56% | 1.12% |
| mbpp | multi_step | tool_call | 0.00% | 0.10% | 0.59% |
| hotpotqa | simple | tool_call | 0.00% | 0.59% | 1.00% |

### Simulated spec-decoding speedup (Monte Carlo, 100 trials)

Expected tokens-emitted per target forward pass:

| family | condition | mean_len | K=2 | K=4 | K=8 |
|---|---|---:|---:|---:|---:|
| mocks | simple | 280 | 2.97 | 4.90 | 8.73 |
| mocks | multi_step | 412 | 2.97 | 4.91 | 8.74 |
| mocks | error_recovery | 396 | 2.97 | 4.89 | 8.61 |
| hotpotqa | simple | 784 | 2.96 | 4.86 | 8.51 |
| hotpotqa | multi_step | 1,052 | 2.95 | 4.82 | 8.38 |
| hotpotqa | error_recovery | 1,475 | 2.95 | 4.84 | 8.42 |
| mbpp | simple | 1,671 | 2.90 | 4.69 | 7.93 |
| mbpp | multi_step | 1,661 | 2.91 | 4.72 | 8.03 |
| mbpp | long_context | 1,980 | 2.91 | 4.71 | 7.99 |

Theoretical ceilings: K=2 → 3.0×, K=4 → 5.0×, K=8 → 9.0×.

### Comparison: Qwen2.5 pair vs Qwen3 pair (same families, same tasks)

Side-by-side on `mbpp/multi_step`:

| metric | Qwen2.5 7B/1.5B | Qwen3 8B/1.7B |
|---|---:|---:|
| reasoning / mean | — | 0.9697 |
| reasoning / p1 | — | **0.554** |
| reasoning / <0.9 fraction | — | **9.93%** |
| response / mean | 0.989 | 0.989 |
| response / p1 | 0.684 | 0.724 |
| response / <0.9 fraction | 3.80% | 3.48% |
| tool_call / mean | 0.998 | 0.9995 |
| tool_call / <0.9 fraction | 0.57% | 0.10% |
| **K=4 simulated speedup** | **4.90** | **4.72** |

Qwen3 is **slightly better** than Qwen2.5 on response and tool_call
tokens on the same MBPP tasks. The overall speedup drops because
the trajectory is now dominated by reasoning tokens (83–99% of
tokens are `reasoning` on MBPP), and reasoning tokens have
materially worse acceptance than the other types.

### Low-acceptance token frequency (tokens with acceptance < 0.5)

`mbpp / reasoning` tail (n = 394 / 84,364 = 0.47% of reasoning tokens):

| count | % of tail | token |
|---:|---:|---|
| **88** | **22.3%** | `Ġthe` |
| 16 | 4.1% | `But` |
| 14 | 3.6% | `Ġis` |
| 13 | 3.3% | `ĠBut` |
| 12 | 3.0% | `Ġfunction` |
| 9 | 2.3% | `.ĊĊ` |
| 9 | 2.3% | `,` |
| 7 | 1.8% | `Ġnot` |
| 6 | 1.5% | `Ġfor`, `'s`, `Ġand`, `.`, `Ġsays` |

`hotpotqa / reasoning` tail (n = 113 / 44,463 = 0.25%):

| count | % of tail | token |
|---:|---:|---|
| 12 | 10.6% | `Ġthe` |
| 5 | 4.4% | `Ġa` |
| 4 | 3.5% | `ĠI`, `ĠThe` |
| 3 | 2.7% | `.`, `Ġis`, `Ġ"` |
| 2 | 1.8% | `Okay`, `So`, `Ġneed`, `Ġlet`, etc. |

### Unterminated-reasoning rate

17 of 50 HotPotQA trajectories (34%) carry the
`unterminated_reasoning` error tag. The pattern: Qwen3 emits
`<think>`, thinks briefly, then emits a `<tool_call>` block and
`<|im_end|>` — **without ever closing `</think>`**. Example:

```
<think> <tool_call>{"name": "get_wiki_page", "arguments": ...}</tool_call><|im_end|>
```

The hardened parser correctly refuses to dispatch the embedded tool
call. The task's eval reports failure (no real dispatch happened)
but the trajectory is still valid per-token measurement data.

## Findings

1. **Reasoning tokens are their own distribution.** The mean
   acceptance on reasoning (~0.97) is ~1 point lower than response
   (~0.99) and 2–3 points lower than tool_call (~0.999), but the
   tail is much heavier: `<0.9` fraction is **10–20× higher** for
   reasoning than for tool_call on the same cell.

2. **MBPP reasoning is the hardest cell.** 9.8–11.5% of reasoning
   tokens fall below 0.9 acceptance; 18.9–22.6% fall below 0.99.
   `p1` is **0.554**, meaning 1% of reasoning tokens have acceptance
   below 56%. This is the single loudest signal in the project to
   date.

3. **Response and tool_call acceptance are _better_ on Qwen3 than
   Qwen2.5** on matched tasks. `tool_call` mean on MBPP rose from
   0.998 to 0.9995 and `<0.9` fraction fell from 0.57% to 0.10%.
   Response acceptance also edged up. The dominant effect on
   speedup is not a regression in the old token types — it is
   introduction of a new, harder-to-accept token type.

4. **Simulated K=4 speedup drops from 4.90× (Qwen2.5 MBPP) to 4.72×
   (Qwen3 MBPP).** A 4% throughput loss on the hardest family. On
   HotPotQA the loss is smaller (4.90 → 4.82, ~2%). On mocks there
   is no regression (4.90 → 4.91). The degree of loss tracks the
   reasoning-token share of each family.

5. **Low-acceptance reasoning tokens cluster on common English
   discourse words.** On MBPP reasoning, `the` alone accounts for
   **22.3%** of the < 0.5 acceptance tail (88 occurrences).
   `But`, `is`, `The`, `,`, `.`, `function`, `not`, `for`, `and`,
   `says` round out the top. This is the same overconfidence
   pattern we saw on Qwen2.5 response tokens, but now far more
   pronounced and concentrated inside `<think>` blocks, because
   the model does most of its natural-language generation there.

6. **Qwen3's reasoning is coherent and multi-step.** Trajectories
   show the model actually using its thinking block productively:
   decomposing the question, deciding which Wikipedia page to
   fetch, comparing facts, deriving the final answer. Typical
   reasoning blocks run 300–2000 tokens per step.

7. **Roughly a third of HotPotQA Qwen3 trajectories hit
   `unterminated_reasoning`.** The model opens `<think>`, thinks,
   emits a tool call, and ends the turn without closing the
   thinking block. Our pre-flight hardening (scanner + parser)
   correctly prevents these from becoming spurious tool dispatches.
   The trajectories still carry valid per-token acceptance data —
   they just don't produce a finish action at the eval level.

## Interpretation

The Qwen2.5 sweep said: "speculative decoding is essentially a
solved problem on same-family agentic workloads". That was a
correct statement **for models without reasoning**. It was also
the less interesting finding.

The Qwen3 sweep says: "reasoning tokens are the one agent-specific
thing that actually moves the acceptance distribution." And they
move it in a specific, interpretable way — the small model
over-commits on common English discourse words inside the
thinking block, where the bigger model maintains more spread over
plausible continuations. The resulting ~4% K=4 speedup loss on
MBPP is modest but real.

**Where this pushes the project**:

- The project's original hypothesis ("agents break speculative
  decoding") is still not fully supported by the data — a 4%
  speedup loss is not "breaking". But the more honest, interesting
  hypothesis is now visible: **reasoning tokens are the
  distribution bottleneck on same-family pairs with thinking
  capability.**
- The distillation target is now extremely well specified: a
  **narrow set of ~100 common English discourse words**
  (`the`, `a`, `is`, `but`, `function`, and friends) where the
  1.7B Qwen3 draft is overconfident relative to the 8B Qwen3
  target, concentrated inside `<think>` blocks. A fine-tune on
  this population alone might close most of the 4% speedup gap.
- Measurement methodology holds up. The per-step scorer (fixed
  for BPE roundtrip), the scanner and parser hardening for
  `<think>` (prevented spurious tool dispatches on ~34% of
  HotPotQA trajectories), and the tail-aware aggregator all
  proved necessary in this sweep. The pre-flight audits paid off.

**What this does not show**:

- Long-form reasoning. Our trajectories are multi-step agentic
  with short reasoning blocks per step (200–2000 tokens). The
  `<think>`-stripping chat template convention means we do not
  measure acceptance on "chain of thought that references prior
  chain of thought". That would require single-turn reasoning
  benchmarks (AIME-style), which is out of scope here.
- Cross-family generalisation. Both the Qwen2.5 and Qwen3 sweeps
  are same-family pairs. A genuine stress test would pair
  Qwen3-8B with a different-family draft. The user has explicitly
  descoped that for now.

## Next steps

- **Optional**: write up a second small note comparing Qwen2.5 and
  Qwen3 side-by-side as the "cross-generation" view. Most of the
  content is already in this note's comparison table.
- **Phase 4 distillation — narrow, well-targeted**. Build a
  distillation dataset from the low-acceptance reasoning tokens
  (the ~400 tokens with acceptance < 0.5 across MBPP + HotPotQA),
  heavily weighted to the `the` / `is` / `but` / `function` cluster.
  LoRA fine-tune Qwen3-1.7B on this set and re-run the same sweep.
  Before/after metric: `<0.9 fraction on reasoning` — if it drops
  from 10% to < 3%, we've proven the fix. Simulated K=4 speedup
  should recover toward 4.90×.
- **Not in scope**: long-form reasoning benchmarks, cross-family
  pairs, or WebArena-scale tasks.
