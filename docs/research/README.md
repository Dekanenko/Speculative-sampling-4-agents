# Research Notes

Permanent record of measurements, findings, and interpretations. One
file per session-level milestone (a full measurement sweep, a
falsified hypothesis, a direction change). Each note is
self-contained — a future session should not need to read prior
notes to understand any single finding.

## How to write a note

Each note: `docs/research/YYYY-MM-DD-<slug>.md`, with the following
sections, in this order:

- **Context** — what question was being answered and why
- **What got built** — the specific scripts / measurements /
  datasets that produced the numbers
- **Measurements** — raw numbers as tables, not prose
- **Findings** — numbered, each with a one-line headline
- **Interpretation** — what the numbers mean at the project level
- **Next steps** — what to do next given this data

## When to write a note

- A measurement pass completes (acceptance rates, speedup sim,
  eval distributions, ...).
- A hypothesis is tested (confirmed or refuted).
- An analysis reveals something that updates project direction.
- A decision is made that depends on data.

Routine code changes and in-progress work belong in the `CLAUDE.md`
changelog and `docs/progress.md`, not here.

## Index

<!-- Newest first. One line per note: date, title, one-line hook. -->

- [2026-04-18-qwen3-reasoning-sweep.md](2026-04-18-qwen3-reasoning-sweep.md) —
  First full Qwen3 pair (8B target, 1.7B draft) sweep across
  mocks, HotPotQA, and MBPP. **Reasoning tokens are their own
  distribution**: on MBPP reasoning, 10–11% fall below 0.9
  acceptance (vs 3.8% for Qwen2.5 response) and `p1` is 0.554.
  Simulated K=4 speedup drops from 4.90× (Qwen2.5) to 4.72×
  (Qwen3) on MBPP — a 4% loss, concentrated on common English
  discourse words inside `<think>` blocks. This is the
  hypothesis-confirming signal the Qwen2.5 pair did not expose,
  and it defines a sharp target for Phase 4 distillation.
- [2026-04-17-qwen3-readiness-audit.md](2026-04-17-qwen3-readiness-audit.md) —
  Pre-flight audit confirming the pipeline supports Qwen3
  reasoning tokens out of the box. Three Qwen3-specific
  behaviors to remember: prior-turn `<think>` is stripped by
  the template, default rendering makes thinking optional, and
  all think/tool_call tags are `special=False` so round-trip is
  preserved.
- [2026-04-17-first-full-sweep.md](2026-04-17-first-full-sweep.md) —
  First end-to-end measurement on the Qwen2.5 same-family pair
  across mocks, HotPotQA, and MBPP. Speculative decoding hits
  ~97% of the theoretical ceiling on all three families; the
  hypothesis "agents break spec decoding" is not supported by
  this data. Remaining gap is concentrated on common English
  discourse words and turn-boundary syntax.
