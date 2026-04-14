# Architecture Decision Records

Every architectural decision is recorded here **in the same session it
is made** — never retroactively. If a decision is reversed, add a new
ADR that supersedes the old one; do not rewrite history.

---

## ADR-001: Custom HF wrapper instead of LangChain / LangGraph
**Date:** 2026-04-13
**Status:** Accepted
**Decision:** Implement a custom `Agent` class that drives
`transformers.AutoModelForCausalLM` directly, rather than building on
top of LangChain, LangGraph, or any other agent framework.
**Alternatives considered:**
- LangChain agents with callbacks to capture intermediate state.
- LangGraph for an explicit state-machine representation.
- Hugging Face `transformers.Agent`.
**Reasoning:** The research question demands **token-level
instrumentation**: per-token logprobs from both target and draft
models, per-token type labels, and full control over the generation
loop. Existing frameworks abstract away the generation call and do
not expose `output_scores` cleanly, and wedging custom logprob capture
into their callback systems would be more code than writing the loop
directly. A thin custom wrapper is simpler, faster, and gives us
exactly the hooks we need.

---

## ADR-002: Qwen2.5 as the primary model pair
**Date:** 2026-04-13
**Status:** Accepted
**Decision:** Use **Qwen2.5-1.5B-Instruct + Qwen2.5-7B-Instruct** as
the primary draft/target pair for all Phase 1 and Phase 2 experiments.
**Alternatives considered:**
- Llama 3.2 1B + Llama 3.1 8B.
- Qwen3 pair as primary.
- DeepSeek R1 distills.
**Reasoning:** Qwen2.5 is the most popular model family in current
agent benchmarks (BFCL, τ-bench), so results are directly comparable
to published work. It ships native tool calling through its chat
template, it has a matching small/large pair in the same tokenizer
family, and the 1.5B draft is small enough to run alongside the 7B
target on a single GPU.

---

## ADR-003: Qwen3 as the secondary model pair
**Date:** 2026-04-13
**Status:** Accepted
**Decision:** Use **Qwen3-1.7B + Qwen3-8B** as the secondary pair,
specifically for measuring acceptance rate on `reasoning` tokens.
**Alternatives considered:**
- Skipping reasoning-token analysis entirely.
- Using DeepSeek R1 distills for thinking tokens.
- Using a separate, non-agentic reasoning model.
**Reasoning:** Qwen3 is the only small open-source model family that
has **both** native tool calling **and** explicit `<think>` tokens in
the same tokenizer family at two sizes. It lets us answer "does
speculative decoding work on reasoning tokens?" without mixing model
families, which would confound the comparison.

---

## ADR-004: DeepSeek R1 distills deprioritised
**Date:** 2026-04-13
**Status:** Accepted
**Decision:** Do **not** include DeepSeek R1 distilled models in the
primary or secondary experimental plan.
**Alternatives considered:**
- Including R1-Distill-Qwen-1.5B as a draft variant.
- Using R1 distills as the reasoning-token pair instead of Qwen3.
**Reasoning:** The R1 distills do not natively support tool calling
through the chat template; getting them to emit structured tool call
JSON requires hand-crafted prompting, which introduces a confound
(prompt engineering quality vs. model capability) that the study is
not designed to measure. Qwen3 gives us native tool calling **and**
explicit thinking tokens, which is a cleaner experimental setup.

---

## ADR-005: Teacher forcing for draft-model scoring
**Date:** 2026-04-13
**Status:** Accepted
**Decision:** Score the draft model by running a **single teacher-forced
forward pass** over the target-generated sequence, rather than running
the draft model autoregressively.
**Alternatives considered:**
- Running a real speculative decoder (draft proposes, target verifies).
- Running the draft autoregressively and aligning outputs.
**Reasoning:** The study measures the **acceptance proxy**
`min(1, exp(target_lp − draft_lp))`, which only requires the draft
model's probability of **each token the target actually emitted**. A
single forward pass gives exactly that, is numerically identical to
step-by-step scoring for this quantity, and is dramatically cheaper
than autoregressive generation. A real speculative decoder is out of
scope for Phase 1 — the goal here is to predict acceptance rates, not
to run a production decoder.

---

## ADR-006: Draft scoring runs offline, not interleaved
**Date:** 2026-04-13
**Status:** Accepted
**Decision:** The Agent writes trajectories containing **only target
logprobs**. Draft-model teacher-forced scoring is performed later, by
a separate script that reads stored trajectory JSONL files and fills
in `draft_logprobs` / `acceptance_proxy` into a new trajectory file.
**Alternatives considered:**
- **Interleaved:** after each agent step, immediately load the draft
  model and teacher-force it over the new tokens before moving on.
**Reasoning:** Phase 4 distillation **requires** re-scoring the
same trajectories with a newly fine-tuned draft model to produce
before/after comparisons on identical data. An offline scoring pass
makes that free — we just re-run the scoring script against the new
draft. Interleaved scoring would force a full re-run of every agent
loop for each new draft variant, which is wasteful. The cost is one
extra script and an intermediate JSONL format, which we need for
reproducibility anyway.

---

## ADR-007: Swappable ModelProfile dataclass per model family
**Date:** 2026-04-13
**Status:** Accepted
**Decision:** Model-family specifics (delimiter token IDs, tool call
parser, reasoning-token support) live in a `ModelProfile` dataclass.
Profiles are registered in `src/agent/profiles/registry.py` as
*factory functions* that take a tokenizer and return a resolved
profile. Switching between Qwen2.5, Qwen3, and Llama3 is a single
argument to `Agent(...)`.
**Alternatives considered:**
- Subclassing `Agent` per model family.
- A dict of hardcoded token IDs keyed on model name.
- Per-model adapter classes with overridden methods.
**Reasoning:** Delimiter token IDs cannot be hardcoded because they
depend on the exact tokenizer instance (different Qwen2.5 releases
may have different merges). A factory function resolving delimiters
against the live tokenizer at construction time guarantees
correctness. Dataclass + factory is simpler than subclassing, easier
to test (the profile is a plain value), and lets tests use fake
tokenizers without touching the Agent class at all.

---

## ADR-008: Token IDs are the single source of truth for labelling
**Date:** 2026-04-13
**Status:** Accepted
**Decision:** Token type labels (`tool_call`, `reasoning`, `response`)
are computed by scanning token ID sequences for delimiter
subsequences. They are **never** computed from the decoded text via
regex.
**Alternatives considered:**
- Regex over the decoded text to find `<tool_call>...</tool_call>`,
  then mapping character offsets back to token indices.
**Reasoning:** BPE token boundaries and text character boundaries
disagree. A single character offset can fall in the middle of a
multi-character token, producing off-by-one errors in the label
array. Matching token ID subsequences is exact and has no
reverse-mapping step. The `DelimiterScanner` handles multi-token
delimiters by matching subsequences, so this works even when
`<tool_call>` tokenises as two or three tokens.
