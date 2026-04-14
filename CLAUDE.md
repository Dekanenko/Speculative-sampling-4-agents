# Speculative Decoding Validation Framework for LLM Agents

## Project Summary

### Research Question
Standard speculative decoding pairs a small draft model with a large target
model to accelerate token generation. It is well-studied for prose generation.
This project measures whether it holds up for **agentic workloads** — where
outputs are structured (tool call JSON, reasoning chains, error recovery
sequences) — and, if not, fixes it via targeted distillation.

### Core Hypothesis
Acceptance rates for agentic token patterns differ significantly across
**token types** (tool call tokens vs. reasoning tokens vs. response tokens)
and across **experimental conditions** (simple tasks vs. multi-step vs.
error recovery vs. long context), even within same-family model pairs.

### Model Pairs
All pairs share a tokenizer family, a prerequisite for valid speculative
decoding.

| Role      | Draft                     | Target                    | Notes                                        |
|-----------|---------------------------|---------------------------|----------------------------------------------|
| Primary   | Qwen2.5-1.5B-Instruct     | Qwen2.5-7B-Instruct       | Native tool calling, no explicit thinking   |
| Secondary | Qwen3-1.7B                | Qwen3-8B                  | Native tool calling + `<think>` tokens       |
| Future    | Llama-3.2-1B-Instruct     | Llama-3.1-8B-Instruct     | Exploratory                                  |

### Taxonomy
Two-dimensional measurement matrix.

**Token types** (what acceptance rate is measured ON):
- `tool_call`  — structured JSON inside `<tool_call>...</tool_call>`
- `reasoning`  — thinking content inside `<think>...</think>` (Qwen3 only)
- `response`   — final natural language answer

**Conditions** (experimental variables):
- `simple`         — single tool call, short context
- `multi_step`     — 3+ chained tool calls, later steps depend on earlier
- `error_recovery` — tool returned error, agent must adapt
- `long_context`   — 8+ steps, large accumulated history

### Technical Approach
- **Custom `Agent` class** wrapping `HuggingFaceForCausalLM` directly
  (no LangChain / LangGraph — we need full visibility into the generation
  loop and token-level instrumentation).
- **Target model** generates with `output_scores=True` to capture logprobs.
- **Draft model** scores the same sequence via a single teacher-forcing
  forward pass (cheap and correct — no autoregressive generation needed).
- **Acceptance rate proxy:**
  `mean(min(1, exp(target_logprob − draft_logprob)))`
- **Token type labels** assigned by scanning for delimiter token IDs.
- **Experiment tracking:** MLflow.
- **Containerisation:** Docker.
- **CI:** GitHub Actions.

### Project Phases
| Phase | Window    | Focus                                      | Status         |
|-------|-----------|--------------------------------------------|----------------|
| 1     | Wks 1–3   | Foundation: harness, tasks, trajectories   | **In progress** |
| 2     | Wks 4–8   | Measurement: validation, experiments       | Pending        |
| 3     | Wks 9–11  | Analysis & Decision; distillation dataset  | Pending        |
| 4     | Wks 12–18 | Distillation: LoRA fine-tune, re-run       | Pending        |
| 5     | Wks 19–21 | Packaging: preprint, blog, open source     | Pending        |

## Architecture Overview
A custom `Agent` class drives a Hugging Face causal LM through a
tool-calling loop and emits structured `TrajectoryStep` records. Each
step carries the target-model logprobs, the draft-model teacher-forced
logprobs over the same tokens, and per-token type labels. Runs are
logged to MLflow; trajectories are stored as versioned JSONL datasets.
See `docs/architecture.md` for full detail.

## Current Status
**Phase 1, Week 1.** Agent harness implemented **and validated on
real hardware**. Model profiles (Qwen2.5, Qwen3, Llama3 stub) with
swappable delimiter configs, `DelimiterScanner` for token-type
labelling, `Trajectory` schema + JSONL IO, mock tools, YAML task
loader, and the `Agent` class (target-only — draft scoring is
deferred to an offline pass). 32 unit tests pass both locally and
on the EC2 dev server. `scripts/run_phase1.py` has been run
successfully against both **Qwen2.5-1.5B-Instruct** (3.1 GB VRAM)
and **Qwen2.5-7B-Instruct** (15.3 GB VRAM) on a g6.xlarge L4,
producing clean trajectories for all three Phase 1 benchmark tasks.
Next: offline draft scoring pass.

## Rules
This project follows four rule files loaded from `.claude/rules/`:
- @.claude/rules/code-quality.md
- @.claude/rules/documentation.md
- @.claude/rules/git.md
- @.claude/rules/experiments.md

## Changelog
<!-- Updated every session -->

| Date       | Session Summary                                                          |
|------------|--------------------------------------------------------------------------|
| 2026-04-13 | Repository initialised. CLAUDE.md, rules, and docs scaffold created.    |
| 2026-04-13 | Phase 1 agent harness implemented. Profiles, scanner, trajectory IO, mock tools, Agent class, 31 unit tests green. |
| 2026-04-14 | Phase 1 harness validated on g6.xlarge L4: Qwen2.5-1.5B and Qwen2.5-7B ran all three benchmark tasks cleanly; trajectories committed under `runs/`. |
