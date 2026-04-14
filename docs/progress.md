# Progress Log

Running log of work on the Speculative Decoding Validation Framework,
structured by phase. Updated whenever a milestone or significant
implementation decision is reached.

---

## Phase 1 — Foundation (Wks 1–3) — **In progress**
**Goal:** Build agent harness, define benchmark tasks, produce
trajectory dataset v1.
**Deliverable:** Repo + harness running 12+ tasks + 60+ stored trajectories.

### Session log
- **2026-04-13** — Repository scaffold and documentation created.
  `CLAUDE.md`, `.claude/rules/` (code-quality, documentation, git,
  experiments), `docs/architecture.md`, `docs/progress.md`, and
  `docs/decisions.md` are in place. No implementation code yet.
- **2026-04-13** — Phase 1 agent harness implemented. Built
  `src/agent/profiles/` (base dataclasses, `_common` shared parser,
  `qwen25`, `qwen3`, `llama3` stub, registry),
  `src/agent/scanner.py` (subsequence-matching delimiter state
  machine), `src/agent/tools.py` (`ToolSpec`, `ToolRegistry`),
  `src/agent/state.py` (`AgentState`), `src/agent/agent.py`
  (`Agent` class with `run`/`_step` loop and target-only logprob
  capture), `src/trajectory/` (schema + JSONL IO with metadata
  header), `src/tasks/` (YAML task loader, three Phase 1 benchmark
  tasks covering simple / multi_step / error_recovery, deterministic
  mock tools), and `src/config.py`. Added 31 unit tests under
  `tests/` mirroring `src/`; all green. Installed `torch` and
  `pytest` into the `sda` conda env. No real models loaded (Mac dev
  machine) — runtime validation is blocked on server access.

---

## Phase 2 — Measurement (Wks 4–8) — Pending
**Goal:** Validation framework, acceptance rate experiments,
statistical analysis.
**Deliverable:** MLflow results, Docker image, Go/No-Go memo.

---

## Phase 3 — Analysis & Decision (Wks 9–11) — Pending
**Goal:** Interpret results, classify outcome, generate distillation
dataset.
**Deliverable:** Analysis report, dataset on Hugging Face Hub.

---

## Phase 4 — Distillation (Wks 12–18) — Pending
**Goal:** LoRA fine-tune draft model, re-run experiments, latency and
quantisation study.
**Deliverable:** Fine-tuned model on Hugging Face Hub, before/after
results.

---

## Phase 5 — Packaging (Wks 19–21) — Pending
**Goal:** Clean codebase, arXiv preprint, blog series, open source release.
