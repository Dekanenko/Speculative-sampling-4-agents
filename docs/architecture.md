# Architecture

This document is the canonical technical design for the Speculative
Decoding Validation Framework. If the code drifts from this document,
**update this document in the same commit as the code change**.

---

## 1. Agent Class

A custom wrapper around `transformers.AutoModelForCausalLM`. No
LangChain, no LangGraph — we need full visibility into the generation
loop and token-level instrumentation.

### Constructor parameters
| Param              | Type                     | Purpose                                                   |
|--------------------|--------------------------|-----------------------------------------------------------|
| `model_name`       | `str`                    | HF model id of the **target** model                       |
| `draft_model_name` | `str \| None`            | HF model id of the draft model (optional at runtime)      |
| `tools`            | `list[ToolSpec]`         | Tool schemas exposed to the model                         |
| `max_steps`        | `int`                    | Hard cap on agent loop iterations                         |
| `device`           | `str`                    | `"cuda"`, `"cpu"`, or explicit device string              |
| `seed`             | `int`                    | Master seed — forwarded to torch, numpy, random           |
| `generation_kwargs`| `dict[str, Any]`         | Temperature, top_p, max_new_tokens, etc.                  |

### Public methods
| Method                                   | Returns                  | Description                                                                 |
|------------------------------------------|--------------------------|-----------------------------------------------------------------------------|
| `run(task: Task) -> Trajectory`          | `Trajectory`             | Execute the full agent loop for a task                                      |
| `step(state: AgentState) -> TrajectoryStep` | `TrajectoryStep`      | Execute a single generation + tool-call cycle                               |
| `score_with_draft(step: TrajectoryStep)` | `TrajectoryStep`         | Teacher-force the draft model over `step.token_ids` and attach draft logprobs |
| `reset()`                                | `None`                   | Clear internal state between tasks                                          |

### Stored state
- Target model + tokenizer.
- Optional draft model + tokenizer (same family, same vocab).
- Tool registry mapping tool names to callables.
- Current conversation messages list.
- Per-run RNG generators (seeded).

---

## 2. Trajectory Step Schema

Each step in a trajectory is serialised as JSON with the following
fields:

```json
{
  "step_index": 0,
  "task_id": "simple_weather_001",
  "condition": "simple",
  "role": "assistant",
  "text": "<tool_call>{\"name\": \"get_weather\", ...}</tool_call>",
  "token_ids": [151644, 77091, ...],
  "token_strings": ["<|im_start|>", "assistant", "..."],
  "token_types": ["response", "response", "tool_call", "tool_call", "..."],
  "target_logprobs": [-0.12, -1.44, -0.08, "..."],
  "draft_logprobs":  [-0.31, -2.01, -0.05, "..."],
  "acceptance_proxy": [1.0, 1.0, 0.97, "..."],
  "tool_call": {
    "name": "get_weather",
    "arguments": {"city": "Berlin"}
  },
  "tool_result": {"temp_c": 12, "conditions": "cloudy"},
  "error": null,
  "wall_time_ms": 842.3,
  "model_target": "Qwen/Qwen2.5-7B-Instruct",
  "model_draft":  "Qwen/Qwen2.5-1.5B-Instruct",
  "seed": 42
}
```

### Field types
| Field              | Type                              |
|--------------------|-----------------------------------|
| `step_index`       | `int`                             |
| `task_id`          | `str`                             |
| `condition`        | `Literal["simple","multi_step","error_recovery","long_context"]` |
| `role`             | `Literal["assistant","tool","user","system"]` |
| `text`             | `str`                             |
| `token_ids`        | `list[int]`                       |
| `token_strings`    | `list[str]`                       |
| `token_types`      | `list[Literal["tool_call","reasoning","response"]]` |
| `target_logprobs`  | `list[float]`                     |
| `draft_logprobs`   | `list[float] \| None`             |
| `acceptance_proxy` | `list[float] \| None`             |
| `tool_call`        | `dict \| None`                    |
| `tool_result`      | `dict \| None`                    |
| `error`            | `str \| None`                     |
| `wall_time_ms`     | `float`                           |
| `model_target`     | `str`                             |
| `model_draft`      | `str \| None`                     |
| `seed`             | `int`                             |

A `Trajectory` is simply `{task_id, condition, steps: list[TrajectoryStep], metadata}`.

---

## 3. Token Type Labelling

Labels are assigned by a **delimiter token ID scan**, not by regex on
the decoded string (token boundaries and text boundaries disagree).

### Procedure
1. At construction time, resolve and cache the token IDs of the
   relevant delimiter strings:
   - `<tool_call>` / `</tool_call>`
   - `<think>` / `</think>` (Qwen3 only)
2. Walk `token_ids` left to right, maintaining a mode stack initialised
   to `"response"`.
3. On encountering an opening delimiter, push the new mode
   (`"tool_call"` or `"reasoning"`).
4. On encountering a closing delimiter, pop back to the previous mode.
5. Assign each token the current top-of-stack mode.
6. The delimiter tokens themselves are labelled with the mode they
   **open or close into** (configurable; default: the inner mode).

Edge cases:
- Multi-token delimiters (e.g. `<`, `tool_call`, `>`) require matching
  a **sequence** of token IDs, not a single one.
- Qwen2.5 has no `<think>` tokens — the reasoning label will be absent.

---

## 4. Logprob Capture Pipeline

Two passes per agent step:

### Pass A — Generation (target model)
- Run `model.generate(..., output_scores=True, return_dict_in_generate=True)`.
- Post-process scores into per-token logprobs by indexing `scores[i]`
  at the sampled token id and applying `log_softmax`.
- Store `target_logprobs` aligned 1:1 with newly generated `token_ids`.

### Pass B — Scoring (draft model, teacher-forced)
- Build the **same prefix + generated sequence** using the draft
  tokenizer (must match, since same family).
- Single forward pass: `draft_model(input_ids).logits`.
- Shift logits left by one and gather the logprob of each target token
  under the draft distribution.
- **No autoregressive generation in the draft pass** — it is exactly
  one forward pass, which is cheap and numerically identical to
  running the draft step-by-step for scoring purposes.

### Acceptance proxy
For each token `i`:
```
acceptance_proxy[i] = min(1.0, exp(target_logprob[i] - draft_logprob[i]))
```
This mirrors the Leviathan et al. speculative-decoding acceptance rule
without requiring an actual draft-target verification loop at generation
time. It is an **upper bound** on what a real speculative decoder would
accept, which is exactly what we want for this study.

---

## 5. MLflow Logging Schema

### Run structure
One MLflow **run** per (task_set × model_pair × seed) tuple. Runs are
grouped into **experiments** by study phase (`phase1_trajectories`,
`phase2_acceptance`, `phase4_distillation_eval`, ...).

### Parameters (logged once per run)
- `model_target`, `model_draft`
- `seed`
- `task_set_version`
- `docker_image_tag`
- `git_commit_sha`
- All `generation_kwargs`

### Metrics (logged per step and aggregated)
- `acceptance_rate_overall`
- `acceptance_rate_tool_call`
- `acceptance_rate_reasoning`
- `acceptance_rate_response`
- `acceptance_rate_by_condition.<condition>`
- `mean_target_logprob`, `mean_draft_logprob`
- `steps_per_task`
- `wall_time_ms_per_step`

### Artifacts
```
run_<id>/
  trajectories/
    task_<id>.jsonl
  config/
    generation_kwargs.json
    tools.json
  logs/
    agent.log
  plots/
    acceptance_by_token_type.png
    acceptance_by_condition.png
```

---

## 6. `src/` Layout (as built in Phase 1 W1)

Implemented. Tests mirror this structure under `tests/`.

```
src/
  agent/
    agent.py              # Agent class (run, _step, logprob extraction)
    state.py              # AgentState dataclass
    scanner.py            # DelimiterScanner (token type labelling)
    tools.py              # ToolSpec, ToolRegistry
    profiles/
      base.py             # ModelProfile, DelimiterSet, ParsedToolCall
      _common.py          # encode_delimiter, parse_xml_tool_calls
      qwen25.py           # Qwen2.5 profile factory
      qwen3.py            # Qwen3 profile factory (with <think>)
      llama3.py           # Llama 3 profile stub (NotImplementedError)
      registry.py         # name → factory lookup
  trajectory/
    schema.py             # TrajectoryStep, Trajectory, TrajectoryMetadata
    io.py                 # JSONL read/write with metadata header
  tasks/
    schema.py             # Task dataclass
    registry.py           # YAML loader
    mock_tools.py         # Deterministic weather / calculator / search
    benchmarks/phase1/    # Phase 1 task YAMLs (simple, multi_step, error_recovery)
  config.py               # Constants, GenerationKwargs
```

**Not yet built** (deferred to later in Phase 1 / Phase 2):
- `src/scoring/` — offline draft teacher-forced scoring pass.
- `src/tracking/` — MLflow run helpers.
- `src/cli.py` — entry points.

Tests mirror `src/` exactly under `tests/`, per the code-quality rules.
