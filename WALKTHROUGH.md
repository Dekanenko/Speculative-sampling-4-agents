# End-to-End Walkthrough

> Temporary explainer document. Not part of the published docs.

This file walks through what actually happens when you call
`Agent.run(...)` on a single task, from the moment the tokenizer is
first asked to render a chat prompt to the moment a `Trajectory` is
serialised to JSONL. It focuses on the `Agent` class and names every
collaborator it touches.

---

## 1. The big picture

The project measures whether speculative decoding is viable for
**agentic workloads**. To answer that, we need, for every token the
*target* model emits, two logprobs:

- `target_lp[i]` — what the large model thought of token `i`
- `draft_lp[i]` — what the small model would have thought of the
  same token in the same position

The acceptance proxy is then `min(1, exp(target_lp − draft_lp))`.

The full system splits that into two phases:

1. **Online phase (this code)**: run a real agent loop using the
   *target* model, record every generated token plus its target
   logprob, and label each token with its type
   (`tool_call` / `reasoning` / `response`). Dump to JSONL.
2. **Offline phase (not yet built)**: load a draft model, read the
   JSONL files, and teacher-force the draft over each stored token
   sequence in a single forward pass. Write the draft logprobs back
   out as enriched JSONL.

The Agent is the heart of phase 1. Everything in `src/` exists to
support it.

---

## 2. File map

```
src/
  config.py                       — constants, GenerationKwargs
  agent/
    agent.py                      — the Agent class (run loop)
    state.py                      — AgentState (messages, step index)
    scanner.py                    — DelimiterScanner (token type labels)
    tools.py                      — ToolSpec, ToolRegistry
    profiles/
      base.py                     — ModelProfile, DelimiterSet, ParsedToolCall
      _common.py                  — encode_delimiter, parse_xml_tool_calls
      qwen25.py, qwen3.py, llama3.py
      registry.py                 — name → factory mapping
  trajectory/
    schema.py                     — TrajectoryStep, Trajectory, TrajectoryMetadata
    io.py                         — JSONL read/write
  tasks/
    schema.py                     — Task dataclass
    registry.py                   — YAML task loader
    mock_tools.py                 — deterministic weather/calculator/search tools
    benchmarks/phase1/*.yaml      — actual Phase 1 task definitions
```

Everything under `tests/` mirrors this tree exactly.

---

## 3. Key idea #1: ModelProfile makes families swappable

Different model families emit different delimiters. Qwen2.5 uses
`<tool_call>...</tool_call>` but no thinking tokens. Qwen3 adds
`<think>...</think>`. Llama 3 uses `<|python_tag|>` entirely.

Rather than subclassing `Agent` per family, we put family-specific
knowledge into a plain dataclass:

```python
@dataclass(frozen=True)
class ModelProfile:
    name: str
    delimiters: DelimiterSet            # token-ID sequences
    tool_call_parser: Callable[[str], list[ParsedToolCall]]
    stop_token_ids: list[int]
    supports_reasoning: bool
```

The registry stores **factory functions**, not resolved profiles:

```python
_REGISTRY = {
    "qwen2.5": qwen25.build,
    "qwen3":   qwen3.build,
    "llama3":  llama3.build,
}
```

Each factory takes the actual tokenizer and encodes delimiter strings
to token IDs against that tokenizer:

```python
def build(tokenizer):
    return ModelProfile(
        delimiters=DelimiterSet(
            tool_call_open  = encode_delimiter(tokenizer, "<tool_call>"),
            tool_call_close = encode_delimiter(tokenizer, "</tool_call>"),
            ...
        ),
        tool_call_parser = parse_xml_tool_calls,
        ...
    )
```

**Why factories, not frozen profiles?** Because different Qwen2.5
releases could have different token merges, so hardcoding token IDs
is wrong. Resolving against the live tokenizer guarantees
correctness.

Swapping families at the call site is trivial:

```python
Agent(model_name="Qwen/Qwen2.5-7B-Instruct", profile_name="qwen2.5", ...)
Agent(model_name="Qwen/Qwen3-8B",            profile_name="qwen3",   ...)
```

---

## 4. Key idea #2: Token IDs are the source of truth

A very tempting approach would be to label tokens by running a regex
over the decoded text (`<tool_call>...</tool_call>`) and mapping
character offsets back to token indices. **This is wrong.** BPE token
boundaries and text character boundaries disagree — a single character
offset can land in the middle of a multi-character token, producing
off-by-one errors.

Instead, the `DelimiterScanner` walks **token IDs** left to right,
maintaining a stack of current modes:

```
mode_stack = ["response"]           # start in response mode
for each token:
    if token IDs starting here match a CLOSE delimiter
       and the top of stack is the matching mode:
           label the delimiter tokens with the current mode
           pop the stack
    elif token IDs starting here match an OPEN delimiter:
           push the new mode
           label the delimiter tokens with the new (inner) mode
    else:
           label the token with the top-of-stack mode
```

Delimiters are matched as **subsequences**, so the scanner works
whether `<tool_call>` tokenises as one token (Qwen: yes, id 151657)
or three (some fictional tokenizer: `<`, `tool_call`, `>`).

Unterminated regions (e.g., the model emitted `<tool_call>` but hit
`max_new_tokens` before `</tool_call>`) are detected: the stack still
has more than one entry at EOF, and `ScanResult.unterminated` is set
to the leftover mode. The Agent turns that into `step.error =
"unterminated_tool_call"`.

---

## 5. Key idea #3: Two passes per step for logprobs

The target model runs `generate()` with `output_scores=True,
return_dict_in_generate=True`. The return value is a dict-like object
with:

- `sequences` — the full `prefix + new_tokens` tensor
- `scores` — a tuple of length `num_new_tokens`, each element a
  tensor of shape `(batch, vocab)` containing the raw logits at that
  generation step

To get the target logprob for new token `i`:

```python
logits_i = output.scores[i][0]                    # shape (vocab,)
logprobs_i = F.log_softmax(logits_i, dim=-1)      # numerically stable
target_lp_i = logprobs_i[new_token_ids[i]].item()
```

We use `log_softmax`, not `log(softmax(...))`, because the acceptance
proxy is the *difference* of logprobs and because `softmax` of a
150k-vocab logit vector produces values small enough that taking
`log()` afterwards loses precision.

We store the list of floats in the trajectory step. The draft model
is **never touched** in this phase — draft scoring happens later,
offline, from the stored JSONL (see ADR-006).

---

## 6. Key idea #4: Tools are baked into the prompt

`model.generate()` doesn't take a `tools=` argument. Tools are a
*prompt-engineering* convention: the chat template reads the tool
schemas and renders them into the system prompt as a `<tools>` block,
then tells the model to reply with `<tool_call>...</tool_call>`
blocks.

So tool injection happens at prompt-rendering time inside
`_render_prefix`:

```python
encoded = tokenizer.apply_chat_template(
    messages,
    tools=self._tools.schemas(),       # flat Qwen format
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
)
input_ids      = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
```

The tool schema format matters and **differs per family**: Qwen wants
the flat form `{"name", "description", "parameters"}`; the OpenAI
wrapper `{"type": "function", "function": {...}}` also "works" but
leaks the wrapper verbatim into the `<tools>` block, which is
off-distribution for Qwen. Llama 3.1 expects a different shape again.

So tool-schema formatting lives on the profile, not on `ToolSpec`.
`ModelProfile.tool_schema_formatter` is a `Callable[[name, desc,
params], dict]`. The Qwen profiles share `qwen_flat_tool_schema` from
`_common.py`; Llama's profile will define its own when it's wired up.
`ToolRegistry.schemas(formatter)` takes the formatter from the active
profile at render time.

After the model emits a `<tool_call>` block, we parse it out of the
decoded text (`parse_xml_tool_calls` → `[ParsedToolCall]`) and
dispatch via `ToolRegistry.call(name, arguments)`. The result dict
comes back as the next `{"role": "tool"}` message. Qwen's chat
template wraps it in `<tool_response>...</tool_response>` and puts it
inside a `user` turn (I was surprised by this — not an `assistant`
turn, a `user` turn — but that's just how the template works).

---

## 7. The Agent loop, step by step

Here's what happens when you call:

```python
agent = Agent(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    profile_name="qwen2.5",
    tools=build_mock_tools(),
    seed=42,
)
traj = agent.run(
    task_id="simple_weather_berlin",
    condition="simple",
    system_prompt="You are a helpful assistant with access to tools.",
    user_prompt="What is the weather in Berlin?",
)
```

### a. `__init__`

1. Imports `transformers` lazily (keeps test-time import cheap).
2. Sets torch/numpy/random seeds.
3. Loads `AutoTokenizer.from_pretrained(model_name)`.
4. Loads `AutoModelForCausalLM.from_pretrained(model_name).to(device)`
   and `.eval()`s it.
5. Calls `build_profile("qwen2.5", tokenizer)` — the qwen25 factory
   runs, encoding `<tool_call>` / `</tool_call>` into their actual
   token IDs (151657, 151658 for Qwen2.5).
6. Builds a `ToolRegistry` from the list of `ToolSpec`s.

### b. `run()`

1. Creates a `TrajectoryMetadata` header (model name, profile name,
   seed, generation kwargs, dataset version).
2. Creates an empty `Trajectory`.
3. **Re-seeds** torch/numpy/random so this run is independent of
   previous runs on the same Agent instance.
4. Creates an `AgentState` and appends `system` + `user` messages.
5. Loops: while `not state.finished and state.step_index <
   max_steps`, call `_step()` and append the returned
   `TrajectoryStep` to the trajectory.
6. Returns the trajectory.

### c. `_step()` — the heart of everything

For each iteration:

#### c.1. Render the prefix
- Call `_render_prefix(state.messages)`.
- This invokes `tokenizer.apply_chat_template` with the current
  messages **plus the tool schemas** and `add_generation_prompt=True`.
- Returns `(input_ids, attention_mask)` tensors. `prefix_len =
  input_ids.shape[1]`.

#### c.2. Generate
- `with torch.no_grad(): model.generate(input_ids, attention_mask,
  output_scores=True, return_dict_in_generate=True,
  **generation_kwargs)`.
- Wall-clock time is captured around the call for
  `step.wall_time_ms`.

#### c.3. Slice + extract logprobs
- `_extract_new_tokens(output, prefix_len)`:
  - `new_token_ids = output.sequences[0][prefix_len:].tolist()`
  - Assert `len(output.scores) == len(new_token_ids)` (this catches
    off-by-one bugs immediately).
  - For each position `i`, take `log_softmax(scores[i][0])` and index
    at `new_token_ids[i]`. Append to a Python list of floats.
  - Returns `(new_token_ids, target_logprobs)`.

#### c.4. Decode + label
Two decodes, because of a real gotcha:

- `text = tokenizer.decode(new_token_ids, skip_special_tokens=False)`
  — kept verbatim (including the trailing `<|im_end|>` that ended
  generation) and stored in `trajectory.text` for debugging.
- `message_text = tokenizer.decode(new_token_ids,
  skip_special_tokens=True)` — strips the sentinels (`<|im_end|>`,
  `<|im_start|>`, …) but **keeps** `<tool_call>` and `</tool_call>`
  because Qwen marks those with `special=False`. This is the version
  that gets appended to `state.messages`.

Why two decodes? If you feed `message_text` back into the chat
template with the sentinels still in it, the template adds its own
`<|im_end|>` on top and you end up with a doubled terminator like
`...</tool_call><|im_end|><|im_end|>`. That's malformed and confuses
the model on subsequent turns. Stripping sentinels before storing the
content solves it cleanly.

`token_strings = tokenizer.convert_ids_to_tokens(new_token_ids)` —
used only for debugging, never for logic.

`scan = label_tokens(new_token_ids, profile.delimiters)` — the
scanner walks the IDs and produces per-token type labels.

If the scan is unterminated, `step.error =
f"unterminated_{scan.unterminated}"`.

#### c.5. Parse tool calls + dispatch
- `parsed_calls = profile.tool_call_parser(message_text)` — regex the
  cleaned decoded text for `<tool_call>{...}</tool_call>` blocks.
- For each parsed call, call `ToolRegistry.call(name, arguments)`
  and collect the result dict. If the tool name is unknown, the result
  is `{"error": "Unknown tool: ..."}`.
- `tool_calls_out` and `tool_results_out` are stored as **lists**,
  because Qwen can emit multiple `<tool_call>` blocks in a single
  assistant turn.

#### c.6. Build the TrajectoryStep
- Populate every field of `TrajectoryStep` with what we gathered:
  token IDs, strings, types, target logprobs, the decoded text,
  parsed tool calls, results, error tag, timing, model id, seed.
- `draft_logprobs` and `acceptance_proxy` are left as `None` — those
  get filled in by the offline draft scoring pass later.

#### c.7. Update state
- Append `{"role": "assistant", "content": message_text}` to
  `state.messages` (the sentinel-stripped version — see c.4).
- `state.step_index += 1`.
- If there were tool calls:
  - Append one `{"role": "tool", "content": json.dumps(result)}`
    message per result.
  - `state.finished = False` — loop continues.
- Else:
  - `state.finished = True` — loop exits.
- Return the step.

### d. After the loop

`run()` returns the `Trajectory`. You then call
`write_trajectory(path, trajectory)` to dump it to JSONL:

```
line 0: {"__kind__": "metadata", "task_id": ..., "model_target": ..., ...}
line 1: {"__kind__": "step", "step_index": 0, "token_ids": [...], ...}
line 2: {"__kind__": "step", "step_index": 1, ...}
...
```

The `__kind__` discriminator lets the reader dispatch to the right
dataclass.

---

## 8. Data structures (quick reference)

```
Task                   ── one benchmark problem (YAML)
  task_id, condition, system_prompt, user_prompt, allowed_tools

ToolSpec               ── one tool
  name, description, parameters (JSON schema), fn (callable)

ToolRegistry           ── name → ToolSpec, dispatches calls

AgentState             ── per-run mutable state
  messages, step_index, finished

ModelProfile           ── family-specific knowledge (frozen)
  name, delimiters, tool_call_parser, stop_token_ids, supports_reasoning

DelimiterSet           ── token-ID sequences (frozen)
  tool_call_open, tool_call_close, think_open, think_close

TrajectoryStep         ── one generation turn, fully reconstructable
  step_index, task_id, condition, role, text,
  token_ids, token_strings, token_types,
  target_logprobs, draft_logprobs, acceptance_proxy,
  tool_calls, tool_results, error,
  wall_time_ms, model_target, model_draft, seed

TrajectoryMetadata     ── JSONL file header
  task_id, condition, model_target, profile_name, seed,
  generation_kwargs, dataset_version,
  git_commit_sha, docker_image_tag

Trajectory             ── metadata + list of steps
```

---

## 9. What is NOT here yet

Deliberately deferred:

- **`src/scoring/`** — offline draft teacher-forced scoring pass.
  Reads JSONL trajectories, loads a draft model, runs one forward
  pass per step, writes enriched trajectories.
- **`src/tracking/`** — MLflow run logging.
- **`src/cli.py`** — a top-level entry point that takes a task set,
  runs the Agent over each task, writes trajectories, and logs to
  MLflow.
- **Llama 3 profile** — the factory raises `NotImplementedError`.
  Delimiter resolution and tool-call parser will be written once the
  Qwen pipeline is validated on a real server.

Actual model runs are blocked on a GPU server — Mac dev machine only
handles tokenizer-level verification, which is what all current unit
tests use.

---

## 10. Where to start reading the code

Recommended order:

1. `src/agent/profiles/base.py` — the data shapes everything uses.
2. `src/agent/profiles/qwen25.py` — a concrete profile factory.
3. `src/agent/scanner.py` — pure logic, unit-testable in isolation.
4. `src/trajectory/schema.py` — the output data structure.
5. `src/agent/agent.py` — the class that wires all of the above
   together. Start at `run()`, then read `_step()` top to bottom.
6. `src/agent/tools.py` and `src/tasks/mock_tools.py` — how the
   Agent's tools are defined and dispatched.

The unit tests in `tests/agent/test_scanner.py` are a particularly
good way to build intuition for the delimiter state machine.
