"""Run the Phase 1 benchmark task set against a target model.

Loads every YAML task under ``src/tasks/benchmarks/phase1/``,
instantiates the Agent with the requested model + profile, runs each
task, and writes one JSONL trajectory per task to
``runs/<model_slug>/<task_id>.jsonl``.

Usage:
    python scripts/run_phase1.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --profile qwen2.5 \
        --dtype bfloat16

Intended to be run on the GPU server. Safe to re-run — existing
trajectory files under the same model slug are overwritten.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.agent.agent import Agent
from src.config import DEFAULT_MAX_STEPS, GenerationKwargs
from src.tasks.mock_tools import build_mock_tools
from src.tasks.registry import load_task_set
from src.trajectory.io import write_trajectory


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _model_slug(model_name: str) -> str:
    """Turn ``Qwen/Qwen2.5-1.5B-Instruct`` into ``Qwen_Qwen2.5-1.5B-Instruct``."""
    return model_name.replace("/", "_")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF target model id")
    parser.add_argument(
        "--profile", required=True, help="Registered profile name (qwen2.5, qwen3, ...)"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(_DTYPE_MAP),
        help="Target model dtype",
    )
    parser.add_argument(
        "--device", default="cuda", help="Torch device (default: cuda)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Per-step generation cap"
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Per-task step cap"
    )
    parser.add_argument(
        "--tasks-dir",
        default="src/tasks/benchmarks/phase1",
        help="Directory with task YAMLs",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Root directory for trajectory JSONL output",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Master seed"
    )
    parser.add_argument(
        "--dataset-version", default="phase1-v0", help="Dataset version tag"
    )
    return parser.parse_args()


def main() -> None:
    """Run all Phase 1 tasks against a single target model."""
    args = _parse_args()

    tasks = load_task_set(args.tasks_dir)
    if not tasks:
        raise SystemExit(f"No tasks found under {args.tasks_dir}")
    print(f"Loaded {len(tasks)} tasks from {args.tasks_dir}")

    out_root = Path(args.runs_dir) / _model_slug(args.model)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device} ({args.dtype})...")
    agent = Agent(
        model_name=args.model,
        profile_name=args.profile,
        tools=build_mock_tools(),
        seed=args.seed,
        max_steps=args.max_steps,
        generation_kwargs=GenerationKwargs(
            do_sample=False, max_new_tokens=args.max_new_tokens
        ),
        device=args.device,
        torch_dtype=_DTYPE_MAP[args.dtype],
    )
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM allocated: {vram_gb:.2f} GB")

    summary: list[tuple[str, int, int, int, str | None]] = []
    total_t0 = time.perf_counter()
    for task in tasks:
        print()
        print(f"== {task.task_id} [{task.condition}] ==")
        t0 = time.perf_counter()
        traj = agent.run(
            task_id=task.task_id,
            condition=task.condition,
            system_prompt=task.system_prompt,
            user_prompt=task.user_prompt,
            dataset_version=args.dataset_version,
        )
        elapsed = time.perf_counter() - t0

        out_path = out_root / f"{task.task_id}.jsonl"
        write_trajectory(out_path, traj)

        tool_call_tokens = sum(
            s.token_types.count("tool_call") for s in traj.steps
        )
        response_tokens = sum(
            s.token_types.count("response") for s in traj.steps
        )
        reasoning_tokens = sum(
            s.token_types.count("reasoning") for s in traj.steps
        )
        total_tokens = tool_call_tokens + response_tokens + reasoning_tokens
        errors = [s.error for s in traj.steps if s.error]
        err_str = ";".join(errors) if errors else None

        print(
            f"  steps={len(traj.steps)} tokens={total_tokens} "
            f"(tool_call={tool_call_tokens}, response={response_tokens}, "
            f"reasoning={reasoning_tokens}) "
            f"wall={elapsed:.1f}s err={err_str}"
        )
        if traj.steps and traj.steps[-1].text:
            final = traj.steps[-1].text.replace("\n", " ")[:200]
            print(f"  final: {final!r}")
        summary.append(
            (task.task_id, len(traj.steps), total_tokens, int(elapsed * 1000), err_str)
        )

    total_elapsed = time.perf_counter() - total_t0
    print()
    print(f"=== Phase 1 run complete in {total_elapsed:.1f}s ===")
    print(f"Trajectories written under {out_root}")
    for task_id, n_steps, n_tokens, wall_ms, err in summary:
        flag = "!" if err else " "
        print(f"  {flag} {task_id:<36} steps={n_steps} tokens={n_tokens} {wall_ms}ms")


if __name__ == "__main__":
    main()
