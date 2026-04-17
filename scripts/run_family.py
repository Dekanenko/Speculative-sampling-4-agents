"""Run every task in a family's split against a target model.

Dispatches by family name: loads the split, builds per-task env +
tools via the family, executes the Agent loop, and writes both the
raw trajectory JSONL and an ``.eval.json`` sidecar with the family's
evaluation result.

Usage:
    PYTHONPATH=. python scripts/run_family.py \
        --family hotpotqa \
        --split dev_sample50 \
        --model Qwen/Qwen2.5-7B-Instruct \
        --profile qwen2.5 \
        --dtype bfloat16

Output layout:
    runs/<model_slug>/<family>/<split>/<task_id>.jsonl
    runs/<model_slug>/<family>/<split>/<task_id>.eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.agent.agent import Agent
from src.config import DEFAULT_MAX_STEPS, GenerationKwargs
from src.tasks import get_family
from src.tasks.families.base import EvaluationResult
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
    parser.add_argument("--family", required=True, help="Task family name")
    parser.add_argument("--split", required=True, help="Split identifier")
    parser.add_argument("--model", required=True, help="HF target model id")
    parser.add_argument(
        "--profile", required=True, help="Registered model profile name"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(_DTYPE_MAP),
        help="Target model dtype",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to transformers from_pretrained",
    )
    return parser.parse_args()


def _write_eval(path: Path, result: EvaluationResult) -> None:
    """Write an EvaluationResult as a .eval.json sidecar."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)


def main() -> None:
    """Run every task in a family's split and write trajectories + eval."""
    args = _parse_args()
    family = get_family(args.family)
    tasks = family.load_tasks(args.split)
    if not tasks:
        raise SystemExit(
            f"No tasks loaded for family={args.family!r} split={args.split!r}"
        )
    print(f"Loaded {len(tasks)} tasks from family={args.family} split={args.split}")

    out_dir = (
        Path(args.runs_dir)
        / _model_slug(args.model)
        / args.family
        / args.split
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} on {args.device} ({args.dtype})...")
    agent = Agent(
        model_name=args.model,
        profile_name=args.profile,
        seed=args.seed,
        max_steps=args.max_steps,
        generation_kwargs=GenerationKwargs(
            do_sample=False, max_new_tokens=args.max_new_tokens
        ),
        device=args.device,
        torch_dtype=_DTYPE_MAP[args.dtype],
        trust_remote_code=args.trust_remote_code,
    )
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM allocated: {vram:.2f} GB")

    total_t0 = time.perf_counter()
    for task in tasks:
        print()
        print(f"== {task.task_id} [{task.condition}] ==")
        env = family.build_env(task)
        try:
            tools = family.build_tools(env)
            t0 = time.perf_counter()
            traj = agent.run(task=task, tools=tools, dataset_split=args.split)
            elapsed = time.perf_counter() - t0
            result = family.evaluate(task, trajectory=traj, env=env)
        finally:
            family.teardown_env(env)

        traj_path = out_dir / f"{task.task_id}.jsonl"
        eval_path = out_dir / f"{task.task_id}.eval.json"
        write_trajectory(traj_path, traj)
        _write_eval(eval_path, result)

        tool_call_tokens = sum(
            s.token_types.count("tool_call") for s in traj.steps
        )
        response_tokens = sum(s.token_types.count("response") for s in traj.steps)
        reasoning_tokens = sum(s.token_types.count("reasoning") for s in traj.steps)
        total_tokens = tool_call_tokens + response_tokens + reasoning_tokens
        errors = [s.error for s in traj.steps if s.error]
        err_str = ";".join(errors) if errors else None

        print(
            f"  steps={len(traj.steps)} tokens={total_tokens} "
            f"(tool_call={tool_call_tokens}, response={response_tokens}, "
            f"reasoning={reasoning_tokens}) wall={elapsed:.1f}s err={err_str}"
        )
        print(
            f"  eval: success={result.success} score={result.score:.3f} "
            f"details={result.details}"
        )
        if traj.steps and traj.steps[-1].text:
            final = traj.steps[-1].text.replace("\n", " ")[:200]
            print(f"  final: {final!r}")

    total_elapsed = time.perf_counter() - total_t0
    print()
    print(
        f"=== {args.family}/{args.split} complete in {total_elapsed:.1f}s, "
        f"{len(tasks)} tasks, trajectories under {out_dir} ==="
    )


if __name__ == "__main__":
    main()
