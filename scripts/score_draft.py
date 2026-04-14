"""Run the offline draft-scoring pass against a directory of trajectories.

Loads a draft model, iterates every ``*.jsonl`` file under
``--input-dir``, runs :func:`src.scoring.draft.score_trajectory` on
each, and writes enriched trajectories under ``--out-dir``. Prints a
per-token-type mean-acceptance summary per trajectory so the
hypothesis signature (tool_call acceptance ≈ 1, response acceptance
< 1) can be eyeballed immediately.

Usage:
    PYTHONPATH=. python scripts/score_draft.py \
        --input-dir runs/Qwen_Qwen2.5-7B-Instruct \
        --draft-model Qwen/Qwen2.5-1.5B-Instruct \
        --out-dir runs/scored/Qwen_Qwen2.5-7B-Instruct__Qwen_Qwen2.5-1.5B-Instruct

The draft tokenizer must be byte-compatible with the target
tokenizer (same family), otherwise the replay invariant check will
abort.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.scoring.draft import score_trajectory
from src.trajectory.io import read_trajectory, write_trajectory
from src.trajectory.schema import Trajectory


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing target-produced trajectory JSONL files",
    )
    parser.add_argument(
        "--draft-model",
        required=True,
        help="HF draft model id (must share the target tokenizer)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write scored trajectory JSONL files into",
    )
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=sorted(_DTYPE_MAP),
        help="Draft model dtype",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass through to from_pretrained",
    )
    return parser.parse_args()


def _print_summary(traj: Trajectory) -> None:
    """Print per-token-type mean acceptance for one trajectory."""
    by_type: dict[str, list[float]] = {}
    total_tokens = 0
    for step in traj.steps:
        if step.acceptance_proxy is None:
            continue
        for label, accept in zip(step.token_types, step.acceptance_proxy):
            by_type.setdefault(label, []).append(accept)
            total_tokens += 1
    print(
        f"  task={traj.metadata.task_id} "
        f"condition={traj.metadata.condition} "
        f"total_tokens={total_tokens}"
    )
    for label in sorted(by_type):
        vals = by_type[label]
        mean = sum(vals) / len(vals)
        lo = min(vals)
        hi = max(vals)
        print(
            f"    {label:<10s} n={len(vals):4d} "
            f"mean={mean:.4f}  min={lo:.4f}  max={hi:.4f}"
        )
    overall = [v for vs in by_type.values() for v in vs]
    if overall:
        mean = sum(overall) / len(overall)
        print(f"    overall    n={len(overall):4d} mean={mean:.4f}")


def main() -> None:
    """Score every trajectory in the input directory."""
    args = _parse_args()
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)

    traj_paths = sorted(input_dir.glob("*.jsonl"))
    if not traj_paths:
        raise SystemExit(f"No trajectory JSONL files found under {input_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(traj_paths)} trajectories in {input_dir}")

    print(f"Loading draft model {args.draft_model} on {args.device} ({args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.draft_model, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        dtype=_DTYPE_MAP[args.dtype],
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM allocated: {vram:.2f} GB")

    print()
    total_t0 = time.perf_counter()
    for path in traj_paths:
        traj = read_trajectory(path)
        print(f"== scoring {path.name} ==")
        t0 = time.perf_counter()
        scored = score_trajectory(
            traj=traj,
            draft_tokenizer=tokenizer,
            draft_model=model,
            draft_model_name=args.draft_model,
            device=args.device,
        )
        elapsed = time.perf_counter() - t0
        out_path = out_dir / path.name
        write_trajectory(out_path, scored)
        _print_summary(scored)
        print(f"  wall={elapsed * 1000:.0f}ms  wrote {out_path}")
        print()

    print(
        f"=== scored {len(traj_paths)} trajectories in "
        f"{time.perf_counter() - total_t0:.1f}s ==="
    )


if __name__ == "__main__":
    main()
