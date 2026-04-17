"""Simulate speculative-decoding speedup from stored acceptance proxies.

For each scored trajectory, Monte-Carlo simulate a speculative
decoder with fixed lookahead ``K`` over the sequence of per-token
acceptance probabilities. Report, per (family, condition, K), the
expected ratio of tokens emitted per target forward pass — i.e.,
the speedup over baseline autoregressive generation.

The simulation:

1. Walk tokens left-to-right. For block starting at position i:
   - For j = 0..K-1, accept position i+j with probability
     ``acceptance_proxy[i+j]`` (Bernoulli). Stop at first rejection.
2. That block consumes exactly one target forward pass.
3. The target verification yields one extra token — the "correction"
   (if a rejection happened) or the "bonus" (if all K accepted).
   Either way: tokens emitted in this block = accepted + 1. Advance
   ``i`` by ``accepted + 1``.

**Caveats**:

- This is an **upper bound** on the real speedup. In practice the
  draft proposes tokens drawn from its own distribution; our
  stored acceptance values assume the draft proposed exactly what
  the target emitted (its favourite). A real speculative decoder
  would see lower acceptance on average.
- Each block counts as one target forward pass, regardless of
  which position rejected. This matches how real spec decoders
  work (one batched forward pass verifies all K proposals).
- We ignore the cost of the draft's ``K`` forward passes per block.
  In practice the draft is much smaller so this is cheap, but a
  rigorous speedup number would subtract it.

Usage:
    PYTHONPATH=. python scripts/simulate_speedup.py \
        --scored-root runs/scored/Qwen_Qwen2.5-7B-Instruct__Qwen_Qwen2.5-1.5B-Instruct \
        --ks 2 4 8
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict
from pathlib import Path

from src.trajectory.io import read_trajectory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", required=True)
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[2, 4, 8],
        help="Lookahead values K to simulate",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _simulate_once(acceptance: list[float], k: int, rng: random.Random) -> tuple[int, int]:
    """Single Monte Carlo run.

    Returns ``(target_passes, tokens_emitted)`` for the whole
    trajectory.
    """
    n = len(acceptance)
    i = 0
    target_passes = 0
    tokens_emitted = 0
    while i < n:
        target_passes += 1
        accepted = 0
        for j in range(k):
            if i + j >= n:
                break
            if rng.random() < acceptance[i + j]:
                accepted += 1
            else:
                break
        # Emit accepted + 1 (correction or bonus). Advance same amount.
        # Clamp if we'd go past the end of the stored tokens.
        step_forward = min(accepted + 1, n - i)
        tokens_emitted += step_forward
        i += step_forward
    return target_passes, tokens_emitted


def _simulate_mean(
    acceptance: list[float], k: int, trials: int, seed: int
) -> float:
    """Average ``tokens_emitted / target_passes`` over ``trials`` runs."""
    rng = random.Random(seed)
    speedups: list[float] = []
    for _ in range(trials):
        passes, emitted = _simulate_once(acceptance, k, rng)
        if passes > 0:
            speedups.append(emitted / passes)
    return statistics.fmean(speedups) if speedups else 0.0


def main() -> None:
    """Simulate and report speedup per (family, condition, K)."""
    args = _parse_args()
    root = Path(args.scored_root)
    paths = sorted(p for p in root.rglob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No scored trajectories under {root}")

    # Gather acceptance sequences per (family, condition, task_id)
    seqs: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for path in paths:
        traj = read_trajectory(path)
        fam = traj.metadata.family
        cond = traj.metadata.condition
        flat: list[float] = []
        for step in traj.steps:
            if step.acceptance_proxy is not None:
                flat.extend(step.acceptance_proxy)
        if flat:
            seqs[(fam, cond)].append(flat)

    print(
        f"Simulating spec-decoding speedup over {len(paths)} trajectories, "
        f"{args.trials} trials per (trajectory, K)"
    )
    print()

    # Header
    k_cols = " | ".join(f"K={k:<2d}" for k in args.ks)
    print(f"| family | condition | n_trajs | mean_len | {k_cols} |")
    print(f"|--------|-----------|--------:|--------:|{'|'.join(['----:'] * len(args.ks))}|")

    for (fam, cond), trajectories in sorted(seqs.items()):
        mean_len = statistics.fmean(len(t) for t in trajectories)
        row_speedups: list[float] = []
        for k in args.ks:
            # Average speedup across trajectories, each averaged across trials
            per_traj = [
                _simulate_mean(t, k, args.trials, args.seed) for t in trajectories
            ]
            row_speedups.append(statistics.fmean(per_traj))
        speedup_cells = " | ".join(f"{s:.2f}" for s in row_speedups)
        print(
            f"| {fam} | {cond} | {len(trajectories)} | {mean_len:.0f} | "
            f"{speedup_cells} |"
        )

    # Also report baseline (expected speedup if all tokens had mean acceptance)
    print()
    print("## Reference: analytical speedup for uniform-p model")
    print(
        "  With lookahead K and uniform acceptance p, "
        "E[tokens/block] = (1 - p^{K+1}) / (1 - p). "
        "K=4 at p=0.99 gives 4.90x; p=0.95 -> 4.52x; p=0.90 -> 4.10x; "
        "p=0.50 -> 1.94x."
    )


if __name__ == "__main__":
    main()
