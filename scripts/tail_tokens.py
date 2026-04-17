"""Inspect which token *strings* end up in the low-acceptance tail.

Walks every scored trajectory, collects tokens whose
``acceptance_proxy < threshold``, and reports the most common
token strings in that tail. This tells us whether the 3-8% of
response tokens that trigger low acceptance cluster on a small
set of predictable strings (and would therefore be ideal targets
for distillation) or are spread out uniformly.

Usage:
    PYTHONPATH=. python scripts/tail_tokens.py \
        --scored-root runs/scored/Qwen_Qwen2.5-7B-Instruct__Qwen_Qwen2.5-1.5B-Instruct \
        --threshold 0.5 \
        --top 30
"""

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.trajectory.io import read_trajectory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Count only tokens with acceptance_proxy < threshold",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of top token strings to print per family",
    )
    return parser.parse_args()


def main() -> None:
    """Gather tail tokens and print per-family frequency tables."""
    args = _parse_args()
    root = Path(args.scored_root)
    paths = sorted(p for p in root.rglob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No scored trajectories under {root}")

    # (family, token_type) -> Counter[token_string]
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    # (family, token_type) -> list of (accept, target_lp, draft_lp) for distribution
    stats: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    total_counts: dict[tuple[str, str], int] = defaultdict(int)

    for path in paths:
        traj = read_trajectory(path)
        fam = traj.metadata.family
        for step in traj.steps:
            if step.acceptance_proxy is None:
                continue
            for j, accept in enumerate(step.acceptance_proxy):
                t_type = step.token_types[j]
                total_counts[(fam, t_type)] += 1
                if accept < args.threshold:
                    tok_str = step.token_strings[j]
                    counters[(fam, t_type)][tok_str] += 1
                    stats[(fam, t_type)].append(
                        (accept, step.target_logprobs[j], step.draft_logprobs[j])
                    )

    print(
        f"Low-acceptance tokens (acceptance < {args.threshold}) across "
        f"{len(paths)} trajectories"
    )
    print()

    for (fam, t_type), counter in sorted(counters.items()):
        total = total_counts[(fam, t_type)]
        tail_n = sum(counter.values())
        pct = 100.0 * tail_n / total if total else 0.0
        st = stats[(fam, t_type)]
        mean_a = statistics.fmean(a for a, _, _ in st) if st else 0.0
        mean_t = statistics.fmean(t for _, t, _ in st) if st else 0.0
        mean_d = statistics.fmean(d for _, _, d in st) if st else 0.0

        print(
            f"## {fam} / {t_type}: "
            f"{tail_n} / {total} tokens in tail ({pct:.2f}%), "
            f"tail mean accept={mean_a:.3f}, "
            f"tail mean target_lp={mean_t:.3f}, tail mean draft_lp={mean_d:.3f}"
        )
        if not counter:
            print("  (empty tail)")
            print()
            continue
        print(f"  top {args.top} most frequent tail tokens:")
        for tok_str, count in counter.most_common(args.top):
            frac = 100.0 * count / tail_n
            print(f"    {count:5d} ({frac:5.1f}% of tail)  {tok_str!r}")
        print()


if __name__ == "__main__":
    main()
