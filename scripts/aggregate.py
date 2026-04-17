"""Aggregate scored trajectories into a per-(family × condition × token_type) summary.

Reads every ``runs/scored/<pair>/<family>/<split>/*.jsonl`` file,
groups per-token acceptance proxies by
``(family, condition, token_type)``, prints a markdown table, and
writes the aggregate as CSV for external analysis.

This is the Stage D deliverable — the first place we can eyeball
whether the hypothesis signature (tool_call tokens have higher
acceptance than response tokens) holds on real tasks compared to
the Phase 1 mocks baseline.

Usage:
    PYTHONPATH=. python scripts/aggregate.py \
        --scored-root runs/scored/Qwen_Qwen2.5-7B-Instruct__Qwen_Qwen2.5-1.5B-Instruct \
        --out-csv runs/scored/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.trajectory.io import read_trajectory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-root",
        required=True,
        help="Root directory containing <family>/<split>/*.jsonl scored trajectories",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="Optional path for a CSV with one row per (family, condition, "
        "token_type) cell. If omitted, only the markdown table is printed.",
    )
    return parser.parse_args()


def _walk_scored(root: Path) -> list[Path]:
    """Return every ``*.jsonl`` under ``root`` (skip eval sidecars)."""
    return sorted(
        p for p in root.rglob("*.jsonl") if not p.name.endswith(".eval.json")
    )


def _aggregate(paths: list[Path]) -> dict[tuple[str, str, str], list[float]]:
    """Group per-token acceptance values by (family, condition, token_type)."""
    cells: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in paths:
        traj = read_trajectory(path)
        family = traj.metadata.family
        condition = traj.metadata.condition
        for step in traj.steps:
            if step.acceptance_proxy is None:
                continue
            for token_type, accept in zip(step.token_types, step.acceptance_proxy):
                cells[(family, condition, token_type)].append(accept)
    return cells


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile on a pre-sorted list (0 <= pct <= 100)."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    rank = max(0, min(n - 1, int(round(pct / 100.0 * (n - 1)))))
    return sorted_vals[rank]


def _summary_row(values: list[float]) -> dict[str, Any]:
    """Summary stats for a list of acceptance values.

    Reports both central-tendency (mean, median) and tail
    (p1, p5, p10, fractions below thresholds). The tail numbers
    matter more than the mean for speculative decoding — real
    speedup is bottlenecked by rejections, not by average-case
    agreement.
    """
    n = len(values)
    if n == 0:
        return {
            "n": 0, "mean": None, "std": None,
            "min": None, "p1": None, "p5": None, "p10": None,
            "median": None, "max": None,
            "frac_below_0_1": None, "frac_below_0_5": None,
            "frac_below_0_9": None, "frac_below_0_99": None,
        }
    sorted_vals = sorted(values)
    mean = statistics.fmean(values)
    return {
        "n": n,
        "mean": mean,
        "std": statistics.pstdev(values),
        "min": sorted_vals[0],
        "p1": _percentile(sorted_vals, 1),
        "p5": _percentile(sorted_vals, 5),
        "p10": _percentile(sorted_vals, 10),
        "median": _percentile(sorted_vals, 50),
        "max": sorted_vals[-1],
        "frac_below_0_1": sum(1 for v in values if v < 0.1) / n,
        "frac_below_0_5": sum(1 for v in values if v < 0.5) / n,
        "frac_below_0_9": sum(1 for v in values if v < 0.9) / n,
        "frac_below_0_99": sum(1 for v in values if v < 0.99) / n,
    }


def main() -> None:
    """Walk the scored root, aggregate, print, and optionally write CSV."""
    args = _parse_args()
    root = Path(args.scored_root)
    if not root.is_dir():
        raise SystemExit(f"Scored root not found: {root}")
    paths = _walk_scored(root)
    if not paths:
        raise SystemExit(f"No scored trajectories found under {root}")

    print(f"Aggregating {len(paths)} scored trajectories under {root}")
    cells = _aggregate(paths)

    rows: list[dict[str, Any]] = []
    for (family, condition, token_type) in sorted(cells):
        stats = _summary_row(cells[(family, condition, token_type)])
        rows.append(
            {
                "family": family,
                "condition": condition,
                "token_type": token_type,
                **stats,
            }
        )

    # Central-tendency + distribution shape
    print()
    print("## Central tendency + tail percentiles")
    print()
    print(
        "| family | condition | token_type | n | mean | std | p1 | p5 | p10 | median | min |"
    )
    print(
        "|--------|-----------|------------|---:|-----:|----:|---:|---:|----:|-------:|----:|"
    )
    for row in rows:
        if row["n"] == 0:
            continue
        print(
            f"| {row['family']} | {row['condition']} | {row['token_type']} | "
            f"{row['n']} | {row['mean']:.4f} | {row['std']:.4f} | "
            f"{row['p1']:.4f} | {row['p5']:.4f} | {row['p10']:.4f} | "
            f"{row['median']:.4f} | {row['min']:.4f} |"
        )

    # Threshold fractions — "what fraction of tokens are in each badness band"
    print()
    print("## Fraction of tokens below thresholds (lower = fewer bad tokens)")
    print()
    print(
        "| family | condition | token_type | n | <0.1 | <0.5 | <0.9 | <0.99 |"
    )
    print(
        "|--------|-----------|------------|---:|-----:|-----:|-----:|------:|"
    )
    for row in rows:
        if row["n"] == 0:
            continue
        print(
            f"| {row['family']} | {row['condition']} | {row['token_type']} | "
            f"{row['n']} | "
            f"{row['frac_below_0_1']*100:5.2f}% | "
            f"{row['frac_below_0_5']*100:5.2f}% | "
            f"{row['frac_below_0_9']*100:5.2f}% | "
            f"{row['frac_below_0_99']*100:5.2f}% |"
        )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "family",
                    "condition",
                    "token_type",
                    "n",
                    "mean",
                    "std",
                    "min",
                    "p1",
                    "p5",
                    "p10",
                    "median",
                    "max",
                    "frac_below_0_1",
                    "frac_below_0_5",
                    "frac_below_0_9",
                    "frac_below_0_99",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
