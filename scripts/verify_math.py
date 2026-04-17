"""Sanity check the logprob / acceptance chain in stored trajectories.

Checks, per scored trajectory:

1. **Array alignment** — ``token_ids``, ``target_logprobs``,
   ``draft_logprobs``, ``acceptance_proxy``, and ``token_types``
   all have the same length on every step.
2. **Logprob bounds** — target and draft logprobs are ``<= 0`` (up
   to a small float-precision tolerance; a tiny positive value is
   permissible when one token dominates the softmax so completely
   that the normalisation rounds).
3. **Acceptance range** — every value is in ``[0, 1]``.
4. **Recomputation match** — recompute ``a = min(1, exp(t_lp - d_lp))``
   directly from stored logprobs and confirm it matches the stored
   ``acceptance_proxy`` within a small tolerance. This is the most
   important check: if the stored ``acceptance_proxy`` doesn't
   match the formula, the scorer has a bug.
5. **Distribution of tied tokens** — report how many tokens have
   ``target_lp >= draft_lp`` (→ acceptance exactly 1.0). For a
   same-family pair with a mostly-confident target this fraction
   is naturally high and explains why so many acceptance values
   sit at exactly 1.0.

Also prints, for each family, a few example low-acceptance tokens
with their raw logprobs so you can inspect whether the numbers
look sensible (e.g. target_lp = -0.1, draft_lp = 0.0 → acceptance
= 0.905 is a legitimate case, not a bug).

Usage:
    PYTHONPATH=. python scripts/verify_math.py \
        --scored-root runs/scored/Qwen_Qwen2.5-7B-Instruct__Qwen_Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

from src.trajectory.io import read_trajectory


TOLERANCE = 1e-4
"""Allowed slack for logprob non-negativity and recomputation mismatch.

fp32 log_softmax + fp64 recomputation can disagree in the 5th-6th
decimal; TOLERANCE is several orders of magnitude larger than that
so we catch real errors without false positives.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-root", required=True)
    parser.add_argument(
        "--per-family-examples",
        type=int,
        default=5,
        help="Number of low-acceptance token examples to print per family",
    )
    return parser.parse_args()


def main() -> None:
    """Run the invariant checks and print a summary."""
    args = _parse_args()
    root = Path(args.scored_root)
    paths = sorted(p for p in root.rglob("*.jsonl"))
    if not paths:
        raise SystemExit(f"No scored trajectories under {root}")

    print(f"Checking {len(paths)} scored trajectories...")
    print()

    total_tokens = 0
    total_alignment_fails = 0
    total_bounds_fails = 0
    total_recompute_fails = 0
    total_tied_at_1 = 0
    total_approx_1 = 0
    recompute_max_diff = 0.0

    # For per-family example printouts:
    low_accept_examples: dict[str, list[tuple[str, float, float, float, str]]] = defaultdict(list)

    for path in paths:
        traj = read_trajectory(path)
        family = traj.metadata.family
        for step_idx, step in enumerate(traj.steps):
            if step.acceptance_proxy is None:
                print(f"WARN {path.name} step {step_idx}: acceptance_proxy is None; skipping")
                continue

            n = len(step.token_ids)
            arrays_ok = (
                len(step.target_logprobs) == n
                and len(step.draft_logprobs or []) == n
                and len(step.acceptance_proxy) == n
                and len(step.token_types) == n
            )
            if not arrays_ok:
                total_alignment_fails += 1
                print(
                    f"FAIL alignment on {path.name} step {step_idx}: "
                    f"token_ids={n} target_lp={len(step.target_logprobs)} "
                    f"draft_lp={len(step.draft_logprobs or [])} "
                    f"accept={len(step.acceptance_proxy)} "
                    f"types={len(step.token_types)}"
                )
                continue

            total_tokens += n

            # Bounds + recompute check
            for j in range(n):
                t_lp = step.target_logprobs[j]
                d_lp = step.draft_logprobs[j]
                stored = step.acceptance_proxy[j]

                if t_lp > TOLERANCE:
                    total_bounds_fails += 1
                if d_lp > TOLERANCE:
                    total_bounds_fails += 1

                if stored < -TOLERANCE or stored > 1.0 + TOLERANCE:
                    total_bounds_fails += 1

                recomputed = min(1.0, math.exp(t_lp - d_lp))
                diff = abs(recomputed - stored)
                if diff > recompute_max_diff:
                    recompute_max_diff = diff
                if diff > TOLERANCE:
                    total_recompute_fails += 1
                    if total_recompute_fails <= 10:
                        print(
                            f"FAIL recompute on {path.name} step {step_idx} tok {j}: "
                            f"t_lp={t_lp:.6f} d_lp={d_lp:.6f} "
                            f"stored={stored:.6f} recomputed={recomputed:.6f} "
                            f"diff={diff:.2e}"
                        )

                # Tied-at-1 tracking
                if stored >= 1.0 - 1e-12:
                    total_tied_at_1 += 1
                if stored >= 1.0 - 1e-4:
                    total_approx_1 += 1

                # Collect low-acceptance examples per family
                if stored < 0.5 and len(low_accept_examples[family]) < args.per_family_examples:
                    token_string = step.token_strings[j]
                    token_type = step.token_types[j]
                    low_accept_examples[family].append(
                        (token_string, t_lp, d_lp, stored, token_type)
                    )

    # Summary
    print()
    print("## Invariant check summary")
    print(f"  total tokens checked     : {total_tokens}")
    print(f"  array alignment failures : {total_alignment_fails}")
    print(f"  logprob bounds failures  : {total_bounds_fails}")
    print(f"  recompute mismatches     : {total_recompute_fails}")
    print(f"  max recompute |stored - recomputed| = {recompute_max_diff:.3e}")
    print()
    pct_tied_exact = 100.0 * total_tied_at_1 / max(1, total_tokens)
    pct_near_one = 100.0 * total_approx_1 / max(1, total_tokens)
    print(f"  tokens with acceptance == 1.0 exactly : {total_tied_at_1} ({pct_tied_exact:.2f}%)")
    print(f"  tokens with acceptance >= 1 - 1e-4    : {total_approx_1} ({pct_near_one:.2f}%)")
    print()
    print(
        "  Interpretation: tokens at exactly 1.0 are cases where the target "
        "assigned >= draft's probability to the emitted token — the min(1,...) "
        "clip fires. This is expected and natural, not a bug."
    )

    print()
    print("## Low-acceptance examples per family (first few)")
    for family, examples in sorted(low_accept_examples.items()):
        print(f"  {family}:")
        for tok_str, t_lp, d_lp, accept, tok_type in examples:
            print(
                f"    token={tok_str!r:20s}  type={tok_type:<10s}  "
                f"target_lp={t_lp:+.4f}  draft_lp={d_lp:+.4f}  "
                f"accept={accept:.4f}  "
                f"(draft is {'more' if d_lp > t_lp else 'less'} confident)"
            )

    # Exit code
    total_fails = total_alignment_fails + total_bounds_fails + total_recompute_fails
    if total_fails > 0:
        print()
        print(f"FAILED: {total_fails} invariant violations")
        raise SystemExit(1)
    print()
    print("PASSED: all invariants hold within tolerance")


if __name__ == "__main__":
    main()
