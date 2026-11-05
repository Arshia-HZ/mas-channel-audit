"""Validate a run: counts, linkage, grading, truncation/overflow, budgets.

    python scripts/21_validate_run.py --out runs/mock

Exit code 0: valid and gate G3 passes. 1: errors. 2: valid but G3 fails.
"""

from __future__ import annotations

import argparse
import sys

from channel_audit.runner import open_run, throughput_summary
from channel_audit.validate import validate_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx = open_run(args.out)
    r = validate_run(ctx)

    print(f"errors: {len(r.errors)}")
    for e in r.errors:
        print(f"  ERROR {e}")
    for w in r.warnings:
        print(f"  WARN  {w}")
    print(
        "gate G3 (no overflow; each revision arm under 5% truncation): "
        f"{'pass' if r.g3_ok else 'FAIL'}"
        + (f" - failing arms: {r.g3_failing_arms}" if r.g3_failing_arms else "")
    )
    print(f"leakage outcomes: {r.leakage_outcomes}")
    t = throughput_summary(ctx.out_dir)
    print(
        f"throughput: {t['completion_tokens_per_s']:.0f} completion tokens/s "
        f"({t['completion_tokens']} tokens, {t['seconds']:.1f} s, "
        f"{t['n_batches']} backend calls)"
    )
    print(f"{'arm':<14}{'n':>7}{'trunc':>8}{'overflow':>10}{'tokens':>10}{'SC k':>7}")
    for arm in sorted(r.counts):
        cost = r.costs.get(arm)
        k = r.matched_k.get(arm)
        print(
            f"{arm:<14}{r.counts[arm]:>7}{r.truncation_rate[arm]:>8.1%}"
            f"{r.overflow_count[arm]:>10}"
            f"{(f'{cost.generated_tokens:.0f}' if cost else '-'):>10}"
            f"{(f'{k:.2f}' if k else '-'):>7}"
        )
    if r.errors:
        return 1
    return 0 if r.g3_ok else 2


if __name__ == "__main__":
    sys.exit(main())
