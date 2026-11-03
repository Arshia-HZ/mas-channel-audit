"""Branch every arm from the stored round 0: one revision per agent.

Needs steps 10 and 12 first (11 is needed only for the budget comparison).

    python scripts/13_branch_arms.py --out runs/mock [--arms self_reflect placebo]
"""

from __future__ import annotations

import argparse

from channel_audit.channels import ARMS
from channel_audit.runner import open_run, run_arm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    args = ap.parse_args()
    ctx = open_run(args.out)
    for arm in args.arms:
        run_arm(ctx, arm)
        n = sum(1 for g in ctx.gens if g.arm == arm)
        print(f"{arm}: {n} generations")


if __name__ == "__main__":
    main()
