"""Failure messages for the failure_cause arm, with leakage filtering.

Agent j's solution is reviewed in one sentence (no final answer) and the
message goes to agent j. Boxes and answer sentences are stripped; a surviving
claim of a round-0 answer is regenerated once, then redacted.

    python scripts/12_failure_messages.py --out runs/mock
"""

from __future__ import annotations

import argparse
from collections import Counter

from channel_audit.runner import open_run, run_failure_messages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx = open_run(args.out)
    run_failure_messages(ctx)
    outcomes = dict(Counter(r.outcome for r in ctx.leakage))
    print(f"failure messages: {len(ctx.leakage)} resolved; leakage outcomes {outcomes}")


if __name__ == "__main__":
    main()
