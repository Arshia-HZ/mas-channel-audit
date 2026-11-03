"""Extract and grade every solution-bearing generation not yet graded.

python scripts/20_grade.py --out runs/mock
"""

from __future__ import annotations

import argparse

from channel_audit.runner import grade_pending, open_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx = open_run(args.out)
    n = grade_pending(ctx)
    print(f"graded {n} new generations; {len(ctx.graded)} total")


if __name__ == "__main__":
    main()
