"""Compare screened models: accuracy by level, disagreement, truncation, G0.

    python scripts/04_screen_report.py --runs runs/screen_a runs/screen_b \
        --json screen_summary.json

Reads finished screening runs; it never loads a model. Choosing the primary
model from this table and writing the choice into PREREG.md is a human step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from channel_audit.runner import grade_pending, open_run
from channel_audit.screening import format_comparison, screen_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json", help="also write the reports to this file")
    args = ap.parse_args()

    reports = []
    for run in args.runs:
        ctx = open_run(run)
        grade_pending(ctx)
        reports.append(screen_report(ctx))
    print(format_comparison(reports))
    if args.json:
        Path(args.json).write_text(
            json.dumps(reports, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
