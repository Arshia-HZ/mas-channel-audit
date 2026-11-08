"""Draw the 40 failure messages to label for gate G4.

    python scripts/05_failure_sample.py --root runs/pilot_qwen3-1.7b --out-dir g4

Writes ``failure_sheet.md`` (read it) and a blank ``labels.json`` (fill each value
with useful, vague or wrong). The draw is a seeded uniform sample over every
message a reader was shown; it never looks at correctness. An existing labels file
is never overwritten. Then run ``05_gate_check.py --labels g4/labels.json`` with the
same seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from channel_audit.failure_review import (
    N_SAMPLE,
    format_label_sheet,
    sample_failure_messages,
    write_label_template,
)
from channel_audit.runner import open_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="directory holding the shard_XX run directories")
    ap.add_argument("--runs", nargs="+", help="shard run directories")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()
    if bool(args.root) == bool(args.runs):
        ap.error("give exactly one of --root and --runs")

    if args.root:
        dirs = sorted(d for d in Path(args.root).iterdir() if (d / "run.json").exists())
    else:
        dirs = [Path(r) for r in args.runs]
    items = sample_failure_messages(
        [open_run(d) for d in dirs],
        n=args.n,
        seed=args.seed,
        variant=args.variant,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_label_template(items, out / "labels.json")
    (out / "failure_sheet.md").write_text(format_label_sheet(items), encoding="utf-8")
    print(f"{len(items)} messages: read {out / 'failure_sheet.md'}")
    print(f"then fill in {out / 'labels.json'}")


if __name__ == "__main__":
    main()
