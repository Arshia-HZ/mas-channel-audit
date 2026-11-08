"""Pilot gates G1 to G4, the arm table and the power inputs.

    python scripts/05_gate_check.py --root runs/pilot_qwen3-1.7b \
        --labels labels.json --json pilot_report.json

Reads finished shards (never loads a model). G4 needs the hand-labelled failure
messages (see ``05_failure_sample.py``); without ``--labels`` it is reported as
pending. ``--confirm-available`` caps the required sample size; by default it is
the confirm count in the frozen split manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from channel_audit.failure_review import (
    format_stats,
    g4_check,
    gold_audit,
    load_labels,
    sample_failure_messages,
)
from channel_audit.gates import format_pilot_report, pilot_report
from channel_audit.runner import open_run


def shard_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if (d / "run.json").exists())


def confirm_available(manifest: Path) -> int:
    return json.loads(manifest.read_text(encoding="utf-8"))["counts"]["confirm"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="directory holding the shard_XX run directories")
    ap.add_argument("--runs", nargs="+", help="shard run directories")
    ap.add_argument("--labels", help="completed labels file for G4")
    ap.add_argument("--seed", type=int, default=0, help="seed of the failure sample")
    ap.add_argument(
        "--variant", default="v1", choices=["v1", "v2"], help="failure-message prompt"
    )
    ap.add_argument("--manifest", default="data/problems/split_manifest.json")
    ap.add_argument("--confirm-available", type=int)
    ap.add_argument("--json", help="also write the report to this file")
    args = ap.parse_args()
    if bool(args.root) == bool(args.runs):
        ap.error("give exactly one of --root and --runs")

    dirs = shard_dirs(Path(args.root)) if args.root else [Path(r) for r in args.runs]
    ctxs = [open_run(d) for d in dirs]
    available = args.confirm_available or confirm_available(Path(args.manifest))
    rep = pilot_report(ctxs, available)

    g4 = audit = None
    if args.labels:
        items = sample_failure_messages(ctxs, seed=args.seed, variant=args.variant)
        g4 = g4_check(load_labels(args.labels, items))
        audit = {"gold": gold_audit(items), "format": format_stats(items)}
    print(format_pilot_report(rep, g4))
    if audit:
        n = audit["format"]["n"]
        print(
            f"failure messages ({args.variant}): "
            f"{audit['format']['single_sentence']}/{n} single sentence, "
            f"{audit['format']['artifacts']} with an [answer removed] artifact, "
            f"{audit['gold']['contains_gold']} containing the gold answer"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps({**rep, "G4": g4, "G4_audit": audit}, indent=1, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
