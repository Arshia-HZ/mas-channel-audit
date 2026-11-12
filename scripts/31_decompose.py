"""Exploratory decomposition of the confirmatory shards (descriptive, labelled).

    python scripts/31_decompose.py --root runs/confirm_qwen3-1.7b_vllm \
        --out-dir results/confirm_qwen3-1.7b_exploratory --expect-problems 600

Change types, source of change, diversity, headroom recovery and conformity, as
PREREG 10.2 (item 9) allows after the analysis freeze. The output is marked
EXPLORATORY and goes in its own directory: the script refuses a directory that holds
the primary reports (analysis.json, RUN.json) or an earlier exploratory result. The
shards are checked first (SHARD_DONE hash, validation, no duplicate problem).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from channel_audit.analysis.contrasts import ANALYSIS_SEED, N_BOOT
from channel_audit.analysis.decompose import decompose, format_decomposition_report
from channel_audit.runner import open_run
from channel_audit.shards import DONE_FILE
from channel_audit.validate import validate_run

RESULT = "exploratory.json"
REFUSED = ("analysis.json", "analysis.md", "RUN.json", "exploratory.md", RESULT)


def fail(message: str) -> int:
    print(f"error: {message}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="directory holding the shard_XX run directories")
    ap.add_argument("--runs", nargs="+", help="shard run directories")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--expect-problems", type=int)
    args = ap.parse_args()
    if bool(args.root) == bool(args.runs):
        ap.error("give exactly one of --root and --runs")

    out = Path(args.out_dir)
    for name in REFUSED:
        if (out / name).exists():
            return fail(
                f"{out / name} already exists: exploratory output has its own directory"
            )

    dirs = (
        sorted(d for d in Path(args.root).iterdir() if (d / "run.json").exists())
        if args.root
        else [Path(r) for r in args.runs]
    )
    for d in dirs:
        done_path = d / DONE_FILE
        if not done_path.exists():
            return fail(f"{d.name}: not marked done ({DONE_FILE} missing)")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        digest = hashlib.sha256((d / "generations.jsonl").read_bytes()).hexdigest()
        if digest != done["generations_sha256"]:
            return fail(
                f"{d.name}: generations changed since the shard was marked done"
            )
    ctxs = [open_run(d) for d in dirs]
    for d, ctx in zip(dirs, ctxs, strict=True):
        report = validate_run(ctx)
        if report.errors:
            return fail(f"{d.name} does not validate: " + "; ".join(report.errors[:5]))
    ids = [p.problem_id for ctx in ctxs for p in ctx.problems]
    if len(set(ids)) != len(ids):
        return fail("a problem appears in more than one shard")
    if args.expect_problems is not None and len(ids) != args.expect_problems:
        return fail(f"expected {args.expect_problems} problems, found {len(ids)}")

    rep = decompose(ctxs, n_boot=args.n_boot, seed=ANALYSIS_SEED)
    text = format_decomposition_report(rep)
    out.mkdir(parents=True, exist_ok=True)
    (out / "exploratory.md").write_text(text + "\n", encoding="utf-8")
    (out / RESULT).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
