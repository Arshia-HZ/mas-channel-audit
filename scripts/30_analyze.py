"""Analyse finished shards: the six primary contrasts and the preregistered extras.

    python scripts/30_analyze.py --root runs/confirm_qwen3-1.7b_vllm \
        --out-dir results/confirm_qwen3-1.7b_analysis --expect-problems 600

This is the confirmatory analysis. It runs once, with the code frozen by
``scripts/29_freeze_analysis.py``. It refuses to run if a frozen or pinned file
changed or the freeze is missing, if ``--expect-problems`` is not given, if a shard
does not validate, no longer matches its SHARD_DONE hash or holds a stored grade the
grader would not give now, or if results already exist in ``--out-dir``.
``--rehearsal`` skips the freeze check but only for mock-backend data, and labels
the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from channel_audit.analysis.contrasts import (
    BEST_LOW_BANDWIDTH,
    N_BOOT,
    analyze,
    format_analysis_report,
)
from channel_audit.analysis.integrity import grade_mismatches
from channel_audit.analysis_freeze import (
    ANALYSIS_FILES,
    FREEZE_FILE,
    PIPELINE_FILES,
    ROOT,
    FrozenAnalysisError,
    build_manifest,
    require_frozen,
)
from channel_audit.runner import open_run
from channel_audit.shards import DONE_FILE
from channel_audit.validate import validate_run

RESULT = "analysis.json"
OUTPUTS = ("RUN.json", "analysis.md", RESULT)  # RESULT last: it marks a finished run


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> int:
    print(f"error: {message}")
    return 1


def shard_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if (d / "run.json").exists())


def version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def environment() -> dict:
    """The versions that can change a number: bootstrap draws and answer equivalence."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{
            pkg: version(pkg)
            for pkg in ("numpy", "math-verify", "pydantic", "latex2sympy2_extended")
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", help="directory holding the shard_XX run directories")
    ap.add_argument("--runs", nargs="+", help="shard run directories")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--freeze-file", default=str(ROOT / FREEZE_FILE))
    ap.add_argument("--rehearsal", action="store_true", help="mock-backend data only")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument(
        "--expect-problems",
        type=int,
        help="how many problems the shards must hold (600 for the confirmatory run)",
    )
    ap.add_argument("--best-arm", default=BEST_LOW_BANDWIDTH)
    args = ap.parse_args()
    if bool(args.root) == bool(args.runs):
        ap.error("give exactly one of --root and --runs")

    out = Path(args.out_dir)
    for name in OUTPUTS:
        if (out / name).exists():
            return fail(f"{out / name} already exists: the analysis runs once")
    if not args.rehearsal:
        if args.expect_problems is None:
            return fail(
                "--expect-problems is required outside a rehearsal: a missing shard "
                "would silently shrink the analysis (the confirmatory run has 600)"
            )
        try:
            frozen = require_frozen(
                args.freeze_file,
                ROOT,
                expected={"analysis": ANALYSIS_FILES, "pipeline": PIPELINE_FILES},
            )
        except FrozenAnalysisError as exc:
            return fail(str(exc))
    else:
        frozen = build_manifest(ROOT)

    env = environment()
    dirs = shard_dirs(Path(args.root)) if args.root else [Path(r) for r in args.runs]
    shards = []
    for d in dirs:
        done_path = d / DONE_FILE
        digest = sha256(d / "generations.jsonl")
        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if digest != done["generations_sha256"]:
                return fail(
                    f"{d.name}: generations changed since the shard was marked done"
                )
        elif not args.rehearsal:
            return fail(f"{d.name}: not marked done ({DONE_FILE} missing)")
        shards.append({"name": d.name, "generations_sha256": digest})

    ctxs = [open_run(d) for d in dirs]
    if args.rehearsal and any(ctx.cfg.backend != "mock" for ctx in ctxs):
        return fail(
            "--rehearsal is for mock-backend data only; real data needs the freeze"
        )
    for d, ctx, shard in zip(dirs, ctxs, shards, strict=True):
        report = validate_run(ctx)
        if report.errors:
            return fail(f"{d.name} does not validate: " + "; ".join(report.errors[:5]))
        bad = grade_mismatches(ctx)
        if bad:
            return fail(
                f"{d.name}: {len(bad)} stored grades differ from a fresh grading "
                f"(first: {bad[0]})"
            )
        shard["n_problems"] = len(ctx.problems)
    ids = [p.problem_id for ctx in ctxs for p in ctx.problems]
    if len(set(ids)) != len(ids):
        return fail("a problem appears in more than one shard")
    if args.expect_problems is not None and len(ids) != args.expect_problems:
        return fail(f"expected {args.expect_problems} problems, found {len(ids)}")

    rep = analyze(ctxs, best_arm=args.best_arm, n_boot=args.n_boot)
    rep["status"]["rehearsal"] = args.rehearsal
    text = format_analysis_report(rep)

    hardware_file = dirs[0].parent / "HARDWARE.json"
    info = {
        "analysis_combined_sha256": frozen["combined_sha256"],
        "frozen_manifest": frozen,
        "freeze_file": None if args.rehearsal else str(args.freeze_file),
        "rehearsal": args.rehearsal,
        "argv": sys.argv[1:],
        "n_boot": args.n_boot,
        "best_arm": args.best_arm,
        "environment": env,
        "hardware": (
            json.loads(hardware_file.read_text(encoding="utf-8"))
            if hardware_file.exists()
            else None
        ),
        "shards": shards,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "RUN.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    (out / "analysis.md").write_text(text + "\n", encoding="utf-8")
    (out / RESULT).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
