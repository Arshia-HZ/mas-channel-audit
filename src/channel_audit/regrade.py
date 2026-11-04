"""Re-grade a finished run into a new directory.

Runs are append-only, so a change to the extractor never rewrites old grades.
Instead the generations and their inputs are copied byte for byte into a new
run directory, and the grades are computed again there.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from channel_audit.runner import grade_pending, open_run

COPIED = (
    "run.json",
    "problems.jsonl",
    "generations.jsonl",
    "prompts.jsonl",
    "leakage.jsonl",
    "throughput.jsonl",
)


def regrade_run(src: str | Path, dst: str | Path) -> int:
    """Copy ``src`` to a new ``dst`` without its grades and grade it again.

    Returns the number of generations graded. ``dst`` must not hold files yet.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise FileExistsError("the destination must be a new directory")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} is not empty; choose a new directory")
    dst.mkdir(parents=True, exist_ok=True)
    for name in COPIED:
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)
    return grade_pending(open_run(dst))
