"""Re-grading a finished run into a new directory; the source is never touched."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from channel_audit.regrade import regrade_run
from channel_audit.runner import grade_pending, open_run, run_all
from channel_audit.schema import Graded
from channel_audit.store import JsonlStore
from helpers import build_ctx

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stale_run(tmp_path):
    """A run whose stored grades are all (wrongly) 'incorrect'."""
    ctx = build_ctx(tmp_path, n_problems=4, replicates=1, base_p=1.0, log_prompts=True)
    run_all(ctx, arms=["self_reflect"])
    store = JsonlStore(ctx.out_dir / "graded.jsonl", Graded)
    for g in ctx.gens:
        if g.arm != "failure_msg":
            store.append(
                Graded(
                    gen_key=g.gen_key,
                    extracted=None,
                    extraction_status="none",
                    correct=False,
                )
            )
    return ctx.out_dir


def test_regrading_writes_fresh_grades_to_a_new_directory(tmp_path):
    src = stale_run(tmp_path)
    before = {n: digest(src / n) for n in ("generations.jsonl", "graded.jsonl")}
    dst = tmp_path / "regraded"
    n = regrade_run(src, dst)
    assert n > 0
    new = open_run(dst)
    assert all(r.correct for r in new.graded)  # base_p=1.0: every answer is right
    old = open_run(src)
    assert not any(r.correct for r in old.graded)
    assert {n: digest(src / n) for n in before} == before  # source untouched


def test_generations_and_inputs_are_copied_byte_for_byte(tmp_path):
    src = stale_run(tmp_path)
    dst = tmp_path / "regraded"
    regrade_run(src, dst)
    for name in ("generations.jsonl", "problems.jsonl", "run.json", "prompts.jsonl"):
        assert digest(src / name) == digest(dst / name), name


def test_an_existing_destination_is_refused(tmp_path):
    src = stale_run(tmp_path)
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "x").write_text("data")
    with pytest.raises(FileExistsError):
        regrade_run(src, tmp_path / "taken")
    with pytest.raises(FileExistsError):
        regrade_run(src, src)


def test_the_regraded_run_is_complete(tmp_path):
    src = stale_run(tmp_path)
    dst = tmp_path / "regraded"
    regrade_run(src, dst)
    ctx = open_run(dst)
    assert grade_pending(ctx) == 0  # nothing left ungraded


def test_regrade_script(tmp_path):
    src = stale_run(tmp_path)
    dst = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "22_regrade.py"),
            "--src",
            str(src),
            "--out",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dst / "graded.jsonl").exists()
    assert "regraded" in proc.stdout
