"""M4 scripts that need no GPU or network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from channel_audit.runner import grade_pending
from channel_audit.screening import run_screen
from helpers import build_ctx

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "script", ["04_freeze_problems.py", "04_screen.py", "04_screen_report.py"]
)
def test_every_m4_script_starts_and_documents_itself(script):
    proc = run(script, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


def test_screen_report_compares_finished_runs_and_writes_json(tmp_path):
    runs = []
    for name, base_p in (("a", 1.0), ("b", 0.0)):
        ctx = build_ctx(tmp_path / name, n_problems=6, replicates=1, base_p=base_p)
        run_screen(ctx)
        grade_pending(ctx)
        runs.append(str(ctx.out_dir))
    out = tmp_path / "summary.json"
    proc = run("04_screen_report.py", "--runs", *runs, "--json", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("mock/model") >= 2
    reports = json.loads(out.read_text(encoding="utf-8"))
    assert [r["accuracy"] for r in reports] == [1.0, 0.0]


def test_screen_report_on_a_run_without_screening_fails_clearly(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3, replicates=1)
    proc = run("04_screen_report.py", "--runs", str(ctx.out_dir))
    assert proc.returncode != 0 and "screen" in proc.stderr
