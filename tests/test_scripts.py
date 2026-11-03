"""The numbered scripts run end to end on fake problems with no GPU."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(script, *args, check=True):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
    )
    if check:
        assert proc.returncode == 0, f"{script}\n{proc.stdout}\n{proc.stderr}"
    return proc


def test_full_mock_pipeline_on_twenty_fake_problems(tmp_path):
    out = str(tmp_path / "run")
    run("10_round0.py", "--out", out, "--n-fake-problems", "20", "--replicates", "2")
    run("11_sc_pool.py", "--out", out)
    run("12_failure_messages.py", "--out", out)
    run("13_branch_arms.py", "--out", out)
    run("20_grade.py", "--out", out)
    proc = run("21_validate_run.py", "--out", out)
    assert "errors: 0" in proc.stdout
    assert "self_reflect" in proc.stdout and "failure_cause" in proc.stdout


def test_scripts_are_resumable(tmp_path):
    out = str(tmp_path / "run")
    args = ("--out", out, "--n-fake-problems", "8", "--replicates", "1")
    run("10_round0.py", *args)
    gens = tmp_path / "run" / "generations.jsonl"
    n1 = gens.read_text().count("\n")
    run("10_round0.py", *args)
    assert gens.read_text().count("\n") == n1


def test_validate_fails_when_a_step_was_skipped(tmp_path):
    out = str(tmp_path / "run")
    run("10_round0.py", "--out", out, "--n-fake-problems", "8", "--replicates", "1")
    proc = run("21_validate_run.py", "--out", out, check=False)
    assert proc.returncode == 1
    assert "missing" in proc.stdout


def test_branching_before_round0_is_refused(tmp_path):
    out = str(tmp_path / "run")
    proc = run("13_branch_arms.py", "--out", out, check=False)
    assert proc.returncode != 0
