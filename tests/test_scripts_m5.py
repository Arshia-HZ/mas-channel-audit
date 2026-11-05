"""M5 scripts that need no GPU or network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.runner import RunConfig
from channel_audit.shards import LazyBackend, run_shards

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def pilot_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("pilot") / "runs"
    problems = fake_problems(16)
    world = MockWorld.from_problems(problems)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        root,
        cfg,
        problems,
        LazyBackend(lambda: MockBackend(world)),
        size=8,
        log=lambda s: None,
    )
    return root


@pytest.mark.parametrize(
    "script", ["05_pilot_shards.py", "05_gate_check.py", "05_failure_sample.py"]
)
def test_every_m5_script_starts_and_documents_itself(script):
    proc = run(script, "--help")
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()


def test_gate_check_prints_the_gates_the_arm_table_and_the_power_inputs(
    pilot_root, tmp_path
):
    out = tmp_path / "report.json"
    proc = run(
        "05_gate_check.py",
        "--root",
        str(pilot_root),
        "--confirm-available",
        "3250",
        "--json",
        str(out),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for text in ("G1 round-0 disagreement", "G2 recoverable", "G3", "G4", "n@3pt"):
        assert text in proc.stdout
    assert "pending" in proc.stdout  # no labels yet
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["n_problems"] == 16 and "contrasts" in rep and rep["G4"] is None


def test_the_failure_sample_script_writes_a_sheet_and_a_blank_label_file(
    pilot_root, tmp_path
):
    out = tmp_path / "g4"
    proc = run("05_failure_sample.py", "--root", str(pilot_root), "--out-dir", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    assert len(labels) == 40 and set(labels.values()) == {""}
    assert "Failure messages to label" in (out / "failure_sheet.md").read_text(
        encoding="utf-8"
    )
    again = run(
        "05_failure_sample.py", "--root", str(pilot_root), "--out-dir", str(out)
    )
    assert again.returncode != 0  # labels are never overwritten


def test_gate_check_computes_g4_from_the_completed_labels(pilot_root, tmp_path):
    out = tmp_path / "g4"
    run("05_failure_sample.py", "--root", str(pilot_root), "--out-dir", str(out))
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    labelled = {k: ("useful" if i % 2 else "vague") for i, k in enumerate(labels)}
    (out / "labels.json").write_text(json.dumps(labelled), encoding="utf-8")
    proc = run(
        "05_gate_check.py",
        "--root",
        str(pilot_root),
        "--labels",
        str(out / "labels.json"),
        "--confirm-available",
        "3250",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "G4 failure messages: 50% not vague" in proc.stdout
    assert "PASS" in proc.stdout.split("G4 failure messages")[1].splitlines()[0]


def test_gate_check_needs_exactly_one_source(pilot_root):
    assert run("05_gate_check.py").returncode != 0
    both = run("05_gate_check.py", "--root", str(pilot_root), "--runs", "x")
    assert both.returncode != 0
