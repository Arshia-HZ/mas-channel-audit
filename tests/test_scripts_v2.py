"""Scripts for the v2 failure-message re-test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.prompt_freeze import (
    FrozenPromptsError,
    require_frozen,
    write_manifest,
)
from channel_audit.runner import RunConfig
from channel_audit.shards import LazyBackend, run_shards, run_variant_shards

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True
    )


def test_the_retest_script_documents_itself():
    proc = run("05_retest_failure.py", "--help")
    assert proc.returncode == 0 and "usage" in proc.stdout.lower()


def test_an_unfrozen_prompt_directory_is_refused(tmp_path):
    (tmp_path / "failure_msg.txt").write_text("<<question>> <<solution>>")
    with pytest.raises(FrozenPromptsError, match="not frozen"):
        require_frozen(tmp_path)
    write_manifest(tmp_path)
    assert "failure_msg.txt" in require_frozen(tmp_path)["files"]


@pytest.fixture(scope="module")
def v2_root(tmp_path_factory):
    base = tmp_path_factory.mktemp("v2scripts")
    problems = fake_problems(16)
    world = MockWorld.from_problems(problems)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        base / "pilot", cfg, problems, LazyBackend(lambda: MockBackend(world)),
        size=8, log=lambda s: None,
    )  # fmt: skip
    run_variant_shards(
        base / "pilot", base / "v2", cfg, problems,
        LazyBackend(lambda: MockBackend(world)), size=8, log=lambda s: None,
    )  # fmt: skip
    return base / "v2"


def test_the_sample_script_draws_v2_messages(v2_root, tmp_path):
    out = tmp_path / "g4"
    proc = run(
        "05_failure_sample.py", "--root", str(v2_root), "--out-dir", str(out),
        "--variant", "v2", "--seed", "1",
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    assert len(labels) == 40 and all(k.startswith("v2|") for k in labels)


def test_the_gate_check_reports_format_and_the_gold_audit_for_v2(v2_root, tmp_path):
    out = tmp_path / "g4"
    run(
        "05_failure_sample.py", "--root", str(v2_root), "--out-dir", str(out),
        "--variant", "v2", "--seed", "1",
    )  # fmt: skip
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    (out / "labels.json").write_text(
        json.dumps({k: "wrong" for k in labels}), encoding="utf-8"
    )
    proc = run(
        "05_gate_check.py", "--root", str(v2_root), "--confirm-available", "3250",
        "--labels", str(out / "labels.json"), "--variant", "v2", "--seed", "1",
        "--json", str(tmp_path / "r.json"),
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "failure messages (v2): 40/40 single sentence" in proc.stdout
    assert "G4 failure messages: 100% not vague" in proc.stdout
    assert "failure_cause_v2" in proc.stdout
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["G4_audit"]["format"]["n"] == 40
