"""scripts/29_freeze_analysis.py and scripts/30_analyze.py, on mock shards only."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.runner import RunConfig
from channel_audit.shards import LazyBackend, run_shards

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
HARDWARE = {"gpu": "mock gpu", "python": "3.12", "packages": {"vllm": "0.0"}}


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True
    )


def said(proc):
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def shard_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("confirm") / "runs"
    problems = fake_problems(24)
    world = MockWorld.from_problems(problems, base_p=0.45)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        root,
        cfg,
        problems,
        LazyBackend(lambda: MockBackend(world)),
        size=12,
        arms=(*ARMS, *VARIANT_ARMS),
        log=lambda s: None,
    )
    (root / "HARDWARE.json").write_text(json.dumps(HARDWARE), encoding="utf-8")
    return root


@pytest.fixture
def freeze_file(tmp_path):
    path = tmp_path / "analysis_freeze.json"
    proc = run("29_freeze_analysis.py", "--out", str(path))
    assert proc.returncode == 0, said(proc)
    return path


def frozen_args(shard_root, out, freeze_file, *extra):
    return (
        "--root",
        str(shard_root),
        "--out-dir",
        str(out),
        "--freeze-file",
        str(freeze_file),
        "--n-boot",
        "100",
        *extra,
    )


@pytest.mark.parametrize("script", ["29_freeze_analysis.py", "30_analyze.py"])
def test_scripts_document_themselves(script):
    proc = run(script, "--help")
    assert proc.returncode == 0 and "usage" in proc.stdout.lower()


def test_freezing_writes_a_manifest_once(tmp_path):
    path = tmp_path / "f.json"
    first = run("29_freeze_analysis.py", "--out", str(path))
    assert first.returncode == 0 and "combined sha256" in first.stdout
    assert {"analysis", "pipeline", "combined_sha256"} <= set(
        json.loads(path.read_text(encoding="utf-8"))
    )
    again = run("29_freeze_analysis.py", "--out", str(path))
    assert again.returncode != 0


def test_a_frozen_run_writes_the_report_the_table_and_the_provenance(
    shard_root, tmp_path, freeze_file
):
    out = tmp_path / "analysis"
    proc = run(
        "30_analyze.py",
        *frozen_args(shard_root, out, freeze_file, "--expect-problems", "24"),
    )
    assert proc.returncode == 0, said(proc)
    rep = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert rep["analyses"]["main"]["n_problems"] == 24
    assert rep["status"]["rehearsal"] is False
    text = (out / "analysis.md").read_text(encoding="utf-8")
    assert "best_low_bandwidth_vs_sc" in text and "REHEARSAL" not in text
    info = json.loads((out / "RUN.json").read_text(encoding="utf-8"))
    frozen = json.loads(freeze_file.read_text(encoding="utf-8"))
    assert info["analysis_combined_sha256"] == frozen["combined_sha256"]
    assert info["frozen_manifest"] == frozen
    assert len(info["shards"]) == 2
    assert all(len(s["generations_sha256"]) == 64 for s in info["shards"])
    assert info["hardware"] == HARDWARE
    assert info["argv"][:2] == ["--root", str(shard_root)]
    env = info["environment"]
    assert {"python", "numpy", "math-verify", "pydantic"} <= set(env)
    assert all(env.values())
    assert proc.stdout.strip()  # the table is also printed


def test_it_runs_once_and_never_overwrites_a_result(shard_root, tmp_path, freeze_file):
    out = tmp_path / "analysis"
    args = frozen_args(shard_root, out, freeze_file, "--expect-problems", "24")
    assert run("30_analyze.py", *args).returncode == 0
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    again = run("30_analyze.py", *args)
    assert again.returncode != 0 and "already" in said(again)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


@pytest.mark.parametrize("leftover", ["analysis.json", "analysis.md", "RUN.json"])
def test_any_earlier_output_blocks_a_new_run(shard_root, tmp_path, leftover):
    out = tmp_path / "o"
    out.mkdir()
    (out / leftover).write_text("old", encoding="utf-8")
    proc = run(
        "30_analyze.py",
        "--root",
        str(shard_root),
        "--out-dir",
        str(out),
        "--rehearsal",
        "--n-boot",
        "100",
    )
    assert proc.returncode != 0 and "already" in said(proc)
    assert (out / leftover).read_text(encoding="utf-8") == "old"


def test_the_problem_count_is_required_outside_a_rehearsal(
    shard_root, tmp_path, freeze_file
):
    proc = run("30_analyze.py", *frozen_args(shard_root, tmp_path / "o", freeze_file))
    assert proc.returncode != 0 and "--expect-problems" in said(proc)
    assert not (tmp_path / "o" / "analysis.json").exists()


def test_without_a_freeze_it_refuses_real_analysis(shard_root, tmp_path):
    proc = run(
        "30_analyze.py",
        "--root",
        str(shard_root),
        "--out-dir",
        str(tmp_path / "o"),
        "--freeze-file",
        str(tmp_path / "missing.json"),
        "--expect-problems",
        "24",
    )
    assert proc.returncode != 0
    assert "29_freeze_analysis" in said(proc)
    assert not (tmp_path / "o" / "analysis.json").exists()


def test_a_changed_analysis_or_pipeline_file_blocks_the_run(shard_root, tmp_path):
    frozen = json.loads(run_freeze_to(tmp_path / "f.json").read_text(encoding="utf-8"))
    for group, rel in (
        ("analysis", "src/channel_audit/analysis/contrasts.py"),
        ("pipeline", "src/channel_audit/runner.py"),
    ):
        bad = json.loads(json.dumps(frozen))
        bad[group][rel] = "0" * 64
        path = tmp_path / f"bad_{group}.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        proc = run(
            "30_analyze.py",
            "--root",
            str(shard_root),
            "--out-dir",
            str(tmp_path / f"o_{group}"),
            "--freeze-file",
            str(path),
            "--expect-problems",
            "24",
        )
        assert proc.returncode != 0 and f"{rel}: changed" in said(proc)


def run_freeze_to(path):
    assert run("29_freeze_analysis.py", "--out", str(path)).returncode == 0
    return path


def test_a_rehearsal_on_mock_data_is_allowed_without_a_freeze_and_is_labelled(
    shard_root, tmp_path
):
    out = tmp_path / "o"
    proc = run(
        "30_analyze.py",
        "--root",
        str(shard_root),
        "--out-dir",
        str(out),
        "--rehearsal",
        "--n-boot",
        "100",
    )
    assert proc.returncode == 0, said(proc)
    assert "REHEARSAL" in (out / "analysis.md").read_text(encoding="utf-8")
    rep = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert rep["status"]["rehearsal"] is True


def test_a_rehearsal_is_refused_on_real_backend_data(shard_root, tmp_path):
    root = tmp_path / "real"
    shutil.copytree(shard_root, root)
    for run_json in root.glob("shard_*/run.json"):
        cfg = json.loads(run_json.read_text(encoding="utf-8"))
        cfg["backend"] = "vllm"
        run_json.write_text(json.dumps(cfg), encoding="utf-8")
    proc = run(
        "30_analyze.py",
        "--root",
        str(root),
        "--out-dir",
        str(tmp_path / "o"),
        "--rehearsal",
        "--n-boot",
        "100",
    )
    assert proc.returncode != 0 and "mock" in said(proc)
    assert not (tmp_path / "o" / "analysis.json").exists()


def test_the_wrong_number_of_problems_is_an_error(shard_root, tmp_path):
    proc = run(
        "30_analyze.py",
        "--root",
        str(shard_root),
        "--out-dir",
        str(tmp_path / "o"),
        "--rehearsal",
        "--expect-problems",
        "600",
    )
    assert proc.returncode != 0 and "600" in said(proc)
    assert not (tmp_path / "o" / "analysis.json").exists()


def test_a_missing_shard_changes_the_count_and_is_caught(shard_root, tmp_path):
    root = tmp_path / "one_of_two"
    shutil.copytree(shard_root, root)
    shutil.rmtree(root / "shard_01")
    proc = run(
        "30_analyze.py",
        "--root",
        str(root),
        "--out-dir",
        str(tmp_path / "o"),
        "--rehearsal",
        "--expect-problems",
        "24",
    )
    assert proc.returncode != 0 and "found 12" in said(proc)


def test_a_shard_that_does_not_validate_stops_the_analysis(shard_root, tmp_path):
    root = tmp_path / "broken"
    shutil.copytree(shard_root, root)
    graded = root / "shard_00" / "graded.jsonl"
    lines = graded.read_text(encoding="utf-8").splitlines()
    graded.write_text("\n".join(lines[:-5]) + "\n", encoding="utf-8")
    proc = run(
        "30_analyze.py",
        "--root",
        str(root),
        "--out-dir",
        str(tmp_path / "o"),
        "--rehearsal",
    )
    assert proc.returncode != 0 and "shard_00" in said(proc)
    assert not (tmp_path / "o" / "analysis.json").exists()


def test_a_stored_grade_that_the_grader_would_not_give_stops_the_analysis(
    shard_root, tmp_path
):
    root = tmp_path / "tampered"
    shutil.copytree(shard_root, root)
    path = root / "shard_01" / "graded.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    rows[7]["correct"] = not rows[7]["correct"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    proc = run(
        "30_analyze.py",
        "--root",
        str(root),
        "--out-dir",
        str(tmp_path / "o"),
        "--rehearsal",
    )
    assert proc.returncode != 0 and "shard_01" in said(proc) and "differ" in said(proc)
    assert not (tmp_path / "o" / "analysis.json").exists()


def test_a_shard_whose_generations_changed_since_it_was_marked_done_is_refused(
    shard_root, tmp_path
):
    root = tmp_path / "edited"
    shutil.copytree(shard_root, root)
    gens = root / "shard_01" / "generations.jsonl"
    gens.write_text(gens.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    proc = run(
        "30_analyze.py",
        "--root",
        str(root),
        "--out-dir",
        str(tmp_path / "o"),
        "--rehearsal",
    )
    assert proc.returncode != 0 and "shard_01" in said(proc)
