"""scripts/31_decompose.py and scripts/32_figures.py on mock shards only."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from channel_audit.analysis.contrasts import analyze
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import LazyBackend, run_shards

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True
    )


def said(proc):
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def shard_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("exploratory") / "runs"
    problems = fake_problems(24)
    world = MockWorld.from_problems(problems, base_p=0.45)
    run_shards(
        root,
        RunConfig(model_id="mock/model", replicates=2, max_tokens=512),
        problems,
        LazyBackend(lambda: MockBackend(world)),
        size=12,
        arms=(*ARMS, *VARIANT_ARMS),
        log=lambda s: None,
    )
    return root


@pytest.fixture(scope="module")
def analysis_json(shard_root, tmp_path_factory):
    ctxs = [open_run(d) for d in sorted(shard_root.glob("shard_*"))]
    path = tmp_path_factory.mktemp("primary") / "analysis.json"
    path.write_text(json.dumps(analyze(ctxs, n_boot=100)), encoding="utf-8")
    return path


@pytest.mark.parametrize("script", ["31_decompose.py", "32_figures.py"])
def test_scripts_document_themselves(script):
    proc = run(script, "--help")
    assert proc.returncode == 0 and "usage" in proc.stdout.lower()


def test_decompose_writes_a_labelled_report_and_never_overwrites(shard_root, tmp_path):
    out = tmp_path / "exploratory"
    args = ("--root", str(shard_root), "--out-dir", str(out), "--n-boot", "100")
    proc = run("31_decompose.py", *args)
    assert proc.returncode == 0, said(proc)
    rep = json.loads((out / "exploratory.json").read_text(encoding="utf-8"))
    assert rep["status"]["exploratory"] is True and rep["n_problems"] == 24
    text = (out / "exploratory.md").read_text(encoding="utf-8")
    assert text.startswith("EXPLORATORY")
    assert proc.stdout.strip()
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    again = run("31_decompose.py", *args)
    assert again.returncode != 0 and "already" in said(again)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_the_problem_count_is_checked_when_given(shard_root, tmp_path):
    proc = run(
        "31_decompose.py",
        "--root",
        str(shard_root),
        "--out-dir",
        str(tmp_path / "o"),
        "--expect-problems",
        "600",
    )
    assert proc.returncode != 0 and "600" in said(proc)
    assert not (tmp_path / "o" / "exploratory.json").exists()


def test_a_shard_changed_since_it_was_marked_done_is_refused(shard_root, tmp_path):
    root = tmp_path / "edited"
    shutil.copytree(shard_root, root)
    gens = root / "shard_01" / "generations.jsonl"
    gens.write_text(gens.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    proc = run("31_decompose.py", "--root", str(root), "--out-dir", str(tmp_path / "o"))
    assert proc.returncode != 0 and "shard_01" in said(proc)


def test_a_shard_that_does_not_validate_is_refused(shard_root, tmp_path):
    root = tmp_path / "broken"
    shutil.copytree(shard_root, root)
    graded = root / "shard_00" / "graded.jsonl"
    lines = graded.read_text(encoding="utf-8").splitlines()
    graded.write_text("\n".join(lines[:-5]) + "\n", encoding="utf-8")
    proc = run("31_decompose.py", "--root", str(root), "--out-dir", str(tmp_path / "o"))
    assert proc.returncode != 0 and "shard_00" in said(proc)


def test_the_primary_reports_are_never_touched(shard_root, tmp_path):
    primary = tmp_path / "primary"
    primary.mkdir()
    (primary / "analysis.json").write_text("{}", encoding="utf-8")
    proc = run("31_decompose.py", "--root", str(shard_root), "--out-dir", str(primary))
    assert proc.returncode != 0
    assert (primary / "analysis.json").read_text(encoding="utf-8") == "{}"


def test_figures_are_drawn_from_the_two_json_reports(
    shard_root, analysis_json, tmp_path
):
    pytest.importorskip("matplotlib")
    exp = tmp_path / "exploratory"
    assert (
        run(
            "31_decompose.py",
            "--root",
            str(shard_root),
            "--out-dir",
            str(exp),
            "--n-boot",
            "100",
        ).returncode
        == 0
    )
    out = tmp_path / "figures"
    args = (
        "--analysis",
        str(analysis_json),
        "--decomposition",
        str(exp / "exploratory.json"),
        "--out-dir",
        str(out),
    )
    proc = run("32_figures.py", *args)
    assert proc.returncode == 0, said(proc)
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "fig2_accuracy_vs_tokens.png",
        "fig3_change_types.png",
        "fig4_source_of_change.png",
    ]
    assert run("32_figures.py", *args).returncode != 0  # never overwritten
