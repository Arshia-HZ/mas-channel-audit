"""The Colab/Kaggle runner notebook: well formed, and runnable on the mock."""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "runner.ipynb"


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_sources(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_a_clean_nbformat_4_file(nb):
    assert nb["nbformat"] == 4
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    assert nb["metadata"]["accelerator"] == "GPU"


def test_every_code_cell_is_valid_python(nb):
    for i, src in enumerate(code_sources(nb)):
        ast.parse(src, filename=f"cell {i}")


def test_it_opens_in_colab_from_this_repository_and_path(nb):
    first = "".join(nb["cells"][0]["source"])
    assert (
        "colab.research.google.com/github/Arshia-HZ/mas-channel-audit/blob/main/"
        "notebooks/runner.ipynb" in first
    )


def test_referenced_scripts_and_configs_exist(nb):
    text = "\n".join(code_sources(nb))
    scripts = set(re.findall(r"scripts/(\w+\.py)", text))
    configs = set(re.findall(r"configs/models/[\w.\-]+\.yaml", text))
    assert scripts and configs
    for s in scripts:
        assert (ROOT / "scripts" / s).is_file(), s
    for c in configs:
        assert (ROOT / c).is_file(), c


def test_settings_defaults_match_the_m3_check(nb):
    ns: dict = {}
    settings = next(
        s for s in code_sources(nb) if s.lstrip().startswith("MODEL_CONFIG")
    )
    exec(settings, ns)
    assert ns["MODEL_CONFIG"] == "configs/models/qwen3-1.7b.yaml"
    assert (ns["BACKEND"], ns["N_PROBLEMS"], ns["REPLICATES"]) == ("vllm", 50, 1)
    assert ns["REF"] == "main"
    assert ns["REPO_URL"].endswith("Arshia-HZ/mas-channel-audit.git")


def test_no_secret_is_embedded_or_printed(nb):
    text = "\n".join(code_sources(nb))
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "print(token" not in text


def test_the_notebook_never_freezes_the_prompts(nb):
    assert "03_freeze_prompts" not in "\n".join(code_sources(nb))


def test_whole_notebook_runs_end_to_end_on_the_mock_backend(nb, tmp_path, monkeypatch):
    """Executes every code cell locally; only the model itself is faked."""
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("MCA_RUN_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "path", list(sys.path))
    problems = fake_problems(18)
    (tmp_path / "check_problems.jsonl").write_text(
        "".join(p.model_dump_json() + "\n" for p in problems), encoding="utf-8"
    )
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run",
        lambda cfg, probs: MockBackend(MockWorld.from_problems(probs)),
    )
    ns: dict = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(nb)):
        src = src.replace("SHOW_PROMPT = True", "SHOW_PROMPT = False")  # needs network
        exec(compile(src, f"<cell {i}>", "exec"), ns)

    assert ns["validate_proc"].returncode == 0, ns["validate_proc"].stdout
    run = Path(ns["RUN"])
    assert run.parent == tmp_path
    assert len(ns["ctx"].gens) > 18 * 3
    assert (run / "review_sample.md").read_text(encoding="utf-8").count("## ") >= 30
    assert (run / "throughput.jsonl").exists()
    assert "match rate: 100.0%" in (run / "repro_report.txt").read_text()
    assert Path(ns["archive"]).exists()
    assert os.path.isdir(ns["REPO"])
