"""The M5 pilot notebook: well formed, and runnable end to end on the mock."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from channel_audit import data
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "pilot.ipynb"


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_sources(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_is_a_clean_nbformat_4_file_with_valid_python(nb):
    assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    for i, src in enumerate(code_sources(nb)):
        ast.parse(src, filename=f"cell {i}")


def test_it_opens_in_colab_from_this_repository(nb):
    assert "blob/main/notebooks/pilot.ipynb" in "".join(nb["cells"][0]["source"])


def test_settings_follow_the_plan(nb):
    ns: dict = {}
    settings = next(
        s for s in code_sources(nb) if s.lstrip().startswith("MODEL_CONFIG")
    )
    exec(settings, ns)
    assert ns["MODEL_CONFIG"] == "configs/models/qwen3-1.7b.yaml"
    assert (ns["REPLICATES"], ns["SHARD_SIZE"], ns["BACKEND"]) == (2, 50, "vllm")
    assert ns["UPLOAD"] is True and ns["MAX_NEW_SHARDS"] is None
    assert ns["HF_REPO"] == "Arshia-HZ/mas-channel-audit-raw"
    assert (ROOT / ns["MODEL_CONFIG"]).is_file()


def test_no_secret_is_embedded_or_printed(nb):
    text = "\n".join(code_sources(nb))
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "print(token" not in text


def fake_rows(config, revision):
    return [
        {
            "problem": f"{config} q{i}",
            "level": f"Level {i % 5 + 1}",
            "type": config,
            "solution": rf"So $\boxed{{{i}}}$.",
        }
        for i in range(130)
    ]


def test_whole_notebook_runs_end_to_end_on_the_mock(nb, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("MCA_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("MCA_SPLIT_DIR", str(tmp_path / "split"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(data, "_hf_rows", fake_rows)
    small = fake_problems(16)  # four (subject, level) groups, all with donors
    monkeypatch.setattr(data, "load_frozen", lambda directory, name: small)
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run",
        lambda cfg, probs: MockBackend(MockWorld.from_problems(probs)),
    )
    ns: dict = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(nb)):
        src = src.replace("SHARD_SIZE = 50", "SHARD_SIZE = 8")
        src = src.replace("UPLOAD = True", "UPLOAD = False")
        exec(compile(src, f"<cell {i}>", "exec"), ns)

    run = Path(ns["RUN"])
    assert run.name == "pilot_qwen3-1.7b_vllm"
    assert ns["ALL_DONE"] and ns["n_shards"] == 2
    assert [r.status for r in ns["results"]] == ["generated", "generated"]
    report = json.loads((run / "pilot_report.json").read_text(encoding="utf-8"))
    assert report["n_problems"] == 16 and report["replicates"] == 2
    assert set(report["gates"]) >= {"G1", "G2", "G3"}
    labels = json.loads((run / "g4" / "labels.json").read_text(encoding="utf-8"))
    assert len(labels) == 40 and set(labels.values()) == {""}
    assert (run / "g4" / "failure_sheet.md").exists()


def test_upload_without_a_token_stops_before_any_generation(nb, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("MCA_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("MCA_SPLIT_DIR", str(tmp_path / "split"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(data, "_hf_rows", fake_rows)
    monkeypatch.setattr(data, "load_frozen", lambda directory, name: fake_problems(16))
    built = []
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run",
        lambda cfg, probs: (
            built.append(1) or MockBackend(MockWorld.from_problems(probs))
        ),
    )
    ns: dict = {"__name__": "__main__"}
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        for i, src in enumerate(code_sources(nb)):
            exec(compile(src, f"<cell {i}>", "exec"), ns)
    assert built == []
