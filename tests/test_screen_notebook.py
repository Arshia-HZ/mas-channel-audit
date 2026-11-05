"""The M4 screening notebook: well formed, and runnable end to end on the mock."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from channel_audit import data
from channel_audit.backends.mock import MockBackend, MockWorld

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "screen.ipynb"


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
    first = "".join(nb["cells"][0]["source"])
    assert "blob/main/notebooks/screen.ipynb" in first


def test_settings_default_to_one_primary_candidate_and_four_samples(nb):
    ns: dict = {}
    settings = next(
        s for s in code_sources(nb) if s.lstrip().startswith("MODEL_CONFIG")
    )
    exec(settings, ns)
    assert ns["MODEL_CONFIG"] == "configs/models/qwen3-1.7b.yaml"
    assert (ns["BACKEND"], ns["N_SAMPLES"], ns["REF"]) == ("vllm", 4, "main")
    assert (ROOT / ns["MODEL_CONFIG"]).is_file()


def test_every_listed_config_exists(nb):
    text = "".join(nb["cells"][0]["source"])
    configs = set(re.findall(r"configs/models/[\w.\-]+\.yaml", text))
    assert len(configs) == 4
    for c in configs:
        assert (ROOT / c).is_file(), c


def test_no_secret_is_embedded_or_printed(nb):
    text = "\n".join(code_sources(nb))
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "print(token" not in text


def fake_rows(config, revision):
    rows = []
    for i in range(130):
        rows.append(
            {
                "problem": f"{config} q{i}",
                "level": f"Level {i % 5 + 1}",
                "type": config,
                "solution": rf"So $\boxed{{{i}}}$.",
            }
        )
    return rows


def test_whole_notebook_runs_end_to_end_on_the_mock(nb, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("MCA_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("MCA_SPLIT_DIR", str(tmp_path / "split"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(data, "_hf_rows", fake_rows)
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run",
        lambda cfg, probs: MockBackend(MockWorld.from_problems(probs)),
    )
    ns: dict = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(nb)):
        exec(compile(src, f"<cell {i}>", "exec"), ns)

    run = Path(ns["RUN"])
    assert run.name == "screen_qwen3-1.7b_vllm"
    assert len(ns["problems"]) == 300
    assert len(ns["ctx"].gens) == 300 * 4
    report = json.loads((run / "screen_report.json").read_text(encoding="utf-8"))
    assert report["n_samples"] == 1200 and report["n_problems"] == 300
    assert Path(ns["archive"]).exists()


def test_notebook_runs_both_prompts_and_writes_a_report_for_each(
    nb, tmp_path, monkeypatch
):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("MCA_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("MCA_SPLIT_DIR", str(tmp_path / "split"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(data, "_hf_rows", fake_rows)
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run",
        lambda cfg, probs: MockBackend(MockWorld.from_problems(probs)),
    )
    ns: dict = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(nb)):
        src = src.replace(
            'SCREEN_PROMPTS = ["round0"]', 'SCREEN_PROMPTS = ["round0", "plain"]'
        )
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    run = Path(ns["RUN"])
    assert len(ns["ctx"].gens) == 300 * 4 * 2
    plain = json.loads((run / "screen_report_plain.json").read_text(encoding="utf-8"))
    assert plain["prompt"] == "plain" and plain["n_samples"] == 1200
    assert (run / "screen_report.json").exists()


def test_settings_default_to_the_study_prompt_only(nb):
    ns: dict = {}
    settings = next(
        s for s in code_sources(nb) if s.lstrip().startswith("MODEL_CONFIG")
    )
    exec(settings, ns)
    assert ns["SCREEN_PROMPTS"] == ["round0"]
