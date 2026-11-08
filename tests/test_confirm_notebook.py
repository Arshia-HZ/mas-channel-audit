"""The M6 confirmatory notebook: well formed, quality only, runnable on the mock."""

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
NOTEBOOK = ROOT / "notebooks" / "confirm.ipynb"


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
    assert "blob/main/notebooks/confirm.ipynb" in "".join(nb["cells"][0]["source"])


def test_settings_follow_the_preregistered_design(nb):
    ns: dict = {}
    settings = next(
        s for s in code_sources(nb) if s.lstrip().startswith("MODEL_CONFIG")
    )
    exec(settings, ns)
    assert ns["MODEL_CONFIG"] == "configs/models/qwen3-1.7b.yaml"
    assert (ns["REPLICATES"], ns["SHARD_SIZE"], ns["N_PROBLEMS"]) == (2, 50, 600)
    assert ns["WITH_V2"] is True and ns["UPLOAD"] is True
    assert ns["MAX_NEW_SHARDS"] is None


def test_the_notebook_never_computes_an_outcome(nb):
    text = "\n".join(code_sources(nb))
    for forbidden in (
        "pilot_report",
        "collect_units",
        "plurality",
        "is_correct",
        "sc_credit",
        ".correct",
    ):
        assert forbidden not in text, forbidden
    assert "run_quality" in text


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


def prepare(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("MCA_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("MCA_SPLIT_DIR", str(tmp_path / "split"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(data, "_hf_rows", fake_rows)
    problems = fake_problems(16)
    pilot = [
        p.model_copy(update={"problem_id": "pilot-" + p.problem_id}) for p in problems
    ]
    monkeypatch.setattr(data, "confirmatory_problems", lambda d, n: problems)
    monkeypatch.setattr(data, "load_frozen", lambda d, name: pilot)
    world = MockWorld.from_problems(problems)
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run", lambda cfg, probs: MockBackend(world)
    )
    return problems


def test_whole_notebook_runs_end_to_end_on_the_mock(nb, tmp_path, monkeypatch, capsys):
    prepare(monkeypatch, tmp_path)
    ns: dict = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(nb)):
        src = src.replace("SHARD_SIZE = 50", "SHARD_SIZE = 8")
        src = src.replace("UPLOAD = True", "UPLOAD = False")
        exec(compile(src, f"<cell {i}>", "exec"), ns)

    run = Path(ns["RUN"])
    assert run.name == "confirm_qwen3-1.7b_vllm"
    assert ns["ALL_DONE"] and [r.status for r in ns["results"]] == ["generated"] * 2
    quality = json.loads((run / "quality_report.json").read_text(encoding="utf-8"))
    assert "failure_cause_v2" in quality["arms"] and "round0" in quality["arms"]
    assert quality["n_problems"] == 16 and set(quality["leakage"]) == {"v1", "v2"}
    out = capsys.readouterr().out.lower()
    assert "accuracy" not in out and "gain" not in out


def test_without_v2_the_run_has_only_the_seven_arms(nb, tmp_path, monkeypatch):
    prepare(monkeypatch, tmp_path)
    ns: dict = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(nb)):
        src = src.replace("SHARD_SIZE = 50", "SHARD_SIZE = 8")
        src = src.replace("UPLOAD = True", "UPLOAD = False")
        src = src.replace("WITH_V2 = True", "WITH_V2 = False")
        exec(compile(src, f"<cell {i}>", "exec"), ns)
    quality = json.loads((Path(ns["RUN"]) / "quality_report.json").read_text("utf-8"))
    assert "failure_cause_v2" not in quality["arms"]


def test_a_confirmatory_problem_that_is_also_in_the_pilot_stops_the_run(
    nb, tmp_path, monkeypatch
):
    problems = prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(data, "load_frozen", lambda d, name: problems[:3])
    ns: dict = {"__name__": "__main__"}
    with pytest.raises(RuntimeError, match="also pilot problems"):
        for i, src in enumerate(code_sources(nb)):
            src = src.replace("UPLOAD = True", "UPLOAD = False")
            exec(compile(src, f"<cell {i}>", "exec"), ns)


def test_upload_without_a_token_stops_before_any_generation(nb, tmp_path, monkeypatch):
    prepare(monkeypatch, tmp_path)
    built = []
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run",
        lambda cfg, probs: built.append(1),
    )
    ns: dict = {"__name__": "__main__"}
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        for i, src in enumerate(code_sources(nb)):
            exec(compile(src, f"<cell {i}>", "exec"), ns)
    assert built == []
