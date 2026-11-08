"""The v2 re-test notebook: well formed, and runnable end to end on the mock."""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from channel_audit import data, prompt_freeze
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.models import load_model_config
from channel_audit.runner import config_for_model
from channel_audit.shards import LazyBackend, run_shards

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "retest.ipynb"


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
    assert "blob/main/notebooks/retest.ipynb" in "".join(nb["cells"][0]["source"])


def test_it_refuses_to_run_until_prompts_v2_is_frozen(nb):
    text = "\n".join(code_sources(nb))
    assert "require_frozen" in text and '"v2"' in text


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
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(data, "_hf_rows", fake_rows)
    problems = fake_problems(16)
    monkeypatch.setattr(data, "load_frozen", lambda directory, name: problems)
    world = MockWorld.from_problems(problems)
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run",
        lambda cfg, probs: MockBackend(world),
    )
    return problems, world


def build_source(problems, world, tmp_path):
    model = load_model_config(ROOT / "configs/models/qwen3-1.7b.yaml")
    cfg = config_for_model(model, replicates=2, log_prompts=True)
    run_shards(
        tmp_path / "runs" / "pilot_qwen3-1.7b_vllm",
        cfg,
        problems,
        LazyBackend(lambda: MockBackend(world)),
        size=8,
        log=lambda s: None,
    )


def test_whole_notebook_runs_end_to_end_on_the_mock(nb, tmp_path, monkeypatch):
    problems, world = prepare(monkeypatch, tmp_path)
    build_source(problems, world, tmp_path)
    monkeypatch.setattr(prompt_freeze, "verify_manifest", lambda d: {"files": {}})
    (tmp_path / "frozen").mkdir()
    monkeypatch.setattr(
        prompt_freeze, "require_frozen", lambda d: {"files": {"failure_msg.txt": "x"}}
    )
    ns: dict = {"__name__": "__main__"}
    for i, src in enumerate(code_sources(nb)):
        src = src.replace("SHARD_SIZE = 50", "SHARD_SIZE = 8")
        src = src.replace("UPLOAD = True", "UPLOAD = False")
        exec(compile(src, f"<cell {i}>", "exec"), ns)

    run = Path(ns["RUN"])
    assert run.name == "pilot_qwen3-1.7b_vllm_v2"
    assert ns["ALL_DONE"] and [r.status for r in ns["results"]] == ["generated"] * 2
    report = json.loads((run / "pilot_report_v2.json").read_text(encoding="utf-8"))
    assert "failure_cause_v2" in report["arms"]
    labels = json.loads((run / "g4_v2" / "labels.json").read_text(encoding="utf-8"))
    assert len(labels) == 40 and all(k.startswith("v2|") for k in labels)
    # the source shards were not modified
    src_done = Path(ns["SOURCE"]) / "shard_00" / "SHARD_DONE.json"
    assert "variant" not in json.loads(src_done.read_text(encoding="utf-8"))


def test_an_unfrozen_v2_stops_the_notebook_before_any_generation(
    nb, tmp_path, monkeypatch
):
    problems, world = prepare(monkeypatch, tmp_path)
    build_source(problems, world, tmp_path)
    built = []
    monkeypatch.setattr(
        "channel_audit.runner.backend_for_run",
        lambda cfg, probs: built.append(1) or MockBackend(world),
    )
    ns: dict = {"__name__": "__main__"}
    with pytest.raises(prompt_freeze.FrozenPromptsError, match="not frozen"):
        for i, src in enumerate(code_sources(nb)):
            src = src.replace("UPLOAD = True", "UPLOAD = False")
            exec(compile(src, f"<cell {i}>", "exec"), ns)
    assert built == []
