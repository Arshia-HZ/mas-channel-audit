"""Runner changes for real backends: lazy engine, prompt log, throughput log."""

from __future__ import annotations

import json
import sys

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.models import ModelConfig
from channel_audit.runner import (
    RunConfig,
    backend_for_run,
    config_for_model,
    create_run,
    open_run,
    run_all,
    run_round0,
    throughput_summary,
)
from helpers import RecordingBackend, build_ctx

MODEL = ModelConfig(
    model_id="org/model",
    backend="vllm",
    dtype="float16",
    revision="d" * 40,
    max_tokens=64,
    max_model_len=512,
)


def test_run_config_embeds_the_model_config_and_round_trips(tmp_path):
    cfg = RunConfig(model_id="org/model", backend="vllm", model=MODEL)
    assert RunConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_config_for_model_takes_every_setting_from_the_model_config():
    cfg = config_for_model(MODEL, replicates=3, log_prompts=True)
    assert (cfg.model_id, cfg.backend) == ("org/model", "vllm")
    assert (cfg.max_tokens, cfg.context_limit) == (64, 512)
    assert cfg.model == MODEL and cfg.replicates == 3 and cfg.log_prompts


def test_real_backends_need_a_model_config():
    problems = fake_problems(2)
    with pytest.raises(ValueError, match="model"):
        backend_for_run(RunConfig(model_id="x", backend="vllm"), problems)
    with pytest.raises(ValueError, match="model"):
        backend_for_run(RunConfig(model_id="x", backend="hf"), problems)


def test_unknown_backend_is_an_error():
    with pytest.raises(ValueError, match="backend"):
        backend_for_run(RunConfig(model_id="x", backend="tgi"), fake_problems(2))


def test_real_backend_without_its_library_fails_clearly(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vllm", None)
    cfg = RunConfig(model_id="org/model", backend="vllm", model=MODEL)
    with pytest.raises(ImportError, match="vllm"):
        backend_for_run(cfg, fake_problems(2))


def test_open_run_does_not_build_a_backend_until_it_is_needed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)  # never load a model in tests
    problems = fake_problems(2)
    cfg = RunConfig(model_id="org/model", backend="vllm", model=MODEL)
    create_run(
        tmp_path / "r", cfg, problems, MockBackend(MockWorld.from_problems(problems))
    )
    ctx = open_run(tmp_path / "r")  # would need vllm if it built the engine eagerly
    assert len(ctx.problems) == 2
    with pytest.raises(ImportError):
        ctx.backend  # noqa: B018 - first use builds the engine


def test_prompts_are_logged_exactly_as_sent_when_requested(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=4, log_prompts=True)
    run_all(ctx, arms=["self_reflect", "failure_cause"])
    sent = {}
    for r in ctx.backend.requests:
        sent.setdefault(json.dumps(r.messages), 0)
    assert ctx.prompts is not None
    logged = {json.dumps(p.messages) for p in ctx.prompts}
    assert logged == set(sent)
    assert len(ctx.prompts) == len(ctx.gens)
    for g in ctx.gens:
        assert g.gen_key in ctx.prompts


def test_prompt_log_is_off_by_default(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3)
    run_round0(ctx)
    assert ctx.prompts is None
    assert not (ctx.out_dir / "prompts.jsonl").exists()


def test_every_backend_call_appends_a_throughput_line(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3, replicates=1)
    run_round0(ctx)
    lines = (ctx.out_dir / "throughput.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["n_requests"] == 9
    assert rec["arms"] == {"round0": 9}
    assert rec["completion_tokens"] == sum(g.completion_tokens for g in ctx.gens)
    assert rec["prompt_tokens"] == sum(g.prompt_tokens for g in ctx.gens)
    assert rec["seconds"] >= 0 and rec["backend"] == "mock"
    assert rec["model_id"] == "mock/model"
    assert "completion_tokens_per_s" in rec


def test_nothing_to_do_logs_nothing(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3, replicates=1)
    run_round0(ctx)
    run_round0(ctx)
    assert len((ctx.out_dir / "throughput.jsonl").read_text().splitlines()) == 1


def test_throughput_summary_aggregates_all_batches(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=4, replicates=1)
    run_all(ctx, arms=["self_reflect"])
    s = throughput_summary(ctx.out_dir)
    assert s["n_requests"] == len(ctx.gens)
    assert s["completion_tokens"] == sum(g.completion_tokens for g in ctx.gens)
    assert s["n_batches"] >= 3
    assert s["completion_tokens_per_s"] > 0


def test_throughput_summary_of_a_run_without_a_log_is_empty(tmp_path):
    assert throughput_summary(tmp_path)["n_requests"] == 0


def test_recording_backend_still_reports_its_name(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=2)
    assert isinstance(ctx.backend, RecordingBackend)
