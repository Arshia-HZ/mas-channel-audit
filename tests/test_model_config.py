"""Model configs: validated, pinned, and Qwen3 fixed to non-thinking mode."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from channel_audit.models import ModelConfig, load_model_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"
FILES = sorted(CONFIG_DIR.glob("*.yaml"))


def base(**over):
    cfg = {
        "model_id": "org/plain-model",
        "backend": "vllm",
        "dtype": "float16",
        "revision": "a" * 40,
        "max_tokens": 1024,
        "max_model_len": 4096,
    }
    return {**cfg, **over}


def test_the_planned_models_all_have_a_config():
    names = {p.stem for p in FILES}
    assert {
        "qwen2.5-math-1.5b",
        "qwen3-1.7b",
        "llama-3.2-3b-instruct",
        "qwen3-4b",
    } <= names


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_every_shipped_config_loads_and_is_pinned(path):
    cfg = load_model_config(path)
    assert re.fullmatch(r"[0-9a-f]{40}", cfg.revision), "pin the model revision"
    assert cfg.dtype == "float16", "T4 has no bfloat16"
    assert cfg.backend in ("vllm", "hf")
    assert cfg.max_tokens < cfg.max_model_len


@pytest.mark.parametrize("path", [p for p in FILES if "qwen3" in p.stem], ids=str)
def test_qwen3_configs_disable_thinking(path):
    assert load_model_config(path).chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize("path", [p for p in FILES if "qwen" in p.stem], ids=str)
def test_qwen_configs_stop_on_im_end_and_endoftext(path):
    assert {151645, 151643} <= set(load_model_config(path).stop_token_ids)


def test_qwen3_without_non_thinking_is_rejected():
    with pytest.raises(ValidationError, match="enable_thinking"):
        ModelConfig(**base(model_id="Qwen/Qwen3-1.7B"))
    with pytest.raises(ValidationError, match="enable_thinking"):
        ModelConfig(
            **base(
                model_id="Qwen/Qwen3-4B",
                chat_template_kwargs={"enable_thinking": True},
            )
        )


def test_non_qwen3_models_do_not_need_the_flag():
    assert ModelConfig(**base()).chat_template_kwargs == {}


def test_unknown_fields_backends_and_dtypes_are_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(**base(surprise=1))
    with pytest.raises(ValidationError):
        ModelConfig(**base(backend="tgi"))
    with pytest.raises(ValidationError):
        ModelConfig(**base(dtype="int4"))


def test_revision_must_be_a_full_commit_hash():
    with pytest.raises(ValidationError):
        ModelConfig(**base(revision="main"))


def test_max_tokens_must_fit_in_the_context():
    with pytest.raises(ValidationError):
        ModelConfig(**base(max_tokens=4096, max_model_len=4096))


def test_config_is_immutable_and_round_trips_json():
    cfg = ModelConfig(**base())
    with pytest.raises(ValidationError):
        cfg.max_tokens = 5
    assert ModelConfig.model_validate_json(cfg.model_dump_json()) == cfg


def test_loader_rejects_a_non_mapping(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_model_config(p)


# The full_solution reader sees its own and two peer solutions (each up to
# max_tokens) plus the problem, then writes its own reply of up to max_tokens.
PROMPT_RESERVE = 1024  # problem statement and instructions
LONG_OUTPUT = 2048  # the limit the T4 check showed is needed


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_context_fits_three_full_solutions_at_the_long_output_limit(path):
    cfg = load_model_config(path)
    if cfg.max_model_len < 8192:
        pytest.skip(f"{path.stem}: native context too small; see its config comment")
    need = 4 * max(cfg.max_tokens, LONG_OUTPUT) + PROMPT_RESERVE
    assert cfg.max_model_len >= need, (cfg.max_model_len, need)
