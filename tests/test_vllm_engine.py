"""vLLM backend against a fake engine: seeds, token counts, finish reasons."""

from __future__ import annotations

import sys

import pytest

from channel_audit.backends.base import GenRequest
from channel_audit.backends.vllm_engine import VllmBackend
from channel_audit.models import ModelConfig
from fakes import FakeCompletion, FakeLLM, FakeSamplingParams, FakeTokenizer

CFG = ModelConfig(
    model_id="org/model",
    backend="vllm",
    dtype="float16",
    revision="b" * 40,
    max_tokens=32,
    max_model_len=200,
    stop_token_ids=[151645, 151643],
    chat_template_kwargs={"enable_thinking": False},
)


def make(llm=None, tok=None):
    return VllmBackend(
        CFG,
        llm=llm or FakeLLM(),
        tokenizer=tok or FakeTokenizer(),
        sampling_params_cls=FakeSamplingParams,
        version="9.9.9",
    )


def req(content="2+2?", seed=11, max_tokens=32, temperature=0.7, top_p=1.0, meta=None):
    return GenRequest(
        messages=[{"role": "user", "content": content}],
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        meta=meta or {"arm": "round0"},
    )


def test_identity_and_version():
    b = make()
    assert (b.name, b.version) == ("vllm", "9.9.9")


def test_sampling_params_carry_seed_decoding_settings_and_stop_ids():
    llm = FakeLLM()
    make(llm).generate([req(seed=5), req(seed=6, temperature=0.3, max_tokens=7)])
    _, params = llm.calls[0]
    assert [p.seed for p in params] == [5, 6]
    assert [p.temperature for p in params] == [0.7, 0.3]
    assert [p.max_tokens for p in params] == [32, 7]
    assert all(p.top_p == 1.0 and p.n == 1 for p in params)
    # no model-default sampling tweaks leak in: top-k off, no repetition penalty
    assert all(p.top_k == -1 and p.repetition_penalty == 1.0 for p in params)
    assert all(p.stop_token_ids == [151645, 151643] for p in params)


def test_prompt_is_rendered_with_the_template_and_sent_as_token_ids():
    llm, tok = FakeLLM(), FakeTokenizer()
    make(llm, tok).generate([req("hello")])
    prompts, _ = llm.calls[0]
    expected = list(
        tok.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ).encode()
    )
    assert prompts[0] == {"prompt_token_ids": expected}
    assert tok.template_calls[-1]["enable_thinking"] is False


def test_token_counts_come_from_the_engine_ids_not_from_words():
    llm = FakeLLM(lambda ids, sp: FakeCompletion("x y z", [1, 2, 3, 4, 5], "stop"))
    [res] = make(llm).generate([req("abc")])
    rendered = FakeTokenizer().apply_chat_template(
        [{"role": "user", "content": "abc"}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert res.prompt_tokens == len(rendered.encode())
    assert res.completion_tokens == 5
    assert res.text == "x y z"
    assert res.finish_reason == "stop"


def test_length_finish_reason_is_reported_as_length():
    llm = FakeLLM(lambda ids, sp: FakeCompletion("cut", [1] * 32, "length"))
    assert make(llm).generate([req()])[0].finish_reason == "length"


def test_results_follow_request_order():
    llm = FakeLLM(
        lambda ids, sp: FakeCompletion(f"seed{sp.seed}", [1, 2], "stop"),
    )
    out = make(llm).generate([req(seed=s) for s in (3, 1, 2)])
    assert [r.text for r in out] == ["seed3", "seed1", "seed2"]


def test_meta_is_ignored_by_the_real_backend():
    llm = FakeLLM()
    make(llm).generate([req(meta={"arm": "placebo", "problem_id": "x"})])
    assert len(llm.calls[0][0]) == 1


def test_over_long_prompt_is_context_overflow_and_skips_the_engine():
    llm = FakeLLM()
    big = "x" * 500  # > max_model_len tokens in the byte tokenizer
    out = make(llm).generate([req(big), req("ok")])
    assert out[0].finish_reason == "context_overflow"
    assert out[0].completion_tokens == 0 and out[0].text == ""
    assert out[0].prompt_tokens > CFG.max_model_len
    assert out[1].finish_reason == "stop"
    assert len(llm.calls[0][0]) == 1  # only the fitting request reached the engine


def test_an_empty_batch_does_not_call_the_engine():
    llm = FakeLLM()
    assert make(llm).generate([]) == []
    assert llm.calls == []


def test_engine_errors_propagate():
    class Boom(FakeLLM):
        def generate(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        make(Boom()).generate([req()])


def test_a_wrong_number_of_outputs_is_an_error():
    class Short(FakeLLM):
        def generate(self, prompts, sampling_params, use_tqdm=True):
            return super().generate(prompts[:-1], sampling_params[:-1], use_tqdm)

    with pytest.raises(RuntimeError, match="outputs"):
        make(Short()).generate([req(), req()])


def test_from_config_explains_how_to_install_vllm(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", None)
    with pytest.raises(ImportError, match=r"mas-channel-audit\[vllm\]"):
        VllmBackend.from_config(CFG)
