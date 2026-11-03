"""HF backend: left padding, token-id stop detection, exact counts."""

from __future__ import annotations

import sys

import pytest

from channel_audit.backends.base import GenRequest
from channel_audit.backends.hf_engine import (
    HfBackend,
    batch_seed,
    left_pad,
    plan_batches,
    split_completion,
)
from channel_audit.models import ModelConfig
from fakes import FakeTokenizer

EOS, PAD = 1, 0  # both are special ids in the fake tokenizer


def cfg(batch_size=4, max_model_len=200):
    return ModelConfig(
        model_id="org/model",
        backend="hf",
        dtype="float16",
        revision="c" * 40,
        max_tokens=10,
        max_model_len=max_model_len,
        batch_size=batch_size,
        stop_token_ids=[EOS],
    )


class Recorder:
    """A fake ``generate_fn``: returns scripted tokens padded to max_new_tokens."""

    def __init__(self, script=None):
        self.calls: list[dict] = []
        self.script = script or (lambda row: [65, 66, EOS])

    def __call__(self, input_ids, attention_mask, **kw):
        self.calls.append({"input_ids": input_ids, "mask": attention_mask, **kw})
        n = kw["max_new_tokens"]
        out = []
        for row, mask in zip(input_ids, attention_mask, strict=True):
            real = [t for t, m in zip(row, mask, strict=True) if m]
            gen = self.script(real)[:n]
            out.append(gen + [kw["pad_token_id"]] * (n - len(gen)))
        return out


def make(rec=None, batch_size=4, **kw):
    return HfBackend(
        cfg(batch_size, **kw),
        tokenizer=FakeTokenizer(),
        generate_fn=rec or Recorder(),
        version="hf-test",
        pad_token_id=PAD,
        eos_token_ids=[EOS],
    )


def req(content="x", seed=1, max_tokens=10, temperature=0.7, top_p=1.0):
    return GenRequest(
        messages=[{"role": "user", "content": content}],
        seed=seed,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        meta={"arm": "round0"},
    )


def test_left_pad_puts_padding_on_the_left_and_masks_it():
    ids, mask = left_pad([[5, 6, 7], [9]], pad_id=0)
    assert ids == [[5, 6, 7], [0, 0, 9]]
    assert mask == [[1, 1, 1], [0, 0, 1]]


def test_left_pad_of_nothing_is_empty():
    assert left_pad([], pad_id=0) == ([], [])


def test_split_completion_stops_at_the_first_eos_and_keeps_it_in_the_count():
    toks, reason = split_completion([65, 66, EOS, 9, 9], {EOS}, max_new_tokens=5)
    assert (toks, reason) == ([65, 66, EOS], "stop")


def test_split_completion_without_eos_is_length():
    toks, reason = split_completion([65, 66, 67], {EOS}, max_new_tokens=3)
    assert (toks, reason) == ([65, 66, 67], "length")


def test_split_completion_eos_as_the_last_allowed_token_is_stop_not_length():
    assert split_completion([65, EOS], {EOS}, max_new_tokens=2)[1] == "stop"


def test_split_completion_handles_pad_equal_to_eos():
    toks, reason = split_completion([EOS, EOS, EOS], {EOS}, max_new_tokens=3)
    assert (toks, reason) == ([EOS], "stop")


def test_plan_batches_cover_every_index_once_in_length_order():
    lengths = [30, 5, 20, 5, 40, 10, 15]
    batches = plan_batches(lengths, batch_size=3)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(7))
    assert all(len(b) <= 3 for b in batches)
    ordered = [lengths[i] for i in flat]
    assert ordered == sorted(ordered)


def test_batch_seed_depends_on_the_set_of_seeds_not_their_order():
    assert batch_seed([3, 1, 2]) == batch_seed([2, 3, 1])
    assert batch_seed([1, 2]) != batch_seed([1, 3])
    assert 0 <= batch_seed([1, 2]) < 2**31 - 1


def test_results_come_back_in_request_order_with_exact_counts():
    def script(row):
        return [65] * (len(row) % 4 + 1) + [EOS]

    rec = Recorder(script)
    reqs = [req("a" * k, seed=k) for k in (9, 2, 5, 7)]
    out = make(rec, batch_size=2).generate(reqs)
    for r, res in zip(reqs, out, strict=True):
        ids = list(
            FakeTokenizer()
            .apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True)
            .encode()
        )
        assert res.prompt_tokens == len(ids)
        assert res.finish_reason == "stop"
        assert res.completion_tokens == len(script(ids))
        assert res.text == "A" * (len(ids) % 4 + 1)  # EOS is special-skipped


def test_batches_are_left_padded_and_masked_when_lengths_differ():
    rec = Recorder()
    make(rec, batch_size=4).generate([req("a"), req("a" * 30)])
    call = rec.calls[0]
    assert len({len(r) for r in call["input_ids"]}) == 1
    padded = [i for i, r in enumerate(call["input_ids"]) if r[0] == PAD]
    assert len(padded) == 1
    assert call["mask"][padded[0]][0] == 0 and call["mask"][padded[0]][-1] == 1


def test_length_finish_when_the_model_never_emits_eos():
    rec = Recorder(lambda row: [65] * 50)
    [res] = make(rec).generate([req(max_tokens=4)])
    assert (res.finish_reason, res.completion_tokens) == ("length", 4)


def test_batch_size_one_uses_each_requests_own_seed():
    rec = Recorder()
    make(rec, batch_size=1).generate([req(seed=21), req("y", seed=22)])
    assert sorted(c["seed"] for c in rec.calls) == [21, 22]
    assert all(len(c["input_ids"]) == 1 for c in rec.calls)


def test_larger_batches_use_a_derived_seed_that_ignores_request_order():
    a, b = Recorder(), Recorder()
    make(a, batch_size=4).generate([req("x", seed=1), req("y", seed=2)])
    make(b, batch_size=4).generate([req("y", seed=2), req("x", seed=1)])
    assert a.calls[0]["seed"] == b.calls[0]["seed"] == batch_seed([1, 2])


def test_decoding_settings_reach_generate_and_split_batches():
    rec = Recorder()
    make(rec, batch_size=8).generate(
        [req("a", max_tokens=5), req("b", max_tokens=6, temperature=0.2)]
    )
    assert len(rec.calls) == 2  # different settings cannot share a batch
    got = {(c["max_new_tokens"], c["temperature"]) for c in rec.calls}
    assert got == {(5, 0.7), (6, 0.2)}
    assert all(c["top_p"] == 1.0 and c["eos_token_ids"] == [EOS] for c in rec.calls)


def test_over_long_prompt_is_context_overflow_and_skips_the_engine():
    rec = Recorder()
    out = make(rec).generate([req("x" * 500), req("ok")])
    assert out[0].finish_reason == "context_overflow" and out[0].completion_tokens == 0
    assert out[1].finish_reason == "stop"
    assert sum(len(c["input_ids"]) for c in rec.calls) == 1


def test_engine_errors_propagate():
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        make(boom).generate([req()])


def test_identity_and_empty_batch():
    b = make()
    assert (b.name, b.version) == ("hf", "hf-test")
    assert b.generate([]) == []


def test_from_config_explains_how_to_install_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"mas-channel-audit\[hf\]"):
        HfBackend.from_config(cfg())
