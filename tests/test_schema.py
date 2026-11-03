"""Typed records reject malformed or tampered data."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from channel_audit.schema import Generation, Graded, Problem
from channel_audit.seeds import derive_seed, gen_key
from conftest import BASE_IDENTITY, make_generation


def test_create_fills_key_and_seed_from_identity():
    g = make_generation()
    assert g.gen_key == gen_key(**BASE_IDENTITY)
    assert g.seed == derive_seed(**BASE_IDENTITY)


def test_generation_round_trips_through_json():
    g = make_generation(parent_gen_keys=["k1", "k2"], text="hi \\boxed{3}")
    assert Generation.model_validate_json(g.model_dump_json()) == g


def test_tampered_gen_key_is_rejected():
    data = make_generation().model_dump()
    data["gen_key"] = "0" * 64
    with pytest.raises(ValidationError):
        Generation.model_validate(data)


def test_tampered_seed_is_rejected():
    data = make_generation().model_dump()
    data["seed"] += 1
    with pytest.raises(ValidationError):
        Generation.model_validate(data)


def test_changing_an_identity_field_without_rekeying_is_rejected():
    data = make_generation().model_dump()
    data["agent"] = 2
    with pytest.raises(ValidationError):
        Generation.model_validate(data)


def test_non_identity_fields_do_not_change_the_key():
    a = make_generation(text="one", completion_tokens=1, backend="vllm")
    b = make_generation(text="two", completion_tokens=2, backend="hf")
    assert a.gen_key == b.gen_key


def test_unknown_fields_are_rejected():
    data = make_generation().model_dump()
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        Generation.model_validate(data)


def test_generation_is_immutable():
    g = make_generation()
    with pytest.raises(ValidationError):
        g.text = "changed"


def test_negative_token_counts_are_rejected():
    with pytest.raises(ValidationError):
        make_generation(completion_tokens=-1)


def test_problem_validates_split_and_level():
    ok = dict(
        problem_id="p",
        dataset="math",
        subject="Algebra",
        level=3,
        question="q",
        gold="1",
        split="screen",
    )
    assert Problem(**ok).split == "screen"
    with pytest.raises(ValidationError):
        Problem(**{**ok, "split": "test"})
    with pytest.raises(ValidationError):
        Problem(**{**ok, "level": 6})


def test_graded_validates_status_and_confidence_range():
    ok = dict(gen_key="k", extracted="1", extraction_status="boxed", correct=True)
    assert Graded(**ok).confidence is None
    assert Graded(**ok, confidence=100).confidence == 100
    with pytest.raises(ValidationError):
        Graded(**{**ok, "extraction_status": "garbled"})
    with pytest.raises(ValidationError):
        Graded(**ok, confidence=101)
    with pytest.raises(ValidationError):
        Graded(**ok, confidence=-1)
