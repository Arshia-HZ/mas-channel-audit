"""gen_key and seed are pure functions of the identity fields."""

from __future__ import annotations

import hashlib
import json

import pytest

from channel_audit.seeds import IDENTITY_FIELDS, derive_seed, gen_key

IDENTITY = {
    "problem_id": "math-0001",
    "replicate": 0,
    "agent": 1,
    "model_id": "mock/model",
    "arm": "round0",
    "round": 0,
    "sample_idx": 0,
    "prompt_version": "v1",
    "prompt_hash": "ab" * 32,
    "temperature": 0.7,
    "max_tokens": 1024,
}

CHANGED = {
    "problem_id": "math-0002",
    "replicate": 1,
    "agent": 2,
    "model_id": "other/model",
    "arm": "answer_only",
    "round": 1,
    "sample_idx": 1,
    "prompt_version": "v2",
    "prompt_hash": "cd" * 32,
    "temperature": 0.8,
    "max_tokens": 2048,
}


def test_identity_fields_match_the_documented_set():
    assert set(IDENTITY_FIELDS) == set(IDENTITY)


def test_same_identity_same_key_and_seed():
    assert gen_key(**IDENTITY) == gen_key(**dict(reversed(IDENTITY.items())))
    assert derive_seed(**IDENTITY) == derive_seed(**IDENTITY)


def test_key_matches_documented_sha256_of_canonical_json():
    canonical = json.dumps(
        [IDENTITY[f] for f in IDENTITY_FIELDS], separators=(",", ":")
    )
    expected = hashlib.sha256(("gen_key|" + canonical).encode()).hexdigest()
    assert gen_key(**IDENTITY) == expected


@pytest.mark.parametrize("field", sorted(CHANGED))
def test_any_changed_field_changes_key_and_seed(field):
    other = {**IDENTITY, field: CHANGED[field]}
    assert gen_key(**other) != gen_key(**IDENTITY)
    assert derive_seed(**other) != derive_seed(**IDENTITY)


def test_int_and_float_temperature_are_the_same_identity():
    assert gen_key(**{**IDENTITY, "temperature": 1}) == gen_key(
        **{**IDENTITY, "temperature": 1.0}
    )


def test_seed_is_a_non_negative_int32():
    seeds = {derive_seed(**{**IDENTITY, "sample_idx": i}) for i in range(200)}
    assert len(seeds) > 190  # no collapse onto a few values
    assert all(0 <= s < 2**31 - 1 for s in seeds)


def test_seed_does_not_depend_on_call_order():
    a = derive_seed(**{**IDENTITY, "agent": 0})
    derive_seed(**{**IDENTITY, "agent": 1})
    assert derive_seed(**{**IDENTITY, "agent": 0}) == a


def test_missing_or_extra_identity_field_is_an_error():
    missing = dict(IDENTITY)
    del missing["arm"]
    with pytest.raises(ValueError):
        gen_key(**missing)
    with pytest.raises(ValueError):
        derive_seed(**{**IDENTITY, "text": "not identity"})
