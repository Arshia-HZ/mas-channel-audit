"""Shared test helpers. No model, no network."""

from __future__ import annotations

import pytest

from channel_audit.schema import Generation

BASE_IDENTITY = {
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


def make_generation(**overrides) -> Generation:
    """Build a valid Generation; any identity field can be overridden."""
    fields = {**BASE_IDENTITY, **{k: v for k, v in overrides.items()}}
    identity_keys = set(BASE_IDENTITY)
    identity = {k: fields.pop(k) for k in list(fields) if k in identity_keys}
    defaults = {
        "backend": "mock",
        "backend_version": "0",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "finish_reason": "stop",
        "text": "x",
        "parent_gen_keys": [],
        "created_at": "2026-01-01T00:00:00Z",
    }
    return Generation.create(**identity, **{**defaults, **fields})


@pytest.fixture
def gen():
    return make_generation
