"""Deterministic identity: gen_key and seed are pure functions of identity fields.

Identity is what makes a generation *that* generation: problem, replicate,
agent, model, arm, round, sample index, the prompt version and hash, and the
decoding settings. Content (text, token counts, backend, timestamps) is not
identity, so a restarted job on the same plan skips finished work.
"""

from __future__ import annotations

import hashlib
import json

IDENTITY_FIELDS: tuple[str, ...] = (
    "problem_id",
    "replicate",
    "agent",
    "model_id",
    "arm",
    "round",
    "sample_idx",
    "prompt_version",
    "prompt_hash",
    "temperature",
    "max_tokens",
)

SEED_MODULUS = 2**31 - 1


def _canonical(fields: dict) -> str:
    given, expected = set(fields), set(IDENTITY_FIELDS)
    if given != expected:
        raise ValueError(
            f"identity fields mismatch: missing={sorted(expected - given)} "
            f"extra={sorted(given - expected)}"
        )
    values = [fields[name] for name in IDENTITY_FIELDS]
    values[IDENTITY_FIELDS.index("temperature")] = float(fields["temperature"])
    return json.dumps(values, separators=(",", ":"))


def _digest(domain: str, fields: dict) -> bytes:
    return hashlib.sha256(f"{domain}|{_canonical(fields)}".encode()).digest()


def gen_key(**fields) -> str:
    """sha256 hex digest of the identity fields."""
    return hashlib.sha256(f"gen_key|{_canonical(fields)}".encode()).hexdigest()


def derive_seed(**fields) -> int:
    """Per-request sampling seed, derived from identity, never a global counter."""
    return int.from_bytes(_digest("seed", fields)[:8], "big") % SEED_MODULUS
