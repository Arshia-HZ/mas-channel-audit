"""Typed records. Every generation is traceable to its prompt, model and seed."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from channel_audit.seeds import IDENTITY_FIELDS, derive_seed, gen_key


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Problem(_Record):
    problem_id: str
    dataset: str
    subject: str
    level: int = Field(ge=1, le=5)
    question: str
    gold: str
    split: Literal["screen", "pilot", "confirm", "check"]


class Generation(_Record):
    gen_key: str  # sha256 of the identity fields (see seeds.IDENTITY_FIELDS)
    problem_id: str
    replicate: int
    agent: int
    model_id: str
    arm: str  # round0, sc_pool, failure_msg, or an arm
    round: int
    sample_idx: int
    prompt_version: str
    prompt_hash: str
    seed: int  # derived from the same identity fields
    temperature: float
    max_tokens: int
    backend: str
    backend_version: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    finish_reason: str
    text: str
    parent_gen_keys: list[str]
    created_at: str

    @property
    def identity(self) -> dict:
        return {name: getattr(self, name) for name in IDENTITY_FIELDS}

    @model_validator(mode="after")
    def _key_and_seed_match_identity(self) -> Self:
        if self.gen_key != gen_key(**self.identity):
            raise ValueError("gen_key does not match the identity fields")
        if self.seed != derive_seed(**self.identity):
            raise ValueError("seed does not match the identity fields")
        return self

    @classmethod
    def create(cls, **fields) -> Self:
        """Build a record, deriving gen_key and seed from the identity fields."""
        identity = {name: fields[name] for name in IDENTITY_FIELDS}
        return cls(**fields, gen_key=gen_key(**identity), seed=derive_seed(**identity))


class Graded(_Record):
    gen_key: str
    extracted: str | None
    extraction_status: Literal["boxed", "none", "truncated"]
    confidence: int | None = Field(default=None, ge=0, le=100)
    correct: bool


class PromptRecord(_Record):
    """The chat messages sent for one generation (logged when requested)."""

    gen_key: str
    messages: list[dict[str, str]]
