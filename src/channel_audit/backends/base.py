"""Backend protocol: one batched ``generate`` call with exact token accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenRequest:
    messages: list[dict[str, str]]  # chat messages; the backend applies the template
    seed: int
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    # Bookkeeping the runner attaches (problem_id, arm, ...). Real backends must
    # ignore it; only the mock reads it, to plant arm effects.
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length" | backend-specific


class Backend(Protocol):
    name: str
    version: str

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]:
        """Results in request order. Failures raise; they are never swallowed."""
        ...
