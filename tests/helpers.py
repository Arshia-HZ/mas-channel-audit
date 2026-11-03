"""Test helpers for the mock pipeline. No model, no network."""

from __future__ import annotations

from collections.abc import Sequence

from channel_audit.backends.base import GenRequest, GenResult
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.runner import RunConfig, RunContext, create_run


class RecordingBackend:
    """Wraps a backend and keeps every request, so tests can read real prompts."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.version = inner.version
        self.requests: list[GenRequest] = []

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]:
        self.requests.extend(requests)
        return self.inner.generate(requests)

    def by_arm(self, arm: str) -> list[GenRequest]:
        return [r for r in self.requests if r.meta["arm"] == arm]


def build_ctx(
    tmp_path,
    *,
    n_problems: int = 12,
    replicates: int = 2,
    effects=None,
    leak_rate: float = 0.0,
    leak_forms=("boxed", "answer_line", "claim"),
    base_p: float | None = None,
    max_tokens: int = 512,
    context_limit: int = 100_000,
    log_prompts: bool = False,
) -> RunContext:
    problems = fake_problems(n_problems)
    mock = MockBackend(
        MockWorld.from_problems(problems, base_p=base_p),
        effects=effects,
        leak_rate=leak_rate,
        leak_forms=leak_forms,
    )
    cfg = RunConfig(
        model_id="mock/model",
        replicates=replicates,
        max_tokens=max_tokens,
        context_limit=context_limit,
        log_prompts=log_prompts,
    )
    return create_run(tmp_path / "run", cfg, problems, RecordingBackend(mock))
