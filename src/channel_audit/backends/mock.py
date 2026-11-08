"""Deterministic mock backend with planted arm effects. No GPU, no network.

A request's outcome depends only on its seed, its problem's difficulty and the
planted effect of its arm, so a pipeline run on the mock has a known answer.
Every produced text carries a ``[trace <seed>]`` marker so tests can tell which
solution ended up in which prompt.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from channel_audit.backends.base import GenRequest, GenResult
from channel_audit.extract import extract_boxed
from channel_audit.schema import Problem

# Added to a problem's base probability of a correct answer, by arm.
PLANTED_EFFECTS: dict[str, float] = {
    "self_reflect": 0.0,
    "answer_only": 0.05,
    "answer_conf": 0.08,
    "dissent_only": 0.02,
    "failure_cause": 0.10,
    "full_solution": 0.03,
    "placebo": -0.04,
}

LEAK_FORMS = ("boxed", "answer_line", "claim")


def fake_problems(n: int, seed: int = 0) -> list[Problem]:
    """Integer-answer problems in 4 (subject, level) groups, cycled."""
    rng = random.Random(seed)
    problems = []
    for i in range(n):
        a, b = rng.randrange(10, 90), rng.randrange(10, 90)
        problems.append(
            Problem(
                problem_id=f"fake-{i:04d}",
                dataset="fake",
                subject=("Algebra", "Geometry")[i % 2],
                level=3 + (i // 2) % 2,
                question=f"Compute {a} + {b}. (fake problem {i})",
                gold=str(a + b),
                split="pilot",
            )
        )
    return problems


@dataclass(frozen=True)
class MockWorld:
    gold: Mapping[str, str]
    base_p: Mapping[str, float]

    @classmethod
    def from_problems(
        cls, problems: Sequence[Problem], base_p: float | None = None
    ) -> MockWorld:
        """Difficulty from a hash of the id (0.30..0.60) unless pinned by base_p."""

        def p_of(pid: str) -> float:
            if base_p is not None:
                return base_p
            h = int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)
            return 0.30 + 0.30 * (h / 0xFFFFFFFF)

        return cls(
            gold={p.problem_id: p.gold for p in problems},
            base_p={p.problem_id: p_of(p.problem_id) for p in problems},
        )


_TOKEN = re.compile(r"\S+\s*")


class MockBackend:
    name = "mock"
    version = "1"

    def __init__(
        self,
        world: MockWorld,
        effects: Mapping[str, float] | None = None,
        leak_rate: float = 0.0,
        leak_forms: Sequence[str] = LEAK_FORMS,
    ) -> None:
        self.world = world
        self.effects = dict(PLANTED_EFFECTS if effects is None else effects)
        self.leak_rate = leak_rate
        self.leak_forms = tuple(leak_forms)

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]:
        return [self._one(r) for r in requests]

    def _one(self, req: GenRequest) -> GenResult:
        rng = random.Random(req.seed)
        arm = req.meta["arm"]
        if arm in ("failure_msg", "failure_msg_v2"):
            text = self._failure_message(req, rng)
        else:
            text = self._solution(req, rng, arm)
        tokens = _TOKEN.findall(text)
        finish = "stop"
        if len(tokens) > req.max_tokens:
            tokens, finish = tokens[: req.max_tokens], "length"
        prompt_tokens = sum(len(m["content"].split()) for m in req.messages)
        return GenResult(
            text="".join(tokens),
            prompt_tokens=prompt_tokens,
            completion_tokens=len(tokens),
            finish_reason=finish,
        )

    def _solution(self, req: GenRequest, rng: random.Random, arm: str) -> str:
        pid = req.meta["problem_id"]
        gold_text = self.world.gold[pid]
        p = min(1.0, max(0.0, self.world.base_p[pid] + self.effects.get(arm, 0.0)))
        if rng.random() < p:
            answer, confidence = gold_text, rng.randint(60, 95)
        else:
            confidence = rng.randint(20, 70)
            try:
                answer = int(gold_text) + rng.choice([-1, 1, 2, 3])
            except ValueError:  # non-integer gold: any different string is wrong
                answer = f"{gold_text}?{rng.randint(1, 3)}"
        steps = "\n".join(
            f"Step {k + 1}: apply the rule here." for k in range(rng.randint(8, 14))
        )
        return (
            f"{steps}\n[trace {req.seed}]\nSo the result is "
            f"\\boxed{{{answer}}}\nConfidence: {confidence}"
        )

    def _failure_message(self, req: GenRequest, rng: random.Random) -> str:
        step = rng.randint(1, 4)
        text = (
            f"[trace {req.seed}] The first step that may be wrong is step "
            f"{step} because the working looks inconsistent."
        )
        reviewed = extract_boxed(req.messages[-1]["content"])
        if reviewed is not None and rng.random() < self.leak_rate:
            form = rng.choice(self.leak_forms)
            if form == "boxed":
                text += f" The correct result is \\boxed{{{reviewed}}}."
            elif form == "answer_line":
                text += f" So the answer is {reviewed}."
            else:
                text += f" Recomputing this gives x = {reviewed}."
        return text
