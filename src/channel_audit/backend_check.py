"""Tools for the M3 backend check: rerun comparison and the hand-review sample."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from channel_audit.channels import ARMS
from channel_audit.runner import RunContext
from channel_audit.schema import Generation
from channel_audit.validate import TRUNCATION_GATE

KINDS = ("round0", "sc_pool", "failure_msg", *ARMS)
MAX_DIVERGENCES = 20


@dataclass
class ReproReport:
    n_a: int
    n_b: int
    only_in_a: list[str]
    only_in_b: list[str]
    n_common: int
    n_identical: int
    by_arm: dict[str, tuple[int, int]]  # arm -> (common, identical)
    divergences: list[dict] = field(default_factory=list)

    @property
    def n_different(self) -> int:
        return self.n_common - self.n_identical

    @property
    def match_rate(self) -> float:
        return self.n_identical / self.n_common if self.n_common else 1.0

    @property
    def perfect(self) -> bool:
        return not (self.only_in_a or self.only_in_b or self.n_different)

    def format(self) -> str:
        lines = [
            f"generations: A={self.n_a} B={self.n_b} common={self.n_common}",
            f"identical text, token counts and finish reason: {self.n_identical}"
            f" / {self.n_common}",
            f"match rate: {self.match_rate:.1%}",
        ]
        if self.only_in_a or self.only_in_b:
            lines.append(
                f"only in A: {len(self.only_in_a)}; only in B: {len(self.only_in_b)}"
            )
        for arm, (common, same) in sorted(self.by_arm.items()):
            lines.append(f"  {arm:<14}{same:>5} / {common}")
        for d in self.divergences:
            lines.append(
                f"  differs {d['arm']} {d['gen_key'][:12]} at char {d['char_index']}"
                f" (tokens {d['tokens_a']} vs {d['tokens_b']})"
            )
        return "\n".join(lines)


def _first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def compare_runs(a: RunContext, b: RunContext) -> ReproReport:
    """Compare two runs of the same plan generation by generation (by gen_key)."""
    in_a = {g.gen_key: g for g in a.gens}
    in_b = {g.gen_key: g for g in b.gens}
    common = [k for k in in_a if k in in_b]
    by_arm: dict[str, list[int]] = {}
    divergences = []
    identical = 0
    for key in common:
        ga, gb = in_a[key], in_b[key]
        same = (
            ga.text == gb.text
            and ga.completion_tokens == gb.completion_tokens
            and ga.finish_reason == gb.finish_reason
        )
        counts = by_arm.setdefault(ga.arm, [0, 0])
        counts[0] += 1
        counts[1] += same
        identical += same
        if not same and len(divergences) < MAX_DIVERGENCES:
            divergences.append(
                {
                    "gen_key": key,
                    "arm": ga.arm,
                    "char_index": _first_difference(ga.text, gb.text),
                    "tokens_a": ga.completion_tokens,
                    "tokens_b": gb.completion_tokens,
                }
            )
    return ReproReport(
        n_a=len(in_a),
        n_b=len(in_b),
        only_in_a=[k for k in in_a if k not in in_b],
        only_in_b=[k for k in in_b if k not in in_a],
        n_common=len(common),
        n_identical=identical,
        by_arm={arm: (c[0], c[1]) for arm, c in by_arm.items()},
        divergences=divergences,
    )


def select_review_sample(
    ctx: RunContext, n: int = 30, seed: int = 0
) -> list[Generation]:
    """``n`` generations spread evenly over every kind (round 0, pool, failure
    messages and each arm). Short kinds are made up from the others."""

    def rank(g: Generation) -> str:
        return hashlib.sha256(f"{seed}|{g.gen_key}".encode()).hexdigest()

    pools = {k: sorted((g for g in ctx.gens if g.arm == k), key=rank) for k in KINDS}
    base = n // len(KINDS)
    taken = {k: pools[k][:base] for k in KINDS}
    left = {k: pools[k][base:] for k in KINDS}
    need = n - sum(len(v) for v in taken.values())
    while need > 0 and any(left.values()):
        for k in KINDS:
            if need > 0 and left[k]:
                taken[k].append(left[k].pop(0))
                need -= 1
    return [g for k in KINDS for g in taken[k]]


def format_review(ctx: RunContext, gens: list[Generation]) -> str:
    """Markdown for reading: problem, the prompt as sent, the output, the grade."""
    problems = {p.problem_id: p for p in ctx.problems}
    out = [f"# Review sample: {len(gens)} outputs from {ctx.out_dir}", ""]
    for i, g in enumerate(gens, start=1):
        p = problems[g.problem_id]
        out += [
            f"## {i}. {g.arm} | {g.problem_id} | agent {g.agent} | "
            f"replicate {g.replicate} | {g.gen_key[:12]}",
            "",
            f"finish: {g.finish_reason} | prompt tokens: {g.prompt_tokens} | "
            f"completion tokens: {g.completion_tokens}",
        ]
        if g.gen_key in ctx.graded:
            r = ctx.graded.get(g.gen_key)
            out.append(
                f"graded: extracted={r.extracted!r} status={r.extraction_status} "
                f"correct: {r.correct}"
            )
        out += ["", f"**Problem** (gold: {p.gold})", "", p.question, ""]
        out += ["### Prompt", ""]
        if ctx.prompts is not None and g.gen_key in ctx.prompts:
            for m in ctx.prompts.get(g.gen_key).messages:
                out += [f"[{m['role']}]", "~~~~", m["content"], "~~~~"]
        else:
            out.append("*(prompt not logged)*")
        out += ["", "### Output", "", "~~~~", g.text, "~~~~", ""]
    return "\n".join(out)


def round0_summary(ctx: RunContext) -> dict:
    """Truncation and accuracy of the round-0 generations stored so far."""
    r0 = [g for g in ctx.gens if g.arm == "round0"]
    if not r0:
        raise ValueError("no round-0 generations stored")
    graded = {r.gen_key: r for r in ctx.graded}
    cut = [g for g in r0 if g.finish_reason == "length"]
    done = [g for g in r0 if g.finish_reason != "length"]

    def accuracy(gens: list[Generation]) -> float | None:
        marks = [graded[g.gen_key].correct for g in gens if g.gen_key in graded]
        return sum(marks) / len(marks) if marks else None

    tokens = sorted(g.completion_tokens for g in r0)

    def pct(q: float) -> int:
        return tokens[min(len(tokens) - 1, int(len(tokens) * q))]

    rate = len(cut) / len(r0)
    return {
        "n": len(r0),
        "max_tokens": r0[0].max_tokens,
        "truncated": len(cut),
        "truncation_rate": rate,
        "gate_g3": rate < TRUNCATION_GATE,
        "accuracy": accuracy(r0),
        "accuracy_finished": accuracy(done),
        "accuracy_truncated": accuracy(cut),
        "tokens_p50": pct(0.5),
        "tokens_p90": pct(0.9),
        "tokens_max": tokens[-1],
        "n_problems_with_truncation": len({g.problem_id for g in cut}),
    }


def format_round0_summary(s: dict) -> str:
    def pc(x):
        return "n/a" if x is None else f"{x:.1%}"

    return "\n".join(
        [
            f"round 0 at max_tokens={s['max_tokens']}: {s['n']} generations",
            f"truncated: {s['truncated']} ({s['truncation_rate']:.1%}) "
            f"on {s['n_problems_with_truncation']} problems",
            f"gate G3 (truncation < {TRUNCATION_GATE:.0%}): "
            f"{'PASS' if s['gate_g3'] else 'FAIL'}",
            f"accuracy: {pc(s['accuracy'])} overall, {pc(s['accuracy_finished'])} "
            f"finished, {pc(s['accuracy_truncated'])} truncated",
            f"completion tokens: median {s['tokens_p50']}, p90 {s['tokens_p90']}, "
            f"max {s['tokens_max']}",
        ]
    )
