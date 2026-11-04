"""M4 screening: sample a model 4 times per problem and summarise what it does.

Screening uses the round-0 prompt, the study decoding (temperature 0.7) and the
model's ``max_tokens``. It never looks at an arm outcome. The report gives, for
the screen split, accuracy by level, disagreement among the samples,
truncation and extraction failure, and gate G0 for the sanity model.
"""

from __future__ import annotations

from dataclasses import dataclass

from channel_audit import channels
from channel_audit.aggregate import cluster_answers
from channel_audit.data import STUDY_LEVELS
from channel_audit.runner import Planned, RunContext, _execute, identity_for
from channel_audit.schema import Generation

SCREEN_ARM = "screen"
N_SAMPLES = 4
CHUNK_PROBLEMS = 50  # problems per backend call, so progress is saved as we go

G0_TOLERANCE_POINTS = 5.0
G0_MAX_EXTRACTION_FAILURE = 0.05

# MATH accuracy (percent) published by the model's authors, for the G0 check.
# Qwen2.5-1.5B-Instruct: "MATH 55.2" in the Qwen2.5 release post
# (https://qwenlm.github.io/blog/qwen2.5-llm/).
PUBLISHED_MATH: dict[str, float] = {"Qwen/Qwen2.5-1.5B-Instruct": 55.2}


def run_screen(
    ctx: RunContext, n_samples: int = N_SAMPLES, chunk: int = CHUNK_PROBLEMS
) -> list[Generation]:
    """Generate the missing screening samples, ``chunk`` problems at a time."""
    out: list[Generation] = []
    for start in range(0, len(ctx.problems), chunk):
        planned = []
        for p in ctx.problems[start : start + chunk]:
            messages = channels.build_round0_prompt(p.question)
            for k in range(n_samples):
                ident = identity_for(ctx.cfg, p.problem_id, 0, 0, SCREEN_ARM, 0, k)
                planned.append(Planned(ident, messages, []))
        out.extend(_execute(ctx, planned))
    return out


@dataclass(frozen=True)
class ProblemStats:
    disagrees: bool  # two or more distinct parsable answers
    mixed: bool  # at least one correct and at least one incorrect sample
    all_wrong: bool
    all_correct: bool


def problem_stats(answers: list[str | None], correct: list[bool]) -> ProblemStats:
    _, reps = cluster_answers(answers)
    return ProblemStats(
        disagrees=len(reps) >= 2,
        mixed=any(correct) and not all(correct),
        all_wrong=not any(correct),
        all_correct=all(correct),
    )


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _summary(stats: list[ProblemStats], correct: list[bool]) -> dict:
    return {
        "n_problems": len(stats),
        "accuracy": _rate(correct),
        "disagreement_rate": _rate([s.disagrees for s in stats]),
        "mixed_rate": _rate([s.mixed for s in stats]),
        "all_correct_rate": _rate([s.all_correct for s in stats]),
        "all_wrong_rate": _rate([s.all_wrong for s in stats]),
    }


def screen_report(ctx: RunContext) -> dict:
    gens = [g for g in ctx.gens if g.arm == SCREEN_ARM]
    if not gens:
        raise ValueError("no screen generations stored; run the screening first")
    missing = [g.gen_key for g in gens if g.gen_key not in ctx.graded]
    if missing:
        raise ValueError(f"{len(missing)} screen generations are not graded yet")

    level = {p.problem_id: p.level for p in ctx.problems}
    by_problem: dict[str, list[Generation]] = {}
    for g in gens:
        by_problem.setdefault(g.problem_id, []).append(g)

    per_problem = {}
    for pid, members in by_problem.items():
        graded = [ctx.graded.get(g.gen_key) for g in members]
        per_problem[pid] = (
            problem_stats([r.extracted for r in graded], [r.correct for r in graded]),
            [r.correct for r in graded],
        )

    def summarize(pids: list[str]) -> dict:
        return _summary(
            [per_problem[p][0] for p in pids],
            [c for p in pids for c in per_problem[p][1]],
        )

    pids = list(per_problem)
    graded_all = [ctx.graded.get(g.gen_key) for g in gens]
    n = len(gens)
    return {
        "model_id": ctx.cfg.model_id,
        "backend": gens[0].backend,
        "backend_version": gens[0].backend_version,
        "temperature": ctx.cfg.temperature,
        "max_tokens": ctx.cfg.max_tokens,
        "n_samples": n,
        **summarize(pids),
        "by_level": {
            lv: summarize([p for p in pids if level[p] == lv])
            for lv in sorted({level[p] for p in pids})
        },
        "study_levels": summarize([p for p in pids if level[p] in STUDY_LEVELS]),
        "truncation_rate": sum(g.finish_reason == "length" for g in gens) / n,
        "overflow_count": sum(g.finish_reason == "context_overflow" for g in gens),
        "extraction_failure_rate": sum(
            r.extraction_status == "none" for r in graded_all
        )
        / n,
        "no_answer_rate": sum(r.extracted is None for r in graded_all) / n,
    }


def g0_check(
    accuracy: float,
    extraction_failure_rate: float,
    published: float,
    tolerance_points: float = G0_TOLERANCE_POINTS,
) -> dict:
    """Gate G0: accuracy near the published level and extraction failure < 5%."""
    diff = accuracy * 100 - published
    accuracy_ok = abs(diff) <= tolerance_points
    extraction_ok = extraction_failure_rate < G0_MAX_EXTRACTION_FAILURE
    return {
        "published": published,
        "difference_points": diff,
        "tolerance_points": tolerance_points,
        "accuracy_ok": accuracy_ok,
        "extraction_ok": extraction_ok,
        "passed": accuracy_ok and extraction_ok,
    }


def _pc(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_comparison(reports: list[dict]) -> str:
    """A text table comparing screened models, plus G0 where a reference exists."""
    lines = [
        f"{'model':<34}{'tok':>5}{'acc':>7}{'acc3-5':>8}{'disagree':>9}"
        f"{'dis3-5':>8}{'mixed3-5':>9}{'trunc':>7}{'extr.fail':>10}"
    ]
    for r in reports:
        s = r["study_levels"]
        lines.append(
            f"{r['model_id']:<34}{r['max_tokens']:>5}{_pc(r['accuracy']):>7}"
            f"{_pc(s['accuracy']):>8}{_pc(r['disagreement_rate']):>9}"
            f"{_pc(s['disagreement_rate']):>8}{_pc(s['mixed_rate']):>9}"
            f"{_pc(r['truncation_rate']):>7}{_pc(r['extraction_failure_rate']):>10}"
        )
    lines.append("")
    lines.append("accuracy by level (all samples)")
    for r in reports:
        cells = "  ".join(
            f"L{lv}: {_pc(v['accuracy'])} (n={v['n_problems']})"
            for lv, v in r["by_level"].items()
        )
        lines.append(f"  {r['model_id']:<32}{cells}")
    for r in reports:
        published = PUBLISHED_MATH.get(r["model_id"])
        if published is None:
            continue
        g = g0_check(r["accuracy"], r["extraction_failure_rate"], published)
        lines.append("")
        lines.append(
            f"G0 {r['model_id']}: {'PASS' if g['passed'] else 'FAIL'} "
            f"(accuracy {_pc(r['accuracy'])} vs published {published}% = "
            f"{g['difference_points']:+.1f} points, allowed +/-"
            f"{g['tolerance_points']:.0f}; extraction failure "
            f"{_pc(r['extraction_failure_rate'])}, must be under 5%)"
        )
    return "\n".join(lines)
