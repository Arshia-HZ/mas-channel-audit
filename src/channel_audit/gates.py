"""M5 pilot gates G1 and G2, the arm table, and the inputs to the power calculation.

All quantities are computed from graded generations and never filter problems or
examples by an arm's outcome. The unit is one problem and one replicate; the
analysis unit is the problem, so replicates are averaged within problem first.

Definitions (also written into PREREG.md):

* G1, round-0 disagreement: the share of replicates whose three round-0 answers
  contain two or more distinct parsable answers. Pass at 15% or more.
* G2, recoverable headroom: the share of replicates where at least one round-0
  answer is correct but no answer in the round-0 plurality is. Pass at 5% or more.
* G3: see ``validate.g3_check`` (zero overflow, each revision arm under 5%).
* G4: the share of sampled failure messages not labelled vague, from
  ``failure_review``. Pass at 50% or more.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean, stdev

from channel_audit.aggregate import cluster_answers, plurality, plurality_credit
from channel_audit.analysis.stats import paired_bootstrap_ci
from channel_audit.budget import (
    arm_costs,
    matched_sample_count,
    sc_credit,
    sc_credit_interp,
    sc_sample_cost,
)
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.grade import is_correct
from channel_audit.runner import N_AGENTS, RunContext, final_generations, round0_gens
from channel_audit.validate import g3_check

G1_MIN = 0.15
G2_MIN = 0.05
LOW_BANDWIDTH = ("answer_only", "answer_conf", "dissent_only", "failure_cause")
Z_ALPHA_BETA = 1.96 + 0.84  # two-sided 5% test with 80% power


@dataclass(frozen=True)
class Unit:
    """One problem and replicate, with every answer that the gates need."""

    problem_id: str
    replicate: int
    subject: str
    level: int
    gold: str
    r0_answers: list[str | None]
    r0_correct: list[bool]
    sc_answers: list[str | None]  # round 0 followed by the pool
    finals: dict[str, list[str | None]]  # arm -> the three final answers
    final_correct: dict[str, list[bool]]


def collect_units(ctxs: Sequence[RunContext], arms: Sequence[str] = ARMS) -> list[Unit]:
    units = []
    for ctx in ctxs:
        pool: dict[tuple[str, int], list] = defaultdict(list)
        for g in ctx.gens:
            if g.arm == "sc_pool":
                pool[(g.problem_id, g.replicate)].append(g)
        for p in ctx.problems:
            for rep in range(ctx.cfg.replicates):
                r0 = round0_gens(ctx, p.problem_id, rep)
                graded_r0 = [ctx.graded.get(g.gen_key) for g in r0]
                pool_gens = sorted(
                    pool[(p.problem_id, rep)], key=lambda g: g.sample_idx
                )
                finals, final_correct = {}, {}
                for arm in arms:
                    graded = [
                        ctx.graded.get(g.gen_key)
                        for g in final_generations(ctx, arm, p.problem_id, rep)
                    ]
                    finals[arm] = [r.extracted for r in graded]
                    final_correct[arm] = [r.correct for r in graded]
                units.append(
                    Unit(
                        problem_id=p.problem_id,
                        replicate=rep,
                        subject=p.subject,
                        level=p.level,
                        gold=p.gold,
                        r0_answers=[r.extracted for r in graded_r0],
                        r0_correct=[r.correct for r in graded_r0],
                        sc_answers=[r.extracted for r in graded_r0]
                        + [ctx.graded.get(g.gen_key).extracted for g in pool_gens],
                        finals=finals,
                        final_correct=final_correct,
                    )
                )
    return units


def g1_disagreement(units: Sequence[Unit]) -> dict:
    flags = [len(cluster_answers(u.r0_answers)[1]) >= 2 for u in units]
    by_level: dict[int, list[bool]] = defaultdict(list)
    for u, flag in zip(units, flags, strict=True):
        by_level[u.level].append(flag)
    rate = sum(flags) / len(flags)
    return {
        "n": len(units),
        "rate": rate,
        "by_level": {lv: sum(v) / len(v) for lv, v in sorted(by_level.items())},
        "threshold": G1_MIN,
        "passed": rate >= G1_MIN,
    }


def g2_headroom(units: Sequence[Unit]) -> dict:
    flags = []
    by_level: dict[int, list[bool]] = defaultdict(list)
    for u in units:
        recoverable = False
        if any(u.r0_correct):
            winners = plurality(u.r0_answers).winners
            recoverable = not any(is_correct(w, u.gold) for w in winners)
        flags.append(recoverable)
        by_level[u.level].append(recoverable)
    rate = sum(flags) / len(flags)
    return {
        "n": len(units),
        "rate": rate,
        "by_level": {lv: sum(v) / len(v) for lv, v in sorted(by_level.items())},
        "threshold": G2_MIN,
        "passed": rate >= G2_MIN,
    }


def required_n(sd: float, d: float) -> int:
    """Problems needed to detect a paired difference ``d`` (both as fractions)."""
    return max(1, math.ceil(Z_ALPHA_BETA**2 * sd**2 / d**2))


def _by_problem(units: Sequence[Unit], value) -> dict[str, float]:
    """Average ``value(unit)`` over the replicates of each problem."""
    acc: dict[str, list[float]] = defaultdict(list)
    for u in units:
        acc[u.problem_id].append(value(u))
    return {pid: mean(v) for pid, v in acc.items()}


def _correct_to_wrong(u: Unit, arm: str) -> float:
    flips = sum(
        was and not now
        for was, now in zip(u.r0_correct, u.final_correct[arm], strict=True)
    )
    return flips / N_AGENTS


def _contrast(diffs: dict[str, float], confirm_available: int) -> dict:
    values = [diffs[k] for k in sorted(diffs)]
    m, lo, hi = paired_bootstrap_ci(values)
    sd = stdev(values) if len(values) > 1 else 0.0
    n3, n2 = required_n(sd, 0.03), required_n(sd, 0.02)
    return {
        "n_problems": len(values),
        "mean": m,
        "ci_low": lo,
        "ci_high": hi,
        "sd": sd,
        "n_for_3_points": n3,
        "n_for_2_points": n2,
        "n_for_3_points_capped": min(n3, confirm_available),
        "n_for_2_points_capped": min(n2, confirm_available),
    }


def pilot_report(
    ctxs: Sequence[RunContext], confirm_available: int, arms: Sequence[str] = ARMS
) -> dict:
    """Gates G1 to G3, the arm table and the paired-difference SDs of the pilot.

    The failure-message v2 arm is included automatically when its generations
    are stored.
    """
    has_v2 = any(g.arm in VARIANT_ARMS for ctx in ctxs for g in ctx.gens)
    if has_v2:
        arms = (*arms, *VARIANT_ARMS)
    units = collect_units(ctxs, arms)
    gens = [g for ctx in ctxs for g in ctx.gens]
    n_pool = ctxs[0].cfg.n_pool
    costs = arm_costs(gens)
    per_sample, _ = sc_sample_cost(gens)

    credit = {
        arm: _by_problem(units, lambda u, a=arm: plurality_credit(u.finals[a], u.gold))
        for arm in arms
    }
    k = {
        arm: matched_sample_count(costs[arm].generated_tokens, per_sample, 3 + n_pool)
        for arm in arms
    }
    sc = {
        arm: _by_problem(
            units, lambda u, a=arm: sc_credit_interp(u.sc_answers, u.gold, k[a])
        )
        for arm in arms
    }
    gain = {arm: {p: credit[arm][p] - sc[arm][p] for p in credit[arm]} for arm in arms}
    gain_mean = {arm: mean(gain[arm].values()) for arm in arms}
    candidates = [a for a in (*LOW_BANDWIDTH, *VARIANT_ARMS) if a in gain_mean]
    best = max(candidates, key=lambda a: gain_mean[a])
    c2w = {
        arm: _by_problem(units, lambda u, a=arm: _correct_to_wrong(u, a))
        for arm in ("self_reflect", "placebo")
    }

    diffs = {
        "full_solution_vs_sc": gain["full_solution"],
        "answer_only_vs_full_solution": {
            p: credit["answer_only"][p] - credit["full_solution"][p]
            for p in credit["full_solution"]
        },
        "placebo_vs_self_reflect_correct_to_wrong": {
            p: c2w["placebo"][p] - c2w["self_reflect"][p] for p in c2w["placebo"]
        },
        "failure_cause_vs_full_solution": {
            p: gain["failure_cause"][p] - gain["full_solution"][p]
            for p in gain["full_solution"]
        },
        "answer_conf_vs_full_solution": {
            p: gain["answer_conf"][p] - gain["full_solution"][p]
            for p in gain["full_solution"]
        },
        "best_low_bandwidth_vs_sc": gain[best],
    }
    if has_v2:
        diffs["failure_cause_v2_vs_full_solution"] = {
            p: gain["failure_cause_v2"][p] - gain["full_solution"][p]
            for p in gain["full_solution"]
        }
    sc_curve = {
        kk: mean(
            _by_problem(
                units, lambda u, k_=kk: sc_credit(u.sc_answers, u.gold, k_)
            ).values()
        )
        for kk in range(1, 3 + n_pool + 1)
    }
    limit = ctxs[0].cfg.context_limit
    g3 = g3_check(gens, limit)
    return {
        "n_problems": len({u.problem_id for u in units}),
        "replicates": ctxs[0].cfg.replicates,
        "model_id": ctxs[0].cfg.model_id,
        "arms": {
            arm: {
                "accuracy": mean(credit[arm].values()),
                "tokens": costs[arm].generated_tokens,
                "matched_k": k[arm],
                "sc_accuracy": mean(sc[arm].values()),
                "gain": gain_mean[arm],
            }
            for arm in arms
        },
        "best_low_bandwidth_arm": best,
        "sc_curve": sc_curve,
        "contrasts": {
            name: _contrast(d, confirm_available) for name, d in diffs.items()
        },
        "gates": {
            "G1": g1_disagreement(units),
            "G2": g2_headroom(units),
            "G3": {
                "ok": g3.ok,
                "overflow_total": g3.overflow_total,
                "failing_arms": g3.failing_arms,
                "flagged_arms": g3.flagged_arms,
                "truncation": g3.truncation,
            },
        },
    }


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def format_pilot_report(rep: dict, g4: dict | None) -> str:
    g = rep["gates"]
    g1, g2, g3 = g["G1"], g["G2"], g["G3"]
    lines = [
        f"{rep['model_id']}: {rep['n_problems']} problems x "
        f"{rep['replicates']} replicates",
        "",
        f"G1 round-0 disagreement: {g1['rate']:.1%} of replicates "
        f"(need {g1['threshold']:.0%}) -> {_verdict(g1['passed'])}",
        "   by level: "
        + ", ".join(f"L{lv} {v:.0%}" for lv, v in g1["by_level"].items()),
        f"G2 recoverable headroom: {g2['rate']:.1%} of replicates "
        f"(need {g2['threshold']:.0%}) -> {_verdict(g2['passed'])}",
        "   by level: "
        + ", ".join(f"L{lv} {v:.1%}" for lv, v in g2["by_level"].items()),
        f"G3 overflow {g3['overflow_total']}, failing arms {g3['failing_arms']}, "
        f"flagged arms {g3['flagged_arms']} -> {_verdict(g3['ok'])}",
    ]
    if g4 is None:
        lines.append("G4 failure messages: pending (label the sample first)")
    else:
        lines.append(
            f"G4 failure messages: {g4['share_not_vague']:.0%} not vague "
            f"({g4['useful']} useful, {g4['wrong']} wrong, {g4['vague']} vague of "
            f"{g4['n']}; need {g4['threshold']:.0%}) -> {_verdict(g4['passed'])}"
        )
    lines += [
        "",
        f"{'arm':<15}{'acc':>7}{'tokens':>8}{'SC k':>6}{'SC acc':>8}{'gain':>8}",
    ]
    for arm, r in rep["arms"].items():
        lines.append(
            f"{arm:<15}{r['accuracy']:>7.1%}{r['tokens']:>8.0f}{r['matched_k']:>6.2f}"
            f"{r['sc_accuracy']:>8.1%}{r['gain']:>+8.1%}"
        )
    curve = rep["sc_curve"]
    lines.append(
        "SC accuracy by samples: "
        + ", ".join(f"k={k} {v:.1%}" for k, v in curve.items())
    )
    lines += [
        f"best low-bandwidth arm (chosen on the pilot): "
        f"{rep['best_low_bandwidth_arm']}",
        "",
        f"{'contrast':<44}{'mean':>8}{'95% CI':>18}{'sd':>7}{'n@3pt':>7}{'n@2pt':>7}",
    ]
    for name, c in rep["contrasts"].items():
        ci = f"[{c['ci_low']:+.1%}, {c['ci_high']:+.1%}]"
        lines.append(
            f"{name:<44}{c['mean']:>+8.1%}{ci:>18}{c['sd']:>7.3f}"
            f"{c['n_for_3_points_capped']:>7}{c['n_for_2_points_capped']:>7}"
        )
    return "\n".join(lines)
