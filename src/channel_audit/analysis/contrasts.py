"""M7: the six primary contrasts, their decision rules and the preregistered extras.

Written and tested on mock data, then frozen (``analysis_freeze``) before it is run
once on the confirmatory shards. Nothing here chooses a problem, an arm or a
threshold from the data: the best low-bandwidth arm was fixed on the pilot, the
margins are the plan's, and every exclusion rests on truncation, never on whether
an answer was right.

The unit is the problem. Replicates are averaged within problem, and arms are
compared with a paired bootstrap over problems. P-values come from the same
resamples as the intervals; the Holm family is contrasts 3 to 6. Contrasts 1 and 2
are decided by their intervals alone, and the failure-message v2 contrast is reported
beside contrast 4, secondary and unadjusted (PREREG sections 7 and 8).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

import numpy as np

from channel_audit.aggregate import plurality_credit
from channel_audit.analysis.stats import bootstrap_means
from channel_audit.budget import (
    BudgetOutOfRange,
    arm_costs,
    matched_sample_count,
    sc_credit,
    sc_sample_cost,
)
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.gates import collect_units, g1_disagreement, g2_headroom
from channel_audit.quality import format_quality_report, run_quality
from channel_audit.runner import N_AGENTS
from channel_audit.validate import g3_check

N_BOOT = 10_000
ANALYSIS_SEED = 0
EQUIV_MARGIN = 0.02  # contrast 1: the 90% interval must lie within +/- 2 points
NONINF_MARGIN = -0.015  # contrast 2: the lower 95% bound must exceed -1.5 points
HOLM_ALPHA = 0.05
BEST_LOW_BANDWIDTH = "dissent_only"  # chosen on the pilot only (PREREG section 8)

C1 = "full_solution_vs_sc"
C2 = "answer_only_vs_full_solution"
C3 = "placebo_vs_self_reflect_correct_to_wrong"
C4 = "failure_cause_vs_full_solution"
C5 = "answer_conf_vs_full_solution"
C6 = "best_low_bandwidth_vs_sc"
C4_V2 = "failure_cause_v2_vs_full_solution"
FAMILY = (C3, C4, C5, C6)

BASES = {"generated": 0, "generated_plus_prompt": 1}  # index into sc_sample_cost
_PARTS = {  # a failure-cause arm also pays for, and is affected by, its messages
    "failure_cause": ("failure_cause", "failure_msg"),
    "failure_cause_v2": ("failure_cause_v2", "failure_msg_v2"),
}


# ---- tests and decision rules -------------------------------------------------


def holm_adjust(p: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    ordered = sorted(p, key=lambda name: p[name])
    adjusted, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p[name]))
        adjusted[name] = running
    return adjusted


def bootstrap_p(means: np.ndarray, alternative: str) -> float:
    """P-value from resampled means: the share on the null side of zero.

    ``"greater"`` tests mean > 0; ``"two-sided"`` doubles the smaller tail. One is
    added to the count and to the total, so a p-value is never exactly zero.
    """
    n = means.size
    at_or_below = (1 + int((means <= 0).sum())) / (n + 1)
    if alternative == "greater":
        return at_or_below
    if alternative == "two-sided":
        at_or_above = (1 + int((means >= 0).sum())) / (n + 1)
        return min(1.0, 2 * min(at_or_below, at_or_above))
    raise ValueError(
        f"alternative must be 'greater' or 'two-sided', not {alternative!r}"
    )


def summarize(
    diffs: Mapping[str, float],
    alternative: str | None,
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    """Mean, 95% and 90% percentile intervals and (if asked) a p-value."""
    values = np.array([diffs[k] for k in sorted(diffs)], dtype=float)
    if not np.isfinite(values).all():
        # NaN compares false both ways, which would read as a tiny p-value
        raise ValueError("a paired difference is not finite")
    means = bootstrap_means(values, n_boot, seed)
    lo95, hi95, lo90, hi90 = np.quantile(means, [0.025, 0.975, 0.05, 0.95])
    return {
        "n_problems": int(values.size),
        "mean": float(values.mean()),
        "ci95_low": float(lo95),
        "ci95_high": float(hi95),
        "ci90_low": float(lo90),
        "ci90_high": float(hi90),
        "alternative": alternative,
        "p": None if alternative is None else bootstrap_p(means, alternative),
    }


def is_equivalent(
    ci90_low: float, ci90_high: float, margin: float = EQUIV_MARGIN
) -> bool:
    return -margin <= ci90_low and ci90_high <= margin


def is_noninferior(ci95_low: float, margin: float = NONINF_MARGIN) -> bool:
    return ci95_low > margin


# ---- per-unit quantities --------------------------------------------------------


@dataclass(frozen=True)
class UnitStats:
    """What the contrasts need from one problem and replicate, computed once."""

    problem_id: str
    replicate: int
    level: int
    credit: dict[str, float]  # arm -> plurality credit of its three final answers
    c2w: dict[str, float]  # arm -> agents right in round 0, wrong after, over 3
    sc: tuple[float, ...]  # self-consistency credit at 1..n samples

    def sc_at(self, k: float) -> float:
        """Self-consistency credit at a fractional sample count (linear between)."""
        if not 1 <= k <= len(self.sc):
            raise BudgetOutOfRange(f"k={k} outside [1, {len(self.sc)}]")
        lo, hi = math.floor(k), math.ceil(k)
        at_lo = self.sc[lo - 1]
        if lo == hi:
            return at_lo
        return at_lo + (k - lo) * (self.sc[hi - 1] - at_lo)


def unit_stats(units, arms: Sequence[str]) -> list[UnitStats]:
    out = []
    for u in units:
        out.append(
            UnitStats(
                problem_id=u.problem_id,
                replicate=u.replicate,
                level=u.level,
                credit={a: plurality_credit(u.finals[a], u.gold) for a in arms},
                c2w={
                    a: sum(
                        was and not now
                        for was, now in zip(
                            u.r0_correct, u.final_correct[a], strict=True
                        )
                    )
                    / N_AGENTS
                    for a in arms
                },
                sc=tuple(
                    sc_credit(u.sc_answers, u.gold, k)
                    for k in range(1, len(u.sc_answers) + 1)
                ),
            )
        )
    return out


def _by_problem(stats: Sequence[UnitStats], value: Callable[[UnitStats], float]):
    acc: dict[str, list[float]] = defaultdict(list)
    for s in stats:
        acc[s.problem_id].append(value(s))
    return {pid: mean(v) for pid, v in acc.items()}


def _diff(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    return {pid: a[pid] - b[pid] for pid in a}


# ---- the contrasts --------------------------------------------------------------


def _row(diffs, alternative, rule, n_boot, seed) -> dict:
    return {
        **summarize(diffs, alternative, n_boot, seed),
        "estimable": True,
        "rule": rule,
        "decision": None,
    }


def _not_estimable(rule: str, why: str) -> dict:
    return {"estimable": False, "rule": rule, "decision": "not estimable", "why": why}


def run_contrasts(
    stats: Sequence[UnitStats],
    gens: Sequence,
    *,
    n_pool: int,
    best_arm: str = BEST_LOW_BANDWIDTH,
    basis: str = "generated",
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
    problem_ids: set[str] | None = None,
    allow_out_of_range: bool = False,
) -> dict:
    """Arm table and contrasts 1 to 6 (plus the v2 contrast) on one set of problems.

    Costs, and so the matched self-consistency sample counts, are those of the
    problems analysed. With ``allow_out_of_range`` an arm whose budget lies beyond
    the drawn pool is marked not estimable instead of raising; the contrasts that
    need it are then not estimable.
    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {sorted(BASES)}, not {basis!r}")
    if problem_ids is not None:
        stats = [s for s in stats if s.problem_id in problem_ids]
        gens = [g for g in gens if g.problem_id in problem_ids]
    if not stats:
        raise ValueError("no problems to analyse")
    arms = [a for a in (*ARMS, *VARIANT_ARMS) if a in stats[0].credit]
    if best_arm not in arms:
        raise ValueError(f"best low-bandwidth arm {best_arm!r} is not in the run")

    costs = arm_costs(gens, arms)
    per_sample = sc_sample_cost(gens)[BASES[basis]]
    credit = {a: _by_problem(stats, lambda s, a=a: s.credit[a]) for a in arms}
    k: dict[str, float] = {}
    sc: dict[str, dict[str, float]] = {}
    gain: dict[str, dict[str, float]] = {}
    for a in arms:
        tokens = (
            costs[a].generated_tokens if basis == "generated" else costs[a].total_tokens
        )
        try:
            k[a] = matched_sample_count(tokens, per_sample, 3 + n_pool)
        except BudgetOutOfRange:
            if not allow_out_of_range:
                raise
            k[a] = tokens / per_sample
            continue
        sc[a] = _by_problem(stats, lambda s, kk=k[a]: s.sc_at(kk))
        gain[a] = _diff(credit[a], sc[a])
    c2w = {
        a: _by_problem(stats, lambda s, a=a: s.c2w[a])
        for a in ("self_reflect", "placebo")
    }

    arm_rows = {}
    for a in arms:
        tokens = (
            costs[a].generated_tokens if basis == "generated" else costs[a].total_tokens
        )
        row = {
            "accuracy": mean(credit[a].values()),
            "tokens": tokens,
            "matched_k": k[a],
            "estimable": a in gain,
            "sc_accuracy": None,
            "gain": None,
            "gain_ci95": None,
        }
        if a in gain:
            s = summarize(gain[a], None, n_boot, seed)
            row.update(
                sc_accuracy=mean(sc[a].values()),
                gain=s["mean"],
                gain_ci95=[s["ci95_low"], s["ci95_high"]],
            )
        arm_rows[a] = row

    def gain_gap(arm: str):
        if arm not in gain or "full_solution" not in gain:
            return None
        return _diff(gain[arm], gain["full_solution"])

    spec = {
        C1: (
            gain.get("full_solution"),
            None,
            "90% CI within +/-2 points of matched SC",
        ),
        C2: (
            _diff(credit["answer_only"], credit["full_solution"]),
            None,
            "lower 95% bound above -1.5 points",
        ),
        C3: (
            _diff(c2w["placebo"], c2w["self_reflect"]),
            "greater",
            "one-sided, Holm-adjusted",
        ),
        C4: (gain_gap("failure_cause"), "two-sided", "two-sided, Holm-adjusted"),
        C5: (gain_gap("answer_conf"), "two-sided", "two-sided, Holm-adjusted"),
        C6: (gain.get(best_arm), "two-sided", "two-sided, Holm-adjusted"),
    }
    contrasts = {}
    for name, (diffs, alternative, rule) in spec.items():
        if diffs is None:
            contrasts[name] = _not_estimable(rule, "needs an arm beyond the drawn pool")
            continue
        row = _row(diffs, alternative, rule, n_boot, seed)
        if name == C1:
            row["decision"] = (
                "equivalent"
                if is_equivalent(row["ci90_low"], row["ci90_high"])
                else "equivalence not shown"
            )
        elif name == C2:
            row["decision"] = (
                "non-inferior"
                if is_noninferior(row["ci95_low"])
                else "non-inferiority not shown"
            )
        contrasts[name] = row

    family = {n: contrasts[n]["p"] for n in FAMILY if contrasts[n]["estimable"]}
    for name, p_holm in holm_adjust(family).items():
        contrasts[name]["p_holm"] = p_holm
        contrasts[name]["reject"] = p_holm < HOLM_ALPHA
        contrasts[name]["decision"] = (
            "rejected (Holm)" if p_holm < HOLM_ALPHA else "not rejected (Holm)"
        )

    secondary = {}
    if "failure_cause_v2" in arms:
        rule = "secondary, two-sided, not in the Holm family"
        diffs = gain_gap("failure_cause_v2")
        if diffs is None:
            secondary[C4_V2] = _not_estimable(
                rule, "needs an arm beyond the drawn pool"
            )
        else:
            secondary[C4_V2] = _row(diffs, "two-sided", rule, n_boot, seed)
            secondary[C4_V2]["decision"] = "secondary, unadjusted"

    return {
        "n_problems": len(credit["self_reflect"]),
        "basis": basis,
        "best_low_bandwidth_arm": best_arm,
        "arms": arm_rows,
        "contrasts": contrasts,
        "secondary": secondary,
    }


# ---- sensitivity ------------------------------------------------------------------


def truncated_problems(gens: Sequence, arms: Sequence[str]) -> set[str]:
    """Problems with a generation cut off at ``max_tokens`` in any of these arms.

    A failure-cause arm includes its failure messages. The test is on the length
    cut-off only; whether an answer was right plays no part.
    """
    names: set[str] = set()
    for arm in arms:
        names.update(_PARTS.get(arm, (arm,)))
    return {
        g.problem_id for g in gens if g.arm in names and g.finish_reason == "length"
    }


def exclusion_sets(gens: Sequence, flagged_arms: Sequence[str]) -> dict[str, set[str]]:
    """Problems to drop in each preregistered truncation sensitivity run.

    ``no_round0_truncation``: any round-0 agent was cut off (PREREG section 3).
    ``no_flagged_arm_truncation``: only when an arm is more than 2 points from
    ``self_reflect``; drops the problems where it or ``self_reflect`` was cut off.
    """
    out = {"no_round0_truncation": truncated_problems(gens, ("round0",))}
    if flagged_arms:
        out["no_flagged_arm_truncation"] = truncated_problems(
            gens, ("self_reflect", *flagged_arms)
        )
    return out


# ---- the whole analysis ----------------------------------------------------------


def analyze(
    ctxs,
    *,
    best_arm: str = BEST_LOW_BANDWIDTH,
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    """Run every preregistered analysis on the pooled shards."""
    cfg = ctxs[0].cfg
    if any(ctx.cfg != cfg for ctx in ctxs):
        raise ValueError("shards were run with different configs")
    gens = [g for ctx in ctxs for g in ctx.gens]
    arms = (
        (*ARMS, *VARIANT_ARMS)
        if any(g.arm in VARIANT_ARMS for g in gens)
        else tuple(ARMS)
    )
    units = collect_units(ctxs, arms)
    stats = unit_stats(units, arms)
    all_ids = {s.problem_id for s in stats}
    g3 = g3_check(gens, cfg.context_limit)

    def run(**kw):
        return run_contrasts(
            stats,
            gens,
            n_pool=cfg.n_pool,
            best_arm=best_arm,
            n_boot=n_boot,
            seed=seed,
            **kw,
        )

    analyses = {"main": run()}
    secondary, empty = [], []
    for level in sorted({s.level for s in stats}):
        ids = {s.problem_id for s in stats if s.level == level}
        analyses[f"level_{level}"] = run(problem_ids=ids)
        secondary.append(f"level_{level}")
    for name, drop in exclusion_sets(gens, g3.flagged_arms).items():
        keep = all_ids - drop
        if not keep:  # nothing left to analyse: say so, never skip silently
            empty.append(name)
            continue
        analyses[name] = run(problem_ids=keep)
        secondary.append(name)
    analyses["prompt_tokens_in_budget"] = run(
        basis="generated_plus_prompt", allow_out_of_range=True
    )
    secondary.append("prompt_tokens_in_budget")

    sc_curve = {
        kk: mean(_by_problem(stats, lambda s, kk=kk: s.sc[kk - 1]).values())
        for kk in range(1, len(stats[0].sc) + 1)
    }
    return {
        "model_id": cfg.model_id,
        "replicates": cfg.replicates,
        "n_problems": len(all_ids),
        "arms_run": list(arms),
        "analyses": analyses,
        "sc_curve": sc_curve,
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
        "quality": run_quality(ctxs),
        "status": {
            "primary": ["main"],
            "preregistered_secondary": secondary,
            "exploratory": [],
            "not_triggered": [] if g3.flagged_arms else ["no_flagged_arm_truncation"],
            "empty": empty,
            "rehearsal": False,
        },
    }


# ---- text report -----------------------------------------------------------------


def _fmt_p(p: float | None) -> str:
    return "-" if p is None else f"{p:.4f}"


def _contrast_lines(contrasts: Mapping[str, dict]) -> list[str]:
    lines = [
        f"{'contrast':<44}{'mean':>8}{'95% CI':>19}{'p':>8}{'p Holm':>8}  decision"
    ]
    for name, c in contrasts.items():
        if not c["estimable"]:
            lines.append(f"{name:<44}{'':>8}{'':>19}{'':>8}{'':>8}  {c['decision']}")
            continue
        ci = f"[{c['ci95_low']:+.1%}, {c['ci95_high']:+.1%}]"
        lines.append(
            f"{name:<44}{c['mean']:>+8.1%}{ci:>19}{_fmt_p(c['p']):>8}"
            f"{_fmt_p(c.get('p_holm')):>8}  {c['decision']}"
        )
    return lines


def _arm_lines(arms: Mapping[str, dict]) -> list[str]:
    lines = [f"{'arm':<18}{'acc':>7}{'tokens':>8}{'SC k':>7}{'SC acc':>8}{'gain':>8}"]
    for arm, r in arms.items():
        sc = "-" if r["sc_accuracy"] is None else f"{r['sc_accuracy']:.1%}"
        gain = "-" if r["gain"] is None else f"{r['gain']:+.1%}"
        lines.append(
            f"{arm:<18}{r['accuracy']:>7.1%}{r['tokens']:>8.0f}{r['matched_k']:>7.2f}"
            f"{sc:>8}{gain:>8}"
        )
    return lines


def format_analysis_report(rep: dict) -> str:
    g = rep["gates"]
    g1, g2, g3 = g["G1"], g["G2"], g["G3"]
    trunc = g3["truncation"]
    revision = [a for a in rep["arms_run"] if a in trunc]
    worst = max((trunc[a] for a in revision), default=0.0)
    lines = []
    if rep["status"].get("rehearsal"):
        lines += ["REHEARSAL on mock data: no confirmatory result", ""]
    lines += [
        f"{rep['model_id']}: {rep['n_problems']} problems x "
        f"{rep['replicates']} replicates",
        "",
        format_quality_report(rep["quality"]),
        "",
        f"G1 round-0 disagreement {g1['rate']:.1%} "
        f"(need {g1['threshold']:.0%}); by level: "
        + ", ".join(f"L{lv} {v:.0%}" for lv, v in g1["by_level"].items()),
        f"G2 recoverable headroom {g2['rate']:.1%} "
        f"(need {g2['threshold']:.0%}); by level: "
        + ", ".join(f"L{lv} {v:.1%}" for lv, v in g2["by_level"].items()),
        f"G3 pooled: overflow {g3['overflow_total']}, "
        f"failing arms {g3['failing_arms']}, "
        f"flagged arms {g3['flagged_arms']}, "
        f"highest revision-arm truncation {worst:.1%} "
        f"-> {'PASS' if g3['ok'] else 'FAIL'}",
        "",
        "SC accuracy by samples: "
        + ", ".join(f"k={k} {v:.1%}" for k, v in rep["sc_curve"].items()),
    ]
    for name, a in rep["analyses"].items():
        label = "PRIMARY" if name in rep["status"]["primary"] else "secondary"
        lines += [
            "",
            f"== {name} ({label}; {a['n_problems']} problems; budget in {a['basis']} "
            f"tokens; Holm over contrasts 3 to 6) ==",
        ]
        if name == "main":
            lines += _arm_lines(a["arms"]) + [""]
        lines += _contrast_lines({**a["contrasts"], **a["secondary"]})
    if rep["status"]["not_triggered"]:
        lines += ["", "not triggered: " + ", ".join(rep["status"]["not_triggered"])]
    if rep["status"]["empty"]:
        lines += [
            "",
            "not run, every problem excluded: " + ", ".join(rep["status"]["empty"]),
        ]
    lines += [
        "",
        "Anything not listed above is exploratory and must be labelled so.",
    ]
    return "\n".join(lines)
