"""Exploratory decomposition of what the channels change (PREREG 10.2, item 9).

Descriptive only, written after the analysis freeze, and never a replacement for a
primary table. Everything here is a mean over problems (replicates are averaged
within problem first) with a percentile bootstrap interval; the intervals are not
adjusted for the many quantities shown, and no test is run.

* change types: for each agent, round 0 against its final answer under an arm;
* source of change: the change split into what happens without any peer content
  (``self_reflect``), what the form of peer content adds (``placebo`` minus
  ``self_reflect``) and what the real content adds (the arm minus ``placebo``);
* diversity: distinct parsable answers among the three agents, before and after;
* headroom: how often an arm recovers a round-0 plurality that is wrong although a
  right answer exists (the cases G2 counts), and keeps one that is right;
* conformity: how often an agent moves to a peer's wrong answer, and to a right one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from statistics import mean

import numpy as np

from channel_audit.aggregate import cluster_answers, plurality, plurality_credit
from channel_audit.analysis.contrasts import ANALYSIS_SEED, N_BOOT, summarize
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.gates import collect_units
from channel_audit.grade import answers_equivalent, is_correct

EXPLORATORY = (
    "EXPLORATORY: descriptive, written after the analysis freeze; "
    "not a preregistered result (PREREG 10.2, item 9)"
)
CHANGE_TYPES = (
    "wrong_to_correct",
    "correct_to_wrong",
    "wrong_to_other_wrong",
    "unchanged_correct",
    "unchanged_wrong",
)
SOURCE_METRICS = ("wrong_to_correct", "correct_to_wrong", "net")
SOURCE_PARTS = ("total", "spontaneous", "form", "content")


# ---- one agent ---------------------------------------------------------------


def _clean(answer: str | None) -> str | None:
    return None if answer is None or not answer.strip() else answer


def _same(a: str | None, b: str | None) -> bool:
    """Same answer: both missing, or both present and equivalent."""
    a, b = _clean(a), _clean(b)
    if a is None or b is None:
        return a is None and b is None
    return answers_equivalent(a, b)


def classify_change(
    before: str | None, before_ok: bool, after: str | None, after_ok: bool
) -> str:
    """One agent's change from round 0 to its final answer."""
    if before_ok and after_ok:
        return "unchanged_correct"
    if after_ok:
        return "wrong_to_correct"
    if before_ok:
        return "correct_to_wrong"
    return "unchanged_wrong" if _same(before, after) else "wrong_to_other_wrong"


def agent_changes(unit, arm: str) -> list[str]:
    return [
        classify_change(b, bo, a, ao)
        for b, bo, a, ao in zip(
            unit.r0_answers,
            unit.r0_correct,
            unit.finals[arm],
            unit.final_correct[arm],
            strict=True,
        )
    ]


# ---- summaries over problems -------------------------------------------------


def _by_problem(units: Sequence, value: Callable[[object], float]) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for u in units:
        acc[u.problem_id].append(value(u))
    return {pid: mean(v) for pid, v in acc.items()}


def _ci(per_problem: Mapping[str, float], n_boot: int, seed: int) -> dict:
    s = summarize(per_problem, None, n_boot, seed)
    return {k: s[k] for k in ("mean", "ci95_low", "ci95_high", "n_problems")}


def _ratio(
    num: Mapping[str, float],
    den: Mapping[str, float],
    problems: Sequence[str],
    n_boot: int,
    seed: int,
) -> dict:
    """Pooled ratio of two counts, resampling problems (all of them, even with 0)."""
    ids = sorted(problems)
    n = np.array([num.get(p, 0.0) for p in ids], dtype=float)
    d = np.array([den.get(p, 0.0) for p in ids], dtype=float)
    out = {
        "value": None,
        "numerator": float(n.sum()),
        "denominator": float(d.sum()),
        "ci95_low": None,
        "ci95_high": None,
        "n_problems": int((d > 0).sum()),
    }
    if d.sum() == 0:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    nb, db = n[idx].sum(axis=1), d[idx].sum(axis=1)
    ok = db > 0
    lo, hi = np.quantile(nb[ok] / db[ok], [0.025, 0.975])
    out.update(value=float(n.sum() / d.sum()), ci95_low=float(lo), ci95_high=float(hi))
    return out


# ---- change types and their sources ------------------------------------------


def _share_table(units: Sequence, arms: Sequence[str]) -> dict:
    """arm -> change type -> problem -> share of agents (replicates averaged)."""
    out = {}
    for arm in arms:
        per: dict[str, dict[str, list[float]]] = {
            t: defaultdict(list) for t in CHANGE_TYPES
        }
        for u in units:
            labels = agent_changes(u, arm)
            for t in CHANGE_TYPES:
                per[t][u.problem_id].append(labels.count(t) / len(labels))
        out[arm] = {t: {p: mean(v) for p, v in per[t].items()} for t in CHANGE_TYPES}
    return out


def _net(shares: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    w, c = shares["wrong_to_correct"], shares["correct_to_wrong"]
    return {p: w[p] - c[p] for p in w}


def change_table(
    units: Sequence,
    arms: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    table = _share_table(units, arms)
    return {
        arm: {
            "shares": {t: _ci(table[arm][t], n_boot, seed) for t in CHANGE_TYPES},
            "net": _ci(_net(table[arm]), n_boot, seed),
        }
        for arm in arms
    }


def source_decomposition(
    units: Sequence,
    arms: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    """Split each arm's rate into spontaneous, form and content parts.

    spontaneous = ``self_reflect``; form = ``placebo`` - ``self_reflect``; content =
    arm - ``placebo``. The three add up to the arm's own rate. ``self_reflect`` is the
    baseline and has no row.
    """
    if "self_reflect" not in arms or "placebo" not in arms:
        raise ValueError("the decomposition needs self_reflect and placebo")
    table = _share_table(units, arms)
    rates = {
        "wrong_to_correct": {a: table[a]["wrong_to_correct"] for a in arms},
        "correct_to_wrong": {a: table[a]["correct_to_wrong"] for a in arms},
        "net": {a: _net(table[a]) for a in arms},
    }
    out: dict = {}
    for metric, r in rates.items():
        sr, pl = r["self_reflect"], r["placebo"]
        out[metric] = {}
        for arm in arms:
            if arm == "self_reflect":
                continue
            parts = {
                "total": r[arm],
                "spontaneous": sr,
                "form": {p: pl[p] - sr[p] for p in sr},
                "content": {p: r[arm][p] - pl[p] for p in sr},
            }
            out[metric][arm] = {k: _ci(v, n_boot, seed) for k, v in parts.items()}
    return out


# ---- diversity ---------------------------------------------------------------


def _distinct(answers: Sequence[str | None]) -> int:
    return len(cluster_answers(answers)[1])


def diversity(
    units: Sequence,
    arms: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    """Distinct parsable answers among the three agents, before and after each arm.

    "Unanimous" means at most one distinct parsable answer.
    """
    before = {id(u): _distinct(u.r0_answers) for u in units}
    out = {
        "round0": _ci(_by_problem(units, lambda u: before[id(u)]), n_boot, seed),
        "unanimous_round0": float(np.mean([before[id(u)] <= 1 for u in units])),
        "arms": {},
    }
    for arm in arms:
        after = {id(u): _distinct(u.finals[arm]) for u in units}
        change = {id(u): after[id(u)] - before[id(u)] for u in units}
        out["arms"][arm] = {
            "after": _ci(_by_problem(units, lambda u, a=after: a[id(u)]), n_boot, seed),
            "change": _ci(
                _by_problem(units, lambda u, c=change: c[id(u)]), n_boot, seed
            ),
            "unanimous_after": float(np.mean([after[id(u)] <= 1 for u in units])),
        }
    return out


# ---- headroom ----------------------------------------------------------------


def _recoverable(u) -> bool:
    """A right answer exists in round 0 but the plurality does not hold it (G2)."""
    if not any(u.r0_correct):
        return False
    return not any(is_correct(w, u.gold) for w in plurality(u.r0_answers).winners)


def headroom(
    units: Sequence,
    arms: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    """Recovery of wrong round-0 pluralities and retention of right ones."""
    problems = sorted({u.problem_id for u in units})
    recoverable = [u for u in units if _recoverable(u)]
    right = [u for u in units if plurality_credit(u.r0_answers, u.gold) == 1.0]
    out = {
        "recoverable_units": len(recoverable),
        "plurality_right_units": len(right),
        "arms": {},
    }

    def pooled(group: Sequence, arm: str) -> dict:
        num: dict[str, float] = defaultdict(float)
        den: dict[str, float] = defaultdict(float)
        for u in group:
            num[u.problem_id] += plurality_credit(u.finals[arm], u.gold)
            den[u.problem_id] += 1
        return _ratio(num, den, problems, n_boot, seed)

    for arm in arms:
        out["arms"][arm] = {
            "recovered": pooled(recoverable, arm),
            "retained": pooled(right, arm),
        }
    return out


# ---- conformity --------------------------------------------------------------


def _peer_moves(u, arm: str) -> tuple[int, int, int, int]:
    """Agents that moved to a peer's wrong / right answer, and agents exposed to one.

    An agent is exposed to a wrong (right) answer when a peer held, in round 0, a
    wrong (right) answer different from its own. It moved when its final answer is
    that answer.
    """
    moved_wrong = exposed_wrong = moved_right = exposed_right = 0
    for i, (own, final) in enumerate(zip(u.r0_answers, u.finals[arm], strict=True)):
        peers = [
            (a, ok)
            for j, (a, ok) in enumerate(zip(u.r0_answers, u.r0_correct, strict=True))
            if j != i and _clean(a) is not None
        ]
        wrong = [a for a, ok in peers if not ok and not _same(own, a)]
        right = [a for a, ok in peers if ok and not _same(own, a)]
        exposed_wrong += bool(wrong)
        exposed_right += bool(right)
        moved_wrong += any(_same(final, a) for a in wrong)
        moved_right += any(_same(final, a) for a in right)
    return moved_wrong, exposed_wrong, moved_right, exposed_right


def conformity(
    units: Sequence,
    arms: Sequence[str],
    n_boot: int = N_BOOT,
    seed: int = ANALYSIS_SEED,
) -> dict:
    problems = sorted({u.problem_id for u in units})
    out = {}
    for arm in arms:
        count: dict[str, dict[str, float]] = {
            k: defaultdict(float)
            for k in (
                "moved_wrong",
                "exposed_wrong",
                "moved_right",
                "exposed_right",
                "agents",
            )
        }
        for u in units:
            mw, xw, mr, xr = _peer_moves(u, arm)
            for key, value in (
                ("moved_wrong", mw),
                ("exposed_wrong", xw),
                ("moved_right", mr),
                ("exposed_right", xr),
                ("agents", len(u.r0_answers)),
            ):
                count[key][u.problem_id] += value

        def ratio(n: str, d: str, count=count) -> dict:
            return _ratio(count[n], count[d], problems, n_boot, seed)

        out[arm] = {
            "copied_wrong_per_agent": ratio("moved_wrong", "agents"),
            "exposed_to_wrong": ratio("exposed_wrong", "agents"),
            "copied_wrong_given_exposed": ratio("moved_wrong", "exposed_wrong"),
            "exposed_to_right": ratio("exposed_right", "agents"),
            "adopted_right_given_exposed": ratio("moved_right", "exposed_right"),
        }
    return out


# ---- everything --------------------------------------------------------------


def decompose(ctxs, *, n_boot: int = N_BOOT, seed: int = ANALYSIS_SEED) -> dict:
    cfg = ctxs[0].cfg
    if any(ctx.cfg != cfg for ctx in ctxs):
        raise ValueError("shards were run with different configs")
    has_v2 = any(g.arm in VARIANT_ARMS for ctx in ctxs for g in ctx.gens)
    arms = (*ARMS, *VARIANT_ARMS) if has_v2 else tuple(ARMS)
    units = collect_units(ctxs, arms)
    return {
        "banner": EXPLORATORY,
        "model_id": cfg.model_id,
        "replicates": cfg.replicates,
        "n_problems": len({u.problem_id for u in units}),
        "arms": list(arms),
        "changes": change_table(units, arms, n_boot, seed),
        "source": source_decomposition(units, arms, n_boot, seed),
        "diversity": diversity(units, arms, n_boot, seed),
        "headroom": headroom(units, arms, n_boot, seed),
        "conformity": conformity(units, arms, n_boot, seed),
        "status": {"exploratory": True},
    }


# ---- text report -------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def _pm(ci: Mapping) -> str:
    return f"{ci['mean']:+.1%} [{ci['ci95_low']:+.1%}, {ci['ci95_high']:+.1%}]"


def _ratio_text(r: Mapping) -> str:
    if r["value"] is None:
        return "-"
    return f"{r['value']:.1%} [{r['ci95_low']:.1%}, {r['ci95_high']:.1%}]"


def format_decomposition_report(rep: Mapping) -> str:
    arms = rep["arms"]
    lines = [
        rep["banner"],
        "",
        f"{rep['model_id']}: {rep['n_problems']} problems x "
        f"{rep['replicates']} replicates."
        " Means over problems; 95% percentile bootstrap intervals, unadjusted.",
        "",
        "== change types (share of agents) ==",
        f"{'arm':<18}{'wrong>right':>12}{'right>wrong':>12}{'wrong>other':>12}"
        f"{'stay right':>12}{'stay wrong':>12}   net (right-to-wrong subtracted)",
    ]
    for arm in arms:
        t = rep["changes"][arm]
        s = t["shares"]
        lines.append(
            f"{arm:<18}"
            + "".join(f"{_pct(s[k]['mean']):>12}" for k in CHANGE_TYPES)
            + f"   {_pm(t['net'])}"
        )
    lines += ["", "== source of change: spontaneous + form + content = total =="]
    for metric in SOURCE_METRICS:
        lines += [
            f"-- {metric} --",
            f"{'arm':<18}{'spontaneous':>13}{'form':>9}{'content':>9}{'total':>9}",
        ]
        for arm, parts in rep["source"][metric].items():
            lines.append(
                f"{arm:<18}"
                + "".join(
                    f"{parts[k]['mean']:>{w}.1%}"
                    for k, w in (
                        ("spontaneous", 13),
                        ("form", 9),
                        ("content", 9),
                        ("total", 9),
                    )
                )
            )
    d = rep["diversity"]
    lines += [
        "",
        "== diversity (distinct parsable answers among the three agents) ==",
        f"round 0: {d['round0']['mean']:.2f} distinct, "
        f"{_pct(d['unanimous_round0'])} with at most one",
        f"{'arm':<18}{'after':>8}{'change':>10}{'at most one':>13}",
    ]
    for arm in arms:
        a = d["arms"][arm]
        lines.append(
            f"{arm:<18}{a['after']['mean']:>8.2f}{a['change']['mean']:>+10.2f}"
            f"{_pct(a['unanimous_after']):>13}"
        )
    h = rep["headroom"]
    lines += [
        "",
        "== headroom: recovered = a wrong round-0 plurality with a right answer present"
        f" ({h['recoverable_units']} cases); retained = a right plurality"
        f" ({h['plurality_right_units']} cases) ==",
        f"{'arm':<18}{'recovered':>26}{'retained':>26}",
    ]
    for arm in arms:
        a = h["arms"][arm]
        lines.append(
            f"{arm:<18}{_ratio_text(a['recovered']):>26}{_ratio_text(a['retained']):>26}"
        )
    lines += [
        "",
        "== conformity: moving to a peer's round-0 answer ==",
        f"{'arm':<18}{'exposed wrong':>15}{'copied wrong | exposed':>30}"
        f"{'adopted right | exposed':>30}",
    ]
    for arm in arms:
        c = rep["conformity"][arm]
        lines.append(
            f"{arm:<18}{_ratio_text(c['exposed_to_wrong']):>15}"
            f"{_ratio_text(c['copied_wrong_given_exposed']):>30}"
            f"{_ratio_text(c['adopted_right_given_exposed']):>30}"
        )
    lines += ["", rep["banner"]]
    return "\n".join(lines)
