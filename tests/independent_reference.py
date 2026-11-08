"""A second implementation of the M7 arithmetic, written separately on purpose.

It reads the raw JSONL files of a run and imports nothing from ``channel_audit``, so
a mistake in ``analysis/contrasts.py`` (a wrong average, budget, tie credit or p-value)
cannot also be in this code. Answers are compared as strings, which is exact for the
integer answers of the mock backend. ``tests/test_analysis_independent.py`` compares it
with the real analysis.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ARMS = [
    "self_reflect",
    "answer_only",
    "answer_conf",
    "dissent_only",
    "failure_cause",
    "full_solution",
    "placebo",
    "failure_cause_v2",
]
MESSAGES = {"failure_cause": "failure_msg", "failure_cause_v2": "failure_msg_v2"}
FAMILY = (
    "placebo_vs_self_reflect_correct_to_wrong",
    "failure_cause_vs_full_solution",
    "answer_conf_vs_full_solution",
    "best_low_bandwidth_vs_sc",
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]


def load(root, *, drop_round0_truncated=False, only_level=None):
    gens, graded, problems = [], {}, {}
    for shard in sorted(Path(root).glob("shard_*")):
        gens += _rows(shard / "generations.jsonl")
        graded.update({g["gen_key"]: g for g in _rows(shard / "graded.jsonl")})
        problems.update({p["problem_id"]: p for p in _rows(shard / "problems.jsonl")})
    keep = set(problems)
    if only_level is not None:
        keep = {p for p in keep if problems[p]["level"] == only_level}
    if drop_round0_truncated:
        keep -= {
            g["problem_id"]
            for g in gens
            if g["arm"] == "round0" and g["finish_reason"] == "length"
        }
    gens = [g for g in gens if g["problem_id"] in keep]
    return gens, graded, {k: v for k, v in problems.items() if k in keep}


def credit(answers, gold):
    """Plurality credit: 1, or 1/m on an m-way tie containing the gold answer."""
    votes = Counter(a for a in answers if a is not None)
    if not votes:
        return 0.0
    top = max(votes.values())
    winners = [a for a, n in votes.items() if n == top]
    return 1 / len(winners) if gold in winners else 0.0


def sc_credit(answers, gold, k):
    """Exact average over every subset of ``k`` of the samples."""
    subsets = list(combinations(range(len(answers)), k))
    return sum(credit([answers[i] for i in s], gold) for s in subsets) / len(subsets)


def interpolate(curve, k):
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    if lo == hi:
        return curve[lo - 1]
    return curve[lo - 1] + (k - lo) * (curve[hi - 1] - curve[lo - 1])


def quantities(root, **kw):
    gens, graded, problems = load(root, **kw)
    by = defaultdict(list)
    for g in gens:
        by[(g["problem_id"], g["replicate"], g["arm"])].append(g)

    def answer(g):
        return graded[g["gen_key"]]["extracted"]

    def correct(g):
        return graded[g["gen_key"]]["correct"]

    units = sorted({(g["problem_id"], g["replicate"]) for g in gens})
    arms = [a for a in ARMS if any(g["arm"] == a for g in gens)]
    score = {a: defaultdict(list) for a in arms}
    flips = {a: defaultdict(list) for a in ("self_reflect", "placebo")}
    curves = defaultdict(list)
    tokens = {a: [] for a in arms}
    for pid, rep in units:
        gold = problems[pid]["gold"]
        r0 = sorted(by[(pid, rep, "round0")], key=lambda g: g["agent"])
        pool = sorted(by[(pid, rep, "sc_pool")], key=lambda g: g["sample_idx"])
        nine = [answer(g) for g in r0] + [answer(g) for g in pool]
        curves[pid].append([sc_credit(nine, gold, k) for k in range(1, 10)])
        for a in arms:
            revised = {g["agent"]: g for g in by[(pid, rep, a)]}
            final = [revised.get(i, r0[i]) for i in range(3)]  # dissent_only keeps r0
            score[a][pid].append(credit([answer(g) for g in final], gold))
            if a in flips:
                flips[a][pid].append(
                    sum(correct(r0[i]) and not correct(final[i]) for i in range(3)) / 3
                )
            cost = sum(g["completion_tokens"] for g in r0)
            cost += sum(g["completion_tokens"] for g in by[(pid, rep, a)])
            if a in MESSAGES:
                cost += sum(g["completion_tokens"] for g in by[(pid, rep, MESSAGES[a])])
            tokens[a].append(cost)
    sample_cost = np.mean(
        [g["completion_tokens"] for g in gens if g["arm"] in ("round0", "sc_pool")]
    )

    def per_problem(d):
        return {p: float(np.mean(v)) for p, v in d.items()}

    accuracy = {a: per_problem(score[a]) for a in arms}
    k = {a: float(np.mean(tokens[a])) / float(sample_cost) for a in arms}
    matched = {
        a: {
            p: float(np.mean([interpolate(c, k[a]) for c in curves[p]])) for p in curves
        }
        for a in arms
    }
    gain = {a: {p: accuracy[a][p] - matched[a][p] for p in accuracy[a]} for a in arms}
    return {
        "arms": arms,
        "accuracy": accuracy,
        "tokens": {a: float(np.mean(tokens[a])) for a in arms},
        "k": k,
        "matched": matched,
        "gain": gain,
        "flips": {a: per_problem(flips[a]) for a in flips},
    }


def bootstrap(d, n_boot, seed=0):
    v = np.array([d[p] for p in sorted(d)])
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return v.mean(), np.quantile(means, [0.025, 0.975, 0.05, 0.95]), means


def holm(p):
    out, running = {}, 0.0
    for i, (name, value) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        running = max(running, min(1.0, (len(p) - i) * value))
        out[name] = running
    return out


def contrasts(q, n_boot, best="dissent_only"):
    def minus(a, b):
        return {p: a[p] - b[p] for p in a}

    diffs = {
        "full_solution_vs_sc": q["gain"]["full_solution"],
        "answer_only_vs_full_solution": minus(
            q["accuracy"]["answer_only"], q["accuracy"]["full_solution"]
        ),
        "placebo_vs_self_reflect_correct_to_wrong": minus(
            q["flips"]["placebo"], q["flips"]["self_reflect"]
        ),
        "failure_cause_vs_full_solution": minus(
            q["gain"]["failure_cause"], q["gain"]["full_solution"]
        ),
        "answer_conf_vs_full_solution": minus(
            q["gain"]["answer_conf"], q["gain"]["full_solution"]
        ),
        "best_low_bandwidth_vs_sc": q["gain"][best],
        "failure_cause_v2_vs_full_solution": minus(
            q["gain"]["failure_cause_v2"], q["gain"]["full_solution"]
        ),
    }
    out, raw = {}, {}
    for name, d in diffs.items():
        mean, (lo95, hi95, lo90, hi90), means = bootstrap(d, n_boot)
        below = (1 + (means <= 0).sum()) / (means.size + 1)
        above = (1 + (means >= 0).sum()) / (means.size + 1)
        p = below if name.startswith("placebo") else min(1.0, 2 * min(below, above))
        out[name] = {
            "mean": mean,
            "ci95_low": lo95,
            "ci95_high": hi95,
            "ci90_low": lo90,
            "ci90_high": hi90,
            "p": p,
        }
        if name in FAMILY:
            raw[name] = p
    for name, adjusted in holm(raw).items():
        out[name]["p_holm"] = adjusted
    return out
