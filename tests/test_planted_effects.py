"""The mock plants known arm effects; the pipeline must recover them.

Intervals are 99% so a chance miss on a null arm is rare (about 1% per arm)."""

from __future__ import annotations

import numpy as np
import pytest

from channel_audit.analysis.stats import paired_bootstrap_ci
from channel_audit.budget import arm_costs, matched_sample_count, matched_sc_accuracy
from channel_audit.runner import final_generations, grade_pending, run_all
from helpers import build_ctx

EFFECTS = {
    "self_reflect": 0.0,
    "answer_only": 0.0,
    "failure_cause": 0.20,
    "placebo": -0.15,
}
ARMS = list(EFFECTS)


def per_problem_accuracy(ctx, arm, problem, reps):
    vals = []
    for rep in range(reps):
        for g in final_generations(ctx, arm, problem.problem_id, rep):
            vals.append(ctx.graded.get(g.gen_key).correct)
    return float(np.mean(vals))


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    ctx = build_ctx(
        tmp_path_factory.mktemp("planted"),
        n_problems=80,
        replicates=2,
        effects=EFFECTS,
        base_p=0.45,
    )
    run_all(ctx, arms=ARMS)
    grade_pending(ctx)
    return ctx


def _accuracy(ctx, arm):
    return np.array([per_problem_accuracy(ctx, arm, p, 2) for p in ctx.problems])


def test_planted_effects_are_recovered_within_their_intervals(ctx):
    base = _accuracy(ctx, "self_reflect")
    for arm, planted in EFFECTS.items():
        if arm == "self_reflect":
            continue
        mean, lo, hi = paired_bootstrap_ci(
            _accuracy(ctx, arm) - base, n_boot=4000, alpha=0.01, seed=0
        )
        assert lo <= planted <= hi, (arm, planted, (mean, lo, hi))


def test_real_effects_exclude_zero_and_the_null_arm_does_not(ctx):
    base = _accuracy(ctx, "self_reflect")

    def ci(arm):
        return paired_bootstrap_ci(
            _accuracy(ctx, arm) - base, n_boot=4000, alpha=0.01, seed=0
        )

    assert ci("failure_cause")[1] > 0
    assert ci("placebo")[2] < 0
    _, lo, hi = ci("answer_only")
    assert lo < 0 < hi


def test_budget_matching_runs_on_the_graded_mock_run(ctx):
    costs = arm_costs(ctx.gens)
    pool = [g for g in ctx.gens if g.arm in ("round0", "sc_pool")]
    per_sample = sum(g.completion_tokens for g in pool) / len(pool)
    k = matched_sample_count(costs["failure_cause"].generated_tokens, per_sample, 9)
    assert 3 < k < 9
    items = []
    for p in ctx.problems:
        for rep in range(2):
            gens = [
                g
                for g in ctx.gens
                if g.problem_id == p.problem_id
                and g.replicate == rep
                and g.arm in ("round0", "sc_pool")
            ]
            gens.sort(key=lambda g: (g.arm != "round0", g.agent, g.sample_idx))
            items.append(([ctx.graded.get(g.gen_key).extracted for g in gens], p.gold))
    assert 0.0 < matched_sc_accuracy(items, k) <= 1.0
