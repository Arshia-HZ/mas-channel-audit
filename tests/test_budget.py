"""Token accounting and self-consistency matching."""

from __future__ import annotations

import pytest

from channel_audit.aggregate import plurality_credit
from channel_audit.budget import (
    BudgetOutOfRange,
    arm_costs,
    matched_sample_count,
    matched_sc_accuracy,
    sc_credit,
    sc_credit_interp,
    sc_sample_cost,
)
from conftest import make_generation


def g(arm, tokens, *, agent=0, sample_idx=0, prompt=10, pid="p", rep=0, round_=0):
    return make_generation(
        problem_id=pid,
        replicate=rep,
        arm=arm,
        agent=agent,
        sample_idx=sample_idx,
        round=round_,
        completion_tokens=tokens,
        prompt_tokens=prompt,
    )


def replicate_gens(pid="p", rep=0):
    out = [g("round0", 100, agent=a, pid=pid, rep=rep) for a in range(3)]
    out += [
        g("sc_pool", 100, agent=-1, sample_idx=i, pid=pid, rep=rep) for i in range(6)
    ]
    out += [
        g("self_reflect", 50, agent=a, pid=pid, rep=rep, round_=1) for a in range(3)
    ]
    out += [
        g("full_solution", 150, agent=a, prompt=400, pid=pid, rep=rep, round_=1)
        for a in range(3)
    ]
    out += [
        g("failure_msg", 20, agent=a, sample_idx=0, pid=pid, rep=rep, round_=1)
        for a in range(3)
    ]
    out += [g("failure_msg", 20, agent=0, sample_idx=1, pid=pid, rep=rep, round_=1)]
    out += [
        g("failure_cause", 60, agent=a, pid=pid, rep=rep, round_=1) for a in range(3)
    ]
    return out


def test_arm_cost_is_round0_plus_the_arms_own_generations():
    costs = arm_costs(replicate_gens())
    assert costs["self_reflect"].generated_tokens == 300 + 150
    assert costs["full_solution"].generated_tokens == 300 + 450


def test_failure_cause_carries_all_failure_message_attempts_including_rejected():
    costs = arm_costs(replicate_gens())
    assert costs["failure_cause"].generated_tokens == 300 + 4 * 20 + 180


def test_other_arms_do_not_pay_for_failure_messages():
    costs = arm_costs(replicate_gens())
    assert costs["self_reflect"].generated_tokens == 450


def test_prompt_tokens_are_the_sensitivity_view():
    costs = arm_costs(replicate_gens())
    # full_solution: 3 round-0 prompts of 10, 3 revision prompts of 400
    assert costs["full_solution"].total_tokens == 750 + 3 * 10 + 3 * 400


def test_arm_cost_is_a_mean_over_replicates():
    gens = (
        replicate_gens(rep=0)
        + [g("round0", 200, agent=a, rep=1) for a in range(3)]
        + [g("self_reflect", 100, agent=a, rep=1, round_=1) for a in range(3)]
    )
    assert arm_costs(gens)["self_reflect"].generated_tokens == pytest.approx(
        (450 + 900) / 2
    )
    assert arm_costs(gens)["self_reflect"].n_replicates == 2


def test_arm_with_no_generations_costs_just_round0():
    gens = [g("round0", 100, agent=a) for a in range(3)]
    assert (
        arm_costs(gens, arms=["dissent_only"])["dissent_only"].generated_tokens == 300
    )


def test_sc_sample_cost_is_the_mean_over_round0_and_pool():
    gens = [g("round0", 100, agent=a) for a in range(3)] + [
        g("sc_pool", 400, agent=-1, sample_idx=i) for i in range(3)
    ]
    gen_cost, total_cost = sc_sample_cost(gens)
    assert gen_cost == 250
    assert total_cost == 260


def test_matched_sample_count_and_range():
    assert matched_sample_count(750, 100, max_samples=9) == 7.5
    with pytest.raises(BudgetOutOfRange):
        matched_sample_count(1000, 100, max_samples=9)
    with pytest.raises(BudgetOutOfRange):
        matched_sample_count(50, 100, max_samples=9)


def test_sc_credit_over_all_samples_equals_plurality_credit():
    answers = ["4", "4", "5", None, "6", "5", "4", "7", "5"]
    assert sc_credit(answers, "4", 9) == pytest.approx(plurality_credit(answers, "4"))
    assert sc_credit(answers, "5", 9) == pytest.approx(plurality_credit(answers, "5"))


def test_sc_credit_k1_is_the_single_sample_accuracy():
    answers = ["4", "4", "5", None]
    assert sc_credit(answers, "4", 1) == pytest.approx(2 / 4)


def test_sc_credit_averages_over_every_subset_with_tie_credit():
    # k=2 of [4, 5, 5]: {4,5} tie -> 1/2 (twice), {5,5} -> 0   => (0.5+0.5+0)/3
    assert sc_credit(["4", "5", "5"], "4", 2) == pytest.approx(1 / 3)


def test_sc_credit_unparsable_samples_never_vote():
    assert sc_credit([None, None, "4"], "4", 2) == pytest.approx(2 / 3)


def test_sc_credit_interpolates_linearly_between_integer_counts():
    answers = ["4", "4", "5", "5", "4", "6"]
    lo, hi = sc_credit(answers, "4", 3), sc_credit(answers, "4", 4)
    assert sc_credit_interp(answers, "4", 3.25) == pytest.approx(0.75 * lo + 0.25 * hi)
    assert sc_credit_interp(answers, "4", 3.0) == pytest.approx(lo)
    assert sc_credit_interp(answers, "4", 4.0) == pytest.approx(hi)


def test_sc_credit_rejects_bad_k():
    with pytest.raises(ValueError):
        sc_credit(["4"], "4", 0)
    with pytest.raises(ValueError):
        sc_credit(["4"], "4", 2)
    with pytest.raises(BudgetOutOfRange):
        sc_credit_interp(["4", "4"], "4", 2.5)


def test_matched_sc_accuracy_is_the_mean_over_replicates():
    items = [(["4", "4", "4"], "4"), (["5", "5", "5"], "4")]
    assert matched_sc_accuracy(items, 3.0) == pytest.approx(0.5)
