"""M5 pilot gates G1 and G2, the arm table and the power inputs, on the mock."""

from __future__ import annotations

import pytest

from channel_audit.aggregate import plurality
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS
from channel_audit.gates import (
    G1_MIN,
    G2_MIN,
    LOW_BANDWIDTH,
    collect_units,
    g1_disagreement,
    g2_headroom,
    pilot_report,
    required_n,
)
from channel_audit.grade import is_correct
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import LazyBackend, run_shards

N_PROBLEMS = 24


def build(tmp_path, *, size=50, n=N_PROBLEMS, base_p=None, effects=None):
    problems = fake_problems(n)
    world = MockWorld.from_problems(problems, base_p=base_p)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        tmp_path,
        cfg,
        problems,
        LazyBackend(lambda: MockBackend(world, effects=effects)),
        size=size,
        log=lambda s: None,
    )
    return [open_run(d) for d in sorted(tmp_path.iterdir()) if d.is_dir()]


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    return build(tmp_path_factory.mktemp("pilot"))


@pytest.fixture(scope="module")
def units(pilot):
    return collect_units(pilot)


def test_one_unit_per_problem_and_replicate_with_everything_graded(units):
    assert len(units) == N_PROBLEMS * 2
    u = units[0]
    assert len(u.r0_answers) == 3 and len(u.sc_answers) == 3 + 6
    assert set(u.finals) == set(ARMS) and all(len(v) == 3 for v in u.finals.values())
    assert len(u.r0_correct) == 3 and isinstance(u.gold, str)


def test_the_thresholds_are_the_plans():
    assert G1_MIN == 0.15 and G2_MIN == 0.05
    assert set(LOW_BANDWIDTH) == {
        "answer_only",
        "answer_conf",
        "dissent_only",
        "failure_cause",
    }


def test_g1_is_the_share_of_replicates_where_round0_answers_differ(units):
    r = g1_disagreement(units)
    manual = sum(
        len({a for a in u.r0_answers if a is not None}) > 1 for u in units
    )  # integer answers: distinct strings are distinct values
    assert r["n"] == len(units)
    assert r["rate"] == pytest.approx(manual / len(units))
    assert r["passed"] is (r["rate"] >= G1_MIN)
    assert set(r["by_level"]) == {3, 4}


def test_g2_counts_replicates_where_the_right_answer_exists_but_loses(units):
    r = g2_headroom(units)
    manual = 0
    for u in units:
        if not any(u.r0_correct):
            continue
        winners = plurality(u.r0_answers).winners
        if not any(is_correct(w, u.gold) for w in winners):
            manual += 1
    assert r["rate"] == pytest.approx(manual / len(units))
    assert r["passed"] is (r["rate"] >= G2_MIN)


def test_everyone_right_means_no_disagreement_and_no_headroom(tmp_path):
    u = collect_units(build(tmp_path, n=12, base_p=1.0, effects={}))
    assert g1_disagreement(u)["rate"] == 0 and not g1_disagreement(u)["passed"]
    assert g2_headroom(u)["rate"] == 0


def test_everyone_wrong_means_disagreement_but_no_headroom(tmp_path):
    u = collect_units(build(tmp_path, n=12, base_p=0.0, effects={}))
    assert g1_disagreement(u)["passed"]
    assert g2_headroom(u)["rate"] == 0  # nothing correct to recover


def test_pooling_shards_gives_the_same_gates_as_one_run(tmp_path, units):
    sharded = build(tmp_path, size=8)
    assert len(sharded) > 1
    pooled = collect_units(sharded)
    assert g1_disagreement(pooled) == g1_disagreement(units)
    assert g2_headroom(pooled) == g2_headroom(units)


def test_required_sample_size_matches_the_plans_formula():
    assert required_n(0.2654, 0.03) == 614
    assert required_n(0.2654, 0.02) in (1380, 1381, 1382)
    assert required_n(0.1, 0.03) < required_n(0.2, 0.03)


def test_the_report_has_every_arm_and_every_primary_contrast(pilot):
    rep = pilot_report(pilot, confirm_available=3250)
    assert set(rep["arms"]) == set(ARMS)
    row = rep["arms"]["failure_cause"]
    assert {"accuracy", "tokens", "matched_k", "sc_accuracy", "gain"} <= set(row)
    assert set(rep["contrasts"]) == {
        "full_solution_vs_sc",
        "answer_only_vs_full_solution",
        "placebo_vs_self_reflect_correct_to_wrong",
        "failure_cause_vs_full_solution",
        "answer_conf_vs_full_solution",
        "best_low_bandwidth_vs_sc",
    }
    for c in rep["contrasts"].values():
        assert c["n_problems"] == N_PROBLEMS
        assert c["sd"] >= 0 and c["ci_low"] <= c["mean"] <= c["ci_high"]
        assert c["n_for_3_points"] >= 1 and c["n_for_2_points"] >= c["n_for_3_points"]
        assert c["n_for_3_points_capped"] <= 3250


def test_the_best_low_bandwidth_arm_is_chosen_from_the_pilot_by_gain_over_sc(pilot):
    rep = pilot_report(pilot, confirm_available=3250)
    best = rep["best_low_bandwidth_arm"]
    assert best in LOW_BANDWIDTH
    gains = {a: rep["arms"][a]["gain"] for a in LOW_BANDWIDTH}
    assert gains[best] == max(gains.values())
    assert rep["contrasts"]["best_low_bandwidth_vs_sc"]["mean"] == pytest.approx(
        gains[best]
    )


def test_planted_effects_are_recovered_in_the_arm_table(pilot):
    arms = pilot_report(pilot, confirm_available=3250)["arms"]
    assert arms["failure_cause"]["accuracy"] > arms["self_reflect"]["accuracy"]
    assert arms["placebo"]["accuracy"] < arms["answer_conf"]["accuracy"]


def test_a_strong_placebo_penalty_shows_up_as_more_correct_to_wrong(tmp_path):
    ctxs = build(
        tmp_path,
        n=12,
        base_p=0.6,
        effects={"placebo": -0.5, "self_reflect": 0.3},
    )
    contrast = pilot_report(ctxs, 3250)["contrasts"]
    assert contrast["placebo_vs_self_reflect_correct_to_wrong"]["mean"] > 0.2


def test_the_report_carries_gates_and_g3(pilot):
    rep = pilot_report(pilot, confirm_available=3250)
    assert set(rep["gates"]) >= {"G1", "G2", "G3"}
    assert rep["gates"]["G3"]["ok"] is True
    assert rep["n_problems"] == N_PROBLEMS and rep["replicates"] == 2
