"""M7: the six primary contrasts, their decision rules and the sensitivity runs.

Everything runs on the mock backend, whose arm effects are planted, so the answer
is known before any confirmatory data is looked at.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from channel_audit.analysis import contrasts as C
from channel_audit.analysis.stats import bootstrap_means, paired_bootstrap_ci
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.budget import BudgetOutOfRange, sc_credit_interp
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.gates import collect_units, pilot_report
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import LazyBackend, run_shards

PRIMARY = (
    "full_solution_vs_sc",
    "answer_only_vs_full_solution",
    "placebo_vs_self_reflect_correct_to_wrong",
    "failure_cause_vs_full_solution",
    "answer_conf_vs_full_solution",
    "best_low_bandwidth_vs_sc",
)
FAMILY = (
    "placebo_vs_self_reflect_correct_to_wrong",
    "failure_cause_vs_full_solution",
    "answer_conf_vs_full_solution",
    "best_low_bandwidth_vs_sc",
)
PLANTED = {
    "self_reflect": 0.0,
    "answer_only": 0.0,
    "answer_conf": 0.0,
    "dissent_only": 0.0,
    "failure_cause": 0.35,
    "failure_cause_v2": 0.0,
    "full_solution": -0.2,
    "placebo": -0.35,
}


def build(tmp_path, *, n=60, size=30, effects=None, base_p=0.45, max_tokens=512):
    problems = fake_problems(n)
    world = MockWorld.from_problems(problems, base_p=base_p)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=max_tokens)
    run_shards(
        tmp_path,
        cfg,
        problems,
        LazyBackend(lambda: MockBackend(world, effects=effects)),
        size=size,
        arms=(*ARMS, *VARIANT_ARMS),
        log=lambda s: None,
    )
    return [open_run(d) for d in sorted(tmp_path.iterdir()) if d.is_dir()]


@pytest.fixture(scope="module")
def ctxs(tmp_path_factory):
    return build(tmp_path_factory.mktemp("planted"), effects=PLANTED)


@pytest.fixture(scope="module")
def gens(ctxs):
    return [g for ctx in ctxs for g in ctx.gens]


@pytest.fixture(scope="module")
def stats(ctxs):
    return C.unit_stats(
        collect_units(ctxs, (*ARMS, *VARIANT_ARMS)), (*ARMS, *VARIANT_ARMS)
    )


@pytest.fixture(scope="module")
def main(stats, gens):
    return C.run_contrasts(stats, gens, n_pool=6, n_boot=2000)


# ---- the plan's constants -------------------------------------------------


def test_the_constants_are_the_preregistered_ones():
    assert C.N_BOOT == 10_000
    assert C.EQUIV_MARGIN == 0.02 and C.NONINF_MARGIN == -0.015
    assert C.HOLM_ALPHA == 0.05
    assert C.BEST_LOW_BANDWIDTH == "dissent_only"  # chosen on the pilot (PREREG 8)
    assert set(C.FAMILY) == set(FAMILY)


# ---- Holm -----------------------------------------------------------------


def test_holm_matches_a_hand_computation():
    adj = C.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.2})
    assert adj == pytest.approx({"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.2})


def test_holm_is_capped_at_one_and_a_single_test_is_unchanged():
    assert C.holm_adjust({"a": 0.9, "b": 0.8}) == {"a": 1.0, "b": 1.0}
    assert C.holm_adjust({"only": 0.03}) == {"only": 0.03}


def test_holm_never_lowers_a_p_value_and_ignores_input_order():
    p = {"a": 0.2, "b": 0.001, "c": 0.04, "d": 0.5}
    adj = C.holm_adjust(p)
    assert all(adj[k] >= p[k] for k in p)
    assert adj == C.holm_adjust(dict(reversed(list(p.items()))))
    ranked = sorted(p, key=p.get)
    assert [adj[k] for k in ranked] == sorted(adj[k] for k in ranked)


# ---- bootstrap p-values and intervals --------------------------------------


def test_bootstrap_means_reproduce_the_ported_interval():
    d = np.random.default_rng(0).normal(0.05, 0.2, 80)
    means = bootstrap_means(d, n_boot=3000, seed=4)
    lo, hi = np.quantile(means, [0.025, 0.975])
    assert paired_bootstrap_ci(d, n_boot=3000, seed=4) == pytest.approx(
        (d.mean(), lo, hi)
    )


def test_a_clear_positive_effect_has_a_tiny_p_in_the_right_direction_only():
    means = bootstrap_means(
        np.full(60, 0.1) + np.random.default_rng(1).normal(0, 0.05, 60), 2000, 0
    )
    assert C.bootstrap_p(means, "greater") < 0.01
    assert C.bootstrap_p(means, "two-sided") < 0.01
    assert C.bootstrap_p(-means, "greater") > 0.99  # the wrong direction


def test_pure_noise_is_not_significant_and_p_is_never_zero():
    noise = np.random.default_rng(2).normal(0, 1, 100)
    assert C.bootstrap_p(bootstrap_means(noise, 2000, 0), "two-sided") > 0.05
    sure = bootstrap_means(np.full(30, 0.3), 500, 0)
    assert C.bootstrap_p(sure, "greater") == pytest.approx(1 / 501)
    assert C.bootstrap_p(sure, "two-sided") == pytest.approx(2 / 501)


def test_two_sided_p_never_exceeds_one():
    zero = bootstrap_means(np.zeros(20), 200, 0)
    assert C.bootstrap_p(zero, "two-sided") == 1.0
    with pytest.raises(ValueError):
        C.bootstrap_p(zero, "less")


def test_summarize_reports_nested_intervals_and_the_mean():
    d = {
        f"p{i}": v for i, v in enumerate(np.random.default_rng(3).normal(0.02, 0.2, 90))
    }
    s = C.summarize(d, "two-sided", n_boot=2000, seed=0)
    assert s["n_problems"] == 90
    assert s["mean"] == pytest.approx(np.mean(list(d.values())))
    assert s["ci95_low"] < s["ci90_low"] < s["mean"] < s["ci90_high"] < s["ci95_high"]
    assert 0 < s["p"] <= 1


# ---- the two interval rules -------------------------------------------------


def test_equivalence_needs_the_whole_90_percent_interval_inside_two_points():
    assert C.is_equivalent(-0.019, 0.015)
    assert C.is_equivalent(-0.02, 0.02)  # on the margin counts as within
    assert not C.is_equivalent(-0.021, 0.01)
    assert not C.is_equivalent(-0.01, 0.025)


def test_non_inferiority_needs_the_lower_bound_strictly_above_minus_1_5_points():
    assert C.is_noninferior(-0.014)
    assert not C.is_noninferior(-0.015)
    assert not C.is_noninferior(-0.03)


# ---- per-problem quantities --------------------------------------------------


def test_the_cached_sc_curve_equals_the_budget_module_at_any_sample_count(ctxs, stats):
    units = collect_units(ctxs, (*ARMS, *VARIANT_ARMS))
    for u, s in list(zip(units, stats, strict=True))[::7]:
        assert (u.problem_id, u.replicate) == (s.problem_id, s.replicate)
        for k in (1, 2.5, 3, 5.26, 7.9, 9):
            assert s.sc_at(k) == pytest.approx(
                sc_credit_interp(u.sc_answers, u.gold, k)
            )


def test_sc_at_rejects_a_count_outside_what_was_drawn(stats):
    with pytest.raises(BudgetOutOfRange):
        stats[0].sc_at(9.5)


def test_correct_to_wrong_counts_agents_right_in_round_0_and_wrong_after(ctxs, stats):
    units = collect_units(ctxs, (*ARMS, *VARIANT_ARMS))
    for u, s in list(zip(units, stats, strict=True))[::5]:
        manual = sum(
            was and not now
            for was, now in zip(u.r0_correct, u.final_correct["placebo"], strict=True)
        )
        assert s.c2w["placebo"] == pytest.approx(manual / 3)


# ---- run_contrasts ------------------------------------------------------------


def test_every_primary_contrast_is_reported_with_the_variant_beside_it(main):
    assert tuple(main["contrasts"]) == PRIMARY
    assert set(main["secondary"]) == {"failure_cause_v2_vs_full_solution"}
    assert main["n_problems"] == 60 and main["basis"] == "generated"
    for row in (*main["contrasts"].values(), *main["secondary"].values()):
        assert row["ci95_low"] <= row["mean"] <= row["ci95_high"]
        assert row["rule"] and row["decision"]


def test_only_contrasts_three_to_six_are_in_the_holm_family(main):
    for name, row in main["contrasts"].items():
        assert ("p_holm" in row) is (name in FAMILY), name
    assert "p_holm" not in main["secondary"]["failure_cause_v2_vs_full_solution"]
    raw = {n: main["contrasts"][n]["p"] for n in FAMILY}
    adj = C.holm_adjust(raw)
    for n in FAMILY:
        assert main["contrasts"][n]["p_holm"] == pytest.approx(adj[n])
        assert main["contrasts"][n]["reject"] is (adj[n] < 0.05)


def test_the_directions_follow_the_preregistered_alternatives(main):
    c = main["contrasts"]
    assert c["placebo_vs_self_reflect_correct_to_wrong"]["alternative"] == "greater"
    for n in (
        "failure_cause_vs_full_solution",
        "answer_conf_vs_full_solution",
        "best_low_bandwidth_vs_sc",
    ):
        assert c[n]["alternative"] == "two-sided"
    assert c["full_solution_vs_sc"]["alternative"] is None  # interval rule only
    assert c["answer_only_vs_full_solution"]["alternative"] is None


def test_the_interval_rules_use_the_90_and_95_percent_intervals(main):
    c1 = main["contrasts"]["full_solution_vs_sc"]
    assert (c1["decision"] == "equivalent") is C.is_equivalent(
        c1["ci90_low"], c1["ci90_high"]
    )
    c2 = main["contrasts"]["answer_only_vs_full_solution"]
    assert (c2["decision"] == "non-inferior") is C.is_noninferior(c2["ci95_low"])


def test_planted_effects_are_recovered(main):
    c = main["contrasts"]
    # failure_cause +0.35 against full_solution -0.2: a large positive gain gap
    fc = c["failure_cause_vs_full_solution"]
    assert fc["mean"] > 0.2 and fc["reject"] is True
    # placebo -0.35 against self_reflect 0: many more correct-to-wrong flips
    pl = c["placebo_vs_self_reflect_correct_to_wrong"]
    assert pl["mean"] > 0.1 and pl["reject"] is True
    # answer_only (0) is far above full_solution (-0.2): non-inferior, not equivalent
    assert c["answer_only_vs_full_solution"]["decision"] == "non-inferior"
    assert c["full_solution_vs_sc"]["decision"] != "equivalent"
    assert c["full_solution_vs_sc"]["mean"] < -0.1
    # answer_conf has no planted effect, the same as full_solution minus 0.2: positive
    assert c["answer_conf_vs_full_solution"]["mean"] > 0.1
    # the arm table carries a budget for every arm
    for arm in (*ARMS, *VARIANT_ARMS):
        row = main["arms"][arm]
        assert row["estimable"] and 1 <= row["matched_k"] <= 9
        assert row["gain_ci95"][0] <= row["gain"] <= row["gain_ci95"][1]


def test_the_best_low_bandwidth_arm_is_the_preregistered_one_not_re_chosen(stats, gens):
    out = C.run_contrasts(stats, gens, n_pool=6, best_arm="answer_only", n_boot=500)
    assert out["best_low_bandwidth_arm"] == "answer_only"
    default = C.run_contrasts(stats, gens, n_pool=6, n_boot=500)
    assert default["best_low_bandwidth_arm"] == "dissent_only"
    assert default["arms"]["dissent_only"]["gain"] == pytest.approx(
        default["contrasts"]["best_low_bandwidth_vs_sc"]["mean"]
    )


def test_it_agrees_with_the_pilot_report_on_the_same_data(ctxs):
    ref = pilot_report(ctxs, confirm_available=3250)
    full = C.run_contrasts(
        C.unit_stats(
            collect_units(ctxs, (*ARMS, *VARIANT_ARMS)), (*ARMS, *VARIANT_ARMS)
        ),
        [g for ctx in ctxs for g in ctx.gens],
        n_pool=6,
        best_arm=ref["best_low_bandwidth_arm"],  # the pilot report picks it by gain
    )
    for name in PRIMARY:
        assert full["contrasts"][name]["mean"] == pytest.approx(
            ref["contrasts"][name]["mean"]
        )
        assert full["contrasts"][name]["ci95_low"] == pytest.approx(
            ref["contrasts"][name]["ci_low"]
        )
        assert full["contrasts"][name]["ci95_high"] == pytest.approx(
            ref["contrasts"][name]["ci_high"]
        )
    for arm in (*ARMS, *VARIANT_ARMS):
        assert full["arms"][arm]["accuracy"] == pytest.approx(
            ref["arms"][arm]["accuracy"]
        )
        assert full["arms"][arm]["gain"] == pytest.approx(ref["arms"][arm]["gain"])
        assert full["arms"][arm]["matched_k"] == pytest.approx(
            ref["arms"][arm]["matched_k"]
        )


def test_same_inputs_give_the_same_numbers(stats, gens):
    a = C.run_contrasts(stats, gens, n_pool=6, n_boot=500)
    b = C.run_contrasts(stats, gens, n_pool=6, n_boot=500)
    assert a == b


def test_a_problem_subset_is_analysed_on_its_own_budget(stats, gens):
    keep = {s.problem_id for s in stats if s.level == 3}
    sub = C.run_contrasts(stats, gens, n_pool=6, n_boot=500, problem_ids=keep)
    assert sub["n_problems"] == len(keep)
    whole = C.run_contrasts(stats, gens, n_pool=6, n_boot=500)
    assert (
        sub["arms"]["full_solution"]["tokens"]
        != whole["arms"]["full_solution"]["tokens"]
    )


def test_an_empty_subset_is_an_error_not_a_silent_result(stats, gens):
    with pytest.raises(ValueError):
        C.run_contrasts(stats, gens, n_pool=6, n_boot=100, problem_ids=set())


def test_prompt_tokens_in_the_budget_can_push_an_arm_past_the_pool(stats, gens):
    heavy = [
        g.model_copy(update={"prompt_tokens": 60_000})
        if g.arm == "full_solution"
        else g
        for g in gens
    ]
    with pytest.raises(BudgetOutOfRange):
        C.run_contrasts(
            stats, heavy, n_pool=6, n_boot=100, basis="generated_plus_prompt"
        )
    out = C.run_contrasts(
        stats,
        heavy,
        n_pool=6,
        n_boot=300,
        basis="generated_plus_prompt",
        allow_out_of_range=True,
    )
    assert out["arms"]["full_solution"]["estimable"] is False
    assert out["contrasts"]["full_solution_vs_sc"]["decision"] == "not estimable"
    assert "p_holm" not in out["contrasts"]["full_solution_vs_sc"]
    assert out["arms"]["answer_only"]["estimable"] is True
    assert (
        out["contrasts"]["failure_cause_vs_full_solution"]["decision"]
        == "not estimable"
    )
    ok = out["contrasts"]["placebo_vs_self_reflect_correct_to_wrong"]
    assert ok["decision"] != "not estimable" and "p_holm" in ok


def test_an_unknown_basis_is_rejected(stats, gens):
    with pytest.raises(ValueError):
        C.run_contrasts(stats, gens, n_pool=6, n_boot=100, basis="vibes")


# ---- truncation sensitivity ----------------------------------------------------


def test_truncated_problems_lists_those_with_a_cut_off_generation_in_the_arms(gens):
    cut = {
        g.gen_key
        for g in gens
        if g.arm == "round0" and g.problem_id.endswith(("0", "1"))
    }
    tweaked = [
        g.model_copy(update={"finish_reason": "length"}) if g.gen_key in cut else g
        for g in gens
    ]
    got = C.truncated_problems(tweaked, ("round0",))
    assert got == {g.problem_id for g in gens if g.gen_key in cut}
    assert C.truncated_problems(gens, ("round0",)) == set()
    assert C.truncated_problems(tweaked, ("answer_only",)) == set()


def test_failure_messages_count_as_part_of_a_failure_cause_arm(gens):
    one = next(g for g in gens if g.arm == "failure_msg")
    tweaked = [
        g.model_copy(update={"finish_reason": "length"})
        if g.gen_key == one.gen_key
        else g
        for g in gens
    ]
    assert C.truncated_problems(tweaked, ("failure_cause",)) == {one.problem_id}
    assert C.truncated_problems(tweaked, ("answer_only",)) == set()
    v2 = next(g for g in gens if g.arm == "failure_msg_v2")
    tweaked = [
        g.model_copy(update={"finish_reason": "length"})
        if g.gen_key == v2.gen_key
        else g
        for g in gens
    ]
    assert C.truncated_problems(tweaked, ("failure_cause_v2",)) == {v2.problem_id}


def test_exclusion_sets_follow_the_preregistered_rules(gens):
    sr = next(g for g in gens if g.arm == "self_reflect")
    ac = next(
        g for g in gens if g.arm == "answer_conf" and g.problem_id != sr.problem_id
    )
    r0 = next(
        g
        for g in gens
        if g.arm == "round0" and g.problem_id not in (sr.problem_id, ac.problem_id)
    )
    cut = {sr.gen_key, ac.gen_key, r0.gen_key}
    tweaked = [
        g.model_copy(update={"finish_reason": "length"}) if g.gen_key in cut else g
        for g in gens
    ]
    sets = C.exclusion_sets(tweaked, flagged_arms=["answer_conf"])
    assert sets["no_round0_truncation"] == {r0.problem_id}
    assert sets["no_flagged_arm_truncation"] == {sr.problem_id, ac.problem_id}
    # self_reflect is the reference: when it truncates, the affected problems go too
    assert "no_flagged_arm_truncation" not in C.exclusion_sets(tweaked, flagged_arms=[])


# ---- analyze --------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(ctxs):
    return C.analyze(ctxs, n_boot=500)


def test_analyze_has_the_main_result_levels_sensitivity_gates_and_quality(report):
    assert set(report["analyses"]) >= {
        "main",
        "level_3",
        "level_4",
        "prompt_tokens_in_budget",
    }
    assert report["analyses"]["main"]["n_problems"] == 60
    assert (
        report["analyses"]["level_3"]["n_problems"]
        + report["analyses"]["level_4"]["n_problems"]
        == 60
    )
    assert set(report["gates"]) == {"G1", "G2", "G3"}
    assert report["gates"]["G1"]["by_level"].keys() == {3, 4}
    assert report["quality"]["arms"]["round0"]["n"] == 60 * 2 * 3
    assert report["model_id"] == "mock/model" and report["replicates"] == 2
    assert "sc_curve" in report and len(report["sc_curve"]) == 9


def test_analyze_labels_the_exploratory_parts(report):
    assert report["status"]["primary"] == ["main"]
    assert "level_3" in report["status"]["preregistered_secondary"]
    assert "prompt_tokens_in_budget" in report["status"]["preregistered_secondary"]
    assert report["status"]["exploratory"] == []


def test_the_report_text_names_every_contrast_and_its_decision(report):
    text = C.format_analysis_report(report)
    for name in PRIMARY:
        assert name in text
    assert "Holm" in text and "failure_cause_v2_vs_full_solution" in text
    assert "G3" in text and "truncation" in text.lower()


def test_round_0_truncation_sensitivity_drops_only_problems_with_a_cut_round_0(
    tmp_path,
):
    ctxs = build(tmp_path, n=40, size=20, effects=PLANTED, max_tokens=75)
    rep = C.analyze(ctxs, n_boot=200)
    gens = [g for ctx in ctxs for g in ctx.gens]
    drop = C.truncated_problems(gens, ("round0",))
    assert 0 < len(drop) < 40
    sens = rep["analyses"]["no_round0_truncation"]
    assert sens["n_problems"] == 40 - len(drop)
    assert rep["analyses"]["main"]["n_problems"] == 40


def test_flagged_arm_sensitivity_appears_only_when_an_arm_is_flagged(report):
    assert report["gates"]["G3"]["flagged_arms"] == []
    assert "no_flagged_arm_truncation" not in report["analyses"]
    assert report["status"]["not_triggered"] == ["no_flagged_arm_truncation"]


# ---- guards found in review -------------------------------------------------------


def test_a_non_finite_difference_is_an_error_not_a_tiny_p_value():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            C.summarize({"a": bad, "b": 0.1, "c": 0.2}, "two-sided", n_boot=50)


def _with_truncation(ctxs, tmp_path, predicate):
    """Copies of the shards where the generations matching ``predicate`` are cut off."""
    out = []
    for ctx in ctxs:
        dst = tmp_path / ctx.out_dir.name
        shutil.copytree(ctx.out_dir, dst)
        path = dst / "generations.jsonl"
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        for row in rows:
            if predicate(row):
                row["finish_reason"] = "length"
        lines = [json.dumps(r) for r in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out.append(open_run(dst))
    return out


def test_a_flagged_arm_runs_its_sensitivity_analysis_end_to_end(ctxs, tmp_path):
    first = {f"fake-{i:04d}" for i in range(6)}
    cut = _with_truncation(
        ctxs,
        tmp_path,
        lambda r: r["arm"] == "self_reflect" and r["problem_id"] in first,
    )
    rep = C.analyze(cut, n_boot=200)
    assert set(rep["gates"]["G3"]["flagged_arms"]) >= {"answer_only", "placebo"}
    sens = rep["analyses"]["no_flagged_arm_truncation"]
    assert sens["n_problems"] == 60 - 6
    assert rep["analyses"]["no_round0_truncation"]["n_problems"] == 60
    assert rep["status"]["not_triggered"] == []
    assert "no_flagged_arm_truncation" in rep["status"]["preregistered_secondary"]


def test_an_analysis_with_no_problems_left_is_reported_not_skipped_silently(
    ctxs, tmp_path
):
    cut = _with_truncation(
        ctxs, tmp_path, lambda r: r["arm"] in ("self_reflect", "round0")
    )
    rep = C.analyze(cut, n_boot=100)
    assert "no_round0_truncation" not in rep["analyses"]
    assert "no_flagged_arm_truncation" not in rep["analyses"]
    assert rep["status"]["empty"] == [
        "no_round0_truncation",
        "no_flagged_arm_truncation",
    ]
    text = C.format_analysis_report(rep)
    assert "every problem excluded" in text and "no_round0_truncation" in text


def test_a_normal_analysis_has_nothing_empty(report):
    assert report["status"]["empty"] == []
