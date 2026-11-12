"""Exploratory decomposition (PREREG 10.2 item 9): change types, source of change,
diversity, headroom recovery and conformity.

Hand-built units first, so every expected number can be checked by hand; then the
mock pipeline, whose planted effects have a known direction.
"""

from __future__ import annotations

import pytest

from channel_audit.analysis import decompose as D
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.gates import Unit, collect_units
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import LazyBackend, run_shards


def unit(pid, r0, r0_ok, finals, final_ok, gold="1", rep=0, level=3):
    return Unit(
        problem_id=pid,
        replicate=rep,
        subject="Algebra",
        level=level,
        gold=gold,
        r0_answers=r0,
        r0_correct=r0_ok,
        sc_answers=[*r0, *["9"] * 6],
        finals=finals,
        final_correct=final_ok,
    )


def approx(x):
    return pytest.approx(x, abs=1e-12)


# ---- classify_change ---------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "before_ok", "after", "after_ok", "label"),
    [
        ("1", True, "1", True, "unchanged_correct"),
        ("2", False, "1", True, "wrong_to_correct"),
        ("1", True, "2", False, "correct_to_wrong"),
        ("2", False, "2", False, "unchanged_wrong"),
        ("2", False, "3", False, "wrong_to_other_wrong"),
        (None, False, None, False, "unchanged_wrong"),
        (None, False, "1", True, "wrong_to_correct"),
        ("2", False, None, False, "wrong_to_other_wrong"),
        ("2", False, "  ", False, "wrong_to_other_wrong"),
        (None, False, " ", False, "unchanged_wrong"),
    ],
)
def test_classify_change(before, before_ok, after, after_ok, label):
    assert D.classify_change(before, before_ok, after, after_ok) == label


def test_equivalent_spellings_of_a_wrong_answer_are_unchanged():
    assert D.classify_change(r"\frac{1}{2}", False, "0.5", False) == "unchanged_wrong"


def test_the_five_types_are_exhaustive_and_named_as_in_the_plan():
    assert D.CHANGE_TYPES == (
        "wrong_to_correct",
        "correct_to_wrong",
        "wrong_to_other_wrong",
        "unchanged_correct",
        "unchanged_wrong",
    )


def test_agent_changes_follow_the_agents_not_the_answers():
    u = unit(
        "p",
        ["1", "2", "3"],
        [True, False, False],
        {
            "a": ["1", "1", "3"],
            "b": ["2", "2", None],
        },
        {"a": [True, True, False], "b": [False, False, False]},
    )
    assert D.agent_changes(u, "a") == [
        "unchanged_correct",
        "wrong_to_correct",
        "unchanged_wrong",
    ]
    assert D.agent_changes(u, "b") == [
        "correct_to_wrong",
        "unchanged_wrong",
        "wrong_to_other_wrong",
    ]


# ---- change_table ------------------------------------------------------------


def changes_units():
    # p1 has two replicates: shares are averaged within the problem first
    r0 = ["2", "2", "1"]
    ok = [False, False, True]
    return [
        unit("p1", r0, ok, {"x": ["1", "2", "1"]}, {"x": [True, False, True]}, rep=0),
        unit("p1", r0, ok, {"x": ["2", "2", "2"]}, {"x": [False, False, False]}, rep=1),
        unit("p2", r0, ok, {"x": ["2", "2", "1"]}, {"x": [False, False, True]}, rep=0),
    ]


def test_shares_are_averaged_within_problem_then_across_problems():
    t = D.change_table(changes_units(), ["x"], n_boot=200, seed=0)["x"]
    s = {k: v["mean"] for k, v in t["shares"].items()}
    # p1: rep0 = w2c, uw, uc ; rep1 = uw, uw, c2w   p2: uw, uw, uc
    assert s["wrong_to_correct"] == approx((1 / 6 + 0) / 2)
    assert s["correct_to_wrong"] == approx((1 / 6 + 0) / 2)
    assert s["unchanged_correct"] == approx((1 / 6 + 1 / 3) / 2)
    assert s["unchanged_wrong"] == approx((3 / 6 + 2 / 3) / 2)
    assert s["wrong_to_other_wrong"] == approx(0.0)
    assert sum(s.values()) == approx(1.0)
    assert t["net"]["mean"] == approx(0.0)
    assert t["shares"]["wrong_to_correct"]["n_problems"] == 2


def test_the_interval_brackets_the_mean():
    t = D.change_table(changes_units(), ["x"], n_boot=500, seed=1)["x"]
    for ci in (*t["shares"].values(), t["net"]):
        assert ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"]


# ---- source_decomposition ----------------------------------------------------


def source_units():
    r0 = ["2", "2", "1"]
    ok = [False, False, True]
    finals = {
        "self_reflect": ["1", "2", "1"],  # one wrong-to-correct
        "placebo": ["2", "2", "2"],  # one correct-to-wrong
        "x": ["1", "1", "1"],  # two wrong-to-correct
    }
    final_ok = {
        "self_reflect": [True, False, True],
        "placebo": [False, False, False],
        "x": [True, True, True],
    }
    return [unit("p", r0, ok, finals, final_ok)]


def test_source_of_change_splits_an_arm_into_spontaneous_form_and_content():
    src = D.source_decomposition(
        source_units(), ["self_reflect", "placebo", "x"], 100, 0
    )
    w2c = src["wrong_to_correct"]["x"]
    assert w2c["total"]["mean"] == approx(2 / 3)
    assert w2c["spontaneous"]["mean"] == approx(1 / 3)  # self_reflect
    assert w2c["form"]["mean"] == approx(0 - 1 / 3)  # placebo minus self_reflect
    assert w2c["content"]["mean"] == approx(2 / 3 - 0)  # arm minus placebo
    c2w = src["correct_to_wrong"]["x"]
    assert (c2w["spontaneous"]["mean"], c2w["form"]["mean"]) == (
        approx(0),
        approx(1 / 3),
    )
    assert c2w["content"]["mean"] == approx(-1 / 3)
    net = src["net"]["x"]
    assert net["spontaneous"]["mean"] == approx(1 / 3)
    assert net["form"]["mean"] == approx(-2 / 3)
    assert net["content"]["mean"] == approx(1.0)


def test_the_three_sources_always_add_up_to_the_total():
    src = D.source_decomposition(
        source_units(), ["self_reflect", "placebo", "x"], 100, 0
    )
    for metric in src.values():
        for row in metric.values():
            parts = sum(row[k]["mean"] for k in ("spontaneous", "form", "content"))
            assert parts == approx(row["total"]["mean"])


def test_placebo_has_no_content_part_and_self_reflect_is_the_baseline_not_a_row():
    src = D.source_decomposition(
        source_units(), ["self_reflect", "placebo", "x"], 100, 0
    )
    assert src["net"]["placebo"]["content"]["mean"] == approx(0.0)
    assert src["net"]["placebo"]["form"]["mean"] == approx(-2 / 3)
    assert "self_reflect" not in src["net"]


# ---- diversity ---------------------------------------------------------------


def test_diversity_counts_distinct_parsable_answers_before_and_after():
    units = [
        unit(
            "p1",
            ["1", "2", "3"],
            [True, False, False],
            {"x": ["1", "1", "1"]},
            {"x": [True] * 3},
        ),
        unit(
            "p2",
            ["1", "1", "2"],
            [True, True, False],
            {"x": ["1", "1", "1"]},
            {"x": [True] * 3},
        ),
        unit(
            "p3",
            ["1", None, "1"],
            [True, False, True],
            {"x": ["1", "1", "1"]},
            {"x": [True] * 3},
        ),
    ]
    d = D.diversity(units, ["x"], n_boot=100, seed=0)
    assert d["round0"]["mean"] == approx(2.0)  # 3, 2 and 1 distinct answers
    assert d["arms"]["x"]["after"]["mean"] == approx(1.0)
    assert d["arms"]["x"]["change"]["mean"] == approx(-1.0)
    assert d["arms"]["x"]["unanimous_after"] == approx(1.0)
    assert d["unanimous_round0"] == approx(1 / 3)


# ---- headroom ----------------------------------------------------------------


def headroom_units():
    return [
        # recoverable: a right answer exists but the plurality ("2") is wrong
        unit(
            "a",
            ["2", "2", "1"],
            [False, False, True],
            {"x": ["1", "1", "2"], "y": ["2", "2", "2"]},
            {"x": [True, True, False], "y": [False] * 3},
        ),
        # the round-0 plurality is the right answer, alone
        unit(
            "b",
            ["1", "1", "2"],
            [True, True, False],
            {"x": ["1", "1", "1"], "y": ["2", "2", "1"]},
            {"x": [True] * 3, "y": [False, False, True]},
        ),
        # a three-way tie that contains the right answer: neither group
        unit(
            "c",
            ["1", "2", "3"],
            [True, False, False],
            {"x": ["1", "1", "1"], "y": ["2", "2", "2"]},
            {"x": [True] * 3, "y": [False] * 3},
        ),
    ]


def test_recovery_is_measured_on_the_cases_the_g2_gate_counts():
    h = D.headroom(headroom_units(), ["x", "y"], n_boot=100, seed=0)
    assert h["recoverable_units"] == 1 and h["plurality_right_units"] == 1
    assert h["arms"]["x"]["recovered"]["value"] == approx(1.0)
    assert h["arms"]["y"]["recovered"]["value"] == approx(0.0)
    assert h["arms"]["x"]["retained"]["value"] == approx(1.0)
    assert h["arms"]["y"]["retained"]["value"] == approx(0.0)


def test_with_no_recoverable_case_the_rate_is_none_not_zero():
    h = D.headroom(headroom_units()[1:], ["x"], n_boot=100, seed=0)
    assert h["recoverable_units"] == 0
    assert h["arms"]["x"]["recovered"]["value"] is None


# ---- conformity --------------------------------------------------------------


def conformity_units():
    return [
        # agent 1 ("2") moves to the wrong answer of agent 2 ("3")
        unit(
            "p1",
            ["1", "2", "3"],
            [True, False, False],
            {"x": ["1", "3", "3"]},
            {"x": [True, False, False]},
        ),
        # everyone already agrees on the same wrong answer: nobody is exposed
        unit(
            "p2",
            ["5", "5", "5"],
            [False] * 3,
            {"x": ["5", "5", "5"]},
            {"x": [False] * 3},
        ),
    ]


def test_conformity_counts_copies_of_a_peers_wrong_answer():
    c = D.conformity(conformity_units(), ["x"], n_boot=100, seed=0)["x"]
    assert c["copied_wrong_per_agent"]["value"] == approx(1 / 6)
    assert c["exposed_to_wrong"]["value"] == approx(3 / 6)
    assert c["copied_wrong_given_exposed"]["value"] == approx(1 / 3)
    assert c["copied_wrong_given_exposed"]["denominator"] == 3


def test_adopting_a_peers_right_answer_is_the_counterpart():
    c = D.conformity(conformity_units(), ["x"], n_boot=100, seed=0)["x"]
    # agents 1 and 2 hold wrong answers and a peer holds the right one; neither moved
    assert c["adopted_right_given_exposed"]["denominator"] == 2
    assert c["adopted_right_given_exposed"]["value"] == approx(0.0)


def test_an_agent_keeping_its_own_answer_never_counts_as_a_copy():
    u = unit(
        "p", ["2", "2", "3"], [False] * 3, {"x": ["2", "2", "3"]}, {"x": [False] * 3}
    )
    c = D.conformity([u], ["x"], n_boot=50, seed=0)["x"]
    assert c["copied_wrong_per_agent"]["value"] == 0.0


# ---- the mock pipeline -------------------------------------------------------

EFFECTS = {
    "self_reflect": 0.0,
    "answer_only": 0.0,
    "answer_conf": 0.0,
    "dissent_only": 0.0,
    "failure_cause": 0.3,
    "failure_cause_v2": 0.0,
    "full_solution": 0.0,
    "placebo": -0.4,
}


@pytest.fixture(scope="module")
def ctxs(tmp_path_factory):
    root = tmp_path_factory.mktemp("decompose")
    problems = fake_problems(40)
    world = MockWorld.from_problems(problems, base_p=0.45)
    run_shards(
        root,
        RunConfig(model_id="mock/model", replicates=2, max_tokens=512),
        problems,
        LazyBackend(lambda: MockBackend(world, effects=EFFECTS)),
        size=20,
        arms=(*ARMS, *VARIANT_ARMS),
        log=lambda s: None,
    )
    return [open_run(d) for d in sorted(root.glob("shard_*"))]


@pytest.fixture(scope="module")
def report(ctxs):
    return D.decompose(ctxs, n_boot=300)


def test_the_report_is_labelled_exploratory_and_names_every_part(report):
    assert report["status"]["exploratory"] is True
    assert "EXPLORATORY" in report["banner"]
    assert set(report) >= {"changes", "source", "diversity", "headroom", "conformity"}
    assert report["n_problems"] == 40 and report["replicates"] == 2
    assert set(report["changes"]) == {*ARMS, *VARIANT_ARMS}


def test_every_arms_change_shares_add_up_to_one(report):
    for arm, t in report["changes"].items():
        total = sum(v["mean"] for v in t["shares"].values())
        assert total == approx(1.0), arm


def test_the_planted_effects_show_in_the_change_types(report):
    ch = report["changes"]
    assert ch["failure_cause"]["net"]["mean"] > ch["self_reflect"]["net"]["mean"] + 0.1
    assert ch["placebo"]["shares"]["correct_to_wrong"]["mean"] > (
        ch["self_reflect"]["shares"]["correct_to_wrong"]["mean"] + 0.1
    )
    src = report["source"]["net"]
    assert src["placebo"]["form"]["mean"] < -0.1  # the form of peer content alone hurts
    assert src["failure_cause"]["content"]["mean"] > 0.1  # the planted gain is content


def test_the_change_shares_equal_a_direct_count_from_the_units(ctxs, report):
    units = collect_units(ctxs, (*ARMS, *VARIANT_ARMS))
    per_problem: dict[str, list[float]] = {}
    for u in units:
        flips = sum(
            was and not now
            for was, now in zip(u.r0_correct, u.final_correct["placebo"], strict=True)
        )
        per_problem.setdefault(u.problem_id, []).append(flips / 3)
    manual = sum(sum(v) / len(v) for v in per_problem.values()) / len(per_problem)
    got = report["changes"]["placebo"]["shares"]["correct_to_wrong"]["mean"]
    assert got == approx(manual)


def test_the_same_inputs_give_the_same_report(ctxs, report):
    assert D.decompose(ctxs, n_boot=300) == report


def test_the_text_report_has_the_banner_and_every_arm(report):
    text = D.format_decomposition_report(report)
    assert text.startswith("EXPLORATORY")
    for arm in (*ARMS, *VARIANT_ARMS):
        assert arm in text
    for part in (
        "change types",
        "source of change",
        "diversity",
        "headroom",
        "conformity",
    ):
        assert part in text.lower()
