"""Pilot report additions: the v2 arm, G2 by level and the self-consistency curve."""

from __future__ import annotations

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS
from channel_audit.failure_review import (
    format_stats,
    gold_audit,
    sample_failure_messages,
)
from channel_audit.gates import (
    collect_units,
    format_pilot_report,
    g2_headroom,
    pilot_report,
)
from channel_audit.runner import (
    RunConfig,
    grade_pending,
    open_run,
    run_arm,
    run_failure_messages,
)
from channel_audit.shards import LazyBackend, run_shards

N = 16


@pytest.fixture(scope="module")
def ctxs(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2")
    problems = fake_problems(N)
    world = MockWorld.from_problems(problems)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        root, cfg, problems, LazyBackend(lambda: MockBackend(world)), size=8,
        log=lambda s: None,
    )  # fmt: skip
    out = []
    for d in sorted(root.iterdir()):
        ctx = open_run(d, MockBackend(world))
        run_failure_messages(ctx, "v2")
        run_arm(ctx, "failure_cause_v2")
        grade_pending(ctx)
        out.append(ctx)
    return out


def test_units_can_carry_the_variant_arm(ctxs):
    units = collect_units(ctxs, (*ARMS, "failure_cause_v2"))
    assert all("failure_cause_v2" in u.finals for u in units)


def test_the_report_detects_and_includes_the_variant_arm(ctxs):
    rep = pilot_report(ctxs, 3250)
    assert "failure_cause_v2" in rep["arms"]
    assert "failure_cause_v2_vs_full_solution" in rep["contrasts"]
    assert rep["contrasts"]["failure_cause_v2_vs_full_solution"]["n_problems"] == N
    row = rep["arms"]["failure_cause_v2"]
    assert row["tokens"] > 0 and row["matched_k"] >= 1


def test_without_v2_generations_the_report_is_unchanged(ctxs, tmp_path):
    problems = fake_problems(8)
    world = MockWorld.from_problems(problems)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        tmp_path, cfg, problems, LazyBackend(lambda: MockBackend(world)), size=8,
        log=lambda s: None,
    )  # fmt: skip
    rep = pilot_report([open_run(d) for d in sorted(tmp_path.iterdir())], 3250)
    assert "failure_cause_v2" not in rep["arms"]
    assert "failure_cause_v2_vs_full_solution" not in rep["contrasts"]


def test_g2_is_also_reported_by_level(ctxs):
    units = collect_units(ctxs)
    g2 = g2_headroom(units)
    assert set(g2["by_level"]) == {3, 4}
    for lv in (3, 4):
        group = [u for u in units if u.level == lv]
        sub = g2_headroom(group)
        assert g2["by_level"][lv] == pytest.approx(sub["rate"])


def test_the_report_carries_the_self_consistency_curve(ctxs):
    rep = pilot_report(ctxs, 3250)
    curve = rep["sc_curve"]
    assert sorted(curve) == list(range(1, 10))
    assert all(0 <= v <= 1 for v in curve.values())
    assert curve[9] >= curve[1] - 0.2  # more samples never collapse accuracy


def test_the_printed_report_shows_g2_by_level_and_the_sc_curve(ctxs):
    text = format_pilot_report(pilot_report(ctxs, 3250), None)
    assert "SC accuracy by samples" in text and "k=3" in text
    assert text.count("by level") >= 2
    assert "failure_cause_v2" in text


def test_failure_samples_can_be_taken_per_variant(ctxs):
    v1 = sample_failure_messages(ctxs, n=5, seed=0)
    v2 = sample_failure_messages(ctxs, n=5, seed=0, variant="v2")
    assert all(not i["record_key"].startswith("v2|") for i in v1)
    assert all(i["record_key"].startswith("v2|") for i in v2)
    assert {i["variant"] for i in v2} == {"v2"}


def test_the_gold_audit_counts_messages_that_contain_the_gold_answer(ctxs):
    items = sample_failure_messages(ctxs, n=10_000, seed=0, variant="v2")
    audit = gold_audit(items)
    assert audit["n"] == len(items) and audit["contains_gold"] == 0
    leaked = [dict(items[0], message=f"The result is {items[0]['gold']}.")]
    assert gold_audit(leaked)["contains_gold"] == 1


def test_format_stats_counts_sentences_and_artifacts(ctxs):
    items = [
        {"message": "Step 2 drops a sign."},
        {"message": "Step 2 is wrong. It also fails here."},
        {"message": "Step 3 gives [answer removed] which is odd."},
    ]
    s = format_stats(items)
    assert s["n"] == 3 and s["single_sentence"] == 2 and s["artifacts"] == 1
