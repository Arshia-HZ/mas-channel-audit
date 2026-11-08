"""Confirmatory design: the first 600 confirm problems and an outcome-free report."""

from __future__ import annotations

import pytest

from channel_audit import data
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.data import (
    N_CONFIRMATORY,
    assign_splits,
    confirmatory_problems,
    load_math_test,
    write_split,
)
from channel_audit.quality import format_quality_report, run_quality
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import LazyBackend, run_shards


def fake_rows(config, revision):
    return [
        {
            "problem": f"{config} q{i}",
            "level": f"Level {i % 5 + 1}",
            "type": config,
            "solution": rf"So $\boxed{{{i}}}$.",
        }
        for i in range(400)
    ]


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("split")
    problems, dropped = load_math_test(loader=fake_rows)
    write_split(assign_splits(problems), d, n_rows=len(problems), dropped=dropped)
    return d


def test_the_plan_uses_six_hundred_confirmatory_problems():
    assert N_CONFIRMATORY == 600


def test_they_are_the_first_600_of_the_frozen_confirm_split(split_dir):
    chosen = confirmatory_problems(split_dir)
    full = data.load_frozen(split_dir, "confirm")
    assert len(chosen) == 600 and len(full) > 600
    assert [p.problem_id for p in chosen] == [p.problem_id for p in full[:600]]
    assert {p.split for p in chosen} == {"confirm"}
    assert {p.level for p in chosen} <= {3, 4, 5}


def test_the_choice_never_overlaps_the_pilot_or_the_screen(split_dir):
    chosen = {p.problem_id for p in confirmatory_problems(split_dir)}
    for name in ("pilot", "screen"):
        other = {p.problem_id for p in data.load_frozen(split_dir, name)}
        assert not chosen & other


def test_fewer_problems_than_asked_is_an_error(split_dir):
    with pytest.raises(ValueError, match="confirm"):
        confirmatory_problems(split_dir, n=10_000)


def test_every_group_has_a_donor_in_the_chosen_set_and_shards_keep_groups_whole(
    split_dir,
):
    from collections import Counter

    from channel_audit.shards import make_shards

    chosen = confirmatory_problems(split_dir)
    groups = Counter((p.subject, p.level) for p in chosen)
    assert min(groups.values()) >= 2
    homes = {}
    for i, shard in enumerate(make_shards(chosen, 50)):
        for p in shard:
            assert homes.setdefault((p.subject, p.level), i) == i


# --- the quality report (what the plan wants reported on every run) ------------


@pytest.fixture(scope="module")
def ctxs(tmp_path_factory):
    root = tmp_path_factory.mktemp("quality")
    problems = fake_problems(16)
    world = MockWorld.from_problems(problems)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        root, cfg, problems, LazyBackend(lambda: MockBackend(world)), size=8,
        arms=(*ARMS, *VARIANT_ARMS), log=lambda s: None,
    )  # fmt: skip
    return [open_run(d) for d in sorted(root.iterdir())]


def test_quality_covers_every_arm_with_the_four_rates(ctxs):
    q = run_quality(ctxs)
    assert set(ARMS) | set(VARIANT_ARMS) | {"round0", "sc_pool"} <= set(q["arms"])
    row = q["arms"]["answer_only"]
    assert {"n", "truncation_rate", "overflow", "extraction_failure_rate"} <= set(row)
    assert row["n"] == 16 * 2 * 3 and row["overflow"] == 0
    assert row["truncation_rate"] == 0 and row["extraction_failure_rate"] == 0


def test_quality_reports_leakage_outcomes_by_variant(ctxs):
    q = run_quality(ctxs)
    assert set(q["leakage"]) == {"v1", "v2"}
    for variant in ("v1", "v2"):
        row = q["leakage"][variant]
        assert row["n"] == 16 * 2 * 3
        assert row["clean"] + row["regenerated_clean"] + row["redacted"] == row["n"]
        assert 0 <= row["rejection_rate"] <= 1


def test_quality_contains_no_accuracy_or_gain(ctxs):
    text = str(run_quality(ctxs)).lower()
    for word in ("accuracy", "gain", "correct"):
        assert word not in text


def test_quality_sees_truncation_and_missing_boxes(tmp_path):
    problems = fake_problems(8)
    world = MockWorld.from_problems(problems)
    cfg = RunConfig(model_id="mock/model", replicates=1, max_tokens=8)
    run_shards(
        tmp_path, cfg, problems, LazyBackend(lambda: MockBackend(world)), size=8,
        log=lambda s: None,
    )  # fmt: skip
    q = run_quality([open_run(d) for d in sorted(tmp_path.iterdir())])
    assert q["arms"]["self_reflect"]["truncation_rate"] == 1.0
    assert (
        q["arms"]["self_reflect"]["extraction_failure_rate"] == 0.0
    )  # truncated, not "no box"


def test_the_printed_quality_report_has_a_row_per_arm(ctxs):
    text = format_quality_report(run_quality(ctxs))
    for arm in (*ARMS, "round0", "sc_pool"):
        assert arm in text
    assert "leakage" in text.lower() and "trunc" in text
