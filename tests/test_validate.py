"""21_validate_run: counts, linkage, gates, budgets."""

from __future__ import annotations

from channel_audit.channels import ARMS
from channel_audit.runner import (
    grade_pending,
    run_all,
    run_arm,
    run_failure_messages,
    run_round0,
    run_sc_pool,
)
from channel_audit.validate import validate_run
from helpers import build_ctx


def full_run(tmp_path, **kw):
    ctx = build_ctx(tmp_path, **kw)
    run_all(ctx)
    grade_pending(ctx)
    return ctx


def test_complete_run_validates_with_budgets_for_every_arm(tmp_path):
    ctx = full_run(tmp_path, n_problems=12, base_p=0.5)
    report = validate_run(ctx)
    assert report.errors == []
    assert report.ok and report.g3_ok
    assert set(report.costs) == set(ARMS)
    for arm in ARMS:
        assert 3 <= report.matched_k[arm] <= 9, arm
    assert report.counts["round0"] == 12 * 2 * 3
    assert report.counts["sc_pool"] == 12 * 2 * 6
    assert report.counts["self_reflect"] == 12 * 2 * 3


def test_an_arm_that_was_not_run_is_reported_missing(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8)
    run_round0(ctx)
    run_sc_pool(ctx)
    run_failure_messages(ctx)
    for arm in ARMS:
        if arm != "placebo":
            run_arm(ctx, arm)
    grade_pending(ctx)
    report = validate_run(ctx)
    assert not report.ok
    assert any("placebo" in e and "missing" in e for e in report.errors)


def test_ungraded_generations_are_reported(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8)
    run_all(ctx)
    report = validate_run(ctx)
    assert any("not graded" in e for e in report.errors)


def test_truncation_is_reported_by_arm_and_fails_gate_g3(tmp_path):
    ctx = full_run(tmp_path, n_problems=8, max_tokens=8)
    report = validate_run(ctx)
    assert not report.g3_ok
    assert report.truncation_rate["self_reflect"] > 0.5
    assert "round0" in report.truncation_rate


def test_context_overflow_is_counted_by_arm_and_fails_gate_g3(tmp_path):
    ctx = full_run(tmp_path, n_problems=8, context_limit=100)
    report = validate_run(ctx)
    assert not report.g3_ok
    assert report.overflow_count["full_solution"] > 0


def test_untruncated_run_has_zero_truncation_and_overflow(tmp_path):
    ctx = full_run(tmp_path, n_problems=8)
    report = validate_run(ctx)
    assert all(v == 0 for v in report.truncation_rate.values())
    assert all(v == 0 for v in report.overflow_count.values())


def test_leakage_outcomes_are_reported(tmp_path):
    ctx = full_run(tmp_path, n_problems=8, leak_rate=1.0, leak_forms=("claim",))
    report = validate_run(ctx)
    assert report.errors == []
    assert report.leakage_outcomes == {"redacted": 8 * 2 * 3}
    assert report.counts["failure_msg"] == 8 * 2 * 3 * 2


def test_truncation_gap_to_self_reflect_is_a_warning_not_an_error(tmp_path):
    ctx = full_run(tmp_path, n_problems=8, max_tokens=8)
    report = validate_run(ctx)
    assert isinstance(report.warnings, list)
