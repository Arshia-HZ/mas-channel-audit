"""Gate G3 as preregistered: zero overflow, each revision arm under 5% truncation."""

from __future__ import annotations

import pytest

from channel_audit.channels import ARMS
from channel_audit.runner import grade_pending, run_all
from channel_audit.validate import g3_check, validate_run
from helpers import build_ctx


@pytest.fixture(scope="module")
def gens(tmp_path_factory):
    ctx = build_ctx(tmp_path_factory.mktemp("g3"), n_problems=20, replicates=1)
    run_all(ctx)
    grade_pending(ctx)
    return list(ctx.gens)


def cut(gens, arm, n):
    """The same generations with the first ``n`` of one arm marked truncated."""
    out, left = [], n
    for g in gens:
        if g.arm == arm and left > 0:
            g, left = g.model_copy(update={"finish_reason": "length"}), left - 1
        out.append(g)
    return out


def test_a_clean_run_passes(gens):
    r = g3_check(gens, limit=100_000)
    assert r.ok and r.overflow_total == 0 and not r.failing_arms and not r.flagged_arms


def test_round0_and_pool_truncation_is_reported_but_never_fails_g3(gens):
    for arm in ("round0", "sc_pool", "failure_msg"):
        r = g3_check(cut(gens, arm, 10_000), limit=100_000)
        assert r.ok, arm
        assert r.truncation[arm] > 0.9


def test_a_revision_arm_at_five_percent_or_more_fails(gens):
    n_arm = sum(g.arm == "answer_only" for g in gens)
    just_under = g3_check(cut(gens, "answer_only", int(n_arm * 0.05) - 1), 100_000)
    assert just_under.ok
    at_limit = g3_check(cut(gens, "answer_only", -(-n_arm * 5 // 100)), 100_000)
    assert not at_limit.ok and at_limit.failing_arms == ["answer_only"]


def test_any_context_overflow_fails_even_in_round0(gens):
    r = g3_check(gens, limit=10)
    assert not r.ok and r.overflow_total > 0


def test_an_arm_two_points_from_self_reflect_is_flagged_not_failed(gens):
    n_arm = sum(g.arm == "placebo" for g in gens)
    r = g3_check(cut(gens, "placebo", max(2, int(n_arm * 0.04))), 100_000)
    assert r.ok and "placebo" in r.flagged_arms


def test_validate_run_uses_the_same_rule(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=1, max_tokens=8)
    run_all(ctx)
    grade_pending(ctx)
    report = validate_run(ctx)
    assert not report.g3_ok
    assert set(report.g3_failing_arms) <= set(ARMS) and report.g3_failing_arms
