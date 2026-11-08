"""The v2 failure-message variant through the runner, validator and budget."""

from __future__ import annotations

import json

import pytest

from channel_audit import channels
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.budget import arm_costs
from channel_audit.channels import ARMS
from channel_audit.leakage import record_key
from channel_audit.runner import (
    RunConfig,
    create_run,
    final_generations,
    grade_pending,
    run_all,
    run_arm,
    run_failure_messages,
)
from channel_audit.validate import g3_check, validate_run
from helpers import RecordingBackend, build_ctx


def with_v2(tmp_path, **kw):
    ctx = build_ctx(tmp_path, **{"n_problems": 8, "replicates": 2, **kw})
    run_all(ctx)
    run_all(ctx, arms=("failure_cause_v2",))
    grade_pending(ctx)
    return ctx


def test_the_v2_variant_adds_failure_messages_and_a_revision_per_agent(tmp_path):
    ctx = with_v2(tmp_path)
    by_arm = {}
    for g in ctx.gens:
        by_arm.setdefault(g.arm, []).append(g)
    n_agents_total = 8 * 2 * 3
    assert len(by_arm["failure_msg"]) == n_agents_total  # clean mock: no retries
    assert len(by_arm["failure_msg_v2"]) == n_agents_total
    assert len(by_arm["failure_cause_v2"]) == n_agents_total
    assert len(by_arm["failure_cause"]) == n_agents_total


def test_v2_records_are_kept_apart_from_v1_records(tmp_path):
    ctx = with_v2(tmp_path)
    v1 = [r for r in ctx.leakage if r.variant == "v1"]
    v2 = [r for r in ctx.leakage if r.variant == "v2"]
    assert len(v1) == len(v2) == 8 * 2 * 3
    assert all(r.record_key.startswith("v2|") for r in v2)
    assert all(not r.record_key.startswith("v2|") for r in v1)


def test_running_v2_never_changes_the_stored_v1_generations(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=1)
    run_all(ctx)
    before = {g.gen_key: g.text for g in ctx.gens}
    run_failure_messages(ctx, "v2")
    run_arm(ctx, "failure_cause_v2")
    after = {g.gen_key: g.text for g in ctx.gens if g.gen_key in before}
    assert after == before


def test_v2_uses_its_own_prompt_and_the_old_revision_prompt_shape(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=1)
    run_all(ctx)
    run_failure_messages(ctx, "v2")
    sent = ctx.backend.by_arm("failure_msg_v2")
    assert sent and all("single sentence" in r.messages[0]["content"] for r in sent)
    old = ctx.backend.by_arm("failure_msg")
    assert all("single sentence" not in r.messages[0]["content"] for r in old)
    run_arm(ctx, "failure_cause_v2")
    revision = ctx.backend.by_arm("failure_cause_v2")
    assert revision and all(
        "Another solver reviewed your work" in r.messages[0]["content"]
        for r in revision
    )


def test_the_v2_revision_reads_the_v2_message_not_the_v1_one(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=1)
    run_all(ctx)
    run_failure_messages(ctx, "v2")
    run_arm(ctx, "failure_cause_v2")
    p = ctx.problems[0]
    v2_text = ctx.leakage.get(record_key(p.problem_id, 0, 0, "v2")).sanitized_text
    v1_text = ctx.leakage.get(record_key(p.problem_id, 0, 0, "v1")).sanitized_text
    assert v2_text != v1_text  # different generations (different arm, different seed)
    first = [
        r
        for r in ctx.backend.by_arm("failure_cause_v2")
        if v2_text in r.messages[0]["content"]
    ]
    assert first, "the v2 revision prompt must contain the v2 message"


def test_v2_messages_are_cut_to_one_sentence_even_when_the_model_rambles(tmp_path):
    ctx = build_ctx(
        tmp_path, n_problems=8, replicates=1, leak_rate=1.0, leak_forms=("boxed",)
    )
    run_all(ctx)
    run_failure_messages(ctx, "v2")
    for r in ctx.leakage:
        if r.variant == "v2":
            assert r.sanitized_text.count(". ") == 0
            assert r.outcome == "clean"  # the leaking second sentence was dropped


def test_a_leak_inside_the_first_sentence_is_still_regenerated_then_redacted(tmp_path):
    problems = fake_problems(8)

    class Leaky(MockBackend):
        def _failure_message(self, req, rng):
            from channel_audit.extract import extract_answer

            reviewed = extract_answer(req.messages[-1]["content"])
            return f"Step 2 is wrong because the result there is {reviewed}."

    cfg = RunConfig(model_id="mock/model", replicates=1, max_tokens=512)
    backend = RecordingBackend(Leaky(MockWorld.from_problems(problems)))
    ctx = create_run(tmp_path / "run", cfg, problems, backend)
    run_all(ctx)
    run_failure_messages(ctx, "v2")
    outcomes = {r.outcome for r in ctx.leakage if r.variant == "v2"}
    assert outcomes <= {"redacted", "regenerated_clean"} and "redacted" in outcomes
    redacted = [r for r in ctx.leakage if r.variant == "v2" and r.outcome == "redacted"]
    assert all("[answer]" in r.sanitized_text for r in redacted)
    assert len([g for g in ctx.gens if g.arm == "failure_msg_v2"]) > 8 * 3 * 1


def test_v2_failure_messages_are_not_graded_as_solutions(tmp_path):
    ctx = with_v2(tmp_path)
    graded = {g.arm for g in ctx.gens if g.gen_key in ctx.graded}
    assert "failure_msg_v2" not in graded and "failure_cause_v2" in graded


def test_validate_finds_a_complete_run_with_the_variant_valid(tmp_path):
    ctx = with_v2(tmp_path)
    report = validate_run(ctx)
    assert report.errors == []
    assert "failure_cause_v2" in report.costs and "failure_cause_v2" in report.matched_k
    assert report.leakage_outcomes.get("v2/clean") == 8 * 2 * 3


def test_validate_reports_missing_v2_generations(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=1)
    run_all(ctx)
    run_failure_messages(ctx, "v2")  # messages, but no v2 revisions
    grade_pending(ctx)
    report = validate_run(ctx)
    assert any("failure_cause_v2" in e and "missing" in e for e in report.errors)


def test_a_plain_v1_run_is_validated_exactly_as_before(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=1)
    run_all(ctx)
    grade_pending(ctx)
    report = validate_run(ctx)
    assert report.errors == [] and "failure_cause_v2" not in report.costs
    assert set(report.leakage_outcomes) <= {"clean", "regenerated_clean", "redacted"}


def test_the_v2_arm_pays_for_its_own_failure_messages_only(tmp_path):
    ctx = with_v2(tmp_path)
    gens = list(ctx.gens)
    costs = arm_costs(gens)
    v1_msgs = sum(g.completion_tokens for g in gens if g.arm == "failure_msg")
    v2_msgs = sum(g.completion_tokens for g in gens if g.arm == "failure_msg_v2")
    r0 = sum(g.completion_tokens for g in gens if g.arm == "round0")
    rev1 = sum(g.completion_tokens for g in gens if g.arm == "failure_cause")
    rev2 = sum(g.completion_tokens for g in gens if g.arm == "failure_cause_v2")
    n = costs["failure_cause_v2"].n_replicates
    assert costs["failure_cause"].generated_tokens == pytest.approx(
        (r0 + rev1 + v1_msgs) / n
    )
    assert costs["failure_cause_v2"].generated_tokens == pytest.approx(
        (r0 + rev2 + v2_msgs) / n
    )


def test_g3_covers_the_variant_arm(tmp_path):
    ctx = with_v2(tmp_path)
    gens = list(ctx.gens)
    cut = [
        g.model_copy(update={"finish_reason": "length"})
        if g.arm == "failure_cause_v2"
        else g
        for g in gens
    ]
    result = g3_check(cut, 100_000)
    assert not result.ok and result.failing_arms == ["failure_cause_v2"]
    assert g3_check(gens, 100_000).ok


def test_final_generations_work_for_the_variant_arm(tmp_path):
    ctx = with_v2(tmp_path)
    p = ctx.problems[0]
    finals = final_generations(ctx, "failure_cause_v2", p.problem_id, 0)
    assert [g.arm for g in finals] == ["failure_cause_v2"] * 3


def test_variant_arm_keys_differ_from_v1_arm_keys(tmp_path):
    ctx = with_v2(tmp_path)
    keys = {}
    for g in ctx.gens:
        keys.setdefault(g.arm, set()).add(g.gen_key)
    assert keys["failure_cause"].isdisjoint(keys["failure_cause_v2"])
    assert keys["failure_msg"].isdisjoint(keys["failure_msg_v2"])
    assert set(ARMS) | set(channels.VARIANT_ARMS) >= {
        "failure_cause",
        "failure_cause_v2",
    }
    json.dumps(sorted(keys))  # arms are plain strings


def test_an_unknown_variant_is_an_error(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=1)
    run_all(ctx)
    with pytest.raises(ValueError, match="variant"):
        run_failure_messages(ctx, "v9")
