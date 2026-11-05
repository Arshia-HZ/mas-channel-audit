"""M4 screening: 4 samples per problem, the report, and gate G0."""

from __future__ import annotations

import json

import pytest

from channel_audit import channels
from channel_audit.runner import grade_pending
from channel_audit.screening import (
    N_SAMPLES,
    PUBLISHED_MATH,
    SCREEN_ARM,
    format_comparison,
    g0_check,
    problem_stats,
    run_screen,
    screen_report,
)
from helpers import build_ctx


def screened(tmp_path, **kw):
    ctx = build_ctx(tmp_path, **{"n_problems": 8, "replicates": 1, **kw})
    run_screen(ctx)
    grade_pending(ctx)
    return ctx


def test_each_problem_gets_four_samples_with_distinct_seeds(tmp_path):
    ctx = screened(tmp_path)
    gens = [g for g in ctx.gens if g.arm == SCREEN_ARM]
    assert N_SAMPLES == 4 and len(gens) == 8 * 4
    for p in ctx.problems:
        mine = [g for g in gens if g.problem_id == p.problem_id]
        assert sorted(g.sample_idx for g in mine) == [0, 1, 2, 3]
        assert len({g.seed for g in mine}) == 4
        assert {(g.agent, g.round, g.replicate) for g in mine} == {(0, 0, 0)}


def test_screening_uses_the_round0_prompt(tmp_path):
    ctx = screened(tmp_path, n_problems=2)
    sent = {json.dumps(r.messages) for r in ctx.backend.by_arm(SCREEN_ARM)}
    expected = {
        json.dumps(channels.build_round0_prompt(p.question)) for p in ctx.problems
    }
    assert sent == expected
    assert {g.prompt_hash for g in ctx.gens} == {channels.template_hash("round0")}


def test_screening_is_resumable_and_idempotent(tmp_path):
    ctx = screened(tmp_path, n_problems=5)
    n = len(ctx.gens)
    run_screen(ctx)
    assert len(ctx.gens) == n and len(ctx.backend.requests) == n


def test_screening_runs_in_chunks_so_progress_is_saved(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=5, replicates=1)
    run_screen(ctx, chunk=2)
    lines = (ctx.out_dir / "throughput.jsonl").read_text().splitlines()
    assert [json.loads(x)["n_requests"] for x in lines] == [8, 8, 4]


def test_problem_stats_definitions():
    same = problem_stats(["3", "3", "3", "3"], [True] * 4)
    assert (same.disagrees, same.mixed, same.all_wrong) == (False, False, False)
    wrong = problem_stats(["1", "2", "2", "5"], [False] * 4)
    assert (wrong.disagrees, wrong.mixed, wrong.all_wrong) == (True, False, True)
    mixed = problem_stats(["3", "3", "4", "3"], [True, True, False, True])
    assert (mixed.disagrees, mixed.mixed, mixed.all_wrong) == (True, True, False)
    # unparsable answers are not votes, so they cannot create disagreement
    assert not problem_stats(
        ["3", None, "3", None], [True, False, True, False]
    ).disagrees
    # equivalent forms are one answer
    assert not problem_stats(
        ["1/2", "0.5", "\\frac{1}{2}", "1/2"], [True] * 4
    ).disagrees


def test_report_with_always_correct_samples(tmp_path):
    ctx = screened(tmp_path, base_p=1.0, effects={})
    r = screen_report(ctx)
    assert r["n_problems"] == 8 and r["n_samples"] == 32
    assert r["accuracy"] == 1.0 and r["disagreement_rate"] == 0.0
    assert r["truncation_rate"] == 0.0 and r["extraction_failure_rate"] == 0.0
    assert r["all_correct_rate"] == 1.0 and r["mixed_rate"] == 0.0


def test_report_with_always_wrong_samples(tmp_path):
    ctx = screened(tmp_path, base_p=0.0, effects={})
    r = screen_report(ctx)
    assert r["accuracy"] == 0.0 and r["all_wrong_rate"] == 1.0
    assert r["mixed_rate"] == 0.0


def test_report_accuracy_by_level_uses_problem_levels(tmp_path):
    ctx = screened(tmp_path, base_p=1.0, effects={}, n_problems=8)
    r = screen_report(ctx)
    assert set(r["by_level"]) == {3, 4}
    assert r["by_level"][3]["n_problems"] == 4
    assert r["by_level"][3]["accuracy"] == 1.0
    assert r["study_levels"]["n_problems"] == 8 and r["study_levels"]["accuracy"] == 1.0


def test_truncated_outputs_are_counted_apart_from_missing_boxes(tmp_path):
    ctx = screened(tmp_path, max_tokens=5, n_problems=4)
    r = screen_report(ctx)
    assert r["truncation_rate"] == 1.0
    assert r["extraction_failure_rate"] == 0.0  # finished-without-box only
    assert r["no_answer_rate"] == 1.0
    assert r["accuracy"] == 0.0


def test_report_needs_screening_generations(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=2, replicates=1)
    with pytest.raises(ValueError, match="screen"):
        screen_report(ctx)


def test_g0_passes_within_tolerance_and_low_extraction_failure():
    ok = g0_check(accuracy=0.52, extraction_failure_rate=0.01, published=55.2)
    assert ok["passed"] and ok["accuracy_ok"] and ok["extraction_ok"]
    assert ok["difference_points"] == pytest.approx(-3.2)


def test_g0_fails_on_a_big_accuracy_gap_in_either_direction():
    assert not g0_check(0.45, 0.0, published=55.2)["passed"]
    assert not g0_check(0.66, 0.0, published=55.2)["passed"]


def test_g0_fails_when_extraction_failure_reaches_five_percent():
    assert not g0_check(0.55, 0.05, published=55.2)["passed"]
    assert g0_check(0.55, 0.049, published=55.2)["passed"]


def test_the_sanity_model_has_a_published_reference():
    assert PUBLISHED_MATH["Qwen/Qwen2.5-1.5B-Instruct"] == 55.2


def test_comparison_table_lists_every_model_and_g0_for_the_sanity_model(tmp_path):
    a = screen_report(screened(tmp_path / "a", base_p=1.0, effects={}))
    b = screen_report(screened(tmp_path / "b", base_p=0.0, effects={}))
    a["model_id"], b["model_id"] = "org/alpha", "Qwen/Qwen2.5-1.5B-Instruct"
    text = format_comparison([a, b])
    assert "org/alpha" in text and "Qwen/Qwen2.5-1.5B-Instruct" in text
    assert "G0" in text and "disagree" in text


# --- plain sanity prompt (published-style, no confidence request) ------------

PLAIN_ARM = "screen_plain"


def test_the_plain_template_is_a_question_and_the_box_instruction_only(tmp_path):
    text = channels.template_text("screen_plain")
    assert "<<question>>" in text and "\\boxed{}" in text
    assert "onfidence" not in text
    msgs = channels.build_plain_prompt("What is 1+1?")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert "What is 1+1?" in msgs[0]["content"] and "<<" not in msgs[0]["content"]


def test_the_plain_arm_has_its_own_template_hash():
    assert channels.template_hash("screen_plain") != channels.template_hash("screen")


def test_plain_screening_uses_a_separate_arm_and_the_plain_prompt(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3, replicates=1)
    run_screen(ctx, prompt="plain")
    gens = [g for g in ctx.gens if g.arm == PLAIN_ARM]
    assert len(gens) == 3 * 4
    sent = {json.dumps(r.messages) for r in ctx.backend.by_arm(PLAIN_ARM)}
    expected = {
        json.dumps(channels.build_plain_prompt(p.question)) for p in ctx.problems
    }
    assert sent == expected
    assert {g.prompt_hash for g in gens} == {channels.template_hash("screen_plain")}


def test_both_prompts_can_share_one_run_directory(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3, replicates=1)
    run_screen(ctx)
    run_screen(ctx, prompt="plain")
    run_screen(ctx, prompt="plain")  # resumable
    arms = [g.arm for g in ctx.gens]
    assert arms.count(SCREEN_ARM) == 12 and arms.count(PLAIN_ARM) == 12


def test_an_unknown_prompt_is_an_error(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=2, replicates=1)
    with pytest.raises(ValueError, match="prompt"):
        run_screen(ctx, prompt="fancy")


def test_report_can_be_taken_per_prompt(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=4, replicates=1, base_p=1.0, effects={})
    run_screen(ctx)
    run_screen(ctx, prompt="plain")
    grade_pending(ctx)
    plain = screen_report(ctx, prompt="plain")
    assert plain["prompt"] == "plain" and plain["n_samples"] == 16
    assert screen_report(ctx)["prompt"] == "round0"
    with pytest.raises(ValueError, match="screen"):
        screen_report(
            build_ctx(tmp_path / "other", n_problems=2, replicates=1), "plain"
        )


def test_the_comparison_names_the_prompt_and_checks_g0_on_the_plain_one(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=4, replicates=1, base_p=1.0, effects={})
    run_screen(ctx)
    run_screen(ctx, prompt="plain")
    grade_pending(ctx)
    reports = [screen_report(ctx), screen_report(ctx, prompt="plain")]
    for r in reports:
        r["model_id"] = "Qwen/Qwen2.5-1.5B-Instruct"
    text = format_comparison(reports)
    assert "plain" in text and "round0" in text
    assert text.count("G0") == 2
