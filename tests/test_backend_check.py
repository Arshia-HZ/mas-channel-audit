"""Repro comparison and the 30-output review sample."""

from __future__ import annotations

from channel_audit.backend_check import (
    compare_runs,
    format_review,
    select_review_sample,
)
from channel_audit.channels import ARMS
from channel_audit.runner import grade_pending, run_all, run_round0
from helpers import build_ctx


def test_two_identical_runs_match_completely(tmp_path):
    a = build_ctx(tmp_path / "a", n_problems=4, replicates=1)
    b = build_ctx(tmp_path / "b", n_problems=4, replicates=1)
    run_round0(a)
    run_round0(b)
    r = compare_runs(a, b)
    assert (r.n_common, r.n_identical, r.n_different) == (12, 12, 0)
    assert r.match_rate == 1.0
    assert not r.only_in_a and not r.only_in_b and not r.divergences


def test_a_changed_generation_is_measured_with_its_divergence_point(tmp_path):
    a = build_ctx(tmp_path / "a", n_problems=4, replicates=1, base_p=1.0)
    b = build_ctx(tmp_path / "b", n_problems=4, replicates=1, base_p=0.0)
    run_round0(a)
    run_round0(b)
    r = compare_runs(a, b)
    assert r.n_common == 12 and r.n_different > 0
    assert r.match_rate < 1.0
    assert r.by_arm["round0"] == (12, r.n_identical)
    d = r.divergences[0]
    assert d["arm"] == "round0" and d["char_index"] >= 0


def test_non_overlapping_runs_report_the_difference(tmp_path):
    a = build_ctx(tmp_path / "a", n_problems=4, replicates=1)
    b = build_ctx(tmp_path / "b", n_problems=6, replicates=1)
    run_round0(a)
    run_round0(b)
    r = compare_runs(a, b)
    assert r.n_common == 12 and len(r.only_in_b) == 6 and not r.only_in_a


def test_report_renders_as_text(tmp_path):
    a = build_ctx(tmp_path / "a", n_problems=3, replicates=1)
    run_round0(a)
    text = compare_runs(a, a).format()
    assert "match rate" in text and "100.0%" in text


def _full(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=12, replicates=1, log_prompts=True)
    run_all(ctx)
    grade_pending(ctx)
    return ctx


def test_review_sample_has_30_outputs_covering_every_arm(tmp_path):
    ctx = _full(tmp_path)
    sample = select_review_sample(ctx, n=30, seed=0)
    assert len(sample) == 30
    kinds = {g.arm for g in sample}
    assert kinds == {"round0", "sc_pool", "failure_msg", *ARMS}
    assert len({g.gen_key for g in sample}) == 30


def test_review_sample_is_deterministic_and_seed_dependent(tmp_path):
    ctx = _full(tmp_path)
    a = [g.gen_key for g in select_review_sample(ctx, n=30, seed=0)]
    assert a == [g.gen_key for g in select_review_sample(ctx, n=30, seed=0)]
    assert a != [g.gen_key for g in select_review_sample(ctx, n=30, seed=1)]


def test_review_sample_fills_from_other_kinds_when_one_is_short(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=1, base_p=1.0, log_prompts=True)
    run_all(ctx)  # everyone agrees, so dissent_only generates nothing
    sample = select_review_sample(ctx, n=30, seed=0)
    assert len(sample) == 30
    assert "dissent_only" not in {g.arm for g in sample}


def test_review_text_shows_prompt_output_and_grade(tmp_path):
    ctx = _full(tmp_path)
    sample = select_review_sample(ctx, n=30, seed=0)
    text = format_review(ctx, sample)
    assert text.count("## ") >= 30
    first = sample[0]
    assert first.gen_key[:12] in text
    assert first.text.strip().splitlines()[0] in text
    assert "Prompt" in text and "Output" in text
    assert "correct" in text.lower()
    assert "Problem" in text


def test_review_without_a_prompt_log_says_so_instead_of_inventing_one(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=6, replicates=1)
    run_round0(ctx)
    text = format_review(ctx, select_review_sample(ctx, n=3, seed=0))
    assert "prompt not logged" in text
