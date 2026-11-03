"""Mock end-to-end pipeline: structure, linkage, leakage, resume."""

from __future__ import annotations

import pytest

from channel_audit import channels
from channel_audit.channels import ARMS, agent_view, disagrees
from channel_audit.extract import extract_boxed
from channel_audit.leakage import find_leaks
from channel_audit.runner import (
    MissingRound0Error,
    RunConfig,
    create_run,
    final_generations,
    open_run,
    round0_gens,
    run_all,
    run_arm,
    run_failure_messages,
    run_round0,
    run_sc_pool,
)
from helpers import build_ctx


def marker(gen):
    return f"[trace {gen.seed}]"


def test_round0_creates_three_independent_agents_per_replicate(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=5, replicates=2)
    run_round0(ctx)
    assert len(ctx.gens) == 5 * 2 * 3
    gens = round0_gens(ctx, ctx.problems[0].problem_id, 1)
    assert [g.agent for g in gens] == [0, 1, 2]
    assert len({g.seed for g in gens}) == 3
    assert all(
        g.arm == "round0" and g.round == 0 and g.parent_gen_keys == [] for g in gens
    )


def test_sc_pool_creates_six_extra_independent_samples(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=4, replicates=2)
    run_round0(ctx)
    run_sc_pool(ctx)
    pool = [g for g in ctx.gens if g.arm == "sc_pool"]
    assert len(pool) == 4 * 2 * 6
    assert len({g.seed for g in pool}) == len(pool)


def test_rerunning_a_step_adds_nothing_and_calls_no_backend(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=4)
    run_round0(ctx)
    n, calls = len(ctx.gens), len(ctx.backend.requests)
    run_round0(ctx)
    assert len(ctx.gens) == n and len(ctx.backend.requests) == calls


def test_resume_in_a_new_process_skips_finished_work(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8)
    run_all(ctx)
    n = len(ctx.gens)
    again = open_run(tmp_path / "run", ctx.backend)
    run_all(again)
    assert len(again.gens) == n


def test_arms_cannot_run_before_round0(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3)
    with pytest.raises(MissingRound0Error):
        run_arm(ctx, "self_reflect")


def test_failure_cause_cannot_run_before_failure_messages(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3)
    run_round0(ctx)
    with pytest.raises(LookupError):
        run_arm(ctx, "failure_cause")


def test_every_arm_in_a_replicate_links_to_the_same_three_round0_keys(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8, replicates=2, base_p=0.5)
    run_all(ctx)
    for problem in ctx.problems:
        for rep in range(2):
            keys = [g.gen_key for g in round0_gens(ctx, problem.problem_id, rep)]
            assert len(keys) == 3
            arm_gens = [
                g
                for g in ctx.gens
                if g.problem_id == problem.problem_id
                and g.replicate == rep
                and g.arm in ARMS
            ]
            assert arm_gens
            for g in arm_gens:
                assert g.parent_gen_keys[:3] == keys, g.arm
                assert all(k in ctx.gens for k in g.parent_gen_keys)


def test_self_reflect_prompts_contain_no_peer_text(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=6)
    run_all(ctx, arms=["self_reflect"])
    reqs = ctx.backend.by_arm("self_reflect")
    assert len(reqs) == 6 * 2 * 3
    for r in reqs:
        r0 = round0_gens(ctx, r.meta["problem_id"], r.meta["replicate"])
        text = r.messages[0]["content"]
        assert marker(r0[r.meta["agent"]]) in text
        for peer in (g for g in r0 if g.agent != r.meta["agent"]):
            assert marker(peer) not in text


def test_answer_channels_contain_no_peer_reasoning(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=6)
    run_all(ctx, arms=["answer_only", "answer_conf"])
    for arm in ("answer_only", "answer_conf"):
        reqs = ctx.backend.by_arm(arm)
        assert len(reqs) == 36
        for r in reqs:
            r0 = round0_gens(ctx, r.meta["problem_id"], r.meta["replicate"])
            text = r.messages[0]["content"]
            for peer in (g for g in r0 if g.agent != r.meta["agent"]):
                assert marker(peer) not in text
                assert f"Solver {peer.agent + 1}:" in text


def test_full_solution_contains_exactly_the_two_peers(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=4)
    run_all(ctx, arms=["full_solution"])
    for r in ctx.backend.by_arm("full_solution"):
        r0 = round0_gens(ctx, r.meta["problem_id"], r.meta["replicate"])
        text = r.messages[0]["content"]
        for g in r0:
            assert (marker(g) in text) is True


def test_placebo_donors_are_other_problems_of_the_same_subject_and_level(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=24, replicates=2)
    run_all(ctx, arms=["placebo", "full_solution"])
    by_id = {p.problem_id: p for p in ctx.problems}
    placebo = [g for g in ctx.gens if g.arm == "placebo"]
    assert len(placebo) == 24 * 2 * 3
    for g in placebo:
        own = by_id[g.problem_id]
        donors = [ctx.gens.get(k) for k in g.parent_gen_keys[3:]]
        assert len(donors) == 2
        for d in donors:
            assert d.problem_id != g.problem_id
            assert d.replicate == g.replicate
            assert d.agent != g.agent
            donor_problem = by_id[d.problem_id]
            assert (donor_problem.subject, donor_problem.level) == (
                own.subject,
                own.level,
            )
        assert len({d.problem_id for d in donors}) == 1


def test_placebo_prompt_shows_donor_text_and_not_the_real_peers(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=12)
    run_all(ctx, arms=["placebo"])
    for r in ctx.backend.by_arm("placebo"):
        r0 = round0_gens(ctx, r.meta["problem_id"], r.meta["replicate"])
        text = r.messages[0]["content"]
        for peer in (g for g in r0 if g.agent != r.meta["agent"]):
            assert marker(peer) not in text
        assert text.count("[trace ") == 3  # own + two donor solutions


def test_placebo_and_full_solution_use_the_same_template_hash(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=8)
    run_all(ctx, arms=["placebo", "full_solution"])
    hashes = {
        g.arm: g.prompt_hash for g in ctx.gens if g.arm in ("placebo", "full_solution")
    }
    assert hashes["placebo"] == hashes["full_solution"]


def test_failure_message_reviews_only_the_readers_own_solution(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=5)
    run_round0(ctx)
    run_failure_messages(ctx)
    reqs = ctx.backend.by_arm("failure_msg")
    assert reqs
    for r in reqs:
        r0 = round0_gens(ctx, r.meta["problem_id"], r.meta["replicate"])
        text = r.messages[0]["content"]
        j = r.meta["agent"]
        assert marker(r0[j]) in text
        assert all(marker(g) not in text for g in r0 if g.agent != j)
        assert "Do not give a final answer" in text


def test_clean_messages_take_one_attempt_each(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=5, leak_rate=0.0)
    run_round0(ctx)
    run_failure_messages(ctx)
    assert len(ctx.leakage) == 5 * 2 * 3
    assert {r.outcome for r in ctx.leakage} == {"clean"}
    assert len([g for g in ctx.gens if g.arm == "failure_msg"]) == 30


def test_boxed_and_answer_line_leaks_are_stripped_without_regeneration(tmp_path):
    ctx = build_ctx(
        tmp_path, n_problems=5, leak_rate=1.0, leak_forms=("boxed", "answer_line")
    )
    run_round0(ctx)
    run_failure_messages(ctx)
    assert {r.outcome for r in ctx.leakage} == {"clean"}
    assert all(r.n_stripped >= 1 for r in ctx.leakage)
    assert all("boxed" not in r.sanitized_text for r in ctx.leakage)
    assert len([g for g in ctx.gens if g.arm == "failure_msg"]) == 30


def test_persistent_claim_leaks_regenerate_once_then_are_redacted_and_recorded(
    tmp_path,
):
    ctx = build_ctx(tmp_path, n_problems=5, leak_rate=1.0, leak_forms=("claim",))
    run_round0(ctx)
    run_failure_messages(ctx)
    assert {r.outcome for r in ctx.leakage} == {"redacted"}
    assert {r.attempts for r in ctx.leakage} == {2}
    assert len([g for g in ctx.gens if g.arm == "failure_msg"]) == 60
    for r in ctx.leakage:
        answers = [
            agent_view(g).answer for g in round0_gens(ctx, r.problem_id, r.replicate)
        ]
        assert find_leaks(r.sanitized_text, answers) == []
        assert ctx.gens.get(r.final_gen_key).sample_idx == 1


def test_a_leak_that_goes_away_on_regeneration_is_recorded_as_regenerated(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=25, leak_rate=0.5, leak_forms=("claim",))
    run_round0(ctx)
    run_failure_messages(ctx)
    outcomes = {r.outcome for r in ctx.leakage}
    assert {"clean", "regenerated_clean"} <= outcomes
    for r in ctx.leakage:
        assert r.attempts == (1 if r.outcome == "clean" else 2)
    n_attempts = sum(r.attempts for r in ctx.leakage)
    assert len([g for g in ctx.gens if g.arm == "failure_msg"]) == n_attempts


def test_failure_cause_prompt_carries_the_readers_own_sanitized_message(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=6, leak_rate=1.0, leak_forms=("claim",))
    run_round0(ctx)
    run_failure_messages(ctx)
    run_arm(ctx, "failure_cause")
    reqs = ctx.backend.by_arm("failure_cause")
    assert len(reqs) == 36
    for r in reqs:
        pid, rep, j = r.meta["problem_id"], r.meta["replicate"], r.meta["agent"]
        record = ctx.leakage.get(f"{pid}|{rep}|{j}")
        text = r.messages[0]["content"]
        assert record.sanitized_text in text
        assert "boxed" not in record.sanitized_text
        r0 = round0_gens(ctx, pid, rep)
        assert all(marker(g) not in text for g in r0 if g.agent != j)
        gen = next(
            g
            for g in ctx.gens
            if g.arm == "failure_cause"
            and g.problem_id == pid
            and g.replicate == rep
            and g.agent == j
        )
        assert gen.parent_gen_keys[3:] == [record.final_gen_key]


def test_dissent_only_generates_nothing_when_every_answer_agrees(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=6, base_p=1.0)
    run_round0(ctx)
    run_arm(ctx, "dissent_only")
    assert not [g for g in ctx.gens if g.arm == "dissent_only"]
    for p in ctx.problems:
        finals = final_generations(ctx, "dissent_only", p.problem_id, 0)
        assert [g.gen_key for g in finals] == [
            g.gen_key for g in round0_gens(ctx, p.problem_id, 0)
        ]


def test_dissent_only_revises_exactly_the_agents_who_see_a_disagreement(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=20, base_p=0.5)
    run_round0(ctx)
    run_arm(ctx, "dissent_only")
    expected = 0
    for p in ctx.problems:
        for rep in range(2):
            views = [agent_view(g) for g in round0_gens(ctx, p.problem_id, rep)]
            for i in range(3):
                expected += disagrees(
                    views[i], [v for j, v in enumerate(views) if j != i]
                )
    got = [g for g in ctx.gens if g.arm == "dissent_only"]
    assert len(got) == expected and 0 < expected < 20 * 2 * 3
    for r in ctx.backend.by_arm("dissent_only"):
        r0 = round0_gens(ctx, r.meta["problem_id"], r.meta["replicate"])
        assert all(
            marker(g) not in r.messages[0]["content"]
            for g in r0
            if g.agent != r.meta["agent"]
        )


def test_final_generations_prefers_the_revision_when_it_exists(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3)
    run_all(ctx, arms=["self_reflect"])
    p = ctx.problems[0].problem_id
    finals = final_generations(ctx, "self_reflect", p, 0)
    assert [g.arm for g in finals] == ["self_reflect"] * 3
    assert [g.agent for g in finals] == [0, 1, 2]


def test_final_generations_for_a_missing_arm_is_an_error(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3)
    run_round0(ctx)
    with pytest.raises(LookupError):
        final_generations(ctx, "self_reflect", ctx.problems[0].problem_id, 0)


def test_generation_records_are_traceable_to_template_and_backend(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3)
    run_all(ctx, arms=["answer_only"])
    for g in ctx.gens:
        assert g.prompt_version == "v1"
        assert g.backend == "mock" and g.backend_version
        assert g.temperature == 0.7 and g.max_tokens == 512
        expected = channels.template_hash(channels.TEMPLATE_FOR[g.arm])
        assert g.prompt_hash == expected
        assert g.model_id == "mock/model"


def test_extracted_answers_come_from_the_text(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3)
    run_round0(ctx)
    g = next(iter(ctx.gens))
    assert agent_view(g).answer == extract_boxed(g.text)


def test_create_run_refuses_a_different_config_for_an_existing_run(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=3)
    with pytest.raises(ValueError):
        create_run(
            tmp_path / "run",
            RunConfig(model_id="other", replicates=2),
            ctx.problems,
            ctx.backend,
        )
