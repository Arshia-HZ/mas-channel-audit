"""The v2 failure-message prompt sits beside the frozen v1 files, untouched."""

from __future__ import annotations

from pathlib import Path

from channel_audit import channels
from channel_audit.prompt_freeze import verify_manifest, write_manifest

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "prompts" / "v1"
V2 = ROOT / "prompts" / "v2"


def test_the_v1_prompts_are_still_frozen_and_unchanged():
    manifest = verify_manifest(V1)
    assert manifest["combined_sha256"].startswith("545ebeded7e1")
    assert "failure_msg_v2.txt" not in manifest["files"]


def test_v2_holds_only_the_new_failure_message_template():
    assert sorted(p.name for p in V2.glob("*.txt")) == ["failure_msg.txt"]


def test_once_v2_is_frozen_its_files_must_match_the_manifest():
    """v2 is a draft until the owner approves the wording and runs the freeze."""
    import pytest

    if not (V2 / "MANIFEST.json").exists():
        pytest.skip("prompts/v2 is not frozen yet (owner approves the wording first)")
    assert set(verify_manifest(V2)["files"]) == {"failure_msg.txt"}
    with pytest.raises(FileExistsError):
        write_manifest(V2)


def test_the_v2_template_asks_for_one_specific_sentence_and_no_solution():
    text = (V2 / "failure_msg.txt").read_text(encoding="utf-8")
    for needle in ("<<question>>", "<<solution>>", "single sentence"):
        assert needle in text
    assert "Do not solve the problem" in text and "final answer" in text
    assert "step number" in text or "quoting" in text
    assert "No error" not in text  # no correctness signal: a step is always named


def test_the_variant_uses_its_own_hash_and_the_old_one_is_unchanged():
    v1 = channels.template_hash("failure_msg")
    v2 = channels.template_hash("failure_msg_v2")
    assert v1 != v2
    assert v1 == verify_manifest(V1)["files"]["failure_msg.txt"]
    if (V2 / "MANIFEST.json").exists():
        assert v2 == verify_manifest(V2)["files"]["failure_msg.txt"]


def test_the_v2_revision_arm_reuses_the_v1_revision_template():
    assert channels.template_hash("failure_cause_v2") == channels.template_hash(
        "failure_cause"
    )
    assert channels.VARIANT_ARMS == ("failure_cause_v2",)
    assert "failure_cause_v2" not in channels.ARMS  # v1 runs and checks are unchanged


def test_building_the_v2_prompt_fills_the_question_and_the_solution():
    msgs = channels.build_failure_msg_prompt("What is 2+2?", "It is 4.", variant="v2")
    body = msgs[0]["content"]
    assert "What is 2+2?" in body and "It is 4." in body and "<<" not in body
    assert "single sentence" in body
    old = channels.build_failure_msg_prompt("What is 2+2?", "It is 4.")
    assert old[0]["content"] != body and "single sentence" not in old[0]["content"]


def test_the_v2_revision_prompt_matches_failure_cause_with_the_new_message():
    own = channels.AgentView(0, "my solution", "4", 80)
    peers = [channels.AgentView(1, "x", "5", 60), channels.AgentView(2, "y", "4", 70)]
    a = channels.build_revision_prompt(
        "failure_cause", "Q?", own, peers, failure_message="Step 2 drops a sign."
    )
    b = channels.build_revision_prompt(
        "failure_cause_v2", "Q?", own, peers, failure_message="Step 2 drops a sign."
    )
    assert a == b


def test_failure_message_arms_are_not_graded_as_solutions():
    assert channels.NON_SOLUTION_ARMS == ("failure_msg", "failure_msg_v2")
