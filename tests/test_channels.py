"""Channels are pure functions from round-0 views to prompts."""

from __future__ import annotations

import hashlib

import pytest

from channel_audit import channels
from channel_audit.channels import (
    ARMS,
    PROMPT_VERSION,
    PROMPTS_DIR,
    AgentView,
    build_failure_msg_prompt,
    build_revision_prompt,
    build_round0_prompt,
    disagrees,
    render,
    strip_confidence,
    template_hash,
)

QUESTION = r"What is $\frac{1}{2}+\frac{1}{3}$?"


def view(agent, answer, conf, tag):
    text = (
        f"{tag} reasoning for agent {agent}.\n\\boxed{{{answer}}}\nConfidence: {conf}"
    )
    return AgentView(agent=agent, text=text, answer=answer, confidence=conf)


OWN = view(0, "4", 61, "OWN")
P1 = view(1, "5", 93, "PEERONE")
P2 = view(2, "6", 17, "PEERTWO")


def content(messages):
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return messages[0]["content"]


def test_self_reflect_has_own_solution_and_no_peer_text():
    c = content(build_revision_prompt("self_reflect", QUESTION, OWN, [P1, P2]))
    assert "OWN reasoning" in c
    assert "PEERONE" not in c and "PEERTWO" not in c
    assert "Solver" not in c


def test_answer_only_has_peer_answers_but_no_reasoning_or_confidence():
    c = content(build_revision_prompt("answer_only", QUESTION, OWN, [P1, P2]))
    assert "Solver 2: 5" in c and "Solver 3: 6" in c
    assert "PEERONE" not in c and "PEERTWO" not in c
    assert "93" not in c and "17" not in c


def test_answer_conf_adds_confidence_but_no_reasoning():
    c = content(build_revision_prompt("answer_conf", QUESTION, OWN, [P1, P2]))
    assert "Solver 2: 5 (confidence 93)" in c
    assert "Solver 3: 6 (confidence 17)" in c
    assert "PEERONE" not in c and "PEERTWO" not in c


def test_missing_answer_and_confidence_are_marked_not_dropped():
    bare = AgentView(agent=1, text="no box", answer=None, confidence=None)
    c = content(build_revision_prompt("answer_conf", QUESTION, OWN, [bare, P2]))
    assert "Solver 2: (no final answer) (confidence not given)" in c


def test_full_solution_shows_peer_text_without_their_confidence_line():
    c = content(build_revision_prompt("full_solution", QUESTION, OWN, [P1, P2]))
    assert "PEERONE reasoning" in c and "PEERTWO reasoning" in c
    assert "Solver 2:" in c and "Solver 3:" in c
    assert "Confidence: 93" not in c and "Confidence: 17" not in c
    assert "Confidence: 61" in c  # own solution is shown as written


def test_placebo_renders_exactly_like_full_solution_for_the_same_texts():
    real = build_revision_prompt("full_solution", QUESTION, OWN, [P1, P2])
    placebo = build_revision_prompt(
        "placebo", QUESTION, OWN, [P1, P2], donor_texts=[P1.text, P2.text]
    )
    assert real == placebo


def test_placebo_uses_donor_text_not_the_real_peers():
    c = content(
        build_revision_prompt(
            "placebo", QUESTION, OWN, [P1, P2], donor_texts=["DONORA", "DONORB"]
        )
    )
    assert "DONORA" in c and "DONORB" in c
    assert "PEERONE" not in c and "PEERTWO" not in c


def test_placebo_requires_donor_texts_of_the_right_length():
    with pytest.raises(ValueError):
        build_revision_prompt("placebo", QUESTION, OWN, [P1, P2])
    with pytest.raises(ValueError):
        build_revision_prompt("placebo", QUESTION, OWN, [P1, P2], donor_texts=["x"])


def test_dissent_only_is_none_when_everyone_agrees():
    same = [view(1, "4", 10, "A"), view(2, "4.0", 20, "B")]
    assert build_revision_prompt("dissent_only", QUESTION, OWN, same) is None


def test_dissent_only_when_a_peer_differs_reveals_nothing_else():
    c = content(build_revision_prompt("dissent_only", QUESTION, OWN, [P1, P2]))
    assert "different answer" in c
    for leak in ("PEERONE", "PEERTWO", "93", "Solver"):
        assert leak not in c


def test_disagreement_rules():
    bare = AgentView(agent=1, text="", answer=None, confidence=None)
    assert disagrees(OWN, [P1, P2])
    assert not disagrees(OWN, [view(1, "4", 1, "x"), view(2, r"\frac{8}{2}", 1, "y")])
    assert disagrees(OWN, [bare, view(2, "4", 1, "y")])  # no answer differs
    assert not disagrees(bare, [bare, bare])  # nothing vs nothing agrees


def test_failure_cause_shows_only_the_sanitized_message():
    c = content(
        build_revision_prompt(
            "failure_cause",
            QUESTION,
            OWN,
            [P1, P2],
            failure_message="Step 2 divides by zero.",
        )
    )
    assert "Step 2 divides by zero." in c
    assert "PEERONE" not in c and "PEERTWO" not in c
    assert "Solver" not in c


def test_failure_cause_requires_a_message():
    with pytest.raises(ValueError):
        build_revision_prompt("failure_cause", QUESTION, OWN, [P1, P2])


def test_failure_msg_prompt_asks_for_no_final_answer_and_hides_confidence():
    c = content(build_failure_msg_prompt(QUESTION, P1.text))
    assert "Do not give a final answer" in c
    assert "PEERONE reasoning" in c
    assert "Confidence: 93" not in c
    assert not c.rstrip().endswith("\\boxed{}.")


def test_round0_prompt_asks_for_box_and_confidence():
    c = content(build_round0_prompt(QUESTION))
    assert QUESTION in c and "Confidence: N" in c
    assert c.rstrip().endswith("Put your final answer in \\boxed{}.")


@pytest.mark.parametrize("arm", ARMS)
def test_every_revision_prompt_ends_with_the_box_instruction(arm):
    msgs = build_revision_prompt(
        arm,
        QUESTION,
        OWN,
        [P1, P2],
        failure_message="m",
        donor_texts=["a", "b"],
    )
    assert content(msgs).rstrip().endswith("Put your final answer in \\boxed{}.")


def test_latex_braces_in_question_survive_rendering():
    assert QUESTION in content(build_round0_prompt(QUESTION))


def test_render_is_single_pass_so_inserted_text_is_never_re_expanded():
    out = render("A <<x>> B <<y>>", x="<<y>>", y="Y")
    assert out == "A <<y>> B Y"


def test_render_rejects_unknown_or_missing_placeholders():
    with pytest.raises(KeyError):
        render("<<x>>", y="1")
    with pytest.raises(KeyError):
        render("<<x>> <<z>>", x="1")


def test_strip_confidence_removes_only_confidence_lines():
    text = "a\nConfidence: 80\nb\n  confidence = 5  \nc"
    assert strip_confidence(text) == "a\nb\nc"


def test_template_files_exist_and_hash_is_sha256_of_lf_normalised_bytes():
    path = PROMPTS_DIR / PROMPT_VERSION / "full_solution.txt"
    data = path.read_bytes().replace(b"\r\n", b"\n")
    assert template_hash("full_solution") == hashlib.sha256(data).hexdigest()


def test_placebo_template_is_byte_identical_to_full_solution():
    assert channels.TEMPLATE_FOR["placebo"] == channels.TEMPLATE_FOR["full_solution"]
    assert template_hash("placebo") == template_hash("full_solution")
    own_file = PROMPTS_DIR / PROMPT_VERSION / "placebo.txt"
    if own_file.exists():
        full = (PROMPTS_DIR / PROMPT_VERSION / "full_solution.txt").read_bytes()
        assert own_file.read_bytes() == full


def test_every_arm_has_a_template_file():
    for name in set(channels.TEMPLATE_FOR.values()):
        assert (PROMPTS_DIR / PROMPT_VERSION / f"{name}.txt").is_file()
