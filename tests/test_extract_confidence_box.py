"""A boxed confidence is never the answer (found by the M4 screening)."""

from __future__ import annotations

import pytest

from channel_audit.channels import agent_view
from channel_audit.extract import extract_answer, extract_boxed, extract_final
from channel_audit.grade import grade_generation
from channel_audit.runner import run_round0
from helpers import build_ctx

NL = "\n"


def test_a_plain_answer_is_unchanged():
    assert extract_answer(r"so \boxed{42}.") == "42"
    assert extract_answer(r"\boxed{\frac{1}{2}} Confidence: 80") == r"\frac{1}{2}"
    assert extract_answer("no box here") is None


def test_a_boxed_confidence_line_is_skipped_for_the_earlier_answer():
    text = (
        r"Final answer: $\boxed{20}$" + NL + NL + "Confidence: 90" + NL
        + r"$$\boxed{\text{Confidence: 90}}$$"
    )  # fmt: skip
    assert extract_answer(text) == "20"
    assert extract_boxed(text) == r"\text{Confidence: 90}"  # low level unchanged


@pytest.mark.parametrize(
    "tail",
    [
        r"\boxed{\text{Confidence: 95}}",
        r"\boxed{Confidence: 95}",
        r"\boxed{\textbf{Confidence}: 95}",
        r"\boxed{95}",
        r"\boxed{95\%}",
        r"\boxed{N = 95}",
        r"\boxed{\text{95}}",
    ],
)
def test_every_boxed_form_of_the_confidence_is_skipped(tail):
    text = r"The answer is \boxed{7}." + NL + "Confidence: 95" + NL + tail
    assert extract_answer(text) == "7"


def test_several_trailing_confidence_boxes_are_all_skipped():
    text = r"\boxed{7} Confidence: 60 \boxed{60} \boxed{\text{Confidence: 60}}"
    assert extract_answer(text) == "7"


def test_a_number_equal_to_the_confidence_is_an_answer_when_nothing_else_is_boxed():
    assert extract_answer(r"\boxed{90}" + NL + "Confidence: 90") == "90"
    assert extract_answer(r"so it is \boxed{95}." + NL + "Confidence: 50") == "95"


def test_the_answer_may_equal_the_confidence_number():
    text = r"\boxed{90}" + NL + "Confidence: 90" + NL + r"\boxed{90}"
    assert extract_answer(text) == "90"


def test_a_lone_confidence_statement_is_not_an_answer():
    assert extract_answer(r"\boxed{\text{Confidence: 90}}") is None


def test_a_number_that_differs_from_the_stated_confidence_is_kept():
    text = r"\boxed{7}" + NL + "Confidence: 95" + NL + r"\boxed{12}"
    assert extract_answer(text) == "12"


def test_an_unclosed_last_box_still_means_no_answer():
    assert extract_answer(r"\boxed{7} and then \boxed{8") is None


def test_the_word_confidence_inside_a_real_answer_is_kept():
    assert extract_answer(r"\boxed{95\%\text{ confidence interval}}") is not None


def test_extract_final_uses_the_answer_rule():
    stopped = r"\boxed{20}" + NL + "Confidence: 90" + NL + r"\boxed{90}"
    r = extract_final(stopped, "stop")
    assert (r.answer, r.status) == ("20", "boxed")
    cut = r"long \boxed{20}" + NL + "Confidence: 90" + NL + r"\boxed{"
    r = extract_final(cut, "length")
    assert r.answer is None and r.status == "truncated"


def test_grading_and_peer_views_use_the_answer_not_the_confidence(tmp_path):
    ctx = build_ctx(tmp_path, n_problems=2, replicates=1, base_p=1.0)
    run_round0(ctx)
    gen = next(iter(ctx.gens))
    gold = next(p.gold for p in ctx.problems if p.problem_id == gen.problem_id)
    edited = gen.model_copy(
        update={"text": gen.text + NL + r"\boxed{\text{Confidence: 88}}"}
    )
    assert grade_generation(edited, gold).correct
    assert agent_view(edited).answer == gold


@pytest.mark.parametrize(
    "tail",
    [
        r"\text{Confidence: } \boxed{90}",
        r"**Confidence**: $\boxed{90}$",
        r"Confidence level: \boxed{90}",
        r"Confidence (out of 100): \boxed{85}",
    ],
)
def test_a_box_after_the_word_confidence_is_skipped_even_without_a_number(tail):
    text = r"\boxed{\frac{7}{3}}" + NL + NL + tail
    assert extract_answer(text) == r"\frac{7}{3}"


def test_a_lone_box_after_the_word_confidence_is_no_answer():
    assert extract_answer(r"The answer is 7." + NL + r"Confidence: \boxed{90}") is None


def test_a_box_on_a_later_line_is_not_affected_by_an_earlier_confidence_word():
    text = "Confidence is high, so:" + NL + r"\boxed{12}"
    assert extract_answer(text) == "12"


# --- the confidence VALUE when the model wraps or boxes it -------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"Confidence: } \boxed{100}", 100),
        (r"\text{Confidence: } \boxed{90}", 90),
        (r"Confidence: $ \boxed{75} $", 75),
        (r"Confidence: $\boxed{60}$", 60),
        (r"Confidence: **70**", 70),
        (r"**Confidence**: 85", 85),
        (r"Confidence: \(95\)", 95),
        (r"\boxed{\text{Confidence: 88}}", 88),
        (r"\boxed{\text{Confidence: } 77}", 77),
        (r"Confidence level: \boxed{40}", 40),
        (r"Confidence: **N = 100** (I believe this is correct)", 100),
        (r'Confidence: N", where N is an integer', None),
        (r"Confidence: high", None),
        (r"Confidence: \boxed{x}", None),
        (r"Confidence is 5 steps", None),
    ],
)
def test_confidence_value_is_read_through_wrappers_and_boxes(text, expected):
    from channel_audit.extract import extract_confidence

    assert extract_confidence(text) == expected


def test_the_last_confidence_still_wins():
    from channel_audit.extract import extract_confidence

    text = "Confidence: 30\n...\n" + r"\text{Confidence: } \boxed{95}"
    assert extract_confidence(text) == 95


def test_a_boxed_confidence_value_is_not_the_answer_and_is_not_lost():
    from channel_audit.extract import extract_confidence

    text = r"\boxed{12}" + NL + r"\text{Confidence: } \boxed{90}"
    assert extract_answer(text) == "12"
    assert extract_confidence(text) == 90
