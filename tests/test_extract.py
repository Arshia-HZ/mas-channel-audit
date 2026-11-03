"""Answer and confidence parsing. Never falls back to raw-text guessing."""

from __future__ import annotations

import pytest

from channel_audit.extract import extract_boxed, extract_confidence, extract_final


def test_simple_box():
    assert extract_boxed(r"so \boxed{42}.") == "42"


def test_nested_braces():
    assert extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{\sqrt{\frac{a}{b}}}") == r"\sqrt{\frac{a}{b}}"


def test_escaped_braces_do_not_unbalance_the_match():
    assert extract_boxed(r"\boxed{\{1, 2\}}") == r"\{1, 2\}"


def test_missing_box_is_none():
    assert extract_boxed("The answer is 42.") is None
    assert extract_boxed("") is None


def test_answer_phrases_are_never_used_as_a_fallback():
    assert extract_boxed("#### 42") is None
    assert extract_boxed("Final answer: 42") is None


def test_multiple_boxes_last_wins():
    assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"


def test_truncated_final_box_is_none_even_if_an_earlier_box_is_complete():
    assert extract_boxed(r"\boxed{3} ... \boxed{12") is None


def test_truncated_single_box_is_none():
    assert extract_boxed(r"the answer is \boxed{\frac{1}{") is None


def test_empty_box_is_none():
    assert extract_boxed(r"\boxed{}") is None
    assert extract_boxed(r"\boxed{   }") is None


def test_whitespace_inside_box_is_stripped_and_space_before_brace_allowed():
    assert extract_boxed(r"\boxed {  7 }") == "7"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("blah\nConfidence: 85", 85),
        ("Confidence: 0", 0),
        ("Confidence: 100", 100),
        ("confidence = 70", 70),
        ("Confidence: 90%", 90),
        ("Confidence: 40\nConfidence: 60", 60),  # last wins
        ("no confidence line", None),
        ("Confidence: 101", None),
        ("Confidence: high", None),
        ("Confidence: 85.5", None),
        ("Confidence: -5", None),
        ("Confidence: 40\nConfidence: 999", None),  # last occurrence is invalid
    ],
)
def test_confidence_parsing(text, expected):
    assert extract_confidence(text) == expected


def test_final_boxed():
    r = extract_final(r"\boxed{5}" + "\nConfidence: 80", finish_reason="stop")
    assert (r.answer, r.status) == ("5", "boxed")


def test_final_none_when_no_box_and_finished():
    r = extract_final("It is five.", finish_reason="stop")
    assert (r.answer, r.status) == (None, "none")


def test_final_truncated_when_cut_off_without_complete_box():
    r = extract_final(r"so \boxed{1", finish_reason="length")
    assert (r.answer, r.status) == (None, "truncated")
    r = extract_final("long reasoning with no box", finish_reason="length")
    assert (r.answer, r.status) == (None, "truncated")


def test_final_complete_box_wins_even_when_length_limited():
    r = extract_final(r"\boxed{9} Confidence: 5", finish_reason="length")
    assert (r.answer, r.status) == ("9", "boxed")


def test_extract_never_raises_on_odd_input():
    for bad in ["\\boxed{", "\\boxed", "{" * 1000, "\x00", "\\boxed{" * 50]:
        extract_boxed(bad)
        extract_confidence(bad)
