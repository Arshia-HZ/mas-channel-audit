"""Failure messages must not carry a peer's final answer."""

from __future__ import annotations

from channel_audit.extract import strip_boxed
from channel_audit.leakage import find_leaks, redact, sanitize_message


def test_strip_boxed_removes_nested_boxes_and_counts_them():
    text, n = strip_boxed(r"so \boxed{\frac{1}{2}} and \boxed{3}.", "X")
    assert (text, n) == ("so X and X.", 2)


def test_strip_boxed_removes_an_unclosed_box_to_the_end():
    text, n = strip_boxed(r"keep \boxed{12 and more", "X")
    assert (text, n) == ("keep X", 1)


def test_strip_boxed_leaves_clean_text_alone():
    assert strip_boxed("nothing here", "X") == ("nothing here", 0)


def test_sanitize_removes_boxes_and_final_answer_sentences():
    msg = (
        "Step 2 drops a sign. So the final answer is 7. "
        "The correct result is \\boxed{7}.\nThe answer: 7"
    )
    clean, n = sanitize_message(msg)
    assert "boxed" not in clean
    assert "final answer" not in clean.lower() and "answer is" not in clean
    assert "Step 2 drops a sign." in clean
    assert n >= 3


def test_sanitize_keeps_a_decimal_inside_a_kept_sentence_intact():
    clean, _ = sanitize_message("Step 3 rounds 7.5 to 8 too early.")
    assert clean == "Step 3 rounds 7.5 to 8 too early."


def test_sanitized_text_never_contains_a_box_even_if_unclosed():
    clean, _ = sanitize_message("oops \\boxed{4")
    assert "boxed" not in clean


def test_claim_of_a_peer_answer_is_a_leak():
    assert find_leaks("This gives x = 7, which is wrong.", ["7"]) == ["7"]
    assert find_leaks("The result is 7.", ["7"]) == ["7"]
    assert find_leaks("so it equals $7$", ["7"]) == ["7"]


def test_a_number_that_is_not_a_claimed_result_is_not_a_leak():
    assert (
        find_leaks("Step 2 is wrong because 7 apples were counted twice.", ["2"]) == []
    )
    assert find_leaks("The first error is at step 7.", ["7"]) == []
    assert find_leaks("This gives x = 17.", ["7"]) == []
    assert find_leaks("This gives x = 7.5.", ["7"]) == []


def test_latex_answers_are_matched_literally():
    assert find_leaks(r"it gives \frac{1}{2} here", [r"\frac{1}{2}"]) == [
        r"\frac{1}{2}"
    ]


def test_none_and_blank_answers_are_ignored():
    assert find_leaks("x = 7", [None, "", "  "]) == []


def test_any_of_several_peer_answers_can_leak():
    assert find_leaks("so x = 9", ["7", "9"]) == ["9"]


def test_redact_replaces_only_the_leaked_value():
    out = redact("This gives x = 7, but step 7 is fine.", ["7"])
    assert out == "This gives x = [answer], but step 7 is fine."
    assert find_leaks(out, ["7"]) == []
