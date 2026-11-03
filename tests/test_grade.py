"""The only grader. Equivalence cases ported from Phase 1 plus new ones."""

from __future__ import annotations

import importlib
import sys

import pytest

from channel_audit import grade
from channel_audit.grade import answers_equivalent, grade_generation, is_correct
from conftest import make_generation


@pytest.mark.parametrize(
    "pred, gold",
    [
        ("42", "42"),
        (r"\frac{1}{2}", "0.5"),
        (r"\frac{1}{2}", "1/2"),
        ("0.50", "0.5"),
        ("12.0", "12"),
        ("1,000", "1000"),  # thousands separator
        ("12,345", "12345"),
        (r"10\text{ cm}", "10"),  # units
        ("10 cm", "10"),
        (r"90^\circ", "90"),
        (r"5\%", "5"),
        (r"\dfrac{3}{4}", r"\frac{3}{4}"),
        (r"\frac34", r"\frac{3}{4}"),
        (r"\sqrt2", r"\sqrt{2}"),
        ("(1,2)", "(1, 2)"),
        (r"\text{(B)}", "(B)"),
        ("x=2", "2"),
    ],
)
def test_equivalent_answers(pred, gold):
    assert is_correct(pred, gold), (pred, gold)


@pytest.mark.parametrize(
    "pred, gold",
    [
        ("17", "42"),
        ("3", "4"),
        (r"\frac{1}{2}", r"\frac{1}{3}"),
        ("(1,2)", "(2,1)"),
        ("1000", "100"),
        ("abc", "abd"),
        ("", "42"),
    ],
)
def test_different_answers_are_not_equal(pred, gold):
    assert not is_correct(pred, gold), (pred, gold)


def test_none_extraction_is_never_correct():
    assert not is_correct(None, "42")
    assert not is_correct(None, "")
    assert not is_correct("42", None)


def test_identical_text_answers_still_match():
    # math_verify cannot parse these; the normalised exact-match path must.
    assert is_correct("abc", "abc")
    assert is_correct(r"\text{Evelyn}", "Evelyn")


def test_grader_is_symmetric_on_equivalence():
    assert answers_equivalent("0.5", r"\frac{1}{2}") == answers_equivalent(
        r"\frac{1}{2}", "0.5"
    )


def test_grade_generation_boxed_correct():
    g = make_generation(text="so \\boxed{\\frac{1}{2}}\nConfidence: 80")
    r = grade_generation(g, gold="0.5")
    assert (r.gen_key, r.extracted, r.extraction_status) == (
        g.gen_key,
        r"\frac{1}{2}",
        "boxed",
    )
    assert r.confidence == 80 and r.correct is True


def test_grade_generation_boxed_wrong():
    g = make_generation(text="\\boxed{3}\nConfidence: 10")
    r = grade_generation(g, gold="4")
    assert r.correct is False and r.extraction_status == "boxed"


def test_grade_generation_without_box_is_incorrect_not_raw_matched():
    g = make_generation(text="The answer is 4.", finish_reason="stop")
    r = grade_generation(g, gold="4")
    assert (r.extracted, r.extraction_status, r.correct) == (None, "none", False)


def test_grade_generation_truncated():
    g = make_generation(text="so \\boxed{4", finish_reason="length")
    r = grade_generation(g, gold="4")
    assert (r.extracted, r.extraction_status, r.correct) == (None, "truncated", False)


def test_import_fails_loudly_without_math_verify(monkeypatch):
    monkeypatch.setitem(sys.modules, "math_verify", None)  # makes import raise
    monkeypatch.delitem(sys.modules, "channel_audit.grade", raising=False)
    with pytest.raises(ImportError, match="math-verify"):
        importlib.import_module("channel_audit.grade")


def test_startup_selftest_detects_a_silently_broken_verifier(monkeypatch):
    # Phase 1 hazard: a verifier that returns False for everything.
    import math_verify

    monkeypatch.setattr(math_verify, "verify", lambda *a, **k: False)
    monkeypatch.delitem(sys.modules, "channel_audit.grade", raising=False)
    with pytest.raises(RuntimeError, match="self-test"):
        importlib.import_module("channel_audit.grade")


def test_module_imported_normally_still_works():
    assert grade.is_correct("2", "2")
