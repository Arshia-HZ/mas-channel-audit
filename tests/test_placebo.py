"""Placebo donors: same subject and level, nearest length, never the same problem."""

from __future__ import annotations

import pytest

from channel_audit.placebo import NoDonorError, pick_donor
from channel_audit.schema import Problem


def prob(pid, subject="Algebra", level=3):
    return Problem(
        problem_id=pid,
        dataset="fake",
        subject=subject,
        level=level,
        question="q",
        gold="1",
        split="pilot",
    )


ME = prob("me")


def test_picks_nearest_token_length():
    pool = [ME, prob("a"), prob("b"), prob("c")]
    tokens = {"a": 100, "b": 480, "c": 900}
    assert pick_donor(ME, pool, 500, tokens, tiebreak="t").problem_id == "b"


def test_never_the_same_problem_even_if_it_is_the_closest():
    pool = [ME, prob("a")]
    tokens = {"me": 500, "a": 5}
    assert pick_donor(ME, pool, 500, tokens, tiebreak="t").problem_id == "a"


def test_same_subject_and_level_only():
    pool = [
        ME,
        prob("other_subject", subject="Geometry"),
        prob("other_level", level=4),
        prob("ok"),
    ]
    tokens = {"other_subject": 500, "other_level": 500, "ok": 1}
    assert pick_donor(ME, pool, 500, tokens, tiebreak="t").problem_id == "ok"


def test_candidates_without_token_counts_are_ignored():
    pool = [ME, prob("a"), prob("b")]
    assert pick_donor(ME, pool, 500, {"b": 1}, tiebreak="t").problem_id == "b"


def test_raises_when_no_donor_exists():
    with pytest.raises(NoDonorError):
        pick_donor(ME, [ME], 500, {"me": 500}, tiebreak="t")
    with pytest.raises(NoDonorError):
        pick_donor(ME, [ME, prob("g", subject="Geometry")], 5, {"g": 5}, tiebreak="t")


def test_ties_break_deterministically_and_depend_on_the_tiebreak_key():
    pool = [ME] + [prob(f"p{i}") for i in range(20)]
    tokens = {f"p{i}": 500 for i in range(20)}
    first = pick_donor(ME, pool, 500, tokens, tiebreak="k1").problem_id
    assert pick_donor(ME, pool, 500, tokens, tiebreak="k1").problem_id == first
    others = {
        pick_donor(ME, pool, 500, tokens, tiebreak=f"k{i}").problem_id
        for i in range(40)
    }
    assert len(others) > 1


def test_pool_order_does_not_change_the_choice():
    pool = [ME] + [prob(f"p{i}") for i in range(10)]
    tokens = {f"p{i}": 500 for i in range(10)}
    a = pick_donor(ME, pool, 500, tokens, tiebreak="k").problem_id
    b = pick_donor(ME, list(reversed(pool)), 500, tokens, tiebreak="k").problem_id
    assert a == b
