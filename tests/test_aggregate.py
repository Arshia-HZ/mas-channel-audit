"""Plurality voting with 1/m tie credit; unparsable answers never vote."""

from __future__ import annotations

import pytest

from channel_audit.aggregate import plurality, plurality_credit


def test_clear_majority_correct():
    assert plurality_credit(["4", "4", "5"], gold="4") == 1.0


def test_clear_majority_wrong():
    assert plurality_credit(["5", "5", "4"], gold="4") == 0.0


def test_three_way_tie_with_correct_among_them_gives_one_third():
    assert plurality_credit(["4", "5", "6"], gold="4") == pytest.approx(1 / 3)


def test_two_way_tie_gives_half():
    assert plurality_credit(["4", "4", "5", "5"], gold="4") == pytest.approx(1 / 2)


def test_tie_without_the_correct_answer_gives_zero():
    assert plurality_credit(["5", "6", "7"], gold="4") == 0.0


def test_correct_answer_outside_the_tied_top_gets_nothing():
    assert plurality_credit(["5", "5", "6", "6", "4"], gold="4") == 0.0


def test_unparsable_answers_do_not_vote():
    # Two Nones must not outvote or tie the single real answer.
    assert plurality_credit([None, None, "4"], gold="4") == 1.0
    assert plurality_credit([None, None, "5"], gold="4") == 0.0
    assert plurality_credit(["", "  ", "4"], gold="4") == 1.0


def test_all_unparsable_scores_zero():
    assert plurality_credit([None, None, None], gold="4") == 0.0
    assert plurality_credit([], gold="4") == 0.0


def test_equivalent_spellings_pool_their_votes():
    assert plurality_credit([r"\frac{1}{2}", "0.5", "7"], gold="1/2") == 1.0
    assert plurality_credit(["0.5", r"\frac{1}{2}", "7"], gold="7") == 0.0


def test_plurality_reports_structure():
    p = plurality(["4", "4", "5", None])
    assert p.n_votes == 3
    assert p.n_tied == 1
    assert p.winners == ["4"]
    p = plurality(["4", "5", "6"])
    assert p.n_tied == 3 and set(p.winners) == {"4", "5", "6"}


def test_order_of_answers_does_not_change_credit():
    a = plurality_credit(["4", "5", "6"], gold="5")
    b = plurality_credit(["6", "5", "4"], gold="5")
    assert a == b == pytest.approx(1 / 3)
