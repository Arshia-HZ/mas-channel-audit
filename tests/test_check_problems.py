"""The M3 check problems come from the MATH train split, never the test set."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from channel_audit.check_problems import (
    parse_level,
    row_to_problem,
    rows_from_jsonl,
    sample_check_problems,
)

SUBJECTS = ["Algebra", "Geometry", "Number Theory", "Prealgebra"]


def row(subject="Algebra", level="Level 3", solution=r"so $\boxed{12}$.", q="Q?"):
    return {"problem": q, "level": level, "type": subject, "solution": solution}


def test_level_parsing():
    assert parse_level("Level 3") == 3
    assert parse_level("Level 5") == 5
    assert parse_level("Level ?") is None
    assert parse_level("junk") is None


def test_row_becomes_a_check_problem_with_gold_from_the_last_box():
    p = row_to_problem(
        row(solution=r"\boxed{1} and finally \boxed{\frac{1}{2}}"), "t-1"
    )
    assert p.problem_id == "t-1"
    assert (p.dataset, p.split, p.level, p.subject) == (
        "math-train",
        "check",
        3,
        "Algebra",
    )
    assert p.gold == r"\frac{1}{2}"
    assert p.question == "Q?"


@pytest.mark.parametrize(
    "bad",
    [row(level="Level ?"), row(solution="no box here"), row(solution=r"\boxed{}")],
)
def test_unusable_rows_are_skipped_not_guessed(bad):
    assert row_to_problem(bad, "x") is None


def rows(n_per_group=6, levels=(1, 2, 3, 4, 5)):
    out = []
    for s in SUBJECTS:
        for lv in levels:
            for i in range(n_per_group):
                out.append(
                    (
                        f"{s}-{lv}-{i}",
                        row(s, f"Level {lv}", rf"\boxed{{{i}}}", f"{s}{lv}{i}"),
                    )
                )
    return [row_to_problem(r, pid) for pid, r in out]


def test_sample_only_uses_requested_levels():
    got = sample_check_problems(rows(), n=24, seed=0)
    assert {p.level for p in got} <= {3, 4, 5}
    assert len(got) == 24


def test_sample_is_deterministic_and_independent_of_input_order():
    a = sample_check_problems(rows(), n=24, seed=1)
    b = sample_check_problems(list(reversed(rows())), n=24, seed=1)
    assert [p.problem_id for p in a] == [p.problem_id for p in b]
    c = sample_check_problems(rows(), n=24, seed=2)
    assert [p.problem_id for p in a] != [p.problem_id for p in c]


def test_every_subject_level_group_has_a_placebo_donor():
    got = sample_check_problems(rows(), n=30, seed=0)
    groups = Counter((p.subject, p.level) for p in got)
    assert min(groups.values()) >= 2
    assert len(groups) == 12  # 4 subjects x levels 3-5


def test_ids_are_unique_and_sorted():
    ids = [p.problem_id for p in sample_check_problems(rows(), n=30, seed=0)]
    assert ids == sorted(set(ids))


def test_too_small_n_for_two_per_group_is_an_error():
    with pytest.raises(ValueError, match="at least"):
        sample_check_problems(rows(), n=10, seed=0)


def test_not_enough_eligible_problems_is_an_error():
    with pytest.raises(ValueError, match="only"):
        sample_check_problems(rows(n_per_group=2), n=100, seed=0)


def test_groups_with_a_single_problem_are_dropped_not_sampled():
    data = rows(n_per_group=3) + [
        row_to_problem(row("Precalculus", "Level 4"), "lonely")
    ]
    got = sample_check_problems(data, n=24, seed=0)
    assert "lonely" not in {p.problem_id for p in got}


def test_rows_from_jsonl_assigns_stable_ids(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(json.dumps(row(q=f"q{i}")) for i in range(3)), encoding="utf-8"
    )
    problems = rows_from_jsonl(path)
    assert [p.problem_id for p in problems] == [
        "math-train-file-0",
        "math-train-file-1",
        "math-train-file-2",
    ]
