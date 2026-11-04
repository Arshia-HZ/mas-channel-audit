"""MATH test set: pinned revision, usable-row filter, and the frozen hash split."""

from __future__ import annotations

import hashlib
import random
import re

import pytest

from channel_audit import data
from channel_audit.data import (
    MATH_REVISION,
    N_PILOT,
    N_SCREEN,
    assign_splits,
    load_frozen,
    load_math_test,
    split_key,
    write_split,
)

PER_CONFIG = 220


def fake_rows(config: str) -> list[dict]:
    """220 rows per subject, levels cycling 1..5, every tenth row unusable."""
    rows = []
    for i in range(PER_CONFIG):
        level = f"Level {i % 5 + 1}"
        solution = f"So the answer is $\\boxed{{{i}}}$."
        if i % 10 == 7:
            solution = "No boxed answer in this one."
        if i % 50 == 49:
            level = "Level ?"
        rows.append(
            {
                "problem": f"{config} question {i}",
                "level": level,
                "type": config,
                "solution": solution,
            }  # fmt: skip
        )
    return rows


@pytest.fixture(scope="module")
def loaded():
    calls = []

    def loader(config, revision):
        calls.append((config, revision))
        return fake_rows(config)

    problems, dropped = load_math_test(loader=loader)
    return problems, dropped, calls


@pytest.fixture(scope="module")
def split(loaded):
    return assign_splits(loaded[0])


def test_the_dataset_revision_is_pinned_to_a_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", MATH_REVISION)


def test_every_subject_is_loaded_at_the_pinned_revision(loaded):
    calls = loaded[2]
    assert {c for c, _ in calls} == set(data.CONFIGS)
    assert {r for _, r in calls} == {MATH_REVISION}


def test_ids_are_stable_row_positions_and_unusable_rows_are_dropped(loaded):
    problems = loaded[0]
    ids = {p.problem_id for p in problems}
    assert "math-test-algebra-0" in ids
    assert "math-test-algebra-7" not in ids  # no boxed gold
    assert "math-test-algebra-49" not in ids  # unparsable level
    assert all(p.dataset == "math-test" for p in problems)
    assert all(1 <= p.level <= 5 for p in problems)
    assert (
        next(p for p in problems if p.problem_id == "math-test-algebra-3").gold == "3"
    )


def test_dropped_rows_are_reported_not_silent(loaded):
    problems, dropped, _ = loaded
    n_rows = PER_CONFIG * len(data.CONFIGS)
    assert len(dropped) + len(problems) == n_rows
    assert "math-test-algebra-7" in dropped


def test_split_sizes_follow_the_plan(split):
    assert len(split.screen) == N_SCREEN == 300
    assert len(split.pilot) == N_PILOT == 200
    assert len(split.confirm) > 0


def test_pilot_and_confirm_are_levels_3_to_5_and_screen_spans_all_levels(split):
    assert {p.level for p in split.pilot} <= {3, 4, 5}
    assert {p.level for p in split.confirm} <= {3, 4, 5}
    assert {p.level for p in split.screen} == {1, 2, 3, 4, 5}


def test_splits_are_disjoint_and_cover_every_usable_problem(loaded, split):
    problems = loaded[0]
    groups = [split.screen, split.pilot, split.confirm]
    ids = [{p.problem_id for p in g} for g in groups]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    excluded = set(split.excluded_ids)
    assert excluded.isdisjoint(ids[0] | ids[1] | ids[2])
    assert ids[0] | ids[1] | ids[2] | excluded == {p.problem_id for p in problems}


def test_only_low_levels_outside_screen_are_excluded(loaded, split):
    problems = loaded[0]
    level = {p.problem_id: p.level for p in problems}
    assert split.excluded_ids and all(level[i] in (1, 2) for i in split.excluded_ids)


def test_each_problem_carries_its_split_name(split):
    assert {p.split for p in split.screen} == {"screen"}
    assert {p.split for p in split.pilot} == {"pilot"}
    assert {p.split for p in split.confirm} == {"confirm"}


def test_the_split_does_not_depend_on_row_order(loaded, split):
    shuffled = list(loaded[0])
    random.Random(5).shuffle(shuffled)
    again = assign_splits(shuffled)
    assert [p.problem_id for p in again.screen] == [p.problem_id for p in split.screen]
    assert [p.problem_id for p in again.pilot] == [p.problem_id for p in split.pilot]
    assert [p.problem_id for p in again.confirm] == [
        p.problem_id for p in split.confirm
    ]


def test_the_hash_is_a_fixed_function_of_the_problem_id():
    expected = hashlib.sha256(b"mas-channel-audit/split/v1|math-test-algebra-0")
    assert split_key("math-test-algebra-0") == expected.hexdigest()
    assert split_key("math-test-algebra-0") != split_key("math-test-algebra-1")


def test_too_few_problems_is_an_error_not_a_short_split(loaded):
    with pytest.raises(ValueError, match="screen"):
        assign_splits(loaded[0][:100])
    levels35 = [p for p in loaded[0] if p.level >= 3]
    only_low = [p for p in loaded[0] if p.level < 3][:50] + levels35[:350]
    with pytest.raises(ValueError, match="pilot"):
        assign_splits(only_low)


def test_write_split_records_revision_salt_counts_and_file_hashes(tmp_path, split):
    manifest = write_split(split, tmp_path, n_rows=1540)
    assert manifest["dataset"] == data.MATH_DATASET
    assert manifest["revision"] == MATH_REVISION
    assert manifest["salt"] == data.SPLIT_SALT
    assert manifest["counts"]["screen"] == 300 and manifest["counts"]["pilot"] == 200
    for name in ("screen", "pilot", "confirm"):
        text = (tmp_path / f"{name}.jsonl").read_bytes()
        assert manifest["files"][name]["sha256"] == hashlib.sha256(text).hexdigest()
    assert (tmp_path / "split_manifest.json").exists()


def test_load_frozen_returns_the_problems_and_checks_the_hash(tmp_path, split):
    write_split(split, tmp_path, n_rows=1540)
    back = load_frozen(tmp_path, "screen")
    assert [p.problem_id for p in back] == [p.problem_id for p in split.screen]
    path = tmp_path / "screen.jsonl"
    path.write_bytes(path.read_bytes().replace(b"question", b"qUestion", 1))
    with pytest.raises(ValueError, match="sha256"):
        load_frozen(tmp_path, "screen")


def test_writing_the_same_split_again_is_fine_but_a_change_is_refused(tmp_path, split):
    first = write_split(split, tmp_path, n_rows=1540)
    assert write_split(split, tmp_path, n_rows=1540) == first
    smaller = type(split)(
        screen=split.screen,
        pilot=split.pilot[:-1],
        confirm=split.confirm,
        excluded_ids=split.excluded_ids,
    )
    with pytest.raises(ValueError, match="frozen"):
        write_split(smaller, tmp_path, n_rows=1540)


def test_a_committed_manifest_lets_the_files_be_rebuilt_and_checked(tmp_path, split):
    write_split(split, tmp_path, n_rows=1540)
    for name in ("screen", "pilot", "confirm"):
        (tmp_path / f"{name}.jsonl").unlink()  # only the manifest is committed
    write_split(split, tmp_path, n_rows=1540)
    assert [p.problem_id for p in load_frozen(tmp_path, "pilot")] == [
        p.problem_id for p in split.pilot
    ]
