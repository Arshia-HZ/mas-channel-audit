"""Stored grades must equal what the frozen grader gives now."""

from __future__ import annotations

import json
import shutil

import pytest

from channel_audit.analysis.integrity import grade_mismatches
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import LazyBackend, run_shards


@pytest.fixture(scope="module")
def shard(tmp_path_factory):
    root = tmp_path_factory.mktemp("integrity")
    problems = fake_problems(8)
    world = MockWorld.from_problems(problems, base_p=0.5)
    cfg = RunConfig(model_id="mock/model", replicates=2, max_tokens=512)
    run_shards(
        root,
        cfg,
        problems,
        LazyBackend(lambda: MockBackend(world)),
        size=50,
        arms=(*ARMS, *VARIANT_ARMS),
        log=lambda s: None,
    )
    return root / "shard_00"


def test_a_clean_shard_has_no_mismatch(shard):
    assert grade_mismatches(open_run(shard)) == []


def _edit_graded(src, dst, edit):
    shutil.copytree(src, dst)
    path = dst / "graded.jsonl"
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    edit(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return open_run(dst)


def test_a_flipped_correct_flag_is_reported_with_the_generation(shard, tmp_path):
    def edit(rows):
        rows[5]["correct"] = not rows[5]["correct"]

    ctx = _edit_graded(shard, tmp_path / "flip", edit)
    bad = grade_mismatches(ctx)
    assert len(bad) == 1 and "/" in bad[0]


def test_a_changed_extracted_answer_or_status_is_reported(shard, tmp_path):
    def edit(rows):
        rows[0]["extracted"] = "999999"
        rows[1]["extraction_status"] = "none"

    assert len(grade_mismatches(_edit_graded(shard, tmp_path / "ext", edit))) == 2


def test_a_missing_grade_is_reported_not_raised(shard, tmp_path):
    def edit(rows):
        del rows[3]

    bad = grade_mismatches(_edit_graded(shard, tmp_path / "gone", edit))
    assert len(bad) == 1 and "no stored grade" in bad[0]
