"""Re-testing the v2 failure channel on copies of the finished pilot shards."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import (
    DONE_FILE,
    UPLOADED_FILE,
    LazyBackend,
    run_shards,
    run_variant_shards,
)

PROBLEMS = fake_problems(16)
WORLD = MockWorld.from_problems(PROBLEMS)


def cfg():
    return RunConfig(model_id="mock/model", replicates=2, max_tokens=512)


def factory():
    return MockBackend(WORLD)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def pilot(tmp_path):
    src = tmp_path / "pilot"
    run_shards(src, cfg(), PROBLEMS, LazyBackend(factory), size=8, log=lambda s: None)
    return src


class Recorder:
    def __init__(self):
        self.calls = []

    def upload(self, local: Path, remote: str):
        self.calls.append((local, remote))


def run_v2(pilot, tmp_path, **kw):
    return run_variant_shards(
        pilot,
        tmp_path / "retest",
        cfg(),
        PROBLEMS,
        kw.pop("backend", LazyBackend(factory)),
        size=8,
        log=lambda s: None,
        **kw,
    )


def test_each_shard_is_copied_and_gets_the_v2_arms(pilot, tmp_path):
    results = run_v2(pilot, tmp_path)
    assert [r.status for r in results] == ["generated", "generated"]
    for name in ("shard_00", "shard_01"):
        ctx = open_run(tmp_path / "retest" / name)
        arms = {g.arm for g in ctx.gens}
        assert {"failure_msg_v2", "failure_cause_v2", "failure_cause"} <= arms
        assert any(r.variant == "v2" for r in ctx.leakage)


def test_the_original_pilot_shards_are_never_touched(pilot, tmp_path):
    before = {
        f"{d.name}/{f.name}": digest(f)
        for d in pilot.iterdir()
        for f in d.iterdir()
        if f.is_file()
    }
    run_v2(pilot, tmp_path)
    after = {
        f"{d.name}/{f.name}": digest(f)
        for d in pilot.iterdir()
        for f in d.iterdir()
        if f.is_file()
    }
    assert before == after


def test_the_copies_keep_every_original_generation_byte_for_byte(pilot, tmp_path):
    run_v2(pilot, tmp_path)
    old = {g.gen_key: g.text for g in open_run(pilot / "shard_00").gens}
    new = {g.gen_key: g.text for g in open_run(tmp_path / "retest/shard_00").gens}
    assert old.items() <= new.items() and len(new) > len(old)


def test_the_done_marker_records_the_variant_and_where_it_came_from(pilot, tmp_path):
    run_v2(pilot, tmp_path)
    done = json.loads((tmp_path / "retest/shard_00" / DONE_FILE).read_text())
    src_done = json.loads((pilot / "shard_00" / DONE_FILE).read_text())
    assert done["variant"] == "v2"
    assert done["source_generations_sha256"] == src_done["generations_sha256"]
    assert done["generations_sha256"] != src_done["generations_sha256"]
    assert len(done["failure_msg_v2_template_sha256"]) == 64


def test_finished_copies_are_skipped_without_loading_the_model(pilot, tmp_path):
    run_v2(pilot, tmp_path)
    built = []
    again = run_v2(pilot, tmp_path, backend=LazyBackend(lambda: built.append(1)))
    assert [r.status for r in again] == ["skipped", "skipped"] and built == []


def test_each_copy_is_uploaded_under_its_own_prefix(pilot, tmp_path):
    rec = Recorder()
    run_v2(pilot, tmp_path, uploader=rec.upload, remote_prefix="pilot/m-v2")
    assert [c[1] for c in rec.calls] == ["pilot/m-v2/shard_00", "pilot/m-v2/shard_01"]
    assert (tmp_path / "retest/shard_00" / UPLOADED_FILE).exists()


def test_a_missing_source_shard_is_restored_from_the_remote(pilot, tmp_path):
    seen = []

    def restore(remote: str, local: Path) -> bool:
        seen.append(remote)
        shutil.copytree(pilot / remote.split("/")[-1], local, dirs_exist_ok=True)
        return True

    gone = tmp_path / "gone"
    shutil.copytree(pilot, gone)
    shutil.rmtree(gone / "shard_01")
    results = run_variant_shards(
        gone,
        tmp_path / "retest",
        cfg(),
        PROBLEMS,
        LazyBackend(factory),
        size=8,
        restorer=restore,
        source_prefix="pilot/m",
        log=lambda s: None,
    )
    assert seen == ["pilot/m/shard_01"] and len(results) == 2


def test_a_source_that_cannot_be_found_is_an_error(pilot, tmp_path):
    shutil.rmtree(pilot / "shard_01")
    with pytest.raises(FileNotFoundError, match="shard_01"):
        run_v2(pilot, tmp_path)


def test_a_different_config_is_refused(pilot, tmp_path):
    other = RunConfig(model_id="mock/model", replicates=2, max_tokens=256)
    with pytest.raises(ValueError, match="config"):
        run_variant_shards(
            pilot, tmp_path / "retest", other, PROBLEMS, LazyBackend(factory),
            size=8, log=lambda s: None,
        )  # fmt: skip


def test_a_session_can_be_bounded(pilot, tmp_path):
    first = run_v2(pilot, tmp_path, max_new_shards=1)
    assert [r.status for r in first] == ["generated"]
    second = run_v2(pilot, tmp_path, max_new_shards=1)
    assert [r.status for r in second] == ["skipped", "generated"]
