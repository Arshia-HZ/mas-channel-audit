"""Sharded runs: 50-problem shards, resumable, validated, uploaded on completion."""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import (
    DONE_FILE,
    SHARD_SIZE,
    UPLOADED_FILE,
    LazyBackend,
    ShardInvalidError,
    hf_downloader,
    hf_uploader,
    make_shards,
    run_shards,
    shard_name,
)

PROBLEMS = fake_problems(10)


def cfg(**kw):
    return RunConfig(model_id="mock/model", replicates=2, max_tokens=512, **kw)


def factory():
    return MockBackend(MockWorld.from_problems(PROBLEMS))


class Recorder:
    def __init__(self):
        self.calls = []

    def upload(self, local: Path, remote: str):
        names = sorted(p.name for p in local.iterdir())
        self.calls.append((local.name, remote, names))


def test_the_shard_size_is_fifty_problems():
    assert SHARD_SIZE == 50


def test_shards_keep_the_input_order_and_the_last_one_may_be_short():
    shards = make_shards(fake_problems(7), size=3, keep_groups=False)
    assert [len(s) for s in shards] == [3, 3, 1]
    ids = [p.problem_id for s in shards for p in s]
    assert ids == [p.problem_id for p in fake_problems(7)]
    assert shard_name(0) == "shard_00" and shard_name(12) == "shard_12"


def test_a_shard_size_below_one_is_an_error():
    with pytest.raises(ValueError):
        make_shards(PROBLEMS, size=0)


def test_subject_level_groups_are_never_split_across_shards():
    problems = fake_problems(40)
    shards = make_shards(problems, size=10)
    assert sorted(p.problem_id for s in shards for p in s) == sorted(
        p.problem_id for p in problems
    )
    homes = {}
    for i, shard in enumerate(shards):
        for p in shard:
            assert homes.setdefault((p.subject, p.level), i) == i
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 2


def test_shard_layout_depends_only_on_subject_and_level():
    problems = fake_problems(40)
    reordered = list(reversed(problems))
    a = [sorted(p.problem_id for p in s) for s in make_shards(problems, size=10)]
    b = [sorted(p.problem_id for p in s) for s in make_shards(reordered, size=10)]
    assert a == b


def test_a_group_with_a_single_problem_cannot_be_sharded():
    problems = fake_problems(8)
    odd = problems[0].model_copy(update={"subject": "Odd", "level": 5})
    with pytest.raises(ValueError, match="placebo"):
        make_shards([odd, *problems[1:]], size=4)


def test_every_shard_runs_validates_and_records_a_done_marker(tmp_path):
    results = run_shards(tmp_path, cfg(), PROBLEMS, LazyBackend(factory), size=4)
    assert [r.name for r in results] == ["shard_00", "shard_01", "shard_02"]
    for r in results:
        assert r.status == "generated" and r.validation_ok and r.g3_ok
        done = json.loads((tmp_path / r.name / DONE_FILE).read_text())
        assert done["n_generations"] > 0 and len(done["generations_sha256"]) == 64
        assert done["arms"] == list(ARMS)
    assert len(open_run(tmp_path / "shard_00").problems) == 3


def test_shards_reproduce_an_unsharded_run(tmp_path):
    run_shards(tmp_path / "s", cfg(), PROBLEMS, LazyBackend(factory), size=4)
    run_shards(tmp_path / "u", cfg(), PROBLEMS, LazyBackend(factory), size=10)
    sharded = {
        g.gen_key: g.text
        for d in sorted((tmp_path / "s").iterdir())
        for g in open_run(d).gens
    }
    whole = {g.gen_key: g.text for g in open_run(tmp_path / "u" / "shard_00").gens}
    assert sharded == whole


def test_finished_shards_are_skipped_and_never_touch_the_backend(tmp_path):
    run_shards(tmp_path, cfg(), PROBLEMS, LazyBackend(factory), size=5)
    built = []

    def spy():
        built.append(1)
        return factory()

    again = run_shards(tmp_path, cfg(), PROBLEMS, LazyBackend(spy), size=5)
    assert [r.status for r in again] == ["skipped", "skipped"]
    assert built == []  # the engine was never loaded


def test_an_interrupted_shard_resumes_and_generates_only_what_is_missing(tmp_path):
    class Dies(MockBackend):
        calls = 0

        def generate(self, requests):
            Dies.calls += 1
            if Dies.calls == 3:
                raise RuntimeError("session ended")
            return super().generate(requests)

    lazy = LazyBackend(lambda: Dies(MockWorld.from_problems(PROBLEMS)))
    with pytest.raises(RuntimeError, match="session ended"):
        run_shards(tmp_path, cfg(), PROBLEMS, lazy, size=5)
    assert not (tmp_path / "shard_00" / DONE_FILE).exists()
    stored = len(open_run(tmp_path / "shard_00").gens)
    assert stored > 0
    results = run_shards(tmp_path, cfg(), PROBLEMS, LazyBackend(factory), size=5)
    assert all(r.validation_ok for r in results)
    assert len(open_run(tmp_path / "shard_00").gens) > stored


def test_each_shard_is_uploaded_when_it_completes(tmp_path):
    rec = Recorder()
    run_shards(
        tmp_path,
        cfg(),
        PROBLEMS,
        LazyBackend(factory),
        size=5,
        uploader=rec.upload,
        remote_prefix="pilot/mock",
    )
    assert [c[1] for c in rec.calls] == ["pilot/mock/shard_00", "pilot/mock/shard_01"]
    assert "generations.jsonl" in rec.calls[0][2] and DONE_FILE in rec.calls[0][2]
    marker = json.loads((tmp_path / "shard_00" / UPLOADED_FILE).read_text())
    assert marker["remote_path"] == "pilot/mock/shard_00"


def test_a_failed_upload_is_retried_on_the_next_run_without_regenerating(tmp_path):
    def broken(local, remote):
        raise OSError("network down")

    with pytest.raises(OSError):
        run_shards(
            tmp_path, cfg(), PROBLEMS, LazyBackend(factory), size=10, uploader=broken
        )
    assert (tmp_path / "shard_00" / DONE_FILE).exists()
    assert not (tmp_path / "shard_00" / UPLOADED_FILE).exists()
    rec = Recorder()
    built = []
    again = run_shards(
        tmp_path,
        cfg(),
        PROBLEMS,
        LazyBackend(lambda: built.append(1) or factory()),
        size=10,
        uploader=rec.upload,
    )
    assert built == [] and len(rec.calls) == 1
    assert [r.status for r in again] == ["uploaded_only"] and again[0].uploaded


def test_a_shard_that_fails_validation_is_not_marked_done_or_uploaded(
    tmp_path, monkeypatch
):
    import channel_audit.shards as shards

    class Bad:
        errors = ["1 generations missing"]
        warnings = []
        g3_ok = True
        g3_failing_arms = []
        counts = {}

    monkeypatch.setattr(shards, "validate_run", lambda ctx: Bad())
    rec = Recorder()
    with pytest.raises(ShardInvalidError, match="missing"):
        run_shards(
            tmp_path, cfg(), PROBLEMS, LazyBackend(factory), size=5, uploader=rec.upload
        )
    assert not (tmp_path / "shard_00" / DONE_FILE).exists() and rec.calls == []


def test_a_g3_failure_is_recorded_but_does_not_stop_the_run(tmp_path):
    tiny = RunConfig(model_id="mock/model", replicates=1, max_tokens=8)
    results = run_shards(tmp_path, tiny, PROBLEMS, LazyBackend(factory), size=5)
    assert all(r.validation_ok and not r.g3_ok for r in results)
    done = json.loads((tmp_path / "shard_00" / DONE_FILE).read_text())
    assert done["g3_ok"] is False and done["g3_failing_arms"]


def test_a_lost_local_shard_is_restored_from_the_remote(tmp_path):
    src = tmp_path / "src"
    run_shards(src, cfg(), PROBLEMS, LazyBackend(factory), size=5)
    fetched = []

    def restore(remote: str, local: Path) -> bool:
        fetched.append(remote)
        shutil.copytree(src / remote.split("/")[-1], local, dirs_exist_ok=True)
        return True

    built = []
    res = run_shards(
        tmp_path / "fresh",
        cfg(),
        PROBLEMS,
        LazyBackend(lambda: built.append(1) or factory()),
        size=5,
        restorer=restore,
        remote_prefix="pilot/mock",
    )
    assert fetched == ["pilot/mock/shard_00", "pilot/mock/shard_01"]
    assert [r.status for r in res] == ["skipped", "skipped"] and built == []


def test_a_restorer_that_finds_nothing_leaves_the_shard_to_be_generated(tmp_path):
    res = run_shards(
        tmp_path,
        cfg(),
        PROBLEMS,
        LazyBackend(factory),
        size=10,
        restorer=lambda remote, local: False,
    )
    assert [r.status for r in res] == ["generated"]


def test_a_session_can_be_bounded_to_a_number_of_new_shards(tmp_path):
    first = run_shards(
        tmp_path, cfg(), PROBLEMS, LazyBackend(factory), size=5, max_new_shards=1
    )
    assert [r.status for r in first] == ["generated"]
    assert not (tmp_path / "shard_01").exists()
    second = run_shards(
        tmp_path, cfg(), PROBLEMS, LazyBackend(factory), size=5, max_new_shards=1
    )
    assert [r.status for r in second] == ["skipped", "generated"]  # skips are free


def test_the_lazy_backend_loads_once_and_only_when_used():
    built = []
    lazy = LazyBackend(lambda: built.append(1) or factory())
    assert built == []
    assert lazy.name == "mock" and lazy.version == "1"
    lazy.generate([])
    assert built == [1]


# --- Hugging Face adapters (fake library, no network) -------------------------


@pytest.fixture
def fake_hub(monkeypatch):
    calls = {}

    class Api:
        def __init__(self, token=None):
            calls["token"] = token

        def upload_folder(self, **kw):
            calls["upload"] = kw

    def snapshot_download(**kw):
        calls["download"] = kw
        shard = Path(kw["local_dir"]) / "pilot" / "x" / "shard_00"
        shard.mkdir(parents=True, exist_ok=True)
        (shard / DONE_FILE).write_text("{}")
        return kw["local_dir"]

    mod = types.SimpleNamespace(HfApi=Api, snapshot_download=snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return calls


def test_the_hf_uploader_pushes_the_shard_folder_to_a_private_dataset(
    fake_hub, tmp_path
):
    up = hf_uploader("me/raw", token="tok")
    up(tmp_path, "pilot/x/shard_00")
    kw = fake_hub["upload"]
    assert kw["repo_id"] == "me/raw" and kw["repo_type"] == "dataset"
    assert kw["path_in_repo"] == "pilot/x/shard_00"
    assert Path(kw["folder_path"]) == tmp_path and fake_hub["token"] == "tok"


def test_the_hf_downloader_fetches_only_the_requested_shard(fake_hub, tmp_path):
    down = hf_downloader("me/raw", token="tok")
    target = tmp_path / "shard_00"
    assert down("pilot/x/shard_00", target) is True
    assert fake_hub["download"]["allow_patterns"] == "pilot/x/shard_00/*"
    assert (target / DONE_FILE).exists()


def test_the_hf_downloader_reports_when_the_remote_has_nothing(
    fake_hub, tmp_path, monkeypatch
):
    mod = sys.modules["huggingface_hub"]
    monkeypatch.setattr(mod, "snapshot_download", lambda **kw: kw["local_dir"])
    assert (
        hf_downloader("me/raw", token="t")("pilot/x/shard_09", tmp_path / "s") is False
    )
