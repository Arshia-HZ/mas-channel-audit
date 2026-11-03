"""Append-only JSONL store: durable, resumable, duplicate-safe."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from channel_audit.schema import Generation, Graded
from channel_audit.store import CorruptStoreError, DuplicateKeyError, JsonlStore
from conftest import make_generation


def gens(n):
    return [make_generation(sample_idx=i, text=f"t{i}") for i in range(n)]


def test_append_and_reload(tmp_path):
    path = tmp_path / "gen.jsonl"
    s = JsonlStore(path, Generation)
    for g in gens(3):
        s.append(g)
    assert len(s) == 3
    again = JsonlStore(path, Generation)
    assert [g.gen_key for g in again] == [g.gen_key for g in gens(3)]
    assert gens(3)[1].gen_key in again
    assert again.get(gens(3)[1].gen_key).text == "t1"


def test_each_record_is_one_line(tmp_path):
    path = tmp_path / "gen.jsonl"
    s = JsonlStore(path, Generation)
    s.append(make_generation(text="multi\nline\ntext"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["text"] == "multi\nline\ntext"


def test_duplicate_append_is_an_error_and_writes_nothing(tmp_path):
    path = tmp_path / "gen.jsonl"
    s = JsonlStore(path, Generation)
    g = make_generation()
    s.append(g)
    before = path.read_bytes()
    with pytest.raises(DuplicateKeyError):
        s.append(make_generation(text="same key, different text"))
    assert path.read_bytes() == before


def test_duplicate_keys_already_in_file_are_detected_on_load(tmp_path):
    path = tmp_path / "gen.jsonl"
    line = make_generation().model_dump_json() + "\n"
    path.write_text(line + line, encoding="utf-8")
    with pytest.raises(DuplicateKeyError):
        JsonlStore(path, Generation)


def test_corrupt_complete_line_is_an_error_not_skipped(tmp_path):
    path = tmp_path / "gen.jsonl"
    good = make_generation().model_dump_json()
    path.write_text(good + "\n{not json}\n" + good + "\n", encoding="utf-8")
    with pytest.raises(CorruptStoreError):
        JsonlStore(path, Generation)


def test_schema_invalid_line_is_an_error(tmp_path):
    path = tmp_path / "gen.jsonl"
    path.write_text('{"gen_key": "x"}\n', encoding="utf-8")
    with pytest.raises(CorruptStoreError):
        JsonlStore(path, Generation)


def test_works_for_graded_records_too(tmp_path):
    s = JsonlStore(tmp_path / "graded.jsonl", Graded)
    s.append(
        Graded(gen_key="k", extracted="1", extraction_status="boxed", correct=True)
    )
    with pytest.raises(DuplicateKeyError):
        s.append(
            Graded(gen_key="k", extracted="2", extraction_status="boxed", correct=False)
        )


def test_missing_parent_directory_is_created(tmp_path):
    s = JsonlStore(tmp_path / "a" / "b" / "gen.jsonl", Generation)
    s.append(make_generation())
    assert len(s) == 1


KILLED_WRITER = textwrap.dedent(
    """
    import os, sys
    from channel_audit.schema import Generation
    from channel_audit.store import JsonlStore

    path, records_file, n_complete = sys.argv[1], sys.argv[2], int(sys.argv[3])
    recs = [
        Generation.model_validate_json(line)
        for line in open(records_file, encoding="utf-8").read().splitlines()
    ]
    store = JsonlStore(path, Generation)
    for r in recs[:n_complete]:
        store.append(r)
    # Die mid-write of the next record: half a line, no newline, no cleanup.
    with open(path, "ab") as fh:
        fh.write(recs[n_complete].model_dump_json().encode()[:40])
        fh.flush()
        os.fsync(fh.fileno())
    os._exit(9)
    """
)


def _kill_writer(tmp_path, records, n_complete):
    path = tmp_path / "gen.jsonl"
    records_file = tmp_path / "records.txt"
    records_file.write_text(
        "\n".join(r.model_dump_json() for r in records), encoding="utf-8"
    )
    script = tmp_path / "killed_writer.py"
    script.write_text(KILLED_WRITER, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), str(path), str(records_file), str(n_complete)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 9, proc.stderr
    return path


def test_killed_writer_leaves_a_readable_file(tmp_path):
    records = gens(5)
    path = _kill_writer(tmp_path, records, n_complete=3)
    assert not path.read_bytes().endswith(b"\n")  # the torn tail is really there
    before = path.read_bytes()
    ro = JsonlStore(path, Generation, readonly=True)
    assert [g.gen_key for g in ro] == [g.gen_key for g in records[:3]]
    assert path.read_bytes() == before  # read-only never touches the file
    with pytest.raises(PermissionError):
        ro.append(records[3])


def test_resume_after_kill_has_no_duplicates_and_no_loss(tmp_path):
    records = gens(5)
    path = _kill_writer(tmp_path, records, n_complete=3)

    store = JsonlStore(path, Generation)  # recovery happens here
    assert store.recovered_bytes == 40
    assert len(store) == 3
    for r in records:  # the resume loop every runner uses
        if r.gen_key not in store:
            store.append(r)
    assert len(store) == 5

    final = JsonlStore(path, Generation)
    keys = [g.gen_key for g in final]
    assert keys == [r.gen_key for r in records]
    assert len(set(keys)) == len(keys)
    assert path.read_bytes().endswith(b"\n")


def test_torn_fragment_is_preserved_not_silently_dropped(tmp_path):
    path = _kill_writer(tmp_path, gens(3), n_complete=2)
    JsonlStore(path, Generation)
    torn = path.with_name(path.name + ".torn")
    assert torn.exists() and len(torn.read_bytes()) == 40


def test_complete_records_are_byte_identical_after_recovery(tmp_path):
    path = _kill_writer(tmp_path, gens(4), n_complete=3)
    complete = path.read_bytes()[: path.read_bytes().rfind(b"\n") + 1]
    JsonlStore(path, Generation)
    assert path.read_bytes() == complete


def test_no_recovery_needed_on_clean_file(tmp_path):
    path = tmp_path / "gen.jsonl"
    s = JsonlStore(path, Generation)
    s.append(make_generation())
    assert JsonlStore(path, Generation).recovered_bytes == 0
    assert not path.with_name(path.name + ".torn").exists()


def test_store_exposes_no_delete_or_overwrite_api():
    public = {n for n in dir(JsonlStore) if not n.startswith("_")}
    assert not public & {"delete", "remove", "pop", "clear", "update", "overwrite"}
