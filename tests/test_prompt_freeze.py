"""Freezing prompts/v1: a manifest of hashes that tests can enforce."""

from __future__ import annotations

import json
import shutil

import pytest

from channel_audit import channels
from channel_audit.prompt_freeze import (
    FrozenPromptsError,
    build_manifest,
    verify_manifest,
    write_manifest,
)


@pytest.fixture
def prompts_copy(tmp_path):
    dst = tmp_path / "v1"
    shutil.copytree(channels.PROMPTS_DIR / channels.PROMPT_VERSION, dst)
    (dst / "MANIFEST.json").unlink(missing_ok=True)
    return dst


def test_manifest_lists_every_template_with_its_template_hash(prompts_copy):
    m = build_manifest(prompts_copy)
    assert set(m["files"]) == {p.name for p in prompts_copy.glob("*.txt")}
    assert m["files"]["full_solution.txt"] == channels.template_hash("full_solution")
    assert m["version"] == "v1"


def test_combined_hash_is_stable_and_changes_with_any_file(prompts_copy):
    a = build_manifest(prompts_copy)["combined_sha256"]
    assert a == build_manifest(prompts_copy)["combined_sha256"]
    f = prompts_copy / "self_reflect.txt"
    f.write_text(f.read_text(encoding="utf-8") + "x\n", encoding="utf-8")
    assert build_manifest(prompts_copy)["combined_sha256"] != a


def test_line_endings_do_not_change_the_hash(prompts_copy):
    a = build_manifest(prompts_copy)["combined_sha256"]
    f = prompts_copy / "round0.txt"
    f.write_bytes(f.read_bytes().replace(b"\n", b"\r\n"))
    assert build_manifest(prompts_copy)["combined_sha256"] == a


def test_write_then_verify_passes(prompts_copy):
    write_manifest(prompts_copy)
    assert verify_manifest(prompts_copy)["combined_sha256"]


def test_verify_detects_an_edited_file(prompts_copy):
    write_manifest(prompts_copy)
    f = prompts_copy / "answer_only.txt"
    f.write_text(f.read_text(encoding="utf-8").replace("Reconsider", "Rethink"))
    with pytest.raises(FrozenPromptsError, match="answer_only.txt"):
        verify_manifest(prompts_copy)


def test_verify_detects_added_and_removed_files(prompts_copy):
    write_manifest(prompts_copy)
    (prompts_copy / "placebo.txt").write_text("new", encoding="utf-8")
    with pytest.raises(FrozenPromptsError, match="placebo.txt"):
        verify_manifest(prompts_copy)
    (prompts_copy / "placebo.txt").unlink()
    (prompts_copy / "round0.txt").unlink()
    with pytest.raises(FrozenPromptsError, match="round0.txt"):
        verify_manifest(prompts_copy)


def test_verify_without_a_manifest_is_an_error(prompts_copy):
    with pytest.raises(FileNotFoundError):
        verify_manifest(prompts_copy)


def test_writing_over_an_existing_manifest_is_refused(prompts_copy):
    write_manifest(prompts_copy)
    with pytest.raises(FileExistsError):
        write_manifest(prompts_copy)


def test_manifest_file_is_plain_json(prompts_copy):
    path = write_manifest(prompts_copy)
    assert json.loads(path.read_text(encoding="utf-8"))["files"]


def test_the_repo_prompts_match_their_manifest_once_frozen():
    live = channels.PROMPTS_DIR / channels.PROMPT_VERSION
    if not (live / "MANIFEST.json").exists():
        pytest.skip("prompts/v1 is not frozen yet (done after the T4 review)")
    verify_manifest(live)
