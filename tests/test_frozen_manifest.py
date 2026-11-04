"""The committed split manifest is the record of what the study problems are."""

from __future__ import annotations

import json
import re
from pathlib import Path

from channel_audit import data

MANIFEST = (
    Path(__file__).resolve().parents[1] / "data" / "problems" / "split_manifest.json"
)


def test_the_manifest_is_committed_and_matches_the_plan():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert m["dataset"] == data.MATH_DATASET
    assert m["revision"] == data.MATH_REVISION
    assert m["salt"] == data.SPLIT_SALT
    assert m["counts"]["screen"] == data.N_SCREEN
    assert m["counts"]["pilot"] == data.N_PILOT
    assert m["counts"]["confirm"] > 0
    assert (
        m["counts"]["screen"]
        + m["counts"]["pilot"]
        + m["counts"]["confirm"]
        + (m["counts"]["excluded"])
        + m["counts"]["dropped"]
        == m["n_rows"]
    )
    for name in data.SPLITS:
        assert re.fullmatch(r"[0-9a-f]{64}", m["files"][name]["sha256"])
