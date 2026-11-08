"""Freeze the analysis code as a manifest of content hashes (M7).

The confirmatory analysis runs once, with the code that was tested on mock data.
The manifest has two groups, both hashed with the line-ending-normalised sha256
used for the prompts:

* ``analysis``: the M7 code itself. It is frozen for good, and a test checks it.
* ``pipeline``: the shared modules it imports. They are pinned for the confirmatory
  run (``scripts/30_analyze.py`` refuses to run if one changed since the freeze, and
  records the hashes), but later work on the generation side may change them without
  breaking the permanent freeze.

``GENERATION_ONLY`` lists the imported modules that cannot change a number: they
are used to produce generations, never to analyse them. A test checks that every
module the script imports is in one of the three lists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FREEZE_FILE = "configs/analysis_freeze.json"

ANALYSIS_FILES = (
    "src/channel_audit/analysis/contrasts.py",
    "src/channel_audit/analysis/integrity.py",
    "src/channel_audit/analysis/stats.py",
    "src/channel_audit/analysis_freeze.py",
    "scripts/30_analyze.py",
)
PIPELINE_FILES = (
    "src/channel_audit/aggregate.py",
    "src/channel_audit/budget.py",
    "src/channel_audit/channels.py",
    "src/channel_audit/extract.py",
    "src/channel_audit/gates.py",
    "src/channel_audit/grade.py",
    "src/channel_audit/leakage.py",
    "src/channel_audit/models.py",
    "src/channel_audit/mv_guard.py",
    "src/channel_audit/quality.py",
    "src/channel_audit/runner.py",
    "src/channel_audit/schema.py",
    "src/channel_audit/seeds.py",
    "src/channel_audit/shards.py",
    "src/channel_audit/store.py",
    "src/channel_audit/validate.py",
)
GENERATION_ONLY = (
    "src/channel_audit/backends/base.py",
    "src/channel_audit/backends/mock.py",
    "src/channel_audit/placebo.py",
)
GROUPS = ("analysis", "pipeline")


class FrozenAnalysisError(RuntimeError):
    """The analysis files no longer match their manifest, or there is none."""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_manifest(
    root: str | Path,
    analysis: Sequence[str] = ANALYSIS_FILES,
    pipeline: Sequence[str] = PIPELINE_FILES,
) -> dict:
    root = Path(root)
    groups = {
        "analysis": {rel: _file_hash(root / rel) for rel in sorted(analysis)},
        "pipeline": {rel: _file_hash(root / rel) for rel in sorted(pipeline)},
    }
    listing = "\n".join(
        f"{group}:{rel}:{digest}"
        for group in GROUPS
        for rel, digest in groups[group].items()
    )
    return {**groups, "combined_sha256": hashlib.sha256(listing.encode()).hexdigest()}


def write_freeze(
    path: str | Path,
    root: str | Path = ROOT,
    analysis: Sequence[str] = ANALYSIS_FILES,
    pipeline: Sequence[str] = PIPELINE_FILES,
) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"{path} exists: the analysis is frozen, never rewrite it"
        )
    manifest = build_manifest(root, analysis, pipeline)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_freeze(
    path: str | Path,
    root: str | Path = ROOT,
    groups: Sequence[str] = GROUPS,
    expected: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Return the manifest if the live files of ``groups`` match it; raise otherwise.

    With ``expected`` (group -> files) the manifest must also list every one of
    those files.
    """
    root = Path(root)
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for group in groups:
        for rel, digest in sorted(frozen.get(group, {}).items()):
            live = root / rel
            if not live.is_file():
                problems.append(f"{rel}: missing")
            elif _file_hash(live) != digest:
                problems.append(f"{rel}: changed")
    for group, files in (expected or {}).items():
        problems += [
            f"{rel}: not in the manifest"
            for rel in sorted(set(files) - set(frozen.get(group, {})))
        ]
    if problems:
        raise FrozenAnalysisError(
            "analysis code differs from its freeze: " + "; ".join(problems)
        )
    return frozen


def require_frozen(
    path: str | Path,
    root: str | Path = ROOT,
    groups: Sequence[str] = GROUPS,
    expected: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """The verified manifest, or an error saying the analysis is not frozen."""
    if not Path(path).exists():
        raise FrozenAnalysisError(
            f"{path} not found: the analysis is not frozen. After the owner approves "
            "it, run scripts/29_freeze_analysis.py"
        )
    return verify_freeze(path, root, groups, expected)
