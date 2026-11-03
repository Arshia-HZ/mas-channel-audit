"""Freeze ``prompts/<version>/`` as a manifest of content hashes.

The manifest lists every template with the same line-ending-normalised sha256
the generations record as ``prompt_hash``, plus one combined hash to cite.
Once written it is never overwritten, and a test verifies the live files
against it, so a silent prompt edit fails CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST = "MANIFEST.json"


class FrozenPromptsError(RuntimeError):
    """The prompt files no longer match their manifest."""


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build_manifest(prompts_dir: str | Path) -> dict:
    d = Path(prompts_dir)
    files = {p.name: _file_hash(p) for p in sorted(d.glob("*.txt"))}
    listing = "\n".join(f"{name}:{digest}" for name, digest in files.items())
    return {
        "version": d.name,
        "files": files,
        "combined_sha256": hashlib.sha256(listing.encode()).hexdigest(),
    }


def write_manifest(prompts_dir: str | Path) -> Path:
    path = Path(prompts_dir) / MANIFEST
    if path.exists():
        raise FileExistsError(f"{path} exists: prompts are frozen, never rewrite it")
    path.write_text(
        json.dumps(build_manifest(prompts_dir), indent=2) + "\n", encoding="utf-8"
    )
    return path


def verify_manifest(prompts_dir: str | Path) -> dict:
    """Return the manifest if the files match it; raise otherwise."""
    path = Path(prompts_dir) / MANIFEST
    frozen = json.loads(path.read_text(encoding="utf-8"))
    live = build_manifest(prompts_dir)["files"]
    problems = []
    for name in sorted(set(frozen["files"]) | set(live)):
        if name not in live:
            problems.append(f"{name}: missing")
        elif name not in frozen["files"]:
            problems.append(f"{name}: not in the manifest")
        elif live[name] != frozen["files"][name]:
            problems.append(f"{name}: changed")
    if problems:
        raise FrozenPromptsError("prompts differ from manifest: " + "; ".join(problems))
    return frozen
