"""Freeze prompts/v1: write MANIFEST.json with the hash of every template.

Run this ONCE, only after the T4 review, with the owner's approval. Afterwards
the prompt files must never change (AGENTS.md rule 2) and the test suite
verifies them against the manifest. It refuses to overwrite a manifest.

    python scripts/03_freeze_prompts.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from channel_audit.channels import PROMPT_VERSION, PROMPTS_DIR
from channel_audit.prompt_freeze import verify_manifest, write_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(PROMPTS_DIR / PROMPT_VERSION))
    args = ap.parse_args()
    try:
        path = write_manifest(args.dir)
    except FileExistsError as exc:
        print(exc)
        return 1
    manifest = verify_manifest(Path(args.dir))
    print(f"wrote {path}")
    print(f"combined sha256: {manifest['combined_sha256']}")
    for name, digest in manifest["files"].items():
        print(f"  {name:<22}{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
