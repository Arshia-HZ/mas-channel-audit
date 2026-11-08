"""Freeze the analysis code: write configs/analysis_freeze.json.

Run this ONCE, after the owner has approved the analysis code and its mock-data
tests, and before 30_analyze.py touches the confirmatory shards. It refuses to
overwrite a manifest. The ``analysis`` group (the M7 code) is checked by the test
suite for good; the ``pipeline`` group (shared modules it imports) is checked by
30_analyze.py at run time.

    python scripts/29_freeze_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from channel_audit.analysis_freeze import FREEZE_FILE, GROUPS, ROOT, write_freeze


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / FREEZE_FILE))
    args = ap.parse_args()
    try:
        manifest = write_freeze(Path(args.out))
    except FileExistsError as exc:
        print(exc)
        return 1
    print(f"wrote {args.out}")
    print(f"combined sha256: {manifest['combined_sha256']}")
    for group in GROUPS:
        print(f"{group}:")
        for rel, digest in manifest[group].items():
            print(f"  {rel:<45}{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
