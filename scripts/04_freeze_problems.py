"""Freeze the MATH test split: 300 screen, 200 pilot, the rest confirm.

    python scripts/04_freeze_problems.py --out data/problems

Downloads the MATH test set at the pinned dataset revision (needs
``pip install 'mas-channel-audit[data]'`` and internet), orders the usable
problems by a salted sha256 of the problem id and writes screen.jsonl,
pilot.jsonl, confirm.jsonl and split_manifest.json. Commit only the manifest:
it holds the revision, the salt and the sha256 of every file. Re-running gives
identical files; a different result for an existing manifest is refused.
"""

from __future__ import annotations

import argparse
from collections import Counter

from channel_audit.check_problems import CONFIGS
from channel_audit.data import (
    SPLITS,
    assign_splits,
    load_math_test,
    write_split,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    problems, dropped = load_math_test()
    split = assign_splits(problems)
    n_rows = len(problems) + len(dropped)
    manifest = write_split(split, args.out, n_rows=n_rows, dropped=dropped)

    print(
        f"{n_rows} rows in {len(CONFIGS)} subjects; {len(problems)} usable, "
        f"{len(dropped)} dropped (no parsable level or boxed gold)"
    )
    for name in SPLITS:
        levels = Counter(p.level for p in getattr(split, name))
        by_level = " ".join(f"L{lv}={levels[lv]}" for lv in sorted(levels))
        entry = manifest["files"][name]
        print(f"{name:<8}{entry['n']:>5}  {by_level}  sha256 {entry['sha256'][:12]}")
    print(f"excluded (level 1-2 outside the screen split): {len(split.excluded_ids)}")
    print(f"wrote {args.out}/ (revision {manifest['revision'][:12]})")


if __name__ == "__main__":
    main()
