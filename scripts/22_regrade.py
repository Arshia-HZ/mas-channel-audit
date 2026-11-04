"""Re-grade a finished run into a new directory (the source is left untouched).

    python scripts/22_regrade.py --src runs/screen_a --out runs/screen_a_regraded

Use after the extractor changed. Generations, prompts and problems are copied
byte for byte; only graded.jsonl is computed again.
"""

from __future__ import annotations

import argparse

from channel_audit.regrade import regrade_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = regrade_run(args.src, args.out)
    print(f"regraded {n} generations into {args.out}")


if __name__ == "__main__":
    main()
