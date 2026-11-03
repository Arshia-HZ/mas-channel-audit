"""Compare two runs of the same plan, generation by generation.

Run script 10 twice into different directories with the same backend and
version, then:

    python scripts/03_repro_check.py --a runs/a --b runs/b

Exit code 0 only if every generation matches exactly. Otherwise the printed
difference (rate, per arm, first diverging character) is what M3 documents.
"""

from __future__ import annotations

import argparse
import sys

from channel_audit.backend_check import compare_runs
from channel_audit.runner import open_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    a, b = open_run(args.a), open_run(args.b)
    report = compare_runs(a, b)
    versions = {g.backend_version for g in (*a.gens, *b.gens)}
    print(f"backend versions seen: {sorted(versions)}")
    print(report.format())
    return 0 if report.perfect else 1


if __name__ == "__main__":
    sys.exit(main())
