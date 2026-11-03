"""Draw the M3 check problems from the MATH train split (levels 3 to 5).

    python scripts/03_make_check_problems.py --n 50 --out check/problems.jsonl

Needs ``pip install 'mas-channel-audit[data]'`` and internet. For offline use
point --from-jsonl at a file of raw MATH rows. The output is never overwritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from channel_audit.check_problems import (
    load_math_train,
    rows_from_jsonl,
    sample_check_problems,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-jsonl", help="raw MATH rows instead of downloading")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    pool = rows_from_jsonl(args.from_jsonl) if args.from_jsonl else load_math_train()
    chosen = sample_check_problems(pool, n=args.n, seed=args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(p.model_dump_json() + "\n" for p in chosen), encoding="utf-8"
    )
    groups = Counter((p.subject, p.level) for p in chosen)
    print(f"wrote {len(chosen)} problems from {len(pool)} usable rows to {out}")
    print(f"groups: {len(groups)}, smallest {min(groups.values())}")


if __name__ == "__main__":
    main()
