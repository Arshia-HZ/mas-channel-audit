"""Write the sample of outputs to read by hand (30 across every arm).

    python scripts/03_review_sample.py --out runs/check --n 30

Start script 10 with --log-prompts so each output sits beside its prompt.
Read for: answer in \\boxed{}, a Confidence line, no copying of peers' text,
failure messages that name a step without an answer, and sane stopping.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from channel_audit.backend_check import format_review, select_review_sample
from channel_audit.runner import open_run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--file", help="default: <run dir>/review_sample.md")
    args = ap.parse_args()
    ctx = open_run(args.out)
    path = Path(args.file) if args.file else ctx.out_dir / "review_sample.md"
    if path.exists():
        raise SystemExit(f"{path} exists; refusing to overwrite")
    sample = select_review_sample(ctx, n=args.n, seed=args.seed)
    path.write_text(format_review(ctx, sample), encoding="utf-8")
    print(f"wrote {len(sample)} outputs to {path}")


if __name__ == "__main__":
    main()
