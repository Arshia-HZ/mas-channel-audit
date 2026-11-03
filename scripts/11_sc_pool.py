"""Self-consistency pool: 6 extra independent samples per problem and replicate.

python scripts/11_sc_pool.py --out runs/mock
"""

from __future__ import annotations

import argparse

from channel_audit.runner import open_run, run_sc_pool


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ctx = open_run(args.out)
    run_sc_pool(ctx)
    print(f"sc_pool: {len(ctx.gens)} generations in {ctx.out_dir}")


if __name__ == "__main__":
    main()
