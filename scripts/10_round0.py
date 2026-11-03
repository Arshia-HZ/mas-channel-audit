"""Round 0: 3 independent agents per problem and replicate (shared by all arms).

M2 runs on the mock backend with fake problems:

    python scripts/10_round0.py --out runs/mock --n-fake-problems 20 --replicates 2

Re-running with the same arguments only produces what is missing. Arguments
that differ from the run's saved config are refused.
"""

from __future__ import annotations

import argparse

from channel_audit.backends.mock import fake_problems
from channel_audit.runner import RunConfig, backend_for_run, create_run, run_round0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--model-id", default="mock/model")
    ap.add_argument("--n-fake-problems", type=int, default=20)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    cfg = RunConfig(
        model_id=args.model_id,
        backend=args.backend,
        replicates=args.replicates,
        max_tokens=args.max_tokens,
    )
    problems = fake_problems(args.n_fake_problems)
    ctx = create_run(args.out, cfg, problems, backend_for_run(cfg, problems))
    run_round0(ctx)
    print(f"round0: {len(ctx.gens)} generations in {ctx.out_dir}")


if __name__ == "__main__":
    main()
