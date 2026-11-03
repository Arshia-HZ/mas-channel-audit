"""Round 0: 3 independent agents per problem and replicate (shared by all arms).

Mock backend with fake problems (no GPU):

    python scripts/10_round0.py --out runs/mock --n-fake-problems 20 --replicates 2

Real model on a problems file (GPU), settings taken from the model config:

    python scripts/10_round0.py --out runs/check --problems check/problems.jsonl \\
        --model-config configs/models/qwen3-1.7b.yaml --replicates 1 --log-prompts

Re-running with the same arguments only produces what is missing. Arguments
that differ from the run's saved config are refused.
"""

from __future__ import annotations

import argparse

from channel_audit.backends.mock import fake_problems
from channel_audit.models import load_model_config
from channel_audit.runner import RunConfig, backend_for_run, create_run, run_round0
from channel_audit.schema import Problem
from channel_audit.store import JsonlStore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--problems", help="JSONL of Problem records")
    src.add_argument("--n-fake-problems", type=int, help="mock backend only")
    ap.add_argument("--model-config", help="YAML in configs/models; default: mock")
    ap.add_argument("--model-id", default="mock/model", help="mock backend only")
    ap.add_argument("--max-tokens", type=int, help="mock backend only")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--log-prompts", action="store_true")
    args = ap.parse_args()

    if args.model_config:
        if args.max_tokens or args.n_fake_problems:
            ap.error("--max-tokens and --n-fake-problems are for the mock backend")
        if not args.problems:
            ap.error("a real model needs --problems")
        model = load_model_config(args.model_config)
        cfg = RunConfig(
            model_id=model.model_id,
            backend=model.backend,
            max_tokens=model.max_tokens,
            context_limit=model.max_model_len,
            replicates=args.replicates,
            model=model,
            log_prompts=args.log_prompts,
        )
    else:
        cfg = RunConfig(
            model_id=args.model_id,
            max_tokens=args.max_tokens or 512,
            replicates=args.replicates,
            log_prompts=args.log_prompts,
        )

    if args.problems:
        problems = list(
            JsonlStore(args.problems, Problem, key="problem_id", readonly=True)
        )
    else:
        problems = fake_problems(args.n_fake_problems or 20)

    ctx = create_run(args.out, cfg, problems, backend_for_run(cfg, problems))
    run_round0(ctx)
    print(f"round0: {len(ctx.gens)} generations in {ctx.out_dir}")


if __name__ == "__main__":
    main()
