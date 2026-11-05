"""Screen one model: 4 samples per problem on the frozen screen split.

    python scripts/04_screen.py --model-config configs/models/qwen3-1.7b.yaml \
        --problems data/problems --out runs/screen_qwen3-1.7b

Uses the model config's decoding. ``--prompt round0`` is the study prompt;
``--prompt plain`` is the published-style prompt for the G0 sanity check.
``--temperature 0`` runs greedy decoding (one sample per problem) for a like-for-like
comparison with published scores; use a new ``--out`` for it.
Resumable: re-running with the same arguments only generates what is missing.
Then compare models with ``scripts/04_screen_report.py``.
"""

from __future__ import annotations

import argparse

from channel_audit.data import load_frozen
from channel_audit.models import load_model_config
from channel_audit.runner import (
    backend_for_run,
    config_for_model,
    create_run,
    grade_pending,
)
from channel_audit.screening import (
    PROMPT_ARMS,
    format_comparison,
    resolve_samples,
    run_screen,
    screen_report,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--problems", required=True, help="directory with the frozen split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, help="default: 4 sampled, 1 greedy")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--prompt", nargs="+", choices=list(PROMPT_ARMS), default=["round0"]
    )
    args = ap.parse_args()

    model = load_model_config(args.model_config)
    problems = load_frozen(args.problems, "screen")
    n_samples = resolve_samples(args.temperature, args.samples)
    cfg = config_for_model(
        model, replicates=1, log_prompts=True, temperature=args.temperature
    )
    ctx = create_run(args.out, cfg, problems, backend_for_run(cfg, problems))
    for prompt in args.prompt:
        run_screen(ctx, n_samples=n_samples, prompt=prompt)
    print(f"{grade_pending(ctx)} generations graded")
    print(format_comparison([screen_report(ctx, p) for p in args.prompt]))


if __name__ == "__main__":
    main()
