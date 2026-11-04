"""Screen one model: 4 samples per problem on the frozen screen split.

    python scripts/04_screen.py --model-config configs/models/qwen3-1.7b.yaml \
        --problems data/problems --out runs/screen_qwen3-1.7b

Uses the round-0 prompt and the model config's decoding. Resumable: re-running
with the same arguments only generates what is missing. Then compare models
with ``scripts/04_screen_report.py``.
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
    N_SAMPLES,
    format_comparison,
    run_screen,
    screen_report,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--problems", required=True, help="directory with the frozen split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=N_SAMPLES)
    args = ap.parse_args()

    model = load_model_config(args.model_config)
    problems = load_frozen(args.problems, "screen")
    cfg = config_for_model(model, replicates=1, log_prompts=True)
    ctx = create_run(args.out, cfg, problems, backend_for_run(cfg, problems))
    run_screen(ctx, n_samples=args.samples)
    print(f"{grade_pending(ctx)} generations graded")
    print(format_comparison([screen_report(ctx)]))


if __name__ == "__main__":
    main()
