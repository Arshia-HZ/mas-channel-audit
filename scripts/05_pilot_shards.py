"""Run the pilot or the confirmatory problems in resumable shards, uploading each.

    python scripts/05_pilot_shards.py --model-config configs/models/qwen3-1.7b.yaml \
        --problems data/problems --split pilot --out runs/pilot_qwen3-1.7b \
        --upload-repo Arshia-HZ/mas-channel-audit-raw

Every (subject, level) group stays whole inside one shard so the placebo arm finds
its donors. A finished shard is validated and recorded in SHARD_DONE.json; with
``--upload-repo`` (token in the HF_TOKEN environment variable) it is then pushed to
the private dataset. Re-running skips finished shards without loading the model,
resumes an interrupted shard, and uploads any shard that finished but did not
upload.

``--split confirm`` runs the first ``--n-problems`` (default 600) confirm problems in
the frozen order, with ``--with-v2`` adding the stricter failure-message arm. It prints
only run quality (truncation, overflow, extraction failure, leakage) and never an
accuracy, so confirmatory outcomes are not looked at before the analysis code is
frozen.
"""

from __future__ import annotations

import argparse
import os

from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.data import N_CONFIRMATORY, confirmatory_problems, load_frozen
from channel_audit.models import load_model_config
from channel_audit.quality import format_quality_report, run_quality
from channel_audit.runner import backend_for_run, config_for_model, open_run
from channel_audit.shards import (
    SHARD_SIZE,
    LazyBackend,
    hf_downloader,
    hf_uploader,
    run_shards,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--problems", required=True, help="directory with the frozen split")
    ap.add_argument("--split", default="pilot", choices=["pilot", "confirm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--n-problems", type=int, help="confirm only; default 600")
    ap.add_argument("--with-v2", action="store_true", help="add failure_cause_v2")
    ap.add_argument("--size", type=int, default=SHARD_SIZE)
    ap.add_argument("--upload-repo", help="private HF dataset, e.g. user/name")
    ap.add_argument("--remote-prefix", help="default: <split>/<model>")
    ap.add_argument(
        "--max-new-shards", type=int, help="stop after this many new shards"
    )
    args = ap.parse_args()

    model = load_model_config(args.model_config)
    if args.split == "confirm":
        problems = confirmatory_problems(
            args.problems, args.n_problems or N_CONFIRMATORY
        )
    else:
        problems = load_frozen(args.problems, args.split)
    cfg = config_for_model(model, replicates=args.replicates, log_prompts=True)
    slug = model.model_id.split("/")[-1].lower()
    prefix = args.remote_prefix or f"{args.split}/{slug}"

    uploader = restorer = None
    if args.upload_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("set HF_TOKEN (write access) to upload shards")
        uploader = hf_uploader(args.upload_repo, token)
        restorer = hf_downloader(args.upload_repo, token)

    results = run_shards(
        args.out,
        cfg,
        problems,
        LazyBackend(lambda: backend_for_run(cfg, problems)),
        size=args.size,
        arms=(*ARMS, *VARIANT_ARMS) if args.with_v2 else ARMS,
        uploader=uploader,
        restorer=restorer,
        remote_prefix=prefix,
        max_new_shards=args.max_new_shards,
    )
    for r in results:
        print(
            f"{r.name}: {r.status}, {r.n_generations} generations, "
            f"G3 {'pass' if r.g3_ok else 'FAIL'}, uploaded {r.uploaded}"
        )
    from pathlib import Path

    done = sorted(
        d for d in Path(args.out).glob("shard_*") if (d / "SHARD_DONE.json").exists()
    )
    if done:
        print()
        print(format_quality_report(run_quality([open_run(d) for d in done])))


if __name__ == "__main__":
    main()
