"""Re-test the stricter failure-message prompt (v2) on copies of the pilot shards.

    python scripts/05_retest_failure.py --model-config configs/models/qwen3-1.7b.yaml \
        --problems data/problems --source runs/pilot --out runs/pilot_v2 \
        --upload-repo Arshia-HZ/mas-channel-audit-raw

Needs ``prompts/v2`` to be frozen (its MANIFEST.json). Each finished pilot shard is
copied byte for byte, the v2 failure messages and their revisions are generated in
the copy, and the copy is validated and uploaded. The original shards are never
changed. Then draw a fresh sample with ``05_failure_sample.py --variant v2 --seed 1``
and check it with ``05_gate_check.py --variant v2 --seed 1 --labels ...``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from channel_audit.channels import PROMPTS_DIR
from channel_audit.data import load_frozen
from channel_audit.models import load_model_config
from channel_audit.prompt_freeze import FrozenPromptsError, require_frozen
from channel_audit.runner import backend_for_run, config_for_model
from channel_audit.shards import (
    SHARD_SIZE,
    LazyBackend,
    hf_downloader,
    hf_uploader,
    run_variant_shards,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--problems", required=True, help="directory with the frozen split")
    ap.add_argument("--source", required=True, help="finished pilot shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--size", type=int, default=SHARD_SIZE)
    ap.add_argument("--upload-repo", help="private HF dataset, e.g. user/name")
    args = ap.parse_args()

    try:
        require_frozen(Path(PROMPTS_DIR) / "v2")
    except FrozenPromptsError as exc:
        raise SystemExit(str(exc)) from None

    model = load_model_config(args.model_config)
    problems = load_frozen(args.problems, "pilot")
    cfg = config_for_model(model, replicates=args.replicates, log_prompts=True)
    slug = model.model_id.split("/")[-1].lower()

    uploader = restorer = None
    if args.upload_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("set HF_TOKEN (write access) to upload shards")
        uploader = hf_uploader(args.upload_repo, token)
        restorer = hf_downloader(args.upload_repo, token)

    results = run_variant_shards(
        args.source,
        args.out,
        cfg,
        problems,
        LazyBackend(lambda: backend_for_run(cfg, problems)),
        size=args.size,
        uploader=uploader,
        restorer=restorer,
        source_prefix=f"pilot/{slug}",
        remote_prefix=f"pilot/{slug}-v2",
    )
    for r in results:
        print(
            f"{r.name}: {r.status}, {r.n_generations} generations, "
            f"G3 {'pass' if r.g3_ok else 'FAIL'}, uploaded {r.uploaded}"
        )


if __name__ == "__main__":
    main()
