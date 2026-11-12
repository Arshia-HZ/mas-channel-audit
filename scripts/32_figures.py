"""Draw the exploratory figures from the two JSON reports (needs matplotlib).

    python scripts/32_figures.py \
        --analysis results/confirm_qwen3-1.7b_analysis/analysis.json \
        --decomposition results/confirm_qwen3-1.7b_exploratory/exploratory.json \
        --out-dir results/confirm_qwen3-1.7b_figures

Figure 2 (accuracy against tokens, with the self-consistency curve), figure 3 (change
types) and figure 4 (source of change). Each carries the EXPLORATORY footer. Existing
figures are never overwritten. Reads reports only; never loads a shard.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from channel_audit.analysis.figures import make_all


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--analysis", required=True, help="analysis.json of the primary run"
    )
    ap.add_argument("--decomposition", required=True, help="exploratory.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    decomposition = json.loads(Path(args.decomposition).read_text(encoding="utf-8"))
    try:
        paths = make_all(analysis, decomposition, args.out_dir)
    except (FileExistsError, ImportError) as exc:
        print(f"error: {exc}")
        return 1
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
