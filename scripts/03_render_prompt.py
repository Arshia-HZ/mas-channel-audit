"""Show a round-0 prompt exactly as the model will receive it. Needs no GPU.

    python scripts/03_render_prompt.py --model-config configs/models/qwen3-1.7b.yaml

Downloads only the tokenizer. Check that the chat template is the model's own
and that Qwen3 shows an empty <think></think> block (non-thinking mode).
"""

from __future__ import annotations

import argparse

from channel_audit.backends.chat import render_prompt_text
from channel_audit.channels import build_round0_prompt
from channel_audit.models import load_model_config

SAMPLE = (
    r"What is the value of $\frac{1}{2}+\frac{1}{3}$? "
    "Express your answer as a fraction."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-config", required=True)
    args = ap.parse_args()
    cfg = load_model_config(args.model_config)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("needs transformers: pip install transformers") from exc
    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)
    text = render_prompt_text(
        tok, build_round0_prompt(SAMPLE), cfg.chat_template_kwargs
    )
    ids = tok.encode(text, add_special_tokens=False)
    print(
        f"model {cfg.model_id} @ {cfg.revision[:12]}  kwargs {cfg.chat_template_kwargs}"
    )
    print(f"prompt tokens: {len(ids)}; stop ids configured: {cfg.stop_token_ids}")
    print("-" * 60)
    print(text)
    print("-" * 60)


if __name__ == "__main__":
    main()
