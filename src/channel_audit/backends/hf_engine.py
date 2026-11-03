"""Hugging Face backend, for the P100 (no vLLM) and for throughput checks.

Do not mix this backend with vLLM within one model's confirmatory run.

Prompts are rendered exactly as in the vLLM backend, left-padded into batches,
and generation stops by token id: a completion ends at the first end-of-turn
id, which counts toward ``completion_tokens`` (as in vLLM). Seeding differs
from vLLM: with ``batch_size == 1`` each request gets its own seed; with larger
batches the whole batch is seeded from its members' seeds, so a request's
sample then depends on which requests share its batch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

from channel_audit.backends.base import GenRequest, GenResult
from channel_audit.backends.chat import render_prompt_ids
from channel_audit.models import ModelConfig
from channel_audit.seeds import SEED_MODULUS

GenerateFn = Callable[..., list[list[int]]]


def left_pad(
    rows: Sequence[Sequence[int]], pad_id: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Pad rows on the left to a common width; return (ids, attention mask)."""
    width = max((len(r) for r in rows), default=0)
    ids = [[pad_id] * (width - len(r)) + list(r) for r in rows]
    mask = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
    return ids, mask


def split_completion(
    row: Sequence[int], eos_ids: set[int], max_new_tokens: int
) -> tuple[list[int], str]:
    """Cut a generated row at its first end-of-turn id (kept) and name why it ended."""
    tokens = list(row)[:max_new_tokens]
    for i, tok in enumerate(tokens):
        if tok in eos_ids:
            return tokens[: i + 1], "stop"
    if len(tokens) < max_new_tokens:
        raise RuntimeError("generation ended before max_new_tokens without an end id")
    return tokens, "length"


def plan_batches(lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Index batches of similar length, to keep padding small."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


def batch_seed(seeds: Sequence[int]) -> int:
    """One seed for a batch, independent of the order of its members."""
    text = ",".join(str(s) for s in sorted(seeds))
    digest = hashlib.sha256(f"batch|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


class HfBackend:
    name = "hf"

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        tokenizer: Any,
        generate_fn: GenerateFn,
        version: str,
        pad_token_id: int,
        eos_token_ids: Sequence[int],
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn
        self.version = version
        self.pad_token_id = pad_token_id
        self.eos_ids = sorted(set(eos_token_ids) | set(cfg.stop_token_ids))

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> HfBackend:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "the hf backend needs torch and transformers: "
                "pip install 'mas-channel-audit[hf]'"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            revision=cfg.revision,
            torch_dtype=getattr(torch, cfg.dtype),
            trust_remote_code=False,
        )
        model = model.to("cuda").eval()
        gen_eos = model.generation_config.eos_token_id
        eos = [gen_eos] if isinstance(gen_eos, int) else list(gen_eos or [])
        if tokenizer.eos_token_id is not None:
            eos.append(tokenizer.eos_token_id)
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id

        def generate_fn(
            input_ids,
            attention_mask,
            *,
            max_new_tokens,
            temperature,
            top_p,
            seed,
            pad_token_id,
            eos_token_ids,
        ):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ids = torch.tensor(input_ids, device=model.device)
            mask = torch.tensor(attention_mask, device=model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    top_k=0,  # off: do not inherit the model's generation_config
                    repetition_penalty=1.0,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_ids,
                )
            return out[:, ids.shape[1] :].tolist()

        version = f"transformers {transformers.__version__}, torch {torch.__version__}"
        return cls(
            cfg,
            tokenizer=tokenizer,
            generate_fn=generate_fn,
            version=version,
            pad_token_id=pad,
            eos_token_ids=eos,
        )

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]:
        results: list[GenResult | None] = [None] * len(requests)
        groups: dict[tuple[float, float, int], list[tuple[int, list[int]]]] = {}
        for i, req in enumerate(requests):
            ids = render_prompt_ids(
                self.tokenizer, req.messages, self.cfg.chat_template_kwargs
            )
            if len(ids) >= self.cfg.max_model_len:
                results[i] = GenResult("", len(ids), 0, "context_overflow")
                continue
            key = (req.temperature, req.top_p, req.max_tokens)
            groups.setdefault(key, []).append((i, ids))
        eos = set(self.eos_ids)
        for (temperature, top_p, max_tokens), members in groups.items():
            lengths = [len(ids) for _, ids in members]
            for batch in plan_batches(lengths, self.cfg.batch_size):
                chosen = [members[j] for j in batch]
                seeds = [requests[i].seed for i, _ in chosen]
                padded, mask = left_pad([ids for _, ids in chosen], self.pad_token_id)
                rows = self.generate_fn(
                    padded,
                    mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seeds[0] if len(seeds) == 1 else batch_seed(seeds),
                    pad_token_id=self.pad_token_id,
                    eos_token_ids=list(self.eos_ids),
                )
                if len(rows) != len(chosen):
                    raise RuntimeError(
                        f"engine returned {len(rows)} rows for {len(chosen)} prompts"
                    )
                for (i, ids), row in zip(chosen, rows, strict=True):
                    tokens, reason = split_completion(row, eos, max_tokens)
                    results[i] = GenResult(
                        text=self.tokenizer.decode(tokens, skip_special_tokens=True),
                        prompt_tokens=len(ids),
                        completion_tokens=len(tokens),
                        finish_reason=reason,
                    )
        return [r for r in results if r is not None]
