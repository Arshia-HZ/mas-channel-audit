"""vLLM backend: the default for confirmatory runs (T4 and up).

Prompts are rendered with the tokenizer's chat template and sent as token ids,
so the prompt token count is exact. Every request carries its own seed.
``completion_tokens`` is the engine's ``len(token_ids)``. A prompt that leaves
no room to generate is recorded as ``finish_reason="context_overflow"`` instead
of being sent (and instead of raising), so validation can report it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from channel_audit.backends.base import GenRequest, GenResult
from channel_audit.backends.chat import render_prompt_ids
from channel_audit.models import ModelConfig


class VllmBackend:
    name = "vllm"

    def __init__(
        self,
        cfg: ModelConfig,
        *,
        llm: Any,
        tokenizer: Any,
        sampling_params_cls: Any,
        version: str,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params_cls = sampling_params_cls
        self.version = version

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> VllmBackend:
        try:
            import vllm
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "the vllm backend needs vLLM: pip install 'mas-channel-audit[vllm]'"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)
        llm = vllm.LLM(
            model=cfg.model_id,
            revision=cfg.revision,
            tokenizer_revision=cfg.revision,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enforce_eager=cfg.enforce_eager,
            trust_remote_code=False,
            seed=0,
        )
        return cls(
            cfg,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params_cls=vllm.SamplingParams,
            version=vllm.__version__,
        )

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]:
        results: list[GenResult | None] = [None] * len(requests)
        prompts, params, slots = [], [], []
        for i, req in enumerate(requests):
            ids = render_prompt_ids(
                self.tokenizer, req.messages, self.cfg.chat_template_kwargs
            )
            if len(ids) >= self.cfg.max_model_len:
                results[i] = GenResult("", len(ids), 0, "context_overflow")
                continue
            prompts.append({"prompt_token_ids": ids})
            params.append(
                self.sampling_params_cls(
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=-1,  # no model-default top-k
                    repetition_penalty=1.0,  # no model-default penalty
                    max_tokens=req.max_tokens,
                    seed=req.seed,
                    stop_token_ids=list(self.cfg.stop_token_ids),
                    n=1,
                )
            )
            slots.append((i, len(ids)))
        if prompts:
            outputs = self.llm.generate(prompts, params, use_tqdm=False)
            if len(outputs) != len(prompts):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
                )
            for (i, n_prompt), out in zip(slots, outputs, strict=True):
                completion = out.outputs[0]
                if completion.finish_reason is None:
                    raise RuntimeError("vLLM returned an unfinished completion")
                results[i] = GenResult(
                    text=completion.text,
                    prompt_tokens=n_prompt,
                    completion_tokens=len(completion.token_ids),
                    finish_reason=completion.finish_reason,
                )
        return [r for r in results if r is not None]
