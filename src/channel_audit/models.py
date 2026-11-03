"""Model configs: everything a backend needs to load and decode one model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    backend: Literal["vllm", "hf"]
    dtype: Literal["float16", "bfloat16", "float32"]
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")  # a commit, never a branch
    max_tokens: int = Field(ge=1)  # identical for every arm of a model
    max_model_len: int = Field(ge=1)  # context window the engine is started with
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)  # vLLM
    enforce_eager: bool = False  # vLLM: skip CUDA graphs
    batch_size: int = Field(default=8, ge=1)  # HF only
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Extra ids that end a turn. Added to whatever the model's own config says.
    stop_token_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.max_tokens >= self.max_model_len:
            raise ValueError("max_tokens must be smaller than max_model_len")
        if "qwen3" in self.model_id.lower():
            if self.chat_template_kwargs.get("enable_thinking") is not False:
                raise ValueError(
                    "Qwen3 must run in non-thinking mode: set "
                    "chat_template_kwargs: {enable_thinking: false}"
                )
        return self


def load_model_config(path: str | os.PathLike) -> ModelConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a model config must be a mapping")
    return ModelConfig(**data)
