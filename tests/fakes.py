"""Fake tokenizer and engines so real-backend code is tested without a GPU."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

SPECIAL = {0, 1}  # ids skipped when decoding with skip_special_tokens=True


class FakeTokenizer:
    """Byte-level tokenizer with a Qwen-style chat template."""

    def __init__(self):
        self.template_calls: list[dict] = []

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, **kwargs
    ):
        assert tokenize is False, "backends must render text, then encode it"
        self.template_calls.append(
            {"add_generation_prompt": add_generation_prompt, **kwargs}
        )
        out = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is False:
                out += "<think>\n\n</think>\n\n"
        return out

    def encode(self, text, add_special_tokens=True):
        ids = list(text.encode("utf-8"))
        return ([2] + ids) if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=False):
        keep = [i for i in ids if not (skip_special_tokens and i in SPECIAL)]
        return bytes(i % 256 for i in keep).decode("utf-8", errors="replace")


@dataclass
class FakeSamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 16
    seed: int | None = None
    stop_token_ids: list[int] = field(default_factory=list)
    n: int = 1
    top_k: int = 0
    repetition_penalty: float = 0.0


@dataclass
class FakeCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str


@dataclass
class FakeRequestOutput:
    prompt_token_ids: list[int]
    outputs: list[FakeCompletion]


class FakeLLM:
    """Duck-typed vllm.LLM: records calls, answers via ``reply``."""

    def __init__(self, reply: Callable | None = None):
        self.calls: list[tuple[list, list]] = []
        self.reply = reply or self._default_reply

    @staticmethod
    def _default_reply(prompt_ids, params):
        token_ids = [104, 105, 0]  # "hi" + a special stop token
        return FakeCompletion("hi", token_ids, "stop")

    def generate(self, prompts, sampling_params, use_tqdm=True):
        self.calls.append((list(prompts), list(sampling_params)))
        return [
            FakeRequestOutput(
                p["prompt_token_ids"], [self.reply(p["prompt_token_ids"], sp)]
            )
            for p, sp in zip(prompts, sampling_params, strict=True)
        ]
