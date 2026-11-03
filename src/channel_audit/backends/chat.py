"""Prompt rendering shared by every real backend.

Messages go through the tokenizer's own chat template, then are encoded
without adding special tokens (the template already contains them). Both
backends therefore send identical token ids for identical messages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def render_prompt_text(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    chat_template_kwargs: Mapping[str, Any],
) -> str:
    if not messages:
        raise ValueError("cannot render an empty conversation")
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        **chat_template_kwargs,
    )


def render_prompt_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    chat_template_kwargs: Mapping[str, Any],
) -> list[int]:
    text = render_prompt_text(tokenizer, messages, chat_template_kwargs)
    return list(tokenizer.encode(text, add_special_tokens=False))
