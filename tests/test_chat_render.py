"""Prompts go through the tokenizer's own chat template, then exact ids."""

from __future__ import annotations

import pytest

from channel_audit.backends.chat import render_prompt_ids, render_prompt_text
from fakes import FakeTokenizer

MSGS = [{"role": "user", "content": "What is 1+1?"}]


def test_text_comes_from_the_chat_template_with_a_generation_prompt():
    tok = FakeTokenizer()
    text = render_prompt_text(tok, MSGS, {})
    assert text == "<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n"
    assert tok.template_calls == [{"add_generation_prompt": True}]


def test_template_kwargs_are_passed_through_for_qwen3_non_thinking():
    tok = FakeTokenizer()
    text = render_prompt_text(tok, MSGS, {"enable_thinking": False})
    assert text.endswith("<think>\n\n</think>\n\n")
    assert tok.template_calls[0]["enable_thinking"] is False


def test_ids_are_the_encoding_of_the_rendered_text_without_extra_specials():
    tok = FakeTokenizer()
    ids = render_prompt_ids(tok, MSGS, {})
    assert ids == list(render_prompt_text(FakeTokenizer(), MSGS, {}).encode())
    assert ids[0] != 2  # no second BOS: the template already supplies specials


def test_empty_conversation_is_an_error():
    with pytest.raises(ValueError):
        render_prompt_ids(FakeTokenizer(), [], {})


def test_rendering_is_deterministic():
    assert render_prompt_ids(FakeTokenizer(), MSGS, {}) == render_prompt_ids(
        FakeTokenizer(), MSGS, {}
    )
