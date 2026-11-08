"""Failure-message leakage: no peer's final answer may reach the reader.

A message is first sanitised (boxes and final-answer sentences removed), then
checked for *claims* of any round-0 answer (``= 7``, ``is 7``, ``gives 7`` ...).
A bare number that is not claimed as a result, such as "step 2", is not a leak.
If a claim survives, the runner regenerates once; if it survives again the
claim is redacted. The outcome is stored as a ``LeakageRecord``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from channel_audit.extract import strip_boxed

BOX_REPLACEMENT = "[answer removed]"
REDACTION = "[answer]"

_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_ANSWER_SENTENCE = re.compile(
    r"final\s+answer|\banswers?\s+(?:is|are)\b|\banswer\s*[:=]", re.IGNORECASE
)
_CLAIM = (
    r"(?:=|:|\bis\b|\bare\b|\bgives?\b|\bgets?\b|\bequals?\b|\byields?\b"
    r"|\bobtain(?:s|ed)?\b|\bresult(?:s|ed)?\b(?:\s+in\b)?|\banswer\b)"
)


class LeakageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_key: str  # "<problem_id>|<replicate>|<agent>"
    problem_id: str
    replicate: int
    agent: int
    attempts: Literal[1, 2]
    outcome: Literal["clean", "regenerated_clean", "redacted"]
    variant: str = "v1"  # which failure-message prompt wrote it (v1 or v2)
    final_gen_key: str  # the failure_msg generation the message came from
    n_stripped: int  # boxes and answer sentences removed from that generation
    leaked: list[str]  # answers still claimed after sanitising, before redaction
    sanitized_text: str  # exactly what the reader sees


def record_key(problem_id: str, replicate: int, agent: int, variant: str = "v1") -> str:
    base = f"{problem_id}|{replicate}|{agent}"
    return base if variant == "v1" else f"{variant}|{base}"


_FIRST_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9*#$\(])")


def first_sentence(text: str, max_chars: int = 300) -> str:
    """The first sentence of ``text``, without markdown markers, at most ``max_chars``.

    The v2 failure message must be one sentence; anything after the first
    sentence (a re-solution, extra steps) is dropped.
    """
    text = re.sub(r"[#*]+", "", " ".join(text.split()))
    text = _FIRST_SENTENCE_END.split(text.strip(), maxsplit=1)[0].strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return text


def sanitize_message(text: str) -> tuple[str, int]:
    """Remove ``\\boxed{}`` and final-answer sentences; return text and count."""
    text, removed = strip_boxed(text, BOX_REPLACEMENT)
    kept = []
    for sentence in _SENTENCES.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if _ANSWER_SENTENCE.search(sentence):
            removed += 1
            continue
        kept.append(sentence)
    return " ".join(kept), removed


def _claim_pattern(answer: str) -> re.Pattern[str]:
    return re.compile(
        rf"({_CLAIM}\s*\$?\s*)({re.escape(answer)})(?!\w)(?!\.\d)", re.IGNORECASE
    )


def _distinct(answers: Sequence[str | None]) -> list[str]:
    return list(dict.fromkeys(a.strip() for a in answers if a and a.strip()))


def find_leaks(text: str, answers: Sequence[str | None]) -> list[str]:
    """Answers that ``text`` claims as a result."""
    return [a for a in _distinct(answers) if _claim_pattern(a).search(text)]


def redact(text: str, answers: Sequence[str | None]) -> str:
    """Replace each claimed answer with a placeholder; other numbers stay."""
    for answer in _distinct(answers):
        text = _claim_pattern(answer).sub(lambda m: m.group(1) + REDACTION, text)
    return text
