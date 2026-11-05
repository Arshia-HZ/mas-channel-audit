"""Channels: pure functions from round-0 records to prompts.

Nothing here calls a model or touches a store. Templates live in
``prompts/<version>/`` and are hashed byte for byte (line endings normalised)
so every generation can name the exact wording it was given. ``placebo`` has no
template of its own: it renders through ``full_solution`` so the wording is
identical by construction.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from channel_audit.extract import extract_answer, extract_confidence
from channel_audit.grade import answers_equivalent
from channel_audit.schema import Generation

PROMPT_VERSION = "v1"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

ARMS: tuple[str, ...] = (
    "self_reflect",
    "answer_only",
    "answer_conf",
    "dissent_only",
    "failure_cause",
    "full_solution",
    "placebo",
)

# Template file (without .txt) used for each kind of generation.
TEMPLATE_FOR: dict[str, str] = {
    "round0": "round0",
    "sc_pool": "round0",
    "screen": "round0",
    "screen_plain": "sanity",
    "failure_msg": "failure_msg",
    **{arm: arm for arm in ARMS},
    "placebo": "full_solution",
}

NO_ANSWER = "(no final answer)"
NO_CONFIDENCE = "(confidence not given)"

_PLACEHOLDER = re.compile(r"<<(\w+)>>")
_CONFIDENCE_LINE = re.compile(
    r"^[ \t]*confidence[ \t]*[:=].*(?:\n|$)", re.IGNORECASE | re.MULTILINE
)


@dataclass(frozen=True)
class AgentView:
    """What a round-0 agent said, as other channels may see it."""

    agent: int
    text: str
    answer: str | None
    confidence: int | None


def agent_view(gen: Generation) -> AgentView:
    return AgentView(
        agent=gen.agent,
        text=gen.text,
        answer=extract_answer(gen.text),
        confidence=extract_confidence(gen.text),
    )


def _template_bytes(name: str) -> bytes:
    path = PROMPTS_DIR / PROMPT_VERSION / f"{TEMPLATE_FOR.get(name, name)}.txt"
    return path.read_bytes().replace(b"\r\n", b"\n")


@cache
def template_text(name: str) -> str:
    return _template_bytes(name).decode("utf-8").rstrip("\n")


def template_hash(name: str) -> str:
    """sha256 of the template an arm or generation kind is rendered from."""
    return hashlib.sha256(_template_bytes(name)).hexdigest()


def render(template: str, **values: str) -> str:
    """Fill ``<<name>>`` placeholders in one pass.

    Single pass means inserted text (a question, a solution) is never
    re-expanded; ``str.format`` is avoided because LaTeX is full of braces.
    """
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def strip_confidence(text: str) -> str:
    """Drop ``Confidence: N`` lines, so only ``answer_conf`` carries a score."""
    return _CONFIDENCE_LINE.sub("", text).rstrip()


def _user(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def build_round0_prompt(question: str) -> list[dict[str, str]]:
    return _user(render(template_text("round0"), question=question))


def build_plain_prompt(question: str) -> list[dict[str, str]]:
    """The published-style prompt for the G0 sanity check (no confidence request)."""
    return _user(render(template_text("screen_plain"), question=question))


def build_failure_msg_prompt(question: str, solution: str) -> list[dict[str, str]]:
    return _user(
        render(
            template_text("failure_msg"),
            question=question,
            solution=strip_confidence(solution),
        )
    )


def disagrees(own: AgentView, peers: Sequence[AgentView]) -> bool:
    """Whether any peer's answer differs from ``own``.

    A missing answer counts as its own value: it differs from any real answer
    and agrees with another missing answer.
    """
    for peer in peers:
        if own.answer is None or peer.answer is None:
            if (own.answer is None) != (peer.answer is None):
                return True
        elif not answers_equivalent(own.answer, peer.answer):
            return True
    return False


def _peer_answers(peers: Sequence[AgentView], with_confidence: bool) -> str:
    lines = []
    for p in peers:
        line = (
            f"Solver {p.agent + 1}: {p.answer if p.answer is not None else NO_ANSWER}"
        )
        if with_confidence:
            conf = (
                NO_CONFIDENCE
                if p.confidence is None
                else f"(confidence {p.confidence})"
            )
            line += f" {conf}"
        lines.append(line)
    return "\n".join(lines)


def _peer_solutions(peers: Sequence[AgentView], texts: Sequence[str]) -> str:
    return "\n\n".join(
        f"Solver {p.agent + 1}:\n{strip_confidence(t)}"
        for p, t in zip(peers, texts, strict=True)
    )


def build_revision_prompt(
    arm: str,
    question: str,
    own: AgentView,
    peers: Sequence[AgentView],
    *,
    failure_message: str | None = None,
    donor_texts: Sequence[str] | None = None,
) -> list[dict[str, str]] | None:
    """The revision prompt for one agent, or None when the arm sends nothing.

    ``peers`` are the other agents' round-0 views in agent order. Only
    ``dissent_only`` can return None (when nobody disagrees). ``placebo`` shows
    ``donor_texts`` (solutions to a different problem) in the peers' slots
    through the very same template and renderer as ``full_solution``.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    values = {"question": question, "own_solution": own.text}
    if arm == "answer_only":
        values["peer_answers"] = _peer_answers(peers, with_confidence=False)
    elif arm == "answer_conf":
        values["peer_answers"] = _peer_answers(peers, with_confidence=True)
    elif arm == "dissent_only":
        if not disagrees(own, peers):
            return None
    elif arm == "failure_cause":
        if failure_message is None:
            raise ValueError("failure_cause needs failure_message")
        values["messages"] = failure_message
    elif arm == "full_solution":
        values["peer_solutions"] = _peer_solutions(peers, [p.text for p in peers])
    elif arm == "placebo":
        if donor_texts is None or len(donor_texts) != len(peers):
            raise ValueError("placebo needs one donor text per peer")
        values["peer_solutions"] = _peer_solutions(peers, donor_texts)
    return _user(render(template_text(arm), **values))
