"""Answer and confidence parsing. The only extractor.

Only a complete ``\\boxed{...}`` counts as an answer. There is no fallback to
"answer is", "####" or last-number guessing: when extraction fails, the
generation is recorded as having no answer.

The answer is the last box that is not a confidence statement: small models
often box their ``Confidence: N`` line as well, and that box must not be graded
as the answer (found by the M4 screening, where it hid 7.7 points of Qwen3's
accuracy).
"""

from __future__ import annotations

import re
from typing import NamedTuple

_BOXED = re.compile(r"\\boxed\s*\{")
_CONFIDENCE = re.compile(r"confidence\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)
# A box that opens with the word "confidence", optionally inside \text{...}.
_CONFIDENCE_BOX = re.compile(
    r"^\W*(?:\\(?:text|textbf|mathbf|mathrm)\s*\{\s*)?\W*confidence", re.IGNORECASE
)
_WRAPPER = re.compile(r"\\(?:text|textbf|mathbf|mathrm)\s*\{([^{}]*)\}")


def extract_boxed(text: str) -> str | None:
    """Content of the LAST ``\\boxed{...}``, or None.

    The last ``\\boxed{`` decides: if it is never closed (truncated output) the
    result is None, even when an earlier box is complete. Braces escaped as
    ``\\{`` and ``\\}`` do not count towards nesting. An empty box is None.
    """
    if not text:
        return None
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    end = _closing_index(text, starts[-1])
    if end is None:
        return None
    return text[starts[-1] : end].strip() or None


def _closing_index(text: str, start: int) -> int | None:
    """Index of the ``}`` closing a group whose ``{`` sits just before ``start``.

    Braces escaped as ``\\{`` and ``\\}`` do not count towards nesting.
    """
    i, depth = start, 1
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def strip_boxed(text: str, replacement: str = "") -> tuple[str, int]:
    """Replace every ``\\boxed{...}`` by ``replacement``; return text and count.

    An unclosed box is removed through the end of the text, so a truncated
    answer cannot survive. Boxes nested inside a removed box are not counted.
    """
    out: list[str] = []
    pos = count = 0
    for m in _BOXED.finditer(text):
        if m.start() < pos:
            continue
        end = _closing_index(text, m.end())
        out.append(text[pos : m.start()])
        out.append(replacement)
        pos = len(text) if end is None else end + 1
        count += 1
    out.append(text[pos:])
    return "".join(out), count


def _labelled_confidence(text: str, start: int) -> bool:
    """True if "confidence" precedes the box opening at ``start`` on its line."""
    line = text[:start].rsplit("\n", 1)[-1]
    return "confidence" in line.lower()


def _is_confidence_box(content: str, confidence: int | None, labelled: bool) -> bool:
    """A box that states the confidence rather than answering the problem."""
    if labelled or _CONFIDENCE_BOX.match(content):
        return True
    if confidence is None:
        return False
    bare = _WRAPPER.sub(r"\1", content)
    bare = re.sub(r"^\s*N\s*=\s*", "", bare)
    bare = bare.replace("\\%", "").replace("%", "").strip()
    return bare == str(confidence)


def extract_answer(text: str) -> str | None:
    """The last box that is not a confidence statement, else None.

    A box is a confidence statement if it opens with the word "confidence", if
    "confidence" comes before it on the same line (``Confidence: \boxed{90}``),
    or if it holds just the stated ``Confidence: N`` number. Such boxes are
    skipped from the end while an earlier box exists. A lone box that says
    "Confidence ..." is no answer; a lone box holding only a number is kept,
    since it may be the answer. As in ``extract_boxed``, an unclosed last box
    means no answer.
    """
    if not text:
        return None
    boxes: list[tuple[str | None, bool]] = []
    for m in _BOXED.finditer(text):
        end = _closing_index(text, m.end())
        content = None if end is None else text[m.end() : end].strip() or None
        boxes.append((content, _labelled_confidence(text, m.start())))
    if not boxes or boxes[-1][0] is None:
        return None
    confidence = extract_confidence(text)
    while len(boxes) > 1 and _is_confidence_box(
        boxes[-1][0] or "", confidence, boxes[-1][1]
    ):
        boxes.pop()
        if boxes[-1][0] is None:
            return None
    content, labelled = boxes[-1]
    if content is None or labelled or _CONFIDENCE_BOX.match(content):
        return None
    return content


def extract_confidence(text: str) -> int | None:
    """The LAST ``Confidence: N`` as an integer in 0..100, else None.

    If the last occurrence is not a valid integer in range the result is None;
    an earlier valid line is not used instead.
    """
    if not text:
        return None
    matches = list(_CONFIDENCE.finditer(text))
    if not matches:
        return None
    raw = matches[-1].group(1)
    if not raw.lstrip("-").isdigit():
        return None
    value = int(raw)
    return value if 0 <= value <= 100 else None


class Extraction(NamedTuple):
    answer: str | None
    status: str  # "boxed" | "none" | "truncated"


def extract_final(text: str, finish_reason: str) -> Extraction:
    """Answer plus why it is missing, if it is.

    A complete box is an answer even if generation hit the length limit; the
    truncation itself is reported separately from ``finish_reason``.
    """
    answer = extract_answer(text)
    if answer is not None:
        return Extraction(answer, "boxed")
    return Extraction(None, "truncated" if finish_reason == "length" else "none")
