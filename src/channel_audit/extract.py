"""Answer and confidence parsing. The only extractor.

Only a complete ``\\boxed{...}`` counts as an answer. There is no fallback to
"answer is", "####" or last-number guessing: when extraction fails, the
generation is recorded as having no answer.
"""

from __future__ import annotations

import re
from typing import NamedTuple

_BOXED = re.compile(r"\\boxed\s*\{")
_CONFIDENCE = re.compile(r"confidence\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)


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
    i, depth, buf = starts[-1], 1, []
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in "{}":
            buf.append(text[i : i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf).strip() or None
        buf.append(ch)
        i += 1
    return None


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
    answer = extract_boxed(text)
    if answer is not None:
        return Extraction(answer, "boxed")
    return Extraction(None, "truncated" if finish_reason == "length" else "none")
