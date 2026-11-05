"""The only grader.

Raises at import if ``math_verify`` is missing or not working. Phase 1 fell
back to string matching silently; here a broken grader is a startup failure,
never a quiet drop in accuracy.

Grading takes an already-extracted answer (see ``extract.py``). Raw generation
text is never graded: no box means no answer means incorrect.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache

from channel_audit.extract import extract_confidence, extract_final
from channel_audit.mv_guard import GuardedCall
from channel_audit.schema import Generation, Graded

try:
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify
except ImportError as exc:  # pragma: no cover - exercised by test via sys.modules
    raise ImportError(
        "channel_audit.grade requires math-verify (pip install math-verify); "
        "refusing to grade without it"
    ) from exc

# math_verify enforces timeouts with SIGALRM on POSIX but with a subprocess on
# Windows, where the subprocess can fail and every answer then parses to []
# (silently "incorrect"). Windows is development-only here, so run without a
# timeout there; Kaggle and CI are POSIX and keep the 5 s default.
_TIMEOUT: int | None = None if sys.platform == "win32" else 5


def _normalize(s: str) -> str:
    """Conservative normalisation used for the exact-match path.

    Ported from Phase 1. Covers answers math_verify cannot parse (plain text
    such as names or multiple-choice letters) and display variants such as
    ``\\dfrac`` vs ``\\frac``.
    """
    s = s.strip().rstrip(".").strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("$", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)
    s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
    s = s.replace(" ", "")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    s = s.lower()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _mv_equal_direct(a: str, b: str) -> bool:
    ga = _mv_parse(f"${a}$", parsing_timeout=_TIMEOUT)
    gb = _mv_parse(f"${b}$", parsing_timeout=_TIMEOUT)
    return bool(_mv_verify(ga, gb, timeout_seconds=_TIMEOUT))


# Windows has no signal-based timeout, so a hard symbolic pair could hang a long
# job. There the comparison runs in a worker that is killed after 10 s (the pair
# then counts as not equivalent); POSIX keeps math_verify's own timeout.
_GUARD = (
    GuardedCall("channel_audit.grade:_mv_equal_direct", timeout=10.0)
    if sys.platform == "win32"
    else None
)


def _mv_equal(a: str, b: str) -> bool:
    return _GUARD(a, b) if _GUARD is not None else _mv_equal_direct(a, b)


@lru_cache(maxsize=65536)
def answers_equivalent(a: str, b: str) -> bool:
    """Whether two extracted answers denote the same value."""
    if not a.strip() or not b.strip():
        return False
    return _normalize(a) == _normalize(b) or _mv_equal(a, b)


def is_correct(extracted: str | None, gold: str | None) -> bool:
    """Grade one EXTRACTED answer against gold. None or blank is never correct.

    The input must come from ``extract.py``; free text is not searched here.
    The no-box case is enforced one level up, in ``grade_generation``.
    """
    if extracted is None or gold is None:
        return False
    return answers_equivalent(extracted, gold)


def grade_generation(gen: Generation, gold: str) -> Graded:
    """Extract and grade one generation."""
    extraction = extract_final(gen.text, gen.finish_reason)
    return Graded(
        gen_key=gen.gen_key,
        extracted=extraction.answer,
        extraction_status=extraction.status,
        confidence=extract_confidence(gen.text),
        correct=is_correct(extraction.answer, gold),
    )


def _self_test() -> None:
    cases = [("0.5", r"\frac{1}{2}", True), ("3", "4", False)]
    for a, b, expected in cases:
        if _mv_equal_direct(a, b) is not expected:
            raise RuntimeError(
                f"math_verify self-test failed: equal({a!r}, {b!r}) != {expected}; "
                "the verifier is not working in this environment"
            )


_self_test()
