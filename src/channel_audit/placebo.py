"""Placebo donor matching.

A donor is another problem with the same subject and level whose round-0
solutions are closest in total length to the peer solutions they replace. The
choice uses lengths only, never correctness or any arm outcome.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from channel_audit.schema import Problem


class NoDonorError(LookupError):
    """No other problem with the same subject and level has round-0 data."""


def pick_donor(
    problem: Problem,
    pool: Sequence[Problem],
    target_tokens: int,
    tokens_by_problem: Mapping[str, int],
    *,
    tiebreak: str,
) -> Problem:
    """Nearest-length donor; ties break on a hash of ``tiebreak`` and the id.

    ``tokens_by_problem`` holds each candidate's token total for the solutions
    that would be shown; candidates missing from it are ignored.
    """
    candidates = [
        q
        for q in pool
        if q.problem_id != problem.problem_id
        and q.subject == problem.subject
        and q.level == problem.level
        and q.problem_id in tokens_by_problem
    ]
    if not candidates:
        raise NoDonorError(
            f"no donor for {problem.problem_id} "
            f"(subject={problem.subject}, level={problem.level})"
        )
    return min(
        candidates,
        key=lambda q: (
            abs(tokens_by_problem[q.problem_id] - target_tokens),
            hashlib.sha256(f"{tiebreak}|{q.problem_id}".encode()).hexdigest(),
        ),
    )
