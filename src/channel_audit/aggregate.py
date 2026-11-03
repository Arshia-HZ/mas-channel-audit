"""Plurality voting with 1/m tie credit.

Used for every arm and for self-consistency. Unparsable answers (None or blank)
never vote. Answers that denote the same value (via ``grade``) pool their votes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from channel_audit.grade import answers_equivalent, is_correct


@dataclass(frozen=True)
class Plurality:
    winners: list[str]  # one representative answer per tied top group
    n_votes: int  # parsable votes cast
    top_count: int  # votes held by each tied top group

    @property
    def n_tied(self) -> int:
        return len(self.winners)


def cluster_answers(
    answers: Sequence[str | None],
) -> tuple[list[int | None], list[str]]:
    """Label each answer by its equivalence group.

    Returns (labels, representatives). A label is the index of the group's
    first member in ``representatives``; unparsable answers get None.
    """
    reps: list[str] = []
    labels: list[int | None] = []
    for answer in answers:
        if answer is None or not answer.strip():
            labels.append(None)
            continue
        for i, rep in enumerate(reps):
            if answers_equivalent(rep, answer):
                labels.append(i)
                break
        else:
            reps.append(answer)
            labels.append(len(reps) - 1)
    return labels, reps


def plurality(answers: Sequence[str | None]) -> Plurality:
    """Group equivalent answers and return the top group(s)."""
    labels, reps = cluster_answers(answers)
    counts = [0] * len(reps)
    for label in labels:
        if label is not None:
            counts[label] += 1
    if not reps:
        return Plurality(winners=[], n_votes=0, top_count=0)
    top = max(counts)
    return Plurality(
        winners=[rep for rep, c in zip(reps, counts, strict=True) if c == top],
        n_votes=sum(counts),
        top_count=top,
    )


def plurality_credit(answers: Sequence[str | None], gold: str) -> float:
    """1 if the unique plurality answer is correct; 1/m if it is one of m tied."""
    p = plurality(answers)
    if p.n_tied == 0:
        return 0.0
    if any(is_correct(w, gold) for w in p.winners):
        return 1.0 / p.n_tied
    return 0.0
