"""Token accounting and budget-matched self-consistency.

An arm's cost per replicate is its round-0 generations plus everything the arm
generated afterwards. ``failure_cause`` also pays for every failure-message
attempt, rejected ones included. Self-consistency draws from the 3 round-0
samples plus the pool; its accuracy at ``k`` samples is the exact average over
all size-``k`` subsets, and fractional ``k`` interpolates linearly.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from channel_audit.aggregate import cluster_answers
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.grade import is_correct
from channel_audit.schema import Generation


class BudgetOutOfRange(ValueError):
    """The matched self-consistency sample count lies outside what was drawn."""


@dataclass(frozen=True)
class ArmCost:
    arm: str
    n_replicates: int
    generated_tokens: float  # mean per replicate, completion tokens only
    total_tokens: float  # same, plus prompt tokens (sensitivity view)


def _cost(gens: Iterable[Generation], include_prompt: bool) -> int:
    return sum(
        g.completion_tokens + (g.prompt_tokens if include_prompt else 0) for g in gens
    )


def arm_costs(
    gens: Iterable[Generation], arms: Sequence[str] | None = None
) -> dict[str, ArmCost]:
    """Mean cost per replicate, for every replicate that has round-0 data.

    Precondition: the run is complete for the arms asked about (see
    ``validate.validate_run``); an arm with no generations in a replicate is
    taken to have generated nothing there, which is correct for ``dissent_only``.
    """
    round0: dict[tuple[str, int], list[Generation]] = defaultdict(list)
    failure: dict[tuple[str, int], list[Generation]] = defaultdict(list)
    failure_v2: dict[tuple[str, int], list[Generation]] = defaultdict(list)
    by_arm: dict[str, dict[tuple[str, int], list[Generation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for g in gens:
        key = (g.problem_id, g.replicate)
        if g.arm == "round0":
            round0[key].append(g)
        elif g.arm == "failure_msg":
            failure[key].append(g)
        elif g.arm == "failure_msg_v2":
            failure_v2[key].append(g)
        elif g.arm in ARMS or g.arm in VARIANT_ARMS:
            by_arm[g.arm][key].append(g)
    if arms is None:
        arms = [a for a in (*ARMS, *VARIANT_ARMS) if a in by_arm]
    out = {}
    for arm in arms:
        totals = {False: [], True: []}
        for key, r0 in round0.items():
            members = list(r0) + list(by_arm[arm].get(key, []))
            if arm == "failure_cause":
                members += failure.get(key, [])
            elif arm == "failure_cause_v2":
                members += failure_v2.get(key, [])
            for include_prompt in (False, True):
                totals[include_prompt].append(_cost(members, include_prompt))
        n = len(totals[False])
        out[arm] = ArmCost(
            arm=arm,
            n_replicates=n,
            generated_tokens=sum(totals[False]) / n if n else math.nan,
            total_tokens=sum(totals[True]) / n if n else math.nan,
        )
    return out


def sc_sample_cost(gens: Iterable[Generation]) -> tuple[float, float]:
    """Mean tokens per self-consistency sample: (generated, generated + prompt)."""
    pool = [g for g in gens if g.arm in ("round0", "sc_pool")]
    if not pool:
        raise ValueError("no round0 or sc_pool generations")
    return _cost(pool, False) / len(pool), _cost(pool, True) / len(pool)


def matched_sample_count(
    arm_tokens: float, per_sample_tokens: float, max_samples: int
) -> float:
    """Self-consistency sample count that costs as much as the arm."""
    k = arm_tokens / per_sample_tokens
    if not 1 <= k <= max_samples:
        raise BudgetOutOfRange(
            f"matched sample count {k:.2f} outside [1, {max_samples}]"
        )
    return k


def sc_credit(answers: Sequence[str | None], gold: str, k: int) -> float:
    """Expected plurality credit of ``k`` samples drawn from ``answers``."""
    n = len(answers)
    if not 1 <= k <= n:
        raise ValueError(f"k={k} outside [1, {n}]")
    labels, reps = cluster_answers(answers)
    correct = {i for i, rep in enumerate(reps) if is_correct(rep, gold)}
    total = 0.0
    n_subsets = 0
    for subset in combinations(range(n), k):
        n_subsets += 1
        counts = Counter(labels[i] for i in subset if labels[i] is not None)
        if not counts:
            continue
        top = max(counts.values())
        winners = [label for label, c in counts.items() if c == top]
        if any(label in correct for label in winners):
            total += 1 / len(winners)
    return total / n_subsets


def sc_credit_interp(answers: Sequence[str | None], gold: str, k: float) -> float:
    """``sc_credit`` at a fractional sample count, linear between integers."""
    if not 1 <= k <= len(answers):
        raise BudgetOutOfRange(f"k={k} outside [1, {len(answers)}]")
    lo, hi = math.floor(k), math.ceil(k)
    at_lo = sc_credit(answers, gold, lo)
    if lo == hi:
        return at_lo
    return at_lo + (k - lo) * (sc_credit(answers, gold, hi) - at_lo)


def matched_sc_accuracy(
    items: Iterable[tuple[Sequence[str | None], str]], k: float
) -> float:
    """Mean interpolated self-consistency credit over (answers, gold) replicates."""
    credits = [sc_credit_interp(answers, gold, k) for answers, gold in items]
    return sum(credits) / len(credits)
