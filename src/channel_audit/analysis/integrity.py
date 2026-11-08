"""Stored grades must be what the frozen grader gives now (M7).

A shard's SHARD_DONE hash covers its generations only. The analysis reads stored
``graded`` records for some quantities (correct-to-wrong changes) and recomputes
correctness from the extracted answers for others (plurality credit), so before it
runs, every stored grade is compared with a fresh grading of the same text.
"""

from __future__ import annotations

from channel_audit.channels import NON_SOLUTION_ARMS
from channel_audit.grade import grade_generation


def grade_mismatches(ctx) -> list[str]:
    """One line per generation whose stored grade differs from a fresh one."""
    gold = {p.problem_id: p.gold for p in ctx.problems}
    bad = []
    for g in ctx.gens:
        if g.arm in NON_SOLUTION_ARMS:
            continue
        where = f"{g.arm} {g.problem_id}/{g.replicate}/{g.agent}"
        if g.gen_key not in ctx.graded:
            bad.append(f"{where}: no stored grade")
            continue
        stored = ctx.graded.get(g.gen_key)
        fresh = grade_generation(g, gold[g.problem_id])
        if (stored.extracted, stored.extraction_status, stored.correct) != (
            fresh.extracted,
            fresh.extraction_status,
            fresh.correct,
        ):
            bad.append(
                f"{where}: stored {stored.extracted!r}/{stored.extraction_status}/"
                f"{stored.correct}, fresh {fresh.extracted!r}/"
                f"{fresh.extraction_status}/{fresh.correct}"
            )
    return bad
