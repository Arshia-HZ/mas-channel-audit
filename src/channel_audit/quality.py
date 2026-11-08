"""Run quality by arm: truncation, context overflow, extraction failure, leakage.

The plan reports these on every run. Nothing here reads a correctness mark or an
accuracy, so a confirmatory run can be checked shard by shard without looking at
any outcome of the contrasts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

from channel_audit.channels import NON_SOLUTION_ARMS
from channel_audit.runner import RunContext

LEAKAGE_OUTCOMES = ("clean", "regenerated_clean", "redacted")


def run_quality(ctxs: Sequence[RunContext]) -> dict:
    """Per-arm rates and the failure-message leakage outcomes, pooled over shards."""
    limit = ctxs[0].cfg.context_limit
    n: Counter = Counter()
    truncated: Counter = Counter()
    overflow: Counter = Counter()
    no_box: Counter = Counter()
    no_answer: Counter = Counter()
    for ctx in ctxs:
        for g in ctx.gens:
            n[g.arm] += 1
            truncated[g.arm] += g.finish_reason == "length"
            overflow[g.arm] += g.prompt_tokens + g.max_tokens > limit
            if g.arm in NON_SOLUTION_ARMS:
                continue
            status = ctx.graded.get(g.gen_key).extraction_status
            no_box[g.arm] += status == "none"
            no_answer[g.arm] += status != "boxed"
    arms = {
        arm: {
            "n": n[arm],
            "truncation_rate": truncated[arm] / n[arm],
            "overflow": overflow[arm],
            **(
                {}
                if arm in NON_SOLUTION_ARMS
                else {
                    "extraction_failure_rate": no_box[arm] / n[arm],
                    "no_answer_rate": no_answer[arm] / n[arm],
                }
            ),
        }
        for arm in n
    }
    outcomes: dict[str, Counter] = defaultdict(Counter)
    for ctx in ctxs:
        for rec in ctx.leakage:
            outcomes[rec.variant][rec.outcome] += 1
    leakage = {}
    for variant, counts in outcomes.items():
        total = sum(counts.values())
        leakage[variant] = {
            "n": total,
            **{o: counts[o] for o in LEAKAGE_OUTCOMES},
            "rejection_rate": 1 - counts["clean"] / total,
        }
    return {
        "n_problems": len({p.problem_id for ctx in ctxs for p in ctx.problems}),
        "replicates": ctxs[0].cfg.replicates,
        "arms": arms,
        "leakage": leakage,
    }


def format_quality_report(q: dict) -> str:
    lines = [
        f"{q['n_problems']} problems x {q['replicates']} replicates",
        "",
        f"{'arm':<18}{'n':>7}{'trunc':>8}{'overflow':>10}"
        f"{'no box':>8}{'no answer':>11}",
    ]
    for arm, r in sorted(q["arms"].items()):
        box = r.get("extraction_failure_rate")
        ans = r.get("no_answer_rate")
        lines.append(
            f"{arm:<18}{r['n']:>7}{r['truncation_rate']:>8.1%}{r['overflow']:>10}"
            f"{('-' if box is None else f'{box:.1%}'):>8}"
            f"{('-' if ans is None else f'{ans:.1%}'):>11}"
        )
    lines += ["", "failure-message leakage (rejected = not clean on the first try)"]
    for variant, r in sorted(q["leakage"].items()):
        lines.append(
            f"  {variant}: {r['n']} messages, {r['clean']} clean, "
            f"{r['regenerated_clean']} regenerated, {r['redacted']} redacted "
            f"(rejection {r['rejection_rate']:.1%})"
        )
    return "\n".join(lines)
