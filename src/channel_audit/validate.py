"""Run validation: expected counts, linkage, grading, gates and budgets."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from channel_audit.budget import (
    ArmCost,
    BudgetOutOfRange,
    arm_costs,
    matched_sample_count,
    sc_sample_cost,
)
from channel_audit.channels import ARMS, agent_view, disagrees
from channel_audit.leakage import record_key
from channel_audit.runner import (
    N_AGENTS,
    POOL_AGENT,
    REVISION_ROUND,
    MissingRound0Error,
    RunContext,
    identity_for,
    round0_gens,
)
from channel_audit.seeds import gen_key

TRUNCATION_GATE = 0.05  # G3: each revision arm must truncate under 5%
TRUNCATION_GAP_WARNING = 0.02  # report when an arm differs from self_reflect


@dataclass(frozen=True)
class G3Result:
    """Gate G3 (PREREG section 3): zero overflow, each revision arm under 5%.

    Round 0, the self-consistency pool and failure messages are reported but have
    no threshold: every arm shares them, so their truncation cannot favour one.
    """

    ok: bool
    overflow_total: int
    truncation: dict[str, float]  # every kind of generation, for the report
    overflow: dict[str, int]
    failing_arms: list[str]  # revision arms at or above the truncation gate
    flagged_arms: list[str]  # revision arms >2 points from self_reflect


def g3_check(gens, limit: int) -> G3Result:
    by_arm: dict[str, list] = defaultdict(list)
    for g in gens:
        by_arm[g.arm].append(g)
    truncation = {
        arm: sum(g.finish_reason == "length" for g in members) / len(members)
        for arm, members in by_arm.items()
    }
    overflow = {
        arm: sum(g.prompt_tokens + g.max_tokens > limit for g in members)
        for arm, members in by_arm.items()
    }
    failing = [a for a in ARMS if truncation.get(a, 0.0) >= TRUNCATION_GATE]
    base = truncation.get("self_reflect")
    flagged = [
        a
        for a in ARMS
        if base is not None
        and a in truncation
        and abs(truncation[a] - base) > TRUNCATION_GAP_WARNING
    ]
    total = sum(overflow.values())
    return G3Result(
        not failing and total == 0, total, truncation, overflow, failing, flagged
    )


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    truncation_rate: dict[str, float] = field(default_factory=dict)
    overflow_count: dict[str, int] = field(default_factory=dict)
    costs: dict[str, ArmCost] = field(default_factory=dict)
    matched_k: dict[str, float] = field(default_factory=dict)
    leakage_outcomes: dict[str, int] = field(default_factory=dict)
    g3_ok: bool = True
    g3_failing_arms: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _expected_keys(ctx: RunContext, report: ValidationReport) -> dict[str, set[str]]:
    cfg, expected = ctx.cfg, defaultdict(set)

    def add(arm, problem_id, rep, agent, round_, sample_idx):
        expected[arm].add(
            gen_key(
                **identity_for(cfg, problem_id, rep, agent, arm, round_, sample_idx)
            )
        )

    for p in ctx.problems:
        for rep in range(cfg.replicates):
            for agent in range(N_AGENTS):
                add("round0", p.problem_id, rep, agent, 0, 0)
                add("failure_msg", p.problem_id, rep, agent, REVISION_ROUND, 0)
                key = record_key(p.problem_id, rep, agent)
                if key in ctx.leakage and ctx.leakage.get(key).attempts == 2:
                    add("failure_msg", p.problem_id, rep, agent, REVISION_ROUND, 1)
            for i in range(cfg.n_pool):
                add("sc_pool", p.problem_id, rep, POOL_AGENT, 0, i)
            try:
                views = [agent_view(g) for g in round0_gens(ctx, p.problem_id, rep)]
            except MissingRound0Error:
                continue  # round0 itself is reported missing; arms cannot be derived
            for arm in ARMS:
                for i in range(N_AGENTS):
                    peers = [v for j, v in enumerate(views) if j != i]
                    if arm == "dissent_only" and not disagrees(views[i], peers):
                        continue
                    add(arm, p.problem_id, rep, i, REVISION_ROUND, 0)
    return expected


def validate_run(ctx: RunContext) -> ValidationReport:
    report = ValidationReport()
    gens = list(ctx.gens)
    by_arm: dict[str, list] = defaultdict(list)
    for g in gens:
        by_arm[g.arm].append(g)
    report.counts = {arm: len(v) for arm, v in by_arm.items()}

    expected = _expected_keys(ctx, report)
    present = {arm: {g.gen_key for g in v} for arm, v in by_arm.items()}
    for arm, keys in expected.items():
        missing = len(keys - present.get(arm, set()))
        if missing:
            report.errors.append(f"{arm}: {missing} of {len(keys)} generations missing")
    for arm, keys in present.items():
        unexpected = len(keys - expected.get(arm, set()))
        if unexpected:
            report.errors.append(f"{arm}: {unexpected} unexpected generations")

    for g in gens:
        if any(k not in ctx.gens for k in g.parent_gen_keys):
            report.errors.append(f"{g.gen_key[:12]} ({g.arm}) has a missing parent")
    r0_keys: dict[tuple[str, int], list[str]] = {}
    for p in ctx.problems:
        for rep in range(ctx.cfg.replicates):
            try:
                r0_keys[(p.problem_id, rep)] = [
                    g.gen_key for g in round0_gens(ctx, p.problem_id, rep)
                ]
            except MissingRound0Error:
                continue
    for g in gens:
        keys = r0_keys.get((g.problem_id, g.replicate))
        if g.arm in ARMS and keys is not None and g.parent_gen_keys[:3] != keys:
            report.errors.append(
                f"{g.arm} {g.problem_id}/{g.replicate}/{g.agent} not linked to round 0"
            )

    ungraded = sum(
        1 for g in gens if g.arm != "failure_msg" and g.gen_key not in ctx.graded
    )
    if ungraded:
        report.errors.append(f"{ungraded} generations not graded")

    g3 = g3_check(gens, ctx.cfg.context_limit)
    report.truncation_rate = g3.truncation
    report.overflow_count = g3.overflow
    report.g3_ok = g3.ok
    report.g3_failing_arms = g3.failing_arms
    base = g3.truncation.get("self_reflect")
    for arm in g3.flagged_arms:
        report.warnings.append(
            f"{arm}: truncation {g3.truncation[arm]:.1%} vs self_reflect {base:.1%}; "
            "repeat the contrast excluding affected replicates"
        )

    report.leakage_outcomes = dict(Counter(r.outcome for r in ctx.leakage))

    if not report.errors:
        per_sample, _ = sc_sample_cost(gens)
        report.costs = arm_costs(gens)
        for arm, cost in report.costs.items():
            try:
                report.matched_k[arm] = matched_sample_count(
                    cost.generated_tokens, per_sample, 3 + ctx.cfg.n_pool
                )
            except BudgetOutOfRange as exc:
                report.errors.append(f"{arm}: budget {exc}")
    return report
