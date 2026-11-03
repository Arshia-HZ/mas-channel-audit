"""Shard execution: plan generations, skip finished ones, append the rest.

Every step is idempotent and resumable: a generation's ``gen_key`` comes from
its identity, so a restarted job only produces what the store lacks. Steps run
in the order round 0, self-consistency pool, failure messages, arm branches;
every arm branches from the same stored round 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from channel_audit import channels
from channel_audit.backends.base import Backend, GenRequest
from channel_audit.backends.mock import MockBackend, MockWorld
from channel_audit.channels import ARMS, AgentView, agent_view, disagrees
from channel_audit.grade import grade_generation
from channel_audit.leakage import (
    LeakageRecord,
    find_leaks,
    record_key,
    redact,
    sanitize_message,
)
from channel_audit.placebo import pick_donor
from channel_audit.schema import Generation, Graded, Problem
from channel_audit.seeds import derive_seed, gen_key
from channel_audit.store import JsonlStore

N_AGENTS = 3
POOL_AGENT = -1  # sc_pool samples belong to no agent
REVISION_ROUND = 1


class MissingRound0Error(LookupError):
    """A step that branches from round 0 ran before round 0 was complete."""


class MissingFailureMessageError(LookupError):
    """failure_cause ran before the failure messages were resolved."""


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    backend: str = "mock"
    temperature: float = 0.7
    max_tokens: int = Field(default=512, ge=1)
    replicates: int = Field(default=2, ge=1)
    n_pool: int = Field(default=6, ge=0)
    prompt_version: str = channels.PROMPT_VERSION
    context_limit: int = Field(default=4096, ge=1)


@dataclass
class RunContext:
    out_dir: Path
    cfg: RunConfig
    backend: Backend
    problems: list[Problem]
    gens: JsonlStore[Generation]
    graded: JsonlStore[Graded]
    leakage: JsonlStore[LeakageRecord]


def backend_for_run(cfg: RunConfig, problems: Sequence[Problem]) -> Backend:
    if cfg.backend == "mock":
        return MockBackend(MockWorld.from_problems(problems))
    raise NotImplementedError(f"backend {cfg.backend!r}: real backends arrive in M3")


def _context(
    out: Path, cfg: RunConfig, problems: list[Problem], backend: Backend
) -> RunContext:
    return RunContext(
        out_dir=out,
        cfg=cfg,
        backend=backend,
        problems=problems,
        gens=JsonlStore(out / "generations.jsonl", Generation),
        graded=JsonlStore(out / "graded.jsonl", Graded),
        leakage=JsonlStore(out / "leakage.jsonl", LeakageRecord, key="record_key"),
    )


def create_run(
    out_dir, cfg: RunConfig, problems: Sequence[Problem], backend: Backend
) -> RunContext:
    """Start a run directory, or re-attach to one with the identical config."""
    if cfg.prompt_version != channels.PROMPT_VERSION:
        raise ValueError(f"unsupported prompt_version {cfg.prompt_version!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "run.json"
    if cfg_path.exists():
        if RunConfig.model_validate_json(cfg_path.read_text("utf-8")) != cfg:
            raise ValueError(f"{cfg_path} holds a different config; use a new run dir")
    else:
        cfg_path.write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    store = JsonlStore(out / "problems.jsonl", Problem, key="problem_id")
    if len(store):
        if [p.problem_id for p in store] != [p.problem_id for p in problems]:
            raise ValueError("run directory holds a different problem set")
    else:
        for p in problems:
            store.append(p)
    return _context(out, cfg, list(store), backend)


def open_run(out_dir, backend: Backend | None = None) -> RunContext:
    """Re-open an existing run directory."""
    out = Path(out_dir)
    cfg = RunConfig.model_validate_json((out / "run.json").read_text("utf-8"))
    problems = list(JsonlStore(out / "problems.jsonl", Problem, key="problem_id"))
    return _context(out, cfg, problems, backend or backend_for_run(cfg, problems))


def identity_for(
    cfg: RunConfig,
    problem_id: str,
    replicate: int,
    agent: int,
    arm: str,
    round_: int,
    sample_idx: int,
) -> dict:
    return {
        "problem_id": problem_id,
        "replicate": replicate,
        "agent": agent,
        "model_id": cfg.model_id,
        "arm": arm,
        "round": round_,
        "sample_idx": sample_idx,
        "prompt_version": cfg.prompt_version,
        "prompt_hash": channels.template_hash(arm),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }


class Planned(NamedTuple):
    identity: dict
    messages: list[dict[str, str]]
    parents: list[str]


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _execute(ctx: RunContext, planned: Sequence[Planned]) -> list[Generation]:
    """Generate what the store lacks; return every planned record in order."""
    todo = [p for p in planned if gen_key(**p.identity) not in ctx.gens]
    if todo:
        requests = [
            GenRequest(
                messages=p.messages,
                seed=derive_seed(**p.identity),
                temperature=ctx.cfg.temperature,
                max_tokens=ctx.cfg.max_tokens,
                meta={
                    "problem_id": p.identity["problem_id"],
                    "replicate": p.identity["replicate"],
                    "agent": p.identity["agent"],
                    "arm": p.identity["arm"],
                },
            )
            for p in todo
        ]
        results = ctx.backend.generate(requests)
        if len(results) != len(requests):
            raise RuntimeError(
                f"backend returned {len(results)} results for {len(requests)} requests"
            )
        for p, res in zip(todo, results, strict=True):
            ctx.gens.append(
                Generation.create(
                    **p.identity,
                    backend=ctx.backend.name,
                    backend_version=ctx.backend.version,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                    finish_reason=res.finish_reason,
                    text=res.text,
                    parent_gen_keys=p.parents,
                    created_at=_now(),
                )
            )
    return [ctx.gens.get(gen_key(**p.identity)) for p in planned]


def round0_gens(ctx: RunContext, problem_id: str, replicate: int) -> list[Generation]:
    """The shared round-0 generations of a replicate, in agent order."""
    out = []
    for agent in range(N_AGENTS):
        key = gen_key(
            **identity_for(ctx.cfg, problem_id, replicate, agent, "round0", 0, 0)
        )
        if key not in ctx.gens:
            raise MissingRound0Error(
                f"round 0 missing for {problem_id} replicate {replicate} agent {agent}"
            )
        out.append(ctx.gens.get(key))
    return out


def run_round0(ctx: RunContext) -> list[Generation]:
    planned = []
    for p in ctx.problems:
        messages = channels.build_round0_prompt(p.question)
        for rep in range(ctx.cfg.replicates):
            for agent in range(N_AGENTS):
                ident = identity_for(ctx.cfg, p.problem_id, rep, agent, "round0", 0, 0)
                planned.append(Planned(ident, messages, []))
    return _execute(ctx, planned)


def run_sc_pool(ctx: RunContext) -> list[Generation]:
    planned = []
    for p in ctx.problems:
        messages = channels.build_round0_prompt(p.question)
        for rep in range(ctx.cfg.replicates):
            for i in range(ctx.cfg.n_pool):
                ident = identity_for(
                    ctx.cfg, p.problem_id, rep, POOL_AGENT, "sc_pool", 0, i
                )
                planned.append(Planned(ident, messages, []))
    return _execute(ctx, planned)


def run_failure_messages(ctx: RunContext) -> None:
    """Agent j's solution is reviewed in one sentence; the reader is agent j.

    Each message is sanitised and checked against all three round-0 answers.
    A surviving claim triggers one regeneration, then redaction. The outcome of
    every message is stored as a ``LeakageRecord``.
    """
    items = []
    for p in ctx.problems:
        for rep in range(ctx.cfg.replicates):
            for j in range(N_AGENTS):
                if record_key(p.problem_id, rep, j) not in ctx.leakage:
                    items.append((p, rep, j))
    if not items:
        return
    r0_cache: dict[tuple[str, int], list[Generation]] = {}

    def r0_of(p: Problem, rep: int) -> list[Generation]:
        if (p.problem_id, rep) not in r0_cache:
            r0_cache[(p.problem_id, rep)] = round0_gens(ctx, p.problem_id, rep)
        return r0_cache[(p.problem_id, rep)]

    def plan(item, attempt: int) -> Planned:
        p, rep, j = item
        reviewed = r0_of(p, rep)[j]
        ident = identity_for(
            ctx.cfg, p.problem_id, rep, j, "failure_msg", REVISION_ROUND, attempt
        )
        messages = channels.build_failure_msg_prompt(p.question, reviewed.text)
        return Planned(ident, messages, [reviewed.gen_key])

    def check(item, gen: Generation):
        p, rep, _ = item
        answers = [agent_view(g).answer for g in r0_of(p, rep)]
        sanitized, n_stripped = sanitize_message(gen.text)
        return answers, sanitized, n_stripped, find_leaks(sanitized, answers)

    def store(item, attempts, outcome, gen, n_stripped, leaked, text) -> None:
        p, rep, j = item
        ctx.leakage.append(
            LeakageRecord(
                record_key=record_key(p.problem_id, rep, j),
                problem_id=p.problem_id,
                replicate=rep,
                agent=j,
                attempts=attempts,
                outcome=outcome,
                final_gen_key=gen.gen_key,
                n_stripped=n_stripped,
                leaked=leaked,
                sanitized_text=text,
            )
        )

    retry = []
    for item, gen in zip(
        items, _execute(ctx, [plan(i, 0) for i in items]), strict=True
    ):
        _, sanitized, n_stripped, leaks = check(item, gen)
        if leaks:
            retry.append(item)
        else:
            store(item, 1, "clean", gen, n_stripped, [], sanitized)
    for item, gen in zip(
        retry, _execute(ctx, [plan(i, 1) for i in retry]), strict=True
    ):
        answers, sanitized, n_stripped, leaks = check(item, gen)
        if leaks:
            store(
                item, 2, "redacted", gen, n_stripped, leaks, redact(sanitized, answers)
            )
        else:
            store(item, 2, "regenerated_clean", gen, n_stripped, [], sanitized)


def run_arm(ctx: RunContext, arm: str) -> list[Generation]:
    """Branch one arm from the shared round 0: one revision per agent."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    planned = []
    for rep in range(ctx.cfg.replicates):
        r0_by_problem = {
            p.problem_id: round0_gens(ctx, p.problem_id, rep) for p in ctx.problems
        }
        for p in ctx.problems:
            r0 = r0_by_problem[p.problem_id]
            views = [agent_view(g) for g in r0]
            for i in range(N_AGENTS):
                peer_idx = [j for j in range(N_AGENTS) if j != i]
                peers = [views[j] for j in peer_idx]
                extra_parents: list[str] = []
                kwargs: dict = {}
                if arm == "failure_cause":
                    key = record_key(p.problem_id, rep, i)
                    if key not in ctx.leakage:
                        raise MissingFailureMessageError(key)
                    record = ctx.leakage.get(key)
                    kwargs["failure_message"] = record.sanitized_text
                    extra_parents = [record.final_gen_key]
                elif arm == "placebo":
                    donor, donor_r0 = _placebo_donor(
                        ctx, p, rep, i, peer_idx, r0, r0_by_problem
                    )
                    kwargs["donor_texts"] = [donor_r0[j].text for j in peer_idx]
                    extra_parents = [donor_r0[j].gen_key for j in peer_idx]
                messages = channels.build_revision_prompt(
                    arm, p.question, views[i], peers, **kwargs
                )
                if messages is None:
                    continue
                ident = identity_for(
                    ctx.cfg, p.problem_id, rep, i, arm, REVISION_ROUND, 0
                )
                parents = [g.gen_key for g in r0] + extra_parents
                planned.append(Planned(ident, messages, parents))
    return _execute(ctx, planned)


def _placebo_donor(ctx, problem, rep, agent, peer_idx, r0, r0_by_problem):
    tokens = {
        pid: sum(gens[j].completion_tokens for j in peer_idx)
        for pid, gens in r0_by_problem.items()
    }
    donor = pick_donor(
        problem,
        ctx.problems,
        sum(r0[j].completion_tokens for j in peer_idx),
        tokens,
        tiebreak=f"{problem.problem_id}|{rep}|{agent}",
    )
    return donor, r0_by_problem[donor.problem_id]


def run_all(ctx: RunContext, arms: Sequence[str] = ARMS) -> None:
    run_round0(ctx)
    run_sc_pool(ctx)
    if "failure_cause" in arms:
        run_failure_messages(ctx)
    for arm in arms:
        run_arm(ctx, arm)


def final_generations(
    ctx: RunContext, arm: str, problem_id: str, replicate: int
) -> list[Generation]:
    """Each agent's final generation under ``arm``, in agent order.

    A ``dissent_only`` agent that saw no disagreement was never asked to
    revise, so its round-0 generation stands. Any other missing revision is an
    error.
    """
    r0 = round0_gens(ctx, problem_id, replicate)
    views: list[AgentView] = [agent_view(g) for g in r0]
    out = []
    for i in range(N_AGENTS):
        key = gen_key(
            **identity_for(ctx.cfg, problem_id, replicate, i, arm, REVISION_ROUND, 0)
        )
        if key in ctx.gens:
            out.append(ctx.gens.get(key))
            continue
        peers = [v for j, v in enumerate(views) if j != i]
        if arm == "dissent_only" and not disagrees(views[i], peers):
            out.append(r0[i])
        else:
            raise LookupError(f"no {arm} revision for {problem_id}/{replicate}/{i}")
    return out


def grade_pending(ctx: RunContext) -> int:
    """Grade every solution-bearing generation not yet graded; return the count."""
    gold = {p.problem_id: p.gold for p in ctx.problems}
    n = 0
    for g in list(ctx.gens):
        if g.arm == "failure_msg" or g.gen_key in ctx.graded:
            continue
        ctx.graded.append(grade_generation(g, gold[g.problem_id]))
        n += 1
    return n
