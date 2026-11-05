"""Gate G4: read a random sample of failure messages and label each one.

The sample is a seeded uniform draw over every message a reader was shown. It
uses no arm outcome and does not grade anything, so it cannot favour examples.
Labels are a human judgement:

* ``useful``: names a specific step or claim of the reviewed solution, with a
  concrete reason, and the concern is real.
* ``vague``: generic advice ("check the arithmetic carefully") that points at no
  specific step or gives no reason.
* ``wrong``: names a specific step but the concern is mistaken (the step is fine
  or the reason is false).

G4 passes when at least half of the sampled messages are not vague. A wrong
message is specific, so it counts as not vague; how often messages are wrong is
reported separately.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from channel_audit.runner import RunContext

N_SAMPLE = 40
G4_MIN_NOT_VAGUE = 0.5
LABELS = ("useful", "vague", "wrong")
SOLUTION_CHARS = 2500  # how much of the reviewed solution the sheet shows


def sample_failure_messages(
    ctxs: Sequence[RunContext], n: int = N_SAMPLE, seed: int = 0
) -> list[dict]:
    records = []
    for ctx in ctxs:
        problems = {p.problem_id: p for p in ctx.problems}
        for rec in ctx.leakage:
            msg_gen = ctx.gens.get(rec.final_gen_key)
            reviewed = ctx.gens.get(msg_gen.parent_gen_keys[0])
            problem = problems[rec.problem_id]
            records.append(
                {
                    "record_key": rec.record_key,
                    "problem_id": rec.problem_id,
                    "replicate": rec.replicate,
                    "agent": rec.agent,
                    "question": problem.question,
                    "gold": problem.gold,
                    "reviewed_solution": reviewed.text,
                    "message": rec.sanitized_text,
                    "outcome": rec.outcome,
                }
            )
    records.sort(key=lambda r: r["record_key"])
    rng = random.Random(seed)
    return rng.sample(records, min(n, len(records)))


def format_label_sheet(items: Sequence[dict]) -> str:
    lines = [
        "# Failure messages to label",
        "",
        "Label each message `useful`, `vague` or `wrong` in the labels file.",
        "",
        "- `useful`: names a specific step or claim, with a concrete reason, and the "
        "concern is real.",
        "- `vague`: generic advice, no specific step or no reason.",
        "- `wrong`: names a specific step but the concern is mistaken.",
        "",
        "The message is exactly what the reader was shown. The reviewed solution and "
        "the gold answer are given only to help you judge it.",
        "",
    ]
    for i, item in enumerate(items, 1):
        solution = item["reviewed_solution"]
        if len(solution) > SOLUTION_CHARS:
            solution = "[...]\n" + solution[-SOLUTION_CHARS:]
        lines += [
            f"## {i}. {item['record_key']}",
            "",
            f"**Problem** (gold answer: {item['gold']})",
            "",
            item["question"],
            "",
            "**Reviewed solution**",
            "",
            "~~~~",
            solution,
            "~~~~",
            "",
            "**Message shown to the reader**",
            "",
            "~~~~",
            item["message"],
            "~~~~",
            "",
        ]
    return "\n".join(lines)


def write_label_template(items: Sequence[dict], path: str | Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; labels are never overwritten")
    path.write_text(
        json.dumps({i["record_key"]: "" for i in items}, indent=1) + "\n",
        encoding="utf-8",
    )


def load_labels(path: str | Path, items: Sequence[dict]) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = [i["record_key"] for i in items]
    missing = [k for k in keys if data.get(k, "") == ""]
    if missing:
        raise ValueError(f"{len(missing)} unlabelled messages, e.g. {missing[0]}")
    bad = {k: v for k, v in data.items() if k in keys and v not in LABELS}
    if bad:
        key, value = next(iter(bad.items()))
        raise ValueError(f"label {value!r} for {key} is not one of {LABELS}")
    return {k: data[k] for k in keys}


def g4_check(labels: Mapping[str, str]) -> dict:
    if not labels:
        raise ValueError("no labels given")
    counts = Counter(labels.values())
    n = len(labels)
    not_vague = n - counts["vague"]
    return {
        "n": n,
        "useful": counts["useful"],
        "vague": counts["vague"],
        "wrong": counts["wrong"],
        "not_vague": not_vague,
        "share_not_vague": not_vague / n,
        "threshold": G4_MIN_NOT_VAGUE,
        "passed": not_vague / n >= G4_MIN_NOT_VAGUE,
    }
