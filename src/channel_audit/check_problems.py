"""Problems for the M3 backend check, drawn from the MATH *train* split.

Train problems never overlap the test-set screen/pilot/confirm splits (M4), so
reading their outputs by hand contaminates nothing. Selection uses only each
problem's subject and level, never any model output. This is not the data
loader for the study: M4 replaces it with the pinned test-set loader.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from channel_audit.extract import extract_boxed
from channel_audit.schema import Problem

DATASET = "EleutherAI/hendrycks_math"
CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)


def parse_level(text: str) -> int | None:
    m = re.fullmatch(r"Level ([1-5])", text.strip())
    return int(m.group(1)) if m else None


def row_to_problem(row: dict, problem_id: str) -> Problem | None:
    """A MATH row as a Problem, or None if its level or boxed answer is unusable."""
    level = parse_level(row["level"])
    gold = extract_boxed(row["solution"])
    if level is None or gold is None:
        return None
    return Problem(
        problem_id=problem_id,
        dataset="math-train",
        subject=row["type"],
        level=level,
        question=row["problem"],
        gold=gold,
        split="check",
    )


def rows_from_jsonl(path: str | Path) -> list[Problem]:
    """Problems from a local file of raw MATH rows (one JSON object per line)."""
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if line.strip():
            problem = row_to_problem(json.loads(line), f"math-train-file-{i}")
            if problem is not None:
                out.append(problem)
    return out


def load_math_train() -> list[Problem]:
    """Every usable problem of the MATH train split (needs ``datasets``)."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "loading MATH needs datasets: pip install 'mas-channel-audit[data]'"
        ) from exc
    out = []
    for config in CONFIGS:
        for i, row in enumerate(load_dataset(DATASET, config, split="train")):
            problem = row_to_problem(row, f"math-train-{config}-{i}")
            if problem is not None:
                out.append(problem)
    return out


def sample_check_problems(
    problems: Iterable[Problem],
    n: int,
    seed: int = 0,
    levels: Sequence[int] = (3, 4, 5),
    min_per_group: int = 2,
) -> list[Problem]:
    """``n`` problems of the given levels, at least ``min_per_group`` per
    (subject, level) group so every placebo has a donor.

    Groups smaller than ``min_per_group`` are dropped. The result does not
    depend on the order of ``problems``.
    """
    groups: dict[tuple[str, int], list[Problem]] = defaultdict(list)
    for p in problems:
        if p.level in levels:
            groups[(p.subject, p.level)].append(p)
    groups = {k: v for k, v in groups.items() if len(v) >= min_per_group}
    needed = min_per_group * len(groups)
    if n < needed:
        raise ValueError(
            f"n={n} is too small: {len(groups)} groups need at least {needed}"
        )
    available = sum(len(v) for v in groups.values())
    if n > available:
        raise ValueError(f"only {available} eligible problems for n={n}")
    rng = random.Random(seed)
    chosen: list[Problem] = []
    rest: list[Problem] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda p: p.problem_id)
        rng.shuffle(members)
        chosen += members[:min_per_group]
        rest += members[min_per_group:]
    rng.shuffle(rest)
    chosen += rest[: n - len(chosen)]
    return sorted(chosen, key=lambda p: p.problem_id)
