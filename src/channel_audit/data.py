"""The study problems: MATH test set at a pinned revision, split by a fixed hash.

Plan (M4): 300 screen, 200 pilot, the rest confirm. All usable test problems
are ordered by ``sha256(SPLIT_SALT | problem_id)``. The first 300 are the
screen split (every level, so the sanity check can be compared with a published
all-level accuracy). Among the others, level 3 to 5 problems go to the pilot
(first 200) and then to confirm; level 1 and 2 leftovers are not used. Only a
problem's id and level are read; no model output ever decides membership.

The split is frozen as files plus a manifest holding the dataset revision, the
salt, the counts and the sha256 of every file, so any later change is caught.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from channel_audit.check_problems import CONFIGS, parse_level
from channel_audit.extract import extract_boxed
from channel_audit.schema import Problem

MATH_DATASET = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"  # pinned commit
SPLIT_SALT = "mas-channel-audit/split/v1"
N_SCREEN = 300
N_PILOT = 200
STUDY_LEVELS = (3, 4, 5)
MANIFEST_NAME = "split_manifest.json"
SPLITS = ("screen", "pilot", "confirm")

RowLoader = Callable[[str, str], Iterable[dict]]


@dataclass(frozen=True)
class SplitResult:
    screen: list[Problem]
    pilot: list[Problem]
    confirm: list[Problem]
    excluded_ids: list[str]  # level 1-2 problems outside the screen split


def split_key(problem_id: str) -> str:
    """The stable hash that orders problems for the split."""
    return hashlib.sha256(f"{SPLIT_SALT}|{problem_id}".encode()).hexdigest()


def _hf_rows(config: str, revision: str) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "loading MATH needs datasets: pip install 'mas-channel-audit[data]'"
        ) from exc
    return load_dataset(MATH_DATASET, config, split="test", revision=revision)


def load_math_test(
    revision: str = MATH_REVISION, loader: RowLoader | None = None
) -> tuple[list[Problem], list[str]]:
    """Usable MATH test problems and the ids of the rows that were dropped.

    A row is dropped when its level or its boxed gold answer cannot be parsed.
    The returned problems have ``split="confirm"`` only as a placeholder; call
    ``assign_splits`` to set the real split.
    """
    rows_for = loader or _hf_rows
    problems, dropped = [], []
    for config in CONFIGS:
        for i, row in enumerate(rows_for(config, revision)):
            problem_id = f"math-test-{config}-{i}"
            level = parse_level(row["level"])
            gold = extract_boxed(row["solution"])
            if level is None or gold is None:
                dropped.append(problem_id)
                continue
            problems.append(
                Problem(
                    problem_id=problem_id,
                    dataset="math-test",
                    subject=row["type"],
                    level=level,
                    question=row["problem"],
                    gold=gold,
                    split="confirm",
                )
            )
    return problems, dropped


def assign_splits(problems: Iterable[Problem]) -> SplitResult:
    ordered = sorted(problems, key=lambda p: split_key(p.problem_id))
    if len(ordered) < N_SCREEN:
        raise ValueError(
            f"need at least {N_SCREEN} problems for the screen split, got "
            f"{len(ordered)}"
        )
    screen, rest = ordered[:N_SCREEN], ordered[N_SCREEN:]
    study = [p for p in rest if p.level in STUDY_LEVELS]
    if len(study) < N_PILOT:
        raise ValueError(
            f"need at least {N_PILOT} level 3-5 problems for the pilot split "
            f"after the screen split, got {len(study)}"
        )
    return SplitResult(
        screen=[p.model_copy(update={"split": "screen"}) for p in screen],
        pilot=[p.model_copy(update={"split": "pilot"}) for p in study[:N_PILOT]],
        confirm=[p.model_copy(update={"split": "confirm"}) for p in study[N_PILOT:]],
        excluded_ids=[p.problem_id for p in rest if p.level not in STUDY_LEVELS],
    )


def _file_bytes(problems: list[Problem]) -> bytes:
    return "".join(p.model_dump_json() + "\n" for p in problems).encode("utf-8")


def write_split(
    split: SplitResult,
    out_dir: str | Path,
    *,
    n_rows: int,
    dropped: Iterable[str] = (),
) -> dict:
    """Write the split files and manifest; refuse to change an existing freeze."""
    out = Path(out_dir)
    dropped = list(dropped)
    blobs = {name: _file_bytes(getattr(split, name)) for name in SPLITS}
    manifest = {
        "dataset": MATH_DATASET,
        "revision": MATH_REVISION,
        "salt": SPLIT_SALT,
        "n_rows": n_rows,
        "counts": {
            **{name: len(getattr(split, name)) for name in SPLITS},
            "excluded": len(split.excluded_ids),
            "dropped": len(dropped),
        },
        "files": {
            name: {
                "sha256": hashlib.sha256(blob).hexdigest(),
                "n": len(getattr(split, name)),
            }
            for name, blob in blobs.items()
        },
        "dropped_ids": dropped,
        "excluded_ids": split.excluded_ids,
    }
    manifest_path = out / MANIFEST_NAME
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old != manifest:
            raise ValueError(
                f"{manifest_path} is already frozen with different content; "
                "choose a new directory instead of changing a frozen split"
            )
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in blobs.items():
        path = out / f"{name}.jsonl"
        if not path.exists():  # only the manifest is committed; files are rebuilt
            path.write_bytes(blob)
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    return manifest


def load_frozen(directory: str | Path, name: str) -> list[Problem]:
    """One frozen split, checked against the sha256 recorded in the manifest."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    base = Path(directory)
    manifest = json.loads((base / MANIFEST_NAME).read_text(encoding="utf-8"))
    blob = (base / f"{name}.jsonl").read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest["files"][name]["sha256"]:
        raise ValueError(f"{name}.jsonl does not match the manifest sha256")
    return [
        Problem.model_validate_json(line)
        for line in blob.decode("utf-8").splitlines()
        if line.strip()
    ]
