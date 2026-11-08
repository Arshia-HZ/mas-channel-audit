"""Sharded runs: 50-problem shards, resumable, validated, uploaded on completion.

A shard is a run directory over a slice of the problems. Finishing a shard means
every arm was generated, graded and validated without errors; that is recorded in
``SHARD_DONE.json``, and the folder is then pushed to the raw-data dataset. A
Colab or Kaggle session that ends loses at most the shard in progress: finished
shards are skipped on the next start (and never load the model), an interrupted
shard resumes from its append-only store, and a shard that finished but was not
uploaded is only uploaded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from channel_audit.backends.base import Backend, GenRequest, GenResult
from channel_audit.channels import ARMS, template_hash
from channel_audit.runner import (
    RunConfig,
    _now,
    create_run,
    grade_pending,
    open_run,
    run_all,
    run_arm,
    run_failure_messages,
)
from channel_audit.schema import Problem
from channel_audit.validate import validate_run

SHARD_SIZE = 50
DONE_FILE = "SHARD_DONE.json"
UPLOADED_FILE = "UPLOADED.json"

Uploader = Callable[[Path, str], None]
Restorer = Callable[[str, Path], bool]


class ShardInvalidError(RuntimeError):
    """A finished shard failed validation; it is neither marked done nor uploaded."""


@dataclass(frozen=True)
class ShardResult:
    name: str
    status: str  # generated | skipped | uploaded_only
    validation_ok: bool
    g3_ok: bool
    uploaded: bool
    n_generations: int


class LazyBackend:
    """Builds the real backend (and loads the model) on first use, once."""

    def __init__(self, factory: Callable[[], Backend]) -> None:
        self._factory = factory
        self._backend: Backend | None = None

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = self._factory()
        return self._backend

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def version(self) -> str:
        return self.backend.version

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResult]:
        return self.backend.generate(requests)


def shard_name(index: int) -> str:
    return f"shard_{index:02d}"


def make_shards(
    problems: Sequence[Problem], size: int = SHARD_SIZE, *, keep_groups: bool = True
) -> list[list[Problem]]:
    """Split the problems into shards of about ``size``.

    With ``keep_groups`` (the default) every (subject, level) group stays whole
    inside one shard, so the placebo arm always finds its donors among the shard's
    own problems and the donor pool does not depend on the shard layout. Groups
    are balanced over ``ceil(n / size)`` shards (largest group first), a function
    of subject and level only. Each shard keeps the input order. A group of one
    problem has no placebo donor and is an error. Without ``keep_groups`` the
    input is simply cut into consecutive slices.
    """
    if size < 1:
        raise ValueError("shard size must be at least 1")
    items = list(problems)
    if not keep_groups:
        return [items[i : i + size] for i in range(0, len(items), size)]
    groups: dict[tuple[str, int], list[int]] = {}
    for i, p in enumerate(items):
        groups.setdefault((p.subject, p.level), []).append(i)
    lonely = [key for key, members in groups.items() if len(members) < 2]
    if lonely:
        raise ValueError(f"placebo needs two or more problems per group: {lonely}")
    n_shards = max(1, -(-len(items) // size))
    loads: list[list[int]] = [[] for _ in range(n_shards)]
    for key in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        smallest = min(range(n_shards), key=lambda j: (len(loads[j]), j))
        loads[smallest].extend(groups[key])
    return [[items[i] for i in sorted(idx)] for idx in loads if idx]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_same_config(directory: Path, cfg: RunConfig) -> None:
    path = directory / "run.json"
    if path.exists() and RunConfig.model_validate_json(path.read_text("utf-8")) != cfg:
        raise ValueError(f"{path} holds a different config; use a new directory")


def _upload(directory: Path, remote: str, uploader: Uploader) -> None:
    uploader(directory, remote)
    (directory / UPLOADED_FILE).write_text(
        json.dumps({"remote_path": remote, "uploaded_at": _now()}, indent=1) + "\n",
        encoding="utf-8",
    )


def run_shards(
    root: str | Path,
    cfg: RunConfig,
    problems: Sequence[Problem],
    backend: Backend | None,
    *,
    size: int = SHARD_SIZE,
    keep_groups: bool = True,
    arms: Sequence[str] = ARMS,
    uploader: Uploader | None = None,
    restorer: Restorer | None = None,
    remote_prefix: str = "",
    max_new_shards: int | None = None,
    log: Callable[[str], None] = print,
) -> list[ShardResult]:
    """Run, validate and upload every shard, skipping what is already finished.

    ``max_new_shards`` stops after that many shards were generated in this call
    (finished shards do not count), so one session can be bounded.
    """
    root = Path(root)
    results = []
    generated = 0
    for index, shard in enumerate(make_shards(problems, size, keep_groups=keep_groups)):
        name = shard_name(index)
        directory = root / name
        remote = f"{remote_prefix}/{name}" if remote_prefix else name
        done_path = directory / DONE_FILE
        if not done_path.exists() and restorer is not None:
            restorer(remote, directory)
        _check_same_config(directory, cfg)

        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            status = "skipped"
            if uploader is not None and not (directory / UPLOADED_FILE).exists():
                _upload(directory, remote, uploader)
                status = "uploaded_only"
            log(f"{name}: {status}")
            results.append(
                ShardResult(
                    name,
                    status,
                    True,
                    done["g3_ok"],
                    (directory / UPLOADED_FILE).exists(),
                    done["n_generations"],
                )
            )
            continue

        if max_new_shards is not None and generated >= max_new_shards:
            log(f"{name}: left for the next session")
            continue
        log(f"{name}: {len(shard)} problems")
        ctx = create_run(directory, cfg, shard, backend)
        run_all(ctx, arms=arms)
        grade_pending(ctx)
        report = validate_run(ctx)
        if report.errors:
            raise ShardInvalidError(f"{name}: " + "; ".join(report.errors[:5]))
        done = {
            "shard": name,
            "n_problems": len(shard),
            "n_generations": len(ctx.gens),
            "generations_sha256": _sha256(directory / "generations.jsonl"),
            "arms": list(arms),
            "g3_ok": report.g3_ok,
            "g3_failing_arms": report.g3_failing_arms,
            "warnings": report.warnings,
            "counts": report.counts,
            "finished_at": _now(),
        }
        done_path.write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
        if uploader is not None:
            _upload(directory, remote, uploader)
        generated += 1
        log(f"{name}: done, G3 {'pass' if report.g3_ok else 'FAIL'}")
        results.append(
            ShardResult(
                name,
                "generated",
                True,
                report.g3_ok,
                uploader is not None,
                len(ctx.gens),
            )
        )
    return results


def run_variant_shards(
    source_root: str | Path,
    dest_root: str | Path,
    cfg: RunConfig,
    problems: Sequence[Problem],
    backend: Backend | None,
    *,
    variant: str = "v2",
    size: int = SHARD_SIZE,
    keep_groups: bool = True,
    uploader: Uploader | None = None,
    restorer: Restorer | None = None,
    source_prefix: str = "",
    remote_prefix: str = "",
    max_new_shards: int | None = None,
    log: Callable[[str], None] = print,
) -> list[ShardResult]:
    """Add a failure-message variant to copies of finished shards.

    Each finished source shard is copied byte for byte into ``dest_root``; the
    variant's messages and revisions are then generated in the copy, which is
    validated and uploaded like any shard. The source shards never change, so
    both variants can be compared on the same round 0 and the same pool.
    """
    source_root, dest_root = Path(source_root), Path(dest_root)
    arm = f"failure_cause_{variant}"
    results = []
    generated = 0
    for index, _ in enumerate(make_shards(problems, size, keep_groups=keep_groups)):
        name = shard_name(index)
        src, dst = source_root / name, dest_root / name
        remote = f"{remote_prefix}/{name}" if remote_prefix else name
        done_path = dst / DONE_FILE
        if done_path.exists():
            done = json.loads(done_path.read_text(encoding="utf-8"))
            status = "skipped"
            if uploader is not None and not (dst / UPLOADED_FILE).exists():
                _upload(dst, remote, uploader)
                status = "uploaded_only"
            log(f"{name}: {status}")
            results.append(
                ShardResult(
                    name, status, True, done["g3_ok"],
                    (dst / UPLOADED_FILE).exists(), done["n_generations"],
                )
            )  # fmt: skip
            continue

        if not (src / DONE_FILE).exists() and restorer is not None:
            source_remote = f"{source_prefix}/{name}" if source_prefix else name
            restorer(source_remote, src)
        if not (src / DONE_FILE).exists():
            raise FileNotFoundError(
                f"{name}: no finished source shard in {source_root}"
            )
        if max_new_shards is not None and generated >= max_new_shards:
            log(f"{name}: left for the next session")
            continue

        if not (dst / "run.json").exists():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(DONE_FILE, UPLOADED_FILE),
                dirs_exist_ok=True,
            )
        _check_same_config(dst, cfg)
        log(f"{name}: failure messages {variant}")
        ctx = open_run(dst, backend)
        run_failure_messages(ctx, variant)
        run_arm(ctx, arm)
        grade_pending(ctx)
        report = validate_run(ctx)
        if report.errors:
            raise ShardInvalidError(f"{name}: " + "; ".join(report.errors[:5]))
        source_done = json.loads((src / DONE_FILE).read_text(encoding="utf-8"))
        done = {
            "shard": name,
            "variant": variant,
            "n_problems": len(ctx.problems),
            "n_generations": len(ctx.gens),
            "source_generations_sha256": source_done["generations_sha256"],
            "generations_sha256": _sha256(dst / "generations.jsonl"),
            f"failure_msg_{variant}_template_sha256": template_hash(
                f"failure_msg_{variant}"
            ),
            "g3_ok": report.g3_ok,
            "g3_failing_arms": report.g3_failing_arms,
            "warnings": report.warnings,
            "counts": report.counts,
            "finished_at": _now(),
        }
        done_path.write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
        if uploader is not None:
            _upload(dst, remote, uploader)
        generated += 1
        log(f"{name}: done, G3 {'pass' if report.g3_ok else 'FAIL'}")
        results.append(
            ShardResult(
                name, "generated", True, report.g3_ok, uploader is not None,
                len(ctx.gens),
            )
        )  # fmt: skip
    return results


# --- Hugging Face adapters; the library is imported only when they are used ----


def hf_uploader(repo_id: str, token: str | None = None) -> Uploader:
    def upload(directory: Path, remote_path: str) -> None:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_folder(
            folder_path=str(directory),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
        )

    return upload


def hf_downloader(repo_id: str, token: str | None = None) -> Restorer:
    def restore(remote_path: str, directory: Path) -> bool:
        from huggingface_hub import snapshot_download

        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=f"{remote_path}/*",
                local_dir=tmp,
                token=token,
            )
            src = Path(tmp, *remote_path.split("/"))
            if not (src.is_dir() and any(src.iterdir())):
                return False
            shutil.copytree(src, directory, dirs_exist_ok=True)
        return True

    return restore
