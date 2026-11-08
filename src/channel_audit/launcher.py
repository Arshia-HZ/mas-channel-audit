"""Run the shard script as a background job from a notebook.

A hosted notebook can lose its browser tab or its kernel. The job is therefore a
separate process that logs to a file; the notebook only starts it, tails the log,
and refuses to start a second copy. Everything it runs is resumable, so a killed
job is simply started again.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "05_pilot_shards.py"
# The version every screening, pilot and confirmatory generation was produced with.
VLLM_VERSION = "0.33.0"
# vLLM's default FlashInfer sampler compiles kernels with nvcc and ninja, which a
# hosted sandbox lacks; the PyTorch sampler needs neither.
JOB_ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
PACKAGES = ("vllm", "torch", "transformers", "numpy", "math-verify", "pydantic")

_children: dict[int, subprocess.Popen] = {}


@dataclass(frozen=True)
class ShardJob:
    """The arguments of ``scripts/05_pilot_shards.py``; the token is never one."""

    split: str
    model_config: str
    problems: str
    out: str
    replicates: int = 2
    size: int = 50
    n_problems: int | None = None
    with_v2: bool = False
    upload_repo: str | None = None
    remote_prefix: str | None = None
    max_new_shards: int | None = None

    def env(self) -> dict[str, str]:
        """Environment variables the job needs; the token is added by the caller."""
        return dict(JOB_ENV)

    def command(self, python: str | None = None) -> list[str]:
        if self.split not in ("pilot", "confirm"):
            raise ValueError(f"split must be 'pilot' or 'confirm', not {self.split!r}")
        cmd = [
            python or sys.executable,
            str(SCRIPT),
            "--model-config",
            self.model_config,
            "--problems",
            self.problems,
            "--split",
            self.split,
            "--out",
            self.out,
            "--replicates",
            str(self.replicates),
            "--size",
            str(self.size),
        ]
        optional = {
            "--n-problems": self.n_problems,
            "--upload-repo": self.upload_repo,
            "--remote-prefix": self.remote_prefix,
            "--max-new-shards": self.max_new_shards,
        }
        for name, value in optional.items():
            if value is not None:
                cmd += [name, str(value)]
        if self.with_v2:
            cmd.append("--with-v2")
        return cmd


MODEL_CONFIG = "configs/models/qwen3-1.7b.yaml"
MODEL_SLUG = "qwen3-1.7b"


def job_for_mode(mode: str, repo: str, run_root: str, hf_repo: str) -> ShardJob:
    """The two things the hosted notebook runs.

    ``confirm`` is the preregistered confirmatory run. ``pilot_check`` is the first
    pilot shard on the new GPU, uploaded under its own prefix so the T4 pilot data
    is never overwritten; it is a hardware comparison on data that was already seen.
    """
    common = dict(
        model_config=f"{repo}/{MODEL_CONFIG}",
        problems=f"{repo}/data/problems",
        upload_repo=hf_repo,
    )
    if mode == "confirm":
        return ShardJob(
            split="confirm",
            out=f"{run_root}/confirm_{MODEL_SLUG}_vllm",
            remote_prefix=f"confirm/{MODEL_SLUG}",
            n_problems=600,
            with_v2=True,
            **common,
        )
    if mode == "pilot_check":
        return ShardJob(
            split="pilot",
            out=f"{run_root}/pilot_check_{MODEL_SLUG}_vllm",
            remote_prefix=f"pilot_check/{MODEL_SLUG}",
            max_new_shards=1,
            **common,
        )
    raise ValueError(f"mode must be 'confirm' or 'pilot_check', not {mode!r}")


def venv_python(venv: str | Path) -> str:
    sub = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return str(Path(venv) / sub)


def setup_commands(repo: str, venv: str) -> list[list[str]]:
    """Create an environment for the job and install the pinned stack into it.

    The job gets its own environment so that installing vLLM never replaces the
    packages the notebook kernel itself is running on.
    """
    py = venv_python(venv)
    pins = [f"vllm=={VLLM_VERSION}", "-e", f"{repo}[data]"]
    if shutil.which("uv"):
        return [
            ["uv", "venv", "--python", "3.12", venv],
            ["uv", "pip", "install", "--python", py, *pins],
        ]
    return [
        [sys.executable, "-m", "venv", venv],
        [py, "-m", "pip", "install", "-q", *pins],
    ]


def freeze_command(repo: str, python: str) -> list[str]:
    """Rebuild the problem split from the pinned MATH revision and check the manifest.

    The ``*.jsonl`` split files are not in git; this recreates them and refuses to
    continue if they differ from ``split_manifest.json``.
    """
    script = str(Path(repo) / "scripts" / "04_freeze_problems.py")
    return [python, script, "--out", f"{repo}/data/problems"]


def state_from_stat(stat: str) -> str | None:
    """The process state letter from a /proc/<pid>/stat line (Z is a zombie)."""
    _, close, tail = stat.rpartition(")")
    rest = tail.split()
    return rest[0] if close and rest else None


def _alive(pid: int) -> bool:
    child = _children.get(pid)
    if child is not None:
        return child.poll() is None
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # a finished job nobody waited for still answers kill(0)
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return True
    return state_from_stat(stat) not in ("Z", "X")


def _read_pid(pid_file: str | Path) -> int | None:
    try:
        return int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def is_running(pid_file: str | Path) -> bool:
    pid = _read_pid(pid_file)
    return pid is not None and _alive(pid)


def start(
    cmd: list[str],
    log_file: str | Path,
    pid_file: str | Path,
    env: dict[str, str] | None = None,
) -> int:
    """Start ``cmd`` detached from this process, logging to ``log_file``."""
    if is_running(pid_file):
        raise RuntimeError(f"a job is already running (pid file {pid_file})")
    log_file, pid_file = Path(log_file), Path(pid_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
    detach = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if os.name == "nt"
        else {"start_new_session": True}
    )
    with open(log_file, "ab") as log:
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=full_env, **detach
        )
    _children[proc.pid] = proc
    pid_file.write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def stop(pid_file: str | Path) -> None:
    pid = _read_pid(pid_file)
    if pid is None or not _alive(pid):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(os.getpgid(pid), signal.SIGTERM)


def tail(log_file: str | Path, lines: int = 40) -> str:
    try:
        with open(log_file, encoding="utf-8", errors="replace") as fh:
            return "\n".join(line.rstrip("\n") for line in deque(fh, maxlen=lines))
    except OSError:
        return ""


_VERSIONS = """
import json, platform
from importlib import metadata

def version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None

packages = {package: version(package) for package in %r}
print(json.dumps({"python": platform.python_version(), "packages": packages}))
"""


def record_hardware(run_dir: str | Path, python: str | None = None) -> dict:
    """Write HARDWARE.json (GPU, driver, package versions) beside the shards.

    Package versions are read from ``python``, the interpreter that runs the job.
    """
    gpu = None
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            gpu = proc.stdout.strip().splitlines()[0]
    out = subprocess.run(
        [python or sys.executable, "-c", _VERSIONS % (PACKAGES,)],
        capture_output=True,
        text=True,
        check=True,
    )
    info = {"gpu": gpu, **json.loads(out.stdout)}
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "HARDWARE.json").write_text(
        json.dumps(info, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return info
