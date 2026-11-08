import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Channel audit on molab: pilot check, then the confirmatory run

    This notebook only **starts and watches** `scripts/05_pilot_shards.py` as a background
    job. It computes nothing from the generations and shows no outcome, so the confirmatory
    rule (no outcome before the analysis code is frozen) holds. Raw shards go to the private
    Hugging Face dataset after each shard is validated.

    **Order:** run `MODE = "pilot_check"` first (one pilot shard, uploaded under its own
    prefix, never over the T4 pilot). Compare it with the T4 pilot. Only then switch to
    `MODE = "confirm"`.

    Attach the GPU from the notebook specs button before starting. A session ends after
    12 hours or 90 idle minutes; start the job again and it continues.
    """)
    return


@app.cell
def _():
    # "pilot_check" runs one pilot shard; "confirm" runs the 600 problems.
    MODE = "pilot_check"
    HF_REPO = "Arshia-HZ/mas-channel-audit-raw"
    REF = "main"  # branch, tag or commit of the repository to run
    REPO_URL = "https://github.com/Arshia-HZ/mas-channel-audit.git"
    return HF_REPO, MODE, REF, REPO_URL


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="1. Clone, install, show the GPU")
    token_input = mo.ui.text(
        kind="password",
        label="HF token with write access (not needed if the HF_TOKEN variable is set)",
    )
    mo.vstack([setup_button, token_input])
    return setup_button, token_input


@app.cell
def _(REF, REPO_URL, mo, setup_button):
    import os
    import subprocess
    import sys
    from pathlib import Path

    mo.stop(not setup_button.value, mo.md("Press **1** to clone and install."))

    REPO = Path.home() / "mas-channel-audit"
    RUN_ROOT = Path.home() / "mas-channel-audit-runs"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    def sh(*cmd, cwd=None):
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        print(proc.stdout, end="")
        print(proc.stderr, end="")
        if proc.returncode:
            raise RuntimeError(f"{' '.join(cmd)} exited with {proc.returncode}")

    if not (REPO / ".git").is_dir():
        sh("git", "clone", "-q", REPO_URL, str(REPO))
    sh("git", "fetch", "-q", "--all", cwd=REPO)
    sh("git", "checkout", "-q", REF, cwd=REPO)
    head = subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=REPO)
    if head.returncode == 0:
        sh("git", "pull", "-q", "--ff-only", cwd=REPO)  # on a branch: take the latest

    sys.path.insert(0, str(REPO / "src"))
    from channel_audit import launcher  # standard library only

    # The job gets its own environment: installing vLLM must not replace the
    # packages this notebook kernel runs on.
    VENV = Path.home() / "mca-venv"
    if not Path(launcher.venv_python(VENV)).exists():
        sh(*launcher.setup_commands(str(REPO), str(VENV))[0])
    sh(*launcher.setup_commands(str(REPO), str(VENV))[1])
    PYTHON = launcher.venv_python(VENV)
    # The split files are not in git: rebuild them and check them against the manifest.
    sh(*launcher.freeze_command(str(REPO), PYTHON))
    sh("git", "log", "--oneline", "-1", cwd=REPO)
    subprocess.run(["nvidia-smi"])
    mo.md(f"Repository ready at `{REPO}`. Runs go to `{RUN_ROOT}`.")
    return PYTHON, Path, REPO, RUN_ROOT, launcher, os


@app.cell
def _(mo):
    launch_button = mo.ui.run_button(label="2. Start the job")
    stop_button = mo.ui.run_button(label="Stop the job")
    mo.hstack([launch_button, stop_button], justify="start")
    return launch_button, stop_button


@app.cell
def _(
    HF_REPO,
    MODE,
    PYTHON,
    Path,
    REPO,
    RUN_ROOT,
    launch_button,
    launcher,
    mo,
    os,
    token_input,
):
    mo.stop(not launch_button.value, mo.md("Press **2** to start the job."))

    job = launcher.job_for_mode(MODE, str(REPO), str(RUN_ROOT), HF_REPO)
    token = os.environ.get("HF_TOKEN") or token_input.value
    if not token:
        raise RuntimeError("upload needs a Hugging Face token with write access")

    launcher.record_hardware(job.out, python=PYTHON)
    LOG = Path(job.out).parent / f"{MODE}.log"
    PID = Path(job.out).parent / f"{MODE}.pid"
    pid = launcher.start(
        job.command(python=PYTHON), LOG, PID, env={**job.env(), "HF_TOKEN": token}
    )
    mo.md(f"Started `{MODE}` as process {pid}. Log: `{LOG}`")
    return


@app.cell
def _(MODE, RUN_ROOT, launcher, mo, stop_button):
    mo.stop(not stop_button.value, None)
    launcher.stop(RUN_ROOT / f"{MODE}.pid")
    mo.md("Stop signal sent. Press **2** again later; finished shards are kept.")
    return


@app.cell
def _(mo):
    refresh = mo.ui.refresh(default_interval="30s", label="Refresh the log")
    refresh
    return (refresh,)


@app.cell
def _(MODE, RUN_ROOT, launcher, mo, refresh):
    refresh.value  # rerun on every refresh
    _pid = RUN_ROOT / f"{MODE}.pid"
    _log = RUN_ROOT / f"{MODE}.log"
    _state = "running" if launcher.is_running(_pid) else "not running"
    _done = len(list(RUN_ROOT.glob("*/shard_*/SHARD_DONE.json")))
    mo.vstack(
        [
            mo.md(f"**{_state}** | finished shards on this machine: {_done}"),
            mo.md(f"```\n{launcher.tail(_log, 40)}\n```"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
