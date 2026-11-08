"""Background launcher used by the molab notebook: no GPU, no network."""

from __future__ import annotations

import json
import sys
import time

import pytest

from channel_audit import launcher
from channel_audit.launcher import ShardJob


def wait_until(predicate, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def job(**kw):
    base = dict(
        split="confirm",
        model_config="configs/models/qwen3-1.7b.yaml",
        problems="data/problems",
        out="/runs/confirm",
    )
    return ShardJob(**{**base, **kw})


def flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_the_confirmatory_command_matches_the_preregistered_design():
    cmd = job(with_v2=True, upload_repo="u/raw", n_problems=600).command()
    assert cmd[0] == sys.executable and cmd[1].endswith("05_pilot_shards.py")
    assert flag(cmd, "--split") == "confirm" and flag(cmd, "--replicates") == "2"
    assert flag(cmd, "--n-problems") == "600" and flag(cmd, "--size") == "50"
    assert "--with-v2" in cmd and flag(cmd, "--upload-repo") == "u/raw"


def test_optional_flags_are_left_out_when_unset():
    cmd = job().command()
    for name in ("--with-v2", "--upload-repo", "--n-problems", "--max-new-shards"):
        assert name not in cmd


def test_a_session_limit_and_a_remote_prefix_are_passed_on():
    cmd = job(max_new_shards=1, remote_prefix="pilot_blackwell/q").command()
    assert flag(cmd, "--max-new-shards") == "1"
    assert flag(cmd, "--remote-prefix") == "pilot_blackwell/q"


def test_confirm_mode_is_the_preregistered_run():
    j = launcher.job_for_mode("confirm", "/repo", "/runs", "u/raw")
    assert (j.split, j.n_problems, j.replicates, j.size) == ("confirm", 600, 2, 50)
    assert j.with_v2 and j.max_new_shards is None
    assert j.remote_prefix == "confirm/qwen3-1.7b" and j.upload_repo == "u/raw"
    assert j.model_config == "/repo/configs/models/qwen3-1.7b.yaml"


def test_pilot_check_mode_is_one_pilot_shard_that_cannot_overwrite_the_t4_pilot():
    j = launcher.job_for_mode("pilot_check", "/repo", "/runs", "u/raw")
    assert (j.split, j.max_new_shards, j.with_v2) == ("pilot", 1, False)
    assert j.remote_prefix == "pilot_check/qwen3-1.7b"
    assert j.remote_prefix != "pilot/qwen3-1.7b"
    assert j.out == "/runs/pilot_check_qwen3-1.7b_vllm"


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        launcher.job_for_mode("everything", "/repo", "/runs", "u/raw")


def test_an_unknown_split_is_refused():
    with pytest.raises(ValueError, match="split"):
        job(split="test").command()


def test_the_install_command_prefers_uv_and_targets_this_interpreter(monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: "/bin/uv")
    cmd = launcher.install_command("/repo")
    assert cmd[:3] == ["uv", "pip", "install"] and sys.executable in cmd
    assert cmd[-1] == "/repo[vllm,data]"
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    cmd = launcher.install_command("/repo")
    assert cmd[:4] == [sys.executable, "-m", "pip", "install"]


def test_a_job_runs_in_the_background_and_logs(tmp_path):
    cmd = [sys.executable, "-c", "print('hello from the job')"]
    pid = launcher.start(cmd, tmp_path / "job.log", tmp_path / "job.pid")
    assert pid > 0
    assert wait_until(lambda: not launcher.is_running(tmp_path / "job.pid"))
    assert "hello from the job" in launcher.tail(tmp_path / "job.log")


def test_a_second_launch_is_refused_while_the_first_runs(tmp_path):
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    log, pidfile = tmp_path / "job.log", tmp_path / "job.pid"
    pid = launcher.start(sleeper, log, pidfile)
    try:
        assert launcher.is_running(pidfile)
        with pytest.raises(RuntimeError, match="already running"):
            launcher.start(sleeper, log, pidfile)
    finally:
        launcher.stop(pidfile)
    assert wait_until(lambda: not launcher.is_running(pidfile))
    assert pid > 0


def test_a_stale_pid_file_does_not_block_a_new_launch(tmp_path):
    pidfile = tmp_path / "job.pid"
    pidfile.write_text("999999999")
    assert not launcher.is_running(pidfile)
    launcher.start([sys.executable, "-c", "pass"], tmp_path / "job.log", pidfile)


def test_the_token_reaches_the_job_but_never_the_command_or_the_log(tmp_path):
    secret = "hf_" + "x" * 30
    code = "import os; print('token set:', bool(os.environ.get('HF_TOKEN')))"
    cmd = [sys.executable, "-c", code]
    launcher.start(
        cmd, tmp_path / "job.log", tmp_path / "job.pid", env={"HF_TOKEN": secret}
    )
    assert wait_until(lambda: not launcher.is_running(tmp_path / "job.pid"))
    text = launcher.tail(tmp_path / "job.log")
    assert "token set: True" in text and secret not in text
    assert secret not in "".join(job(upload_repo="u/raw").command())


def test_tail_returns_only_the_last_lines(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")
    lines = launcher.tail(log, lines=5).splitlines()
    assert lines == [f"line {i}" for i in range(95, 100)]
    assert launcher.tail(tmp_path / "missing.log") == ""


def test_the_hardware_record_is_written_next_to_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)  # no nvidia-smi
    info = launcher.record_hardware(tmp_path)
    saved = json.loads((tmp_path / "HARDWARE.json").read_text(encoding="utf-8"))
    assert saved == info
    assert info["gpu"] is None and "python" in info and "packages" in info
