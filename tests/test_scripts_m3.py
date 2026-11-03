"""M3 scripts that need no GPU: problems, repro check, review sample, freeze."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script, *args, check=True):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args], capture_output=True, text=True
    )
    if check:
        assert proc.returncode == 0, f"{script}\n{proc.stdout}\n{proc.stderr}"
    return proc


def write_rows(path, n_per_group=4):
    rows = []
    for s in ("Algebra", "Geometry"):
        for lv in (3, 4, 5):
            for i in range(n_per_group):
                rows.append(
                    {
                        "problem": f"{s} L{lv} #{i}: what is {i} + {lv}?",
                        "level": f"Level {lv}",
                        "type": s,
                        "solution": rf"It is $\boxed{{{i + lv}}}$.",
                    }
                )
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_make_check_problems_from_local_rows(tmp_path):
    rows, out = tmp_path / "rows.jsonl", tmp_path / "problems.jsonl"
    write_rows(rows)
    proc = run(
        "03_make_check_problems.py",
        "--from-jsonl",
        str(rows),
        "--n",
        "16",
        "--out",
        str(out),
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 16 and "16" in proc.stdout
    assert all(json.loads(line)["split"] == "check" for line in lines)


def test_round0_script_accepts_a_problems_file_and_logs_prompts(tmp_path):
    rows, problems = tmp_path / "rows.jsonl", tmp_path / "problems.jsonl"
    write_rows(rows)
    run(
        "03_make_check_problems.py",
        "--from-jsonl",
        str(rows),
        "--n",
        "16",
        "--out",
        str(problems),
    )
    out = tmp_path / "run"
    run(
        "10_round0.py",
        "--out",
        str(out),
        "--problems",
        str(problems),
        "--replicates",
        "1",
        "--log-prompts",
    )
    assert (out / "prompts.jsonl").exists()
    assert (out / "throughput.jsonl").exists()
    assert len((out / "generations.jsonl").read_text().splitlines()) == 16 * 3


def test_repro_check_and_review_sample_on_two_mock_runs(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for out in (a, b):
        run(
            "10_round0.py",
            "--out",
            out,
            "--n-fake-problems",
            "8",
            "--replicates",
            "1",
            "--log-prompts",
        )
        run("11_sc_pool.py", "--out", out)
        run("12_failure_messages.py", "--out", out)
        run("13_branch_arms.py", "--out", out)
        run("20_grade.py", "--out", out)
    proc = run("03_repro_check.py", "--a", a, "--b", b)
    assert "100.0%" in proc.stdout
    review = tmp_path / "review.md"
    run("03_review_sample.py", "--out", a, "--n", "30", "--file", str(review))
    assert review.read_text(encoding="utf-8").count("## ") >= 30


def test_repro_check_exits_nonzero_when_runs_differ(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    run("10_round0.py", "--out", a, "--n-fake-problems", "4", "--replicates", "1")
    run("10_round0.py", "--out", b, "--n-fake-problems", "6", "--replicates", "1")
    proc = run("03_repro_check.py", "--a", a, "--b", b, check=False)
    assert proc.returncode == 1


def test_freeze_script_writes_a_manifest_and_refuses_to_overwrite(tmp_path):
    dst = tmp_path / "v1"
    shutil.copytree(ROOT / "prompts" / "v1", dst)
    (dst / "MANIFEST.json").unlink(missing_ok=True)
    proc = run("03_freeze_prompts.py", "--dir", str(dst))
    assert (dst / "MANIFEST.json").exists() and "combined" in proc.stdout
    again = run("03_freeze_prompts.py", "--dir", str(dst), check=False)
    assert again.returncode != 0
