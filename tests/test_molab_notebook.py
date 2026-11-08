"""The marimo notebook for molab: well formed, thin, and safe with the token."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "confirm_molab.py"


@pytest.fixture(scope="module")
def source():
    return NOTEBOOK.read_text(encoding="utf-8")


def test_it_is_valid_python_and_a_marimo_app(source):
    tree = ast.parse(source)
    assert "app = marimo.App(" in source
    cells = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and any("app.cell" in ast.unparse(d) for d in n.decorator_list)
    ]
    assert len(cells) >= 6
    assert 'if __name__ == "__main__":' in source and "app.run()" in source


def test_it_ships_set_to_the_cheap_hardware_check_not_the_full_run(source):
    assert re.search(r'^\s*MODE = "pilot_check"', source, re.M)


def test_nothing_starts_without_a_button_press(source):
    assert source.count("mo.ui.run_button(") >= 2
    assert source.count("mo.stop(") >= 2


def test_no_secret_is_embedded_or_printed(source):
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source)
    assert "print(token" not in source and "print(HF_TOKEN" not in source
    assert 'kind="password"' in source


def test_the_notebook_only_launches_it_never_computes_an_outcome(source):
    for forbidden in (
        "pilot_report",
        "collect_units",
        "plurality",
        "is_correct",
        "sc_credit",
        ".correct",
        "accuracy",
    ):
        assert forbidden not in source, forbidden
    assert "launcher" in source


def test_the_notebook_runs_the_repository_at_a_chosen_ref(source):
    assert "REPO_URL" in source and "REF" in source
    assert "git" in source and "setup_commands" in source


def test_the_job_gets_its_own_environment_the_split_and_the_sampler_setting(source):
    assert "venv_python" in source and "freeze_command" in source
    assert "job.env()" in source and "python=PYTHON" in source


def test_marimo_accepts_the_file(source):
    pytest.importorskip("marimo")
    proc = subprocess.run(
        [sys.executable, "-m", "marimo", "check", str(NOTEBOOK)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
