"""M0 scaffold checks: the package installs, CI is wired, AGENTS.md is intact."""

import re
import tomllib
from pathlib import Path

import channel_audit

ROOT = Path(__file__).resolve().parents[1]


def test_package_importable_with_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", channel_audit.__version__)


def test_pyproject_pins_python_311_and_dev_tools():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert cfg["project"]["name"] == "mas-channel-audit"
    assert cfg["project"]["requires-python"].startswith(">=3.11")
    dev = " ".join(cfg["project"]["optional-dependencies"]["dev"])
    assert "pytest" in dev and "ruff" in dev


def test_agents_md_has_all_ten_rules():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    numbers = [int(n) for n in re.findall(r"^(\d+)\. ", text, flags=re.MULTILINE)]
    assert numbers == list(range(1, 11))


def test_ci_runs_ruff_and_pytest_on_python_311():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "3.11" in ci
    assert "ruff check" in ci
    assert "pytest" in ci


def test_gitignore_keeps_raw_generations_out_of_git():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "raw/" in ignored
    assert ".venv/" in ignored
