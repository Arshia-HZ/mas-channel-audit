"""The analysis code is frozen as a hash manifest before it touches real data.

Two groups: the analysis files are frozen for good (CI checks them), and the pipeline
files they rely on are pinned for the confirmatory run (the script checks them), so
that later work on the generation side does not break the permanent freeze.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from channel_audit import analysis_freeze as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def make_tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    for name, text in (("a.py", "x = 1\n"), ("b.py", "y = 2\n"), ("c.py", "z = 3\n")):
        (tmp_path / "pkg" / name).write_text(text, encoding="utf-8")
    return ("pkg/a.py", "pkg/b.py"), ("pkg/c.py",)


def test_the_manifest_has_two_groups_with_hashes_and_a_combined_hash(tmp_path):
    analysis, pipeline = make_tree(tmp_path)
    m = F.build_manifest(tmp_path, analysis, pipeline)
    assert set(m["analysis"]) == set(analysis) and set(m["pipeline"]) == set(pipeline)
    assert all(len(h) == 64 for g in ("analysis", "pipeline") for h in m[g].values())
    assert len(m["combined_sha256"]) == 64
    assert F.build_manifest(tmp_path, analysis, pipeline) == m


def test_the_combined_hash_changes_with_any_file_in_either_group(tmp_path):
    analysis, pipeline = make_tree(tmp_path)
    before = F.build_manifest(tmp_path, analysis, pipeline)["combined_sha256"]
    (tmp_path / "pkg" / "c.py").write_text("z = 4\n", encoding="utf-8")
    assert F.build_manifest(tmp_path, analysis, pipeline)["combined_sha256"] != before


def test_line_endings_do_not_change_the_hash(tmp_path):
    analysis, pipeline = make_tree(tmp_path)
    before = F.build_manifest(tmp_path, analysis, pipeline)
    (tmp_path / "pkg" / "a.py").write_bytes(b"x = 1\r\n")
    assert F.build_manifest(tmp_path, analysis, pipeline) == before


def test_writing_never_overwrites_a_manifest(tmp_path):
    analysis, pipeline = make_tree(tmp_path)
    path = tmp_path / "freeze.json"
    F.write_freeze(path, tmp_path, analysis, pipeline)
    assert json.loads(path.read_text(encoding="utf-8"))["analysis"]
    with pytest.raises(FileExistsError):
        F.write_freeze(path, tmp_path, analysis, pipeline)


def test_a_changed_file_in_either_group_fails_verification_and_names_it(tmp_path):
    analysis, pipeline = make_tree(tmp_path)
    path = tmp_path / "freeze.json"
    F.write_freeze(path, tmp_path, analysis, pipeline)
    assert F.verify_freeze(path, tmp_path)["analysis"]
    (tmp_path / "pkg" / "b.py").write_text("y = 3\n", encoding="utf-8")
    with pytest.raises(F.FrozenAnalysisError, match=r"pkg/b\.py: changed"):
        F.verify_freeze(path, tmp_path)
    (tmp_path / "pkg" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "pkg" / "c.py").write_text("z = 9\n", encoding="utf-8")
    with pytest.raises(F.FrozenAnalysisError, match=r"pkg/c\.py: changed"):
        F.verify_freeze(path, tmp_path)


def test_the_permanent_check_looks_only_at_the_analysis_group(tmp_path):
    analysis, pipeline = make_tree(tmp_path)
    path = tmp_path / "freeze.json"
    F.write_freeze(path, tmp_path, analysis, pipeline)
    (tmp_path / "pkg" / "c.py").write_text("z = 9\n", encoding="utf-8")  # pipeline
    assert F.verify_freeze(path, tmp_path, groups=("analysis",))["analysis"]
    (tmp_path / "pkg" / "a.py").write_text("x = 9\n", encoding="utf-8")  # analysis
    with pytest.raises(F.FrozenAnalysisError, match=r"pkg/a\.py: changed"):
        F.verify_freeze(path, tmp_path, groups=("analysis",))


def test_a_missing_file_fails_verification(tmp_path):
    analysis, pipeline = make_tree(tmp_path)
    path = tmp_path / "freeze.json"
    F.write_freeze(path, tmp_path, analysis, pipeline)
    (tmp_path / "pkg" / "a.py").unlink()
    with pytest.raises(F.FrozenAnalysisError, match=r"pkg/a\.py: missing"):
        F.verify_freeze(path, tmp_path)


def test_a_manifest_that_leaves_out_an_expected_file_fails(tmp_path):
    analysis, pipeline = make_tree(tmp_path)
    path = tmp_path / "freeze.json"
    F.write_freeze(path, tmp_path, analysis[:1], pipeline)
    with pytest.raises(F.FrozenAnalysisError, match=r"pkg/b\.py: not in the manifest"):
        F.verify_freeze(
            path, tmp_path, expected={"analysis": analysis, "pipeline": pipeline}
        )


def test_require_frozen_says_what_to_do_when_there_is_no_manifest(tmp_path):
    with pytest.raises(F.FrozenAnalysisError, match="29_freeze_analysis"):
        F.require_frozen(tmp_path / "none.json", tmp_path)


def test_every_listed_file_exists_and_the_groups_do_not_overlap():
    both = (*F.ANALYSIS_FILES, *F.PIPELINE_FILES)
    assert len(both) == len(set(both))
    for rel in both:
        assert (ROOT / rel).is_file(), rel
    assert not set(F.GENERATION_ONLY) & set(both)


def test_the_analysis_group_is_the_m7_code_itself():
    assert set(F.ANALYSIS_FILES) == {
        "src/channel_audit/analysis/contrasts.py",
        "src/channel_audit/analysis/integrity.py",
        "src/channel_audit/analysis/stats.py",
        "src/channel_audit/analysis_freeze.py",
        "scripts/30_analyze.py",
    }


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:  # module level; imports inside functions never run here
        for n in ast.walk(node) if isinstance(node, (ast.If, ast.Try)) else [node]:
            if isinstance(n, ast.ImportFrom) and (n.module or "").startswith(
                "channel_audit"
            ):
                names.add(n.module)
                names.update(f"{n.module}.{a.name}" for a in n.names)
            elif isinstance(n, ast.Import):
                names.update(
                    a.name for a in n.names if a.name.startswith("channel_audit")
                )
    return names


def _closure(entry: Path) -> set[str]:
    seen: set[str] = set()
    todo = [entry]
    while todo:
        for name in _module_imports(todo.pop()):
            base = SRC / name.replace(".", "/")
            for cand in (base.with_suffix(".py"), base / "__init__.py"):
                rel = cand.relative_to(ROOT).as_posix()
                if cand.is_file() and rel not in seen:
                    seen.add(rel)
                    todo.append(cand)
    return {r for r in seen if not r.endswith("__init__.py")}


def test_every_module_the_script_imports_is_frozen_pinned_or_generation_only():
    """A module that can change a number must not be missing from the freeze."""
    covered = {*F.ANALYSIS_FILES, *F.PIPELINE_FILES, *F.GENERATION_ONLY}
    missing = _closure(ROOT / "scripts" / "30_analyze.py") - covered
    assert not missing, f"imported by 30_analyze.py but not in the freeze: {missing}"


def test_a_committed_freeze_matches_the_live_analysis_files():
    """Once the owner freezes the analysis, any later edit to its code fails here."""
    path = ROOT / F.FREEZE_FILE
    if not path.exists():
        pytest.skip("analysis not frozen yet")
    frozen = F.verify_freeze(
        path, ROOT, groups=("analysis",), expected={"analysis": F.ANALYSIS_FILES}
    )
    assert set(frozen["analysis"]) == set(F.ANALYSIS_FILES)
