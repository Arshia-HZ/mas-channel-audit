"""The real analysis agrees with a separately written one on planted mock data."""

from __future__ import annotations

import numpy as np
import pytest

from channel_audit.analysis.contrasts import analyze
from channel_audit.backends.mock import MockBackend, MockWorld, fake_problems
from channel_audit.channels import ARMS, VARIANT_ARMS
from channel_audit.runner import RunConfig, open_run
from channel_audit.shards import LazyBackend, run_shards
from independent_reference import contrasts, quantities

N_BOOT = 1000
EFFECTS = {
    "failure_cause": 0.15,
    "full_solution": -0.05,
    "placebo": -0.2,
    "answer_conf": 0.1,
}


def build(root, max_tokens):
    problems = fake_problems(40)
    world = MockWorld.from_problems(problems, base_p=0.5)
    run_shards(
        root,
        RunConfig(model_id="mock/model", replicates=2, max_tokens=max_tokens),
        problems,
        LazyBackend(lambda: MockBackend(world, effects=EFFECTS)),
        size=20,
        arms=(*ARMS, *VARIANT_ARMS),
        log=lambda s: None,
    )
    return root


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    root = build(tmp_path_factory.mktemp("clean"), 512)
    ctxs = [open_run(d) for d in sorted(root.glob("shard_*"))]
    return root, analyze(ctxs, n_boot=N_BOOT)


@pytest.fixture(scope="module")
def truncating(tmp_path_factory):
    root = build(tmp_path_factory.mktemp("cut"), 75)  # about 40% of outputs cut off
    ctxs = [open_run(d) for d in sorted(root.glob("shard_*"))]
    return root, analyze(ctxs, n_boot=N_BOOT)


def compare(root, analysis, **kw):
    q = quantities(root, **kw)
    for arm in q["arms"]:
        row = analysis["arms"][arm]
        assert row["accuracy"] == pytest.approx(
            np.mean(list(q["accuracy"][arm].values()))
        )
        assert row["tokens"] == pytest.approx(q["tokens"][arm])
        assert row["matched_k"] == pytest.approx(q["k"][arm])
        assert row["sc_accuracy"] == pytest.approx(
            np.mean(list(q["matched"][arm].values()))
        )
        assert row["gain"] == pytest.approx(np.mean(list(q["gain"][arm].values())))
    for name, ref in contrasts(q, N_BOOT).items():
        got = analysis["contrasts"].get(name) or analysis["secondary"][name]
        for key, value in ref.items():
            if got[key] is None:  # contrasts 1 and 2 have no p-value by design
                assert key == "p" and name in (
                    "full_solution_vs_sc",
                    "answer_only_vs_full_solution",
                )
                continue
            assert got[key] == pytest.approx(value, abs=1e-9), (name, key)


def test_main_analysis_agrees_arm_by_arm_and_contrast_by_contrast(clean):
    root, rep = clean
    compare(root, rep["analyses"]["main"])


def test_a_level_agrees(clean):
    root, rep = clean
    compare(root, rep["analyses"]["level_3"], only_level=3)
    compare(root, rep["analyses"]["level_4"], only_level=4)


def test_the_round_0_truncation_run_agrees_when_outputs_are_cut_off(truncating):
    root, rep = truncating
    assert rep["analyses"]["main"]["n_problems"] == 40
    sens = rep["analyses"]["no_round0_truncation"]
    assert 0 < sens["n_problems"] < 40
    compare(root, rep["analyses"]["main"])
    compare(root, sens, drop_round0_truncated=True)
