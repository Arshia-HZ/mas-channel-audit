"""Paired bootstrap, ported from Phase 1 under new tests."""

from __future__ import annotations

import numpy as np
import pytest

from channel_audit.analysis.stats import paired_bootstrap_ci


def test_mean_and_interval_for_a_clear_positive_effect():
    diffs = np.full(100, 0.1) + np.random.default_rng(0).normal(0, 0.05, 100)
    mean, lo, hi = paired_bootstrap_ci(diffs, n_boot=2000, seed=1)
    assert mean == pytest.approx(diffs.mean())
    assert lo < mean < hi
    assert lo > 0


def test_interval_covers_zero_for_pure_noise():
    diffs = np.random.default_rng(3).normal(0, 1, 200)
    _, lo, hi = paired_bootstrap_ci(diffs, n_boot=2000, seed=1)
    assert lo < 0 < hi


def test_deterministic_given_seed_and_different_across_seeds():
    d = np.random.default_rng(0).normal(0, 1, 50)
    assert paired_bootstrap_ci(d, n_boot=500, seed=7) == paired_bootstrap_ci(
        d, n_boot=500, seed=7
    )
    assert paired_bootstrap_ci(d, n_boot=500, seed=7) != paired_bootstrap_ci(
        d, n_boot=500, seed=8
    )


def test_constant_differences_give_a_degenerate_interval():
    assert paired_bootstrap_ci([0.2] * 10, n_boot=200, seed=0) == (
        pytest.approx(0.2),
        pytest.approx(0.2),
        pytest.approx(0.2),
    )


def test_wider_alpha_gives_a_narrower_interval():
    d = np.random.default_rng(5).normal(0, 1, 100)
    _, lo95, hi95 = paired_bootstrap_ci(d, alpha=0.05, n_boot=3000, seed=2)
    _, lo80, hi80 = paired_bootstrap_ci(d, alpha=0.20, n_boot=3000, seed=2)
    assert (hi80 - lo80) < (hi95 - lo95)


def test_empty_input_is_an_error():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], n_boot=10)
