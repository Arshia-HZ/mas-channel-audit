"""Problem-level resampling statistics (ported from Phase 1)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_means(
    diffs: Sequence[float] | np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> np.ndarray:
    """Means of ``n_boot`` resamples, drawn as ``paired_bootstrap_ci`` draws them."""
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        raise ValueError("bootstrap_means needs at least one difference")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    return d[idx].mean(axis=1)


def paired_bootstrap_ci(
    diffs: Sequence[float] | np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of per-problem paired differences with a percentile bootstrap CI.

    ``diffs`` holds one value per problem (replicates already averaged within
    problem). Returns (mean, lower, upper) for a (1 - alpha) interval.
    """
    d = np.asarray(diffs, dtype=float)
    if d.size == 0:
        raise ValueError("paired_bootstrap_ci needs at least one difference")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)
