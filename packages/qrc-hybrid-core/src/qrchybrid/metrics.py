"""Statistical helpers extracted from the QRC validation core."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def bh_correct(p_values: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg correction preserving monotonic adjusted p-values."""
    arr = np.array(p_values, dtype=float)
    n = len(arr)
    if n == 0:
        return []

    order = np.argsort(arr)
    adjusted = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = arr[order[i]] * n / rank
        running_min = min(running_min, candidate)
        adjusted[order[i]] = running_min
    return adjusted.tolist()


def ci95(values: Sequence[float]) -> tuple[float, float]:
    """Two-sided 95% confidence interval for the sample mean using Student's t."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        x = float(arr[0])
        return x, x

    mean = float(arr.mean())
    sem = float(stats.sem(arr))
    t_value = float(stats.t.ppf(0.975, arr.size - 1))
    return mean - t_value * sem, mean + t_value * sem


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    random_seed: int = 0,
    n_boot: int = 500,
) -> tuple[float, float]:
    """Non-parametric percentile CI for sample mean."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(random_seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, arr.size, arr.size)
        samples[i] = float(arr[idx].mean())
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def within_tolerance(ratio: float, tolerance: float) -> bool:
    """Whether ratio lies within symmetric tolerance around 1.0."""
    return (1.0 - tolerance) <= ratio <= (1.0 + tolerance)
