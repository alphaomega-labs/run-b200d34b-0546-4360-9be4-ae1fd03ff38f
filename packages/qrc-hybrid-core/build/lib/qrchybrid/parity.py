"""Parity-conditioned advantage summaries for quantum-vs-classical comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from scipy import stats

from .metrics import bh_correct, ci95


@dataclass(frozen=True)
class ParityThresholds:
    """Frozen acceptance tuple for practical advantage checks."""

    delta_min: float = 0.015
    p_max: float = 0.01
    r_min: float = 0.65


@dataclass(frozen=True)
class StageGateDecision:
    """Gate result deciding whether downstream mechanism stages should run."""

    should_continue: bool
    max_hard_delta: float


def summarize_parity_advantage(
    seed_level: pd.DataFrame,
    *,
    thresholds: ParityThresholds | None = None,
) -> pd.DataFrame:
    """Aggregate seed-level parity metrics to dataset-level summary.

    Required columns:
    - dataset
    - hard_dataset
    - delta_par
    - delta_naive
    - robustness_r
    """
    thresholds = thresholds or ParityThresholds()

    required = {
        "dataset",
        "hard_dataset",
        "delta_par",
        "delta_naive",
        "robustness_r",
    }
    missing = sorted(required.difference(seed_level.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows: list[dict[str, float | int | bool | str]] = []
    for dataset, frame in seed_level.groupby("dataset"):
        delta = frame["delta_par"].to_numpy()
        t_stat, p_raw = stats.ttest_1samp(delta, 0.0)
        ci_low, ci_high = ci95(delta)
        rows.append(
            {
                "dataset": str(dataset),
                "delta_par_mean": float(delta.mean()),
                "delta_par_ci_low": float(ci_low),
                "delta_par_ci_high": float(ci_high),
                "delta_naive_mean": float(frame["delta_naive"].mean()),
                "p_raw": float(p_raw),
                "robustness_r": float(frame["robustness_r"].mean()),
                "n_seeds": int(len(frame)),
                "t_stat": float(t_stat),
                "hard_dataset": bool(frame["hard_dataset"].iloc[0]),
            }
        )

    summary = pd.DataFrame(rows)
    summary["p_bh"] = bh_correct(summary["p_raw"].tolist())
    summary["A_adv"] = (
        (summary["delta_par_mean"] >= thresholds.delta_min)
        & (summary["p_bh"] <= thresholds.p_max)
        & (summary["robustness_r"] >= thresholds.r_min)
    )
    return summary.sort_values("dataset").reset_index(drop=True)


def stage_gate_from_summary(
    summary: pd.DataFrame,
    *,
    min_hard_delta: float = 0.005,
) -> StageGateDecision:
    """Apply stage-gate policy used by the staged validation pipeline."""
    hard = summary.loc[summary["hard_dataset"], "delta_par_mean"]
    max_hard_delta = float(hard.max()) if not hard.empty else float("-inf")
    return StageGateDecision(
        should_continue=max_hard_delta >= min_hard_delta,
        max_hard_delta=max_hard_delta,
    )


def threshold_sensitivity(
    summary: pd.DataFrame,
    *,
    thresholds: ParityThresholds | None = None,
    multipliers: Iterable[float] = (0.9, 1.0, 1.1),
) -> pd.DataFrame:
    """Evaluate A_adv pass counts under small threshold perturbations."""
    thresholds = thresholds or ParityThresholds()
    rows: list[dict[str, float | int]] = []

    for multiplier in multipliers:
        delta_min = thresholds.delta_min * multiplier
        r_min = max(0.0, min(1.0, thresholds.r_min * multiplier))
        passed = (
            (summary["delta_par_mean"] >= delta_min)
            & (summary["p_bh"] <= thresholds.p_max)
            & (summary["robustness_r"] >= r_min)
        )
        rows.append(
            {
                "multiplier": float(multiplier),
                "delta_min": float(delta_min),
                "r_min": float(r_min),
                "pass_count": int(passed.sum()),
            }
        )

    return pd.DataFrame(rows)
