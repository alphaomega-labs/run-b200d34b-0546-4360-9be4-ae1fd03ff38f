"""Operator-vs-dynamics attribution utilities extracted from stage-3 analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import ci95, within_tolerance


def _anova_components(frame: pd.DataFrame) -> dict[str, float]:
    """Balanced crossed-effects ANOVA component estimates."""
    mean_total = float(frame["y"].mean())
    op_levels = sorted(frame["operator_policy"].unique())
    dyn_levels = sorted(frame["dynamics_family"].unique())
    n_op = len(op_levels)
    n_dyn = len(dyn_levels)

    cell_sizes = frame.groupby(["operator_policy", "dynamics_family"]).size()
    if cell_sizes.empty or cell_sizes.nunique() != 1:
        raise ValueError("Attribution requires a balanced operator x dynamics design.")
    n_rep = int(cell_sizes.iloc[0])

    means_op = frame.groupby("operator_policy")["y"].mean()
    means_dyn = frame.groupby("dynamics_family")["y"].mean()
    means_cell = frame.groupby(["operator_policy", "dynamics_family"])["y"].mean()

    ss_op = n_dyn * n_rep * float(((means_op - mean_total) ** 2).sum())
    ss_dyn = n_op * n_rep * float(((means_dyn - mean_total) ** 2).sum())
    ss_int = n_rep * float(
        sum(
            (
                means_cell.loc[(op, dyn)]
                - means_op.loc[op]
                - means_dyn.loc[dyn]
                + mean_total
            )
            ** 2
            for op in op_levels
            for dyn in dyn_levels
        )
    )
    ss_error = float(
        sum(
            (row.y - means_cell.loc[(row.operator_policy, row.dynamics_family)]) ** 2
            for row in frame.itertuples()
        )
    )

    df_op = n_op - 1
    df_dyn = n_dyn - 1
    df_int = (n_op - 1) * (n_dyn - 1)
    df_error = n_op * n_dyn * (n_rep - 1)

    ms_op = ss_op / df_op if df_op > 0 else 0.0
    ms_dyn = ss_dyn / df_dyn if df_dyn > 0 else 0.0
    ms_int = ss_int / df_int if df_int > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    sigma_op = (ms_op - ms_int) / (n_dyn * n_rep)
    sigma_dyn = (ms_dyn - ms_int) / (n_op * n_rep)
    sigma_op_clip = max(0.0, sigma_op)
    sigma_dyn_clip = max(0.0, sigma_dyn)
    denom = sigma_op_clip + sigma_dyn_clip
    r_op = sigma_op_clip / denom if denom > 0 else float("nan")

    return {
        "ms_operator": float(ms_op),
        "ms_dynamics": float(ms_dyn),
        "ms_interaction": float(ms_int),
        "ms_error": float(ms_error),
        "sigma_op2_hat": float(sigma_op),
        "sigma_dyn2_hat": float(sigma_dyn),
        "R_op_hat": float(r_op),
        "n_op": float(n_op),
        "n_dyn": float(n_dyn),
        "n_r": float(n_rep),
    }


def evaluate_operator_attribution(
    frame: pd.DataFrame,
    *,
    budget_tolerance: float = 0.05,
    bootstrap_samples: int = 300,
    bootstrap_seed: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | bool]]:
    """Compute dataset-level operator attribution with parity filtering.

    Required columns:
    - dataset
    - operator_policy
    - dynamics_family
    - y
    - budget_parity_ratio
    - delta_par
    - delta_naive
    """
    required = {
        "dataset",
        "operator_policy",
        "dynamics_family",
        "y",
        "budget_parity_ratio",
        "delta_par",
        "delta_naive",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    mask = frame["budget_parity_ratio"].apply(
        lambda value: within_tolerance(float(value), budget_tolerance)
    )
    filtered = frame.loc[mask].copy()
    violations = int((~mask).sum())
    if filtered.empty:
        raise ValueError("All rows violate budget parity tolerance.")

    rows: list[dict[str, float | str]] = []
    for dataset, group in filtered.groupby("dataset"):
        components = _anova_components(group)

        rng = np.random.default_rng(bootstrap_seed + len(str(dataset)))
        r_values: list[float] = []
        cells = list(group.groupby(["operator_policy", "dynamics_family"]))
        for _ in range(bootstrap_samples):
            sampled = []
            for _, cell in cells:
                sampled.append(
                    cell.sample(
                        n=len(cell),
                        replace=True,
                        random_state=int(rng.integers(0, 1_000_000)),
                    )
                )
            boot = pd.concat(sampled, ignore_index=True)
            r_hat = _anova_components(boot)["R_op_hat"]
            if not math.isnan(r_hat):
                r_values.append(float(r_hat))

        if r_values:
            ci_low = float(np.percentile(r_values, 2.5))
            ci_high = float(np.percentile(r_values, 97.5))
        else:
            ci_low = float("nan")
            ci_high = float("nan")

        shrinkage = (group["delta_naive"] - group["delta_par"]).to_numpy(dtype=float)
        shrink_ci_low, shrink_ci_high = ci95(shrinkage)

        rows.append(
            {
                "dataset": str(dataset),
                **components,
                "bootstrap_CI_R_op_low": ci_low,
                "bootstrap_CI_R_op_high": ci_high,
                "delta_shrink_mean": float(shrinkage.mean()),
                "delta_shrink_ci_low": float(shrink_ci_low),
                "delta_shrink_ci_high": float(shrink_ci_high),
                "budget_parity_ratio_mean": float(group["budget_parity_ratio"].mean()),
            }
        )

    effects = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    shrinkage = effects[
        [
            "dataset",
            "delta_shrink_mean",
            "delta_shrink_ci_low",
            "delta_shrink_ci_high",
        ]
    ].copy()

    report: dict[str, float | int | bool] = {
        "budget_parity_records_raw": int(len(frame)),
        "budget_parity_violation_count_raw": violations,
        "budget_parity_within_tolerance_raw": bool(violations == 0),
        "r_op_lb_gt_05_count": int((effects["bootstrap_CI_R_op_low"] > 0.5).sum()),
    }

    return effects, shrinkage, report
