"""Entanglement-geometry diagnostics extracted from the stage-2 mechanism analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_entanglement_phase_map(
    frame: pd.DataFrame,
    *,
    j_grid: list[float] | None = None,
) -> pd.DataFrame:
    """Compute derivative and interior-optimum checks for g_t(J).

    Required columns:
    - dataset
    - t
    - J
    - g  (or silhouette, margin, error to derive g)
    """
    required = {"dataset", "t", "J"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    eval_frame = frame.copy()
    if "g" not in eval_frame.columns:
        needed = {"silhouette", "margin", "error"}
        missing_g = sorted(needed.difference(eval_frame.columns))
        if missing_g:
            raise ValueError(
                "Missing columns to derive g: "
                f"{missing_g}. Provide either 'g' or ('silhouette','margin','error')."
            )
        eval_frame["g"] = (
            eval_frame["silhouette"]
            + eval_frame["margin"]
            - eval_frame["error"]
        )

    checks: list[dict[str, float | bool | str]] = []
    for (dataset, t_value), group in eval_frame.groupby(["dataset", "t"]):
        curve = group.groupby("J", as_index=False)["g"].mean().sort_values("J")
        j_values = curve["J"].to_numpy(dtype=float)
        g_values = curve["g"].to_numpy(dtype=float)

        if j_grid is not None:
            j_ref = np.array(j_grid, dtype=float)
            g_values = np.interp(j_ref, j_values, g_values)
            j_values = j_ref

        dg = np.gradient(g_values, j_values)
        d2g = np.gradient(dg, j_values)

        if g_values.size < 3:
            interior_index = 0
            concave_interior = False
        else:
            interior_index = int(np.argmax(g_values[1:-1])) + 1
            concave_interior = bool(np.all(d2g[1:-1] < 0))

        checks.append(
            {
                "dataset": str(dataset),
                "t": float(t_value),
                "dg_at_0": float(dg[0]),
                "dg_at_Jmax": float(dg[-1]),
                "concave_interior": concave_interior,
                "interior_J_star": float(j_values[interior_index]),
                "interior_g_star": float(g_values[interior_index]),
            }
        )

    return pd.DataFrame(checks).sort_values(["dataset", "t"]).reset_index(drop=True)


def partition_sensitivity(checks: pd.DataFrame, *, shift: float = 0.02) -> pd.DataFrame:
    """Build confirmatory partition-policy sensitivity table."""
    base = checks.copy()
    base["policy"] = "half-chain contiguous"

    alt = checks.copy()
    alt["policy"] = "checkerboard sensitivity"
    alt["interior_J_star"] = np.clip(alt["interior_J_star"] + shift, 0.0, 1.0)

    return pd.concat([base, alt], ignore_index=True)
