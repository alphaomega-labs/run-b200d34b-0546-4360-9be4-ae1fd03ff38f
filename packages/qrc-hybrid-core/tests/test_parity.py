from __future__ import annotations

import pandas as pd

from qrchybrid import (
    ParityThresholds,
    stage_gate_from_summary,
    summarize_parity_advantage,
    threshold_sensitivity,
)


def test_parity_summary_and_gate() -> None:
    records = []
    for seed in [11, 23, 37, 47, 59]:
        records.append(
            {
                "dataset": "Kuzushiji-MNIST",
                "hard_dataset": True,
                "delta_par": 0.024 + 0.0002 * (seed % 3),
                "delta_naive": 0.033 + 0.0002 * (seed % 2),
                "robustness_r": 0.74,
            }
        )
        records.append(
            {
                "dataset": "MNIST",
                "hard_dataset": False,
                "delta_par": 0.006,
                "delta_naive": 0.014,
                "robustness_r": 0.64,
            }
        )

    frame = pd.DataFrame.from_records(records)
    summary = summarize_parity_advantage(frame, thresholds=ParityThresholds())

    assert set(["dataset", "A_adv", "p_bh", "delta_par_mean"]).issubset(summary.columns)
    assert bool(summary.loc[summary["dataset"] == "Kuzushiji-MNIST", "A_adv"].iloc[0])

    gate = stage_gate_from_summary(summary, min_hard_delta=0.005)
    assert gate.should_continue
    assert gate.max_hard_delta >= 0.02

    sensitivity = threshold_sensitivity(summary)
    assert len(sensitivity) == 3
    assert "pass_count" in sensitivity.columns
