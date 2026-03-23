from __future__ import annotations

import pandas as pd

from qrchybrid import (
    ParityThresholds,
    evaluate_entanglement_phase_map,
    stage_gate_from_summary,
    summarize_parity_advantage,
)

seed_level = pd.DataFrame(
    {
        "dataset": ["Fashion-MNIST", "Fashion-MNIST", "Fashion-MNIST", "Fashion-MNIST", "Fashion-MNIST"],
        "hard_dataset": [True, True, True, True, True],
        "delta_par": [0.018, 0.017, 0.019, 0.018, 0.020],
        "delta_naive": [0.028, 0.027, 0.029, 0.028, 0.030],
        "robustness_r": [0.69, 0.70, 0.68, 0.71, 0.70],
    }
)

summary = summarize_parity_advantage(seed_level, thresholds=ParityThresholds())
print(summary[["dataset", "delta_par_mean", "A_adv"]])
print(stage_gate_from_summary(summary))

phase_samples = pd.DataFrame(
    {
        "dataset": ["Fashion-MNIST"] * 5,
        "t": [0.5] * 5,
        "J": [0.0, 0.1, 0.2, 0.3, 0.4],
        "g": [0.18, 0.24, 0.28, 0.27, 0.21],
    }
)
checks = evaluate_entanglement_phase_map(phase_samples)
print(checks)
