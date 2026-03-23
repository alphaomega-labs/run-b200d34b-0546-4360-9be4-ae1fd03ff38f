"""Reusable analytics for parity-conditioned quantum reservoir evaluation."""

from .attribution import evaluate_operator_attribution
from .entanglement import evaluate_entanglement_phase_map, partition_sensitivity
from .parity import (
    ParityThresholds,
    StageGateDecision,
    stage_gate_from_summary,
    summarize_parity_advantage,
    threshold_sensitivity,
)

__all__ = [
    "ParityThresholds",
    "StageGateDecision",
    "evaluate_operator_attribution",
    "evaluate_entanglement_phase_map",
    "partition_sensitivity",
    "summarize_parity_advantage",
    "stage_gate_from_summary",
    "threshold_sensitivity",
]
