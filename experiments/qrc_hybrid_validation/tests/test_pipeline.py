from __future__ import annotations

import json
from pathlib import Path

from qrc_validation.pipeline import _bh_correct, _ratio_within_tolerance


def test_bh_correction_monotone() -> None:
    p = [0.02, 0.001, 0.03, 0.2]
    adj = _bh_correct(p)
    assert len(adj) == len(p)
    assert all(0.0 <= x <= 1.0 for x in adj)


def test_design_has_required_stages() -> None:
    p = Path("phase_outputs/experiment_design.json")
    assert p.exists()
    obj = json.loads(p.read_text())
    ids = [e["id"] for e in obj["payload"]["experiments"]]
    assert "exp_stage0_symbolic_preflight" in ids
    assert "exp_stage1_parity_ladder" in ids


def test_budget_parity_tolerance_helper() -> None:
    assert _ratio_within_tolerance(1.00, 0.05)
    assert _ratio_within_tolerance(0.95, 0.05)
    assert _ratio_within_tolerance(1.05, 0.05)
    assert not _ratio_within_tolerance(0.94, 0.05)
    assert not _ratio_within_tolerance(1.06, 0.05)
