# QRC Hybrid Validation Experiments

This package executes the staged validation plan from `phase_outputs/experiment_design.json`.

## Stages
- `stage0`: Symbolic and theorem preflight (SymPy identity checks A-D).
- `stage1`: Parity-conditioned dataset ladder with frozen acceptance tuple.
- `stage2`: Entanglement phase map (conditional on stage gate).
- `stage3`: Operator attribution and variance-share analysis (conditional on stage gate).

## Environment
Use the experiment virtualenv under `experiments/.venv`.

## Run
```bash
experiments/.venv/bin/python run_experiments.py --workspace-root .
```

## Outputs
- Main run report: `experiments/qrc_hybrid_validation/outputs/run_result.json`
- Summary for writing phase: `experiments/qrc_hybrid_validation/outputs/results_summary.json`
- SymPy report: `experiments/qrc_hybrid_validation/outputs/sympy_validation_report.json`
- Experiment log: `experiments/qrc_hybrid_validation/experiment_log.jsonl`
- Figures: `paper/figures/stage*.pdf`
- Tables: `paper/tables/*.csv`
- Datasets: `paper/data/*.csv`
