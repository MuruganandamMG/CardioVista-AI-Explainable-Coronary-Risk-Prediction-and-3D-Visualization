# CardioVista AI — Agent Guide

## Benchmark requests

- When the user says **"test it"** for a benchmark, save a concise model-performance report in `benchmark/`. Cover accuracy, precision, recall, F1, specificity, ROC-AUC, confusion-matrix counts, thresholds, and baseline comparison for all four targets.
- Name the first report `bench1.md`. Name subsequent reports `bench2_YYYY-MM-DD_HH-mm-ss.md`, `bench3_YYYY-MM-DD_HH-mm-ss.md`, and so on, using the creation date and time in **Asia/Calcutta (IST)**.
- Determine the next number from existing `bench<number>` reports. Never overwrite a report or restart numbering when the date changes.
- Include the actual creation date, time, and timezone in every report, including `bench1.md`.
- Include only the timestamp, model/run ID, evaluation sample size and source, actual metric tables, and material limitations. Exclude speed measurements, environment inventories, test-suite logs, and boilerplate unless explicitly requested. Never invent results.
- Use synthetic inputs for prediction benchmarks. Preserve existing model artifacts and evaluation reports; do not retrain or repeat held-out evaluation merely because the user says "test it".
- Setting up this workflow does not run a benchmark or consume `bench1`.

## Mission

CardioVista AI is a reproducible, explainable Python ML project for four
binary predictions from the Z-Alizadeh Sani coronary-disease dataset:

- `cad`: overall coronary artery disease
- `lad`, `lcx`, `rca`: stenosis labels for the three coronary vessels

The project is educational and decision-support software, not a diagnostic
device. Keep this disclaimer with any user-facing prediction surface:

> For educational and decision-support purposes only. Not a substitute for
> diagnostic imaging or professional clinical assessment.

## Non-negotiable rules

1. Preserve `data/raw/z_alizadeh_sani_extension.xlsx` unchanged. Read only
   `Sheet 1 - Table 1`.
2. Treat `Cath`, `LAD`, `LCX`, and `RCA` strictly as targets. Never let them,
   identifiers, or fields derived from them become model features.
3. Prevent leakage. Fit imputation, encoding, scaling, constant-column
   filtering, model selection, calibration, and thresholds using development
   data only. The persisted test partition is reserved for final evaluation.
4. Do not silently repair source labels or force CAD and vessel predictions to
   agree. Report inconsistencies as model/data limitations.
5. Report probabilities as disease-label probabilities—not stenosis
   percentages, lesion locations, causal explanations, or future-event risk.
6. Use synthetic records only in examples, demos, and API fixtures. Do not
   expose patient rows from the source workbook.

## Repository map

| Location | Responsibility |
| --- | --- |
| `src/cardio_risk/data.py` | Workbook loading, target separation, audit, fixed split |
| `src/cardio_risk/schema.py` | Approved 55-feature boundary and request validation |
| `src/cardio_risk/preprocessing.py` | Training-only preprocessing pipelines |
| `src/cardio_risk/train.py` | Candidate search, calibration, threshold selection, frozen fits |
| `src/cardio_risk/evaluate.py` | Metrics and held-out evaluation |
| `src/cardio_risk/predict.py` | Bundle persistence and four-target inference |
| `src/cardio_risk/explain.py` | Score-scale local/global explanations |
| `src/cardio_risk/api.py` | Local FastAPI health, schema, and prediction interface |
| `src/cardio_risk/cli.py` | Reproducible command-line workflow |
| `tests/` | Regression, leakage, schema, training, inference, and API checks |
| `configs/training.json` | Fixed baseline search settings |
| `docs/superpowers/plans/2026-10-07-cardiovascular-ml.md` | Detailed design and acceptance criteria |
| `docs/implementation-progress.md` | Execution ledger and current milestone status |

## Working conventions

- Prefer small, focused changes and update the matching test first or with the
  implementation.
- Use the existing feature names exactly at the model/API boundary. A request
  must include every approved feature; `null` is the supported way to mark a
  missing value.
- Keep the target order fixed: `cad`, `lad`, `lcx`, `rca`.
- Preserve deterministic settings: the saved split uses seed `42`; reuse the
  persisted `data/processed/split.json` rather than generating another split.
- Do not overwrite an existing run directory. New training attempts need a new
  run ID so artifacts and reports remain traceable.
- Store artifacts and reports together under `artifacts/<run-id>/` and
  `reports/<run-id>/`. Evaluation must use the saved, frozen bundle.
- Explanation output describes the underlying model score. If a model is
  calibrated, clearly distinguish its score probability from the final served
  probability.
- Keep dependencies lightweight and CPU-compatible. The intended baseline is
  logistic regression and CatBoost; do not add models or services without a
  demonstrated need.

## Useful commands

Run these from the repository root with the project environment active:

```powershell
python -m pytest -q
python -m cardio_risk.cli audit --data data/raw/z_alizadeh_sani_extension.xlsx
python -m cardio_risk.cli split --data data/raw/z_alizadeh_sani_extension.xlsx
python -m cardio_risk.cli train --config configs/training.json --run-id <run-id>
python -m cardio_risk.cli evaluate --run-id <run-id>
python -m cardio_risk.cli predict --run-id <run-id> --input <synthetic-json>
```

Before handing work off, run the smallest relevant tests and then the full
suite when practical. Report actual results and limitations; never invent an
accuracy target or call development estimates held-out performance.

## Current implementation focus

The data audit and fixed development/test split are in place. The project is
working through the model-training, inference, explanation, API, and final
reproducibility milestones. Consult the implementation ledger and detailed
plan before changing a workflow that affects saved artifacts, evaluation, or
the API contract.

## What good work looks like

A successful contribution is reproducible from raw data to a versioned bundle,
has tests for its behavior and leakage boundaries, preserves clinical and
statistical limitations in its outputs, and makes the next contributor's job
clearer—not merely more feature-rich.
