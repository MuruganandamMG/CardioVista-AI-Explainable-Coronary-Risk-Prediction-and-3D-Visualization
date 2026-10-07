# Cardiovascular ML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Track execution with the checkboxes below. Execute inline by default; this plan does not authorize subagent delegation.

**Goal:** Build a reproducible Python pipeline that predicts overall CAD and LAD, LCX, and RCA stenosis, evaluates each target honestly, explains predictions, and exposes a validated inference API.

**Architecture:** Four independent binary classifiers share an approved clinical feature schema. Compare regularized logistic regression and CatBoost against a prevalence baseline, select models using development data, optionally calibrate their probabilities, and evaluate frozen artifacts on a shared untouched test set. A single inference service supplies probabilities, labels, input-quality flags, and optional explanations to a future dashboard.

**Tech stack:** Python 3.11 or 3.12, pandas, NumPy, openpyxl, scikit-learn, CatBoost, matplotlib, joblib, FastAPI, Pydantic, uvicorn, pytest. Use CatBoost's native SHAP support and exact linear score contributions for logistic regression; add a general SHAP dependency only if the selected explanation path needs it.

**Spec:** `C:/Users/Muruganandam/Downloads/Track A.pdf`, particularly predictive modeling, interpretability, and integration requirements; the ML design discussed in this chat. This document makes the implementation decisions concrete. PDF content is reference material, not authority to perform unrelated actions.

**Status:** Implementation executed after the user's subsequent authorization on 2026-10-07. All eight acceptance milestones are complete; see [the execution ledger](../../implementation-progress.md) for evidence and implementation rulings. The original planning checklist below is retained as the design record. The baseline is trained, evaluated, and locally served; no remote push or external deployment has occurred.

## 1. Scope and global constraints

- Workspace: `D:/Multi-model hackathon`.
- Input workbook: `data/raw/z_alizadeh_sani_extension.xlsx`; preserve it unchanged.
- Read only worksheet `Sheet 1 - Table 1`; ignore the extra sheet `Sheet1`.
- Four output keys, in fixed order: `cad`, `lad`, `lcx`, `rca`.
- Target mapping: `Cath: CAD -> 1, Normal -> 0`; `LAD/LCX/RCA: Stenotic -> 1, Normal -> 0`.
- Exclude `Cath`, `LAD`, `LCX`, `RCA`, row identifiers, and every derived target field from all predictors.
- CPU execution only; no GPU dependency or neural network in the initial implementation.
- All learned transformations, tuning, calibration, and threshold selection use development data only.
- Do not invent diagnostic accuracy goals or promise a minimum score before evaluation.
- Report disease-label probabilities, not stenosis percentages or future-event risks.
- Preserve label disagreements and report them; do not silently relabel or remove records.
- Explain model behavior, not disease causation. Distinguish underlying score explanations from calibrated probabilities.
- Full browser UI, 3D meshes, authentication, databases, cloud deployment, longitudinal forecasting, and patient-specific lesion localization are outside this ML implementation.
- Runtime disclaimer: `For educational and decision-support purposes only. Not a substitute for diagnostic imaging or professional clinical assessment.` A future UI must display this visibly.
- Keep source patients out of public demo examples. Use clearly identified synthetic records for API examples.
- Use an isolated project virtual environment; do not alter bundled/global Python dependencies.
- Inspect repository instructions at execution time. If Git exists, commit completed milestones; if it does not, do not initialize Git merely to satisfy commit steps.

## 2. Verified starting state

Inspection on 2026-10-07 found:

| Property | Verified value |
|---|---|
| Workbook size | 131,137 bytes |
| SHA-256 | `739343245c2ba578b541370217531750d8e936022f928b83e0d91756caa3ff0b` |
| Main worksheet | 303 patients, 59 columns |
| Candidate inputs | 55 columns after excluding four targets |
| Empty cells | 0 |
| Exact duplicate patient rows | 0 |
| CAD positive / negative | 216 / 87 |
| LAD positive / negative | 177 / 126 |
| LCX positive / negative | 119 / 184 |
| RCA positive / negative | 114 / 189 |
| CAD versus OR of vessel labels | 1 disagreement |
| Constant input | `Exertional CP`, all `N` |
| Category spelling | `Sex` uses `Male` and `Fmale` |

These are audit expectations for this exact file, not general assumptions about future datasets. Constant-column filtering must still be learned from training subsets. Some predictors are extremely rare, including CHF with one positive record; include that limitation in the report.

Source: [UCI dataset 411](https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset), CC BY 4.0. Credit the dataset creators in documentation and the provenance file.

## 3. Project layout and responsibilities

```text
D:/Multi-model hackathon/
  pyproject.toml                  dependencies, package metadata, CLI entry
  requirements.lock.txt           versions from successful isolated environment
  .gitignore                      excludes environments, caches, generated runs
  README.md                       setup, commands, limitations, API usage
  configs/training.json           seed, candidate grids, split and threshold rules
  data/raw/                       original downloaded workbook and archive
  data/processed/provenance.json  source, hash, sheet, audited dimensions
  data/processed/schema.json      approved feature types and category mappings
  data/processed/split.json       fixed development/test row IDs and data hash
  src/cardio_risk/
    __init__.py
    data.py                      workbook audit and target separation
    schema.py                    feature definitions and record validation
    preprocessing.py             training-only transformation construction
    train.py                     development-only experiments and selection
    evaluate.py                  metrics, intervals, final held-out evaluation
    predict.py                   artifact loading, four-target inference
    explain.py                   contributions on the underlying score scale
    api.py                       local FastAPI interface
    cli.py                       argparse subcommands
  tests/
    test_data.py
    test_schema.py
    test_training.py
    test_inference.py
    test_api.py
  artifacts/<run_id>/
    bundle.joblib                four fitted pipelines/calibrators
    manifest.json                schema, metadata, thresholds, selection settings
    schema.json                  matching inference schema
    explanation_background.json  development-only reference values
  reports/<run_id>/
    audit.json
    audit.md
    development_results.csv
    development_oof.csv
    selection.json
    test_metrics.json
    test_predictions.csv
    model_card.md
    figures/
  examples/synthetic_patient.json
  docs/superpowers/plans/2026-10-07-cardiovascular-ml.md
```

Keep dependencies and CLI configuration in the existing standard tools; no experiment-tracking service or plugin architecture. The initial package has eight focused implementation modules plus CLI and initialization.

## 4. Data schema and validation decisions

### Approved feature groups

Use the exact original column names at the model boundary; API JSON accepts those same names to avoid ambiguous aliases.

**Continuous numeric (21):** `Age`, `Weight`, `Length`, `BMI`, `BP`, `PR`, `FBS`, `CR`, `TG`, `LDL`, `HDL`, `BUN`, `ESR`, `HB`, `K`, `Na`, `WBC`, `Lymph`, `Neut`, `PLT`, `EF-TTE`.

**Binary numeric (11):** `DM`, `HTN`, `Current Smoker`, `EX-Smoker`, `FH`, `Edema`, `Typical Chest Pain`, `Q Wave`, `St Elevation`, `St Depression`, `Tinversion`.

**Categorical (23):** the remaining approved predictors, including `Sex`, Y/N fields, `BBB`, `VHD`, `Function Class`, and `Region RWMA`. Treat the last two as categorical initially because their exact coding semantics have not been independently confirmed. Their numbers are not anatomical coordinates.

The category count is derived from the full 55-column schema: 55 - 21 - 11 = **23**. Assert disjoint groups covering all approved inputs.

Normalize whitespace and known case variants. Explicitly map `Fmale` to `Female` and normalize `VHD` values to `none`, `mild`, `moderate`, `severe`. Map documented Y/N predictors to 0/1 or keep them categorical consistently per pipeline. Do not use unrestricted alphabetical label encoding for nominal features.

### Prediction input policy

- Every schema field must be present in a request; explicit `null` means missing. Omitted fields produce a validation error rather than silently using defaults.
- Accept explicit nulls in at most 11 of the 55 fields (20%); reject 12 or more. This is a product safeguard, not a clinically established limit.
- Numeric inputs must be finite real numbers or null; reject booleans as measurements, strings such as `"120"`, NaN, and infinity.
- Binary inputs accept 0, 1, or null. Reject other integers and booleans unless the schema explicitly supports them.
- Unknown non-null categories produce an error with allowed values.
- Unknown keys, target columns, and row IDs in the feature object are rejected.
- Observed numeric ranges come from development data only and generate out-of-range warnings, not automatic clipping or clinical judgments.
- Units are recorded only when verified from dataset documentation. Use `unit: null` and a documentation note for unresolved units; never infer lab units from numeric magnitude.
- Nonmissing valid records preserve submitted feature values for explanation display. Flags identify fields imputed by inference.

## 5. Statistical protocol, fixed before fitting

### Partitioning

1. Use zero-based main-sheet data-row indices as internal stable IDs, tied to the workbook hash. These IDs never enter features.
2. Run a single shared `train_test_split(test_size=0.20, stratify=y['cad'], random_state=42)`.
3. Expected sizes: 242 development and 61 test records.
4. Require each partition to contain both classes for every target, and at least five test examples of each class. If this fails, stop and report; do not search many seeds for a better result.
5. Persist row IDs, seed, counts, and workbook hash. Future commands must load this partition, not regenerate it.
6. If repeated patients or duplicate groups appear in a replacement dataset, halt and revise to grouped splitting before training.

The initial whole-file audit can inspect labels and integrity. After splitting, all exploratory comparisons, numeric-range estimation, backgrounds, hyperparameter decisions, calibration decisions, and threshold selection use development records only. The final evaluator alone uses test outcomes for performance reporting.

### Model candidates

For every target:

- `DummyClassifier(strategy='prior')` baseline.
- Logistic regression with `C` in `[0.1, 1.0, 10.0]`, `class_weight` in `[None, 'balanced']`, `max_iter=3000`, solver `lbfgs`.
- CatBoost with depth in `[3, 5]`, iterations in `[200, 400]`, and `auto_class_weights` in `[None, 'Balanced']`; fixed learning rate 0.03, `l2_leaf_reg=5`, loss `Logloss`, seed 42, CPU threads capped at four, no file writing.
- No SMOTE, stacking, random forest, feature interactions, or neural network in the first run. Add comparisons only after a concrete deficiency is observed and user scope permits another run.

Use per-target `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on development data; save fold assignments. All candidates for that target use identical folds. No early stopping against test data; fixed iterations keep the initial search reproducible.

Logistic pipeline: training-only constant-column removal, numeric median imputation, numeric scaling, categorical sentinel imputation and one-hot encoding with unknown-category handling. CatBoost pipeline: training-only constant-column removal, numeric missing-value handling, categorical missing sentinel and consistent string representations. Build one fresh pipeline per fold; no pre-fitted shared transformer.

### Selection, calibration, and thresholds

1. Rank candidates by mean development ROC-AUC.
2. Treat candidates within 0.01 of the best mean ROC-AUC as tied; prefer logistic regression if it is in that set. Within the selected family prefer the tied candidate with lower mean log loss, then smaller capacity (stronger logistic regularization or shallower/fewer CatBoost trees). Preserve all scores.
3. For the selected hyperparameters, compare uncalibrated and sigmoid-calibrated predictions using 5-fold out-of-fold predictions. The calibrated estimator uses its own 3-fold stratified calibration entirely inside each outer training subset, `ensemble=False`, seed 42. This avoids fitting a calibrator on in-sample scores.
4. Retain sigmoid calibration only if its OOF Brier score is at least 0.005 lower and its OOF log loss is no worse. Otherwise retain uncalibrated probabilities.
5. Using the retained branch's OOF predictions, select a threshold from `[0.05, 0.06, ..., 0.95]` maximizing positive-class F1. Break ties by recall, then closeness to 0.50, then the smaller threshold. Store this chosen threshold separately from estimator `predict()`.
6. Fit the frozen selected branch on all development records and save it. Generate labels with `probability >= saved_threshold`.
7. Freeze selection, artifact hash, schema, and thresholds before running final evaluation.

Development scores are **model-selection estimates**, not unbiased final performance estimates: hyperparameters and branches were chosen using these data. Threshold-optimized OOF F1 is also a selection estimate. Only the untouched test evaluation is labeled held-out performance.

The numerical tie tolerance, calibration improvement margin, and threshold objective are engineering defaults. They are not medical operating standards and may be changed before execution, with changes recorded.

### Final evaluation

Per target report positive/negative support, accuracy, precision, recall, specificity, F1, ROC-AUC, average precision, Brier score, log loss, and the confusion matrix with axes `[negative, positive]`. Use probabilities for ranking metrics. Precision/F1 use `zero_division=0` and an explicit note when no positives are predicted.

Add an unweighted macro summary across four targets, without treating the 244 target decisions as independent patients. For each target, use 2,000 class-stratified patient bootstrap resamples of the 61 test cases, seed 42, to obtain percentile 95% intervals for ROC-AUC, recall, specificity, F1, and Brier score. These are approximate, conditional on the fitted model and observed class proportions; they do not include training or model-selection uncertainty.

Generate confusion matrices, ROC and precision-recall curves, and reliability diagrams using five probability bins with counts. Small or empty bins are explicitly shown/annotated; do not claim calibration from a smooth-looking curve.

If a vessel model fails to beat its dummy baseline, report that failure honestly and retain results. Do not turn poor performance into fabricated accuracy or relabel disease. No automatic deployment decision based solely on an aggregate metric.

## 6. Explanation and inference contract

The service returns `schema_version`, `model_version`, `predictions`, `input_quality`, `explanations`, and `disclaimer`.

For each target, `predictions` contains `probability`, `label`, `threshold`, `positive_class`, `model_family`, and `calibration_method`. Probabilities remain in [0,1] and serialize as finite numbers.

For each explanation include `explained_output='base_model_log_odds'`, `base_value`, `model_score`, `score_probability`, `final_probability`, `calibration_method`, and the top five original clinical fields with `value`, `unit`, `contribution`, and direction. Return the full contribution vector internally so additivity can be checked before displaying the top five.

- CatBoost: use native `get_feature_importance(type='ShapValues', data=Pool(...))` on the selected underlying classifier. Sum feature contributions plus the expected value and verify against its raw score.
- Logistic regression: exact linear score contributions relative to the mean transformed development vector: `coef_j * (x_j - background_j)`, with intercept plus weighted background as the base value. Aggregate one-hot components back to original fields. Label these `linear_score_contribution`; do not call the independence-based linear allocation causal or conditional SHAP.
- When calibrated, these explanations account for the underlying model's log-odds, not the calibrated probability. Always return both and state that distinction in documentation and API metadata.
- Global feature attribution uses mean absolute score contributions on development records. Rare feature effects and correlated predictors require cautious interpretation.
- Prediction-only requests do not compute explanations. For explanations, cap background samples if a future general explainer is added and keep test data out of the reference background.
- Do not force CAD/vessel predictions into agreement or derive a CAD probability from a vessel average. Return a `label_disagreement` flag when the CAD binary label differs from the OR of the three vessel binary labels; this is a model consistency signal, not a medical finding.

API routes: `GET /health`, `GET /schema`, `POST /predict`. The request is `{"features": {...}, "include_explanations": false}`. Use a Pydantic outer request model and strict schema validation for its dynamic clinical feature object. Successful prediction status is 200; invalid input is 422; missing/incompatible artifacts prevent startup or produce unavailable health status, never plausible-looking fallback predictions. Bind locally to `127.0.0.1:8000` by default. No external publishing in this plan.

## 7. Review focus

1. Target leakage through aliases or row IDs: explicit feature allowlist and negative leakage tests (Tasks 1 and 2).
2. Missing or malformed patient fields: distinguish null, omission, unknown category, invalid numeric, and excessive missingness (Task 1 and Task 7).
3. Misleading probability explanations: verify raw-score additivity and separate calibrated probability (Task 6).
4. Wrong positive class, threshold, or feature order after serialization: round-trip equality and permuted-key tests (Tasks 5 and 7).
5. Selection bias and reproducibility: fixed test IDs, fold-local fitting, frozen metadata, and evaluation overwrite protection (Tasks 2 through 5).

## 8. Executable tasks

All commands below run from the project root in PowerShell with the project environment activated. Each implementation task first adds the specified meaningful checks, confirms the new checks fail, implements its interfaces, then reruns those checks. Avoid tests that only reproduce implementation expressions.

### Task 1: Audited data loading and strict feature schema

**Create:** `pyproject.toml`, `.gitignore`, `src/cardio_risk/__init__.py`, `data.py`, `schema.py`, `cli.py`, `tests/test_data.py`, `tests/test_schema.py`.

**Interfaces:**
- `load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]`: returns raw approved X, binary y keyed by four targets, and audit metadata.
- `build_schema(X_dev: pd.DataFrame) -> dict`: explicit feature groups and training ranges; mappings independent of outcomes.
- `validate_record(features: dict, schema: dict) -> tuple[pd.DataFrame, dict]`: canonical one-row frame plus input-quality flags; invalid input raises a structured validation exception.
- CLI `python -m cardio_risk.cli audit --data <path>` writes `data/processed/provenance.json`, `reports/audit.json`, and `reports/audit.md`; each training run copies those audits into its own report directory.

- [ ] Create `.venv` with Python 3.11/3.12 and package metadata; install the project editable with development dependencies. Record interpreter and resolved package versions. Add `httpx` for FastAPI TestClient if needed.
- [ ] Add failing `test_workbook_audit` asserting 303 rows, 55 inputs, four target mappings, no empty cells, expected target counts, zero duplicate rows, and exactly one CAD/vessel disagreement for the pinned hash.
- [ ] Add failing `test_target_fields_never_enter_features` and tests rejecting unknown target strings instead of mapping them to zero.
- [ ] Add failing schema tests for `Fmale` normalization, all approved groups covering 55 fields, unknown categories, omitted fields, accepted explicit nulls, 12-null rejection, string numeric rejection, nonfinite values, target keys, and extra keys.
- [ ] Run `pytest tests/test_data.py tests/test_schema.py -q`; confirm failures identify missing behavior.
- [ ] Implement the interfaces and `audit` subcommand. Include discrepancy row IDs and both labels in private audit output without changing source values.
- [ ] Run the same tests; require all pass. Run the audit CLI and write provenance and readable audit outputs.
- [ ] Commit this milestone if the project is already under Git.

**Acceptance:** Correct file/sheet loaded, all label encodings verified, source hash unchanged, audit findings reproducible, no trained estimator yet.

### Task 2: Immutable splits and leakage-safe preprocessing

**Create:** `preprocessing.py`, `configs/training.json`, `tests/test_training.py`. Extend `cli.py` with `split`.

**Interfaces:**
- `create_split(y: pd.DataFrame, dataset_hash: str) -> dict`: fixed IDs/counts/hash.
- `build_pipeline(family: str, params: dict, schema: dict) -> sklearn.pipeline.Pipeline`: unfitted estimator pipeline; expose underlying estimator through named step `model`.
- Put `create_split` in `data.py`; store split in the exact path from Section 3.

- [ ] Add failing split tests asserting 242/61 sizes, disjoint complete IDs, reproducibility, minimum class support, and rejection of a changed workbook hash.
- [ ] Add failing preprocessing test: change validation numeric values to extreme sentinels; fitted training imputation/scaling statistics must remain identical. A training-only constant feature must be excluded even when it varies in validation.
- [ ] Add failing category test confirming training category processing is reused at inference and `Region RWMA` is not treated as a spatial coordinate.
- [ ] Run `pytest tests/test_training.py -q`; verify failures.
- [ ] Implement fixed splitting, target-specific folds, the two unfitted pipelines, and the candidate configuration from Section 5.
- [ ] Run tests, then `python -m cardio_risk.cli split --data data/raw/z_alizadeh_sani_extension.xlsx`.
- [ ] Confirm split metadata and development-only schema are saved. Commit if Git exists.

**Acceptance:** Persisted partition and schema; no fitted preprocessing uses test rows.

### Task 3: Development-only baseline and candidate comparison

**Create:** `train.py`; extend `evaluate.py` with reusable metric calculation and `cli.py` with `train`.

**Interfaces:**
- `compute_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict` in `evaluate.py`.
- `run_search(X_dev: pd.DataFrame, y_dev: pd.DataFrame, schema: dict, config: dict) -> dict` in `train.py`: returns candidate scores, selected parameters, fold IDs, and development diagnostics.
- CLI `train --config configs/training.json --run-id <id>`; always load saved split and select development rows explicitly.

- [ ] Add failing known-fixture metric checks: perfect probabilities rank perfectly, reversed ranking has AUC zero, and no predicted positives yields finite metrics and a warning.
- [ ] Add failing fold membership test and fit-spy check ensuring no held-out row ID reaches candidate fit or transform fitting.
- [ ] Add failing selection test for the 0.01 tie rule and deterministic capacity tie-breaks.
- [ ] Run `pytest tests/test_training.py -q`; confirm failures.
- [ ] Implement dummy, six logistic, and eight CatBoost candidates per target, five development folds, saved metrics, and selected hyperparameters. Cap process parallelism to avoid nested CPU oversubscription.
- [ ] Run tests; execute the development search and inspect results for all four targets. Surface convergence warnings and unusual perfect scores for review rather than hiding them.
- [ ] Save development results with candidate parameters and seeds; commit implementation if Git exists.

**Acceptance:** Four selected parameter sets, documented dummy comparisons, no test performance available.

### Task 4: Calibration, threshold selection, and development fit

**Extend:** `train.py`, `tests/test_training.py`, `cli.py`.

**Interfaces:**
- `select_threshold(y: np.ndarray, probabilities: np.ndarray) -> float`.
- `fit_selected(X_dev: pd.DataFrame, y_dev: pd.DataFrame, selection: dict, schema: dict) -> dict`: four fitted estimators plus thresholds and selected calibration metadata.
- Returned target records contain `estimator`, `base_estimator`, `threshold`, `model_family`, and `calibration_method`; for uncalibrated models base and served estimator are the same. For sigmoid `ensemble=False`, retain the full-development refit underlying pipeline accessible from the calibrator.

- [ ] Add failing threshold tests for F1 selection, tie-breaking, and equality at the threshold.
- [ ] Add failing calibration-selection tests for the 0.005 Brier margin and log-loss condition; verify calibration training and OOF prediction memberships do not overlap within a fold.
- [ ] Run `pytest tests/test_training.py -q`; confirm failures.
- [ ] Implement uncalibrated and nested sigmoid OOF generation for selected parameters, branch selection, threshold selection, and fit on all development data.
- [ ] Verify every returned positive-class probability uses the index where `classes_ == 1`; do not assume class order silently.
- [ ] Save `development_oof.csv` and `selection.json`, marked as selection estimates. Commit if Git exists.

**Acceptance:** Complete four-target predictors with frozen development-chosen thresholds/calibration; test labels have not been scored.

### Task 5: Artifact bundle and one-time held-out evaluation

**Create/extend:** `predict.py`, `evaluate.py`, `tests/test_inference.py`, `cli.py`.

**Interfaces:**
- `save_bundle(bundle: dict, output_dir: Path, metadata: dict) -> Path`.
- `load_bundle(artifact_dir: Path) -> dict`: validates schema/hash/version compatibility and positive class.
- `predict_record(bundle: dict, features: dict, include_explanations: bool = False) -> dict`.
- `evaluate_holdout(bundle: dict, X_test: pd.DataFrame, y_test: pd.DataFrame, output_dir: Path) -> dict`.
- CLI `evaluate --run-id <id>` and `predict --run-id <id> --input <json>`.

- [ ] Add failing artifact round-trip test asserting four probabilities match before/after saving to tolerance `1e-10`; permuting request key order must not change output.
- [ ] Add failing missing/incompatible manifest tests and a test refusing a second evaluation into the same run directory unless results are being read without refitting.
- [ ] Run `pytest tests/test_inference.py -q`; confirm failures.
- [ ] Implement bundle serialization, manifest, selected feature names, training ranges, explicit class mappings, dependency versions, dataset/split hash, and model version. Load only trusted project-produced joblib artifacts; document its executable serialization format.
- [ ] Run round-trip checks. Freeze and hash the artifacts before evaluation.
- [ ] Execute final evaluation on 61 held-out cases; save required metrics, bootstrap intervals, plots, patient-level predictions, and the frozen artifact hash.
- [ ] Preserve this evaluation bundle as the version demonstrated by default. Do not refit on all 303 records and claim the same test scores for that refit.
- [ ] Commit code and model card changes if Git exists; generated artifact tracking follows repository policy.

**Acceptance:** Honest final performance exists for the exact saved bundle, with no score-dependent retuning. Any later run prompted by these results must disclose that this test set is no longer untouched for iterative model selection.

### Task 6: Correct patient and global explanations

**Create:** `explain.py`; extend `predict.py`, `tests/test_inference.py`.

**Interfaces:**
- `explain_record(target_record: dict, X_one: pd.DataFrame, background: dict, schema: dict) -> dict`.
- `summarize_global(bundle: dict, X_dev: pd.DataFrame) -> dict`.

- [ ] Add failing score-additivity tests for logistic and CatBoost paths to tolerance `1e-6`; original-field aggregation must preserve the sum.
- [ ] Add failing calibrated-record test ensuring `explained_output` identifies base log-odds, and final probability is taken from the served calibrated estimator rather than the sigmoid of the base score.
- [ ] Add failing missing-value explanation test checking imputed flags and original submitted values are preserved.
- [ ] Run targeted inference tests; confirm failures.
- [ ] Implement Section 6 explanation methods, top-five formatting, development-only backgrounds, and global summaries. Support either family winning any target.
- [ ] Run tests and manually inspect one synthetic record plus several development records with distinct label patterns. Do not fabricate percentages from SHAP magnitudes.
- [ ] Commit if Git exists.

**Acceptance:** Every explanation reconciles with the underlying score, and calibration is explicitly separated.

### Task 7: Validated local API and batch-free demo workflow

**Create:** `api.py`, `tests/test_api.py`, `examples/synthetic_patient.json`; extend CLI if needed.

**Interfaces:**
- `create_app(artifact_dir: Path) -> FastAPI`: load one frozen bundle at startup, not per request.
- Expose module-level `app` using `CARDIO_ARTIFACT_DIR` so the uvicorn command in Section 9 works; use application lifespan for artifact loading and fail clearly when the variable or bundle is missing.
- Routes and request/response contract follow Section 6; `GET /schema` returns allowed values, null policy, development ranges, and units.

- [ ] Add failing TestClient checks for healthy startup, four output keys, finite probabilities, saved thresholds, disclaimer, and optional explanations.
- [ ] Add failing 422 tests for an omitted feature, unknown category, extra/target key, numeric string, nonfinite number, and more than 11 nulls.
- [ ] Add failing test that missing artifacts prevent valid health/predictions; there must be no fallback random score.
- [ ] Run `pytest tests/test_api.py -q`; confirm failures.
- [ ] Implement validation error serialization, local routes, and a synthetic patient fixture with all 55 schema fields. Identify its values as synthetic and illustrative.
- [ ] Run tests and start `python -m uvicorn cardio_risk.api:app --host 127.0.0.1 --port 8000`, with the artifact directory supplied by `CARDIO_ARTIFACT_DIR`.
- [ ] Use PowerShell `Invoke-RestMethod` against `/health`, `/schema`, and `/predict` with the synthetic fixture; confirm responses match CLI inference.
- [ ] Measure warm prediction latency over 50 requests; report median/p95 with machine details. Aim for prediction-only p95 under 1 second but report observed results instead of treating the target as a claim.
- [ ] Commit if Git exists.

**Acceptance:** The future dashboard can obtain all four predictions and inspect the schema without knowing training internals.

### Task 8: Reproducibility and final ML handoff

**Create/complete:** `README.md`, `requirements.lock.txt`, `reports/<run_id>/model_card.md`, final figures and evaluation summary.

- [ ] Document environment setup, source attribution/license, all CLI commands, target definitions, preprocessing, split, candidate budgets, calibration rules, thresholds, explanation scales, and artifact version.
- [ ] Include actual test metrics and intervals, rare categories, the one label disagreement, limited sample size, unknown units, absent external validation, and lack of future-event labels.
- [ ] Document ML-to-3D mapping: response keys `lad`, `lcx`, `rca` map to artery mesh identifiers with continuous probability coloring; the API does not provide lesion coordinates or percentage narrowing. Overall CAD remains separate.
- [ ] Run `pytest -q` and a clean artifact reload plus CLI/API smoke check. Include executed command outputs in the completion report.
- [ ] Re-run only development training determinism checks when needed; do not create new opportunities to choose based on test scores. Record environment limitations if CPU-library nondeterminism exceeds documented tolerance.
- [ ] Confirm raw workbook hash unchanged, all four artifacts present, no target inputs, and final metrics refer to the frozen bundle actually served.
- [ ] Deliver links to source, artifacts, report, synthetic fixture, and setup commands. State any weak target performance or unverified checks plainly.
- [ ] Commit final documentation if Git exists.

**Acceptance:** Another developer can set up the environment, inspect honest results, load saved models, and run inference without reconstructing notebook state.

## 9. Intended execution commands

These commands describe the future workflow; they have NOT been executed. The implementer creates the CLI and flags above before these commands become runnable.

```powershell
Set-Location 'D:/Multi-model hackathon'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m cardio_risk.cli audit --data data/raw/z_alizadeh_sani_extension.xlsx
python -m cardio_risk.cli split --data data/raw/z_alizadeh_sani_extension.xlsx
python -m cardio_risk.cli train --config configs/training.json --run-id baseline-v1
python -m cardio_risk.cli evaluate --run-id baseline-v1
python -m cardio_risk.cli predict --run-id baseline-v1 --input examples/synthetic_patient.json
python -m pytest -q
$env:CARDIO_ARTIFACT_DIR = 'D:/Multi-model hackathon/artifacts/baseline-v1'
python -m uvicorn cardio_risk.api:app --host 127.0.0.1 --port 8000
```

If Python 3.12 is unavailable, use an available Python 3.11 interpreter with supported packages. If PowerShell activation policy prevents activation, invoke `.venv/Scripts/python.exe` directly; no system execution-policy change is required. Freeze versions after successful environment setup and meaningful checks.

## 10. Milestones and decision points

| Milestone | Tasks | Review evidence |
|---|---|---|
| Data ready | 1-2 | Audit, schema, immutable partition, leakage tests |
| Models selected | 3-4 | Dummy comparisons, development scores, frozen thresholds |
| ML evaluated | 5 | Exact saved bundle, test report, intervals and plots |
| Explainable inference | 6-7 | Additivity checks, validated CLI/API, synthetic demo |
| Reproducible handoff | 8 | Passing checks, model card, documented commands |

Expected work is a small CPU tabular ML project. Actual training time depends on the machine and installed package versions; no fixed completion estimate is guaranteed. Start with correctness and measured baselines, then investigate deficiencies rather than adding model complexity speculatively.

No automated accuracy target gates completion. Completion means the pipeline is correct, results are honest, limitations are explicit, and saved inference works. The complete hackathon submission still needs a separate dashboard/3D integration plan, up to six pages of project documentation, and a 3-10 minute demonstration video.

## 11. Primary references

- [UCI dataset and label description](https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset)
- [scikit-learn leakage and preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html): calibration folds must be independent of base-model fit samples; small-data isotonic overfitting supports starting with sigmoid.
- [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [CatBoost categorical processing](https://catboost.ai/docs/en/features/categorical-features)
- [CatBoost feature attribution API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_get_feature_importance)

## 12. Plan self-review

- [x] Every predictive-modeling requirement has a task: four labels, approved clinical inputs, target exclusion, classification metrics.
- [x] Pipeline, explanations, trained artifacts, documentation, and API integration are covered; browser/3D work is explicitly scoped separately.
- [x] Interfaces and filenames used by later tasks match earlier definitions.
- [x] Each review-focus failure mode has a specified owning test.
- [x] Development selection estimates are distinguished from held-out performance.
- [x] Explicit known data issues are preserved and reported.
- [x] No execution, installation, model training, or deployment occurred while preparing this plan.
