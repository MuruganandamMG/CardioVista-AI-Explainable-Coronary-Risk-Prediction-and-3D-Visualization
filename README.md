# CardioVista AI

Explainable coronary disease classification in Python. A saved bundle predicts four binary labels: overall CAD and LAD, LCX, and RCA stenosis. The local API supplies probabilities, selected thresholds, input-quality flags, and optional patient explanations for a future dashboard and 3D heart viewer.

## Dataset

The official [UCI Extension of Z-Alizadeh Sani dataset](https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset) is saved at `data/raw/z_alizadeh_sani_extension.xlsx`. It has 303 patients, 55 candidate clinical inputs, and four labels. Use worksheet `Sheet 1 - Table 1`.

Dataset creators: R. Alizadehsani, M. Roshanzamir, and Z. Sani. Dataset license: CC BY 4.0; DOI: 10.24432/C5461K. The dataset license does not automatically assign a license to this project's code.

`Cath`, `LAD`, `LCX`, and `RCA` are always excluded from predictors. The original file is preserved. The audit reports one inconsistent CAD/vessel label record without relabeling it; `Exertional CP` is constant, and training-only filters remove constant predictors. `Fmale` is normalized to `Female`. Extremely rare categories have limited supporting data.

## Setup on Windows

Use Python 3.11 or 3.12. From this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

The working project already has `.venv`. Activation is optional; invoking its Python directly avoids PowerShell execution-policy changes. The lockfile describes the tested environment. Serialized models require matching numerical-library versions and trusted local artifacts.

## Reproduce training

```powershell
.\.venv\Scripts\python.exe -m cardio_risk.cli audit
.\.venv\Scripts\python.exe -m cardio_risk.cli split
.\.venv\Scripts\python.exe -m cardio_risk.cli train --config configs/training.json --run-id baseline-v1
.\.venv\Scripts\python.exe -m cardio_risk.cli evaluate --run-id baseline-v1
```

The completed run lives in `artifacts/baseline-v1` and `reports/baseline-v1`. Training refuses to overwrite an existing run. Evaluation refuses to repeat into a run that has already begun evaluation. Read `test_metrics.json` to inspect existing results. An interrupted run is retained for diagnosis. If restarting training, use a new run ID and disclose reuse of the same test set after its results have been seen.

The shared fixed split uses 242 development and 61 held-out patients, stratified on CAD with seed 42. Development-only five-fold comparison evaluates a dummy prevalence baseline, six regularized logistic regression settings, and eight CatBoost settings per target. Numeric imputation, scaling, category encoding, and constant filtering are fitted independently within training folds.

Candidates are selected by ROC-AUC with a 0.01 tie preference for logistic regression, then log loss and lower capacity. Selected models compare raw probabilities with nested three-fold sigmoid calibration using development OOF predictions. Calibration is retained only for a Brier improvement of at least 0.005 without worse log loss. Per-target thresholds maximize development OOF F1, with recall and proximity to 0.5 resolving ties.

Development scores are model-selection estimates. Final held-out reports include accuracy, precision, recall, specificity, F1, ROC-AUC, average precision, Brier score, log loss, confusion matrices, and conditional bootstrap intervals. The evaluated bundle stays fitted on development data; it is not silently replaced with an all-data refit.

## Predict from a file

`examples/synthetic_patient.json` contains an illustrative synthetic record; it is not a copied patient or a diagnostic example.

```powershell
.\.venv\Scripts\python.exe -m cardio_risk.cli predict --run-id baseline-v1 --input examples/synthetic_patient.json
.\.venv\Scripts\python.exe -m cardio_risk.cli predict --run-id baseline-v1 --input examples/synthetic_patient.json --explain
```

In Python:

```python
import json
from pathlib import Path
from cardio_risk.predict import load_bundle, predict_record

bundle = load_bundle(Path("artifacts/baseline-v1"))
patient = json.loads(Path("examples/synthetic_patient.json").read_text())
result = predict_record(bundle, patient["features"], include_explanations=True)
print(result["predictions"])
```

All 55 feature keys must be provided using the schema's original column names. Explicit `null` marks a missing value; at most 11 nulls are accepted. Missing keys, unknown categories, target/extra keys, numeric strings, booleans as measurements, and nonfinite numbers are rejected. Finite measurements outside development ranges receive warnings and are not clipped. Numeric nulls use training medians and categorical nulls use a missing sentinel. Units remain unset until independently verified; do not infer lab units from values.

## Local API

```powershell
$env:CARDIO_ARTIFACT_DIR = (Resolve-Path 'artifacts/baseline-v1').Path
.\.venv\Scripts\python.exe -m uvicorn cardio_risk.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API explorer. Routes:

- `GET /health`: loaded version and artifact hash.
- `GET /schema`: required feature names, types, categories, missing-input policy, and development ranges.
- `POST /predict`: `{"features": {...}, "include_explanations": false}`.

From another PowerShell terminal:

```powershell
$patient = Get-Content 'examples/synthetic_patient.json' -Raw
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body $patient
```

Invalid requests return 422. Missing or incompatible model artifacts prevent startup. There are no fabricated fallback predictions, external services, or database requirements.

## Explanations and visualization handoff

CatBoost uses native tree SHAP. Logistic regression uses exact linear contributions relative to the mean transformed development vector. One-hot contributions are aggregated back to original fields, and every full contribution vector is checked for score additivity. Explanations contain the base score, model log-odds, underlying score probability, final served probability, and top five contributors. When calibration is active, contributions explain the underlying log-odds rather than the calibrated probability. Feature attribution does not establish causation.

Response keys `lad`, `lcx`, and `rca` should map to corresponding artery mesh IDs in the future 3D viewer. Use their probabilities for a documented continuous color scale. The dataset supplies vessel-level labels, not lesion coordinates. A probability of 0.8 is not 80% narrowing. Overall CAD is separately predicted; disagreement with vessel binary predictions is flagged rather than forced into agreement.

## Tests and project artifacts

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover target exclusion, label mapping, strict input validation, fixed partitions, fold-local preprocessing, model selection, probability metrics, artifact round trips, score explanations, and API responses. Test model fits use small development-only fixtures, not the final holdout evaluation.

Read [the implementation plan](docs/superpowers/plans/2026-10-07-cardiovascular-ml.md), [execution ledger](docs/implementation-progress.md), and the generated `reports/baseline-v1/model_card.md`. Generated binaries remain local under ignored `artifacts/`; reports contain the metrics and frozen hash. Source-patient prediction tables remain local and ignored.

## Intended use and limits

For educational and decision-support purposes only. Not a substitute for diagnostic imaging or professional clinical assessment.

This model predicts the dataset's current disease labels, not future heart attacks or mortality. Only 303 source records and 61 test records are available; confidence intervals can be broad and rare categories poorly estimated. Bootstrap intervals condition on fitted models and observed test class proportions and omit training/model-selection uncertainty. There is no external validation, and performance with missing-input patterns has not been clinically validated.

The ML backend is the current deliverable. The browser dashboard, interactive 3D viewer, and complete hackathon submission video require subsequent development.
