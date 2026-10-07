# CardioVista AI — How to start

This guide runs the saved Python ML models and the local prediction API on Windows. Use **PowerShell**. The current project includes a working `.venv` and the trained `baseline-v1` models, so start with the quick start below. The backend and API explorer are available; a dashboard and 3D heart viewer are future work.

## 1. Quick start: run the saved ML models

Open PowerShell and run:

```powershell
Set-Location 'D:\Multi-model hackathon'
.\.venv\Scripts\python.exe -m cardio_risk.cli predict --run-id baseline-v1 --input examples/synthetic_patient.json
```

This prints JSON predictions for `cad`, `lad`, `lcx`, and `rca`, including each probability, decision threshold, and predicted label. The input is a synthetic example, not a source patient record. You do not need to retrain before predicting.

To include explanations:

```powershell
.\.venv\Scripts\python.exe -m cardio_risk.cli predict --run-id baseline-v1 --input examples/synthetic_patient.json --explain
```

The explanations describe contributions to the underlying model score; they do not establish causes. Probabilities describe disease labels, not the percentage of artery narrowing or future heart-attack risk.

## 2. Start the API

In a PowerShell terminal:

```powershell
Set-Location 'D:\Multi-model hackathon'
$env:CARDIO_ARTIFACT_DIR = (Resolve-Path 'artifacts/baseline-v1').Path
.\.venv\Scripts\python.exe -m uvicorn cardio_risk.api:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. Wait for `Application startup complete`, then open:

- API explorer: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>
- Input schema: <http://127.0.0.1:8000/schema>

In the API explorer, expand `POST /predict`, click **Try it out**, paste the complete contents of `examples/synthetic_patient.json`, and click **Execute**. Set `include_explanations` to `true` in that JSON to request explanations.

Press **Ctrl+C** in the server terminal to stop it. Run the same startup commands to restart it. Set `CARDIO_ARTIFACT_DIR` in each new server terminal before launching Uvicorn.

### Call the API from a second terminal

Leave the server running and open another PowerShell terminal:

```powershell
Set-Location 'D:\Multi-model hackathon'
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health'

$body = Get-Content 'examples/synthetic_patient.json' -Raw
$result = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body $body
$result | ConvertTo-Json -Depth 20
```

To request explanations:

```powershell
$patient = Get-Content 'examples/synthetic_patient.json' -Raw | ConvertFrom-Json
$patient.include_explanations = $true
$body = $patient | ConvertTo-Json -Depth 10
$result = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/predict' -Method Post -ContentType 'application/json' -Body $body
$result | ConvertTo-Json -Depth 20
```

## 3. First-time setup on another machine

Install **Python 3.12** and open PowerShell in your project folder. The following assumes the Python launcher `py` is installed:

```powershell
Set-Location 'D:\Multi-model hackathon'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pip check
```

Adjust the project path if your checkout is elsewhere. If `py` is unavailable, use your installed Python 3.12 executable to create the environment:

```powershell
& 'C:\path\to\Python312\python.exe' -m venv .venv
```

Replace that placeholder with the actual executable path. Activation is optional: all commands in this guide invoke the virtual environment's Python directly.

**Trained artifacts are excluded from Git.** A fresh clone alone does not contain the saved models. Copy the entire `artifacts/baseline-v1` folder from this trusted project, or train a new run using section 6. The complete folder includes `bundle.joblib`, `manifest.json`, `schema.json`, and `explanation_background.json`. Only load trusted model artifacts, and use the matching locked dependencies.

## 4. Prepare your own input

Make a separate synthetic input file:

```powershell
Copy-Item 'examples/synthetic_patient.json' 'examples/my_patient.json'
```

Edit the copy, preserving the request structure and exact feature names. Then run:

```powershell
.\.venv\Scripts\python.exe -m cardio_risk.cli predict --run-id baseline-v1 --input examples/my_patient.json --explain
```

Input requirements:

- Include all 55 feature keys inside `features`. See `GET /schema` or `artifacts/baseline-v1/schema.json` for allowed values.
- Use JSON numbers for numeric features, including numeric binary `0`/`1`; do not use quoted numbers or booleans.
- Use the allowed categorical values exactly. Some numeric-looking category codes are categorical fields; follow the schema.
- Use `null` for an unavailable value. At most 11 feature values may be missing; omitting a key is invalid.
- Exclude target columns (`Cath`, `LAD`, `LCX`, `RCA`) and any extra feature keys.
- Keep examples and demos synthetic; do not copy patient rows from the workbook.

Numeric values outside the observed development ranges produce quality warnings. Review these warnings alongside the predictions.

## 5. Run inference inside Python

Save this as a Python script in the project root and run it with `.\.venv\Scripts\python.exe your_script.py`:

```python
import json
from pathlib import Path

from cardio_risk.predict import load_bundle, predict_record

bundle = load_bundle(Path("artifacts/baseline-v1"))
patient = json.loads(
    Path("examples/synthetic_patient.json").read_text(encoding="utf-8")
)
result = predict_record(bundle, patient["features"], include_explanations=True)
print(json.dumps(result, indent=2))
```

Load the bundle once and reuse it for subsequent predictions, as the API does.

## 6. Train a new model run (optional)

Training uses the saved dataset, configuration, and fixed development/test split. It runs on CPU and can take many minutes. Choose a **new run ID** every time; existing artifact/report directories cannot be overwritten.

```powershell
Set-Location 'D:\Multi-model hackathon'
$runId = 'experiment-v2'

# These checks should both return False before starting a new run.
Test-Path "artifacts/$runId"
Test-Path "reports/$runId"

.\.venv\Scripts\python.exe -m cardio_risk.cli audit
.\.venv\Scripts\python.exe -m cardio_risk.cli split
.\.venv\Scripts\python.exe -m cardio_risk.cli train --config configs/training.json --run-id $runId
```

If either path already exists, change `$runId` before training. Proceed only after training completes successfully. The `split` command reuses and validates the persisted partition rather than generating a new one.

For the new run's final evaluation and prediction:

```powershell
.\.venv\Scripts\python.exe -m cardio_risk.cli evaluate --run-id $runId
.\.venv\Scripts\python.exe -m cardio_risk.cli predict --run-id $runId --input examples/synthetic_patient.json --explain
```

Evaluation is allowed once per run. `baseline-v1` has already been evaluated; read its reports instead of trying to evaluate it again. Reusing the same test set after viewing its results does not provide a fresh independent validation set.

To serve the new run, stop the existing server with Ctrl+C, then:

```powershell
$env:CARDIO_ARTIFACT_DIR = (Resolve-Path "artifacts/$runId").Path
.\.venv\Scripts\python.exe -m uvicorn cardio_risk.api:app --host 127.0.0.1 --port 8000
```

## 7. Read results and run checks

```powershell
Get-Content 'reports/baseline-v1/model_card.md'
Get-Content 'reports/baseline-v1/test_metrics.json'
.\.venv\Scripts\python.exe -m pytest -q
```

| Location | Contents |
| --- | --- |
| `configs/training.json` | Training and model-selection settings |
| `data/raw/z_alizadeh_sani_extension.xlsx` | Original dataset; preserve unchanged |
| `data/processed/split.json` | Fixed development/test partition |
| `artifacts/<run-id>/` | Saved models, schema, and compatibility metadata |
| `reports/<run-id>/model_card.md` | Evaluation summary and limitations |
| `reports/<run-id>/test_metrics.json` | Detailed held-out metrics |
| `reports/<run-id>/figures/` | Evaluation plots |

## 8. Troubleshooting

| Problem | What to do |
| --- | --- |
| `.venv` Python not found | Run first-time setup with Python 3.12. |
| `No module named cardio_risk` | From the project root, run `.\.venv\Scripts\python.exe -m pip install --no-deps -e .`. |
| Model folder missing | Copy the complete trusted artifact folder or train a new run. Check the run ID and current directory. |
| Library compatibility error | Use Python 3.12 and install `requirements.lock.txt` in the project environment. |
| Port 8000 already in use | Reuse the running server, stop it in its terminal, or launch with `--port 8001` and change the URLs to port 8001. |
| Connection refused | Start the API, keep its terminal open, and check its startup log and port. |
| HTTP 422 / invalid input | Read the response details; check all 55 keys, allowed categories, numeric types, and missing-value limits. |
| Run already exists / evaluation already started | Choose a new training run ID. Read existing evaluation reports; preserve the existing run for diagnosis. |
| PowerShell blocks environment activation | Invoke `.\.venv\Scripts\python.exe` directly, as shown above. |

> For educational and decision-support purposes only. Not a substitute for diagnostic imaging or professional clinical assessment.
