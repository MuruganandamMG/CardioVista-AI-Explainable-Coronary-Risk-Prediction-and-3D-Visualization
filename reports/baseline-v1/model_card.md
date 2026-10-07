# CardioVista AI model card

Model version: baseline-v1. Development: 242 patients; held-out test: 61 patients.

Frozen bundle SHA-256: `4cf2e286ef9b8f10c4a414a2f9f68f98ee525dcc7a5e9b954b672a64faf8bba7`

| Target | Model | Accuracy | Recall | Specificity | F1 | ROC-AUC (95% CI) |
|---|---|---:|---:|---:|---:|---|
| CAD | logistic | 0.787 | 0.791 | 0.778 | 0.840 | 0.913 (0.827-0.981) |
| LAD | logistic | 0.721 | 0.971 | 0.407 | 0.795 | 0.817 (0.705-0.908) |
| LCX | catboost | 0.607 | 0.769 | 0.486 | 0.625 | 0.702 (0.563-0.833) |
| RCA | catboost | 0.623 | 0.850 | 0.512 | 0.596 | 0.749 (0.613-0.863) |

## Interpretation and limitations

Selected decision thresholds: CAD 0.68, LAD 0.29, LCX 0.38, RCA 0.39. All models retained raw probabilities because sigmoid calibration did not meet the predeclared development-only improvement criterion.

The vessel thresholds favor positive-case recall and generate many false positives: specificity is 40.7% for LAD, 48.6% for LCX, and 51.2% for RCA. RCA accuracy (62.3%) is below its majority-class dummy baseline (67.2%), despite better ROC-AUC and probability losses. Thresholds are experimental F1 operating points, not clinical standards. No post-test retuning was performed.

Predicts the dataset's current disease labels, not future events or percentage narrowing. All four target columns are excluded from inputs. Source labels are preserved, including one CAD/vessel disagreement. Development cross-validation scores and tuned-threshold scores are selection estimates; only this evaluation is held out.

Intervals are conditional on the fitted model and observed test class proportions; they omit training/selection uncertainty. The sample is small, rare categories are poorly supported, and there is no external validation. Units await verified documentation. Explicit missing inputs are imputed with training medians or categorical sentinels; performance under missing-input patterns has not been clinically validated.

CatBoost explanations use native tree SHAP. Logistic explanations use exact linear contributions relative to the transformed development mean. Both explain base log-odds; calibration probabilities are shown separately. Neither method establishes causation.

Data source: UCI dataset 411, Alizadehsani/Roshanzamir/Sani; CC BY 4.0. Model predictions are for educational/decision-support purposes and do not replace diagnostic imaging.

## Reproducibility and serving verification

Saved artifact reload succeeded with the locked Python 3.12 environment. Reconstructing all four estimators from their saved selected parameters and development records reproduced probabilities exactly (maximum absolute difference 0.0). The live HTTP API matched direct Python inference including full explanation vectors. Explanation additivity errors were at most 8.33e-17 on the synthetic example.

Fifty warm local HTTP predictions measured 60.16 ms median and 83.32 ms p95 on an Intel Core i7-13620H, 16 logical processors. These timings describe this local run, not a production service guarantee. Source workbook hash was unchanged. Detailed verification is in `verification.json`.
