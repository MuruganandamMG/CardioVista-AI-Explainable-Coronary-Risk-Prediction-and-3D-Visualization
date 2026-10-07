# Benchmark 1 — Model performance

Report created: 2026-10-07 20:09:17 IST. Corrected to report model performance.

Model: `baseline-v1`. Evaluation: 61 held-out patients. Source: [saved evaluation metrics](../reports/baseline-v1/test_metrics.json). These are existing held-out results, not a new evaluation.

| Target | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CAD | 78.69% | 89.47% | 79.07% | 83.95% | 77.78% | 0.9134 |
| LAD | 72.13% | 67.35% | 97.06% | 79.52% | 40.74% | 0.8170 |
| LCX | 60.66% | 52.63% | 76.92% | 62.50% | 48.57% | 0.7022 |
| RCA | 62.30% | 45.95% | 85.00% | 59.65% | 51.22% | 0.7488 |

| Target | True positives | False positives | True negatives | False negatives | Threshold | Dummy accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CAD | 34 | 4 | 14 | 9 | 0.68 | 70.49% |
| LAD | 33 | 16 | 11 | 1 | 0.29 | 55.74% |
| LCX | 20 | 18 | 17 | 6 | 0.38 | 57.38% |
| RCA | 17 | 20 | 21 | 3 | 0.39 | 67.21% |

Accuracy measures all correct classifications; precision measures how many positive predictions are correct; recall measures how many actual positives are detected; F1 balances precision and recall; specificity measures actual negatives correctly rejected; ROC-AUC measures discrimination across thresholds. Dummy accuracy is the majority-class reference.

RCA accuracy is below the dummy baseline. Vessel predictions produce many false positives. Results are from a small internal test set, without external validation.
