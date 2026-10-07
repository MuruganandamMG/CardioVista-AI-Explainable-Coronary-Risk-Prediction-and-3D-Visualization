"""Probability and classification metrics; held-out reporting."""

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(y, predicted)), "precision": float(precision_score(y, predicted, zero_division=0)), "recall": float(recall_score(y, predicted, zero_division=0)), "specificity": float(tn / (tn + fp)) if tn + fp else 0.0, "f1": float(f1_score(y, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, "average_precision": float(average_precision_score(y, p)) if y.sum() else 0.0, "brier_score": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "positive_support": int(y.sum()), "negative_support": int(len(y)-y.sum()), "threshold": float(threshold), "warnings": [] if predicted.any() else ["No positive predictions"]}


def bootstrap_intervals(y, p, threshold, repetitions=2000):
    rng = np.random.default_rng(42)
    names = ["roc_auc", "recall", "specificity", "f1", "brier_score"]
    samples = {name: [] for name in names}
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    for _ in range(repetitions):
        ids = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        metrics = compute_metrics(y[ids], p[ids], threshold)
        for name in names:
            samples[name].append(metrics[name])
    return {name: np.quantile(values, [.025, .975]).tolist() for name, values in samples.items()}


def plot_target(y, p, metrics, target, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    matrix = np.asarray(metrics["confusion_matrix"])
    axes[0, 0].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2): axes[0, 0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[0, 0].set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label", title="Confusion matrix")
    fpr, tpr, _ = roc_curve(y, p)
    axes[0, 1].plot(fpr, tpr, label=f"AUC {metrics['roc_auc']:.3f}")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[0, 1].legend()
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1, 0].plot(recall, precision)
    axes[1, 0].axhline(y.mean(), linestyle="--", color="gray")
    axes[1, 0].set(xlabel="Recall", ylabel="Precision", title=f"PR: AP {metrics['average_precision']:.3f}")
    bins = np.minimum((p*5).astype(int), 4)
    axes[1, 1].plot([0, 1], [0, 1], "--", color="gray")
    empty = []
    for index in range(5):
        mask = bins == index
        if mask.any():
            xp, yp = p[mask].mean(), y[mask].mean()
            axes[1, 1].scatter([xp], [yp])
            axes[1, 1].annotate(f"n={mask.sum()}", (xp, yp), xytext=(-6 if xp > .85 else 4, -14 if yp > .85 else 6), ha="right" if xp > .85 else "left", textcoords="offset points")
        else: empty.append(index+1)
    axes[1, 1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed positive frequency", title=f"Reliability (5 bins); empty: {empty}")
    fig.suptitle(f"{target.upper()}: held-out evaluation, n={len(y)}")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def evaluate_holdout(bundle: dict, X_test, y_test, output_dir) -> dict:
    from pathlib import Path
    import pandas as pd
    from .data import write_json
    from .train import positive_probability
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "test_metrics.json").exists() or (output_dir / ".evaluation_started").exists():
        raise FileExistsError("This run has already begun evaluation; read existing results")
    ids = bundle["metadata"]["split"]["test"]
    if list(X_test.index) != ids or not X_test.index.equals(y_test.index):
        raise ValueError("Evaluation inputs do not match the frozen held-out partition")
    with (output_dir / ".evaluation_started").open("x") as marker:
        marker.write("Frozen models only; do not use this test set for subsequent selection.\n")
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    results = {"population": "untouched test", "records": len(X_test), "artifact_hash": bundle.get("artifact_hash"), "interval_method": "2000 class-stratified patient bootstraps; percentile 95%; conditional on fitted model and observed proportions", "targets": {}}
    predictions = pd.DataFrame(index=X_test.index)
    for target, record in bundle["targets"].items():
        y = y_test[target].to_numpy()
        p = positive_probability(record["estimator"], X_test)
        metrics = compute_metrics(y, p, record["threshold"])
        metrics["model_family"] = record["model_family"]
        metrics["calibration_method"] = record["calibration_method"]
        metrics["dummy_metrics"] = compute_metrics(y, np.repeat(record["development_prevalence"], len(y)), .5)
        print(f"Evaluating {target}: AUC={metrics['roc_auc']:.3f}; computing intervals", flush=True)
        metrics["confidence_intervals_95"] = bootstrap_intervals(y, p, record["threshold"])
        results["targets"][target] = metrics
        predictions[f"{target}_label"] = y
        predictions[f"{target}_probability"] = p
        plot_target(y, p, metrics, target, figures / f"{target}_evaluation.png")
    names = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "average_precision", "brier_score", "log_loss"]
    results["macro"] = {name: float(np.mean([record[name] for record in results["targets"].values()])) for name in names}
    predictions.to_csv(output_dir / "test_predictions.csv", index_label="row_id")
    write_json(output_dir / "test_metrics.json", results)
    lines = ["# CardioVista AI model card", "", f"Model version: {bundle['metadata']['model_version']}. Development: 242 patients; held-out test: {len(X_test)} patients.", "", f"Frozen bundle SHA-256: `{results['artifact_hash']}`", "", "| Target | Model | Accuracy | Recall | Specificity | F1 | ROC-AUC (95% CI) |", "|---|---|---:|---:|---:|---:|---|"]
    for target, record in results["targets"].items():
        low, high = record["confidence_intervals_95"]["roc_auc"]
        lines.append(f"| {target.upper()} | {record['model_family']} | {record['accuracy']:.3f} | {record['recall']:.3f} | {record['specificity']:.3f} | {record['f1']:.3f} | {record['roc_auc']:.3f} ({low:.3f}-{high:.3f}) |")
    lines.extend(["", "## Interpretation and limitations", "", "Predicts the dataset's current disease labels, not future events or percentage narrowing. All four target columns are excluded from inputs. Source labels are preserved, including one CAD/vessel disagreement. Development cross-validation scores and tuned-threshold scores are selection estimates; only this evaluation is held out.", "", "Intervals are conditional on the fitted model and observed test class proportions; they omit training/selection uncertainty. The sample is small, rare categories are poorly supported, and there is no external validation. Units await verified documentation. Explicit missing inputs are imputed with training medians or categorical sentinels; performance under missing-input patterns has not been clinically validated.", "", "CatBoost explanations use native tree SHAP. Logistic explanations use exact linear contributions relative to the transformed development mean. Both explain base log-odds; calibration probabilities are shown separately. Neither method establishes causation.", "", "Data source: UCI dataset 411, Alizadehsani/Roshanzamir/Sani; CC BY 4.0. Model predictions are for educational/decision-support purposes and do not replace diagnostic imaging."])
    (output_dir / "model_card.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return results
