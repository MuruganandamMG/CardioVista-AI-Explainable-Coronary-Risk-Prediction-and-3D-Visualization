"""Development-only candidate comparison, calibration, and fitting."""

import itertools

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from threadpoolctl import threadpool_limits

from .evaluate import compute_metrics
from .preprocessing import build_pipeline


def positive_probability(estimator, X):
    classes = list(estimator.classes_)
    if classes != [0, 1]:
        raise ValueError("Expected binary classes 0 and 1")
    return estimator.predict_proba(X)[:, classes.index(1)]


def select_candidate(candidates):
    scored = [row for row in candidates if row["family"] != "dummy"]
    best = max(row["mean_roc_auc"] for row in scored)
    tied = [row for row in scored if row["mean_roc_auc"] >= best - .01 - 1e-12]
    if any(row["family"] == "logistic" for row in tied):
        tied = [row for row in tied if row["family"] == "logistic"]
    return min(tied, key=lambda row: (row["mean_log_loss"], row["params"].get("C", row["params"].get("depth", 0)), row["params"].get("iterations", 0)))


def candidates(config):
    yield "dummy", {}
    for C, weight in itertools.product(config["logistic_C"], [None, "balanced"]):
        yield "logistic", {"C": C, "class_weight": weight}
    for depth, iterations, weight in itertools.product(config["catboost_depth"], config["catboost_iterations"], [None, "Balanced"]):
        params = {"depth": depth, "iterations": iterations, "learning_rate": .03, "l2_leaf_reg": 5, "loss_function": "Logloss", "random_seed": 42, "thread_count": 4, "verbose": False, "allow_writing_files": False}
        if weight:
            params["auto_class_weights"] = weight
        yield "catboost", params


def run_search(X_dev: pd.DataFrame, y_dev: pd.DataFrame, schema: dict, config: dict) -> dict:
    if not X_dev.index.equals(y_dev.index):
        raise ValueError("Feature/label row IDs differ")
    all_scores, selected, folds = [], {}, {}
    with threadpool_limits(limits=4):
        for target in y_dev:
            labels = y_dev[target]
            cv = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X_dev, labels))
            folds[target] = [{"train": X_dev.index[train].tolist(), "validation": X_dev.index[validation].tolist()} for train, validation in cv]
            scores = []
            for family, params in candidates(config):
                results = []
                for train, validation in cv:
                    if family == "dummy":
                        model = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), labels.iloc[train])
                        p = positive_probability(model, np.zeros((len(validation), 1)))
                    else:
                        model = build_pipeline(family, params, schema)
                        model.fit(X_dev.iloc[train], labels.iloc[train])
                        p = positive_probability(model, X_dev.iloc[validation])
                    results.append(compute_metrics(labels.iloc[validation].to_numpy(), p, .5))
                row = {"target": target, "family": family, "params": params, "fold_metrics": results}
                for metric in ("roc_auc", "log_loss", "accuracy", "recall", "f1", "brier_score", "average_precision"):
                    values = [result[metric] for result in results]
                    row[f"mean_{metric}"] = float(np.mean(values))
                    row[f"std_{metric}"] = float(np.std(values, ddof=1))
                scores.append(row)
                print(f"{target}: {family} AUC={row['mean_roc_auc']:.3f}", flush=True)
            selected[target] = select_candidate(scores)
            all_scores.extend(scores)
    return {"candidates": all_scores, "selected": selected, "folds": folds}


def select_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    rows = [(threshold, compute_metrics(y, probabilities, threshold)) for threshold in np.round(np.arange(.05, .951, .01), 2)]
    return float(max(rows, key=lambda item: (item[1]["f1"], item[1]["recall"], -abs(item[0]-.5), -item[0]))[0])


def prefer_calibration(raw, calibrated):
    return calibrated["brier_score"] <= raw["brier_score"] - .005 and calibrated["log_loss"] <= raw["log_loss"]


def calibrated_pipeline(pipeline):
    return CalibratedClassifierCV(pipeline, method="sigmoid", cv=StratifiedKFold(3, shuffle=True, random_state=42), ensemble=False, n_jobs=1)


def fit_selected(X_dev: pd.DataFrame, y_dev: pd.DataFrame, selection: dict, schema: dict) -> dict:
    records, oof = {}, pd.DataFrame(index=X_dev.index)
    with threadpool_limits(limits=4):
        for target, candidate in selection["selected"].items():
            family, params = candidate["family"], candidate["params"]
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            pipeline = build_pipeline(family, params, schema)
            raw_p = cross_val_predict(pipeline, X_dev, y_dev[target], cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            cal_p = cross_val_predict(calibrated_pipeline(build_pipeline(family, params, schema)), X_dev, y_dev[target], cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            raw_metrics = compute_metrics(y_dev[target].to_numpy(), raw_p, .5)
            cal_metrics = compute_metrics(y_dev[target].to_numpy(), cal_p, .5)
            calibrated = prefer_calibration(raw_metrics, cal_metrics)
            p = cal_p if calibrated else raw_p
            threshold = select_threshold(y_dev[target].to_numpy(), p)
            estimator = calibrated_pipeline(pipeline) if calibrated else pipeline
            estimator.fit(X_dev, y_dev[target])
            base = estimator.calibrated_classifiers_[0].estimator if calibrated else estimator
            records[target] = {"estimator": estimator, "base_estimator": base, "threshold": threshold, "model_family": family, "calibration_method": "sigmoid" if calibrated else "none", "params": params, "development_metrics": compute_metrics(y_dev[target].to_numpy(), p, threshold), "calibration_comparison": {"raw": raw_metrics, "sigmoid": cal_metrics}, "development_prevalence": float(y_dev[target].mean())}
            if family == "logistic":
                records[target]["background"] = base[:-1].transform(X_dev).mean(axis=0).tolist()
            oof[f"{target}_label"] = y_dev[target]
            oof[f"{target}_probability"] = p
            print(f"Selected {target}: {family}, calibration={records[target]['calibration_method']}, threshold={threshold:.2f}", flush=True)
    return {"targets": records, "schema": schema, "oof": oof}
