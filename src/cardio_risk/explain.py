"""Additive underlying-score explanations, never causal claims."""

import numpy as np
from catboost import Pool
from scipy.special import expit


def score_contributions(record, X):
    pipeline = record["base_estimator"]
    model = pipeline.named_steps["model"]
    features = list(X)
    contributions = np.zeros((len(X), len(features)))
    if record["model_family"] == "catboost":
        transformed = pipeline[:-1].transform(X)
        shap = model.get_feature_importance(type="ShapValues", data=Pool(transformed, cat_features=model.get_param("cat_features")))
        for i, name in enumerate(transformed):
            contributions[:, features.index(name)] = shap[:, i]
        baseline = shap[:, -1]
        score = np.asarray(model.predict(transformed, prediction_type="RawFormulaVal"))
        method = "tree_shap"
    else:
        transformed = pipeline[:-1].transform(X)
        background = np.asarray(record["background"])
        coef = model.coef_[0]
        encoded = (transformed-background)*coef
        encoder = pipeline.named_steps["encoding"]
        names = list(encoder.transformers_[0][2])
        categorical_names = encoder.transformers_[1][2]
        if categorical_names:
            onehot = encoder.named_transformers_["categorical"]
            for name, categories in zip(categorical_names, onehot.categories_):
                names.extend([name]*len(categories))
        for i, name in enumerate(names):
            contributions[:, features.index(name)] += encoded[:, i]
        baseline = np.repeat(float(model.intercept_[0]+np.dot(coef, background)), len(X))
        score = np.asarray(pipeline.decision_function(X))
        method = "linear_score_contribution"
    if not np.allclose(baseline+contributions.sum(axis=1), score, atol=1e-6, rtol=0):
        raise ValueError("Explanation does not reconcile with the underlying model score")
    return contributions, baseline, score, method


def explain_record(target_record: dict, X_one, background: dict, schema: dict) -> dict:
    from .train import positive_probability
    contributions, baseline, score, method = score_contributions(target_record, X_one)
    fields = []
    for i, name in enumerate(X_one):
        value = X_one.iloc[0][name]
        value = None if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)) else value.item() if hasattr(value, "item") else value
        amount = float(contributions[0, i])
        fields.append({"feature": name, "value": value, "unit": schema["features"][name]["unit"], "contribution": amount, "direction": "increases_score" if amount > 0 else "decreases_score" if amount < 0 else "neutral"})
    return {"method": method, "explained_output": "base_model_log_odds", "base_value": float(baseline[0]), "model_score": float(score[0]), "score_probability": float(expit(score[0])), "final_probability": float(positive_probability(target_record["estimator"], X_one)[0]), "calibration_method": target_record["calibration_method"], "note": "Contributions explain the underlying model log-odds, not a calibrated probability or disease causation.", "top_features": sorted(fields, key=lambda field: abs(field["contribution"]), reverse=True)[:5], "all_contributions": fields}


def summarize_global(bundle: dict, X_dev) -> dict:
    result = {}
    for target, record in bundle["targets"].items():
        contributions, _, _, method = score_contributions(record, X_dev)
        result[target] = {"method": method, "population": "development", "scale": "base_model_log_odds", "features": sorted([{"feature": name, "mean_absolute_contribution": float(np.mean(np.abs(contributions[:, i])))} for i, name in enumerate(X_dev)], key=lambda field: field["mean_absolute_contribution"], reverse=True)}
    return result
