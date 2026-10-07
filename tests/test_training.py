from pathlib import Path
import numpy as np
import pytest


def development_data():
    from cardio_risk.data import load_dataset, create_split
    from cardio_risk.schema import build_schema
    X, y, audit = load_dataset(Path("data/raw/z_alizadeh_sani_extension.xlsx"))
    split = create_split(y, audit["sha256"])
    ids = split["development"]
    return X.loc[ids], y.loc[ids], build_schema(X.loc[ids])


def test_fixed_partition_and_hash_guard(tmp_path):
    from cardio_risk.data import load_dataset, create_split, write_json, load_split
    X, y, audit = load_dataset(Path("data/raw/z_alizadeh_sani_extension.xlsx"))
    split = create_split(y, audit["sha256"])
    assert len(split["development"]) == 242 and len(split["test"]) == 61
    assert set(split["development"]).isdisjoint(split["test"])
    assert sorted(split["development"] + split["test"]) == list(range(303))
    assert create_split(y, audit["sha256"]) == split
    write_json(tmp_path / "split.json", split)
    with pytest.raises(ValueError, match="hash"):
        load_split(tmp_path / "split.json", "different")


@pytest.mark.parametrize("family", ["logistic", "catboost"])
def test_pipeline_fits_training_only_and_reuses_categories(family):
    from cardio_risk.preprocessing import build_pipeline
    X, y, schema = development_data()
    train = X.iloc[:180].copy()
    train["CHF"] = "N"
    train.iloc[0, train.columns.get_loc("Weight")] = np.nan
    model = build_pipeline(family, {"iterations": 5, "depth": 2} if family == "catboost" else {}, schema)
    model.fit(train, y.cad.iloc[:180])
    validation = X.iloc[180:].copy()
    validation["CHF"] = "Y"
    validation["Weight"] = 100000
    learned = model.named_steps["clinical"]
    assert "CHF" not in learned.columns_
    assert learned.medians_["Weight"] < 200
    predictions = model.predict_proba(validation)
    assert predictions.shape == (len(validation), 2)
    assert np.isfinite(predictions).all()
    assert learned.medians_["Weight"] < 200


def test_metric_fixture():
    from cardio_risk.evaluate import compute_metrics
    y = np.array([0, 0, 1, 1])
    assert compute_metrics(y, np.array([.1, .2, .8, .9]), .5)["roc_auc"] == 1
    assert compute_metrics(y, np.array([.9, .8, .2, .1]), .5)["roc_auc"] == 0
    metrics = compute_metrics(y, np.array([.1, .1, .1, .1]), .5)
    assert metrics["precision"] == 0 and metrics["f1"] == 0
    assert metrics["warnings"] == ["No positive predictions"]


def test_simple_model_tie_selection():
    from cardio_risk.train import select_candidate
    candidates = [
        {"family": "catboost", "params": {"depth": 3, "iterations": 200}, "mean_roc_auc": .90, "mean_log_loss": .3},
        {"family": "logistic", "params": {"C": .1}, "mean_roc_auc": .895, "mean_log_loss": .4},
        {"family": "logistic", "params": {"C": 10}, "mean_roc_auc": .89, "mean_log_loss": .5},
    ]
    assert select_candidate(candidates)["params"] == {"C": .1}


def test_threshold_and_calibration_selection():
    from cardio_risk.train import select_threshold, prefer_calibration
    assert select_threshold(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == .5
    assert prefer_calibration({"brier_score": .2, "log_loss": .5}, {"brier_score": .194, "log_loss": .49})
    assert not prefer_calibration({"brier_score": .2, "log_loss": .5}, {"brier_score": .196, "log_loss": .49})
    assert not prefer_calibration({"brier_score": .2, "log_loss": .5}, {"brier_score": .19, "log_loss": .51})


def test_search_folds_exclude_holdout(data):
    from cardio_risk.train import run_search
    X, y, schema, split = data
    result = run_search(X, y[["cad"]], schema, {"logistic_C": [.1], "catboost_depth": [2], "catboost_iterations": [5]})
    assert result["selected"]["cad"]["family"] in ("logistic", "catboost")
    seen = []
    for fold in result["folds"]["cad"]:
        assert set(fold["train"]).isdisjoint(fold["validation"])
        assert set(fold["train"] + fold["validation"]) == set(split["development"])
        assert set(fold["train"] + fold["validation"]).isdisjoint(split["test"])
        seen.extend(fold["validation"])
    assert sorted(seen) == split["development"]


@pytest.mark.parametrize("damage", ["overlap", "duplicate", "missing", "seed", "counts", "string_id"])
def test_saved_partition_integrity(tmp_path, data, damage):
    import copy
    from cardio_risk.data import load_split, write_json
    _, _, _, good = data
    split = copy.deepcopy(good)
    if damage == "overlap": split["development"][0] = split["test"][0]
    if damage == "duplicate": split["development"][0] = split["development"][1]
    if damage == "missing": split["test"].pop()
    if damage == "seed": split["seed"] = 99
    if damage == "counts": split["class_counts"]["test"]["cad"]["1"] = 0
    if damage == "string_id": split["test"][0] = str(split["test"][0])
    write_json(tmp_path / "split.json", split)
    with pytest.raises(ValueError, match="partition"):
        load_split(tmp_path / "split.json", good["sha256"])


def test_fit_selected_trains_independent_targets_and_nested_subsets(data, monkeypatch):
    from cardio_risk.train import fit_selected
    from cardio_risk.preprocessing import ClinicalTransform
    X, y, schema, split = data
    memberships = []
    original_fit = ClinicalTransform.fit
    def record_fit(self, X_fit, y_fit=None):
        memberships.append(set(X_fit.index))
        return original_fit(self, X_fit, y_fit)
    monkeypatch.setattr(ClinicalTransform, "fit", record_fit)
    selection = {"selected": {target: {"family": "logistic", "params": {"C": .1}} for target in y}}
    fitted = fit_selected(X, y, selection, schema)
    assert len({id(record["estimator"]) for record in fitted["targets"].values()}) == 4
    assert not np.array_equal(fitted["targets"]["cad"]["base_estimator"].named_steps["model"].coef_, fitted["targets"]["lad"]["base_estimator"].named_steps["model"].coef_)
    assert all(ids <= set(split["development"]) for ids in memberships)
    assert all(ids.isdisjoint(split["test"]) for ids in memberships)
    assert any(len(ids) < 140 for ids in memberships)  # Inner calibration refits within outer training subsets.
    assert list(fitted["oof"].index) == split["development"]
