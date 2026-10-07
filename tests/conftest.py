from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data():
    from cardio_risk.data import load_dataset, create_split
    from cardio_risk.schema import build_schema
    X, y, audit = load_dataset(Path("data/raw/z_alizadeh_sani_extension.xlsx"))
    split = create_split(y, audit["sha256"])
    ids = split["development"]
    return X.loc[ids], y.loc[ids], build_schema(X.loc[ids]), split


@pytest.fixture(params=["logistic", "catboost"], scope="module")
def fitted_bundle(request, data):
    from cardio_risk.preprocessing import build_pipeline
    from cardio_risk.train import calibrated_pipeline
    X, y, schema, split = data
    family = request.param
    params = {"iterations": 5, "depth": 2, "verbose": False, "allow_writing_files": False, "thread_count": 2} if family == "catboost" else {"C": .1}
    base = build_pipeline(family, params, schema)
    estimator = calibrated_pipeline(base).fit(X, y.cad)
    base = estimator.calibrated_classifiers_[0].estimator
    record = {"estimator": estimator, "base_estimator": base, "threshold": .45, "model_family": family, "calibration_method": "sigmoid", "development_prevalence": float(y.cad.mean())}
    if family == "logistic":
        record["background"] = base[:-1].transform(X).mean(axis=0).tolist()
    return {"targets": {target: dict(record) for target in y}, "schema": schema, "metadata": {"model_version": "test", "dataset_hash": split["sha256"], "split": split}}
