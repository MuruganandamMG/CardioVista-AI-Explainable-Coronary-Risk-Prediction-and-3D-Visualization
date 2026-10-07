"""Fold-local preprocessing for the two candidate model families."""

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import APPROVED, DOMAINS, canonical_frame


class ClinicalTransform(TransformerMixin, BaseEstimator):
    """Remove training constants and impute using training-only statistics."""

    def fit(self, X, y=None):
        if set(X) != APPROVED:
            raise ValueError("Forbidden or missing predictor columns")
        X = canonical_frame(X)
        self.columns_ = [name for name in X if X[name].nunique(dropna=False) > 1]
        self.numeric_ = [name for name in self.columns_ if name not in DOMAINS]
        self.categorical_ = [name for name in self.columns_ if name in DOMAINS]
        self.medians_ = X[self.numeric_].median().fillna(0).to_dict()
        self.fit_row_ids_ = list(X.index)
        return self

    def transform(self, X):
        if set(X) != APPROVED:
            raise ValueError("Forbidden or missing predictor columns")
        result = canonical_frame(X)[self.columns_].copy()
        for name in self.numeric_:
            result[name] = result[name].astype(float).fillna(self.medians_[name])
        for name in self.categorical_:
            result[name] = result[name].fillna("__MISSING__").astype(str)
        return result


def numeric_columns(X):
    return [name for name in X if name not in DOMAINS]


def categorical_columns(X):
    return [name for name in X if name in DOMAINS]


def build_pipeline(family: str, params: dict, schema: dict) -> Pipeline:
    if set(schema["features"]) != APPROVED:
        raise ValueError("Schema does not match the approved feature set")
    steps = [("clinical", ClinicalTransform())]
    if family == "logistic":
        steps.append(("encoding", ColumnTransformer([
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ])))
        model = LogisticRegression(max_iter=3000, solver="lbfgs", **params)
    elif family == "catboost":
        # Native CatBoost categorical names are the training-varying fields.
        model = ClinicalCatBoost(**{"verbose": False, "allow_writing_files": False, "thread_count": 4, **params})
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline([*steps, ("model", model)])


class ClinicalCatBoost(CatBoostClassifier):
    """Supply categorical names after training-only constant filtering."""

    def fit(self, X, y=None, **kwargs):
        self.set_params(cat_features=categorical_columns(X))
        return super().fit(X, y, **kwargs)
