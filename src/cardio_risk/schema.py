"""Explicit clinical feature boundary and strict inference validation."""

import math
from numbers import Real

import numpy as np
import pandas as pd

NUMERIC = ["Age", "Weight", "Length", "BMI", "BP", "PR", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT", "EF-TTE"]
BINARY = ["DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Edema", "Typical Chest Pain", "Q Wave", "St Elevation", "St Depression", "Tinversion"]
YN = ["Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP", "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur", "Dyspnea", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "LVH", "Poor R Progression"]
DOMAINS = {**{name: ["N", "Y"] for name in YN}, "Sex": ["Male", "Female"], "BBB": ["N", "LBBB", "RBBB"], "VHD": ["none", "mild", "moderate", "severe"], "Function Class": ["0", "1", "2", "3"], "Region RWMA": ["0", "1", "2", "3", "4"]}
APPROVED = set(NUMERIC + BINARY + list(DOMAINS))
DISCLAIMER = "For educational and decision-support purposes only. Not a substitute for diagnostic imaging or professional clinical assessment."


class InputError(ValueError):
    """Validation errors suitable for a 422 response."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__(str(errors))


def finite_number(value):
    try:
        return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def normalize_category(name, value):
    if pd.isna(value):
        return None
    if name in ("Function Class", "Region RWMA") and isinstance(value, Real) and not isinstance(value, bool):
        return str(int(value)) if finite_number(value) and value == int(value) else "__INVALID__"
    value = str(value).strip()
    if name == "Sex":
        return {"fmale": "Female", "female": "Female", "male": "Male"}.get(value.lower(), value)
    if name == "VHD":
        return "none" if value.lower() in ("n", "none") else value.lower()
    return value.upper()


def canonical_frame(frame):
    result = frame.copy()
    for name in DOMAINS:
        if name in result:
            result[name] = result[name].map(lambda value: normalize_category(name, value)).astype(object)
    return result


def build_schema(X_dev: pd.DataFrame) -> dict:
    if set(X_dev) != APPROVED:
        raise ValueError("Feature columns do not match the 55-field allowlist")
    X_dev = canonical_frame(X_dev)
    features = {}
    for name in X_dev:
        spec = {"type": "categorical" if name in DOMAINS else "binary" if name in BINARY else "numeric", "unit": None}
        if name in DOMAINS:
            spec["allowed_values"] = DOMAINS[name]
        else:
            values = pd.to_numeric(X_dev[name], errors="raise")
            spec["observed_min"] = float(values.min())
            spec["observed_max"] = float(values.max())
        features[name] = spec
    return {"schema_version": "1.0", "features": features, "max_missing": 11, "units_note": "Units await independent verification from dataset documentation; do not infer units from magnitude."}


def validate_record(features: dict, schema: dict) -> tuple[pd.DataFrame, dict]:
    expected = schema["features"]
    errors = []
    if not isinstance(features, dict):
        raise InputError([{"message": "features must be an object"}])
    for name in sorted(set(expected) - set(features)):
        errors.append({"field": name, "message": "Required field omitted; explicit null is accepted"})
    for name in sorted(set(features) - set(expected)):
        errors.append({"field": name, "message": "Unknown or forbidden feature"})
    missing = [name for name in expected if features.get(name) is None]
    if len(missing) > schema["max_missing"]:
        errors.append({"message": "More than 11 fields are missing"})
    values, out_of_range = {}, []
    for name, spec in expected.items():
        value = features.get(name)
        if value is None:
            values[name] = None if spec["type"] == "categorical" else np.nan
            continue
        if spec["type"] == "categorical":
            valid_type = isinstance(value, str) or (name in ("Function Class", "Region RWMA") and isinstance(value, Real) and not isinstance(value, bool))
            normalized = normalize_category(name, value) if valid_type else None
            if normalized not in spec["allowed_values"]:
                errors.append({"field": name, "message": "Invalid category", "allowed_values": spec["allowed_values"]})
            values[name] = normalized
        else:
            if not finite_number(value):
                errors.append({"field": name, "message": "Expected a finite number"})
                continue
            if spec["type"] == "binary" and value not in (0, 1):
                errors.append({"field": name, "message": "Expected 0 or 1"})
            values[name] = float(value)
            if not spec["observed_min"] <= value <= spec["observed_max"]:
                out_of_range.append(name)
    if errors:
        raise InputError(errors)
    return pd.DataFrame([values], columns=list(expected)), {"missing_features": missing, "imputed_features": missing, "out_of_range_features": out_of_range}
