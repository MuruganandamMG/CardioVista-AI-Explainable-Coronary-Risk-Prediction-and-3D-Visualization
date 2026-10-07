"""Trusted local artifacts and consistent four-target inference."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np

from .data import TARGET_COLUMNS, write_json
from .schema import DISCLAIMER, validate_record
from .train import positive_probability

RUNTIME_PACKAGES = ["numpy", "pandas", "scikit-learn", "catboost", "joblib"]


def save_bundle(bundle: dict, output_dir: Path, metadata: dict) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle["metadata"] = {**metadata, "runtime_versions": {name: version(name) for name in RUNTIME_PACKAGES}}
    path = output_dir / "bundle.joblib"
    joblib.dump(bundle, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version": bundle["schema"]["schema_version"], "bundle_sha256": digest, **bundle["metadata"], "targets": list(TARGET_COLUMNS)}
    write_json(output_dir / "schema.json", bundle["schema"])
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "explanation_background.json", {target: record.get("background") for target, record in bundle["targets"].items()})
    bundle["artifact_hash"] = digest
    return path


def load_bundle(artifact_dir: Path) -> dict:
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != "1.0" or manifest["targets"] != list(TARGET_COLUMNS):
        raise ValueError("Incompatible artifact manifest")
    path = artifact_dir / "bundle.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("Artifact hash mismatch")
    for name, saved_version in manifest["runtime_versions"].items():
        if version(name) != saved_version:
            raise ValueError(f"Artifact requires {name}=={saved_version}; use requirements.lock.txt")
    # joblib is executable serialization: only load trusted project-produced files.
    bundle = joblib.load(path)
    schema = json.loads((artifact_dir / "schema.json").read_text(encoding="utf-8"))
    if schema != bundle["schema"] or bundle["metadata"]["model_version"] != manifest["model_version"]:
        raise ValueError("Bundle/schema metadata mismatch")
    if list(bundle["targets"]) != list(TARGET_COLUMNS):
        raise ValueError("Missing target models")
    bundle["artifact_hash"] = manifest["bundle_sha256"]
    return bundle


def predict_record(bundle: dict, features: dict, include_explanations: bool = False) -> dict:
    frame, flags = validate_record(features, bundle["schema"])
    predictions, explanations = {}, {}
    for target, record in bundle["targets"].items():
        p = float(positive_probability(record["estimator"], frame)[0])
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Model returned invalid probability")
        predictions[target] = {"probability": p, "label": int(p >= record["threshold"]), "threshold": record["threshold"], "positive_class": "CAD" if target == "cad" else "Stenotic", "model_family": record["model_family"], "calibration_method": record["calibration_method"]}
        if include_explanations:
            from .explain import explain_record
            explanations[target] = explain_record(record, frame, {}, bundle["schema"])
    flags["label_disagreement"] = predictions["cad"]["label"] != int(any(predictions[name]["label"] for name in ("lad", "lcx", "rca")))
    return {"schema_version": bundle["schema"]["schema_version"], "model_version": bundle["metadata"]["model_version"], "predictions": predictions, "input_quality": flags, "explanations": explanations, "disclaimer": DISCLAIMER}
