import json
import numpy as np
import pytest


def test_roundtrip_and_key_order(tmp_path, fitted_bundle, data):
    from cardio_risk.predict import save_bundle, load_bundle, predict_record
    X, _, _, _ = data
    features = X.iloc[0].to_dict()
    before = predict_record(fitted_bundle, features)
    save_bundle(fitted_bundle, tmp_path / "bundle", fitted_bundle["metadata"])
    loaded = load_bundle(tmp_path / "bundle")
    after = predict_record(loaded, dict(reversed(list(features.items()))))
    for target in ("cad", "lad", "lcx", "rca"):
        assert before["predictions"][target]["probability"] == pytest.approx(after["predictions"][target]["probability"], abs=1e-10)
        assert after["predictions"][target]["threshold"] == .45
    assert after["explanations"] == {}


def test_manifest_tampering_and_missing_artifact(tmp_path, fitted_bundle):
    from cardio_risk.predict import save_bundle, load_bundle
    with pytest.raises((FileNotFoundError, ValueError)): load_bundle(tmp_path / "absent")
    save_bundle(fitted_bundle, tmp_path / "bundle", fitted_bundle["metadata"])
    manifest_file = tmp_path / "bundle" / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["schema_version"] = "bad"
    manifest_file.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): load_bundle(tmp_path / "bundle")


def test_score_explanations_and_missing_flag(fitted_bundle, data):
    from cardio_risk.predict import predict_record
    X, _, _, _ = data
    features = X.iloc[0].to_dict()
    features["Weight"] = None
    result = predict_record(fitted_bundle, features, include_explanations=True)
    assert result["input_quality"]["imputed_features"] == ["Weight"]
    for target, explanation in result["explanations"].items():
        total = explanation["base_value"] + sum(item["contribution"] for item in explanation["all_contributions"])
        assert total == pytest.approx(explanation["model_score"], abs=1e-6)
        assert explanation["explained_output"] == "base_model_log_odds"
        assert explanation["calibration_method"] == "sigmoid"
        assert explanation["final_probability"] == result["predictions"][target]["probability"]
        assert len(explanation["top_features"]) == 5
        weight = next(item for item in explanation["all_contributions"] if item["feature"] == "Weight")
        assert weight["value"] is None
        assert np.isfinite(total)


def test_evaluation_refuses_repeat_and_wrong_partition(tmp_path, fitted_bundle, data):
    from cardio_risk.evaluate import evaluate_holdout
    X, y, _, _ = data
    with pytest.raises(ValueError, match="partition"):
        evaluate_holdout(fitted_bundle, X.iloc[:20], y.iloc[:20], tmp_path)
    (tmp_path / "test_metrics.json").write_text("{}")
    with pytest.raises(FileExistsError):
        evaluate_holdout(fitted_bundle, X, y, tmp_path)
