import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, fitted_bundle):
    from cardio_risk.predict import save_bundle
    from cardio_risk.api import create_app
    save_bundle(fitted_bundle, tmp_path / "bundle", fitted_bundle["metadata"])
    with TestClient(create_app(tmp_path / "bundle")) as connection:
        yield connection


def test_health_schema_and_predictions(client, data):
    X, _, _, _ = data
    assert client.get("/health").json()["status"] == "ready"
    assert len(client.get("/schema").json()["features"]) == 55
    result = client.post("/predict", json={"features": json.loads(Path("examples/synthetic_patient.json").read_text())["features"], "include_explanations": True})
    assert result.status_code == 200
    body = result.json()
    assert list(body["predictions"]) == ["cad", "lad", "lcx", "rca"]
    assert "diagnostic imaging" in body["disclaimer"]
    assert len(body["explanations"]) == 4
    assert all(0 <= prediction["probability"] <= 1 for prediction in body["predictions"].values())


@pytest.mark.parametrize("case", ["omitted", "extra", "target", "string", "category", "nulls", "outer_extra"])
def test_invalid_requests_return_422(client, data, case):
    X, _, _, _ = data
    fields = json.loads(Path("examples/synthetic_patient.json").read_text())["features"]
    payload = {"features": fields}
    if case == "omitted": fields.pop("Age")
    if case == "extra": fields["row_id"] = 4
    if case == "target": fields["Cath"] = "CAD"
    if case == "string": fields["Age"] = "50"
    if case == "category": fields["Sex"] = "unknown"
    if case == "nulls":
        for name in list(fields)[:12]: fields[name] = None
    if case == "outer_extra": payload["something"] = 1
    assert client.post("/predict", json=payload).status_code == 422


def test_missing_artifacts_fail_startup(tmp_path):
    from cardio_risk.api import create_app
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "absent")): pass


@pytest.mark.parametrize("field", ["Age", "Function Class"])
def test_huge_integer_returns_validation_error(client, data, field):
    X, _, _, _ = data
    features = json.loads(Path("examples/synthetic_patient.json").read_text())["features"]
    features[field] = 10**400
    result = client.post("/predict", json={"features": features})
    assert result.status_code == 422
