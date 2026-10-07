import pytest
from pathlib import Path


def record_and_schema():
    from cardio_risk.data import load_dataset
    from cardio_risk.schema import build_schema
    X, _, _ = load_dataset(Path("data/raw/z_alizadeh_sani_extension.xlsx"))
    return X.iloc[0].to_dict(), build_schema(X.iloc[:200])


def test_normalization_and_schema_coverage():
    from cardio_risk.schema import validate_record
    record, schema = record_and_schema()
    record["Sex"] = " Fmale "
    frame, flags = validate_record(record, schema)
    assert frame.iloc[0]["Sex"] == "Female"
    assert len(schema["features"]) == 55
    assert flags["imputed_features"] == []


@pytest.mark.parametrize("change", ["missing", "target", "unknown", "string", "nan", "infinity", "boolean", "category", "too_missing", "bad_binary"])
def test_invalid_input_is_rejected(change):
    from cardio_risk.schema import validate_record, InputError
    record, schema = record_and_schema()
    if change == "missing": record.pop("Age")
    if change == "target": record["LAD"] = "Normal"
    if change == "unknown": record["row_id"] = 0
    if change == "string": record["Age"] = "65"
    if change == "nan": record["Age"] = float("nan")
    if change == "infinity": record["Age"] = float("inf")
    if change == "boolean": record["Age"] = True
    if change == "category": record["Sex"] = "unknown"
    if change == "bad_binary": record["DM"] = 2
    if change == "too_missing":
        for name in list(record)[:12]: record[name] = None
    with pytest.raises(InputError): validate_record(record, schema)


def test_nulls_and_range_warnings_do_not_clip():
    from cardio_risk.schema import validate_record
    record, schema = record_and_schema()
    record["Age"] = 1000
    record["Weight"] = None
    frame, flags = validate_record(record, schema)
    assert frame.iloc[0]["Age"] == 1000
    assert flags["imputed_features"] == ["Weight"]
    assert flags["out_of_range_features"] == ["Age"]
