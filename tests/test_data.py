from pathlib import Path

import pytest

DATA = Path("data/raw/z_alizadeh_sani_extension.xlsx")


def test_workbook_audit_and_target_exclusion():
    from cardio_risk.data import load_dataset
    X, y, audit = load_dataset(DATA)
    assert X.shape == (303, 55)
    assert list(y) == ["cad", "lad", "lcx", "rca"]
    assert y.sum().tolist() == [216, 177, 119, 114]
    assert not {"Cath", "LAD", "LCX", "RCA"}.intersection(X)
    assert audit["missing_cells"] == 0
    assert audit["duplicate_rows"] == 0
    assert len(audit["label_disagreements"]) == 1
    assert audit["sha256"] == "739343245c2ba578b541370217531750d8e936022f928b83e0d91756caa3ff0b"


def test_unknown_target_is_rejected():
    import pandas as pd
    from cardio_risk.data import separate_targets
    frame = pd.read_excel(DATA, sheet_name="Sheet 1 - Table 1")
    frame.loc[0, "Cath"] = "unknown"
    with pytest.raises(ValueError, match="Cath"):
        separate_targets(frame)
