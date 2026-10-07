"""Workbook audit, target mapping, and fixed partition creation."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import APPROVED, canonical_frame

TARGET_COLUMNS = {"cad": "Cath", "lad": "LAD", "lcx": "LCX", "rca": "RCA"}
SOURCE = "https://archive.ics.uci.edu/dataset/411/extention+of+z+alizadeh+sani+dataset"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def separate_targets(frame):
    required = APPROVED | set(TARGET_COLUMNS.values())
    if set(frame) != required:
        raise ValueError("Workbook does not match the approved clinical/target columns")
    y = pd.DataFrame(index=frame.index)
    for target, column in TARGET_COLUMNS.items():
        mapping = {"Normal": 0, "CAD" if target == "cad" else "Stenotic": 1}
        values = frame[column].astype(str).str.strip()
        if not values.isin(mapping).all():
            raise ValueError(f"Unknown target value in {column}")
        y[target] = values.map(mapping).astype(int)
    return canonical_frame(frame.drop(columns=list(TARGET_COLUMNS.values()))), y


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = Path(path)
    raw = pd.read_excel(path, sheet_name="Sheet 1 - Table 1")
    X, y = separate_targets(raw)
    disagreements = y.index[y.cad != y[["lad", "lcx", "rca"]].any(axis=1).astype(int)]
    audit = {"source": SOURCE, "license": "CC BY 4.0", "creators": "Alizadehsani, R.; Roshanzamir, M.; Sani, Z.", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sheet": "Sheet 1 - Table 1", "records": len(raw), "columns": len(raw.columns), "inputs": len(X.columns), "missing_cells": int(raw.isna().sum().sum()), "duplicate_rows": int(raw.duplicated().sum()), "class_counts": {target: {"positive": int(y[target].sum()), "negative": int((y[target] == 0).sum())} for target in y}, "constant_columns": [name for name in X if X[name].nunique(dropna=False) == 1], "label_disagreements": [{"row_id": int(i), **{name: int(y.loc[i, name]) for name in y}} for i in disagreements]}
    return X, y, audit


def create_split(y: pd.DataFrame, dataset_hash: str) -> dict:
    development, test = train_test_split(y.index.to_numpy(), test_size=0.2, stratify=y.cad, random_state=42)
    counts = {}
    for partition, ids in (("development", development), ("test", test)):
        counts[partition] = {}
        for target in y:
            count = y.loc[ids, target].value_counts()
            if len(count) != 2 or (partition == "test" and count.min() < 5):
                raise ValueError(f"Insufficient class support: {partition}/{target}")
            counts[partition][target] = {str(k): int(v) for k, v in count.items()}
    return {"sha256": dataset_hash, "seed": 42, "development": sorted(map(int, development)), "test": sorted(map(int, test)), "class_counts": counts}


def load_split(path, dataset_hash):
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    if split["sha256"] != dataset_hash:
        raise ValueError("Dataset hash differs from saved partition")
    return split
