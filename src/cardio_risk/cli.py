"""Repeatable project commands; paths resolve relative to the working folder."""

import argparse
import json
from pathlib import Path

from .data import create_split, load_dataset, load_split, write_json
from .schema import build_schema


def audit(data):
    _, _, details = load_dataset(Path(data))
    write_json("data/processed/provenance.json", details)
    write_json("reports/audit.json", details)
    Path("reports/audit.md").write_text("# Dataset audit\n\n```json\n" + json.dumps(details, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(details, indent=2))


def split_data(data):
    X, y, details = load_dataset(Path(data))
    if details["duplicate_rows"]:
        raise ValueError("Duplicate patients require a grouped split")
    path = Path("data/processed/split.json")
    if path.exists():
        split = load_split(path, details["sha256"], y)
    else:
        split = create_split(y, details["sha256"])
        write_json(path, split)
    write_json("data/processed/schema.json", build_schema(X.loc[split["development"]]))
    print(f"Fixed split: {len(split['development'])} development / {len(split['test'])} test")


def train(config_path, run_id):
    from .train import fit_selected, run_search
    from .predict import save_bundle
    config = json.loads(Path(config_path).read_text())
    artifact_dir, report_dir = Path("artifacts") / run_id, Path("reports") / run_id
    if artifact_dir.exists() or report_dir.exists():
        raise FileExistsError("Run already exists; choose a new run ID")
    X, y, details = load_dataset(Path(config["data"]))
    split = load_split("data/processed/split.json", details["sha256"], y)
    ids = split["development"]
    schema = build_schema(X.loc[ids])
    report_dir.mkdir(parents=True)
    write_json(report_dir / "audit.json", details)
    (report_dir / "audit.md").write_text(Path("reports/audit.md").read_text(encoding="utf-8"), encoding="utf-8")
    selection = run_search(X.loc[ids], y.loc[ids], schema, config)
    write_json(report_dir / "search.json", selection)
    import pandas as pd
    pd.DataFrame([{key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items() if key != "fold_metrics"} for row in selection["candidates"]]).to_csv(report_dir / "development_results.csv", index=False)
    fitted = fit_selected(X.loc[ids], y.loc[ids], selection, schema)
    fitted.pop("oof").to_csv(report_dir / "development_oof.csv", index_label="row_id")
    metadata = {"model_version": run_id, "dataset_hash": details["sha256"], "split": split, "config": config}
    save_bundle(fitted, artifact_dir, metadata)
    from .explain import summarize_global
    write_json(report_dir / "global_attributions.json", summarize_global(fitted, X.loc[ids]))
    write_json(report_dir / "selection.json", {target: {k: v for k, v in record.items() if k not in ("estimator", "base_estimator", "background")} for target, record in fitted["targets"].items()})
    print(f"Frozen bundle saved: {artifact_dir}")


def main():
    parser = argparse.ArgumentParser(description="CardioVista reproducible ML commands")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "split"):
        sub.add_parser(name).add_argument("--data", default="data/raw/z_alizadeh_sani_extension.xlsx")
    training = sub.add_parser("train")
    training.add_argument("--config", default="configs/training.json")
    training.add_argument("--run-id", required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--run-id", required=True)
    prediction = sub.add_parser("predict")
    prediction.add_argument("--run-id", required=True)
    prediction.add_argument("--input", required=True)
    prediction.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    if args.command == "audit": audit(args.data)
    elif args.command == "split": split_data(args.data)
    elif args.command == "train": train(args.config, args.run_id)
    elif args.command == "predict":
        from .predict import load_bundle, predict_record
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = predict_record(load_bundle(Path("artifacts") / args.run_id), payload.get("features", payload), args.explain)
        print(json.dumps(result, indent=2, allow_nan=False))
    elif args.command == "evaluate":
        from .predict import load_bundle
        from .evaluate import evaluate_holdout
        bundle = load_bundle(Path("artifacts") / args.run_id)
        X, y, details = load_dataset(Path(bundle["metadata"]["config"]["data"]))
        split = load_split("data/processed/split.json", details["sha256"], y)
        if split != bundle["metadata"]["split"]:
            raise ValueError("Current split differs from frozen artifact")
        ids = split["test"]
        result = evaluate_holdout(bundle, X.loc[ids], y.loc[ids], Path("reports") / args.run_id)
        print(json.dumps(result["targets"], indent=2))


if __name__ == "__main__":
    main()
