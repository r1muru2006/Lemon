import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import METADATA_FILE, MODEL_MAIN_FILE
from utils.io import write_json
from utils.metrics import binary_metrics


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def scores_from_model(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        raw = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)
        return raw
    return model.predict(x).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained model on test data")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    test_df = load_frame(args.test)
    model = joblib.load(args.model_dir / MODEL_MAIN_FILE)

    import json

    with (args.model_dir / METADATA_FILE).open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    feats = metadata["feature_columns"]
    medians = metadata["impute_medians"]
    threshold = float(metadata["threshold"])

    for c in feats:
        if c not in test_df.columns:
            test_df[c] = medians[c]

    x_test = test_df[feats].copy()
    for c in feats:
        x_test[c] = x_test[c].replace([np.inf, -np.inf], np.nan).fillna(medians[c])

    y_true = test_df["label_binary"].astype(int).to_numpy()
    y_score = scores_from_model(model, x_test)
    y_pred = (y_score >= threshold).astype(int)

    result = binary_metrics(y_true, y_score, threshold)

    family_report = {}
    if "attack_family" in test_df.columns:
        fam = test_df["attack_family"].astype(str).fillna("unknown")
        for name in sorted(fam.unique().tolist()):
            mask = fam == name
            n = int(mask.sum())
            if n == 0:
                continue
            detected_rate = float((y_pred[mask] == 1).mean())
            family_report[name] = {
                "count": n,
                "pred_attack_rate": detected_rate,
            }

    args.report_dir.mkdir(parents=True, exist_ok=True)

    cm = pd.crosstab(
        pd.Series(y_true, name="y_true"),
        pd.Series(y_pred, name="y_pred"),
        dropna=False,
    )
    cm.to_csv(args.report_dir / "confusion_matrix.csv")

    payload = {
        "test_metrics": result,
        "n_test": int(len(test_df)),
        "family_report": family_report,
    }
    write_json(payload, args.report_dir / "metrics_test.json")

    print(f"Saved: {args.report_dir / 'metrics_test.json'}")
    print(f"Saved: {args.report_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
