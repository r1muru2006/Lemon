import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DEFAULT_MAX_FPR, METADATA_FILE, MODEL_BASELINE_FILE, MODEL_MAIN_FILE
from utils.io import ensure_dir, write_json
from utils.metrics import binary_metrics, find_threshold_for_max_fpr


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "label_binary",
        "label_raw",
        "attack_family",
        "source_file",
        "event_time",
        "_event_time",
        "_window_start",
    }
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if not cols:
        raise ValueError("No numeric features found for training")
    return cols


def prepare_xy(
    df: pd.DataFrame,
    features: List[str],
    medians: Dict[str, float] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, float]]:
    x = df[features].copy()
    if medians is None:
        medians = {c: float(x[c].median()) for c in features}
    for c in features:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan)
        x[c] = x[c].fillna(medians[c])
    y = df["label_binary"].astype(int).to_numpy()
    return x, y, medians


def train_baseline(x_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    return clf


def train_main_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    seed: int,
):
    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos

    try:
        xgb_module = importlib.import_module("xgboost")
        XGBClassifier = getattr(xgb_module, "XGBClassifier")

        model = XGBClassifier(
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        return model, "xgboost"
    except Exception:
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
        model.fit(x_train, y_train)
        return model, "random_forest"


def scores_from_model(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x)
        raw = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)
        return raw
    return model.predict(x).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train baseline and main DDoS detectors"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--max-fpr", type=float, default=DEFAULT_MAX_FPR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_df = load_frame(args.train)
    val_df = load_frame(args.val)

    feats = feature_columns(train_df)

    x_train, y_train, medians = prepare_xy(train_df, feats)
    x_val, y_val, _ = prepare_xy(val_df, feats, medians=medians)

    baseline = train_baseline(x_train, y_train)
    baseline_scores = scores_from_model(baseline, x_val)
    baseline_metrics = binary_metrics(y_val, baseline_scores, threshold=0.5)

    main_model, main_name = train_main_model(
        x_train, y_train, x_val, y_val, seed=args.seed
    )
    main_scores = scores_from_model(main_model, x_val)
    tuned_thr = find_threshold_for_max_fpr(y_val, main_scores, max_fpr=args.max_fpr)
    main_metrics = binary_metrics(y_val, main_scores, threshold=tuned_thr)

    ensure_dir(args.model_dir)
    ensure_dir(args.report_dir)

    joblib.dump(baseline, args.model_dir / MODEL_BASELINE_FILE)
    joblib.dump(main_model, args.model_dir / MODEL_MAIN_FILE)

    metadata = {
        "model_name": main_name,
        "feature_columns": feats,
        "impute_medians": medians,
        "threshold": float(tuned_thr),
        "max_fpr_target": float(args.max_fpr),
    }
    write_json(metadata, args.model_dir / METADATA_FILE)

    report = {
        "baseline": baseline_metrics,
        "main": main_metrics,
        "main_model_name": main_name,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
    }
    write_json(report, args.report_dir / "metrics_offline.json")

    print(f"Saved model: {args.model_dir / MODEL_MAIN_FILE}")
    print(f"Saved metadata: {args.model_dir / METADATA_FILE}")
    print(f"Saved report: {args.report_dir / 'metrics_offline.json'}")


if __name__ == "__main__":
    main()
