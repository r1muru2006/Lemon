import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import METADATA_FILE, MODEL_MAIN_FILE
from utils.io import write_json


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def pick_time_column(df: pd.DataFrame) -> str:
    candidates = ["event_time", "_event_time", "end_ts", "start_ts", "timestamp"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No time column found for online simulation")


def parse_time(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().mean() >= 0.5:
        return dt.dt.tz_convert(None)
    if pd.api.types.is_numeric_dtype(series):
        sec = pd.to_datetime(series, errors="coerce", unit="s", utc=True)
        return sec.dt.tz_convert(None)
    return dt.dt.tz_convert(None)


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
        description="Simulate online DDoS detection on flow data"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    df = load_frame(args.input)
    if len(df) == 0:
        raise ValueError("Input dataset is empty")

    with (args.model_dir / METADATA_FILE).open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = joblib.load(args.model_dir / MODEL_MAIN_FILE)
    feats = metadata["feature_columns"]
    medians = metadata["impute_medians"]
    threshold = float(metadata["threshold"])

    time_col = pick_time_column(df)
    df["_sim_time"] = parse_time(df[time_col])
    df = df.sort_values("_sim_time").reset_index(drop=True)

    for c in feats:
        if c not in df.columns:
            df[c] = medians[c]

    x_all = df[feats].copy()
    for c in feats:
        x_all[c] = x_all[c].replace([np.inf, -np.inf], np.nan).fillna(medians[c])

    y_true = (
        df["label_binary"].astype(int).to_numpy()
        if "label_binary" in df.columns
        else None
    )

    alerts = []
    first_attack_time = None
    first_detect_time = None
    all_scores = np.zeros(len(df), dtype=float)

    t0 = time.perf_counter()
    for i in range(0, len(df), args.batch_size):
        j = min(i + args.batch_size, len(df))
        x_batch = x_all.iloc[i:j]
        scores = scores_from_model(model, x_batch)
        preds = (scores >= threshold).astype(int)
        all_scores[i:j] = scores

        for k, pred in enumerate(preds):
            if pred != 1:
                continue
            idx = i + k
            alerts.append(
                {
                    "row_index": int(idx),
                    "time": str(df.loc[idx, "_sim_time"]),
                    "score": float(scores[k]),
                    "src_ip": df.loc[idx, "src_ip"] if "src_ip" in df.columns else "",
                    "dst_ip": df.loc[idx, "dst_ip"] if "dst_ip" in df.columns else "",
                }
            )

        if y_true is not None:
            batch_true = y_true[i:j]
            if first_attack_time is None and (batch_true == 1).any():
                local_idx = int(np.where(batch_true == 1)[0][0])
                first_attack_time = df.loc[i + local_idx, "_sim_time"]

            batch_detect = (preds == 1) & (batch_true == 1)
            if first_detect_time is None and batch_detect.any():
                local_idx = int(np.where(batch_detect)[0][0])
                first_detect_time = df.loc[i + local_idx, "_sim_time"]

    elapsed = max(time.perf_counter() - t0, 1e-9)
    throughput = float(len(df) / elapsed)

    detection_delay_seconds = None
    if first_attack_time is not None and first_detect_time is not None:
        detection_delay_seconds = float(
            (first_detect_time - first_attack_time).total_seconds()
        )

    alerts_df = pd.DataFrame(alerts)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    alerts_path = args.report_dir / "alerts.csv"
    alerts_df.to_csv(alerts_path, index=False)

    eval_payload = {
        "rows": int(len(df)),
        "alerts": int(len(alerts_df)),
        "alerts_per_minute": float((len(alerts_df) / elapsed) * 60.0),
        "throughput_flows_per_sec": throughput,
        "detection_delay_seconds": detection_delay_seconds,
        "threshold": threshold,
    }

    if y_true is not None:
        preds_all = (all_scores >= threshold).astype(int)
        tp = int(((preds_all == 1) & (y_true == 1)).sum())
        fp = int(((preds_all == 1) & (y_true == 0)).sum())
        fn = int(((preds_all == 0) & (y_true == 1)).sum())
        tn = int(((preds_all == 0) & (y_true == 0)).sum())
        eval_payload.update(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
            }
        )

    write_json(eval_payload, args.report_dir / "online_eval.json")

    print(f"Saved: {alerts_path}")
    print(f"Saved: {args.report_dir / 'online_eval.json'}")


if __name__ == "__main__":
    main()
