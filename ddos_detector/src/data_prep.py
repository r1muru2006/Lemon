import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ATTACK_FAMILY_KEYWORDS,
    LABEL_CANDIDATES,
    LEAKY_COLUMNS,
    TIMESTAMP_CANDIDATES,
)
from utils.io import ensure_dir, write_json


def read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError as exc:
            last_err = exc
    raise last_err if last_err is not None else ValueError(f"Cannot read {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        new_c = re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip().lower()).strip("_")
        col_map[c] = new_c
    return df.rename(columns=col_map)


def find_column(columns, candidates) -> Optional[str]:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    for c in columns:
        for needle in candidates:
            if needle in c:
                return c
    return None


def parse_event_time(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().mean() >= 0.6:
        return dt.dt.tz_convert(None)

    if pd.api.types.is_numeric_dtype(series):
        as_seconds = pd.to_datetime(series, errors="coerce", unit="s", utc=True)
        as_millis = pd.to_datetime(series, errors="coerce", unit="ms", utc=True)
        use_millis = as_millis.notna().mean() > as_seconds.notna().mean()
        dt = as_millis if use_millis else as_seconds
        if dt.notna().mean() > 0:
            return dt.dt.tz_convert(None)

    return pd.Series(pd.NaT, index=series.index)


def attack_family_from_label(label: str) -> str:
    low = str(label).strip().lower()
    if re.search(r"benign|normal", low):
        return "benign"
    for k, v in ATTACK_FAMILY_KEYWORDS.items():
        if k in low:
            return v
    return "other_attack"


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    medians = df[numeric_cols].median(numeric_only=True)
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    return df


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if "event_time" in df.columns and df["event_time"].notna().mean() > 0.6:
        tmp = df.dropna(subset=["event_time"]).copy()
        tmp["_day"] = tmp["event_time"].dt.date.astype(str)
        days = sorted(tmp["_day"].unique().tolist())

        if len(days) >= 3:
            n_days = len(days)
            train_end = max(1, int(n_days * train_ratio))
            val_end = max(train_end + 1, int(n_days * (train_ratio + val_ratio)))
            val_end = min(val_end, n_days - 1)

            train_days = set(days[:train_end])
            val_days = set(days[train_end:val_end])
            test_days = set(days[val_end:])

            train_df = tmp[tmp["_day"].isin(train_days)].copy()
            val_df = tmp[tmp["_day"].isin(val_days)].copy()
            test_df = tmp[tmp["_day"].isin(test_days)].copy()
            for out_df in (train_df, val_df, test_df):
                out_df.drop(columns=["_day"], inplace=True, errors="ignore")
            return train_df, val_df, test_df, "split_by_day"

        sorted_df = tmp.sort_values("event_time").reset_index(drop=True)
        n = len(sorted_df)
        i_train = int(n * train_ratio)
        i_val = int(n * (train_ratio + val_ratio))
        return (
            sorted_df.iloc[:i_train].copy(),
            sorted_df.iloc[i_train:i_val].copy(),
            sorted_df.iloc[i_val:].copy(),
            "split_by_time_index",
        )

    y = df["label_binary"].astype(int)
    train_df, rem_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=y,
        random_state=seed,
    )
    rem_ratio = val_ratio / max(1 - train_ratio, 1e-9)
    val_df, test_df = train_test_split(
        rem_df,
        test_size=(1 - rem_ratio),
        stratify=rem_df["label_binary"].astype(int),
        random_state=seed,
    )
    return train_df.copy(), val_df.copy(), test_df.copy(), "stratified_random"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CICDDoS CSV files into train/val/test parquet"
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    csv_files = sorted(args.raw_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {args.raw_dir}")

    frames = []
    for p in csv_files:
        df = read_csv_robust(p)
        df = normalize_columns(df)
        df["source_file"] = p.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    if args.max_rows > 0:
        data = data.iloc[: args.max_rows].copy()

    label_col = find_column(data.columns, LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError("Cannot find label column in input CSV")

    data["label_raw"] = data[label_col].astype(str).str.strip().str.lower()
    data["label_binary"] = (
        ~data["label_raw"].str.contains(r"benign|normal", regex=True)
    ).astype(int)
    data["attack_family"] = data["label_raw"].map(attack_family_from_label)

    ts_col = find_column(data.columns, TIMESTAMP_CANDIDATES)
    if ts_col is not None:
        data["event_time"] = parse_event_time(data[ts_col])

    drop_cols = [c for c in data.columns if c in LEAKY_COLUMNS]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    data = clean_numeric(data)

    if "event_time" in data.columns:
        data = data.sort_values("event_time", na_position="last").reset_index(drop=True)

    train_df, val_df, test_df, split_strategy = split_dataframe(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    ensure_dir(args.out_dir)
    train_path = args.out_dir / "train.parquet"
    val_path = args.out_dir / "val.parquet"
    test_path = args.out_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    summary = {
        "input_files": [str(p) for p in csv_files],
        "rows_total": int(len(data)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "label_column": label_col,
        "timestamp_column": ts_col,
        "split_strategy": split_strategy,
        "attack_rate_total": float(data["label_binary"].mean()),
    }
    write_json(summary, args.out_dir / "prep_summary.json")

    print("Saved:")
    print(f"- {train_path}")
    print(f"- {val_path}")
    print(f"- {test_path}")
    print(f"Split strategy: {split_strategy}")


if __name__ == "__main__":
    main()
