import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import EPS, TIMESTAMP_CANDIDATES


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


def choose_time_col(df: pd.DataFrame) -> str:
    c = find_column(df.columns, TIMESTAMP_CANDIDATES)
    if c is None:
        if "end_ts" in df.columns:
            return "end_ts"
        if "start_ts" in df.columns:
            return "start_ts"
        raise ValueError("No usable timestamp column found")
    return c


def parse_time(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().mean() >= 0.6:
        return dt.dt.tz_convert(None)

    if pd.api.types.is_numeric_dtype(series):
        sec = pd.to_datetime(series, unit="s", errors="coerce", utc=True)
        ms = pd.to_datetime(series, unit="ms", errors="coerce", utc=True)
        dt = ms if ms.notna().mean() > sec.notna().mean() else sec
        return dt.dt.tz_convert(None)

    return dt.dt.tz_convert(None)


def entropy(series: pd.Series) -> float:
    counts = series.value_counts(normalize=True)
    if len(counts) == 0:
        return 0.0
    return float(-(counts * np.log2(counts + EPS)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build window-based features on top of flow features"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-seconds", type=int, default=1)
    args = parser.parse_args()

    if args.input.suffix.lower() == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    if len(df) == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".parquet":
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Input is empty, saved empty file to {args.output}")
        return

    src_col = find_column(df.columns, ["src_ip", "source_ip", "src"])
    dst_col = find_column(df.columns, ["dst_ip", "destination_ip", "dst"])
    sport_col = find_column(df.columns, ["src_port", "source_port", "sport"])

    if src_col is None or dst_col is None:
        raise ValueError("Source/Destination IP columns not found")

    time_col = choose_time_col(df)
    df["_event_time"] = parse_time(df[time_col])
    if df["_event_time"].isna().all():
        raise ValueError("Cannot parse timestamps for window aggregation")

    df["_window_start"] = df["_event_time"].dt.floor(f"{args.window_seconds}s")

    syn_col = "syn_count" if "syn_count" in df.columns else None
    df["_syn_metric"] = df[syn_col].astype(float) if syn_col else 0.0

    keys_dst = ["_window_start", dst_col]
    keys_src = ["_window_start", src_col]

    flows_per_dst = df.groupby(keys_dst).size().rename("flows_per_dst").reset_index()
    flows_per_src = df.groupby(keys_src).size().rename("flows_per_src").reset_index()

    unique_src_per_dst = (
        df.groupby(keys_dst)[src_col]
        .nunique()
        .rename("unique_src_per_dst")
        .reset_index()
    )
    unique_dst_per_src = (
        df.groupby(keys_src)[dst_col]
        .nunique()
        .rename("unique_dst_per_src")
        .reset_index()
    )

    syn_sum = df.groupby(keys_dst)["_syn_metric"].sum().rename("syn_sum").reset_index()
    syn_rate = flows_per_dst.merge(syn_sum, on=keys_dst, how="left")
    syn_rate["syn_rate_per_dst"] = syn_rate["syn_sum"] / syn_rate["flows_per_dst"].clip(
        lower=1
    )
    syn_rate = syn_rate[keys_dst + ["syn_rate_per_dst"]]

    ent_src_ip = (
        df.groupby(keys_dst)[src_col]
        .apply(entropy)
        .rename("entropy_src_ip_on_dst_window")
        .reset_index()
    )

    if sport_col is not None:
        ent_src_port = (
            df.groupby(keys_dst)[sport_col]
            .apply(entropy)
            .rename("entropy_src_port_on_dst_window")
            .reset_index()
        )
    else:
        ent_src_port = flows_per_dst.copy()
        ent_src_port["entropy_src_port_on_dst_window"] = 0.0
        ent_src_port = ent_src_port[keys_dst + ["entropy_src_port_on_dst_window"]]

    window_total = (
        df.groupby("_window_start").size().rename("window_flow_total").reset_index()
    )

    merged = df.merge(flows_per_dst, on=keys_dst, how="left")
    merged = merged.merge(flows_per_src, on=keys_src, how="left")
    merged = merged.merge(unique_src_per_dst, on=keys_dst, how="left")
    merged = merged.merge(unique_dst_per_src, on=keys_src, how="left")
    merged = merged.merge(syn_rate, on=keys_dst, how="left")
    merged = merged.merge(ent_src_ip, on=keys_dst, how="left")
    merged = merged.merge(ent_src_port, on=keys_dst, how="left")
    merged = merged.merge(window_total, on=["_window_start"], how="left")

    merged["src_burst_score"] = merged["flows_per_src"] / merged[
        "window_flow_total"
    ].clip(lower=1)
    merged["dst_concentration_score"] = merged["flows_per_dst"] / merged[
        "window_flow_total"
    ].clip(lower=1)

    merged = merged.drop(columns=["_syn_metric"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        merged.to_parquet(args.output, index=False)
    else:
        merged.to_csv(args.output, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(merged)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
