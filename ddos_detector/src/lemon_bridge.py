import argparse
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


LEMON_LINE_RE = re.compile(
    r"(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s*,\s*(?P<count>-?\d+(?:\.\d+)?)\s*,\s*(?P<est>-?\d+(?:\.\d+)?)"
)


def parse_time(ts_text: Optional[str]) -> datetime:
    if not ts_text:
        return datetime.now(UTC).replace(tzinfo=None)
    return datetime.fromisoformat(ts_text)


def parse_lemon_log(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LEMON_LINE_RE.search(line)
            if m is None:
                continue
            rows.append(
                {
                    "dst_ip": m.group("ip"),
                    "lemon_counter": float(m.group("count")),
                    "lemon_est": float(m.group("est")),
                }
            )
    return rows


def build_detector_frame(
    lemon_rows: List[dict],
    start_time: datetime,
    step_seconds: float,
    min_counter: float,
) -> pd.DataFrame:
    cooked = []
    for idx, row in enumerate(lemon_rows):
        counter = max(float(row["lemon_counter"]), 0.0)
        est = max(float(row["lemon_est"]), 1.0)
        if counter < min_counter:
            continue

        event_time = start_time + timedelta(seconds=idx * step_seconds)

        cooked.append(
            {
                "event_time": event_time,
                "src_ip": "0.0.0.0",
                "dst_ip": row["dst_ip"],
                "source_port": 0,
                "destination_port": 0,
                "protocol": 17,
                "flows_per_dst": counter,
                "window_flow_total": counter,
                "unique_src_per_dst": max(1.0, min(counter, est)),
                "unique_dst_per_src": 1.0,
                "syn_rate_per_dst": 0.0,
                "entropy_src_ip_on_dst_window": float(np.log2(est)),
                "entropy_src_port_on_dst_window": 0.0,
                "src_burst_score": 1.0,
                "dst_concentration_score": 1.0,
                "lemon_counter": counter,
                "lemon_est": est,
            }
        )

    return pd.DataFrame(cooked)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Lemon controller output into detector-compatible feature rows"
    )
    parser.add_argument("--lemon-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--start-time",
        type=str,
        default="",
        help="ISO format timestamp, default uses current UTC time",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=1.0,
        help="Time delta between rows when log has no timestamp",
    )
    parser.add_argument(
        "--min-counter",
        type=float,
        default=1.0,
        help="Drop weak rows below this Lemon counter",
    )
    args = parser.parse_args()

    rows = parse_lemon_log(args.lemon_log)
    if not rows:
        raise ValueError(f"No valid Lemon rows found in {args.lemon_log}")

    start_time = parse_time(args.start_time)
    out_df = build_detector_frame(
        lemon_rows=rows,
        start_time=start_time,
        step_seconds=args.step_seconds,
        min_counter=args.min_counter,
    )

    if out_df.empty:
        raise ValueError("All Lemon rows were filtered out by --min-counter")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        out_df.to_parquet(args.output, index=False)
    else:
        out_df.to_csv(args.output, index=False)

    print(f"Input rows parsed: {len(rows)}")
    print(f"Rows exported: {len(out_df)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
