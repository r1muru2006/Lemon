import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scapy.utils import PcapReader
from tqdm import tqdm

from config import EPS
from utils.net import extract_5tuple, tcp_flag_dict


@dataclass
class FlowState:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    label_raw: str = "benign"

    start_ts: float = 0.0
    last_ts: float = 0.0
    last_pkt_ts: Optional[float] = None

    total_pkt: int = 0
    total_bytes: int = 0
    fwd_pkt_count: int = 0
    bwd_pkt_count: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0

    min_pkt_size: int = 10**9
    max_pkt_size: int = 0
    sum_pkt_size: int = 0

    syn_count: int = 0
    ack_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    urg_count: int = 0

    iats: List[float] = field(default_factory=list)

    def update(self, ts: float, pkt_len: int, is_forward: bool, flags: dict) -> None:
        if self.total_pkt == 0:
            self.start_ts = ts

        if self.last_pkt_ts is not None:
            self.iats.append(max(ts - self.last_pkt_ts, 0.0))

        self.last_pkt_ts = ts
        self.last_ts = ts

        self.total_pkt += 1
        self.total_bytes += pkt_len
        self.sum_pkt_size += pkt_len
        self.min_pkt_size = min(self.min_pkt_size, pkt_len)
        self.max_pkt_size = max(self.max_pkt_size, pkt_len)

        if is_forward:
            self.fwd_pkt_count += 1
            self.fwd_bytes += pkt_len
        else:
            self.bwd_pkt_count += 1
            self.bwd_bytes += pkt_len

        self.syn_count += flags["syn"]
        self.ack_count += flags["ack"]
        self.rst_count += flags["rst"]
        self.psh_count += flags["psh"]
        self.urg_count += flags["urg"]

    def to_record(self) -> dict:
        dur = max(self.last_ts - self.start_ts, 0.0)
        iat_arr = (
            np.array(self.iats, dtype=float)
            if self.iats
            else np.array([0.0], dtype=float)
        )

        return {
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "start_ts": self.start_ts,
            "end_ts": self.last_ts,
            "event_time": pd.to_datetime(self.last_ts, unit="s", utc=True).tz_convert(
                None
            ),
            "flow_duration_ms": dur * 1000.0,
            "total_pkt_count": self.total_pkt,
            "total_bytes": self.total_bytes,
            "fwd_pkt_count": self.fwd_pkt_count,
            "bwd_pkt_count": self.bwd_pkt_count,
            "fwd_bytes": self.fwd_bytes,
            "bwd_bytes": self.bwd_bytes,
            "pkt_rate": self.total_pkt / max(dur, EPS),
            "byte_rate": self.total_bytes / max(dur, EPS),
            "avg_pkt_size": self.sum_pkt_size / max(self.total_pkt, 1),
            "min_pkt_size": 0 if self.min_pkt_size == 10**9 else self.min_pkt_size,
            "max_pkt_size": self.max_pkt_size,
            "syn_count": self.syn_count,
            "ack_count": self.ack_count,
            "rst_count": self.rst_count,
            "psh_count": self.psh_count,
            "urg_count": self.urg_count,
            "tcp_flag_ratio_syn_ack": self.syn_count / max(self.ack_count, 1),
            "bwd_fwd_pkt_ratio": self.bwd_pkt_count / max(self.fwd_pkt_count, 1),
            "iat_mean": float(iat_arr.mean()),
            "iat_std": float(iat_arr.std()),
            "iat_min": float(iat_arr.min()),
            "iat_max": float(iat_arr.max()),
            "burstiness": float(iat_arr.std() / max(iat_arr.mean(), EPS)),
            "label_raw": self.label_raw,
            "label_binary": 0 if self.label_raw == "benign" else 1,
            "attack_family": "benign" if self.label_raw == "benign" else "other_attack",
        }


def reverse_key(key: Tuple[str, str, int, int, int]) -> Tuple[str, str, int, int, int]:
    return (key[1], key[0], key[3], key[2], key[4])


def is_expired(
    st: FlowState, now_ts: float, idle_timeout: float, active_timeout: float
) -> bool:
    idle_expired = (now_ts - st.last_ts) > idle_timeout
    active_expired = (st.last_ts - st.start_ts) > active_timeout
    return idle_expired or active_expired


def flush_expired(
    flow_map: Dict[Tuple[str, str, int, int, int], FlowState],
    now_ts: float,
    idle_timeout: float,
    active_timeout: float,
) -> List[dict]:
    expired_keys = []
    records = []
    for k, st in flow_map.items():
        if is_expired(st, now_ts, idle_timeout, active_timeout):
            expired_keys.append(k)
    for k in expired_keys:
        records.append(flow_map[k].to_record())
        del flow_map[k]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PCAP into NetFlow-like records"
    )
    parser.add_argument("--pcap", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-timeout", type=float, default=10.0)
    parser.add_argument("--active-timeout", type=float, default=30.0)
    parser.add_argument("--label", type=str, default="benign")
    parser.add_argument("--limit-packets", type=int, default=0)
    args = parser.parse_args()

    flow_map: Dict[Tuple[str, str, int, int, int], FlowState] = {}
    rows: List[dict] = []

    packet_count = 0
    with PcapReader(str(args.pcap)) as pcap:
        for pkt in tqdm(pcap, desc="Parsing packets"):
            key = extract_5tuple(pkt)
            if key is None:
                continue

            ts = float(pkt.time)
            plen = int(len(pkt))
            flags = tcp_flag_dict(pkt)

            rev = reverse_key(key)
            if key in flow_map:
                state = flow_map[key]
                is_forward = True
            elif rev in flow_map:
                state = flow_map[rev]
                is_forward = False
            else:
                state = FlowState(
                    src_ip=key[0],
                    dst_ip=key[1],
                    src_port=key[2],
                    dst_port=key[3],
                    protocol=key[4],
                    label_raw=args.label,
                )
                flow_map[key] = state
                is_forward = True

            state.update(ts=ts, pkt_len=plen, is_forward=is_forward, flags=flags)
            rows.extend(
                flush_expired(flow_map, ts, args.idle_timeout, args.active_timeout)
            )

            packet_count += 1
            if args.limit_packets > 0 and packet_count >= args.limit_packets:
                break

    for st in list(flow_map.values()):
        rows.append(st.to_record())

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        out_df.to_csv(args.output, index=False)
    else:
        out_df.to_parquet(args.output, index=False)

    print(f"Input PCAP: {args.pcap}")
    print(f"Packets parsed: {packet_count}")
    print(f"Flows generated: {len(out_df)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
