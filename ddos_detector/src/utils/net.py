from typing import Optional, Tuple

from scapy.layers.inet import IP, TCP, UDP


def extract_5tuple(pkt) -> Optional[Tuple[str, str, int, int, int]]:
    if IP not in pkt:
        return None

    ip = pkt[IP]
    src_ip = ip.src
    dst_ip = ip.dst
    proto = int(ip.proto)

    src_port = 0
    dst_port = 0
    if TCP in pkt:
        src_port = int(pkt[TCP].sport)
        dst_port = int(pkt[TCP].dport)
    elif UDP in pkt:
        src_port = int(pkt[UDP].sport)
        dst_port = int(pkt[UDP].dport)

    return (src_ip, dst_ip, src_port, dst_port, proto)


def tcp_flag_dict(pkt) -> dict:
    flags = {"syn": 0, "ack": 0, "rst": 0, "psh": 0, "urg": 0}
    if TCP not in pkt:
        return flags

    tcp_flags = int(pkt[TCP].flags)
    flags["syn"] = 1 if tcp_flags & 0x02 else 0
    flags["ack"] = 1 if tcp_flags & 0x10 else 0
    flags["rst"] = 1 if tcp_flags & 0x04 else 0
    flags["psh"] = 1 if tcp_flags & 0x08 else 0
    flags["urg"] = 1 if tcp_flags & 0x20 else 0
    return flags
