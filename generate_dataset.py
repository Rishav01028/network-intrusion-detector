"""
generate_dataset.py
Generates a synthetic, NSL-KDD-style labeled network traffic dataset.

Why synthetic: the real NSL-KDD dataset (~150k rows) has to be downloaded from
UNB/Kaggle, which needs internet access. This script generates data with the
same feature schema and realistic-ish per-class distributions, so the whole
pipeline (features -> model -> evaluation -> dashboard) runs end-to-end
offline. Swap this for the real NSL-KDD CSV any time by keeping the same
column names (see README).

Attack categories (matching NSL-KDD's top-level grouping):
    normal - regular traffic
    dos    - Denial of Service (e.g. SYN flood): huge packet counts, short duration
    probe  - port/network scanning: many connections, tiny payloads
    r2l    - Remote-to-Local: failed logins, odd services
    u2r    - User-to-Root: rare, root-shell / privilege-escalation traffic
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "ssh", "dns", "telnet", "private"]
FLAGS = ["SF", "S0", "REJ", "RSTR"]


def _sample_categorical(options, n, weights=None):
    return RNG.choice(options, size=n, p=weights)


def _make_class(label, n):
    if label == "normal":
        duration = RNG.exponential(scale=50, size=n)
        src_bytes = RNG.normal(500, 150, n).clip(0)
        dst_bytes = RNG.normal(800, 200, n).clip(0)
        count = RNG.integers(1, 10, n)
        srv_count = RNG.integers(1, 10, n)
        num_failed_logins = np.zeros(n)
        protocol = _sample_categorical(PROTOCOLS, n, [0.7, 0.25, 0.05])
        service = _sample_categorical(SERVICES, n)
        flag = _sample_categorical(FLAGS, n, [0.85, 0.05, 0.05, 0.05])

    elif label == "dos":
        duration = RNG.exponential(scale=1, size=n)
        src_bytes = RNG.normal(20, 10, n).clip(0)
        dst_bytes = RNG.normal(0, 5, n).clip(0)
        count = RNG.integers(200, 500, n)
        srv_count = RNG.integers(200, 500, n)
        num_failed_logins = np.zeros(n)
        protocol = _sample_categorical(PROTOCOLS, n, [0.9, 0.05, 0.05])
        service = _sample_categorical(SERVICES, n)
        flag = _sample_categorical(FLAGS, n, [0.1, 0.7, 0.1, 0.1])

    elif label == "probe":
        duration = RNG.exponential(scale=0.5, size=n)
        src_bytes = RNG.normal(10, 5, n).clip(0)
        dst_bytes = RNG.normal(5, 3, n).clip(0)
        count = RNG.integers(50, 200, n)
        srv_count = RNG.integers(1, 5, n)  # many hosts, few per-service
        num_failed_logins = np.zeros(n)
        protocol = _sample_categorical(PROTOCOLS, n, [0.6, 0.1, 0.3])
        service = _sample_categorical(SERVICES, n)
        flag = _sample_categorical(FLAGS, n, [0.2, 0.2, 0.5, 0.1])

    elif label == "r2l":
        duration = RNG.exponential(scale=20, size=n)
        src_bytes = RNG.normal(300, 100, n).clip(0)
        dst_bytes = RNG.normal(100, 50, n).clip(0)
        count = RNG.integers(1, 5, n)
        srv_count = RNG.integers(1, 5, n)
        num_failed_logins = RNG.integers(1, 5, n)
        protocol = _sample_categorical(PROTOCOLS, n, [0.9, 0.05, 0.05])
        service = _sample_categorical(["ftp", "telnet", "smtp"], n)
        flag = _sample_categorical(FLAGS, n, [0.3, 0.1, 0.5, 0.1])

    else:  # u2r
        duration = RNG.exponential(scale=100, size=n)
        src_bytes = RNG.normal(1000, 300, n).clip(0)
        dst_bytes = RNG.normal(1000, 300, n).clip(0)
        count = RNG.integers(1, 3, n)
        srv_count = RNG.integers(1, 3, n)
        num_failed_logins = RNG.integers(0, 2, n)
        protocol = _sample_categorical(PROTOCOLS, n, [0.95, 0.03, 0.02])
        service = _sample_categorical(["ssh", "telnet"], n)
        flag = _sample_categorical(FLAGS, n, [0.7, 0.1, 0.1, 0.1])

    return pd.DataFrame({
        "duration": duration,
        "protocol_type": protocol,
        "service": service,
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "count": count,
        "srv_count": srv_count,
        "num_failed_logins": num_failed_logins,
        "label": label,
    })


def generate(n_normal=4000, n_dos=1500, n_probe=1000, n_r2l=400, n_u2r=100) -> pd.DataFrame:
    frames = [
        _make_class("normal", n_normal),
        _make_class("dos", n_dos),
        _make_class("probe", n_probe),
        _make_class("r2l", n_r2l),
        _make_class("u2r", n_u2r),
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/traffic_dataset.csv", index=False)
    print(f"Wrote {len(df)} rows to data/traffic_dataset.csv")
    print(df["label"].value_counts())
