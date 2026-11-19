"""Measure /score latency on real validation rows.

Usage: python -m fraudguard.serving.latency [url] [n]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from fraudguard.data.load import load_config
from fraudguard.serving.app import row_to_payload


def main(url="http://localhost:8000", n=200):
    cfg = load_config()
    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "full.parquet")
    rows = df[df["split"] == "valid"].sample(n, random_state=0)
    s = requests.Session()
    ms = []
    for _, row in rows.iterrows():
        body = row_to_payload(row)
        t = time.perf_counter()
        r = s.post(f"{url}/score", json=body)
        ms.append((time.perf_counter() - t) * 1000)
        r.raise_for_status()
    p50, p95 = np.percentile(ms[5:], [50, 95])  # first calls warm up
    out = (
        f"# /score latency\n\n{n} validation rows, one request each, client-side, {url}\n\n"
        "| p50 | p95 | max |\n|---|---|---|\n"
        f"| {p50:.0f} ms | {p95:.0f} ms | {max(ms):.0f} ms |\n"
    )
    Path(cfg["paths"]["results_dir"], "latency.md").write_text(out, "utf-8")
    print(out)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["http://localhost:8000"]), *map(int, sys.argv[2:3]))
