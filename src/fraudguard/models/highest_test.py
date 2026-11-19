"""Test-set numbers for the "highest" risk level, at the threshold chosen on validation.
Reads the saved test scores (no re-scoring, no re-tuning): python -m fraudguard.models.highest_test
"""

import json
from pathlib import Path

import pandas as pd

from fraudguard.data.load import load_config
from fraudguard.models.dashboard_export import point


def f4(x):
    return f"{x + 1e-12:.4f}"  # round half up, as the dashboard does (recall is exactly 0.40625)


def main() -> None:
    cfg = load_config()
    res, proc = Path(cfg["paths"]["results_dir"]), Path(cfg["paths"]["processed_dir"])
    th = json.loads((res / "threshold.json").read_text("utf-8"))
    thr, cost = th["highest"]["threshold"], th["cost_per_false_alarm"]
    ts = pd.read_parquet(proc / "test_scores.parquet")
    amt = pd.read_parquet(proc / "full.parquet", columns=["split", "TransactionAmt"])
    amt = amt.loc[amt["split"] == "test", "TransactionAmt"].to_numpy()
    assert len(amt) == len(ts), "test_scores and full.parquet test rows differ"
    y, p = ts["y"].to_numpy(), ts["p"].to_numpy()
    r = point(y, p, amt, thr, cost)
    r["precision"] = r["caught_n"] / (r["caught_n"] + r["false_alarms"])
    out = {"split": "test", "threshold_chosen_on": "valid", "cost_per_false_alarm": cost, **r}
    (res / "highest_test.json").write_text(json.dumps(out, indent=2), "utf-8")
    (res / "highest_test.md").write_text(
        f"# Highest risk level on test\n\nThreshold {thr:.4f}, chosen on validation (<= 1% false "
        f"alarm rate there). Saved test scores, not re-scored or re-tuned.\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| precision | {f4(r['precision'])} |\n| recall | {f4(r['recall'])} |\n"
        f"| false alarm rate | {r['false_alarm_rate']:.2%} |\n"
        f"| fraud caught / missed (count) | {r['caught_n']:,} / {r['missed_n']:,} |\n"
        f"| false alarms | {r['false_alarms']:,} |\n"
        f"| fraud amount caught / missed | {r['fraud_caught']:,.2f} / {r['fraud_missed']:,.2f} |\n"
        f"| false alarm cost ({cost:g} each) | {r['false_alarm_cost']:,.2f} |\n"
        f"| net saving | {r['net_saving']:,.2f} |\n",
        "utf-8",
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
