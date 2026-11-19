"""Load, validate, time-split. Writes data/processed/full.parquet (time-sorted, with a `split`
column) and eval/results/split.json. Features are built on the full sorted data (past rows only)
and then filtered by `split`, so valid/test rows can see earlier history."""
import json
from pathlib import Path

from fraudguard.data.load import load_config, load_raw
from fraudguard.data.split import time_split
from fraudguard.data.validate import validate


def main() -> None:
    cfg = load_config()
    df = load_raw(cfg)
    validate(df)
    df, bounds = time_split(df, cfg["split"]["train_frac"], cfg["split"]["valid_frac"])
    out = Path(cfg["paths"]["processed_dir"]) / "full.parquet"
    df.to_parquet(out, index=False)
    info = dict(bounds, time_column="TransactionDT", rows_total=len(df), splits={})
    for name, g in df.groupby("split", observed=True):
        info["splits"][name] = {
            "rows": len(g),
            "fraud_rate": round(float(g["isFraud"].mean()), 5),
            "dt_min": int(g["TransactionDT"].min()),
            "dt_max": int(g["TransactionDT"].max()),
        }
    res = Path(cfg["paths"]["results_dir"]) / "split.json"
    res.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()