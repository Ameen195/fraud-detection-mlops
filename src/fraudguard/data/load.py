from pathlib import Path

import pandas as pd
import yaml

TXN_CATS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"] + [
    f"M{i}" for i in range(1, 10)
]


def load_config(path="config/settings.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_raw(cfg: dict) -> pd.DataFrame:
    """Read transaction + identity CSVs and left-merge on TransactionID (memory-friendly dtypes)."""
    raw = Path(cfg["paths"]["raw_dir"])
    txn_path = raw / cfg["data"]["transaction_file"]
    cols = pd.read_csv(txn_path, nrows=0).columns
    dtype = {c: "float32" for c in cols}
    dtype.update({c: "category" for c in TXN_CATS})
    dtype.update(TransactionID="int32", TransactionDT="int32", isFraud="int8")
    dtype["TransactionAmt"] = "float64"
    txn = pd.read_csv(txn_path, dtype=dtype)
    ident = pd.read_csv(raw / cfg["data"]["identity_file"])
    for c in ident.select_dtypes("object"):
        ident[c] = ident[c].astype("category")
    return txn.merge(ident, on="TransactionID", how="left", validate="one_to_one")