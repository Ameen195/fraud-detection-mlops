import pandas as pd

REQUIRED = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1"]


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError on structural problems."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df["TransactionID"].duplicated().any():
        raise ValueError("duplicate TransactionID")
    if not df["isFraud"].isin([0, 1]).all():
        raise ValueError("isFraud must be 0/1")
    if df["TransactionDT"].isna().any() or (df["TransactionDT"] < 0).any():
        raise ValueError("TransactionDT has nulls or negatives")
