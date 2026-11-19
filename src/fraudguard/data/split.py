import pandas as pd


def time_split(df: pd.DataFrame, train_frac: float, valid_frac: float, time_col="TransactionDT"):
    """Sort by time and add a `split` column (train/valid/test). Never shuffles.

    Returns (sorted_df, boundaries). Boundaries are the first time value of valid and test.
    Rows sharing a boundary timestamp all go to the later split, so splits never overlap in time.
    """
    if not (0 < train_frac and train_frac + valid_frac < 1):
        raise ValueError("need 0 < train_frac and train_frac + valid_frac < 1")
    df = df.sort_values([time_col, "TransactionID"], kind="stable").reset_index(drop=True)
    n = len(df)
    valid_start = df[time_col].iloc[int(n * train_frac)]
    test_start = df[time_col].iloc[int(n * (train_frac + valid_frac))]
    split = pd.Series("train", index=df.index)
    split[df[time_col] >= valid_start] = "valid"
    split[df[time_col] >= test_start] = "test"
    df["split"] = split.astype("category")
    return df, {"valid_start": int(valid_start), "test_start": int(test_start)}