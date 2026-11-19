"""Behavioral features built only from strictly earlier rows (TransactionDT < current row's).

Run on the full data (any row order), then filter by split, so valid/test rows see earlier history.
Same function for training and serving: to score one transaction, pass it together with its history.
Requires 0 <= TransactionDT < 2**31 and no NaN in TransactionAmt.
"""

import numpy as np
import pandas as pd

# No user ID in IEEE-CIS: a card = its card fields (addr1 left out so new-address can fire).
CARD_COLS = ["card1", "card2", "card3", "card5", "card6"]
WINDOWS = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}
NEW_VALUE_COLS = {"device": "DeviceInfo", "email": "P_emaildomain", "addr": "addr1"}
FREQ_COLS = ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo"]
_ALL = 2**31 - 1  # "window" covering all past
_SHIFT = np.int64(2**32)  # key = group * 2**32 + dt keeps groups apart when sorted

ENGINEERED = (
    ["card_n_past", "card_amt_mean_past", "amt_to_card_mean", "amt_z_card", "secs_since_last"]
    + [f"card_{k}_{w}" for w in WINDOWS for k in ("n", "amt")]
    + [f"new_{k}_for_card" for k in NEW_VALUE_COLS]
    + [f"freq_{c}" for c in FREQ_COLS]
)


def _codes(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df.groupby(cols, dropna=False, observed=True, sort=False).ngroup().to_numpy(np.int64)


class _Past:
    """Per row, index range [lo, hi) into group-then-time sorted rows: same group, earlier dt."""

    def __init__(self, codes: np.ndarray, dt: np.ndarray):
        self.codes, self.dt = codes, dt
        key = codes * _SHIFT + dt
        self.order = np.argsort(key, kind="stable")
        self.skey = key[self.order]
        self.hi = np.searchsorted(self.skey, key, side="left")  # excludes same-dt rows and itself

    def lo(self, window: int) -> np.ndarray:
        return np.searchsorted(self.skey, self.codes * _SHIFT + (self.dt - window), side="left")

    def sum(self, values: np.ndarray, lo: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(values[self.order])])
        return cs[self.hi] - cs[lo]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["TransactionDT"].to_numpy(np.int64)
    amt = df["TransactionAmt"].to_numpy(np.float64)
    if np.isnan(amt).any():
        raise ValueError("TransactionAmt has NaN")  # would poison every running sum
    if dt.min() < 0 or dt.max() >= _ALL:
        raise ValueError("TransactionDT out of range")
    out = pd.DataFrame(index=df.index)

    card = _Past(_codes(df, CARD_COLS), dt)
    lo = card.lo(_ALL)
    n = (card.hi - lo).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = card.sum(amt, lo) / n
        std = np.sqrt(np.clip(card.sum(amt**2, lo) / n - mean**2, 0, None))
        out["card_n_past"] = n
        out["card_amt_mean_past"] = mean
        out["amt_to_card_mean"] = np.where(mean > 0, amt / mean, np.nan)
        out["amt_z_card"] = np.where((n >= 2) & (std > 1e-9), (amt - mean) / std, np.nan)
    last_dt = dt[card.order][np.maximum(card.hi - 1, 0)]
    out["secs_since_last"] = np.where(n > 0, dt - last_dt, np.nan)
    for w, secs in WINDOWS.items():
        lo_w = card.lo(secs)  # window [t - secs, t)
        out[f"card_n_{w}"] = card.hi - lo_w
        out[f"card_amt_{w}"] = card.sum(amt, lo_w)

    for k, col in NEW_VALUE_COLS.items():
        pair = _Past(_codes(df, CARD_COLS + [col]), dt)
        seen = pair.hi - pair.lo(_ALL)
        out[f"new_{k}_for_card"] = np.where(df[col].isna(), np.nan, (seen == 0).astype(float))

    for col in FREQ_COLS:
        p = _Past(_codes(df, [col]), dt)
        out[f"freq_{col}"] = np.where(df[col].isna(), np.nan, p.hi - p.lo(_ALL))

    return out[ENGINEERED]
