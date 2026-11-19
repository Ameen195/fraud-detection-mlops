import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from fraudguard.features.build import build_features


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    cat = lambda vals: pd.Categorical(rng.choice(vals, n))  # noqa: E731
    return pd.DataFrame(
        {
            "TransactionDT": np.sort(rng.integers(0, 2000, n)) * 300,  # ~7 days, many ties
            "TransactionAmt": rng.gamma(2, 50, n).round(2),
            "card1": rng.integers(0, 6, n),
            "card2": rng.choice([1.0, 2.0, np.nan], n),
            "card3": 150.0,
            "card5": 226.0,
            "card6": cat(["debit", "credit"]),
            "addr1": rng.choice([10.0, 20.0, np.nan], n),
            "P_emaildomain": cat(["a.com", "b.com", None]),
            "R_emaildomain": cat(["a.com", None]),
            "DeviceInfo": cat(["ios", "win", "mac", None]),
            "isFraud": rng.integers(0, 2, n),
        }
    )


@pytest.mark.parametrize("q", [0.2, 0.5, 0.8])
def test_no_future_rows_used(q):
    df = _synthetic()
    cutoff = df["TransactionDT"].quantile(q)
    past = df["TransactionDT"] <= cutoff
    full = build_features(df)

    # 1) deleting all later rows changes nothing for rows at or before the cutoff
    assert_frame_equal(full[past], build_features(df[past]))

    # 2) scrambling every column of later rows (cards, amounts, devices, labels) changes nothing
    other = _synthetic(seed=1)
    fut = df.copy()
    for c in df.columns.drop("TransactionDT"):
        fut.loc[~past, c] = other.loc[~past, c]
    assert_frame_equal(full[past], build_features(fut)[past])


def test_row_order_does_not_matter():
    df = _synthetic()
    shuffled = df.sample(frac=1, random_state=0)
    assert_frame_equal(build_features(df), build_features(shuffled).loc[df.index])


def test_hand_checked_values():
    df = pd.DataFrame(
        {
            "TransactionDT": [0, 1000, 1000, 5000, 90000],
            "TransactionAmt": [10.0, 30.0, 50.0, 20.0, 40.0],
            "card1": 1,
            "card2": 1.0,
            "card3": 1.0,
            "card5": 1.0,
            "card6": "debit",
            "addr1": [1.0, 1.0, 2.0, 1.0, 1.0],
            "P_emaildomain": "a.com",
            "R_emaildomain": None,
            "DeviceInfo": ["a", "a", "b", "a", None],
        }
    )
    f = build_features(df)
    # same-timestamp rows do not see each other
    assert f["card_n_past"].tolist() == [0, 1, 1, 3, 4]
    assert f["secs_since_last"].tolist()[1:] == [1000, 1000, 4000, 85000]
    assert np.isnan(f["secs_since_last"][0])
    assert f.loc[3, "card_amt_mean_past"] == 30.0
    assert f.loc[3, "amt_to_card_mean"] == pytest.approx(20 / 30)
    assert f["card_n_1h"].tolist() == [0, 1, 1, 0, 0]
    assert f["card_n_24h"].tolist() == [0, 1, 1, 3, 1]  # row 4: window [3600, 90000)
    assert f.loc[4, "card_amt_24h"] == 20.0 and f.loc[4, "card_amt_7d"] == 110.0
    assert f["new_device_for_card"].tolist()[:4] == [1, 0, 1, 0]
    assert np.isnan(f.loc[4, "new_device_for_card"])
    assert f["new_addr_for_card"].tolist() == [1, 0, 1, 0, 0]
    assert f["freq_P_emaildomain"].tolist() == [0, 1, 1, 3, 4]
    assert f["freq_R_emaildomain"].isna().all()


def test_nan_amount_rejected():
    df = _synthetic(10)
    df.loc[3, "TransactionAmt"] = np.nan
    with pytest.raises(ValueError):
        build_features(df)
