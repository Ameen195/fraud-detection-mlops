import pandas as pd
import pytest

from fraudguard.data.split import time_split
from fraudguard.data.validate import validate


def _df(n=100):
    # deliberately shuffled order
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            "TransactionDT": list(range(n))[::-1],
            "isFraud": [0, 1] * (n // 2),
            "TransactionAmt": 1.0,
            "card1": 1,
        }
    ).sample(frac=1, random_state=0)


def test_split_is_time_ordered_and_disjoint():
    out, b = time_split(_df(), 0.6, 0.2)
    tr, va, te = (out[out.split == s]["TransactionDT"] for s in ("train", "valid", "test"))
    assert tr.max() < va.min() and va.max() < te.min()
    assert (len(tr), len(va), len(te)) == (60, 20, 20)
    assert b["valid_start"] == va.min() and b["test_start"] == te.min()


def test_split_keeps_ties_together():
    df = _df(10)
    df["TransactionDT"] = 5
    out, _ = time_split(df, 0.6, 0.2)
    assert out["split"].nunique() == 1


def test_bad_fractions():
    with pytest.raises(ValueError):
        time_split(_df(), 0.9, 0.2)


def test_validate():
    validate(_df())
    with pytest.raises(ValueError):
        validate(_df().assign(isFraud=2))
    with pytest.raises(ValueError):
        validate(_df().drop(columns="card1"))
    with pytest.raises(ValueError):
        validate(pd.concat([_df(), _df()]))
