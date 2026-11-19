import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pandas.testing import assert_frame_equal

from fraudguard.features.build import ENGINEERED, build_features
from fraudguard.serving.app import HIST_COLS, History, State, create_app
from tests.test_features import _synthetic


def _serve_features(hist: History, df: pd.DataFrame, rows) -> pd.DataFrame:
    req = df.loc[rows, HIST_COLS].astype(
        {c: object for c in ["card6", "P_emaildomain", "R_emaildomain", "DeviceInfo"]}
    )
    return hist.features(req.reset_index(drop=True)).set_axis(rows)


def test_train_serve_features_identical():
    """Same rows -> same features offline (whole frame) and in serving (rows + their history)."""
    df = _synthetic()
    offline = build_features(df)
    rows = list(range(0, len(df), 47)) + [5, 6, 7]  # includes neighbours with tied timestamps
    for r in rows:
        for hist_rows in (df, df[df.index < r]):  # replay (row already in history) / production
            assert_frame_equal(
                _serve_features(History(hist_rows), df, [r]), offline.loc[[r]], check_exact=False
            )
    # a batch sees its own earlier rows; the rest of the history is everything else
    assert_frame_equal(
        _serve_features(History(df.drop(index=rows)), df, rows),
        offline.loc[rows],
        check_exact=False,
    )


class _Model:
    feature_name_ = ["TransactionAmt", "card6", *ENGINEERED]

    def predict_proba(self, X):
        assert list(X.columns) == self.feature_name_
        p = np.full(len(X), 0.5)
        return np.c_[1 - p, p]


class _Explainer:
    def shap_values(self, row):
        return np.arange(row.shape[1], dtype=float)[None, :]


@pytest.fixture(scope="module")
def client():
    st = State(
        _Model(),
        _Explainer(),
        {"card6": ["credit", "debit"]},
        History(_synthetic()),
        0.4,
        {"version": "1"},
    )
    with TestClient(create_app(lambda: st)) as c:
        yield c


def test_score_shape_and_decision(client):
    r = client.post("/score", json={"TransactionDT": 10**6, "TransactionAmt": 20, "card6": "debit"})
    body = r.json()
    assert r.status_code == 200 and body["decision"] == "flag" and body["probability"] == 0.5
    assert len(body["reasons"]) == 3


def test_batch_health_model(client):
    t = {"TransactionDT": 10**6, "TransactionAmt": 20}
    assert len(client.post("/score/batch", json={"transactions": [t, t]}).json()["results"]) == 2
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model").json()["version"] == "1"


@pytest.mark.parametrize(
    "bad",
    [
        {"TransactionDT": 1},  # missing amount
        {"TransactionDT": -1, "TransactionAmt": 1},
        {"TransactionDT": 1, "TransactionAmt": "x"},
        {"TransactionDT": 1, "TransactionAmt": 1, "nope": 1},  # unknown column
        {"TransactionDT": 1, "TransactionAmt": 1, "card6": [1]},
    ],
)
def test_validation_422(client, bad):
    assert client.post("/score", json=bad).status_code == 422
