"""Phase 5 serving API. Run: uvicorn fraudguard.serving.app:app

Features come from the SAME build_features as training. A transaction is scored together with its
history (earlier rows sharing a card / address / email / device), kept in memory.
shortcut: in-memory history from full.parquet, never updated by scoring; production needs a
feature store (online per-card state) instead.
"""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraudguard.data.load import load_config
from fraudguard.features.build import (
    CARD_COLS,
    ENGINEERED,
    FREQ_COLS,
    NEW_VALUE_COLS,
    apply_categories,
    build_features,
    categorical_levels,
)
from fraudguard.models.final_eval import top_reasons

_GROUP_COLS = list(dict.fromkeys(CARD_COLS + FREQ_COLS + list(NEW_VALUE_COLS.values())))
HIST_COLS = ["TransactionDT", "TransactionAmt", *_GROUP_COLS]
# a history row matters if it shares the full card, or any single freq / new-value column
KEYS = [CARD_COLS] + [[c] for c in _GROUP_COLS if c not in CARD_COLS or c == "card1"]


def _match(h: pd.DataFrame, row, cols) -> np.ndarray:
    m = np.ones(len(h), bool)
    for c in cols:
        m &= h[c].isna().to_numpy() if pd.isna(row[c]) else (h[c] == row[c]).to_numpy()
    return m


class History:
    def __init__(self, df: pd.DataFrame):
        self.df = df[HIST_COLS].sort_values("TransactionDT", kind="stable").reset_index(drop=True)
        self.dt = self.df["TransactionDT"].to_numpy()

    def is_text(self, col: str) -> bool:
        return isinstance(self.df[col].dtype, pd.CategoricalDtype) or self.df[col].dtype == object

    def features(self, req: pd.DataFrame) -> pd.DataFrame:
        """Engineered features for the rows of req (HIST_COLS, default RangeIndex)."""
        h = self.df.iloc[: np.searchsorted(self.dt, req["TransactionDT"].max())]  # strictly earlier
        m = np.zeros(len(h), bool)
        for _, row in req.iterrows():
            for cols in KEYS:
                if len(cols) > 1 or pd.notna(row[cols[0]]):  # a NaN single key never counts
                    m |= _match(h, row, cols)
        both = pd.concat([h[m], req[HIST_COLS]], ignore_index=True) if m.any() else req[HIST_COLS]
        return build_features(both).iloc[-len(req) :].reset_index(drop=True)


class State:
    def __init__(self, model, explainer, levels, history, threshold, info, raw_cols=()):
        self.model, self.explainer, self.levels = model, explainer, levels
        self.history, self.threshold, self.info = history, threshold, info
        self.names = list(model.feature_name_)
        self.base = [c for c in self.names if c not in ENGINEERED]
        # raw_cols: dataset columns the model ignores (dropped V*) are accepted, not scored
        self.known = set(self.base) | set(HIST_COLS) | {"TransactionID"} | set(raw_cols)

    def make_X(self, raw: pd.DataFrame) -> pd.DataFrame:
        req = raw.reindex(columns=HIST_COLS)
        for c in HIST_COLS:
            req[c] = req[c].astype(object) if self.history.is_text(c) else pd.to_numeric(req[c])
        req["TransactionDT"] = req["TransactionDT"].astype("int64")
        X = raw.reindex(columns=self.base)
        for c in self.base:
            if c not in self.levels:
                X[c] = pd.to_numeric(X[c])  # ValueError on non-numeric text
        X = apply_categories(X, self.levels)
        return pd.concat([X, self.history.features(req)], axis=1)[self.names]


def load_state(cfg=None) -> State:
    cfg = cfg or load_config()
    mc, res = cfg["mlflow"], Path(cfg["paths"]["results_dir"])
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", mc["tracking_uri"]))
    model = mlflow.lightgbm.load_model(f"models:/{mc['model_name']}@champion")
    mv = mlflow.MlflowClient().get_model_version_by_alias(mc["model_name"], "champion")
    path = Path(cfg["paths"]["processed_dir"]) / "full.parquet"
    cats = [
        f.name
        for f in pq.read_schema(path)
        if pa.types.is_dictionary(f.type) and f.name in model.feature_name_
    ]
    levels = categorical_levels(pd.read_parquet(path, columns=cats))
    threshold = json.loads((res / "threshold.json").read_text("utf-8"))["threshold"]
    final = res / "final_test.json"
    metrics = json.loads(final.read_text("utf-8")).get("metrics", {}) if final.exists() else {}
    info = {
        "name": mc["model_name"],
        "version": mv.version,
        "alias": "champion",
        "threshold": threshold,
        "test_metrics": metrics,
    }
    history = History(pd.read_parquet(path, columns=HIST_COLS))
    explainer = shap.TreeExplainer(model)
    return State(model, explainer, levels, history, threshold, info, pq.read_schema(path).names)


class Transaction(BaseModel):
    """TransactionDT and TransactionAmt are required; any other raw column is optional."""

    model_config = ConfigDict(extra="allow")
    TransactionDT: int = Field(ge=0, lt=2**31 - 1)
    TransactionAmt: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _extras(self):
        for k, v in (self.model_extra or {}).items():
            if not (v is None or isinstance(v, str | int | float)) or isinstance(v, bool):
                raise ValueError(f"{k}: must be a string, number or null")
        return self


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


def row_to_payload(row: pd.Series) -> dict:
    """One dataset row as an API request body (NaN dropped)."""
    drop = {"isFraud", "split"}
    return {
        k: v if isinstance(v, str) else float(v)
        for k, v in row.items()
        if k not in drop and pd.notna(v)
    }


def create_app(load=load_state) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.s = load()
        yield

    app = FastAPI(title="FraudGuard", lifespan=lifespan)

    def score(txns: list[Transaction]) -> list[dict]:
        s = app.state.s
        rows = [t.model_dump() for t in txns]
        bad = {k for r in rows for k in r} - s.known
        if bad:
            raise HTTPException(422, f"unknown fields: {sorted(bad)}")
        try:
            X = s.make_X(pd.DataFrame(rows))
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        p = s.model.predict_proba(X)[:, 1]
        return [
            {
                "probability": float(pi),
                "decision": "flag" if pi >= s.threshold else "allow",
                "reasons": [
                    {"feature": f, "shap": v} for f, v in top_reasons(s.explainer, X.iloc[[i]])
                ],
            }
            for i, pi in enumerate(p)
        ]

    @app.post("/score")
    def score_one(t: Transaction):
        return score([t])[0]

    @app.post("/score/batch")
    def score_batch(b: Batch):
        return {"results": score(b.transactions)}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/model")
    def model_info():
        return app.state.s.info

    return app


app = create_app()
