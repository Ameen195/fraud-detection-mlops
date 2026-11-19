"""Evidently data drift (features vs. train) and prediction drift (score PSI vs. a reference)."""

import os

import numpy as np
import pandas as pd
import pandas.core.dtypes.common as _pd_common

os.environ.setdefault("DO_NOT_TRACK", "1")  # keep Evidently offline
# shortcut: evidently 0.7.23 imports a helper pandas 3 removed; drop once evidently supports it
if not hasattr(_pd_common, "infer_dtype_from_object"):
    _pd_common.infer_dtype_from_object = lambda dtype: np.dtype(dtype).type

from evidently import DataDefinition, Dataset, Report  # noqa: E402
from evidently.metrics import ValueDrift  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402


def _dataset(df: pd.DataFrame) -> Dataset:
    cat = [c for c in df if isinstance(df[c].dtype, pd.CategoricalDtype)]
    num = [c for c in df if c not in cat]
    dd = DataDefinition(numerical_columns=num, categorical_columns=cat)
    return Dataset.from_pandas(df, data_definition=dd)


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, html: str | None = None) -> dict:
    """Per-column drift with Evidently defaults (normed Wasserstein / Jensen-Shannon, >= 0.1)."""
    snap = Report([DataDriftPreset(columns=list(ref.columns))]).run(_dataset(cur), _dataset(ref))
    if html:
        snap.save_html(html)
    cols = {}
    for m in snap.dict()["metrics"]:
        c = m["config"]
        if c["type"].endswith("ValueDrift"):
            cols[c["column"]] = {
                "score": float(m["value"]),
                "drifted": m["value"] >= c["threshold"],
            }
    return {"share": float(np.mean([v["drifted"] for v in cols.values()])), "columns": cols}


def score_psi(ref_scores, cur_scores) -> float:
    """PSI of the model score distribution (Evidently's psi method)."""
    ref, cur = (
        pd.DataFrame({"score": np.asarray(s, dtype=float)}) for s in (ref_scores, cur_scores)
    )
    snap = Report([ValueDrift(column="score", method="psi")]).run(_dataset(cur), _dataset(ref))
    return float(snap.dict()["metrics"][0]["value"])
