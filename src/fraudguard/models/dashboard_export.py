"""Phase 7 data export for the dashboard. Run once after final_eval:
python -m fraudguard.models.dashboard_export

Validation: adds the "highest" risk level (threshold at 1% FPR) to threshold.json and writes the
threshold curve. Test: only reads the saved test_scores.parquet (no re-scoring) for the PR curve.
Samples (raw competition rows) go to data/processed, which is gitignored.
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from fraudguard.data.load import load_config
from fraudguard.models.evaluate import metrics
from fraudguard.models.final_eval import at_threshold
from fraudguard.models.train import load_xy
from fraudguard.serving.app import row_to_payload

CURVE_QUANTILES = np.linspace(0.5, 0.9995, 200)  # below the median score nobody would set a policy
# per risk band: (true fraud, legit) sample counts; mixed on purpose so misses and false alarms show
SAMPLE_MIX = {"highest": (6, 4), "elevated": (5, 7), "normal": (2, 12)}


def threshold_at_fpr(y, p, max_fpr):
    """Lowest threshold (flag p >= thr) whose false-positive rate is <= max_fpr."""
    fpr, _, thr = roc_curve(y, p)
    return float(thr[np.flatnonzero(fpr <= max_fpr)[-1]])


def point(y, p, amt, thr, cost):
    r = at_threshold(y, p, amt, thr, cost)
    flag = p >= thr
    r["caught_n"] = int((flag & (y == 1)).sum())
    r["missed_n"] = int(y.sum()) - r["caught_n"]
    r["false_alarm_rate"] = r["false_alarms"] / int((y == 0).sum())
    return r


def pr_points(y, p, thr):
    flag = p >= thr
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    return {
        "threshold": thr,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / int(y.sum()),
        "false_alarm_rate": fp / int((y == 0).sum()),
    }


def main() -> None:
    cfg = load_config()
    res, proc = Path(cfg["paths"]["results_dir"]), Path(cfg["paths"]["processed_dir"])
    th = json.loads((res / "threshold.json").read_text("utf-8"))
    final = json.loads((res / "final_test.json").read_text("utf-8"))
    cost, elevated = th["cost_per_false_alarm"], th["threshold"]
    mc = cfg["mlflow"]
    mlflow.set_tracking_uri(mc["tracking_uri"])
    model = mlflow.lightgbm.load_model(f"models:/{mc['model_name']}@champion")
    X, y, split = load_xy(cfg)
    va = (split == "valid").to_numpy()
    Xva, yva = X.loc[va, list(model.feature_name_)], y[va].to_numpy()
    pva, ava = model.predict_proba(Xva)[:, 1], Xva["TransactionAmt"].to_numpy()

    # 1. "highest" level, validation only
    highest = threshold_at_fpr(yva, pva, 0.01)
    hi = point(yva, pva, ava, highest, cost)
    assert hi["false_alarm_rate"] <= 0.01 and highest > elevated, hi
    assert np.isclose(hi["recall"], metrics(yva, pva)["recall_at_1pct_fpr"]), hi
    assert np.isclose(point(yva, pva, ava, elevated, cost)["net_saving"], th["valid"]["net_saving"])
    th["highest"] = {"chosen_on": "valid", "max_false_alarm_rate": 0.01, **hi}
    (res / "threshold.json").write_text(json.dumps(th, indent=2), "utf-8")

    # 2. threshold curve on validation (slider grid includes both chosen thresholds exactly)
    grid = np.unique(np.r_[np.quantile(pva, CURVE_QUANTILES), elevated, highest])
    curve = {
        "split": "valid",
        "cost_per_false_alarm": cost,
        "fraud_n": int(yva.sum()),
        "legit_n": int((yva == 0).sum()),
        "elevated": elevated,
        "highest": highest,
        "points": [point(yva, pva, ava, float(t), cost) for t in grid],
    }
    at_e = next(q for q in curve["points"] if q["threshold"] == elevated)
    assert at_e["false_alarms"] == th["valid"]["false_alarms"], at_e
    (res / "threshold_curve.json").write_text(json.dumps(curve), "utf-8")

    # 3. test PR curve from the saved scores (no new scoring), downsampled evenly in recall
    ts = pd.read_parquet(proc / "test_scores.parquet")
    yte, pte = ts["y"].to_numpy(), ts["p"].to_numpy()
    pr, rc, _ = precision_recall_curve(yte, pte)
    pr, rc = pr[::-1], rc[::-1]  # recall ascending
    idx = np.unique(np.searchsorted(rc, np.linspace(0, 1, 300)).clip(0, len(rc) - 1))
    e_te = pr_points(yte, pte, elevated)
    assert np.isclose(e_te["recall"], final["at_threshold"]["recall"]), e_te
    out = {
        "split": "test",
        "pr_auc": final["metrics"]["pr_auc"],
        "base_rate": float(yte.mean()),
        "points": [[float(rc[i]), float(pr[i])] for i in idx],
        "operating_points": {"elevated": e_te, "highest": pr_points(yte, pte, highest)},
    }
    (res / "pr_curve.json").write_text(json.dumps(out), "utf-8")

    # 4. sample transactions for the review queue (gitignored: raw rows are not redistributed)
    raw = pd.read_parquet(proc / "full.parquet")
    te = raw[raw["split"] == "test"].reset_index(drop=True)
    assert len(te) == len(ts) and (te["isFraud"].to_numpy() == yte).all()
    band = np.where(pte >= highest, "highest", np.where(pte >= elevated, "elevated", "normal"))
    rng = np.random.default_rng(cfg["train"]["seed"])
    picks = []
    for b, (n_fraud, n_legit) in SAMPLE_MIX.items():
        for label, n in ((1, n_fraud), (0, n_legit)):
            pool = np.flatnonzero((band == b) & (yte == label))
            picks += list(rng.choice(pool, min(n, len(pool)), replace=False))
    samples = [
        {"id": int(te.at[i, "TransactionID"]), "label": int(yte[i]), "payload": row_to_payload(r)}
        for i, r in te.loc[sorted(picks)].iterrows()
    ]
    (proc / "dashboard_samples.json").write_text(json.dumps(samples), "utf-8")
    print(f"highest={highest:.4f} elevated={elevated:.4f} samples={len(samples)}")


if __name__ == "__main__":
    main()
