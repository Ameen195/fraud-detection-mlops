"""Phase 4: threshold on validation, one-time test scoring, SHAP, error analysis.

Run from the repo root: uv run python -m fraudguard.models.final_eval
Refuses to run if eval/results/final_test.json exists (the test set is scored once).
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_curve, roc_curve

from fraudguard.data.load import load_config
from fraudguard.models.evaluate import metrics
from fraudguard.models.train import load_xy

SENS_COSTS = [1, 5, 20, 50]
SHAP_SAMPLE = 5000


def best_threshold(y, p, amt, cost):
    """Threshold minimising cost*false_alarms + missed fraud amount (flag p >= thr)."""
    y, p, amt = np.asarray(y), np.asarray(p), np.asarray(amt, dtype=float)
    o = np.argsort(-p, kind="stable")
    ps, ys, a = p[o], y[o], amt[o]
    fp = np.cumsum(ys == 0)
    missed = (ys * a).sum() - np.cumsum(ys * a)
    total = cost * fp + missed
    ok = np.r_[ps[:-1] != ps[1:], True]  # only cut between distinct scores
    k = int(np.flatnonzero(ok)[np.argmin(total[ok])])
    return float(ps[k])


def at_threshold(y, p, amt, thr, cost):
    y, amt = np.asarray(y), np.asarray(amt, dtype=float)
    flag = np.asarray(p) >= thr
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    caught, total = float(amt[flag & (y == 1)].sum()), float(amt[y == 1].sum())
    return {
        "threshold": thr,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(int(y.sum()), 1),
        "false_alarms": fp,
        "fraud_caught": caught,
        "fraud_missed": total - caught,
        "false_alarm_cost": cost * fp,
        "net_saving": caught - cost * fp,
    }


def top_reasons(explainer, row, k=3):
    """Top-k (feature, shap value) for one transaction (1-row DataFrame); reused by the API."""
    sv = _pos(explainer.shap_values(row))[0]
    return [(row.columns[i], float(sv[i])) for i in np.argsort(-np.abs(sv))[:k]]


def _pos(sv):
    return sv[1] if isinstance(sv, list) else sv


def segments(X):
    amt = pd.cut(X["TransactionAmt"], [0, 50, 100, 250, 1000, np.inf], right=False)
    cols = {
        "amount band": amt,
        "ProductCD": X["ProductCD"],
        "card type (card6)": X["card6"],
        "new device for card": X["new_device_for_card"],
        "new email domain for card": X["new_email_for_card"],
    }
    return {k: v.astype(str).fillna("missing") for k, v in cols.items()}  # NaN rows must show


def seg_table(name, s, y, flag):
    d = pd.DataFrame({"s": s.to_numpy(), "y": y, "f": flag})
    d["tp"], d["fa"] = d.f & (d.y == 1), d.f & (d.y == 0)
    g = d.groupby("s")
    t = pd.DataFrame(
        {"rows": g.size(), "fraud": g["y"].sum(), "tp": g["tp"].sum(), "fa": g["fa"].sum()}
    )
    rows = [f"### {name}", "", "| value | rows | fraud | recall | false alarms | FA rate |"]
    rows += ["|---|---|---|---|---|---|"]
    for v, r in t.iterrows():
        rec = f"{r.tp / r.fraud:.3f}" if r.fraud else "n/a"
        rows.append(
            f"| {v} | {int(r.rows)} | {int(r.fraud)} | {rec} | "
            f"{int(r.fa)} | {r.fa / (r.rows - r.fraud):.4f} |"
        )
    return rows + [""]


def main() -> None:
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"])
    if (res / "final_test.json").exists():
        sys.exit("final_test.json exists: the test set was already scored. Not re-scoring.")
    mc, cost = cfg["mlflow"], cfg["business"]["cost_per_false_alarm"]
    mlflow.set_tracking_uri(mc["tracking_uri"])
    model = mlflow.lightgbm.load_model(f"models:/{mc['model_name']}@champion")
    X, y, split = load_xy(cfg)
    X = X[list(model.feature_name_)]
    va, te = (split == "valid").to_numpy(), (split == "test").to_numpy()

    # 1. validation only: threshold + cost sensitivity
    yva, Xva = y[va].to_numpy(), X[va]
    pva, ava = model.predict_proba(Xva)[:, 1], Xva["TransactionAmt"].to_numpy()
    thr = best_threshold(yva, pva, ava, cost)
    sens = {}
    for c in SENS_COSTS:
        sens[c] = at_threshold(yva, pva, ava, best_threshold(yva, pva, ava, c), c)
    th = {
        "cost_per_false_alarm": cost,
        "threshold": thr,
        "chosen_on": "valid",
        "valid": at_threshold(yva, pva, ava, thr, cost),
        "sensitivity_valid": {str(k): v for k, v in sens.items()},
    }
    (res / "threshold.json").write_text(json.dumps(th, indent=2), "utf-8")

    # 2. the one and only test scoring
    yte, Xte = y[te].to_numpy(), X[te]
    pte, ate = model.predict_proba(Xte)[:, 1], Xte["TransactionAmt"].to_numpy()
    proc = Path(cfg["paths"]["processed_dir"])
    pd.DataFrame({"y": yte, "p": pte}).to_parquet(proc / "test_scores.parquet")  # gitignored
    m = metrics(yte, pte)
    biz = at_threshold(yte, pte, ate, thr, cost)
    legit = int((yte == 0).sum())
    biz["false_alarm_rate"] = biz["false_alarms"] / legit
    mv = mlflow.MlflowClient().get_model_version_by_alias(mc["model_name"], "champion").version
    out = {
        "model": f"{mc['model_name']}@champion v{mv}",
        "test_rows": int(te.sum()),
        "legit_rows": legit,
        "metrics": m,
        "at_threshold": biz,
        "cost_per_false_alarm": cost,
    }
    (res / "final_test.json").write_text(json.dumps(out, indent=2), "utf-8")

    md = [
        f"# Test results (Phase 4): {out['model']}",
        "",
        f"Threshold {thr:.4f}, chosen on validation (cost per false alarm = {cost}). "
        "Test set scored once.",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v:.4f} |" for k, v in m.items()],
        f"| precision @ threshold | {biz['precision']:.4f} |",
        f"| recall @ threshold | {biz['recall']:.4f} |",
        "",
        "| business (test) | value |",
        "|---|---|",
        f"| fraud amount caught | {biz['fraud_caught']:,.2f} |",
        f"| fraud amount missed | {biz['fraud_missed']:,.2f} |",
        f"| false alarms | {biz['false_alarms']:,} |",
        f"| false alarm rate (of {legit:,} legitimate test transactions) "
        f"| {biz['false_alarm_rate']:.2%} |",
        f"| false alarm cost | {biz['false_alarm_cost']:,.2f} |",
        f"| net saving (caught - false alarm cost) | {biz['net_saving']:,.2f} |",
        "",
        "Assumptions: the cost of 5 per false alarm assumes a flag triggers a cheap step-up "
        "check (like an OTP), not a blocked payment. If a flag blocks the payment, the real "
        "cost is higher; see the cost sensitivity table. Recall @5% FPR "
        f"({m['recall_at_5pct_fpr']:.4f}) is the stricter operating point: it limits false "
        f"alarms to 5% of legitimate transactions, versus {biz['false_alarm_rate']:.2%} at the "
        "chosen threshold.",
        "",
        "## Cost sensitivity (validation only)",
        "",
        "| cost per false alarm | threshold | recall | precision | false alarms |",
        "|---|---|---|---|---|",
        *[
            f"| {c} | {s['threshold']:.4f} | {s['recall']:.4f} | {s['precision']:.4f} | "
            f"{s['false_alarms']:,} |"
            for c, s in sens.items()
        ],
    ]
    (res / "final_test.md").write_text("\n".join(md) + "\n", "utf-8")

    # 3. plots
    pr, rc, _ = precision_recall_curve(yte, pte)
    plt.figure(figsize=(5, 4))
    plt.plot(rc, pr)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"PR curve (test, AP={m['pr_auc']:.3f})")
    plt.tight_layout()
    plt.savefig(res / "pr_curve.png", dpi=120)
    plt.close()
    fpr, tpr, _ = roc_curve(yte, pte)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.xscale("log")
    plt.xlim(1e-4, 1)
    for f in (0.01, 0.05):
        plt.axvline(f, ls=":", c="gray")
    plt.xlabel("false-positive rate (log)")
    plt.ylabel("recall")
    plt.title("Recall vs FPR (test)")
    plt.tight_layout()
    plt.savefig(res / "recall_vs_fpr.png", dpi=120)
    plt.close()

    # 4. SHAP (global, on a validation sample)
    ex = shap.TreeExplainer(model.booster_)
    samp = Xva.sample(min(SHAP_SAMPLE, len(Xva)), random_state=cfg["train"]["seed"])
    sv = _pos(ex.shap_values(samp))
    imp = pd.Series(np.abs(sv).mean(0), index=samp.columns).sort_values(ascending=False)
    (res / "shap_global.json").write_text(json.dumps(imp.head(30).to_dict(), indent=2), "utf-8")
    top = imp.head(20)[::-1]
    plt.figure(figsize=(6, 6))
    plt.barh(top.index, top.values)
    plt.xlabel("mean |SHAP| (validation sample)")
    plt.tight_layout()
    plt.savefig(res / "shap_global.png", dpi=120)
    plt.close()

    # 5. error analysis (from the test predictions already computed)
    flag = pte >= thr
    miss, fa = (yte == 1) & ~flag, (yte == 0) & flag
    ea = [
        "# Error analysis (test set)",
        "",
        f"Threshold {thr:.4f}. Fraud: {int(yte.sum()):,}; missed: {int(miss.sum()):,}; "
        f"false alarms: {int(fa.sum()):,}. Tables are generated by "
        "`fraudguard.models.final_eval`.",
        "",
    ]
    for name, s in segments(Xte).items():
        ea += seg_table(name, s, yte, flag)
    ea += [
        "### Top SHAP features (validation sample)",
        "",
        *[f"- {k}: {v:.4f}" for k, v in imp.head(10).items()],
        "",
    ]
    Path("docs").mkdir(exist_ok=True)
    Path("docs/ERROR_ANALYSIS.md").write_text("\n".join(ea), "utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
