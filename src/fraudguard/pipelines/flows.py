"""Phase 6: Prefect flows for monitoring and retraining, plus a replay of later traffic.

- monitor_batch(week): score one weekly batch with the static and the live (retrained) champion,
  Evidently feature drift vs. train, score PSI vs. week 0, and performance for weeks whose
  labels have just arrived (label_delay_days after the transaction).
- retrain(week, reason): champion/challenger using only labels known at that time.
- replay(): validation + test as weekly production traffic in time order; writes
  eval/results/monitoring.{json,md} and replay_pr_auc.png.

Replay models go to the `fraudguard_replay` registry name; the served `fraudguard` is untouched.
Run from the repo root: uv run python -m fraudguard.pipelines.flows
"""

import json
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from prefect import flow

from fraudguard.data.load import load_config
from fraudguard.models.evaluate import metrics
from fraudguard.models.train import load_xy
from fraudguard.monitoring.drift import feature_drift, score_psi
from fraudguard.monitoring.performance import (
    DAY,
    labels_known_week,
    retrain_masks,
    should_promote,
    week_edges,
)

KEYS = ["pr_auc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"]  # threshold-free only
SCORES_DIR = Path("data/processed/replay_scores")  # gitignored
REPORTS_DIR = Path("eval/drift_reports")  # gitignored (Evidently HTML is large)


@cache
def _ctx():
    """Data, static champion and replay settings, loaded once per process."""
    cfg = load_config()
    mc, mon = cfg["mlflow"], cfg["monitoring"]
    mlflow.set_tracking_uri(mc["tracking_uri"])
    static = mlflow.lightgbm.load_model(f"models:/{mc['model_name']}@champion")
    X, y, split = load_xy(cfg)
    X = X[list(static.feature_name_)]
    dt = pd.read_parquet(
        Path(cfg["paths"]["processed_dir"]) / "full.parquet", columns=["TransactionDT"]
    )["TransactionDT"].to_numpy(np.int64)
    split = split.to_numpy()
    tr = split == "train"
    prod = ~tr
    edges = week_edges(
        int(dt[prod].min()),
        int(dt[split == "test"].min()),
        int(dt[prod].max()),
        mon["batch_days"] * DAY,
    )
    drift_cols = list(json.loads(Path("eval/results/shap_global.json").read_text("utf-8")))
    drift_cols = drift_cols[: mon["drift_features"]]
    ref = X.loc[tr, drift_cols].sample(mon["reference_rows"], random_state=cfg["train"]["seed"])
    return SimpleNamespace(
        cfg=cfg,
        mon=mon,
        static=static,
        static_version=mlflow.MlflowClient()
        .get_model_version_by_alias(mc["model_name"], "champion")
        .version,
        X=X,
        y=y.to_numpy(),
        dt=dt,
        split=split,
        train_end=int(dt[tr].max()),
        edges=edges,
        delay=mon["label_delay_days"] * DAY,
        drift_cols=drift_cols,
        ref=ref,
    )


def _champion():
    """The live replay champion, loaded from the registry like production would."""
    name = _ctx().mon["replay_model_name"]
    mv = mlflow.MlflowClient().get_model_version_by_alias(name, "champion")
    return mlflow.lightgbm.load_model(f"models:/{name}@champion"), mv


def _week_scores(week: int) -> pd.DataFrame:
    return pd.read_parquet(SCORES_DIR / f"week_{week:02d}.parquet")


def _perf(df: pd.DataFrame, col: str) -> dict:
    m = metrics(df["y"], df[col])
    return {k: m[k] for k in KEYS}


@flow(log_prints=True)
def monitor_batch(week: int) -> dict:
    c = _ctx()
    lo, hi = c.edges[week], c.edges[week + 1]
    rows = (c.dt >= lo) & (c.dt < hi)
    Xb = c.X[rows]
    model, mv = _champion()
    p_static = c.static.predict_proba(Xb)[:, 1]
    scores = pd.DataFrame(
        {"y": c.y[rows], "static": p_static, "retrain": model.predict_proba(Xb)[:, 1]}
    )
    scores.to_parquet(SCORES_DIR / f"week_{week:02d}.parquet")

    drift = feature_drift(c.ref, Xb[c.drift_cols], html=str(REPORTS_DIR / f"week_{week:02d}.html"))
    psi = score_psi(_week_scores(0)["static"], p_static)
    alarm = drift["share"] >= c.mon["drift_share_alarm"] or psi >= c.mon["score_psi_alarm"]

    # performance for every earlier week whose labels have all arrived by the end of this week
    arrived = {
        j: _perf(_week_scores(j), "static")
        for j in range(week + 1)
        if labels_known_week(c.edges, j, c.delay) == week
    }
    for j, m in arrived.items():
        print(f"labels arrived for week {j}: static PR-AUC {m['pr_auc']:.4f}")
    print(
        f"week {week}: rows {int(rows.sum())}, drift share {drift['share']:.2f}, "
        f"score PSI {psi:.3f}, alarm {alarm}, live champion v{mv.version}"
    )
    return {
        "week": week,
        "split": str(c.split[rows][0]),
        "day_start": round(lo / DAY, 1),
        "day_end": round(hi / DAY, 1),
        "rows": int(rows.sum()),
        "drift_share": drift["share"],
        "drifted": sorted(k for k, v in drift["columns"].items() if v["drifted"]),
        "score_psi": psi,
        "alarm": bool(alarm),
        "live_champion_version": int(mv.version),
        "labels_arrived_for_weeks": list(arrived),
    }


@flow(log_prints=True)
def retrain(week: int, reason: str) -> dict:
    """Champion/challenger at the end of `week`, using only labels known by then."""
    c = _ctx()
    now = c.edges[week + 1] - 1
    train_m, eval_m, eval_start = retrain_masks(c.dt, now, c.delay, c.mon["eval_window_days"] * DAY)
    champ, mv = _champion()
    out = {"week": week, "reason": reason, "champion_version": int(mv.version)}
    champ_end = int(mv.tags["train_end_dt"])
    if eval_start < champ_end:
        print(f"week {week}: retrain skipped, no labeled data newer than the champion's training")
        return {**out, "decision": "skipped", "why": "eval window overlaps champion training data"}

    Xtr = c.X[train_m].copy()
    for col in Xtr.select_dtypes("category"):
        Xtr[col] = Xtr[col].cat.remove_unused_categories()  # levels from training data only
    challenger = lgb.LGBMClassifier(**champ.get_params()).fit(Xtr, c.y[train_m])
    Xev, yev = c.X[eval_m], c.y[eval_m]
    m_champ = {k: v for k, v in metrics(yev, champ.predict_proba(Xev)[:, 1]).items() if k in KEYS}
    m_chall = {
        k: v for k, v in metrics(yev, challenger.predict_proba(Xev)[:, 1]).items() if k in KEYS
    }
    promote = should_promote(m_champ, m_chall, c.mon["min_promotion_gain"])

    name = c.mon["replay_model_name"]
    with mlflow.start_run(run_name=f"replay_retrain_week{week:02d}"):
        mlflow.log_params(
            {
                "week": week,
                "reason": reason,
                "train_rows": int(train_m.sum()),
                "train_end_day": round(eval_start / DAY, 1),
                "eval_rows": int(eval_m.sum()),
                "label_delay_days": c.mon["label_delay_days"],
            }
        )
        mlflow.log_metrics({f"champion_{k}": v for k, v in m_champ.items()})
        mlflow.log_metrics({f"challenger_{k}": v for k, v in m_chall.items()})
        info = mlflow.lightgbm.log_model(challenger, name="model")
    new = mlflow.register_model(info.model_uri, name)
    client = mlflow.MlflowClient()
    client.set_model_version_tag(name, new.version, "train_end_dt", str(int(eval_start)))
    client.set_registered_model_alias(name, "champion" if promote else "candidate", new.version)
    decision = "promoted" if promote else "kept as candidate"
    print(
        f"week {week}: challenger v{new.version} PR-AUC {m_chall['pr_auc']:.4f} vs champion "
        f"v{mv.version} {m_champ['pr_auc']:.4f} on days {eval_start / DAY:.1f}-"
        f"{(now - c.delay) / DAY:.1f}"
        f" -> {decision}"
    )
    return {
        **out,
        "decision": decision,
        "challenger_version": int(new.version),
        "train_rows": int(train_m.sum()),
        "train_end_day": round(eval_start / DAY, 1),
        "eval_days": [round(eval_start / DAY, 1), round((now - c.delay) / DAY, 1)],
        "decided_at_day": round(now / DAY, 1),
        "champion": m_champ,
        "challenger": m_chall,
    }


def _start_replay_registry(c) -> int:
    """Start the replay registry fresh: the served champion becomes v1 and the live champion."""
    mc, name = c.cfg["mlflow"], c.mon["replay_model_name"]
    client = mlflow.MlflowClient()
    if client.search_registered_models(filter_string=f"name = '{name}'"):
        client.delete_registered_model(name)  # replay-owned name; the served model is untouched
    src = client.get_model_version_by_alias(mc["model_name"], "champion")
    mv = mlflow.register_model(src.source, name)
    client.set_model_version_tag(name, mv.version, "train_end_dt", str(c.train_end))
    client.set_registered_model_alias(name, "champion", mv.version)
    return int(mv.version)


@flow(log_prints=True)
def replay() -> dict:
    c = _ctx()
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(c.cfg["mlflow"]["experiment"])  # created in Phase 3
    start_version = _start_replay_registry(c)

    weeks, decisions, last = [], [], 0  # last: weeks elapsed at the last retrain that ran
    for w in range(len(c.edges) - 1):
        r = monitor_batch(w)
        weeks.append(r)
        due = (w + 1) - last >= c.mon["retrain_every_weeks"]
        if r["alarm"] or due:
            d = retrain(w, "drift" if r["alarm"] else "schedule")
            decisions.append(d)
            if d["decision"] != "skipped":
                last = w + 1

    # week by week on the eventual true labels: static champion vs retraining policy
    for r in weeks:
        s = _week_scores(r["week"])
        r["fraud_rate"] = float(s["y"].mean())
        r["static"], r["retrain"] = _perf(s, "static"), _perf(s, "retrain")
        r["labels_known_at_week"] = labels_known_week(c.edges, r["week"], c.delay)
    pooled = {}
    for sp in ("valid", "test"):
        s = pd.concat([_week_scores(r["week"]) for r in weeks if r["split"] == sp])
        pooled[sp] = {
            "static": _perf(s, "static"),
            "retrain": _perf(s, "retrain"),
            "mean_drift_share": float(
                np.mean([r["drift_share"] for r in weeks if r["split"] == sp])
            ),
            "alarm_weeks": sum(r["alarm"] for r in weeks if r["split"] == sp),
            "weeks": sum(r["split"] == sp for r in weeks),
        }
    out = {
        "settings": dict(c.mon),
        "static_model": f"{c.cfg['mlflow']['model_name']} v{c.static_version}",
        "replay_start_version": start_version,
        "drift_reference": f"{c.mon['reference_rows']} random train rows; score PSI vs week 0",
        "pooled": pooled,
        "weeks": weeks,
        "retrain_decisions": decisions,
    }
    res = Path(c.cfg["paths"]["results_dir"])
    (res / "monitoring.json").write_text(json.dumps(out, indent=2), "utf-8")
    (res / "monitoring.md").write_text(_markdown(out), "utf-8")
    _chart(weeks, decisions, res / "replay_pr_auc.png")
    print(_markdown(out))
    return out


def _markdown(o: dict) -> str:
    s = o["settings"]
    md = [
        "# Monitoring replay (Phase 6)",
        "",
        f"Validation + test replayed as weekly production batches. Static model: "
        f"{o['static_model']}. Label delay {s['label_delay_days']} days. Drift: Evidently on the "
        f"top {s['drift_features']} SHAP features vs {o['drift_reference']}. Alarm if drift share"
        f" >= {s['drift_share_alarm']} or score PSI >= {s['score_psi_alarm']}. PR-AUC uses the "
        "eventual true labels; 'labels at' is the week those labels became known.",
        "",
        "| week | split | days | rows | fraud rate | drift share | score PSI | alarm "
        "| labels at | PR-AUC static | PR-AUC retrain | live model |",
        "|---" * 12 + "|",
    ]
    for r in o["weeks"]:
        known = r["labels_known_at_week"]
        md.append(
            f"| {r['week']} | {r['split']} | {r['day_start']:.0f}-{r['day_end']:.0f} "
            f"| {r['rows']:,} | {r['fraud_rate']:.4f} | {r['drift_share']:.2f} "
            f"| {r['score_psi']:.3f} "
            f"| {'yes' if r['alarm'] else 'no'} | {'after replay' if known is None else known} "
            f"| {r['static']['pr_auc']:.4f} | {r['retrain']['pr_auc']:.4f} "
            f"| v{r['live_champion_version']} |"
        )
    md += [
        "",
        "| pooled | weeks | alarm weeks | mean drift share | model "
        "| PR-AUC | recall@1%FPR | recall@5%FPR |",
        "|---" * 8 + "|",
    ]
    for sp, p in o["pooled"].items():
        for model in ("static", "retrain"):
            m = p[model]
            md.append(
                f"| {sp} | {p['weeks']} | {p['alarm_weeks']} | {p['mean_drift_share']:.2f} "
                f"| {model} | {m['pr_auc']:.4f} | {m['recall_at_1pct_fpr']:.4f} "
                f"| {m['recall_at_5pct_fpr']:.4f} |"
            )
    md += ["", "## Retrain decisions", ""]
    for d in o["retrain_decisions"]:
        line = f"- end of week {d['week']} ({d['reason']}): {d['decision']}"
        if "challenger" in d:
            line += (
                f"; challenger v{d['challenger_version']} PR-AUC {d['challenger']['pr_auc']:.4f}"
                f" vs champion v{d['champion_version']} {d['champion']['pr_auc']:.4f} on days "
                f"{d['eval_days'][0]:.0f}-{d['eval_days'][1]:.0f} (trained up to day "
                f"{d['train_end_day']:.0f})"
            )
        md.append(line)
    return "\n".join(md) + "\n"


def _chart(weeks: list, decisions: list, path: Path) -> None:
    x = [r["week"] for r in weeks]
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"  # dataviz palette
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for key, color, label in (
        ("static", blue, "static champion"),
        ("retrain", orange, "retraining"),
    ):
        y = [r[key]["pr_auc"] for r in weeks]
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=label)
        ax.annotate(
            label,
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=ink,
        )
    first_test = next(r["week"] for r in weeks if r["split"] == "test")
    ax.axvline(first_test - 0.5, color=muted, ls="--", lw=1)
    ax.text(first_test - 0.4, ax.get_ylim()[1], "test period", va="top", fontsize=8, color=muted)
    for d in decisions:
        if d["decision"] == "promoted":
            ax.axvline(d["week"] + 0.5, color=orange, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xlabel("week of replay (dotted: new model promoted)", color=muted, fontsize=9)
    ax.set_ylabel("PR-AUC (true labels)", color=muted, fontsize=9)
    ax.set_title("Weekly PR-AUC: static champion vs retraining", fontsize=10, color=ink, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_xlim(x[0] - 0.5, x[-1] + 1.8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    replay()
