"""Phase 3: features -> logistic regression, XGBoost, LightGBM (+ Optuna) -> MLflow -> registry.

Picks and registers the best model by validation PR-AUC and writes eval/results/models.{json,md}.
The test split is never used here; it is scored once in Phase 4.
Run from the repo root: uv run python -m fraudguard.models.train
"""

import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudguard.data.load import load_config
from fraudguard.features.build import build_features
from fraudguard.models.evaluate import metrics

NOT_FEATURES = ["TransactionID", "TransactionDT", "isFraud", "split"]


def load_xy(cfg):
    df = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "full.parquet")
    X = pd.concat([df.drop(columns=NOT_FEATURES), build_features(df)], axis=1)
    return X, df["isFraud"], df["split"]


def logreg(seed):
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    cat = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20),
    )
    pre = ColumnTransformer(
        [
            ("num", num, make_column_selector(dtype_exclude="category")),
            ("cat", cat, make_column_selector(dtype_include="category")),
        ]
    )
    return make_pipeline(
        pre, LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    )


def run(name, model, params, data, fit_kw=None):
    """Fit, score on validation, log params/metrics/model to MLflow."""
    Xtr, ytr, Xva, yva = data
    with mlflow.start_run(run_name=name):
        model.fit(Xtr, ytr, **(fit_kw or {}))
        m = metrics(yva, model.predict_proba(Xva)[:, 1])
        mlflow.log_params({**params, "n_features": Xtr.shape[1]})
        mlflow.log_metrics({f"valid_{k}": v for k, v in m.items()})
        # native flavors: mlflow.sklearn's skops check rejects LightGBM/XGBoost objects
        if isinstance(model, lgb.LGBMModel):
            info = mlflow.lightgbm.log_model(model, name="model")
        elif isinstance(model, xgb.XGBModel):
            info = mlflow.xgboost.log_model(model, name="model")
        else:
            trusted = ["numpy.dtype", "sklearn.compose._column_transformer.make_column_selector"]
            info = mlflow.sklearn.log_model(model, name="model", skops_trusted_types=trusted)
    print(f"{name:16s} " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
    return m, info.model_uri


def main() -> None:
    cfg = load_config()
    tc, mc = cfg["train"], cfg["mlflow"]
    seed = tc["seed"]
    results = Path(cfg["paths"]["results_dir"])
    mlflow.set_tracking_uri(mc["tracking_uri"])
    if mlflow.get_experiment_by_name(mc["experiment"]) is None:
        art = Path(mc["artifact_dir"]).resolve().as_uri()
        mlflow.create_experiment(mc["experiment"], artifact_location=art)
    mlflow.set_experiment(mc["experiment"])

    X, y, split = load_xy(cfg)
    tr, va = (split == "train").to_numpy(), (split == "valid").to_numpy()
    ytr, yva = y[tr], y[va]
    w = float((ytr == 0).sum() / (ytr == 1).sum())
    out = {"valid": {}, "class_weight_pos": round(w, 3)}
    uris = {}

    def lgbm(**p):
        return lgb.LGBMClassifier(**p, random_state=seed, verbose=-1)

    # V* check: default class-weighted LightGBM with and without the 339 V columns
    v_cols = [c for c in X.columns if c.startswith("V") and c[1:].isdigit()]
    Xnv = X.drop(columns=v_cols)
    p_w = {"scale_pos_weight": w}
    m_v, uri_v = run("lgbm_weighted_withV", lgbm(**p_w), p_w, (X[tr], ytr, X[va], yva))
    m_nv, uri_nv = run("lgbm_weighted", lgbm(**p_w), p_w, (Xnv[tr], ytr, Xnv[va], yva))
    gap = m_v["pr_auc"] - m_nv["pr_auc"]
    out["v_check"] = {
        "pr_auc_with_v": m_v["pr_auc"],
        "pr_auc_without_v": m_nv["pr_auc"],
        "gap": gap,
        "threshold": tc["v_gap_max"],
    }
    (results / "v_check.json").write_text(json.dumps(out["v_check"], indent=2), "utf-8")
    if tc["v_columns"] == "auto" and gap >= tc["v_gap_max"]:
        sys.exit(f"V* gap {gap:.4f} >= {tc['v_gap_max']}: decide, then set train.v_columns")
    keep_v = tc["v_columns"] == "keep"
    if not keep_v:
        X = Xnv
    del Xnv
    gc.collect()
    out["v_columns"] = "kept" if keep_v else "dropped"
    chosen = (m_v, uri_v) if keep_v else (m_nv, uri_nv)
    out["valid"]["lgbm_weighted"], uris["lgbm_weighted"] = chosen
    data = (X[tr], ytr, X[va], yva)
    out["n_features"] = X.shape[1]

    out["valid"]["logreg"], uris["logreg"] = run(
        "logreg", logreg(seed), {"class_weight": "balanced"}, data
    )
    out["valid"]["xgboost"], uris["xgboost"] = run(
        "xgboost",
        xgb.XGBClassifier(
            tree_method="hist",
            enable_categorical=True,
            scale_pos_weight=w,
            random_state=seed,
            n_jobs=-1,
        ),
        {"scale_pos_weight": w},
        data,
    )
    out["valid"]["lgbm_unweighted"], uris["lgbm_unweighted"] = run(
        "lgbm_unweighted", lgbm(), {"scale_pos_weight": 1.0}, data
    )

    # Optuna: early stopping on the last 20% of train (by time) so validation stays clean
    Xtr = data[0]
    cut = int(len(Xtr) * 0.8)
    fit, es = slice(0, cut), slice(cut, None)

    def params_of(t):
        return dict(
            learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
            num_leaves=t.suggest_int("num_leaves", 16, 256, log=True),
            min_child_samples=t.suggest_int("min_child_samples", 20, 500, log=True),
            subsample=t.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=t.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
            scale_pos_weight=w if t.suggest_categorical("class_weighted", [True, False]) else 1.0,
            n_estimators=1500,
        )

    es_kw = lambda: dict(  # noqa: E731
        eval_set=[(Xtr.iloc[es], ytr.iloc[es])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    def objective(t):
        p = params_of(t)
        model = lgbm(**p).fit(Xtr.iloc[fit], ytr.iloc[fit], **es_kw())
        t.set_user_attr("best_iteration", model.best_iteration_)
        score = metrics(yva, model.predict_proba(data[2])[:, 1])["pr_auc"]
        with mlflow.start_run(run_name=f"optuna_{t.number}", nested=True):
            mlflow.log_params(t.params)
            mlflow.log_metric("valid_pr_auc", score)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with mlflow.start_run(run_name="optuna_lgbm"):
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=tc["optuna_trials"])
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    best_p = params_of(optuna.trial.FixedTrial(study.best_params))
    out["optuna"] = {
        "trials": len(study.trials),
        "best_params": study.best_params,
        "best_iteration": study.best_trial.user_attrs["best_iteration"],
    }
    out["valid"]["lgbm_tuned"], uris["lgbm_tuned"] = run(
        "lgbm_tuned",
        lgbm(**best_p),
        {**study.best_params, "early_stopping": "last20pct_train"},
        (Xtr.iloc[fit], ytr.iloc[fit], data[2], yva),
        es_kw(),
    )

    # tuned model is champion only if it beats the best default model by min_tuning_gain
    pr = {k: v["pr_auc"] for k, v in out["valid"].items()}
    best_default = max((k for k in pr if k != "lgbm_tuned"), key=pr.get)
    tuning_gain = pr["lgbm_tuned"] - pr[best_default]
    best = "lgbm_tuned" if tuning_gain >= tc["min_tuning_gain"] else best_default
    out["optuna"]["gain_over_best_default"] = tuning_gain
    mv = mlflow.register_model(uris[best], mc["model_name"])
    mlflow.MlflowClient().set_registered_model_alias(mc["model_name"], "champion", mv.version)
    out["registered"] = {
        "model": best,
        "name": mc["model_name"],
        "version": int(mv.version),
        "alias": "champion",
    }

    (results / "models.json").write_text(json.dumps(out, indent=2), "utf-8")
    keys = ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "recall_at_5pct_fpr"]
    md = [
        "# Validation metrics (Phase 3)",
        "",
        f"V* check: PR-AUC with V {m_v['pr_auc']:.4f}, without {m_nv['pr_auc']:.4f}"
        f" (gap {gap:.4f}); V columns {out['v_columns']}. Features: {out['n_features']}.",
        "",
        "| model | " + " | ".join(keys) + " |",
        "|---" * (len(keys) + 1) + "|",
    ]
    for name, m in sorted(out["valid"].items(), key=lambda kv: -kv[1]["pr_auc"]):
        md.append(f"| {name} | " + " | ".join(f"{m[k]:.4f}" for k in keys) + " |")
    md += [
        "",
        f"Optuna ({len(study.trials)} trials): PR-AUC gain over best default ({best_default})"
        f" = {tuning_gain:+.4f}, "
        + (
            "kept."
            if best == "lgbm_tuned"
            else f"below min_tuning_gain {tc['min_tuning_gain']}, so no measurable gain; not used."
        ),
        f"Champion: `{mc['model_name']}` v{mv.version} @champion ({best}). Test set not used.",
    ]
    (results / "models.md").write_text("\n".join(md) + "\n", "utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
