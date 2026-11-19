# Validation metrics (Phase 3)

V* check: PR-AUC with V 0.5410, without 0.5337 (gap 0.0074); V columns dropped. Features: 111.

| model | pr_auc | roc_auc | recall_at_1pct_fpr | recall_at_5pct_fpr |
|---|---|---|---|---|
| lgbm_tuned | 0.5483 | 0.9101 | 0.4424 | 0.6439 |
| lgbm_unweighted | 0.5477 | 0.9106 | 0.4487 | 0.6484 |
| lgbm_weighted | 0.5337 | 0.9040 | 0.4255 | 0.6376 |
| xgboost | 0.4974 | 0.8814 | 0.4032 | 0.5968 |
| logreg | 0.2805 | 0.8389 | 0.1841 | 0.3843 |

Optuna (20 trials): PR-AUC gain over best default (lgbm_unweighted) = +0.0006, below min_tuning_gain 0.005, so no measurable gain; not used.
Champion: `fraudguard` v2 @champion (lgbm_unweighted). Test set not used.
