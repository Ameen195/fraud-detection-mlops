import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def metrics(y, p) -> dict:
    """PR-AUC (primary), ROC-AUC, and recall at 1% / 5% false-positive rate."""
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "recall_at_1pct_fpr": float(np.max(tpr[fpr <= 0.01])),
        "recall_at_5pct_fpr": float(np.max(tpr[fpr <= 0.05])),
    }
