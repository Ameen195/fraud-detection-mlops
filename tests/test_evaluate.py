import pytest

from fraudguard.models.evaluate import metrics


def test_perfect_scores():
    m = metrics([0, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9])
    assert m == {k: 1.0 for k in m}


def test_recall_at_fpr():
    # 100 negatives, 4 positives; one negative outscores half the positives
    y = [0] * 100 + [1] * 4
    p = [0.0] * 99 + [0.7] + [0.9, 0.8, 0.6, 0.5]
    m = metrics(y, p)
    assert m["recall_at_1pct_fpr"] == pytest.approx(1.0)  # 1/100 FP allowed -> all caught
    assert metrics(y, [0.0] * 98 + [0.7, 0.7] + [0.9, 0.8, 0.6, 0.5])["recall_at_1pct_fpr"] == 0.5


def test_threshold_cost():
    from fraudguard.models.final_eval import at_threshold, best_threshold

    y, p, a = [0, 0, 1, 1], [0.1, 0.8, 0.7, 0.9], [10, 10, 100, 5]
    assert best_threshold(y, p, a, 5) == 0.7  # catching the 100 beats 1 false alarm (5)
    assert best_threshold(y, p, a, 500) == 0.9  # false alarms too costly: flag only the top
    r = at_threshold(y, p, a, 0.7, 5)
    assert (r["false_alarms"], r["fraud_caught"], r["fraud_missed"]) == (1, 105.0, 0.0)
