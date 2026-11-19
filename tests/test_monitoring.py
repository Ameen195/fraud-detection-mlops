import numpy as np
import pandas as pd

from fraudguard.monitoring.performance import (
    DAY,
    labels_known_week,
    retrain_masks,
    should_promote,
    week_edges,
)


def test_week_edges_align_to_split():
    start, split, end = 100, 100 + 10 * DAY, 100 + 24 * DAY
    e = week_edges(start, split, end, 7 * DAY)
    assert e[0] == start and e[-1] == end + 1
    assert split in e  # no batch spans the validation/test boundary
    assert e[1] - e[0] == 3 * DAY  # partial first batch, then full weeks
    assert all(b - a <= 7 * DAY for a, b in zip(e, e[1:]))


def test_retrain_never_uses_labels_newer_than_delay():
    dt = np.arange(0, 100 * DAY, 3600)
    now, delay, window = 80 * DAY, 30 * DAY, 14 * DAY
    train, ev, eval_start = retrain_masks(dt, now, delay, window)
    assert dt[train | ev].max() <= now - delay  # no label that has not arrived yet
    assert dt[train].max() <= eval_start < dt[ev].min()  # eval window is after training
    assert dt[ev].max() - dt[ev].min() < window
    assert not (train & ev).any()


def test_labels_known_week():
    e = [0, 7 * DAY, 14 * DAY, 21 * DAY, 28 * DAY]
    assert labels_known_week(e, 0, 10 * DAY) == 2  # week 0 ends day 7, labels by day 17
    assert labels_known_week(e, 3, 10 * DAY) is None  # after the replay ends


def test_should_promote():
    champ = {"pr_auc": 0.50}
    assert should_promote(champ, {"pr_auc": 0.506}, 0.005)
    assert not should_promote(champ, {"pr_auc": 0.503}, 0.005)


def test_drift_detects_shift():
    from fraudguard.monitoring.drift import feature_drift, score_psi

    rng = np.random.default_rng(0)

    def frame(shift):
        return pd.DataFrame(
            {
                "a": rng.normal(shift, 1, 3000),
                "b": rng.normal(0, 1, 3000),
                "c": pd.Categorical(rng.choice(["x", "y"], 3000)),
            }
        )

    ref = frame(0)
    assert feature_drift(ref, frame(0))["share"] == 0
    d = feature_drift(ref, frame(2))
    assert d["columns"]["a"]["drifted"] and not d["columns"]["b"]["drifted"]
    assert score_psi(rng.random(3000), rng.random(3000)) < 0.1
    assert score_psi(rng.random(3000), rng.random(3000) ** 4) > 0.1
