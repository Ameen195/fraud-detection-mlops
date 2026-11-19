"""Label timing and champion/challenger rules for the replay. Pure functions, no I/O.

A label for a transaction at time t is known at time `now` only if t + label_delay <= now.
"""

import numpy as np

DAY = 86400


def week_edges(start: int, split_at: int, end: int, step: int) -> list[int]:
    """Batch edges [e_i, e_i+1) covering [start, end], aligned to split_at so no batch spans it."""
    first = split_at - ((split_at - start) // step) * step
    edges = list(range(first, end + 1, step))
    if edges[0] > start:
        edges.insert(0, start)
    return edges + [end + 1]


def labels_known_week(edges: list[int], week: int, delay: int) -> int | None:
    """First batch at whose end every label of `week` has arrived (None: not within the replay)."""
    ready = edges[week + 1] - 1 + delay
    return next((w for w in range(week, len(edges) - 1) if edges[w + 1] - 1 >= ready), None)


def retrain_masks(dt: np.ndarray, now: int, delay: int, window: int):
    """Train on labeled rows before the eval window; eval on the newest `window` of labeled rows."""
    cutoff = now - delay  # newest transaction whose label is known at `now`
    eval_start = cutoff - window
    return dt <= eval_start, (dt > eval_start) & (dt <= cutoff), eval_start


def should_promote(champion: dict, challenger: dict, min_gain: float) -> bool:
    """Threshold-free rule: challenger PR-AUC must beat the champion's by min_gain."""
    return challenger["pr_auc"] >= champion["pr_auc"] + min_gain
