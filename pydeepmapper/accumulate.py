"""Iterate-N accumulation, the PURE aggregation of per-pass findings.

DeepMapper runs the ``arrange -> train -> attribute`` loop N times and accumulates
per-feature findings into robust, stability-scored attributions. These functions
turn a list of per-pass results into the final ranking + stability statistics.
No torch, no IO, stdlib only, so they are testable in isolation.

These functions are property-tested in isolation
(selection frequency = the stability-selection statistic; median importance; Borda
mean rank; the per-pass min_accuracy quality gate).
"""
from __future__ import annotations

from typing import List, Sequence


def accept_pass(accuracy: float, min_accuracy: float) -> bool:
    """The per-pass quality gate: a pass contributes only if it cleared the bar."""
    return accuracy >= min_accuracy


def ranks_from_importances(importances: Sequence[float]) -> List[int]:
    """Convert an importance vector to ranks (rank 1 = most important).

    Ties are broken by original index (stable), so the result is a permutation of
    ``1..n`` and feeds :func:`mean_rank` / top-k selection directly.
    """
    order = sorted(range(len(importances)), key=lambda i: (-importances[i], i))
    ranks = [0] * len(importances)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def top_k_set(importances: Sequence[float], k: int) -> List[int]:
    """Indices of the ``k`` most important features in one pass (descending)."""
    order = sorted(range(len(importances)), key=lambda i: (-importances[i], i))
    return order[:k]


def selection_frequency(top_sets: Sequence[Sequence[int]], n_features: int) -> List[float]:
    """Per-feature fraction of passes that selected it (the stability statistic).

    ``top_sets``: one selected-index list per accepted pass. Returns a value in
    ``[0, 1]`` per feature: how often it landed in the top-k. Empty input -> all 0.
    """
    freq = [0.0] * n_features
    n = len(top_sets)
    if n == 0:
        return freq
    counts = [0] * n_features
    for s in top_sets:
        for i in s:
            counts[i] += 1
    return [c / n for c in counts]


def mean_rank(rank_lists: Sequence[Sequence[float]]) -> List[float]:
    """Borda aggregation: per-feature mean rank across passes (lower = better)."""
    n_passes = len(rank_lists)
    n_feat = len(rank_lists[0])
    out = [0.0] * n_feat
    for r in rank_lists:
        for i in range(n_feat):
            out[i] += r[i]
    return [v / n_passes for v in out]


def median_importance(importance_lists: Sequence[Sequence[float]]) -> List[float]:
    """Per-feature median importance across passes (outlier-robust central estimate)."""
    n_feat = len(importance_lists[0])
    out = [0.0] * n_feat
    for i in range(n_feat):
        col = sorted(lst[i] for lst in importance_lists)
        m = len(col)
        out[i] = col[m // 2] if m % 2 else 0.5 * (col[m // 2 - 1] + col[m // 2])
    return out


def stability_selection_bound(pi_thr: float, q: int, n_features: int) -> float:
    """Meinshausen-Buhlmann expected-false-positives bound E[V] <= q^2 / ((2*pi-1)*F).

    ``pi_thr``: selection-frequency threshold (must be > 0.5). ``q``: average number
    of features selected per pass. Returns the upper bound on expected false positives.
    """
    if not 0.5 < pi_thr <= 1.0:
        raise ValueError("pi_thr must be in (0.5, 1.0]")
    return (q * q) / ((2.0 * pi_thr - 1.0) * n_features)
