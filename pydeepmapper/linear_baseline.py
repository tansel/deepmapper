"""Deterministic linear backbone for DeepMapper, penalised logistic regression on the FULL feature set,
with no imagification and no multi-pass accumulation.

Why this exists (the methodological point): for a linear classifier the iterate-N-accumulate machinery is
redundant. (i) A flatten+linear model is permutation-equivariant, so the feature->pixel arrangement is
irrelevant and averaging over random arrangements removes nothing. (ii) Logistic regression is convex, so
a deterministic solver (lbfgs to convergence) returns a unique optimum, every pass would be identical, so
N=1 is exact. (iii) Integrated gradients on a linear model equals coef*(x - baseline), so the attribution
ranking is exactly the (standardised) coefficients, no gradient sampling needed.

Imagification and iterate-N-accumulate are therefore for arrangement-SENSITIVE backbones (CNN/ViT); the
linear analysis needs neither. This module makes that concrete and gives an exact, reproducible baseline.
"""
from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit(X, y, feature_names: Optional[Sequence[str]] = None, C: float = 1.0, max_iter: int = 5000):
    """Fit the deterministic linear backbone. Returns (clf, scaler, ranking) where ranking is a list of
    (feature_name_or_index, importance) sorted descending. Importance = max_class |standardised coef|."""
    X = np.asarray(X, np.float32); y = np.asarray(y).astype(int)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs").fit(scaler.transform(X), y)
    coef = np.abs(clf.coef_).max(0)                       # (p,), binary or one-vs-rest max
    order = np.argsort(coef)[::-1]
    names = list(feature_names) if feature_names is not None else list(range(X.shape[1]))
    ranking = [(names[i], float(coef[i])) for i in order]
    return clf, scaler, ranking


def top_genes(ranking, k: int) -> List:
    """Top-k feature names from a `fit` ranking."""
    return [name for name, _ in ranking[:k]]
