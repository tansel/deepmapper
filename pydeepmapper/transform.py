"""DeepMapper feature->pixel transform, the PURE, contracted core.

A feature vector is laid out into a small ``dim x dim`` pseudo-image by padding to
a perfect square and reshaping. Different *arrangements* come from a seeded
permutation of the feature order; per-pixel attributions are projected back to
feature space through the inverse permutation (a clean 1-pixel-per-feature gather, DeepMapper's differentiating property).

The scalar/list functions here are
stdlib-only and deterministic, so they are testable without numpy or torch. Only
the array builders (``to_images`` / ``pixels_to_features``) need numpy, imported
lazily so importing this module never requires it.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence


def square_dim(n_features: int, buffer: int = 0) -> int:
    """Smallest square side ``dim`` with ``dim*dim >= n_features + buffer``.

    Mirrors the canonical ``DeepMapper.map`` sizing. Minimal by construction:
    ``(dim-1)**2 < n_features + buffer <= dim**2``.
    """
    if not isinstance(n_features, int) or n_features < 1:
        raise ValueError("n_features must be a positive int")
    if not isinstance(buffer, int) or buffer < 0:
        raise ValueError("buffer must be a non-negative int")
    minsize = n_features + buffer
    dim = math.isqrt(minsize)
    if dim * dim < minsize:
        dim += 1
    return dim


def image_side(n_features: int) -> int:
    """Square side for ``n_features`` with no buffer (batch-facing alias)."""
    return square_dim(n_features, 0)


def permutation(seed: int, n: int) -> List[int]:
    """A deterministic, seeded permutation of ``range(n)`` (one *arrangement*).

    Same ``seed`` always yields the same permutation (reproducible runs). Uses the
    same ``random.Random(seed).shuffle`` mechanism as the canonical
    ``DeepMapper.shuffle_to_seed``.
    """
    lst = list(range(n))
    random.Random(seed).shuffle(lst)
    return lst


def inverse_permutation(perm: Sequence[int]) -> List[int]:
    """Inverse of a permutation: ``inv[perm[i]] == i`` and ``perm[inv[i]] == i``."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


# --- array builders (numpy, imported lazily) --------------------------------
def to_images(X, buffer: int = 0, perm: Optional[Sequence[int]] = None):
    """Lay a ``(n_samples, n_features)`` matrix out as ``(n_samples, dim, dim, 1)``.

    If ``perm`` is given, features are reordered by it first (the arrangement);
    feature ``perm[j]`` lands at flattened pixel ``j``. Padding to ``dim*dim`` is
    zeros. (Deviation from the legacy code: we pad to exactly ``dim*dim`` rather
    than using ``np.resize``, which silently *repeats* data when a buffer is set.)
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_samples, n_features)")
    n, f = X.shape
    if perm is not None:
        if len(perm) != f:
            raise ValueError("perm length must equal n_features")
        X = X[:, list(perm)]
    dim = square_dim(f, buffer)
    pad = dim * dim - f
    Xp = np.pad(X, ((0, 0), (0, pad))) if pad else X
    return Xp.reshape(n, dim, dim, 1)


def pixels_to_features(attr_image, perm: Optional[Sequence[int]], n_features: int):
    """Project a per-pixel attribution image back to per-feature importances.

    Inverse of :func:`to_images`' layout: flattened pixel ``j`` (for ``j <
    n_features``) carries feature ``perm[j]``, so ``out[perm[j]] = a[j]``. Padding
    pixels are dropped. With ``perm=None`` the identity arrangement is assumed.
    """
    import numpy as np

    a = np.asarray(attr_image, dtype=np.float32).reshape(-1)
    if perm is None:
        return a[:n_features]
    out = np.zeros(n_features, dtype=np.float32)
    for j in range(n_features):
        out[perm[j]] = a[j]
    return out
