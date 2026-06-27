"""Property tests for pydeepmapper.transform. Stdlib + pytest + hypothesis only (no numpy/torch)."""
from __future__ import annotations

import os
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydeepmapper import transform as T  # noqa: E402


# --- square_dim: result*result >= n+buffer AND (result-1)^2 < n+buffer -------
@given(n=st.integers(min_value=1, max_value=200000),
       buf=st.integers(min_value=0, max_value=5000))
def test_square_dim_is_minimal_square(n, buf):
    d = T.square_dim(n, buf)
    assert isinstance(d, int) and d >= 1
    assert d * d >= n + buf                      # holds every feature (+buffer)
    assert (d - 1) * (d - 1) < n + buf           # and is the SMALLEST such square


def test_square_dim_rejects_nonpositive():
    for bad in (0, -1):
        try:
            T.square_dim(bad)
            assert False, "expected ValueError"
        except ValueError:
            pass


# --- permutation: bijection on range(n), deterministic in the seed -----------
@given(seed=st.integers(), n=st.integers(min_value=0, max_value=500))
def test_permutation_is_a_bijection(seed, n):
    p = T.permutation(seed, n)
    assert isinstance(p, list) and len(p) == n
    assert sorted(p) == list(range(n))           # every index exactly once


@given(seed=st.integers(), n=st.integers(min_value=0, max_value=500))
def test_permutation_is_deterministic(seed, n):
    assert T.permutation(seed, n) == T.permutation(seed, n)


@given(n=st.integers(min_value=2, max_value=500))
def test_distinct_seeds_usually_differ(n):
    # not a hard guarantee, but for n>=2 two fixed different seeds should differ
    assert T.permutation(1, n) != T.permutation(999983, n)


# --- inverse_permutation: inv[perm[i]] == i and perm[inv[i]] == i ------------
@given(seed=st.integers(), n=st.integers(min_value=0, max_value=500))
@settings(max_examples=200)
def test_inverse_round_trips(seed, n):
    p = T.permutation(seed, n)
    inv = T.inverse_permutation(p)
    assert len(inv) == len(p)
    for i in range(n):
        assert inv[p[i]] == i
        assert p[inv[i]] == i


# --- image_side mirrors square_dim with buffer 0 ----------------------------
@given(n=st.integers(min_value=1, max_value=200000))
def test_image_side_matches_square_dim(n):
    assert T.image_side(n) == T.square_dim(n, 0)
