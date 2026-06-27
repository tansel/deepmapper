"""Property tests for pydeepmapper.accumulate. Stdlib + pytest + hypothesis only."""
from __future__ import annotations

import os
import sys

from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydeepmapper import accumulate as A  # noqa: E402


# --- accept_pass: result == (accuracy >= min_accuracy) -----------------------
@given(a=st.floats(min_value=0, max_value=1),
       m=st.floats(min_value=0, max_value=1))
def test_accept_pass_is_threshold(a, m):
    assert A.accept_pass(a, m) == (a >= m)


# --- selection_frequency: length n_features, every value in [0,1] ------------
@given(n=st.integers(min_value=1, max_value=50),
       data=st.data())
def test_selection_frequency_range_and_length(n, data):
    n_passes = data.draw(st.integers(min_value=0, max_value=20))
    top_sets = [
        data.draw(st.lists(st.integers(min_value=0, max_value=n - 1),
                           min_size=0, max_size=n, unique=True))
        for _ in range(n_passes)
    ]
    freq = A.selection_frequency(top_sets, n)
    assert len(freq) == n
    assert all(0.0 <= p <= 1.0 for p in freq)


def test_selection_frequency_counts_correctly():
    # feature 1 in all 3 passes -> 1.0; feature 2 in 2/3; feature 3 in 1/3; feature 0 never
    assert A.selection_frequency([[1, 2], [1, 3], [1, 2]], 4) == [0.0, 1.0, 2 / 3, 1 / 3]


def test_selection_frequency_empty_is_zero():
    assert A.selection_frequency([], 5) == [0.0] * 5


# --- ranks_from_importances: a permutation of 1..n, most-important == rank 1 --
@given(imps=st.lists(st.floats(min_value=-1e6, max_value=1e6,
                               allow_nan=False, allow_infinity=False),
                     min_size=1, max_size=60))
def test_ranks_are_a_permutation(imps):
    r = A.ranks_from_importances(imps)
    assert sorted(r) == list(range(1, len(imps) + 1))
    # the argmax feature gets rank 1
    best = max(range(len(imps)), key=lambda i: (imps[i], -i))
    assert r[best] == 1


# --- mean_rank: length preserved, every mean within [1, n_passes-range] ------
@given(n_feat=st.integers(min_value=1, max_value=30),
       n_pass=st.integers(min_value=1, max_value=10))
def test_mean_rank_length_and_floor(n_feat, n_pass):
    rank_lists = [list(range(1, n_feat + 1)) for _ in range(n_pass)]
    mr = A.mean_rank(rank_lists)
    assert len(mr) == n_feat
    assert all(v >= 1 for v in mr)


# --- stability_selection_bound: positive, shrinks as F grows -----------------
def test_stability_bound_monotone_in_features():
    b_small = A.stability_selection_bound(0.8, q=10, n_features=100)
    b_big = A.stability_selection_bound(0.8, q=10, n_features=10000)
    assert b_small > b_big > 0
