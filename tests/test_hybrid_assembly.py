"""Property tests for pydeepmapper.hybrid_assembly: the semantic invariants (idempotent tagging,
namespace separation, sc-table assembly). Stdlib + pytest + hypothesis only (no torch,
no external bio tools, only the PURE core is exercised here)."""
from __future__ import annotations

import os
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydeepmapper import hybrid_assembly as H  # noqa: E402

# transcript-id-like tokens (some may already carry the DENOVO_ prefix)
_ids = st.lists(st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789", min_size=1,
                        max_size=12), min_size=0, max_size=30)


# --- is_denovo: bool, and exactly the DENOVO_ prefix test --------------------
@given(t=st.text(min_size=0, max_size=20))
def test_is_denovo_is_bool_and_prefix(t):
    r = H.is_denovo(t)
    assert isinstance(r, bool)
    assert r == t.startswith(H.DENOVO_PREFIX)


# --- tag_denovo: length preserved, every output tagged, idempotent ----------
@given(ids=_ids)
def test_tag_denovo_invariants(ids):
    tagged = H.tag_denovo(ids)
    assert len(tagged) == len(ids)                       # contract: length preserved
    assert all(isinstance(t, str) for t in tagged)       # contract: forall t is str
    assert all(H.is_denovo(t) for t in tagged)           # every output is de-novo
    assert H.tag_denovo(tagged) == tagged                # idempotent (no double-prefix)
    # an already-tagged input is left untouched
    for orig, out in zip(ids, tagged):
        assert out == (orig if H.is_denovo(orig) else H.DENOVO_PREFIX + orig)


# --- merge_reference_and_denovo: length = sum, reference kept, no collision --
@given(ref=_ids, dn=_ids)
def test_merge_lengths_and_namespace(ref, dn):
    merged = H.merge_reference_and_denovo(ref, dn)
    assert len(merged) == len(ref) + len(dn)             # contract
    assert merged[:len(ref)] == list(ref)                # reference verbatim, first
    # de-novo half is fully namespaced
    assert all(H.is_denovo(t) for t in merged[len(ref):])
    # a non-tagged reference id can never be mistaken for a de-novo one
    for r in ref:
        if not H.is_denovo(r):
            assert not H.is_denovo(r)


# --- denovo_fraction: in [0,1] and equals the true share --------------------
@given(ids=_ids)
def test_denovo_fraction_range(ids):
    f = H.denovo_fraction(ids)
    assert isinstance(f, float)
    assert 0.0 <= f <= 1.0
    if ids:
        assert f == sum(H.is_denovo(t) for t in ids) / len(ids)
    else:
        assert f == 0.0


# --- build_sc_table: shape, column order, no NaN, values preserved ----------
@given(data=st.data())
def test_build_sc_table_assembly(data):
    pd = pytest.importorskip("pandas")
    n_samples = data.draw(st.integers(min_value=1, max_value=5))
    samples = [f"S{i}" for i in range(n_samples)]
    all_tx = data.draw(st.lists(st.text(alphabet="abcdEFGH_0123456789", min_size=1, max_size=8),
                                min_size=1, max_size=12, unique=True))
    per_sample = {}
    for s in samples:
        chosen = data.draw(st.lists(st.sampled_from(all_tx), min_size=0, max_size=len(all_tx),
                                    unique=True))
        per_sample[s] = {t: float(data.draw(st.integers(0, 1000))) for t in chosen}

    mat = H.build_sc_table(per_sample)
    assert list(mat.columns) == samples                  # column order preserved
    assert not mat.isna().any().any()                    # missing filled 0, no NaN
    union = set().union(*[set(d) for d in per_sample.values()])
    assert set(mat.index) == union                       # every transcript present
    for s in samples:                                    # values preserved, gaps = 0
        for t in mat.index:
            assert mat.loc[t, s] == per_sample[s].get(t, 0.0)
