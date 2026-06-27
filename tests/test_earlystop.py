"""Tests for pydeepmapper.earlystop.EarlyStopper (plateau early-stopping).
Reproduces Keras EarlyStopping(monitor, patience, min_delta, restore_best).
Stdlib + pytest + hypothesis only, no torch."""
from __future__ import annotations

import os
import sys

from hypothesis import given
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydeepmapper.earlystop import EarlyStopper  # noqa: E402


def run(values, patience=3, min_delta=0.0, min_epochs=0, mode="min"):
    """Feed a sequence; return (stop_epoch or None, best_epoch)."""
    es = EarlyStopper(patience, min_delta, min_epochs, mode)
    for i, v in enumerate(values):
        if es.step(i, v):
            return i, es.best_epoch
    return None, es.best_epoch


# --- monotonic improvement never stops ---------------------------------------
def test_monotonic_improvement_never_stops():
    stop, best = run([1.0 - 0.01 * i for i in range(50)], patience=5)
    assert stop is None
    assert best == 49                       # last (lowest) epoch is best


# --- plateau stops exactly patience epochs after the best --------------------
def test_stops_after_patience():
    # best at epoch 2 (0.5), then flat; patience 3 -> stop at epoch 2+3 = 5
    stop, best = run([0.9, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5], patience=3)
    assert best == 2
    assert stop == 5


# --- min_epochs floor blocks an early stop -----------------------------------
def test_min_epochs_floor():
    vals = [0.5] + [0.6] * 30               # best at 0, then all worse
    stop_no_floor, _ = run(vals, patience=3, min_epochs=0)
    stop_floor, best_floor = run(vals, patience=3, min_epochs=20)
    assert stop_no_floor == 3               # 0 best, 3 bad -> stop at 3
    # warmup ignores epochs 0..18; best taken at 19, then 3 bad -> stop at 22
    assert best_floor == 19
    assert stop_floor == 22


# --- best-epoch respects the floor (the refinement) --------------------------
def test_best_epoch_respects_floor():
    vals = [0.5] * 20
    vals[2] = 0.1                           # true minimum sits inside the warmup
    _, best = run(vals, patience=3, min_epochs=10)
    assert best >= 9                        # NOT epoch 2; restored model trained >= min_epochs


# --- min_delta: sub-threshold gains count as no improvement ------------------
def test_min_delta_counts_tiny_gains_as_bad():
    # drops of 0.0005 < min_delta 0.01 -> not improvements
    stop, best = run([1.0, 0.9995, 0.9990, 0.9985], patience=2, min_delta=0.01)
    assert best == 0
    assert stop == 2


# --- max mode (accuracy: higher is better) -----------------------------------
def test_max_mode_tracks_increasing_metric():
    stop, best = run([0.6, 0.7, 0.8, 0.8, 0.8], patience=2, mode="max")
    assert best == 2
    assert stop == 4


# --- best is never worse than any seen value (property) ----------------------
@given(vals=st.lists(st.floats(min_value=0, max_value=10, allow_nan=False),
                     min_size=1, max_size=40))
def test_best_is_the_minimum_seen(vals):
    es = EarlyStopper(patience=10**9, min_delta=0.0, min_epochs=0, mode="min")
    for i, v in enumerate(vals):
        es.step(i, v)
    assert es.best <= min(vals) + 1e-9
