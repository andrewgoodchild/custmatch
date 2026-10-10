"""Metric correctness. Every number in docs/report/ comes out of these functions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from custmatch.core import pair_scores, partition_scores  # noqa: E402


def test_pair_scores_hand_worked():
    #      true:  T  T  F  F  T
    #      pred:  T  F  T  F  T   -> tp=2, fp=1, fn(among these)=1
    y = np.array([True, True, False, False, True])
    pred = np.array([True, False, True, False, True])
    # 4 true pairs exist in total, one of which blocking never proposed
    r = pair_scores(y, pred, n_true_all=4)
    assert r["precision"] == pytest.approx(200 / 3, abs=0.01)   # 2/3
    assert r["recall"] == 50.0                                   # 2/4
    assert r["false_merges"] == 1


def test_recall_counts_pairs_blocking_never_proposed():
    """Recall must be against ALL true pairs, or blocking losses vanish from the report."""
    y = np.array([True])
    pred = np.array([True])
    assert pair_scores(y, pred, n_true_all=1)["recall"] == 100.0
    assert pair_scores(y, pred, n_true_all=10)["recall"] == 10.0


def test_partition_scores_perfect_and_singleton():
    true = np.array([0, 0, 1, 1])
    assert partition_scores(true.copy(), true, n_true_all=2)["f1"] == 100.0
    # everything split apart: no predicted pairs at all
    singles = np.arange(4)
    r = partition_scores(singles, true, n_true_all=2)
    assert r["precision"] == 0.0 and r["recall"] == 0.0 and r["clusters"] == 4


def test_partition_scores_one_giant_cluster():
    """The failure mode from section 7: everything merged into one blob."""
    true = np.array([0, 0, 1, 1, 2, 2])
    blob = np.zeros(6, dtype=int)
    r = partition_scores(blob, true, n_true_all=3)
    assert r["recall"] == 100.0           # every true pair is inside the blob
    assert r["precision"] == pytest.approx(20.0)   # 3 of 15 implied pairs are right
    assert r["largest_cluster"] == 6
    assert r["adjusted_rand"] == 0.0


def test_all_positive_f1_equals_collapse_signature():
    """A collapsed run scores exactly 2p/(1+p). This is the detector in the report, so it
    has to be arithmetically true, not folklore."""
    for positive_rate in (0.094, 0.102, 0.15, 0.42):
        n = 100_000
        n_pos = int(n * positive_rate)
        y = np.zeros(n, dtype=bool)
        y[:n_pos] = True
        pred = np.ones(n, dtype=bool)          # predict "match" for everything
        f1 = pair_scores(y, pred, n_true_all=float(n_pos))["f1"]
        expected = 100 * 2 * positive_rate / (1 + positive_rate)
        assert f1 == pytest.approx(expected, abs=0.05)
