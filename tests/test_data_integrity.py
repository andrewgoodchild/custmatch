"""Invariants of the generated datasets.

Skipped when the data has not been built (it is gitignored - see docs/reproducing.md for how to rebuild).
These guard properties the results depend on: if splits leak, every held-out number is wrong.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/scripts"))

PERSON = ROOT / "data/person_em"
CUSTOMER = ROOT / "data/customer"
person_em = pytest.mark.skipif(not (PERSON / "test.npz").exists(),
                               reason="run research/scripts/person_em_build.py first")
customer = pytest.mark.skipif(not (CUSTOMER / "source_crm.parquet").exists(),
                              reason="run research/scripts/customer_data.py first")


@person_em
def test_person_splits_are_cluster_disjoint():
    """No person may appear in two splits, or the held-out scores are contaminated."""
    splink_datasets = pytest.importorskip("splink").splink_datasets     # the research data, not custmatch
    src = splink_datasets.historical_50k.reset_index(drop=True)
    seen = {}
    for name in ("train", "valid", "test"):
        d = dict(np.load(PERSON / f"{name}.npz"))
        ids = np.unique(np.concatenate([d["left"], d["right"]]))
        clusters = set(src.loc[ids, "cluster"])
        for other, other_clusters in seen.items():
            overlap = clusters & other_clusters
            assert not overlap, f"{name} shares {len(overlap)} people with {other}"
        seen[name] = clusters


@person_em
def test_person_labels_match_true_clusters():
    splink_datasets = pytest.importorskip("splink").splink_datasets     # the research data, not custmatch
    src = splink_datasets.historical_50k.reset_index(drop=True)
    d = dict(np.load(PERSON / "test.npz"))
    expected = src.loc[d["left"], "cluster"].to_numpy() == src.loc[d["right"], "cluster"].to_numpy()
    assert np.array_equal(d["y"].astype(bool), expected)


@person_em
def test_test_split_keeps_every_candidate_pair():
    """The clustering pass needs the full candidate graph, not a sample."""
    d = dict(np.load(PERSON / "test.npz"))
    meta = dict(np.load(PERSON / "test_meta.npz", allow_pickle=True))
    assert len(d["y"]) > 100_000
    assert int(meta["n_true_pairs"]) > len(d["y"][d["y"].astype(bool)]), \
        "true pairs must exceed retrieved pairs - blocking is never perfect"


@customer
def test_customer_sources_share_people_but_not_fields():
    a = pd.read_parquet(CUSTOMER / "source_crm.parquet")
    b = pd.read_parquet(CUSTOMER / "source_billing.parquet")
    assert len(set(a.simulant_id) & set(b.simulant_id)) > 1000
    assert "date_of_birth" in a.columns and "date_of_birth" not in b.columns
    assert "ssn" in b.columns and "ssn" not in a.columns


@customer
def test_injected_junk_is_actually_shared_across_unrelated_people():
    a = pd.read_parquet(CUSTOMER / "source_crm.parquet")
    b = pd.read_parquet(CUSTOMER / "source_billing.parquet")
    df = pd.concat([a, b])
    truth = pd.read_parquet(CUSTOMER / "contact_truth.parquet")
    junk = set(truth.loc[truth.contact_class == "junk", "email"])
    assert junk, "no junk emails were injected"
    for value in junk:
        holders = df.loc[df.email == value, "simulant_id"].nunique()
        assert holders > 5, f"{value} held by only {holders} people - not a junk hub"


@customer
def test_household_contacts_are_shared_within_a_household():
    a = pd.read_parquet(CUSTOMER / "source_crm.parquet")
    hh = a[a.email != ""].groupby("email").household_id.nunique()
    shared = hh[hh == 1]
    assert len(shared) > 0, "expected some contacts confined to a single household"
