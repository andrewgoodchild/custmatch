"""Clustering behaviour, including regression tests for two bugs this project actually hit:

  * connected components chaining separate households into one 2,252-record cluster
  * NetworkX silently dropping isolated records, which undercounts singleton customers
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from custmatch.core import components, household_partition  # noqa: E402


def _kuzu_graph():
    """The research graph code (scripts/kuzu_graph.py) needs kuzu, which custmatch does not:
    `pip install kuzu` to run these two tests."""
    pytest.importorskip("kuzu")
    import kuzu_graph
    return kuzu_graph


def test_components_keeps_isolated_records_as_singletons():
    """Records with no accepted edge are singleton customers, not missing rows."""
    pairs = np.array([[0, 1]])
    labels = components(5, pairs, np.array([True]))
    assert len(labels) == 5
    assert labels[0] == labels[1]
    assert len({labels[2], labels[3], labels[4], labels[0]}) == 4


def test_networkx_from_edges_alone_would_lose_singletons():
    """Documents the trap: building a graph from an edge list drops isolated nodes."""
    g = nx.Graph()
    g.add_edges_from([(0, 1)])
    assert nx.number_connected_components(g) == 1        # wrong for a 5-record file
    g.add_nodes_from(range(5))
    assert nx.number_connected_components(g) == 4        # right


def test_chain_of_edges_merges_everything():
    """The percolation failure: a path graph collapses into one cluster."""
    pairs = np.array([[i, i + 1] for i in range(9)])
    labels = components(10, pairs, np.ones(len(pairs), dtype=bool))
    assert len(set(labels)) == 1


def test_bridge_aware_cuts_the_weak_link_in_a_chain():
    edges = pd.DataFrame({
        "src": [0, 1, 2, 3, 4, 5, 6],
        "dst": [1, 2, 0, 4, 5, 3, 3],      # two triangles joined by edge 6-3
        "prob": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.40],
    })
    edges.loc[6, ["src", "dst"]] = [2, 3]   # the weak bridge between the triangles
    kg = _kuzu_graph()
    keep, removed = kg.bridge_aware(edges, "prob", low=0.30, bridge_min=0.90)
    assert removed == 1
    labels = kg.labels_from_keep(edges, keep, np.arange(6))
    assert len(set(labels)) == 2            # two triangles, not one blob


def test_bridge_aware_leaves_strong_bridges_alone():
    edges = pd.DataFrame({
        "src": [0, 1, 2, 3, 4, 2],
        "dst": [1, 2, 0, 4, 5, 3],
        "prob": [0.99, 0.99, 0.99, 0.99, 0.99, 0.99],   # the joining edge is confident
    })
    kg = _kuzu_graph()
    keep, removed = kg.bridge_aware(edges, "prob", low=0.30, bridge_min=0.90)
    assert removed == 0
    assert len(set(kg.labels_from_keep(edges, keep, np.arange(6)))) == 1


def test_household_partition_does_not_chain_addresses():
    """Regression: closure over address edges merged separate homes through a person who
    appears at two of them, producing one 2,252-record 'household'."""
    df = pd.DataFrame({
        "addr_number": ["10", "10", "20", "20"],
        "addr_street": ["elm st", "elm st", "oak st", "oak st"],
        "addr_unit": ["", "", "", ""],
        "email": ["", "", "", ""],
        "phone": ["", "", "", ""],
    })
    # records 1 and 2 are the SAME customer, living at one address and billed at the other
    identity = np.array([0, 1, 1, 2])
    labels, n_group = household_partition(df, {"email": {}, "phone": {}}, identity)
    assert len(set(labels)) == 2, "two addresses must remain two households"
    assert n_group == 0


def test_household_partition_places_a_customer_at_its_modal_address():
    df = pd.DataFrame({
        "addr_number": ["10", "10", "99"],
        "addr_street": ["elm st", "elm st", "far rd"],
        "addr_unit": ["", "", ""],
        "email": ["", "", ""], "phone": ["", "", ""],
    })
    identity = np.array([0, 0, 0])          # one customer, three records, one odd address
    labels, _ = household_partition(df, {"email": {}, "phone": {}}, identity)
    assert len(set(labels)) == 1


def test_household_falls_back_to_shared_contact_when_address_missing():
    df = pd.DataFrame({
        "addr_number": ["", ""], "addr_street": ["", ""], "addr_unit": ["", ""],
        "email": ["home@mail.com", "home@mail.com"], "phone": ["", ""],
    })
    profiles = {"email": {"home@mail.com": ("household", 0.35)}, "phone": {}}
    labels, _ = household_partition(df, profiles, np.array([0, 1]))
    assert len(set(labels)) == 1
