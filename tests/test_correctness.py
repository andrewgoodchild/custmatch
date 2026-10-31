"""Correctness against a reference: clustering, blocking, the match weight, Soundex.

The kind of checks Splink's own suite makes (test_cc_random_graphs, test_blocking,
test_total_comparison_count, test_cluster_at_multiple_thresholds, test_correctness_of_convergence,
test_accuracy, test_term_frequencies, test_extreme_match_weights): each result is compared with a
brute-force or hand-computed answer, on random inputs where that helps, so a quiet bug in a core
step fails here rather than costing accuracy unnoticed.
"""
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from custmatch import io
from custmatch.core import components

ROOT = Path(__file__).resolve().parents[1]


def _partition(labels):
    """A clustering as a set of frozensets of record indices, whatever its label values."""
    groups = {}
    for i, c in enumerate(labels):
        groups.setdefault(c, set()).add(i)
    return {frozenset(g) for g in groups.values()}


# ------------------------------------------------------------------ clustering
@pytest.mark.parametrize("seed", range(20))
def test_connected_components_match_networkx_on_random_graphs(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 300))
    m = int(rng.integers(0, 2 * n + 1))
    pairs = rng.integers(0, n, size=(m, 2)) if m else np.zeros((0, 2), int)
    keep = rng.random(m) < 0.6
    g = nx.Graph()
    g.add_nodes_from(range(n))                      # isolated records are customers too
    g.add_edges_from(map(tuple, pairs[keep]))
    expected = {frozenset(c) for c in nx.connected_components(g)}
    assert _partition(components(n, pairs, keep)) == expected


@pytest.mark.parametrize("seed", range(10))
def test_a_higher_threshold_only_splits_customers(seed):
    """Clusterings at increasing thresholds are nested: every customer at a stricter threshold lies
    inside one customer at a looser one - raising the threshold never joins records."""
    rng = np.random.default_rng(seed)
    n = 400
    pairs = rng.integers(0, n, size=(1200, 2))
    score = rng.random(len(pairs))
    previous = None
    for t in (0.2, 0.4, 0.6, 0.8, 0.95):
        part = _partition(components(n, pairs, score > t))
        if previous is not None:
            assert all(any(c <= p for p in previous) for c in part), f"threshold {t} joined records"
            assert len(part) >= len(previous)
        previous = part


# ------------------------------------------------------------------ blocking
@pytest.mark.parametrize("seed", range(10))
def test_block_pairs_match_brute_force(seed, monkeypatch):
    """generate()'s pairs from its block entries, made over many ranges of records (a small
    core.PAIR_CHUNK): each pair once and sorted, with each rule's pairs and new pairs."""
    import pandas as pd
    from custmatch import core
    from custmatch.blocking import _pairs_numpy
    rng = np.random.default_rng(seed)
    monkeypatch.setattr(core, "PAIR_CHUNK", int(rng.integers(1, 50)))
    m = int(rng.integers(0, 600))
    table = pd.DataFrame({"rule": rng.integers(0, 3, m), "key": rng.integers(0, 25, m),
                          "rid": rng.integers(0, 200, m)}).drop_duplicates()
    pairs, counts, new = _pairs_numpy(table, 3, 200)
    by_rule = [set() for _ in range(3)]
    for (r, _), g in table.groupby(["rule", "key"]):
        by_rule[r] |= set(combinations(sorted(g.rid), 2))
    assert [tuple(p) for p in pairs.tolist()] == sorted(set().union(*by_rule))
    assert counts.tolist() == [len(s) for s in by_rule]
    seen, expected_new = set(), []
    for s in by_rule:
        expected_new.append(len(s - seen))
        seen |= s
    assert new.tolist() == expected_new


@pytest.mark.parametrize("seed", range(5))
def test_link_bridges_match_networkx(seed):
    """links.csv's is_bridge: the vectorised search finds networkx's bridges, on graphs with
    trees, cycles, bridges between cycles and isolated records."""
    from custmatch.explain import _bridges
    rng = np.random.default_rng(seed)
    for _ in range(60):
        n = int(rng.integers(2, 40))
        e = np.unique(np.sort(rng.integers(0, n, (int(rng.integers(1, 3 * n)), 2)), axis=1), axis=0)
        e = e[e[:, 0] != e[:, 1]]
        g = nx.Graph()
        g.add_edges_from(map(tuple, e.tolist()))
        bridges = {frozenset(x) for x in nx.bridges(g)}
        assert _bridges(n, e).tolist() == [frozenset(x) in bridges for x in map(tuple, e.tolist())]


def test_components_by_threshold_match_components():
    from custmatch.core import components_by_threshold, threshold_bands
    rng = np.random.default_rng(0)
    for n in (2, 50, 3000):
        pairs = np.sort(rng.integers(0, n, (3 * n, 2)), axis=1).astype(np.int32)
        score = rng.random(len(pairs)).round(2)          # ties at the thresholds themselves
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        got = components_by_threshold(n, pairs, score, grid)
        banded = components_by_threshold(n, pairs, None, grid, band=threshold_bands(score, grid))
        for t in grid:
            assert np.array_equal(got[t], components(n, pairs, score > t)), (n, t)
            assert np.array_equal(banded[t], got[t]), (n, t)


def test_fixed_rule_blocking_proposes_exactly_the_pairs_sharing_a_key():
    """With no block over max_block, the candidates are exactly the pairs sharing some rule's key,
    and each rule's pair counts are right (Splink's test_total_comparison_count)."""
    from custmatch.blocking import Keys, fixed_rules, generate
    from custmatch.core import profile_contacts
    df, cfg = io.load(ROOT / "examples/demo/config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    keys = Keys(df, prof)
    rules = fixed_rules(cfg, df)
    pairs, report = generate(df, keys, rules, max_block=10 ** 6)
    expected, first_rule = set(), {}
    for n, r in enumerate(rules):
        k = keys.rule(r)
        by = pd.Series(np.arange(len(df)))[k >= 0].groupby(k[k >= 0])
        rule_pairs = {p for g in by.groups.values() if len(g) > 1 for p in combinations(sorted(g), 2)}
        assert report["rules"][n]["pairs"] == len(rule_pairs), r
        for p in rule_pairs:
            first_rule.setdefault(p, n)
        expected |= rule_pairs
    assert set(map(tuple, pairs.tolist())) == expected
    for n, entry in enumerate(report["rules"]):
        assert entry["new_pairs"] == sum(1 for v in first_rule.values() if v == n)


def test_default_blocking_proposes_every_pair_sharing_a_personal_email():
    """core.block: two records sharing an email that is neither junk nor in an oversized block are
    always candidates; records sharing nothing are never."""
    from custmatch.core import block, profile_contacts
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({c: [""] * n for c in io.CANONICAL})
    df["source"] = "s"
    df["record_id"] = [str(i) for i in range(n)]
    df["first_name"] = [f"f{i}" for i in range(n)]          # no two records share a name
    df["last_name"] = [f"l{i}" for i in range(n)]
    df["email"] = [f"user{int(e)}@mail.com" for e in rng.integers(0, 120, n)]
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    pairs, _ = block(df, prof)
    got = set(map(tuple, pairs.tolist()))
    em = df.email.to_numpy()
    junk = {v for v, (cls, _) in prof["email"].items() if cls == "junk"}   # many records, nothing else shared
    expected = {(a, b) for a, b in combinations(range(n), 2) if em[a] == em[b] and em[a] not in junk}
    assert junk and got == expected



def test_hashed_blocking_keys_find_the_same_candidates_as_block():
    """The real-time matcher's index (core.blocking_hashes) and the batch blocker (core.block) use
    one key definition: grouping records by key hash, under the same block-size rule, gives
    exactly block's candidate pairs - with extra and similarity (LSH) keys switched on."""
    from custmatch import schema
    from custmatch.core import block, blocking_hashes, profile_contacts
    df, cfg = io.load(ROOT / "examples/demo/config.json")
    extra = schema.extra_keys(cfg | {"extra_blocking": [["postcode"], ["last_name:soundex", "date_of_birth"]]})
    lsh = {"fields": ["first_name", "last_name"], "bands": 8, "rows": 2}
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    for max_block in (60, 8):
        pairs, _ = block(df, prof, extra_keys=extra, max_block=max_block, lsh=lsh)
        h, r = blocking_hashes(df, prof, extra, lsh)
        groups = pd.Series(r).groupby(h).agg(lambda x: sorted(set(x)))
        expected = {p for g in groups if 2 <= len(g) <= max_block for p in combinations(g, 2)}
        assert set(map(tuple, pairs.tolist())) == expected


def test_blocking_report_by_key_adds_up_to_the_candidate_pairs():
    """block(by_key=...) (Splink's cumulative comparisons chart): each key's new pairs add up to
    exactly the candidate pairs, and asking for the report changes nothing."""
    from custmatch import schema
    from custmatch.core import block, profile_contacts
    df, cfg = io.load(ROOT / "examples/demo/config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    rows = []
    with_report, _ = block(df, prof, extra_keys=schema.extra_keys(cfg), by_key=rows)
    without, _ = block(df, prof, extra_keys=schema.extra_keys(cfg))
    assert np.array_equal(with_report, without)
    assert sum(r["new"] for r in rows) == rows[-1]["cumulative"] == len(without)
    assert all(r["new"] <= r["pairs"] for r in rows)

# ------------------------------------------------------------------ the match weight
def _records(**cols):
    n = len(next(iter(cols.values())))
    df = pd.DataFrame({c: [""] * n for c in io.CANONICAL})
    df["source"], df["record_id"] = "s", [str(i) for i in range(n)]
    for c, v in cols.items():
        df[c] = v
    return df


def test_match_weight_equals_the_hand_computed_fellegi_sunter_weight():
    """log2(m/u) per field, the agreeing value's own frequency in place of u (term frequency),
    nothing for a missing value - checked against the formula on a tiny file."""
    from custmatch.matchweight import Scorer
    ssn = ["111", "111", "111", "222", "222", "333", ""]
    df = _records(ssn=ssn)
    m, u = [0.1, 0.9], [0.95, 0.05]
    model = {"fields": [{"column": "ssn", "kind": "exact", "tf": True, "m": m, "u": u}], "lambda": 0.1}
    sc = Scorer(df, {"email": {}, "phone": {}}, model)
    pairs = np.array([[0, 1], [3, 4], [0, 3], [0, 6]])
    w = sc.field_weights(pairs)[:, 0]
    filled = 6                                          # records with an ssn
    assert np.isclose(w[0], np.log2(0.9 / (3 / filled)))       # agree on a common value
    assert np.isclose(w[1], np.log2(0.9 / (2 / filled)))       # agree on a rarer one: more weight
    assert np.isclose(w[2], np.log2(0.1 / 0.95))               # disagree
    assert w[3] == 0.0                                          # one side missing
    assert w[1] > w[0] > 0 > w[2]


def test_match_weight_em_recovers_known_probabilities():
    """Simulated pairs with known m, u and match share: EM, fitted without labels, gets them back
    (Splink's test_correctness_of_convergence)."""
    from custmatch.matchweight import fit
    rng = np.random.default_rng(1)
    n_pairs, share, m_true = 4000, 0.3, {"f1": 0.9, "f2": 0.8, "f3": 0.85}
    cols = {f: [] for f in m_true}
    for k in range(n_pairs):
        match = k < share * n_pairs
        for f, m in m_true.items():
            a = f"{f}_{rng.integers(10 ** 6)}"
            b = a if (match and rng.random() < m) else f"{f}_{rng.integers(10 ** 6)}"
            cols[f] += [a, b]
    df = pd.DataFrame({c: [""] * (2 * n_pairs) for c in io.CANONICAL})
    df["source"], df["record_id"] = "s", [str(i) for i in range(2 * n_pairs)]
    for f, v in cols.items():
        df[f] = v
    cfg = {"extra_fields": [{"name": f, "compare": "exact"} for f in m_true]}
    pairs = np.arange(2 * n_pairs).reshape(-1, 2)
    model = fit(df, {"email": {}, "phone": {}}, pairs, cfg, seed=0)
    assert abs(model["lambda"] - share) < 0.03, model["lambda"]
    for spec in model["fields"]:
        assert abs(spec["m"][1] - m_true[spec["column"]]) < 0.04, (spec["column"], spec["m"])
        assert spec["u"][1] < 0.01                     # random records almost never agree


def test_match_weights_stay_finite_at_the_extremes():
    """A field every record shares, and one no two records share: weights stay finite
    (Splink's test_extreme_match_weights)."""
    from custmatch.matchweight import weights
    n = 400
    df = _records(ssn=["same"] * n, first_name=[f"name{i}" for i in range(n)])
    pairs = np.array([[i, i + 1] for i in range(0, n - 1, 2)])
    (w,), sc = weights(df, {"email": {}, "phone": {}}, [pairs], {}, return_scorer=True)
    assert np.isfinite(w).all() and np.isfinite(sc.field_weights(pairs)).all()
    for spec in sc.model["fields"]:
        assert all(0 < x < 1 for x in spec["m"] + spec["u"]), spec


# ------------------------------------------------------------------ Soundex
@pytest.mark.parametrize("name,code", [
    ("Robert", "R163"), ("Rupert", "R163"), ("Rubin", "R150"), ("Ashcraft", "A261"), ("Ashcroft", "A261"),
    ("Tymczak", "T522"), ("Pfister", "P236"), ("Honeyman", "H555"), ("Lee", "L000"), ("Gutierrez", "G362"),
    ("Jackson", "J250"), ("Washington", "W252"), ("Smith", "S530"), ("Smyth", "S530"), ("O'Brien", "O165"),
    ("van Minh", "V555"), ("", ""), ("123", "")])
def test_soundex_matches_the_standard(name, code):
    """American Soundex reference codes (the US National Archives' rules: h and w do not separate
    equal codes, so Ashcraft is A261), and what custmatch does at the edges."""
    from custmatch.blocking import soundex
    assert soundex(name) == code
