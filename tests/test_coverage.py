"""Unit tests for branches the end-to-end tests in test_custmatch.py do not reach: steward
decisions, monitoring alerts, survivorship rules, the model helpers, LLM labelling (with the
HTTP API faked) and the CLI's error paths."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from custmatch import golden, io, models, monitor, stewardship


# ------------------------------------------------------------------ steward decisions
def test_overrides_reject_a_bad_decision_and_skip_unknown_records(tmp_path):
    keys = np.array(["s:1", "s:2", "s:3"])
    pd.DataFrame([{"a": "s:1", "b": "s:2", "decision": "maybe"}]).to_csv(tmp_path / "o.csv", index=False)
    with pytest.raises(io.ConfigError, match="same or different"):
        stewardship.load_overrides({"overrides": "o.csv"}, tmp_path, keys)
    pd.DataFrame([{"a": "s:2", "b": "s:1", "decision": " Same "},
                  {"a": "s:1", "b": "gone:9", "decision": "different"},
                  {"a": "s:3", "b": "s:1", "decision": "different"}]).to_csv(tmp_path / "o.csv", index=False)
    must, cannot = stewardship.load_overrides({"overrides": "o.csv"}, tmp_path, keys)
    assert must.tolist() == [[0, 1]] and cannot.tolist() == [[0, 2]]           # sorted pairs; unknown waits
    assert [len(x) for x in stewardship.load_overrides({}, tmp_path, keys)] == [0, 0]


def test_trusted_ids_ignore_placeholders_and_missing_columns():
    df = pd.DataFrame({"loyalty": ["L1", "L1", "L1", "", "X", "X", "X", "X", "X", "X"]})
    pairs = stewardship.trusted_pairs(df, {"trusted_ids": ["loyalty", "absent"], "trusted_max": 5})
    assert pairs.tolist() == [[0, 1], [0, 2]]                                # X on 6 records: a placeholder


def test_apply_decisions_adds_missing_must_links_and_zeroes_cannot_links():
    pairs = np.array([[0, 1], [1, 2]])
    p, s = stewardship.apply_decisions(pairs, np.array([0.2, 0.9]), np.array([[0, 1], [0, 3]]), np.array([[1, 2]]))
    assert p.tolist() == [[0, 1], [1, 2], [0, 3]] and s.tolist() == [1.0, 0.0, 1.0]
    same = stewardship.apply_decisions(pairs, np.array([0.2, 0.9]), np.zeros((0, 2)), np.zeros((0, 2)))
    assert same[0] is pairs


def test_enforce_cannot_cuts_the_weakest_link_on_the_path():
    # a chain 0-1-2-3; 0 and 3 must be apart; the weakest link (1-2, 0.6) is the one cut
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    score = np.array([0.95, 0.6, 0.9])
    labels_, cut = stewardship.enforce_cannot(4, pairs, score, 0.5, np.zeros(4, int), np.array([[0, 3]]))
    assert cut == 1 and labels_[0] == labels_[1] and labels_[2] == labels_[3] and labels_[0] != labels_[3]
    # already apart: nothing to do
    apart = np.array([0, 0, 1, 1])
    assert stewardship.enforce_cannot(4, pairs, score, 0.5, apart, np.array([[0, 3]]))[1] == 0
    assert stewardship.enforce_cannot(4, pairs, score, 0.5, apart, np.zeros((0, 2)))[1] == 0


def test_join_must_merges_customers():
    assert stewardship.join_must(np.array([0, 1, 2]), np.array([[0, 2]])).tolist() == [0, 1, 0]


def test_highest_issued_reads_clusters_and_history(tmp_path):
    assert stewardship.highest_issued(tmp_path) == -1
    pd.DataFrame({"customer_id": [3, 7]}).to_csv(tmp_path / "clusters.csv", index=False)
    pd.DataFrame({"customer_id": ["12", ""], "other_customer_ids": ["9;15", ""]}).to_csv(tmp_path / "history.csv", index=False)
    assert stewardship.highest_issued(tmp_path) == 15


# ------------------------------------------------------------------ monitoring
def _snap(d, **over):
    base = {"records": 100, "customers": 60, "match_rate": 50.0, "match_rate_by_source": {"a": 50.0, "b": 50.0},
            "mean_customer_size": 1.6, "largest_customer": 4, "threshold": 0.5, "pairs_scored": 100,
            "pairs_near_threshold": 5, "score_histogram": [50, 5, 5, 5, 5, 5, 5, 5, 5, 10],
            "fill_rate": {"email": 80.0}}
    d.mkdir()
    monitor.write(d, base | over)
    return d


def test_monitor_flags_each_kind_of_drift(tmp_path):
    a = _snap(tmp_path / "a")
    b = _snap(tmp_path / "b", match_rate_by_source={"a": 50.0, "b": 30.0}, largest_customer=40, threshold=0.6,
              score_histogram=[5, 5, 5, 5, 5, 5, 5, 5, 5, 55])
    table, alerts = monitor.compare([a, b])
    text = " | ".join(alerts)
    for expected in ("source b", "largest customer 4 -> 40", "threshold 0.500 -> 0.600", "score distribution"):
        assert expected in text, (expected, alerts)
    assert table.score_psi.iloc[0] is None or np.isnan(table.score_psi.iloc[0])
    with pytest.raises(FileNotFoundError, match="monitor.json"):
        monitor.compare([a, tmp_path / "missing"])
    assert monitor.psi([0, 0], [1, 2]) == 0.0


def test_monitor_snapshot_of_an_empty_output():
    df = pd.DataFrame({"source": [], "record_id": [], "email": []}, dtype=object)
    m = monitor.snapshot(df, np.array([], int), np.array([]), 0.5)
    assert m["records"] == 0 and m["match_rate"] == 0.0 and m["largest_customer"] == 0


# ------------------------------------------------------------------ survivorship
@pytest.mark.parametrize("rule,expected", [
    ("first_seen", "Jon"), ("most_common", "John"), ("longest", "Jonathan"),
    ("source_priority", "Jonathan"), ("most_complete", "John")])
def test_each_survivorship_rule(rule, expected):
    values, sources = ["Jon", "John", "John", "Jonathan", ""], ["web", "app", "app", "store", "store"]
    assert golden.survive(values, sources, rule, {"store": 0, "app": 1, "web": 2}) == expected


def test_survivorship_edge_cases():
    assert golden.survive(["", " "], ["a", "b"], "most_common") == ""
    assert golden.survive(["x", "y"], ["a", "b"], "most_recent", dates=["", ""]) in ("x", "y")   # no dates: falls back
    assert golden.survive(["x", "y"], ["a", "b"], {"prefer": ["b"]}) == "y"
    with pytest.raises(ValueError):
        golden.survive(["x"], ["a"], "newest")
    assert golden.check({"email": {"prefer": ["a"], "else": "x"}}, ["email"])   # unknown key
    assert golden.check({"email": {"prefer": ["a"], "then": "newest"}}, ["email"])


def test_golden_address_parts_come_from_one_record():
    df = pd.DataFrame({"source": ["a", "b"], "addr_number": ["1", "22"], "addr_street": ["", "high st"],
                       "addr_unit": ["", ""], "updated_at": ["2024-01-01", "2020-01-01"], "first_name": ["x", "x"]})
    g = golden.build_golden(df, np.array([0, 0]), {"address": "most_recent"}, ["addr_number", "addr_street", "addr_unit"])
    assert g[0]["addr_number"] == "1" and g[0]["addr_street"] == ""        # the newer record, whole
    g = golden.build_golden(df, np.array([0, 1]), "most_complete", ["first_name"])
    assert g == {0: {"first_name": "x"}, 1: {"first_name": "x"}}


# ------------------------------------------------------------------ model helpers
def test_model_factory_and_helpers():
    with pytest.raises(ValueError, match="matcher_model"):
        models.factory({"matcher_model": "svm"})
    assert models.factory({"matcher_model": "forest"})() is not None
    with pytest.raises(ValueError, match="removed in 0.1.0"):
        models.factory({"matcher_model": "catboost"})


def test_single_class_models():
    class OneClass:
        classes_ = np.array([1])

        def predict_proba(self, X):
            return np.ones((len(X), 1))
    assert models.p_match(OneClass(), np.zeros((2, 1))).tolist() == [1.0, 1.0]
    assert len(models.p_match(OneClass(), np.zeros((0, 1)))) == 0


# ------------------------------------------------------------------ the command line
def test_cli_error_paths(tmp_path, capsys):
    from custmatch import cli
    assert cli.main(["run", "nope.json", "--out", str(tmp_path / "o")]) == 2
    assert "needs --labels" in capsys.readouterr().err
    assert cli.main(["add", "missing.json", "--state", str(tmp_path / "none"), "--out", str(tmp_path / "o")]) == 2
    assert cli.main(["forget", "missing.csv", "--state", str(tmp_path / "none"), "--out", str(tmp_path / "o")]) == 2
    assert cli.main(["monitor", str(tmp_path / "none")]) == 2


def test_python_dash_m_custmatch(monkeypatch):
    import runpy
    monkeypatch.setattr(sys, "argv", ["custmatch", "--help"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("custmatch", run_name="__main__")
    assert e.value.code == 0


# ------------------------------------------------------------------ twins and accented names
def test_twin_pairs_flags_twins_but_not_nicknames_typos_or_initials():
    df = pd.DataFrame({"first_name": ["james", "robert", "jim", "jmaes", "j", "andrew", "ándréw", "james"],
                       "last_name": ["smith"] * 7 + ["jones"],
                       "date_of_birth": ["1980-01-01"] * 8})
    from custmatch import country
    df["first_roots"] = ["|".join(country.first_roots(x)) for x in df.first_name]
    pairs = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [5, 6], [0, 7]])
    assert stewardship.twin_pairs(df, pairs).tolist() == [True, False, False, False, False, False]
    df.loc[1, "date_of_birth"] = "1980-01-02"
    assert not stewardship.twin_pairs(df, pairs[:1]).any()                   # different birthdays: not twins
    assert len(stewardship.twin_pairs(df, np.zeros((0, 2), int))) == 0


def test_accented_names_get_their_nickname_roots():
    from custmatch import country
    # Ándréw meets Andy through the (downloaded) US nickname list; no Wikidata list needed
    assert set(country.first_roots("ándréw")) & set(country.first_roots("andy"))
    assert "jose" in country.first_roots("josé")


# ------------------------------------------------------------------ third-party data, fetched
def test_third_party_name_data_is_downloaded_checked_and_cached(tmp_path, monkeypatch):
    import hashlib
    import io as stdio
    import urllib.request
    from custmatch import country
    body = b"name1,relationship,name2\nrobert,has_nickname,bob\n"
    monkeypatch.setenv("CUSTMATCH_DATA", str(tmp_path))
    monkeypatch.setitem(country.THIRD_PARTY, "test.csv", ("https://example.org/t.csv", hashlib.sha256(body).hexdigest()))
    calls = []
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout, **kw: calls.append(url) or stdio.BytesIO(body))
    assert country.data_file("test.csv").read_bytes() == body
    assert country.data_file("test.csv").exists() and len(calls) == 1              # cached: fetched once
    monkeypatch.setitem(country.THIRD_PARTY, "bad.csv", ("https://example.org/b.csv", "0" * 64))
    with pytest.raises(io.ConfigError, match="checksum"):
        country.data_file("bad.csv")
    assert not (tmp_path / "bad.csv").exists()

    def offline(url, timeout, **kw):
        raise OSError("no network")
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    monkeypatch.setitem(country.THIRD_PARTY, "off.csv", ("https://example.org/o.csv", "0" * 64))
    with pytest.raises(io.ConfigError, match="nicknames"):
        country.data_file("off.csv")
    assert country.data_file("not_third_party.csv") == tmp_path / "not_third_party.csv"


# ------------------------------------------------------------------ ideas from anonlink
def test_unique_source_pairs_flags_two_records_of_one_source_in_a_customer():
    df = pd.DataFrame({"source": ["roll", "roll", "web", "roll", "web"]})
    labels_ = np.array([0, 0, 0, 1, 1])
    pairs = stewardship.unique_source_pairs(df, labels_, {"roll"})
    assert pairs.tolist() == [[0, 1]]                                         # web may repeat
    assert stewardship.unique_sources({"unique_sources": True}, df) == {"roll", "web"}
    assert stewardship.unique_sources({}, df) == set()
    assert len(stewardship.unique_source_pairs(df, labels_, set())) == 0


def test_lsh_blocking_joins_names_with_typos_in_both_and_respects_with():
    from custmatch.blocking import lsh_keys
    from collections import defaultdict
    df = pd.DataFrame({"first_name": ["martha", "mar6ha", "martha", "zed"],
                       "last_name": ["little", "litttle", "little", "quux"],
                       "town": ["ws", "ws", "raleigh", "ws"]})
    blocks = defaultdict(set)
    for i, k in lsh_keys(df, {"fields": ["first_name", "last_name"], "with": "town"}):
        blocks[k].add(i)
    together = {tuple(sorted(v)) for v in blocks.values() if len(v) > 1}
    assert together == {(0, 1)}                         # not across towns, not with the stranger
    no_with = defaultdict(set)
    for i, k in lsh_keys(df, {"fields": ["first_name", "last_name"]}):
        no_with[k].add(i)
    assert any({0, 2} <= v for v in no_with.values())


# ------------------------------------------------------------------ inputs real users produce
# (scripts/robustness.py runs the CLI over the same cases end to end)
def _demo(tmp_path):
    import shutil
    d = tmp_path / "demo"
    shutil.copytree(Path(__file__).resolve().parents[1] / "examples/demo", d)
    return d


def _set(d, **changes):
    cfg = json.loads((d / "config.json").read_text()) | changes
    (d / "config.json").write_text(json.dumps(cfg))


def test_bad_config_and_files_are_clear_config_errors(tmp_path):
    d = _demo(tmp_path)
    (d / "config.json").write_text("{ not json")
    with pytest.raises(io.ConfigError, match="not valid JSON"):
        io.load(d / "config.json")
    d = _demo(tmp_path / "2")
    pd.read_csv(d / "web.csv", dtype=str).assign(first_name="josé").to_csv(d / "web.csv", index=False, encoding="latin-1")
    with pytest.raises(io.ConfigError, match='set "encoding"'):
        io.load(d / "config.json")
    cfg = json.loads((d / "config.json").read_text())
    cfg["sources"][0]["encoding"] = "latin-1"
    (d / "config.json").write_text(json.dumps(cfg))
    df, _ = io.load(d / "config.json")
    assert "josé" in set(df.first_name)
    d = _demo(tmp_path / "3")
    for n in ("web.csv", "app.csv", "store.csv"):
        pd.read_csv(d / n, dtype=str).head(0).to_csv(d / n, index=False)
    with pytest.raises(io.ConfigError, match="no records"):
        io.load(d / "config.json")


def test_config_mistakes_that_used_to_pass_silently(tmp_path):
    for changes, msg in (({"country": "XX"}, "unknown country"),
                         ({"extra_fields": [{"name": "loyalty_tier", "compare": "exact"}]}, "not mapped"),
                         ({"extra_fields": [{"name": "postcode", "compare": "fuzzy"}]}, "compare must be")):
        d = _demo(tmp_path / msg.replace(" ", "_"))
        _set(d, **changes)
        with pytest.raises(io.ConfigError, match=msg):
            df, cfg = io.load(d / "config.json")
            from custmatch.schema import extra_fields
            extra_fields(cfg)
    d = _demo(tmp_path / "blank_ids")
    w = pd.read_csv(d / "web.csv", dtype=str, keep_default_na=False)
    w.loc[:3, "record_id"] = ""
    w.to_csv(d / "web.csv", index=False)
    with pytest.raises(io.ConfigError, match="without a record_id"):
        io.load(d / "config.json")
    cfg = {"sources": [{"name": "web", "columns": {"record_id": "id", "favourite_colour": "c"}}]}
    assert any("favourite_colour" in w for w in io.unknown_keys(cfg))


def test_skipped_common_blocks_are_reported_but_blank_ones_are_not(tmp_path):
    from custmatch.blocking import candidates_for
    from custmatch.core import profile_contacts
    d = _demo(tmp_path)
    df, cfg = io.load(d / "config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    report = candidates_for(df, prof, cfg)[2]
    assert report["records_only_in_skipped_blocks"] < 0.01 * len(df)       # only blank keys skipped on the demo
    for n in ("web.csv", "app.csv", "store.csv"):
        x = pd.read_csv(d / n, dtype=str, keep_default_na=False)
        x["first_name"], x["last_name"] = "john", "smith"
        x.to_csv(d / n, index=False)
    df, cfg = io.load(d / "config.json")
    report = candidates_for(df, {f: profile_contacts(df, f) for f in ("email", "phone")}, cfg)[2]
    # everyone is john smith: those with no email or phone to meet on were never compared
    assert 0 < report["records_only_in_skipped_blocks"] < len(df)
    assert "smith" in report["largest_skipped_blocks"][0]["key"]


def test_trusted_ids_reach_the_run_without_being_an_extra_field(tmp_path):
    """A trusted-ID column mapped in "columns" is kept by the loader and links its records, as the
    guide says - it used to be dropped unless also declared in "extra_fields"."""
    d = pd.DataFrame({"id": ["1", "2", "3"], "first": ["ann", "zed", "bo"], "last": ["lee", "quist", "ng"],
                      "card": ["L1", "L1", ""]})
    d.to_csv(tmp_path / "s.csv", index=False)
    cfg = {"sources": [{"name": "s", "path": "s.csv", "columns": {"record_id": "id", "first_name": "first",
                                                                  "last_name": "last", "loyalty": "card"}}],
           "trusted_ids": ["loyalty"]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    df, cfg = io.load(tmp_path / "config.json")
    assert list(df.loyalty.str.lower()) == ["l1", "l1", ""]
    assert stewardship.trusted_pairs(df, cfg).tolist() == [[0, 1]]


def test_feature_names_line_up_with_the_feature_columns():
    """core.feature_names names every column core.features builds (the dashboard's waterfall and
    feature importance use them), with and without the rich features and extra fields."""
    from custmatch import schema
    from custmatch.core import feature_names, features, profile_contacts, record_arrays
    df, cfg = io.load(Path(__file__).resolve().parents[1] / "examples/demo/config.json")
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    for rich in (True, False):
        opts = schema.feature_options(cfg | {"rich_features": rich})
        X = features(df, np.array([[0, 1], [2, 3]]), prof, **(opts | {"records": record_arrays(df)}))
        assert X.shape[1] == len(feature_names(df.columns, opts["extra_fields"], opts["rich"]))
