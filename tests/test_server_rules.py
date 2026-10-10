"""The batch run's rules in the real-time matcher (server/custmatch_server/matcher.py): stewards'
overrides through the API, trusted IDs, duplicate-free sources and the twin guard - each applied as
records arrive, kept across a matcher restart, and agreed with by the next batch run.
SQLite by default; CUSTMATCH_RULES_DB_URL=postgresql+psycopg://... runs it on a scratch Postgres
database (its schema is dropped). Skipped where the server's dependencies are absent."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/demo"
pytestmark = pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("fastapi", "sqlalchemy", "httpx")),
                                reason="pip install ./server (fastapi, sqlalchemy) to run the service tests")


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    sys.path.insert(0, str(ROOT / "server"))
    from fastapi.testclient import TestClient

    from custmatch_server import db
    from custmatch_server.api import create_app
    from custmatch_server.loader import load_file
    from custmatch_server.worker import run_batch
    tmp = tmp_path_factory.mktemp("rules")
    cfg = json.loads((DEMO / "config.json").read_text())
    for s in cfg["sources"]:
        d = pd.read_csv(DEMO / s["path"], dtype=str, keep_default_na=False)
        if s["name"] == "store":                        # a loyalty number on one store record
            d["loyalty"] = ""
            d.loc[0, "loyalty"] = "L-0001"
            s["columns"]["loyalty"] = "loyalty"
        d.to_csv(tmp / s["path"], index=False)
    cfg |= {"trusted_ids": ["loyalty"], "unique_sources": ["web"], "twin_guard": True}
    (tmp / "config.json").write_text(json.dumps(cfg))
    (tmp / "labels.csv").write_bytes((DEMO / "labels.csv").read_bytes())
    from test_support import fresh_engine
    eng = fresh_engine(os.environ.get("CUSTMATCH_RULES_DB_URL") or f"sqlite:///{tmp / 'rules.db'}")
    for s in cfg["sources"]:
        load_file(eng, cfg, s["name"], tmp / s["path"], mode="full")

    def batch():
        run_id, _ = run_batch(eng, tmp / "config.json", tmp / "labels.csv", tmp / "data", log=lambda *_: None)
        with eng.connect() as c:
            art = Path(db.current(c)["artifact_uri"])
        cl = pd.read_csv(art / "clusters.csv", dtype=str)
        return dict(zip(cl.source + ":" + cl.record_id, cl.customer_id))
    first = batch()
    from custmatch_server import auth
    key = auth.create_key(eng, "rules-test", "admin")
    client = TestClient(create_app(eng, require_keys=True), headers={"Authorization": f"Bearer {key}"})
    return {"eng": eng, "client": client, "batch": batch, "first": first, "cfg": cfg, "tmp": tmp}


def _customer(client, key):
    from test_support import customer_of
    return customer_of(client, key)


def _two_customers(first, source_a="app", source_b="store", skip=0):
    """Two records, of two customers, from the given sources."""
    a = [k for k in first if k.startswith(source_a + ":")][skip]
    b = next(k for k in first if k.startswith(source_b + ":") and first[k] != first[a])
    return a, b


def test_same_merges_now_survives_a_restart_and_the_batch_run_agrees(service):
    from custmatch_server import db
    from custmatch_server.matcher import Matcher
    client, first = service["client"], service["first"]
    a, b = _two_customers(first, skip=3)
    r = client.post("/overrides", json={"a": a, "b": b, "decision": "same", "created_by": "test"}).json()
    assert r["applied"].startswith("merged"), r
    assert _customer(client, a) == _customer(client, b)
    with service["eng"].connect() as c:
        m = Matcher(service["eng"], db.current(c))     # a restart replays the merge
    assert m.lookup(*a.split(":")) == m.lookup(*b.split(":"))
    again = service["batch"]()
    assert again[a] == again[b], "the batch run did not honour the steward's 'same'"


def _current(service):
    cl = pd.read_csv(sorted((service["tmp"] / "data/runs").glob("run-*"))[-1] / "result/clusters.csv", dtype=str)
    return dict(zip(cl.source + ":" + cl.record_id, cl.customer_id))


def test_different_splits_a_customer_at_the_batch_run(service):
    client, current = service["client"], _current(service)
    groups = pd.Series(list(current)).groupby(pd.Series(list(current.values())))
    big = next(list(g) for _, g in groups if len(g) >= 3 and not any(k.startswith("web:") for k in g))
    a, b = big[:2]
    r = client.post("/overrides", json={"a": a, "b": b, "decision": "different"}).json()
    assert "split at the next batch run" in r["applied"], r
    again = service["batch"]()
    assert again[a] != again[b], "the batch run did not split the steward's 'different'"


def test_different_blocks_a_later_same_across_the_two_customers(service):
    client, current = service["client"], _current(service)
    by = pd.Series(list(current)).groupby(pd.Series(list(current.values())))
    pairs = [list(g) for _, g in by if len(g) >= 2 and not any(k.startswith("web:") for k in g)]
    (a, a2), (b, b2) = pairs[0][:2], pairs[1][:2]
    r = client.post("/overrides", json={"a": a, "b": b, "decision": "different"}).json()
    assert r["applied"] == "stored; already so", r
    r = client.post("/overrides", json={"a": a2, "b": b2, "decision": "same"}).json()
    assert "not applied now" in r["applied"] and "different" in r["applied"], r
    assert _customer(client, a2) != _customer(client, b2)


def test_a_shared_trusted_id_joins_whatever_the_names_say(service):
    client = service["client"]
    store = pd.read_csv(service["tmp"] / "store.csv", dtype=str, keep_default_na=False)
    holder = f"store:{store.record_id[0]}"
    fields = {"first_name": "Xanthe", "last_name": "Oyelaran", "email": "xo@nowhere.example", "loyalty": "L-0001"}
    r = client.post("/match", json={"source": "store", "record_id": "rules-loyal", "fields": fields, "write": True}).json()
    assert r["customer_id"] == _customer(client, holder), r
    assert any("trusted ID loyalty" in x for x in r["rules"]), r


def test_a_duplicate_free_source_never_holds_two_records_of_one_customer(service):
    client = service["client"]
    web = pd.read_csv(service["tmp"] / "web.csv", dtype=str, keep_default_na=False)
    row = web.iloc[5].to_dict()
    r = client.post("/match", json={"source": "web", "record_id": "rules-dup", "fields": row, "write": True}).json()
    assert r["customer_id"] != _customer(client, f"web:{row['record_id']}"), r
    assert any("duplicate-free source" in x for x in r["rules"]), r


def test_a_twin_is_kept_apart(service):
    client = service["client"]
    app = pd.read_csv(service["tmp"] / "app.csv", dtype=str, keep_default_na=False)
    row = app[(app.date_of_birth != "") & (app.last_name != "") & (app.first_name.str.len() >= 3)].iloc[0].to_dict()
    twin = dict(row, first_name="Quintessa" if row["first_name"].lower() != "quintessa" else "Bartholomew")
    original = _customer(client, f"app:{row['record_id']}")
    r = client.post("/match", json={"source": "app", "record_id": "rules-twin", "fields": twin, "write": False}).json()
    assert r["customer_id"] != original, r
    same = client.post("/match", json={"source": "app", "record_id": "rules-same", "fields": row, "write": False}).json()
    assert same["customer_id"] == original, same          # the guard is about the name, not the record


def test_steward_decisions_become_active_labels():
    """worker._with_decisions: same -> 1, different -> 0, marked active (they train the model but
    stay out of the threshold estimate); a decision replaces a label on the same pair, either way round."""
    sys.path.insert(0, str(ROOT / "server"))
    from custmatch_server.worker import _with_decisions
    lab = pd.DataFrame({"a": ["s:1", "s:3"], "b": ["s:2", "s:4"], "label": ["1", "0"], "by": ["random", "random"]})
    dec = pd.DataFrame({"a": ["s:2", "s:5"], "b": ["s:1", "s:6"], "decision": ["different", "same"]})
    out, n = _with_decisions(lab, dec)
    got = {tuple(sorted((a, b))): (lbl, by) for a, b, lbl, by in zip(out.a, out.b, out.label, out.by)}
    assert n == 2 and got == {("s:1", "s:2"): ("0", "active"), ("s:3", "s:4"): ("0", "random"),
                              ("s:5", "s:6"): ("1", "active")}


def test_a_record_joins_a_customer_only_if_it_fits_most_of_it():
    """One strong link to one record of a customer is not enough: the record must match at least
    half of the customer's records."""
    sys.path.insert(0, str(ROOT / "server"))
    import numpy as np

    from custmatch_server.matcher import COHERENCE, Matcher
    m = Matcher.__new__(Matcher)
    m.coherence, m.model = COHERENCE, {"threshold": 0.5}
    members = {1: [10, 11, 12], 2: [20, 21, 22], 3: [30], 4: [40, 41]}
    scores = {10: 0.95, 11: 0.02, 12: 0.1, 20: 0.9, 21: 0.8, 22: 0.1, 30: 0.9, 40: 0.95, 41: 0.02}
    m._members = lambda c: members[c]
    m._score = lambda row, idx: np.array([scores[i] for i in idx])
    assert "1 of 3" in m._incoherent({}, 1)        # one record of it matches, the others are someone else
    assert m._incoherent({}, 2) is None             # matches 2 of 3
    assert m._incoherent({}, 3) is None             # a one-record customer: its link decides
    assert m._incoherent({}, 4) is None             # 1 of 2 is half: still joins (stricter cost recall)
    m.coherence = 0
    assert m._incoherent({}, 1) is None             # the check off
