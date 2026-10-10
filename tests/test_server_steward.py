"""The steward's tools (server/README.md, "The dashboard"): previews, splitting a record
out, merging customers, revoking decisions, four-eyes approval, pinned golden values with
lineage, review claims and filters, comparing any two records, activity and audit. Each is
checked now and, where it matters, after the next batch run. SQLite by default;
CUSTMATCH_STEWARD_DB_URL=postgresql+psycopg://... runs them on a scratch Postgres database (its
schema is dropped). Skipped where the server's dependencies are absent."""
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


@pytest.fixture()
def svc(tmp_path, monkeypatch, request):
    sys.path.insert(0, str(ROOT / "server"))
    from fastapi.testclient import TestClient

    from custmatch_server import auth, db
    from custmatch_server.api import create_app
    from custmatch_server.loader import load_file
    from custmatch_server.worker import run_batch
    if getattr(request, "param", None) == "four_eyes":
        monkeypatch.setenv("CUSTMATCH_FOUR_EYES", "on")
    cfg = json.loads((DEMO / "config.json").read_text())
    from test_support import fresh_engine
    eng = fresh_engine(os.environ.get("CUSTMATCH_STEWARD_DB_URL") or f"sqlite:///{tmp_path / 's.db'}")
    for s in cfg["sources"]:
        load_file(eng, cfg, s["name"], DEMO / s["path"], mode="full")

    def batch():
        run_batch(eng, DEMO / "config.json", DEMO / "labels.csv", tmp_path / "data", log=lambda *_: None)
        with eng.connect() as c:
            cl = pd.read_csv(Path(db.current(c)["artifact_uri"]) / "clusters.csv", dtype=str)
        return dict(zip(cl.source + ":" + cl.record_id, cl.customer_id))
    first = batch()
    app = create_app(eng, require_keys=True)
    ann = TestClient(app, headers={"Authorization": f"Bearer {auth.create_key(eng, 'ann', 'steward')}"})
    bob = TestClient(app, headers={"Authorization": f"Bearer {auth.create_key(eng, 'bob', 'steward')}"})
    return {"ann": ann, "bob": bob, "batch": batch, "first": first, "eng": eng}


def _customer_of(client, key):
    from test_support import customer_of
    return customer_of(client, key)


def _big(first, n=3):
    """A customer of at least n records."""
    groups = pd.Series(list(first)).groupby(pd.Series(list(first.values())))
    return next(list(g) for _, g in groups if len(g) >= n)


def test_preview_changes_nothing_then_split_out_now_and_kept_apart_later(svc):
    ann, first = svc["ann"], svc["first"]
    recs = _big(first)
    cid = _customer_of(ann, recs[0])
    pv = ann.post("/split", json={"record_key": recs[0], "dry_run": True}).json()
    assert pv["preview"] and _customer_of(ann, recs[0]) == cid
    done = ann.post("/split", json={"record_key": recs[0], "note": "different person"}).json()
    new = done["to_customer"]
    assert _customer_of(ann, recs[0]) == new != cid
    cust = ann.get(f"/customers/{cid}").json()
    assert recs[0] not in cust["records"] + cust["provisional_records"] and recs[0] in cust["moved_out"]
    assert ann.get(f"/customers/{new}").json()["provisional_records"] == [recs[0]]
    audit = ann.get(f"/customers/{cid}/audit").json()
    assert any(a["action"] == "split" and a["actor"] == "ann" for a in audit)
    again = svc["batch"]()
    assert all(again[recs[0]] != again[o] for o in recs[1:]), "the next run merged the split record back"


def test_merge_two_customers_now_and_kept_together_later(svc):
    ann, first = svc["ann"], svc["first"]
    a = next(k for k in first if k.startswith("app:"))
    b = next(k for k in first if k.startswith("store:") and first[k] != first[a])
    ca, cb = _customer_of(ann, a), _customer_of(ann, b)
    pv = ann.post(f"/customers/{ca}/merge", json={"other": int(cb), "dry_run": True}).json()
    assert pv["preview"] and "merged customer" in pv["applied"] and _customer_of(ann, b) == cb
    ann.post(f"/customers/{ca}/merge", json={"other": int(cb)})
    assert _customer_of(ann, a) == _customer_of(ann, b)
    again = svc["batch"]()
    assert again[a] == again[b]


def test_revoking_a_decision_lets_the_model_decide_again(svc):
    ann, first = svc["ann"], svc["first"]
    recs = _big(first, 2)
    r = ann.post("/overrides", json={"a": recs[0], "b": recs[1], "decision": "different"}).json()
    assert "split at the next batch run" in r["applied"]
    oid = next(o["id"] for o in ann.get("/overrides").json() if {o["a"], o["b"]} == set(recs[:2]))
    assert "revoked" in ann.delete(f"/overrides/{oid}").json()["applied"]
    assert not ann.get("/overrides").json()
    again = svc["batch"]()
    assert again[recs[0]] == again[recs[1]], "a revoked 'different' still split them"
    csv = ann.get("/overrides?format=csv")
    assert csv.headers["content-type"].startswith("text/csv")


@pytest.mark.parametrize("svc", ["four_eyes"], indirect=True)
def test_four_eyes_a_decision_waits_for_a_second_steward(svc):
    ann, bob, first = svc["ann"], svc["bob"], svc["first"]
    a = next(k for k in first if k.startswith("app:"))
    b = next(k for k in first if k.startswith("store:") and first[k] != first[a])
    r = ann.post("/overrides", json={"a": a, "b": b, "decision": "same"}).json()
    assert r["applied"].startswith("proposed") and _customer_of(ann, a) != _customer_of(ann, b)
    oid = ann.get("/overrides?status=proposed").json()[0]["id"]
    assert ann.post(f"/overrides/{oid}/approve", json={}).status_code == 409        # not your own
    assert bob.post(f"/overrides/{oid}/approve", json={}).json()["applied"].startswith("merged")
    assert _customer_of(ann, a) == _customer_of(ann, b)
    st = bob.get("/stewards").json()
    assert st["four_eyes"] and {s["steward"] for s in st["stewards"]} == {"ann", "bob"}


def test_pinned_value_beats_survivorship_now_and_after_the_batch_run(svc):
    ann, first = svc["ann"], svc["first"]
    recs = _big(first, 2)
    cid = _customer_of(ann, recs[0])
    lin = ann.get(f"/customers/{cid}/lineage").json()
    fn = next(f for f in lin["fields"] if f["field"] == "first_name")
    assert sum(v["won"] for v in fn["values"]) == 1 and all(v["records"] for v in fn["values"])
    ann.post("/pins", json={"record_key": recs[0], "field": "first_name", "value": "Pinnedname"})
    c = ann.get(f"/customers/{cid}").json()
    assert c["golden_record"]["first_name"] == "Pinnedname" and c["pinned"]["first_name"]["by"] == "ann"
    svc["batch"]()
    cid2 = _customer_of(ann, recs[0])
    assert ann.get(f"/customers/{cid2}").json()["golden_record"]["first_name"] == "Pinnedname"
    pid = c["pinned"]["first_name"]["id"]
    assert ann.delete(f"/pins/{pid}").json()["unpinned"] == "first_name"


def test_review_claims_and_filters(svc):
    ann, bob = svc["ann"], svc["bob"]
    a_page = ann.get("/review?limit=5").json()
    assert a_page["pairs"]
    b_page = bob.get("/review?limit=50").json()
    seen_by_ann = {(p["a"], p["b"]) for p in a_page["pairs"]}
    assert not seen_by_ann & {(p["a"], p["b"]) for p in b_page["pairs"]}, "two stewards were shown the same pair"
    assert b_page["held_by_others"] == len(seen_by_ann)
    web = ann.get("/review?limit=50&source=web").json()
    assert all(p["a"].startswith("web:") or p["b"].startswith("web:") for p in web["pairs"])
    linked = ann.get("/review?limit=50&call=linked").json()
    assert all(p["linked"] for p in linked["pairs"])


def test_compare_says_why_two_records_were_never_compared(svc):
    ann, first = svc["ann"], svc["first"]
    recs = _big(first, 2)
    same = ann.get("/compare", params={"a": recs[0], "b": recs[1]}).json()
    stranger = next(k for k in first if first[k] != first[recs[0]])
    far = ann.get("/compare", params={"a": recs[0], "b": stranger}).json()
    assert "contributions" in same and far["blocking"]["compared"] is False


def _pair_of_customers(first):
    a = next(k for k in first if k.startswith("app:"))
    b = next(k for k in first if k.startswith("store:") and first[k] != first[a])
    return a, b


def test_merge_keeps_the_customer_named(svc):
    ann, first = svc["ann"], svc["first"]
    a, b = _pair_of_customers(first)
    ca, cb = _customer_of(ann, a), _customer_of(ann, b)
    keep, other = (cb, ca) if a < b else (ca, cb)        # name the one whose record sorts last
    ann.post(f"/customers/{keep}/merge", json={"other": int(other)})
    assert _customer_of(ann, a) == _customer_of(ann, b) == keep


@pytest.mark.parametrize("svc", ["four_eyes"], indirect=True)
def test_four_eyes_covers_merges_splits_and_revokes_and_keeps_applied_decisions(svc):
    ann, bob, first = svc["ann"], svc["bob"], svc["first"]
    a, b = _pair_of_customers(first)
    ca, cb = _customer_of(ann, a), _customer_of(ann, b)
    # a merge waits
    assert ann.post(f"/customers/{ca}/merge", json={"other": int(cb)}).json()["applied"].startswith("proposed")
    assert _customer_of(ann, a) != _customer_of(ann, b)
    pid = ann.get("/overrides?status=proposed").json()[0]["id"]
    pv = bob.post(f"/overrides/{pid}/approve", json={"dry_run": True}).json()
    assert pv["preview"] and _customer_of(ann, a) != _customer_of(ann, b)          # a preview changes nothing
    bob.post(f"/overrides/{pid}/approve", json={})
    assert _customer_of(ann, a) == _customer_of(ann, b)
    applied = ann.get("/overrides?status=applied").json()[0]
    # a proposal to change it leaves the applied decision in force
    ann.post("/overrides", json={"a": applied["a"], "b": applied["b"], "decision": "different"})
    st = {(o["status"], o["decision"]) for o in ann.get("/overrides").json()}
    assert ("applied", "same") in st and ("proposed", "different") in st
    # revoking an applied decision is itself a proposal
    assert ann.delete(f"/overrides/{applied['id']}").json()["applied"].startswith("proposed")
    assert any(o["status"] == "applied" for o in ann.get("/overrides").json())
    # a split waits too
    recs = _big(first, 2)
    assert ann.post("/split", json={"record_key": recs[0]}).json()["applied"].startswith("proposed")
    assert _customer_of(ann, recs[0]) == _customer_of(ann, recs[1])
    sid = next(o["id"] for o in ann.get("/overrides?status=proposed").json() if o["decision"] == "split")
    bob.post(f"/overrides/{sid}/approve", json={})
    assert _customer_of(ann, recs[0]) != _customer_of(ann, recs[1])


def test_real_time_ids_survive_the_batch_run_and_are_never_reissued(svc):
    ann, first = svc["ann"], svc["first"]
    person = {"first_name": "Ottoline", "last_name": "Quaggin", "email": "oq@x.example"}
    m_key = ann.post("/match", json={"source": "store", "record_id": "rt-1", "write": True, "fields": person}).json()
    new_id = m_key["customer_id"]
    assert int(new_id) > max(int(v) for v in first.values())
    again = svc["batch"]()
    assert again["store:rt-1"] == new_id, "the batch run moved the real-time customer to another ID"
    holders = [k for k, v in again.items() if v == new_id]
    assert holders == ["store:rt-1"], "another customer was given the real-time customer's ID"


def test_a_write_after_the_runs_snapshot_stays_pending(svc):
    from custmatch_server import db
    from custmatch_server.schema import realtime_records
    from custmatch_server.worker import _realtime_snapshot
    from sqlalchemy import select
    ann, eng = svc["ann"], svc["eng"]
    ann.post("/match", json={"source": "store", "record_id": "rt-before", "write": True,
                              "fields": {"first_name": "Ana", "last_name": "Before", "email": "ab@x.example"}})
    snap, _, _ = _realtime_snapshot(eng)
    ann.post("/match", json={"source": "store", "record_id": "rt-after", "write": True,
                              "fields": {"first_name": "Ana", "last_name": "After", "email": "aa@x.example"}})
    with eng.connect() as c:
        run = db.current(c)
    from sqlalchemy import update
    with eng.begin() as c:                                # what publish(reconcile=snap) does, in isolation
        for table, upto in snap.items():
            if upto is not None:
                c.execute(update(table).where(table.c.reconciled_run.is_(None)).where(table.c.seq <= upto)
                          .values(reconciled_run=run["id"]))
        state = dict(c.execute(select(realtime_records.c.record_id, realtime_records.c.reconciled_run)).all())
    assert state["rt-before"] == run["id"] and state["rt-after"] is None


def test_a_matcher_that_missed_another_process_s_writes_is_rebuilt(svc):
    from custmatch_server import db
    from custmatch_server.matcher import Matcher, StaleMatcher
    ann, eng = svc["ann"], svc["eng"]
    with eng.connect() as c:
        run = db.current(c)
    other = Matcher(eng, run, holder="another-process")                # a second API process
    ann.post("/match", json={"source": "store", "record_id": "rt-x", "write": True,
                              "fields": {"first_name": "Xan", "last_name": "Writer", "email": "xw@x.example"}})
    from custmatch_server.schema import leases
    with eng.begin() as c:                                               # the API's lease runs out
        c.execute(leases.update().values(expires_at=db.now()))
    with pytest.raises(StaleMatcher):
        other.match("store", "rt-y", {"first_name": "Yan"}, write=True)
    # through the API the rebuild is automatic: a write after another process wrote succeeds
    assert ann.post("/match", json={"source": "store", "record_id": "rt-z", "write": True,
                                     "fields": {"first_name": "Zed"}}).status_code == 200


def test_trusted_ids_join_records_that_arrived_in_real_time(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "server"))
    from custmatch_server import db
    from custmatch_server.loader import load_file
    from custmatch_server.matcher import Matcher
    from custmatch_server.worker import run_batch
    cfg = json.loads((DEMO / "config.json").read_text())
    for s in cfg["sources"]:
        d = pd.read_csv(DEMO / s["path"], dtype=str, keep_default_na=False)
        if s["name"] == "store":
            d["loyalty"] = ""
            s["columns"]["loyalty"] = "loyalty"
        d.to_csv(tmp_path / s["path"], index=False)
    cfg["trusted_ids"] = ["loyalty"]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    eng = db.engine_from_env(f"sqlite:///{tmp_path / 't.db'}")
    for s in cfg["sources"]:
        load_file(eng, cfg, s["name"], tmp_path / s["path"], mode="full")
    run_batch(eng, tmp_path / "config.json", DEMO / "labels.csv", tmp_path / "data", log=lambda *_: None)
    with eng.connect() as c:
        m = Matcher(eng, db.current(c), holder="t")
    one = m.match("store", "loyal-1", {"first_name": "Ann", "last_name": "Xu", "loyalty": "L-77"}, write=True)
    two = m.match("store", "loyal-2", {"first_name": "Bea", "last_name": "Yo", "loyalty": "L-77"}, write=True)
    assert one["customer_id"] == two["customer_id"]
