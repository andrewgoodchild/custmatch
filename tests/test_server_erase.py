"""Erasure requests in the service (server/custmatch_server/erase.py): one operation reaches the
source records and their versions, what stewards wrote about them, the real-time matcher's rows
and memory, the current run's tables and files, and the retired runs' files. SQLite; skipped
where the server's dependencies are absent."""
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/demo"
pytestmark = pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("fastapi", "sqlalchemy", "httpx")),
                                reason="pip install ./server (fastapi, sqlalchemy) to run the service tests")


@pytest.fixture()
def svc(tmp_path):
    sys.path.insert(0, str(ROOT / "server"))
    from custmatch_server import auth, db
    from custmatch_server.api import create_app
    from custmatch_server.loader import load_file
    from custmatch_server.worker import run_batch
    from fastapi.testclient import TestClient
    from test_support import fresh_engine
    cfg = json.loads((DEMO / "config.json").read_text())
    eng = fresh_engine(f"sqlite:///{tmp_path / 'e.db'}")
    for s in cfg["sources"]:
        load_file(eng, cfg, s["name"], DEMO / s["path"], mode="full")
    data = tmp_path / "data"
    run_batch(eng, DEMO / "config.json", DEMO / "labels.csv", data, log=lambda *_: None)
    with eng.connect() as c:
        cl = pd.read_csv(Path(db.current(c)["artifact_uri"]) / "clusters.csv", dtype=str)
    first = dict(zip(cl.source + ":" + cl.record_id, cl.customer_id))
    client = TestClient(create_app(eng, require_keys=True),
                        headers={"Authorization": f"Bearer {auth.create_key(eng, 'ann', 'steward')}"})
    return {"eng": eng, "client": client, "first": first, "data": data}


def _rows(eng, table, **where):
    from sqlalchemy import select
    sel = select(table)
    for k, v in where.items():
        sel = sel.where(table.c[k] == v)
    with eng.connect() as c:
        return c.execute(sel).mappings().all()


def test_erasure_reaches_every_copy_and_the_matcher_reloads(svc):
    from custmatch_server import db
    from custmatch_server.erase import erase
    from custmatch_server.schema import (
        customers,
        golden_pins,
        links,
        overrides,
        realtime_records,
        record_customer,
        source_record_versions,
        source_records,
        steward_log,
    )
    eng, client, first = svc["eng"], svc["client"], svc["first"]
    by = pd.Series(list(first)).groupby(pd.Series(list(first.values())))
    recs = next(list(g) for _, g in by if len(g) >= 2 and any(k.startswith("web:") for k in g))
    key = next(k for k in recs if k.startswith("web:"))
    other = next(k for k in recs if k != key)
    source, record_id = key.split(":", 1)
    email = client.get(f"/sources/{source}/records/{record_id}").json()["fields"]["email"]
    # what stewards and the real-time matcher wrote about the person
    assert client.post("/pins", json={"record_key": key, "field": "email", "value": "pinned@example.com"}).status_code == 200
    assert client.post("/overrides", json={"a": key, "b": other, "decision": "same", "note": "a steward's note"}).status_code == 200
    rt = client.post("/match", json={"source": "web", "record_id": "rt-1", "write": True,
                                     "fields": {"first_name": "Zuleika", "last_name": "Quenneville",
                                                "email": "zq@nowhere.example", "mobile": "0499 000 111"}}).json()
    assert rt["written"]
    with eng.connect() as c:
        run1 = db.current(c)
    assert _rows(eng, realtime_records, source="web", record_id="rt-1")
    # a dry run reports where the data is and changes nothing
    dry = erase(eng, [key, "web:rt-1"], DEMO / "config.json", DEMO / "labels.csv", svc["data"], dry_run=True, log=lambda *_: None)
    assert dry["dry_run"] and dry["run"] is None and dry["rows"]["source_records"] == 2
    assert dry["rows"]["record_customer"] == 1 and dry["rows"]["overrides"] == 1 and dry["rows"]["golden_pins"] == 1
    assert _rows(eng, source_records, source=source, record_id=record_id)
    with pytest.raises(ValueError, match="not in the database"):
        erase(eng, ["web:nobody"], DEMO / "config.json", DEMO / "labels.csv", svc["data"], log=lambda *_: None)
    report = erase(eng, [key, "web:rt-1"], DEMO / "config.json", DEMO / "labels.csv", svc["data"],
                   note="ticket 42", actor="ops", log=lambda *_: None)
    assert report["run"] is not None and report["retired"] == [run1["id"]]
    # the database: the records, their versions, the decisions, the pin, the real-time rows
    assert not _rows(eng, source_records, source=source, record_id=record_id)
    assert not _rows(eng, source_records, source="web", record_id="rt-1")
    assert not _rows(eng, source_record_versions, source=source, record_id=record_id)
    assert not _rows(eng, overrides, a=min(key, other), b=max(key, other))
    assert not _rows(eng, golden_pins, record_key=key)
    assert not _rows(eng, realtime_records, source="web", record_id="rt-1")
    log = [dict(r) for r in _rows(eng, steward_log) if key in (r["a"], r["b"])]
    assert {r["action"] for r in log} >= {"pinned", "decided", "erased"}
    assert all(r["detail"] is None and r["note"] is None for r in log if r["action"] != "erased")
    erased = next(r for r in log if r["action"] == "erased")
    assert erased["actor"] == "ops" and erased["note"] == "ticket 42"
    # the runs: the new one holds nothing of them, the old one is retired and its files are gone
    with eng.connect() as c:
        cur = db.current(c)
    assert cur["id"] == report["run"] and not _rows(eng, record_customer, run_id=cur["id"], source=source, record_id=record_id)
    assert not [r for r in _rows(eng, links, run_id=cur["id"]) if key in (f"{r['source_a']}:{r['record_a']}", f"{r['source_b']}:{r['record_b']}")]
    assert not [r for r in _rows(eng, customers, run_id=cur["id"]) if r["email"] == "pinned@example.com"]
    assert not _rows(eng, record_customer, run_id=run1["id"])
    assert _rows(eng, db.runs, id=run1["id"])[0]["status"] == "retired"
    old = Path(run1["artifact_uri"])
    assert not old.exists() and not list((old.parent / "inputs").glob("*.parquet"))
    assert (old.parent / "inputs" / "config.json").exists()
    new = Path(cur["artifact_uri"])
    state = pd.read_parquet(new / "state" / "records.parquet")
    assert not ((state.source == source) & (state.record_id == record_id)).any()
    review = pd.read_csv(new / "review.csv", dtype=str, keep_default_na=False)
    assert not (review.a.eq(key) | review.b.eq(key)).any()
    # the API, once its matcher has reloaded from the new run (the customer's other records may
    # share the email: a search for it must not find the erased record)
    time.sleep(1.1)
    assert client.get(f"/records/{source}/{record_id}").status_code == 404
    assert client.get("/records/web/rt-1").status_code == 404
    assert client.get(f"/sources/{source}/records/{record_id}").status_code == 404
    assert key not in {r["key"] for r in client.get("/search", params={"q": email}).json()}
    assert client.get("/search", params={"q": "zq@nowhere.example"}).json() == []
    assert key not in client.get(f"/customers/{first[other]}").json()["records"]


def test_erasure_before_any_run_deletes_the_rows_only(tmp_path):
    sys.path.insert(0, str(ROOT / "server"))
    from custmatch_server.erase import erase
    from custmatch_server.loader import load_file
    from custmatch_server.schema import runs, source_records
    from test_support import fresh_engine
    cfg = json.loads((DEMO / "config.json").read_text())
    eng = fresh_engine(f"sqlite:///{tmp_path / 'n.db'}")
    load_file(eng, cfg, "web", DEMO / "web.csv", mode="full")
    key = "web:" + pd.read_csv(DEMO / "web.csv", dtype=str).record_id[0]
    report = erase(eng, [key], DEMO / "config.json", DEMO / "labels.csv", tmp_path / "data", log=lambda *_: None)
    assert report["run"] is None and report["retired"] == []
    assert not _rows(eng, source_records, source="web", record_id=key.split(":", 1)[1]) and not _rows(eng, runs)


def test_erase_from_the_command_line(tmp_path, monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "server"))
    from custmatch_server.__main__ import main
    from custmatch_server.loader import load_file
    from test_support import fresh_engine
    cfg = json.loads((DEMO / "config.json").read_text())
    url = f"sqlite:///{tmp_path / 'c.db'}"
    eng = fresh_engine(url)
    load_file(eng, cfg, "web", DEMO / "web.csv", mode="full")
    eng.dispose()
    monkeypatch.setenv("CUSTMATCH_DB_URL", url)
    monkeypatch.setenv("CUSTMATCH_DATA_DIR", str(tmp_path / "data"))
    ids = pd.read_csv(DEMO / "web.csv", dtype=str).record_id[:2]
    pd.DataFrame({"source": "web", "record_id": ids}).to_csv(tmp_path / "erase.csv", index=False)
    assert main(["erase", str(DEMO / "config.json"), "--labels", str(DEMO / "labels.csv"), "--file", str(tmp_path / "erase.csv"),
                 "--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] and out["erased"] == [f"web:{i}" for i in ids] and out["rows"]["source_records"] == 2
