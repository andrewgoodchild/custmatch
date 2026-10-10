"""Smoke test of the service on the bundled demo (1,766 synthetic records), against SQLite or Postgres.

  1. Each source is loaded (a full load) from 90% of the demo; a batch run over the database is
     published, recording the newest load it saw.
  2. Lookups through the API return the batch run's customer for every record.
  3. Later loads: an incremental web load changes some emails (and repeats a record unchanged), a
     full app load leaves three records out (marked deleted); counts and version history checked.
  4. The held-back records arrive one at a time through POST /match (write=true); latency.
  5. Two records of one brand-new person arrive at once: one customer, not two.
  6. A second matcher (as a second process would) cannot write while the first holds the lease.
  7. A second batch run includes the changed and API records, leaves out the deleted ones, and
     reconciles the API's records; a third retires the first run (its partitions dropped on Postgres).
  8. API keys: no key, a revoked key and a key without the role are refused; overrides carry the
     key's name.
  9. The database refuses a second "current run".

    CUSTMATCH_DB_URL=... (default: a temporary SQLite file) python server/smoke_test.py
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import insert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from fastapi.testclient import TestClient  # noqa: E402

from custmatch import io  # noqa: E402
from custmatch_server import auth, db  # noqa: E402
from custmatch_server.api import create_app  # noqa: E402
from custmatch_server.loader import load_file  # noqa: E402
from custmatch_server.matcher import Matcher  # noqa: E402
from custmatch_server.schema import current_run  # noqa: E402
from custmatch_server.worker import run_batch  # noqa: E402
from test_support import fresh_engine  # noqa: E402

DEMO = ROOT / "examples/demo"


def split_demo(tmp):
    cfg = json.loads((DEMO / "config.json").read_text())
    rng = np.random.default_rng(0)
    held, base = {}, {}
    for s in cfg["sources"]:
        d = pd.read_csv(DEMO / s["path"], dtype=str, keep_default_na=False)
        hold = rng.random(len(d)) < 0.10
        d[~hold].to_csv(tmp / s["path"], index=False)
        held[s["name"]], base[s["name"]] = d[hold], d[~hold]
    (tmp / "config.json").write_text(json.dumps(cfg))
    keys = set(io.key(io.load(tmp / "config.json")[0]))
    lab = pd.read_csv(DEMO / "labels.csv", dtype=str, keep_default_na=False)
    lab[lab.a.isin(keys) & lab.b.isin(keys)].to_csv(tmp / "labels.csv", index=False)
    return cfg, held, base


def groups(assign):
    g = {}
    for k, c in assign.items():
        g.setdefault(c, set()).add(k)
    return {k: frozenset(g[c]) for k, c in assign.items()}


def main():
    out, failures = {}, []
    tmp = Path(tempfile.mkdtemp())
    url = os.environ.get("CUSTMATCH_DB_URL") or f"sqlite:///{tmp / 'smoke.db'}"
    try:
        eng = fresh_engine(url)                          # on Postgres: a clean slate on a reused database
        out["database"] = eng.dialect.name
        cfg, held, base = split_demo(tmp)
        log = lambda *_: None  # noqa: E731
        id_col = {s["name"]: s["columns"]["record_id"] for s in cfg["sources"]}

        # 1. load each source, then the first batch run
        first = {s["name"]: load_file(eng, cfg, s["name"], tmp / s["path"], mode="full") for s in cfg["sources"]}
        out["1_first_loads"] = {k: {"inserted": v["inserted"], "load": v["load_id"]} for k, v in first.items()}
        run1, s1 = run_batch(eng, tmp / "config.json", tmp / "labels.csv", tmp / "data", log=log)
        admin = auth.create_key(eng, "smoke-admin", "admin")
        client = TestClient(create_app(eng, require_keys=True), headers={"Authorization": f"Bearer {admin}"})
        cur = client.get("/runs/current").json()
        out["1_batch_run"] = {"run": run1, "records": s1["records"], "customers": s1["customers"],
                              "input_load_id": cur["input_load_id"]}
        if cur["input_load_id"] != max(v["load_id"] for v in first.values()):
            failures.append("the run did not record the newest load it saw")

        # 2. every record's lookup equals the batch output
        cl = pd.read_csv(tmp / "data/runs" / f"run-{run1:05d}" / "result/clusters.csv", dtype=str)
        bad = sum(client.get(f"/records/{s}/{r}").json()["customer_id"] != c
                  for s, r, c in zip(cl.source, cl.record_id, cl.customer_id))
        out["2_lookups"] = {"records": len(cl), "mismatches": int(bad)}
        if bad:
            failures.append("lookups differ from the batch run")
        one = cl.customer_id.value_counts().index[0]
        cust = client.get(f"/customers/{one}").json()
        if len(cust["records"]) != int(cl.customer_id.value_counts().iloc[0]) or not cust["golden_record"]:
            failures.append("customer lookup is incomplete")

        # 3. later loads: changed emails (incremental), three app records gone (full)
        web = base["web"].copy()
        changed_ids = list(web[id_col["web"]].head(5))
        web.loc[web[id_col["web"]].isin(changed_ids), "email"] = [f"changed{i}@mail.example" for i in range(5)]
        web.head(10).to_csv(tmp / "web_update.csv", index=False)          # 5 changed + 5 unchanged
        upd = load_file(eng, cfg, "web", tmp / "web_update.csv", mode="incremental")
        app = base["app"]
        gone_ids = list(app[id_col["app"]].head(3))
        app[~app[id_col["app"]].isin(gone_ids)].to_csv(tmp / "app_full.csv", index=False)
        full = load_file(eng, cfg, "app", tmp / "app_full.csv", mode="full")
        hist = client.get(f"/sources/web/records/{changed_ids[0]}").json()
        out["3_later_loads"] = {"web_incremental": {k: upd[k] for k in ("inserted", "updated", "unchanged", "deleted")},
                                "app_full": {k: full[k] for k in ("inserted", "updated", "unchanged", "deleted")},
                                "versions_of_a_changed_record": [v["change"] for v in hist["versions"]]}
        if (upd["updated"], upd["unchanged"], full["deleted"]) != (5, 5, 3) or \
                [v["change"] for v in hist["versions"]] != ["insert", "update"]:
            failures.append("load counts or version history wrong")

        # 4. the held-back records through POST /match
        lat = []
        for name, d in held.items():
            for row in d.to_dict("records"):
                body = {"source": name, "record_id": row[id_col[name]], "fields": row, "write": True}
                t0 = time.perf_counter()
                r = client.post("/match", json=body)
                lat.append(time.perf_counter() - t0)
                if r.status_code != 200:
                    failures.append(f"match failed: {r.status_code} {r.text[:200]}")
                    break
        out["4_realtime_writes"] = {"records": len(lat), "ms_p50": round(1000 * np.percentile(lat, 50), 1),
                                    "ms_p95": round(1000 * np.percentile(lat, 95), 1)}
        name, row = "web", held["web"].iloc[-1].to_dict()
        if not client.get(f"/records/web/{row[id_col['web']]}").json().get("provisional"):
            failures.append("a real-time record is not marked provisional")

        # 5. one brand-new person, twice at once
        src = next(s for s in cfg["sources"] if s["name"] == "web")["columns"]
        person = {src["first_name"]: "zephyrine", src["last_name"]: "quillfeather", src["email"]: "zq@nowhere.example"}
        got = {}

        def send(rid):
            got[rid] = client.post("/match", json={"source": "web", "record_id": rid, "fields": person,
                                                   "write": True}).json()["customer_id"]
        ts = [threading.Thread(target=send, args=(rid,)) for rid in ("smoke-a", "smoke-b")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        out["5_concurrent_new_person"] = "one customer" if got["smoke-a"] == got["smoke-b"] else "TWO customers"
        if got["smoke-a"] != got["smoke-b"]:
            failures.append("a new person arriving twice at once became two customers")

        # 6. a second writer is refused while the lease is held
        with eng.connect() as c:
            run = db.current(c)
        try:
            Matcher(eng, run).match("web", "smoke-c", person, write=True)
            out["6_second_writer"] = "ALLOWED"
            failures.append("a second writer was allowed")
        except PermissionError:
            out["6_second_writer"] = "refused"

        # 7. second batch run: changes in, deletions out, API records reconciled; a third retires run 1
        rt = pd.read_sql("select source, record_id, customer_id from realtime_records", eng)
        rt_view = dict(zip(zip(rt.source, rt.record_id), rt.customer_id.astype(str)))
        run2, s2 = run_batch(eng, tmp / "config.json", tmp / "labels.csv", tmp / "data", log=log)
        cl2 = pd.read_csv(tmp / "data/runs" / f"run-{run2:05d}" / "result/clusters.csv", dtype=str)
        batch2 = dict(zip(zip(cl2.source, cl2.record_id), cl2.customer_id))
        before = dict(zip(zip(cl.source, cl.record_id), cl.customer_id)) | rt_view
        before = {k: v for k, v in before.items() if k in batch2}         # deleted records left the file
        g_rt, g_b = groups(before), groups(batch2)
        same = sum(g_rt[k] == g_b[k] for k in g_rt)
        deleted_in_run = sum(("app", r) in batch2 for r in gone_ids)
        out["7_second_batch_run"] = {"run": run2, "records": s2["records"], "records_by_source": s2["records_by_source"],
                                     "deleted_records_in_the_run": deleted_in_run,
                                     "same_group_as_before": f"{same} of {len(g_rt)} ({100 * same / len(g_rt):.1f}%)"}
        if deleted_in_run:
            failures.append("records deleted by a full load were still matched")
        if client.get("/records/web/" + row[id_col["web"]]).json().get("provisional"):
            failures.append("a reconciled record is still provisional")
        email = client.get(f"/sources/web/records/{changed_ids[0]}").json()["fields"]["email"]
        if email != "changed0@mail.example":
            failures.append("the changed email did not reach the stored record")
        time.sleep(1.1)                                  # the API re-reads the current run each second
        run3, _ = run_batch(eng, tmp / "config.json", tmp / "labels.csv", tmp / "data", log=log)
        statuses = dict(pd.read_sql("select id, status from runs order by id", eng).values)
        left = pd.read_sql(f"select count(*) as n from record_customer where run_id = {run1}", eng).n[0]
        out["7_after_third_run"] = {"run_statuses": statuses, "rows_left_for_run_1": int(left)}
        hist3 = pd.read_csv(tmp / "data/runs" / f"run-{run3:05d}" / "result/history.csv", dtype=str)
        n_events = int(pd.read_sql("select count(*) as n from customer_events", eng).n[0])
        out["7_events"] = {"history_rows": len(hist3), "customer_events_rows": n_events}
        if n_events != len(hist3):
            failures.append("customer_events does not match the run history (duplicated or missing events)")
        if statuses.get(run1) != "retired" or left:
            failures.append("the oldest run was not retired")
        if eng.dialect.name == "postgresql":
            parts = pd.read_sql("select tablename from pg_tables where tablename like 'record_customer_r%%'", eng)
            out["7_partitions"] = sorted(parts.tablename)
            if f"record_customer_r{run1}" in set(parts.tablename):
                failures.append("the retired run's partition is still there")

        # 8. API keys: none refused, a read key cannot match, a revoked key refused, /health open,
        #    an override's author is the key's name
        bare = TestClient(client.app)
        reader = auth.create_key(eng, "smoke-reader", "read")
        gone = auth.create_key(eng, "smoke-gone", "steward")
        auth.revoke(eng, "smoke-gone")
        rid = cl.record_id[0]
        rec_url = f"/records/{cl.source[0]}/{rid}"
        as_reader = {"Authorization": f"Bearer {reader}"}
        codes = {"no key": bare.get(rec_url).status_code,
                 "read key, lookup": bare.get(rec_url, headers=as_reader).status_code,
                 "read key, match": bare.post("/match", headers=as_reader, json={
                     "source": "web", "record_id": "x", "fields": {}, "write": False}).status_code,
                 "revoked key": bare.get(rec_url, headers={"Authorization": f"Bearer {gone}"}).status_code,
                 "health, no key": bare.get("/health").status_code}
        out["8_api_keys"] = codes
        if codes != {"no key": 401, "read key, lookup": 200, "read key, match": 403, "revoked key": 401,
                     "health, no key": 200}:
            failures.append("API key checks wrong")
        k = [f"{s}:{r}" for s, r in zip(cl.source, cl.record_id)]
        client.post("/overrides", json={"a": k[0], "b": k[1], "decision": "different", "created_by": "someone else"})
        by = {(o["a"], o["b"]): o["created_by"] for o in client.get("/overrides").json()}
        if by.get(tuple(sorted(k[:2]))) != "smoke-admin":
            failures.append("an override's author was not the key's name")

        # 8b. the steward's endpoints: the review queue (a decision takes its pair off), search, the page
        rv = client.get("/review?limit=1").json()
        first_pair = rv["pairs"][0] if rv["pairs"] else None
        if first_pair:
            client.post("/overrides", json={"a": first_pair["a"], "b": first_pair["b"], "decision": "different"})
        rv2 = client.get("/review?limit=1").json()
        found = client.get("/search", params={"q": cl.record_id[0]}).json()
        page = bare.get("/dashboard")
        # the dashboard's views: a pair's waterfall adds up to its score; customers to inspect,
        # a customer's graph, links by score band, the model and data
        k0, k1 = cl.groupby("customer_id").filter(lambda g: len(g) >= 2).iloc[:2][["source", "record_id"]].agg(":".join, axis=1)
        ex = client.get("/explain", params={"a": k0, "b": k1}).json()
        adds_up = abs(ex["base"] + sum(c["contribution"] for c in ex["contributions"])
                      + (ex.get("adjustment") or {}).get("contribution", 0) - ex["score"]) < 0.01
        big = client.get("/customers?order=size&limit=1").json()["customers"][0]
        gr = client.get(f"/customers/{big['customer_id']}/graph").json()
        lk = client.get("/links?min=0.6&max=1&limit=3").json()
        q = client.get("/quality").json()
        out["8c_dashboard_views"] = {"waterfall_adds_up": adds_up, "graph_nodes": len(gr["nodes"]),
                                     "links_in_band": lk["total"], "threshold_sweep_points": len(q["threshold_sweep"] or {}),
                                     "features_named": len(q["feature_importance"])}
        rl = client.get("/runs").json()
        by_key = q["blocking"].get("pairs_by_key") or []
        out["8c_dashboard_views"] |= {"runs_listed": len(rl), "blocking_keys": len(by_key),
                                      "unlinkable_pct": (q["unlinkables"] or {}).get("share_at_threshold")}
        if not rl or not rl[0]["summary"].get("feature_importance") or q["unlinkables"] is None or \
                not by_key or by_key[-1]["cumulative"] != q["blocking"]["candidate_pairs"]:
            failures.append("runs, unlinkables or blocking-by-key wrong")
        if not adds_up or len(gr["nodes"]) != big["records"] or not lk["links"] or not q["threshold_sweep"]:
            failures.append("dashboard views wrong")
        out["8b_steward"] = {"review_pairs": rv["remaining"], "after_a_decision": rv2["remaining"],
                             "search_hits": len(found), "dashboard": page.status_code}
        if not first_pair or not first_pair["a_fields"] or rv2["remaining"] != rv["remaining"] - 1 or \
                not any(h["key"] == f"{cl.source[0]}:{cl.record_id[0]}" for h in found) or page.status_code != 200:
            failures.append("review queue, search or dashboard wrong")

        # 9. one current run, enforced by the database
        try:
            with eng.begin() as c:
                c.execute(insert(current_run).values(id=2, run_id=run3))
            out["9_second_current_run"] = "ALLOWED"
            failures.append("the database allowed a second current run")
        except Exception as e:
            out["9_second_current_run"] = f"refused ({type(e).__name__})"
        out["loads"] = [{k: r[k] for k in ("id", "source", "mode", "inserted", "updated", "deleted", "unchanged")}
                        | {"origin": Path(str(r["origin"])).name} for r in client.get("/loads").json()]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out["result"] = "PASS" if not failures else f"FAIL: {failures}"
    print(json.dumps(out, indent=1, default=str))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
