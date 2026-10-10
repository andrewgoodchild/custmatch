"""Scale test of the service: the synthetic customer file copied K times (research/scripts/custmatch_scale.py:
copy-specific names and contacts, so copies never match each other), through every step, timed.

  1. full loads of both sources (300 records held back to arrive through the API)
  2. the first batch run - its input read from the database, published - and its clustered F1
  3. the real-time matcher's start (index build) and memory
  4. 300 records through POST /match (write); 1,000 record lookups
  5. an incremental load changing 1% of CRM records, and a full billing load missing 0.5%
  6. a second and a third batch run (the third retires the first); database size

    CUSTMATCH_DB_URL=postgresql+psycopg://... python server/scale_test.py --copies 5
Writes research/results/server_scale.json (one entry per size).
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "research/scripts"))
import custmatch_scale as CS  # noqa: E402
import custmatch_validate as V  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from custmatch import labels  # noqa: E402
from custmatch.core import partition_scores  # noqa: E402
from custmatch_server import db  # noqa: E402
from custmatch_server.api import create_app  # noqa: E402
from custmatch_server.loader import load_file  # noqa: E402
from custmatch_server.matcher import Matcher  # noqa: E402
from custmatch_server.worker import run_batch  # noqa: E402
from test_support import fresh_engine  # noqa: E402

OUT = ROOT / "research/results/server_scale.json"
HELD = 300


class Peak:
    """The process's peak resident memory while a block runs."""

    def __init__(self):
        self.proc = psutil.Process()

    def __enter__(self):
        self.peak, self.stop = self.proc.memory_info().rss, threading.Event()
        self.base = self.peak

        def run():
            while not self.stop.is_set():
                self.peak = max(self.peak, self.proc.memory_info().rss)
                time.sleep(0.05)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        self.t0 = time.time()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.t.join()
        self.seconds = round(time.time() - self.t0, 1)
        self.gb = round(self.peak / 1024 ** 3, 2)


def truth_for(copies):
    gold = {}
    t = V.truth()
    for c in range(copies):
        for key, (s, _) in t.items():
            src, rid = key.split(":", 1)
            gold[f"{src}:{rid}-x{c}"] = f"{s}-x{c}"
    return gold


def clustered_f1(result_dir, gold):
    c = pd.read_csv(Path(result_dir) / "clusters.csv", dtype=str)
    keys = c.source + ":" + c.record_id
    pid = np.array([gold.get(k, k) for k in keys])
    _, n = np.unique(pid, return_counts=True)
    return partition_scores(c.customer_id.to_numpy(), pid, float(np.sum(n * (n - 1) / 2)))["f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=5)
    a = ap.parse_args()
    url = os.environ["CUSTMATCH_DB_URL"]
    eng = fresh_engine(url)
    tmp = Path(tempfile.mkdtemp())
    res = {"copies": a.copies}
    log = lambda *_: None  # noqa: E731

    cfg_path = CS.scaled(a.copies, tmp)
    cfg = json.loads(Path(cfg_path).read_text())
    for s in cfg["sources"]:
        s["path"] = str((Path(cfg_path).parent / s["path"]).resolve())
    gold = truth_for(a.copies)
    # hold back records to arrive through the API
    crm = next(s for s in cfg["sources"] if s["name"] == "crm")
    d = pd.read_parquet(crm["path"]).fillna("").astype(str)
    rng = np.random.default_rng(0)
    hold = np.zeros(len(d), bool)
    hold[rng.choice(len(d), HELD, replace=False)] = True
    held = d[hold]
    d[~hold].to_parquet(tmp / "crm_base.parquet", index=False)
    crm["path"] = str(tmp / "crm_base.parquet")
    (tmp / "config.json").write_text(json.dumps(cfg))
    labels.sample(tmp / "config.json", 1000, tmp / "labels.csv", seed=1)
    lab = pd.read_csv(tmp / "labels.csv", dtype=str, keep_default_na=False)
    lab["label"] = [int(gold[x] == gold[y]) for x, y in zip(lab.a, lab.b)]
    lab.to_csv(tmp / "labels.csv", index=False)

    # 1. full loads
    with Peak() as p:
        loads = {s["name"]: load_file(eng, cfg, s["name"], s["path"], mode="full") for s in cfg["sources"]}
    res["1_full_loads"] = {"records": sum(v["inserted"] for v in loads.values()), "seconds": p.seconds, "peak_gb": p.gb}
    print("loads", res["1_full_loads"], flush=True)

    # 2. first batch run
    with Peak() as p:
        run1, s1 = run_batch(eng, tmp / "config.json", tmp / "labels.csv", tmp / "data", log=log)
    with eng.connect() as c:
        run = db.current(c)
    res["2_batch_run_1"] = {"records": s1["records"], "customers": s1["customers"], "seconds": p.seconds,
                            "peak_gb": p.gb, "clustered_f1": clustered_f1(run["artifact_uri"], gold)}
    print("batch 1", res["2_batch_run_1"], flush=True)

    # 3. the matcher's start
    with Peak() as p:
        m = Matcher(eng, run)
    res["3_matcher_start"] = {"seconds": p.seconds, "peak_gb": p.gb,
                              "added_gb": round((p.peak - p.base) / 1024 ** 3, 2), "index_keys": len(m.index)}
    print("matcher", res["3_matcher_start"], flush=True)
    del m

    # 4. real-time writes and lookups through the API
    from custmatch_server import auth
    key = auth.create_key(eng, "scale-test", "admin")
    client = TestClient(create_app(eng, require_keys=True), headers={"Authorization": f"Bearer {key}"})
    t0 = time.time()
    client.post("/match", json={"source": "crm", "record_id": "warmup", "fields": {}, "write": False})
    res["4_api_first_call_seconds"] = round(time.time() - t0, 1)      # loads the matcher
    lat = []
    for row in held.to_dict("records"):
        t0 = time.perf_counter()
        r = client.post("/match", json={"source": "crm", "record_id": row["record_id"], "fields": row, "write": True})
        lat.append(time.perf_counter() - t0)
        assert r.status_code == 200, r.text[:300]
    keys = pd.read_sql(f"select source, record_id from record_customer where run_id = {run1} order by random() limit 1000", eng)
    lk = []
    for s, rid in zip(keys.source, keys.record_id):
        t0 = time.perf_counter()
        assert client.get(f"/records/{s}/{rid}").status_code == 200
        lk.append(time.perf_counter() - t0)
    res["4_realtime"] = {"writes": len(lat), "ms_p50": round(1000 * np.percentile(lat, 50), 1),
                         "ms_p95": round(1000 * np.percentile(lat, 95), 1),
                         "lookup_ms_p50": round(1000 * np.percentile(lk, 50), 1),
                         "lookup_ms_p95": round(1000 * np.percentile(lk, 95), 1)}
    print("realtime", res["4_realtime"], flush=True)

    # 5. later loads: 1% of CRM changed (incremental), billing missing 0.5% (full)
    base = pd.read_parquet(tmp / "crm_base.parquet").fillna("").astype(str)
    ch = base.sample(frac=0.01, random_state=1).copy()
    ch["email"] = [f"changed{i}@mail.example" for i in range(len(ch))]
    ch.to_parquet(tmp / "crm_changes.parquet", index=False)
    with Peak() as p:
        inc = load_file(eng, cfg, "crm", tmp / "crm_changes.parquet", mode="incremental")
    bill_src = next(s for s in cfg["sources"] if s["name"] == "billing")
    bill = pd.read_parquet(bill_src["path"]).fillna("").astype(str)
    bill.sample(frac=0.995, random_state=1).to_parquet(tmp / "billing_full.parquet", index=False)
    with Peak() as p2:
        full = load_file(eng, cfg, "billing", tmp / "billing_full.parquet", mode="full")
    res["5_later_loads"] = {"incremental_updated": inc["updated"], "incremental_seconds": p.seconds,
                            "full_deleted": full["deleted"], "full_unchanged": full["unchanged"],
                            "full_seconds": p2.seconds, "peak_gb": max(p.gb, p2.gb)}
    ch10 = base.sample(frac=0.10, random_state=2).copy()                # a bigger re-delivery: 10% changed
    ch10["email"] = [f"again{i}@mail.example" for i in range(len(ch10))]
    ch10.to_parquet(tmp / "crm_changes10.parquet", index=False)
    with Peak() as p3:
        inc10 = load_file(eng, cfg, "crm", tmp / "crm_changes10.parquet", mode="incremental")
    res["5_later_loads"].update(incremental_10pct_updated=inc10["updated"], incremental_10pct_seconds=p3.seconds)
    print("later loads", res["5_later_loads"], flush=True)

    # 6. second and third batch runs
    with Peak() as p:
        run2, s2 = run_batch(eng, tmp / "config.json", tmp / "labels.csv", tmp / "data", log=log)
    res["6_batch_run_2"] = {"records": s2["records"], "seconds": p.seconds, "peak_gb": p.gb,
                            "labels_dropped": s2["labels_dropped_for_deleted_records"]}
    with Peak() as p:
        run3, _ = run_batch(eng, tmp / "config.json", tmp / "labels.csv", tmp / "data", log=log)
    sizes = pd.read_sql("""select relname as t, pg_total_relation_size(c.oid) as b from pg_class c
                           join pg_namespace n on n.oid = c.relnamespace
                           where n.nspname = 'public' and c.relkind in ('r', 'p') order by b desc""", eng)
    res["6_batch_run_3"] = {"seconds": p.seconds, "peak_gb": p.gb,
                            "statuses": dict(pd.read_sql("select id, status from runs", eng).values)}
    total = int(pd.read_sql("select pg_database_size(current_database()) as b", eng).b[0])
    res["6_database"] = {"total_mb": round(total / 1024 ** 2),
                         "largest_tables_mb": {t: round(b / 1024 ** 2) for t, b in sizes.head(6).values}}
    print("batch 2/3", res["6_batch_run_2"], res["6_batch_run_3"], res["6_database"], flush=True)

    all_res = json.loads(OUT.read_text()) if OUT.exists() else {}
    all_res[f"copies_{a.copies}"] = res
    OUT.write_text(json.dumps(all_res, indent=1, default=str))


if __name__ == "__main__":
    main()
