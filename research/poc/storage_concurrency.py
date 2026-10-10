"""PoC: storage and concurrency gotchas for a custmatch service (docs/report/12-the-service.md, "The service").

  1. DuckDB database file: can a second process open it while one writes?
  2. Parquet runs on object storage: does a reader see a half-published run, (a) reading a run's
     files by glob, (b) reading through a manifest swapped in one write after the run is complete?
  3. Iceberg (pyiceberg, SQLite catalog): two writers committing at once.
  4. Erasure: after deleting a person's rows from an Iceberg table, is the data still readable?

Local files stand in for S3: one object PUT is atomic on both, which is all the manifest pattern
relies on. (S3 has no atomic rename of a directory, so "write to a temp dir and rename" is not used.)

    python research/poc/storage_concurrency.py
"""
import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RESULTS = {}


# ------------------------------------------------------------------ 1. DuckDB file locking
def _hold_writer(path, ready, done):
    con = duckdb.connect(str(path))
    con.execute("create table if not exists t as select 1 as x")
    ready.set()
    done.wait(10)
    con.close()


def duckdb_file_locking(tmp):
    path = tmp / "service.duckdb"
    ready, done = mp.Event(), mp.Event()
    p = mp.Process(target=_hold_writer, args=(path, ready, done))
    p.start()
    ready.wait(10)
    out = {}
    for mode in ("read_write", "read_only"):
        try:
            con = duckdb.connect(str(path), read_only=(mode == "read_only"))
            con.execute("select count(*) from t").fetchall()
            con.close()
            out[mode] = "opened"
        except Exception as e:                            # duckdb.IOException: the file is locked
            out[mode] = f"refused: {type(e).__name__}: {str(e)[:90]}"
    done.set()
    p.join()
    # the alternative: no database file at all, every process queries the Parquet files directly
    pq.write_table(pa.table({"x": [1, 2, 3]}), tmp / "shared.parquet")
    n = [duckdb.connect().execute(f"select count(*) from read_parquet('{tmp / 'shared.parquet'}')").fetchone()[0]
         for _ in range(3)]
    out["in-memory DuckDB per process over shared Parquet"] = f"{len(n)} readers, all ok"
    RESULTS["1_duckdb_file_locking"] = out


# ------------------------------------------------------------------ 2. publishing runs
def _write_run(run_dir, n_files, rows, slow):
    run_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_files):
        pq.write_table(pa.table({"customer_id": np.arange(rows) + k * rows}), run_dir / f"part-{k}.parquet")
        time.sleep(slow)


def _writer(base, n_runs, n_files, rows, use_manifest):
    for r in range(n_runs):
        run = base / f"run-{r:03d}"
        _write_run(run, n_files, rows, 0.02)
        if use_manifest:                                  # publish: one small object written at once
            tmpf = base / f".manifest-{r}.json"
            tmpf.write_text(json.dumps({"run": run.name, "files": sorted(p.name for p in run.glob("*.parquet")),
                                        "rows": n_files * rows}))
            os.replace(tmpf, base / "manifest.json")      # on S3: a single PUT of manifest.json


def _reader(base, seconds, use_manifest, q):
    seen, partial, t0 = 0, 0, time.time()
    con = duckdb.connect()
    while time.time() - t0 < seconds:
        try:
            if use_manifest:
                m = json.loads((base / "manifest.json").read_text())
                files = [str(base / m["run"] / f) for f in m["files"]]
                n = con.execute(f"select count(*) from read_parquet({files})").fetchone()[0]
                partial += n != m["rows"]
            else:
                runs = sorted(base.glob("run-*"))
                if not runs:
                    continue
                n = con.execute(f"select count(*) from read_parquet('{runs[-1]}/*.parquet')").fetchone()[0]
                partial += n != 10 * 1000                 # a complete run has 10 files of 1,000 rows
            seen += 1
        except Exception:
            partial += 1
    q.put((seen, partial))


def publishing(tmp):
    out = {}
    for use_manifest in (False, True):
        base = tmp / ("manifest" if use_manifest else "glob")
        base.mkdir()
        if use_manifest:
            _write_run(base / "run-init", 10, 1000, 0)
            (base / "manifest.json").write_text(json.dumps({"run": "run-init", "files": [f"part-{k}.parquet" for k in range(10)], "rows": 10000}))
        q = mp.Queue()
        w = mp.Process(target=_writer, args=(base, 8, 10, 1000, use_manifest))
        rd = mp.Process(target=_reader, args=(base, 3.0, use_manifest, q))
        rd.start(); w.start(); w.join(); rd.join()
        seen, partial = q.get()
        out["manifest" if use_manifest else "glob of the newest run"] = f"{seen} reads, {partial} saw a partial run"
    RESULTS["2_publishing_runs"] = out


# ------------------------------------------------------------------ 3 and 4. Iceberg
def _iceberg_catalog(tmp):
    from pyiceberg.catalog.sql import SqlCatalog
    wh = tmp / "warehouse"
    wh.mkdir(exist_ok=True)
    return SqlCatalog("custmatch", uri=f"sqlite:///{tmp / 'catalog.db'}", warehouse=f"file://{wh}")


def _iceberg_append(tmp, k, q):
    cat = _iceberg_catalog(tmp)
    t = cat.load_table("cm.customers")
    rows = pa.table({"record_id": [f"w{k}-{i}" for i in range(1000)], "customer_id": list(range(1000)),
                     "email": [f"p{k}{i}@mail.com" for i in range(1000)]})
    tries = 0
    while True:
        tries += 1
        try:
            t.append(rows)
            q.put((k, "ok", tries))
            return
        except Exception as e:                            # CommitFailedException: another writer won
            if tries >= 5:
                q.put((k, f"gave up: {type(e).__name__}", tries))
                return
            t = cat.load_table("cm.customers")            # reload the latest metadata and retry


def iceberg(tmp):
    cat = _iceberg_catalog(tmp)
    cat.create_namespace("cm")
    schema = pa.schema([("record_id", pa.string()), ("customer_id", pa.int64()), ("email", pa.string())])
    t = cat.create_table("cm.customers", schema=schema)
    t.append(pa.table({"record_id": ["a1", "a2"], "customer_id": [1, 1], "email": ["jane@mail.com", "jane@mail.com"]}))
    q = mp.Queue()
    ps = [mp.Process(target=_iceberg_append, args=(tmp, k, q)) for k in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    outcomes = [q.get() for _ in ps]
    t = cat.load_table("cm.customers")
    RESULTS["3_iceberg_concurrent_appends"] = {
        "writers": [f"writer {k}: {s} after {n} attempt(s)" for k, s, n in sorted(outcomes)],
        "rows_after": t.scan().to_arrow().num_rows, "expected_rows": 2 + 4 * 1000}
    # 4. erasure: delete Jane, then look at the table now and at the snapshot before the delete
    before = t.current_snapshot().snapshot_id
    from pyiceberg.expressions import EqualTo
    t.delete(EqualTo("email", "jane@mail.com"))
    t = cat.load_table("cm.customers")
    now = t.scan(row_filter=EqualTo("email", "jane@mail.com")).to_arrow().num_rows
    old = t.scan(row_filter=EqualTo("email", "jane@mail.com"), snapshot_id=before).to_arrow().num_rows
    out = {"jane_rows_in_current_table": now, "jane_rows_via_time_travel_to_before_the_delete": old,
           "data_files_on_disk": len(list((tmp / "warehouse").rglob("*.parquet")))}
    try:                                                  # expiring old snapshots is what really erases
        expire = t.maintenance.expire_snapshots()
        expire.older_than(pd.Timestamp.now("UTC").to_pydatetime()).commit()
        t = cat.load_table("cm.customers")
        try:
            out["after_expiring_old_snapshots"] = t.scan(row_filter=EqualTo("email", "jane@mail.com"),
                                                         snapshot_id=before).to_arrow().num_rows
        except Exception as e:
            out["after_expiring_old_snapshots"] = f"old snapshot gone ({type(e).__name__})"
        out["data_files_on_disk_after_expiry"] = len(list((tmp / "warehouse").rglob("*.parquet")))
    except Exception as e:
        out["expire_snapshots"] = f"not available in pyiceberg {__import__('pyiceberg').__version__}: {type(e).__name__}: {str(e)[:80]}"
    RESULTS["4_iceberg_erasure"] = out


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        for step in (duckdb_file_locking, publishing, iceberg):
            sub = tmp / step.__name__
            sub.mkdir()
            try:
                step(sub)
            except Exception as e:
                RESULTS[step.__name__] = f"FAILED: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(RESULTS, indent=1, default=str))


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
