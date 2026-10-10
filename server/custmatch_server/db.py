"""The database: the engine, migrations, publishing a run, retiring old runs, and the matcher's lease.
The tables are in schema.py and are created by the migrations (Alembic), never by create_all.

Publishing a run is one transaction under a lock: readers see the old run or the new one, never a
mix, and two batch runs cannot publish at once. On Postgres it fills the run's own partitions with
COPY; retiring a run drops them.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, delete, insert, select, text, update

from .schema import (PARTITIONED, current_run, customer_events, customers, golden_pins, leases, links, loads,
                     record_customer, runs)

KEEP_RUNS = 2          # the current run and the one before it, for rollback
GOLDEN = ["first_name", "middle_initial", "last_name", "date_of_birth", "email", "phone",
          "addr_number", "addr_street", "addr_unit"]


def now():
    return datetime.now(timezone.utc)


def as_json(v):
    """A JSON column's value: Postgres returns the object, SQLite the string."""
    return json.loads(v) if isinstance(v, str) else v


def as_list(v):
    """A config value that may be one path or a list of them, as a list."""
    return [v] if isinstance(v, str) else list(v or [])


def holder_name():
    """A lease holder's name: one per process (host, pid and a random suffix)."""
    import socket
    import uuid
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def engine_from_env(url=None, migrate=True):
    url = url or os.environ.get("CUSTMATCH_DB_URL", "sqlite:///custmatch.db")
    kw = {"connect_args": {"timeout": 30}} if url.startswith("sqlite") else {}
    eng = create_engine(url, **kw)
    if eng.dialect.name == "sqlite":
        with eng.begin() as c:
            c.exec_driver_sql("PRAGMA journal_mode=WAL")       # readers do not block the writer
        from sqlalchemy import event

        @event.listens_for(eng, "connect")
        def _fk(dbapi, _):                                     # SQLite enforces foreign keys only when asked
            dbapi.execute("PRAGMA foreign_keys=ON")
    if migrate:
        upgrade(eng)
    return eng


def upgrade(eng):
    """Bring the schema to the latest migration. Processes starting together (the API and the
    scheduler) would both create the tables and one would fail: on Postgres the migration runs
    under an advisory lock, so the second waits and then finds nothing to do; and anywhere, an
    error from a migration that another process completed meanwhile is not an error."""
    from alembic import command
    from alembic.config import Config
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory
    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).parent / "migrations"))
    head = ScriptDirectory.from_config(cfg).get_current_head()
    try:
        with eng.begin() as c:
            if c.dialect.name == "postgresql":
                c.execute(text("SELECT pg_advisory_xact_lock(424245)"))
            cfg.attributes["connection"] = c
            command.upgrade(cfg, "head")
    except Exception:
        with eng.connect() as c:
            if MigrationContext.configure(c).get_current_revision() != head:
                raise


def lock(conn, key):
    """A transaction-scoped lock: an advisory lock on Postgres; on SQLite a write as the
    transaction's first statement takes the database's single write lock."""
    if conn.dialect.name == "postgresql":
        conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": key})
    else:
        conn.execute(text("UPDATE leases SET holder = holder WHERE 0"))


def current(conn):
    """The current run's row, or None."""
    return conn.execute(select(runs).join(current_run, current_run.c.run_id == runs.c.id)).mappings().first()


def latest_load_id(conn):
    return conn.execute(select(loads.c.id).where(loads.c.status == "done").order_by(loads.c.id.desc())).scalar()


def new_run(eng):
    with eng.begin() as c:
        return c.execute(insert(runs).values(status="running", created_at=now())).inserted_primary_key[0]


def _copy_values(col):
    """A column as COPY wants it, converted once per column: JSON objects as text, missing as None."""
    vals = col.to_numpy(dtype=object)
    if col.dtype == object and any(isinstance(v, (dict, list)) for v in vals):
        return [json.dumps(v) if isinstance(v, (dict, list)) else v for v in vals]
    missing = pd.isna(col).to_numpy()
    if missing.any():
        vals = vals.copy()
        vals[missing] = None
    return vals.tolist()


def _bulk(conn, table, frame):
    """Insert a DataFrame: COPY on Postgres (millions of rows a minute), executemany elsewhere."""
    if not len(frame):
        return
    if conn.dialect.name == "postgresql":
        cols = list(frame.columns)
        raw = conn.connection.driver_connection
        columns = [_copy_values(frame[c]) for c in cols]
        with raw.cursor().copy(f"COPY {table.name} ({', '.join(cols)}) FROM STDIN") as cp:
            for row in zip(*columns):
                cp.write_row(row)
    else:
        rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
        for k in range(0, len(rows), 5000):
            conn.execute(insert(table), rows[k:k + 5000])


def _int(s):
    return pd.to_numeric(s.replace("", None), errors="coerce").astype("Int64")


def publish(eng, run_id, result_dir, summary, input_load_id, reconcile=None):
    """Load a finished custmatch run's outputs and make it current, in one transaction.
    reconcile: {table: the highest seq the run saw} - the real-time rows the run included are
    marked reconciled in the same transaction (rows written while it ran stay pending)."""
    out = Path(result_dir)
    csv = lambda path: pd.read_csv(path, dtype=str, keep_default_na=False)  # noqa: E731
    cl, gold, lk = csv(out / "clusters.csv"), csv(out / "golden_records.csv"), csv(out / "links.csv")
    hist = csv(out / "history.csv") if (out / "history.csv").exists() \
        else pd.DataFrame(columns=["time", "event", "customer_id", "other_customer_ids", "records", "detail"])
    with eng.begin() as c:
        lock(c, 424242)
        prev = current(c)
        # history.csv is cumulative (the previous run's history plus this run's events): only the
        # rows this run added are its events
        if prev is not None and (Path(prev["artifact_uri"]) / "history.csv").exists():
            n_before = len(csv(Path(prev["artifact_uri"]) / "history.csv"))
            hist = hist.iloc[n_before:]
        created = {}
        if prev is not None:                                   # a customer carried over keeps its first run
            created = dict(c.execute(select(customers.c.customer_id, customers.c.created_run)
                                     .where(customers.c.run_id == prev["id"])).all())
        if c.dialect.name == "postgresql":
            for t in PARTITIONED:
                c.execute(text(f"CREATE TABLE {t}_r{run_id} PARTITION OF {t} FOR VALUES IN ({run_id})"))
        # customers: golden record, size, household, first run
        cl["cid"] = _int(cl.customer_id)
        size = cl.groupby("cid").size()
        hh = cl.assign(h=_int(cl.household_id)).groupby("cid").h.first()
        g = gold.assign(cid=_int(gold.customer_id))
        extra_cols = [x for x in gold.columns if x not in GOLDEN + ["customer_id"]]
        cust = pd.DataFrame({"run_id": run_id, "customer_id": g.cid,
                             "household_id": g.cid.map(hh), "record_count": g.cid.map(size).fillna(0).astype(int),
                             "created_run": [int(created.get(int(x), run_id)) for x in g.cid]})
        for f in GOLDEN:
            cust[f] = g[f] if f in g.columns else None
        cust["golden_extra"] = [{k: r[k] for k in extra_cols} if extra_cols else None
                                for r in g.to_dict("records")]
        # stewards' pinned values beat survivorship: each follows its record into the record's customer
        pins = c.execute(select(golden_pins.c.record_key, golden_pins.c.field, golden_pins.c.value)
                         .order_by(golden_pins.c.created_at)).all()
        if pins:
            cid_of = dict(zip(cl.source + ":" + cl.record_id, cl.cid))
            row_of = {int(x): k for k, x in enumerate(cust.customer_id)}
            for key, field, value in pins:
                k = row_of.get(int(cid_of[key])) if key in cid_of else None
                if k is None:
                    continue
                if field in GOLDEN:
                    cust.loc[k, field] = value
                else:
                    cust.at[k, "golden_extra"] = dict(cust.at[k, "golden_extra"] or {}, **{field: value})
        _bulk(c, customers, cust)
        _bulk(c, record_customer, pd.DataFrame({
            "run_id": run_id, "source": cl.source, "record_id": cl.record_id, "customer_id": cl.cid,
            "household_id": _int(cl.household_id),
            "confidence": pd.to_numeric(cl.get("confidence", pd.Series([""] * len(cl))).replace("", None), errors="coerce")}))
        a, b = lk.a.str.split(":", n=1, expand=True), lk.b.str.split(":", n=1, expand=True)
        _bulk(c, links, pd.DataFrame({
            "run_id": run_id, "source_a": a[0], "record_a": a[1], "source_b": b[0], "record_b": b[1],
            "customer_id": _int(lk.customer_id), "score": pd.to_numeric(lk.score, errors="coerce"),
            "is_bridge": lk.get("is_bridge", pd.Series(["False"] * len(lk))).eq("True").astype(int),
            "evidence": lk.get("evidence", pd.Series([""] * len(lk)))}).drop_duplicates(
            ["source_a", "record_a", "source_b", "record_b"]))
        _bulk(c, customer_events, pd.DataFrame({
            "run_id": run_id, "at": pd.to_datetime(hist.time, utc=True, errors="coerce"), "event": hist.event,
            "customer_id": _int(hist.customer_id), "other_customer_ids": hist.other_customer_ids,
            "record_count": _int(hist.records), "detail": hist.detail}))
        c.execute(update(runs).where(runs.c.id == run_id).values(
            status="published", published_at=now(), summary=summary, input_load_id=input_load_id,
            artifact_uri=str(Path(result_dir).resolve())))
        if prev is None:
            c.execute(insert(current_run).values(id=1, run_id=run_id))
        else:
            c.execute(update(current_run).where(current_run.c.id == 1).values(run_id=run_id))
        for table, upto in (reconcile or {}).items():
            if upto is not None:
                c.execute(update(table).where(table.c.reconciled_run.is_(None)).where(table.c.seq <= upto)
                          .values(reconciled_run=run_id))
        retired = _retire_old(c)
    return len(cl), retired


def _retire_old(c):
    """Runs beyond the newest KEEP_RUNS published ones: their partitions dropped (Postgres) or rows
    deleted (SQLite), status 'retired'. The run row stays, as the catalog's record."""
    keep = [r for (r,) in c.execute(select(runs.c.id).where(runs.c.status == "published")
                                     .order_by(runs.c.id.desc()).limit(KEEP_RUNS))]
    old = [r for (r,) in c.execute(select(runs.c.id).where(runs.c.status == "published")
                                    .where(runs.c.id.not_in(keep)))]
    for r in old:
        if c.dialect.name == "postgresql":
            # detach, then drop, children before parents: a foreign key between partitioned tables
            # depends on every partition, so a referenced partition cannot be dropped in place
            for t in ("links", "record_customer", "customers"):
                c.execute(text(f"ALTER TABLE {t} DETACH PARTITION {t}_r{r}"))
                c.execute(text(f"DROP TABLE {t}_r{r}"))
        else:
            for t in (links, record_customer, customers):
                c.execute(delete(t).where(t.c.run_id == r))
        c.execute(update(runs).where(runs.c.id == r).values(status="retired"))
    return old


def fail_run(eng, run_id, message):
    with eng.begin() as c:
        c.execute(update(runs).where(runs.c.id == run_id).values(status="failed", error=message))


# ------------------------------------------------------------------ leases (the matcher's, the scheduler's)
def release_lease(eng, name, holder):
    """Give a lease up at once (on shutdown), if `holder` holds it - so a successor need not wait
    for it to expire."""
    with eng.begin() as c:
        c.execute(update(leases).where(leases.c.name == name).where(leases.c.holder == holder)
                  .values(expires_at=now() - timedelta(seconds=1)))


def take_lease(eng, name, holder, seconds=60, report=False):
    """True if `holder` now holds the lease: it was free, expired, or already this holder's.
    report=True returns (taken, held_throughout): held_throughout is False when the lease was
    free, expired or someone else's - another process may have written meanwhile."""
    t = now()
    with eng.begin() as c:
        lock(c, 424243)
        row = c.execute(select(leases).where(leases.c.name == name)).mappings().first()
        expires = row["expires_at"] if row else None
        if expires is not None and expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)          # SQLite drops the time zone
        if row and row["holder"] != holder and expires > t:
            return (False, False) if report else False
        held = bool(row) and row["holder"] == holder and expires > t
        values = {"holder": holder, "expires_at": t + timedelta(seconds=seconds)}
        if row:
            c.execute(update(leases).where(leases.c.name == name).values(**values))
        else:
            c.execute(insert(leases).values(name=name, **values))
    return (True, held) if report else True

