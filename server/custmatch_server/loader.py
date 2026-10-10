"""Loading source records into the database, load by load.

A load is one delivery of one source's records - a file, a warehouse sync, or the API's records of
a day. Each record's content hash (its fields, in the source's own column names) is compared with
the stored one: new records are inserted, changed ones updated, unchanged ones left alone, and every
insert, update and delete is kept as a version. A *full* load is the source's complete contents, so
records missing from it are marked deleted; an *incremental* load only adds and changes.
"""
import hashlib
import json

import pandas as pd
from sqlalchemy import and_, bindparam, insert, select, text, update

from . import db
from .schema import loads, source_record_versions, source_records


def content_hash(fields):
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_frame(eng, source, frame, id_column, origin, mode="incremental"):
    """Load a DataFrame of one source's records. Returns the load's counts."""
    if mode not in ("full", "incremental"):
        raise ValueError("mode must be 'full' or 'incremental'")
    frame = frame.fillna("").astype(str)
    if id_column not in frame.columns:
        raise ValueError(f"{source}: the record-id column {id_column!r} is not in the data")
    frame = frame.drop_duplicates(subset=[id_column], keep="last")
    with eng.begin() as c:
        load_id = c.execute(insert(loads).values(source=source, origin=origin, mode=mode, status="running",
                                                 started_at=db.now(), rows_in=len(frame))).inserted_primary_key[0]
    try:
        counts = _apply(eng, source, frame, id_column, load_id, mode)
    except Exception as e:
        with eng.begin() as c:
            c.execute(update(loads).where(loads.c.id == load_id).values(
                status="failed", finished_at=db.now(), error=f"{type(e).__name__}: {e}"))
        raise
    with eng.begin() as c:
        c.execute(update(loads).where(loads.c.id == load_id).values(status="done", finished_at=db.now(), **counts))
    return {"load_id": load_id, **counts}


def _apply(eng, source, frame, id_column, load_id, mode):
    rows = frame.to_dict("records")
    incoming = {r[id_column]: (r, content_hash(r)) for r in rows}
    t = db.now()
    with eng.begin() as c:
        db.lock(c, 424244)                                   # one load per database at a time
        stored = {rid: (h, deleted) for rid, h, deleted in c.execute(
            select(source_records.c.record_id, source_records.c.content_hash, source_records.c.deleted_at)
            .where(source_records.c.source == source))}
        new = [rid for rid in incoming if rid not in stored]
        changed = [rid for rid, (_, h) in incoming.items()
                   if rid in stored and (stored[rid][0] != h or stored[rid][1] is not None)]
        gone = [rid for rid, (_, d) in stored.items() if d is None and rid not in incoming] if mode == "full" else []
        if new:
            db._bulk(c, source_records, pd.DataFrame({
                "source": source, "record_id": new, "fields": [incoming[r][0] for r in new],
                "content_hash": [incoming[r][1] for r in new], "first_load_id": load_id,
                "last_load_id": load_id, "updated_at": t, "deleted_at": None}))
        _bulk_update(c, source, load_id, t,
                     [(rid, incoming[rid][0], incoming[rid][1]) for rid in changed], gone)
        versions = [{"source": source, "record_id": r, "load_id": load_id, "change": change,
                     "fields": incoming[r][0] if change != "delete" else None,
                     "content_hash": incoming[r][1] if change != "delete" else None}
                    for rids, change in ((new, "insert"), (changed, "update"), (gone, "delete")) for r in rids]
        db._bulk(c, source_record_versions, pd.DataFrame(versions, columns=[
            "source", "record_id", "load_id", "change", "fields", "content_hash"]))
    return {"inserted": len(new), "updated": len(changed), "deleted": len(gone),
            "unchanged": len(incoming) - len(new) - len(changed)}


def _bulk_update(c, source, load_id, t, changed, gone):
    """Write changed records [(record_id, fields, hash)] and mark `gone` deleted, in bulk: on
    Postgres the rows are copied to a temporary table and applied with one UPDATE ... FROM (one
    statement per record managed ~600 a second); elsewhere an executemany."""
    if not changed and not gone:
        return
    if c.dialect.name == "postgresql":
        c.execute(text("CREATE TEMP TABLE _changes (record_id text PRIMARY KEY, fields jsonb, "
                       "content_hash text, gone boolean) ON COMMIT DROP"))
        raw = c.connection.driver_connection
        with raw.cursor().copy("COPY _changes (record_id, fields, content_hash, gone) FROM STDIN") as cp:
            for rid, f, h in changed:
                cp.write_row([rid, json.dumps(f), h, False])
            for rid in gone:
                cp.write_row([rid, None, None, True])
        c.execute(text("""
            UPDATE source_records s SET
                fields = CASE WHEN ch.gone THEN s.fields ELSE ch.fields END,
                content_hash = CASE WHEN ch.gone THEN s.content_hash ELSE ch.content_hash END,
                deleted_at = CASE WHEN ch.gone THEN :t ELSE NULL END,
                last_load_id = :load, updated_at = :t
            FROM _changes ch WHERE s.source = :source AND s.record_id = ch.record_id"""),
                  {"t": t, "load": load_id, "source": source})
        return
    where = and_(source_records.c.source == bindparam("_s"), source_records.c.record_id == bindparam("_r"))
    if changed:
        c.execute(update(source_records).where(where).values(
            fields=bindparam("_f"), content_hash=bindparam("_h"), last_load_id=load_id, updated_at=t,
            deleted_at=None), [{"_s": source, "_r": rid, "_f": f, "_h": h} for rid, f, h in changed])
    if gone:
        c.execute(update(source_records).where(where).values(deleted_at=t, last_load_id=load_id, updated_at=t),
                  [{"_s": source, "_r": rid} for rid in gone])


def load_file(eng, config, source, path, mode="incremental"):
    """Load one source's file (CSV or Parquet), using the config's column mapping for its id."""
    src = next((s for s in config["sources"] if s["name"] == source), None)
    if src is None:
        raise ValueError(f"source {source!r} is not in the config")
    path = str(path)
    frame = pd.read_parquet(path) if path.endswith((".parquet", ".pq")) else \
        pd.read_csv(path, dtype=str, keep_default_na=False, encoding=src.get("encoding") or "utf-8")
    return load_frame(eng, source, frame[mapped_columns(src, frame.columns)], src["columns"]["record_id"],
                      origin=path, mode=mode)


def mapped_columns(src, available):
    """The columns the config maps for this source, in the file: only what matching uses is stored -
    a file's other columns (anything a system happens to export) never reach the database."""
    return [c for c in dict.fromkeys(src["columns"].values()) if c in set(available)]


def api_load(conn, source):
    """The API's load for this source today: one load per source per day, its counts kept current."""
    day = db.now().date().isoformat()
    origin = f"api:{day}"
    lid = conn.execute(select(loads.c.id).where(loads.c.source == source).where(loads.c.origin == origin)).scalar()
    if lid is None:
        lid = conn.execute(insert(loads).values(source=source, origin=origin, mode="incremental", status="done",
                                                started_at=db.now(), finished_at=db.now(), rows_in=0, inserted=0,
                                                updated=0, deleted=0, unchanged=0)).inserted_primary_key[0]
    return lid


def write_api_record(conn, source, record_id, fields):
    """Store one record that arrived through the API as a source record, in today's API load."""
    lid = api_load(conn, source)
    h = content_hash(fields)
    t = db.now()
    have = conn.execute(select(source_records.c.content_hash).where(source_records.c.source == source)
                        .where(source_records.c.record_id == record_id)).first()
    if have is None:
        conn.execute(insert(source_records).values(source=source, record_id=record_id, fields=fields,
                                                   content_hash=h, first_load_id=lid, last_load_id=lid,
                                                   updated_at=t))
        change = "insert"
    else:
        conn.execute(update(source_records).where(source_records.c.source == source)
                     .where(source_records.c.record_id == record_id)
                     .values(fields=fields, content_hash=h, last_load_id=lid, updated_at=t, deleted_at=None))
        change = "update"
    exists = conn.execute(select(source_record_versions.c.load_id).where(source_record_versions.c.source == source)
                          .where(source_record_versions.c.record_id == record_id)
                          .where(source_record_versions.c.load_id == lid)).first()
    if exists is None:
        conn.execute(insert(source_record_versions).values(source=source, record_id=record_id, load_id=lid,
                                                           change=change, fields=fields, content_hash=h))
    conn.execute(update(loads).where(loads.c.id == lid).values(
        rows_in=loads.c.rows_in + 1, inserted=loads.c.inserted + int(change == "insert"),
        updated=loads.c.updated + int(change == "update"), finished_at=t))
    return lid
