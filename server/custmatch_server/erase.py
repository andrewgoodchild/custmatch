"""Erasing people from the service, for erasure requests: one operation that reaches every copy.

  custmatch-server erase CONFIG --labels LABELS KEY... | --file RECORDS.csv [--note TEXT] [--dry-run]

A key is source:record_id (RECORDS.csv has source and record_id columns, as `custmatch forget`
takes). The operation:

  1. deletes the records from source_records and every version of them in source_record_versions;
  2. deletes what was written about them: stewards' decisions on their pairs, pinned golden values,
     review claims, and the real-time matcher's rows; in the steward log the actions naming them
     keep their action, actor, time and keys but lose their detail and note (a pinned value, a
     steward's free text), and an "erased" action is logged per record, by key only;
  3. runs a batch (worker.run_batch) over the remaining records, published with every earlier run
     retired - a rollback to a run holding the person would undo the erasure - and the retired
     runs' files deleted (db.remove_run_files). The real-time matcher reloads from the new run
     within a second, and its replay of real-time records finds none for them.

Without a published run there is nothing to rebuild: steps 1 and 2 only. --dry-run reports where
the records' data is and changes nothing.

Not reached: the source systems (load the records again and they are back: erase them there
first), your labels file (a pair naming the record is dropped at the next run), files in the
inbox's done/ folder, and backups (they age out; server/README.md "Backup and restore").
"""
import json

from sqlalchemy import and_, delete, func, insert, or_, select, update

from . import db
from .schema import (
    golden_pins,
    links,
    overrides,
    realtime_moves,
    realtime_records,
    record_customer,
    review_claims,
    source_record_versions,
    source_records,
    steward_log,
)


def parse_keys(keys):
    """[(source, record_id)] from source:record_id keys, in order, without repeats."""
    out = []
    for key in dict.fromkeys(keys):
        source, sep, record_id = str(key).partition(":")
        if not sep or not source or not record_id:
            raise ValueError(f"a record key is source:record_id, not {key!r}")
        out.append((source, record_id))
    return out


def _is(table, pairs, source="source", record_id="record_id"):
    return or_(*[and_(table.c[source] == s, table.c[record_id] == r) for s, r in pairs])


def where(c, keys):
    """How many rows name these records, table by table, and the current run's id if it holds any."""
    pairs = parse_keys(keys)
    cur = db.current(c)
    run_id = cur["id"] if cur else -1
    count = lambda sel: int(c.execute(select(func.count()).select_from(sel.subquery())).scalar())  # noqa: E731
    rows = {"source_records": count(select(source_records).where(_is(source_records, pairs))),
            "source_record_versions": count(select(source_record_versions).where(_is(source_record_versions, pairs))),
            "overrides": count(select(overrides).where(overrides.c.a.in_(keys) | overrides.c.b.in_(keys))),
            "steward_log": count(select(steward_log).where(steward_log.c.a.in_(keys) | steward_log.c.b.in_(keys))),
            "golden_pins": count(select(golden_pins).where(golden_pins.c.record_key.in_(keys))),
            "review_claims": count(select(review_claims).where(review_claims.c.a.in_(keys) | review_claims.c.b.in_(keys))),
            "realtime_records": count(select(realtime_records).where(_is(realtime_records, pairs))),
            "realtime_moves": count(select(realtime_moves).where(_is(realtime_moves, pairs))),
            "record_customer": count(select(record_customer).where(record_customer.c.run_id == run_id)
                                     .where(_is(record_customer, pairs))),
            "links": count(select(links).where(links.c.run_id == run_id)
                           .where(_is(links, pairs, "source_a", "record_a") | _is(links, pairs, "source_b", "record_b")))}
    known = {f"{s}:{r}" for s, r in c.execute(select(source_records.c.source, source_records.c.record_id)
                                              .where(_is(source_records, pairs)))}
    known |= {f"{s}:{r}" for s, r in c.execute(select(record_customer.c.source, record_customer.c.record_id)
                                               .where(record_customer.c.run_id == run_id).where(_is(record_customer, pairs)))}
    return rows, known, (cur["id"] if cur else None)


def erase(eng, keys, config_path, labels_path, data_dir, note=None, actor=None, dry_run=False, log=print):
    """Erase these records (source:record_id) from the service: the module docstring says what that
    reaches. Returns a report: the keys, the rows found per table, the run published and the runs
    retired (None and [] without a run), and whether it was a dry run."""
    keys = [str(k) for k in dict.fromkeys(keys)]
    pairs = parse_keys(keys)
    with eng.connect() as c:
        rows, known, current = where(c, keys)
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise ValueError(f"{len(unknown)} record(s) are not in the database or the current run, e.g. {unknown[:3]}")
    report = {"erased": keys, "rows": rows, "run": None, "retired": [], "dry_run": dry_run}
    if dry_run:
        return report
    with eng.begin() as c:
        db.lock(c, 424244)                                   # the loader's lock: no load lands meanwhile
        c.execute(delete(source_record_versions).where(_is(source_record_versions, pairs)))
        c.execute(delete(source_records).where(_is(source_records, pairs)))
        c.execute(delete(overrides).where(overrides.c.a.in_(keys) | overrides.c.b.in_(keys)))
        c.execute(delete(golden_pins).where(golden_pins.c.record_key.in_(keys)))
        c.execute(delete(review_claims).where(review_claims.c.a.in_(keys) | review_claims.c.b.in_(keys)))
        c.execute(delete(realtime_records).where(_is(realtime_records, pairs)))
        c.execute(delete(realtime_moves).where(_is(realtime_moves, pairs)))
        c.execute(update(steward_log).where(steward_log.c.a.in_(keys) | steward_log.c.b.in_(keys))
                  .values(detail=None, note=None))
        for key in keys:
            c.execute(insert(steward_log).values(at=db.now(), actor=actor, action="erased", a=key, note=note,
                                                 detail=json.dumps({"rows": rows})))
    log(f"erased {len(keys)} record(s) from the database: " + ", ".join(f"{t} {n}" for t, n in rows.items() if n))
    if current is None:
        return report
    from .worker import run_batch
    run_id, summary = run_batch(eng, config_path, labels_path, data_dir, log=log, keep_runs=1)
    report |= {"run": run_id, "retired": summary["retired_runs"]}
    return report
