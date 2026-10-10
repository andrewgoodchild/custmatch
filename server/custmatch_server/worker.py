"""A batch run over the source records in the database - every live record of every configured
source, including those that arrived through the API - with customer IDs carried over from the
current run, published in one transaction. The batch run is the source of truth: it confirms or moves
every real-time decision, and records the newest load it saw (runs.input_load_id)."""
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from custmatch import pipeline

from . import db
from .schema import overrides, realtime_merges, realtime_moves, realtime_records, source_records


def _inputs(conn, cfg, work):
    """Each configured source's live records, from the database, as a Parquet file in work."""
    cfg = json.loads(json.dumps(cfg))
    counts = {}
    for src in cfg["sources"]:
        rows = conn.execute(select(source_records.c.fields).where(source_records.c.source == src["name"])
                            .where(source_records.c.deleted_at.is_(None))).scalars().all()
        rows = [db.as_json(r) for r in rows]
        cols = list(dict.fromkeys(list(src["columns"].values()) + [k for r in rows[:1] for k in r]))
        frame = pd.DataFrame(rows, columns=cols).fillna("").astype(str)
        path = work / f"{src['name']}.parquet"
        frame.to_parquet(path, index=False)
        src["path"] = str(path)
        src.pop("encoding", None)
        counts[src["name"]] = len(frame)
    return cfg, counts


def _importance(state):
    """The model's feature importances, named: to compare one run's model with another's."""
    import pyarrow.parquet as pq

    from custmatch.core import feature_names
    from custmatch import schema
    from custmatch.models import load_model
    model = load_model(state)["forest"]
    if not hasattr(model, "feature_importances_"):
        return {}
    cfg = json.loads((state / "config.json").read_text())
    opts = schema.feature_options(cfg)
    names = feature_names(pq.read_schema(state / "records.parquet").names, opts["extra_fields"], opts["rich"])
    names += [f"feature {k}" for k in range(len(names), len(model.feature_importances_))]
    return {n: round(float(v), 4) for n, v in zip(names, model.feature_importances_)}


def _with_decisions(lab, decisions):
    """The labels plus stewards' decisions as labels: same -> 1, different -> 0, marked
    by = "active" (chosen by a person, not drawn at random: they train the model but stay out
    of the threshold's precision and recall estimate, as active-learning labels do). A decision
    replaces an existing label on the same pair."""
    if not len(decisions):
        return lab, 0
    if "by" not in lab.columns:
        lab = lab.assign(by="random")
    pair = lambda a, b: [tuple(sorted(x)) for x in zip(a, b)]  # noqa: E731
    taken = set(pair(decisions.a, decisions.b))
    lab = lab[[x not in taken for x in pair(lab.a, lab.b)]]
    extra = pd.DataFrame({"a": decisions.a, "b": decisions.b,
                          "label": (decisions.decision == "same").astype(int).astype(str), "by": "active"})
    return pd.concat([lab, extra[[c for c in lab.columns if c in extra.columns]]], ignore_index=True), len(extra)


def _realtime_snapshot(eng):
    """Taken before the run reads its inputs: the highest seq of each real-time table (rows up
    to it are in the run's input, so it reconciles them; later ones stay pending), each pending
    real-time record's customer as a vote for keeping that ID, and the highest customer ID the
    real-time matcher ever issued, so the run never gives a new customer one of them."""
    from sqlalchemy import func
    with eng.connect() as c:
        snap = {t: c.execute(select(func.max(t.c.seq))).scalar()
                for t in (realtime_records, realtime_merges, realtime_moves)}
        rows = c.execute(select(realtime_records.c.source, realtime_records.c.record_id, realtime_records.c.customer_id)
                         .where(realtime_records.c.reconciled_run.is_(None))).all()
        merges = c.execute(select(realtime_merges.c.from_customer, realtime_merges.c.into_customer)
                           .where(realtime_merges.c.reconciled_run.is_(None)).order_by(realtime_merges.c.seq)).all()
        moves = c.execute(select(realtime_moves.c.source, realtime_moves.c.record_id, realtime_moves.c.to_customer)
                          .where(realtime_moves.c.reconciled_run.is_(None)).order_by(realtime_moves.c.seq)).all()
        issued = max([-1] + [int(x) for x in (
            c.execute(select(func.max(realtime_records.c.customer_id))).scalar(),
            c.execute(select(func.max(realtime_moves.c.to_customer))).scalar(),
            c.execute(select(func.max(realtime_merges.c.into_customer))).scalar()) if x is not None])
    into = {int(a): int(b) for a, b in merges}

    def resolve(cid):
        seen = set()
        while cid in into and cid not in seen:
            seen.add(cid)
            cid = into[cid]
        return cid
    votes = {f"{s}:{r}": resolve(int(cid)) for s, r, cid in rows}
    votes.update({f"{s}:{r}": int(to) for s, r, to in moves})       # a record split out keeps its new ID
    return snap, votes, issued


def run_batch(eng, config_path, labels_path, data_dir, log=print, decisions_as_labels=True):
    """decisions_as_labels: stewards' decisions also train the model (as well as overriding their
    own pairs) - server/learning_test.py measures what that gives."""
    config_path, data_dir = Path(config_path).resolve(), Path(data_dir)
    cfg = json.loads(config_path.read_text())
    if cfg.get("overrides"):                                # paths in the config stay relative to it
        cfg["overrides"] = [str((config_path.parent / x).resolve()) for x in db.as_list(cfg["overrides"])]
    with eng.connect() as c:
        prev = db.current(c)
        input_load = db.latest_load_id(c)
    snap, votes, issued = _realtime_snapshot(eng)          # before the inputs: what this run will reconcile
    run_id = db.new_run(eng)
    out = data_dir / "runs" / f"run-{run_id:05d}"
    work = out / "inputs"
    work.mkdir(parents=True, exist_ok=True)
    try:
        with eng.connect() as c:
            run_cfg, counts = _inputs(c, cfg, work)
        if not sum(counts.values()):
            raise ValueError("no source records in the database: load some first (custmatch-server load)")
        # stewards' decisions in the database join any overrides file the config names
        with eng.connect() as c:
            ov = pd.read_sql(select(overrides.c.a, overrides.c.b, overrides.c.decision)
                             .where(overrides.c.status == "applied")              # not proposals awaiting approval
                             .where(overrides.c.decision.in_(("same", "different"))), c)
        if len(ov):
            ov.to_csv(work / "overrides.csv", index=False)
            run_cfg["overrides"] = db.as_list(run_cfg.get("overrides")) + [str(work / "overrides.csv")]
        (work / "config.json").write_text(json.dumps(run_cfg))
        # labels on records a later full load deleted cannot train the run: dropped, and reported
        lab = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
        learned = 0
        if decisions_as_labels and len(ov):
            lab, learned = _with_decisions(lab, ov)
        with eng.connect() as c:
            live = {f"{s}:{r}" for s, r in c.execute(select(source_records.c.source, source_records.c.record_id)
                                                     .where(source_records.c.deleted_at.is_(None)))}
        keep = lab.a.isin(live) & lab.b.isin(live)
        lab[keep].to_csv(work / "labels.csv", index=False)
        dropped_labels = int((~keep).sum())
        s = pipeline.run(work / "config.json", work / "labels.csv", out / "result", log=log,
                         previous_dir=Path(prev["artifact_uri"]) if prev else None,
                         previous_extra=votes, issued=issued)
        summary = {"records": s["records"], "customers": s["customers"], "threshold": s["threshold"],
                   "records_by_source": counts, "labels_dropped_for_deleted_records": dropped_labels,
                   "labels_from_decisions": learned,
                   "estimated": {k: v for k, v in (s.get("cross_fitted_closure_aware") or {}).items()
                                 if k in ("precision", "recall", "f1")},
                   "labelled_pairs": s.get("labelled_pairs"), "candidate_pairs": s.get("candidate_pairs"),
                   "feature_importance": _importance(out / "result" / "state"),
                   "warnings": s["warnings"]}
        # the real-time rows the run saw are confirmed or moved by it, in the publish transaction
        n, retired = db.publish(eng, run_id, out / "result", summary, input_load, reconcile=snap)
        log(f"run {run_id} published: {n:,} records, {summary['customers']:,} customers, "
            f"input up to load {input_load}; retired runs {retired}")
        return run_id, summary
    except Exception as e:
        db.fail_run(eng, run_id, f"{type(e).__name__}: {e}")
        raise

