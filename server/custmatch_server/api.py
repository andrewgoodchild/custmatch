"""The HTTP API (FastAPI): lookups from the database, real-time matching through the matcher.

  GET  /health                           the process is up (no key needed)
  GET  /status                            the health checks (health.py): runs, customers, review queue, loads
  GET  /runs/current                      the current run: when, from which loads, its summary
  GET  /runs                              every run: status, when, records, customers, threshold, estimated
                                          accuracy and the model's feature importances - to compare runs
  GET  /loads                             every load: source, when, from where, what changed
  GET  /sources/{source}/records/{id}     a source record: its fields now, and every version by load
  GET  /records/{source}/{record_id}      the record's customer: the current run, or a real-time match
  GET  /customers/{customer_id}           the customer's golden record and records (?fields=true: their fields)
  GET  /customers/{customer_id}/events    the customer's merges, splits and creation, run by run
  POST /match                             {"source", "record_id", "fields", "write": false}
  GET  /review?limit=&offset=             the current run's uncertain pairs not yet decided, both records' fields
  GET  /search?q=                         records by key (source:record_id), record id, email, phone or name
  GET  /explain?a=&b=                     why two records score what they score: each feature's share (a waterfall)
  GET  /customers?order=size|weakest|bridges|sparse|random   customers to inspect, with link statistics
  GET  /customers/{customer_id}/graph     a customer's records and the links between them, with scores
  GET  /links?min=&max=&limit=            a random sample of accepted links in a score band, both records summarised
  GET  /quality                           the model, the threshold and the data: threshold sweep, score
                                          histogram, feature importance, completeness and common values
                                          by source, blocking, junk and shared contact values
  GET  /dashboard                         the steward's page (it calls this API with the steward's key)
  GET  /overrides                         stewards' decisions
  POST /overrides                         {"a", "b", "decision": "same" | "different", "note", "dry_run"}
  POST /overrides/{id}/approve | reject   four-eyes; DELETE /overrides/{id} revokes
  POST /split                             {"record_key", "note", "dry_run"}: a record out of its customer
  POST /customers/{id}/merge              {"other", "note", "dry_run"}: merge another customer into it

A customer ID in an answer has the real-time matcher's merges since the batch run applied to it.

The current run is re-read at most once a second; the matcher reloads when it changes.

Every route but /health needs an API key (auth.py): `Authorization: Bearer <key>`, with the role
the route needs - read for lookups, match for POST /match, steward for POST /overrides. An
override's created_by is the key's name. CUSTMATCH_AUTH=off turns this off, for a local trial.
"""
import base64
import hashlib
import io as _io
import json
import logging
import os
import re

import threading
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi import Query
from pydantic import BaseModel, Field
from sqlalchemy import func, insert, select

from custmatch.io import ConfigError

from . import auth, db
from . import health as health_checks
from .db import as_json as _json
from .matcher import Matcher, StaleMatcher, small_proba
from .schema import (customer_events, customers, golden_pins, links, loads, overrides, realtime_merges, realtime_records,
                     record_customer, review_claims, runs, source_record_versions, source_records, steward_log)

GOLDEN = db.GOLDEN
CLAIM_MINUTES = 15          # a review pair shown to one steward is held from the others this long


class OverrideRequest(BaseModel):
    a: str
    b: str
    decision: str
    created_by: str | None = None
    note: str | None = None
    dry_run: bool = False


class ActionRequest(BaseModel):
    note: str | None = None
    dry_run: bool = False


class MergeRequest(ActionRequest):
    other: int


class PinRequest(BaseModel):
    record_key: str
    field: str
    value: str
    note: str | None = None


KEY_PART = r"^[^\x00-\x1f\x7f]+$"      # no control characters (record ids may hold any other character)


class MatchRequest(BaseModel):
    source: str = Field(max_length=64, pattern=KEY_PART)
    record_id: str = Field(max_length=256, pattern=KEY_PART)
    fields: dict
    write: bool = False




def create_app(eng, require_keys=None):
    """require_keys: None reads CUSTMATCH_AUTH (on unless "off")."""
    if require_keys is None:
        require_keys = os.environ.get("CUSTMATCH_AUTH", "on").lower() != "off"
    if not require_keys:
        logging.getLogger("custmatch").warning("CUSTMATCH_AUTH=off: the API answers anyone who can reach it")
    keys = auth.Keys(eng)

    def need(role):
        def check(authorization: str | None = Header(default=None)):
            if not require_keys:
                return None
            token = authorization.removeprefix("Bearer ").strip() if authorization else None
            who = keys.identify(token)
            if who is None:
                raise HTTPException(401, "an API key is needed: Authorization: Bearer <key> "
                                         "(custmatch-server key create NAME --role ROLE)",
                                    headers={"WWW-Authenticate": "Bearer"})
            if not auth.allows(who[1], role):
                raise HTTPException(403, f"this key's role is {who[1]}; this needs {role}")
            return who
        return check
    read, match_role, steward = Depends(need("read")), Depends(need("match")), Depends(need("steward"))

    four_eyes = os.environ.get("CUSTMATCH_FOUR_EYES", "off").lower() == "on"   # decisions need a second steward
    if four_eyes and not require_keys:
        raise RuntimeError("CUSTMATCH_FOUR_EYES=on needs authentication: with CUSTMATCH_AUTH=off nobody is "
                           "identified, so nobody can be told apart from the proposer")
    app = FastAPI(title="custmatch")
    page = (Path(__file__).parent / "static" / "dashboard.html").read_text()
    script = re.search(r"<script>(.*?)</script>", page, re.S).group(1)
    csp = ("default-src 'none'; script-src 'sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
           + "'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; "
           "form-action 'none'; frame-ancestors 'none'")

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")         # responses hold personal data
        if request.url.path == "/dashboard":
            response.headers["Content-Security-Policy"] = csp
        return response
    state = {"matcher": None, "run": None, "checked": 0.0}
    holder = db.holder_name()                    # this process's lease name
    load_lock = threading.Lock()

    def current():
        if time.monotonic() - state["checked"] > 1.0:
            with eng.connect() as c:
                state["run"] = db.current(c)
            state["checked"] = time.monotonic()
        return state["run"]

    def merged(c):
        """Customer id -> the customer it was merged into since the batch run (chains followed)."""
        rows = c.execute(select(realtime_merges.c.from_customer, realtime_merges.c.into_customer)
                         .where(realtime_merges.c.reconciled_run.is_(None))
                         .order_by(realtime_merges.c.seq)).all()
        into = {}
        for gone, target in rows:
            into[int(gone)] = int(target)

        def resolve(cid):
            seen = set()
            while cid in into and cid not in seen:
                seen.add(cid)
                cid = into[cid]
            return cid
        return resolve

    def matcher():
        run = current()
        if run is None:
            raise HTTPException(503, "no published run yet: load records and run a batch first")
        with load_lock:
            m = state["matcher"]
            if m is None or m.run_id != run["id"]:
                try:
                    if m is None:
                        state["matcher"] = m = Matcher(eng, run, holder=holder)
                    else:                                # no write on the old one may land between its
                        with m.lock:                     # last write and the new one's reading of them
                            state["matcher"] = m = Matcher(eng, run, holder=holder)
                except ConfigError as e:                 # e.g. a model this install cannot load
                    raise HTTPException(503, str(e)) from None
        return m

    def need_run():
        run = current()
        if run is None:
            raise HTTPException(404, "no published run")
        return run

    def actor(who):
        """The caller's name (its key's), or None when authentication is off."""
        return who[0] if who else None

    def write(fn):
        """A write through the matcher: a matcher whose memory another process made stale (it took
        the lease meanwhile) is rebuilt and the write retried; errors become HTTP statuses."""
        try:
            try:
                return fn(matcher())
            except StaleMatcher:
                with load_lock:
                    state["matcher"] = None
                return fn(matcher())
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'")) from None
        except PermissionError as e:
            raise HTTPException(409, str(e)) from None
        except ValueError as e:
            raise HTTPException(400, str(e)) from None

    @app.get("/health")
    def health():
        return {"ok": True, "database": eng.dialect.name}

    @app.get("/runs/current", dependencies=[read])
    def run_current():
        run = need_run()
        return {"id": run["id"], "published_at": run["published_at"], "input_load_id": run["input_load_id"],
                "summary": _json(run["summary"])}

    @app.get("/status", dependencies=[read])
    def status():
        results = health_checks.checks(eng)
        return {"status": health_checks.worst(results), "checks": results}

    @app.get("/runs", dependencies=[read])
    def list_runs():
        with eng.connect() as c:
            rows = c.execute(select(runs).order_by(runs.c.id.desc()).limit(50)).mappings().all()
        return [{"id": r["id"], "status": r["status"], "created_at": r["created_at"], "published_at": r["published_at"],
                 "input_load_id": r["input_load_id"], "error": r["error"], "summary": _json(r["summary"])} for r in rows]

    @app.get("/loads", dependencies=[read])
    def list_loads():
        with eng.connect() as c:
            rows = c.execute(select(loads).order_by(loads.c.id)).mappings().all()
        return [dict(r) for r in rows]

    @app.get("/sources/{source}/records/{record_id:path}", dependencies=[read])
    def source_record(source: str, record_id: str):
        with eng.connect() as c:
            row = c.execute(select(source_records).where(source_records.c.source == source)
                            .where(source_records.c.record_id == record_id)).mappings().first()
            if row is None:
                raise HTTPException(404, "source record not found")
            versions = c.execute(select(source_record_versions, loads.c.origin, loads.c.started_at)
                                 .join(loads, loads.c.id == source_record_versions.c.load_id)
                                 .where(source_record_versions.c.source == source)
                                 .where(source_record_versions.c.record_id == record_id)
                                 .order_by(source_record_versions.c.load_id)).mappings().all()
        return {"source": source, "record_id": record_id, "fields": _json(row["fields"]),
                "deleted_at": row["deleted_at"], "first_load_id": row["first_load_id"],
                "last_load_id": row["last_load_id"],
                "versions": [{"load_id": v["load_id"], "origin": v["origin"], "at": v["started_at"],
                              "change": v["change"], "fields": _json(v["fields"])} for v in versions]}

    @app.get("/records/{source}/{record_id:path}", dependencies=[read])
    def record(source: str, record_id: str):
        run = current()
        with eng.connect() as c:
            row = c.execute(select(record_customer).where(record_customer.c.run_id == (run["id"] if run else -1))
                            .where(record_customer.c.source == source)
                            .where(record_customer.c.record_id == record_id)).mappings().first()
            rt = c.execute(select(realtime_records).where(realtime_records.c.source == source)
                           .where(realtime_records.c.record_id == record_id)
                           .where(realtime_records.c.reconciled_run.is_(None))).mappings().first()
            resolve = merged(c)
        live = matcher().lookup(source, record_id) if run is not None else None
        if row is None and rt is None and live is None:
            raise HTTPException(404, "record not found")
        batch = None if row is None else str(row["customer_id"])
        fallback = rt["customer_id"] if rt is not None else (row["customer_id"] if row is not None else None)
        out = {"source": source, "record_id": record_id,
               "customer_id": live or (None if fallback is None else str(resolve(int(fallback)))),
               "provisional": rt is not None or live != batch, "batch_customer_id": batch}
        if rt is not None:
            out["matched_against_run"] = rt["matched_against_run"]
        if row is not None:
            out |= {"household_id": None if row["household_id"] is None else str(row["household_id"]), "run": row["run_id"]}
        return out

    def membership(customer_id):
        """(the customer's records now, those of them it held in the batch run)."""
        m = matcher()
        now = [m._key(i) for i in m._members(int(customer_id))]
        with eng.connect() as c:
            batch = {f"{s_}:{r}" for s_, r in c.execute(select(record_customer.c.source, record_customer.c.record_id)
                                                        .where(record_customer.c.run_id == current()["id"])
                                                        .where(record_customer.c.customer_id == customer_id))}
        return now, batch

    def pins_for(c, keys):
        """Pinned golden values on these records, latest first per field."""
        rows = c.execute(select(golden_pins).where(golden_pins.c.record_key.in_(list(keys)))
                         .order_by(golden_pins.c.created_at.desc())).mappings().all() if keys else []
        out = {}
        for r in rows:
            out.setdefault(r["field"], dict(r))
        return out

    @app.get("/customers/{customer_id}", dependencies=[read])
    def customer(customer_id: int, fields: bool = False):
        run = need_run()
        now, batch = membership(customer_id)
        with eng.connect() as c:
            cust = c.execute(select(customers).where(customers.c.run_id == run["id"])
                             .where(customers.c.customer_id == customer_id)).mappings().first()
            if not now:
                into = merged(c)(customer_id)
                if into != customer_id:
                    return {"customer_id": str(customer_id), "merged_into": str(into), "records": [],
                            "provisional_records": []}
                raise HTTPException(404, "customer not found")
            pins = pins_for(c, now)
            household = []
            if cust is not None and cust["household_id"] is not None:
                household = [{"customer_id": str(r["customer_id"]), "records": r["record_count"],
                              "name": f"{r['first_name'] or ''} {r['last_name'] or ''}".strip()}
                             for r in c.execute(select(customers.c.customer_id, customers.c.first_name, customers.c.last_name,
                                                       customers.c.record_count)
                                                .where(customers.c.run_id == run["id"])
                                                .where(customers.c.household_id == cust["household_id"])
                                                .where(customers.c.customer_id != customer_id).limit(20)).mappings()]
            out = {"record_fields": fields_of(c, now)} if fields else {}
        golden = None
        if cust is not None:
            golden = {f: cust[f] for f in GOLDEN} | (_json(cust["golden_extra"]) or {})
            for f, pin in pins.items():
                golden[f] = pin["value"]
        return out | {"customer_id": str(customer_id), "run": run["id"],
                      "household_id": None if cust is None or cust["household_id"] is None else str(cust["household_id"]),
                      "created_run": None if cust is None else cust["created_run"],
                      "golden_record": golden, "pinned": {f: {"value": p_["value"], "by": p_["created_by"], "id": p_["id"],
                                                             "record_key": p_["record_key"]} for f, p_ in pins.items()},
                      "household": household,
                      "records": [k for k in now if k in batch],
                      "provisional_records": [k for k in now if k not in batch],
                      "moved_out": sorted(batch - set(now))}

    @app.get("/customers/{customer_id}/lineage", dependencies=[read])
    def lineage(customer_id: int):
        """Each golden field: every value the customer's records hold, which records hold it,
        whether it won (survivorship) and whether a steward pinned it."""
        m = matcher()
        now, _ = membership(customer_id)
        if not now:
            raise HTTPException(404, "customer not found")
        rows = m._rows([m._index_of(k) for k in now])
        gold = customer(customer_id)["golden_record"] or {}
        with eng.connect() as c:
            pins = pins_for(c, now)
        out = []
        for f in [f for f in GOLDEN if f in rows.columns] + [f for f in gold if f not in GOLDEN and f in rows.columns]:
            vals = {}
            for k, v in zip(now, rows[f]):
                if str(v).strip():
                    vals.setdefault(str(v), []).append(k)
            out.append({"field": f, "golden": gold.get(f), "pinned": f in pins,
                        "values": [{"value": v, "records": ks, "won": str(gold.get(f)) == v} for v, ks in
                                   sorted(vals.items(), key=lambda x: -len(x[1]))]})
        return {"customer_id": str(customer_id), "fields": out}

    @app.get("/customers/{customer_id}/audit", dependencies=[read])
    def audit(customer_id: int):
        """Stewards' actions on this customer's records: decisions, approvals, revocations, splits, pins."""
        now, batch = membership(customer_id)
        keys = list(set(now) | batch)
        with eng.connect() as c:
            rows = c.execute(select(steward_log).where(steward_log.c.a.in_(keys) | steward_log.c.b.in_(keys)
                                                       | (steward_log.c.customer_id == customer_id))
                             .order_by(steward_log.c.id.desc()).limit(200)).mappings().all()
        return [dict(r) | {"detail": _json(r["detail"])} for r in rows]

    @app.get("/customers/{customer_id}/events", dependencies=[read])
    def events(customer_id: int):
        with eng.connect() as c:
            rows = c.execute(select(customer_events).where(customer_events.c.customer_id == customer_id)
                             .order_by(customer_events.c.id)).mappings().all()
        return [dict(r) for r in rows]

    def fields_of(c, keys):
        """{key: the source record's fields} for record keys source:record_id."""
        out = {}
        by_source = {}
        for k in keys:
            src, _, rid = k.partition(":")
            by_source.setdefault(src, []).append(rid)
        for src, rids in by_source.items():
            for rid, f in c.execute(select(source_records.c.record_id, source_records.c.fields)
                                    .where(source_records.c.source == src)
                                    .where(source_records.c.record_id.in_(rids))):
                out[f"{src}:{rid}"] = _json(f)
        return out

    @app.get("/review", dependencies=[read])
    def review(limit: int = Query(20, ge=0, le=200), offset: int = Query(0, ge=0), source: str | None = None,
               call: str | None = None, min_score: float = 0.0, max_score: float = 1.0, who=read):
        """The run's review band (pairs near the threshold, most uncertain first), less the pairs
        a steward has decided since, and less the pairs shown to another steward in the last
        CLAIM_MINUTES (claimed: two stewards never work the same pair at once). Filters: a source
        either record comes from; the model's call (linked / apart); a score range."""
        run = need_run()
        path = Path(run["artifact_uri"]) / "review.csv"
        rv = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(columns=["a", "b"])
        me = who[0] if who else "anonymous"
        claims = who is None or auth.allows(who[1], "steward")
        t = db.now()
        with eng.begin() as c:
            decided = {(a, b) for a, b in c.execute(select(overrides.c.a, overrides.c.b))}
            claimed = {(a, b) for a, b in c.execute(select(review_claims.c.a, review_claims.c.b)
                                                    .where(review_claims.c.holder != me)
                                                    .where(review_claims.c.expires_at > t))}
            pairs = [tuple(sorted((a, b))) for a, b in zip(rv.a, rv.b)]
            rv = rv[[p not in decided for p in pairs]]
            if len(rv):
                sc = pd.to_numeric(rv.score)
                keep = (sc >= min_score) & (sc <= max_score)
                if source:
                    keep &= rv.a.str.startswith(source + ":") | rv.b.str.startswith(source + ":")
                if call in ("linked", "apart"):
                    keep &= rv.linked.eq("True") == (call == "linked")
                rv = rv[keep]
            free = [tuple(sorted(x)) not in claimed for x in zip(rv.a, rv.b)]
            held = int(len(rv) - sum(free))
            rv = rv[free]
            page = rv.iloc[offset:offset + limit]
            for a, b in (zip(page.a, page.b) if claims else ()):     # claim what this steward is shown
                x, y = sorted((a, b))
                c.execute(review_claims.delete().where(review_claims.c.a == x).where(review_claims.c.b == y))
                c.execute(insert(review_claims).values(a=x, b=y, holder=me,
                                                       expires_at=t + timedelta(minutes=CLAIM_MINUTES)))
            keys = list(page.a) + list(page.b)
            fields = fields_of(c, keys)
            resolve = merged(c)
            cust = {}
            for k in keys:
                src, _, rid = k.partition(":")
                row = c.execute(select(record_customer.c.customer_id).where(record_customer.c.run_id == run["id"])
                                .where(record_customer.c.source == src)
                                .where(record_customer.c.record_id == rid)).first()
                cust[k] = None if row is None else str(resolve(row[0]))
        return {"remaining": int(len(rv)), "held_by_others": held, "run": run["id"],
                "pairs": [{"a": a, "b": b, "score": float(sc), "linked": lk == "True",
                           "evidence": ev, "a_fields": fields.get(a), "b_fields": fields.get(b),
                           "a_customer": cust.get(a), "b_customer": cust.get(b)}
                          for a, b, sc, lk, ev in zip(page.a, page.b, page.score, page.linked,
                                                      page.get("evidence", pd.Series([""] * len(page))))]}

    @app.get("/search", dependencies=[read])
    def search(q: str = Query(max_length=200), limit: int = Query(20, ge=0, le=200)):
        """Records whose key, record id, email or phone equals q, or whose name contains it (all
        words, any order). The run's records and the real-time ones, from the matcher's memory."""
        q = q.strip()
        if len(q) < 2:
            raise HTTPException(400, "search for at least two characters")
        m = matcher()
        rec = m.rec
        ql = q.lower()
        hit = (rec.record_id == q) | (rec.email == ql) | ((rec.source + ":" + rec.record_id) == q)
        digits = "".join(ch for ch in q if ch.isdigit())
        if len(digits) >= 6:
            hit |= rec.phone.str.endswith(digits[-9:])
        words = ql.split()
        if words and not digits:
            name = rec.first_name + " " + rec.last_name
            by_name = np.ones(len(rec), bool)
            for w in words:
                by_name &= name.str.contains(w, regex=False).to_numpy()
            hit |= by_name
        idx = np.flatnonzero(hit.to_numpy())[:limit].tolist()
        n = len(rec)
        for k, row in enumerate(m.added):                    # records added since the run
            if len(idx) >= limit:
                break
            if q in (row["record_id"], f"{row['source']}:{row['record_id']}") or row["email"] == ql or \
                    (words and all(w in f"{row['first_name']} {row['last_name']}" for w in words)):
                idx.append(n + k)
        with eng.connect() as c:
            resolve = merged(c)
        out = []
        for i in idx[:limit]:
            r = m._rows([i]).iloc[0]
            out.append({"key": m._key(i), "customer_id": str(resolve(m._customer(i))),
                        "name": f"{r.first_name} {r.last_name}".strip(), "email": r.email,
                        "date_of_birth": r.date_of_birth})
        return out

    per_run = {}                                    # computed once per run: (run id, name) -> value

    def cached(name, fn):
        run = current()
        key = (run["id"] if run else None, name)
        if key not in per_run:
            if len(per_run) > 16:
                per_run.clear()
            per_run[key] = fn(run)
        return per_run[key]

    def brief(m, key):
        i = m._index_of(key)
        if i is None:
            return {"key": key}
        r = m._rows([i]).iloc[0]
        return {"key": key, "name": f"{r.first_name} {r.last_name}".strip(), "email": r.email, "phone": r.phone,
                "date_of_birth": r.date_of_birth, "street": f"{r.addr_number} {r.addr_street}".strip()}

    @app.get("/explain", dependencies=[read])
    def explain(a: str, b: str, score: float | None = None):
        """The model's score for the pair, feature by feature. The batch run's score can differ: it
        nudges a pair toward the previous run's decision (the stability margin) and applies
        overrides. That difference - from the run's link, or `score` (a review pair's) - is shown
        as a last step, so the waterfall adds up to the score the steward sees."""
        try:
            e = matcher().explain(a, b)
        except KeyError as err:
            raise HTTPException(404, str(err)) from None
        except NotImplementedError as err:
            raise HTTPException(501, str(err)) from None
        if score is None and current() is not None:
            (sa, _, ra), (sb, _, rb) = a.partition(":"), b.partition(":")
            with eng.connect() as c:
                score = c.execute(select(links.c.score).where(links.c.run_id == current()["id"]).where(
                    ((links.c.source_a == sa) & (links.c.record_a == ra) & (links.c.source_b == sb) & (links.c.record_b == rb))
                    | ((links.c.source_a == sb) & (links.c.record_a == rb) & (links.c.source_b == sa) & (links.c.record_b == ra))
                )).scalar()
        if score is not None and "contributions" in e and abs(score - e["score"]) >= 0.001:
            e["model_score"] = e["score"]
            e["adjustment"] = {"feature": "the batch run's adjustment (stability: kept near the previous run's decision; "
                                          "overrides)", "value": "", "contribution": round(score - e["score"], 4)}
            e["score"] = round(score, 4)
        return e

    def link_stats(run):
        with eng.connect() as c:
            lk = pd.read_sql(select(links.c.customer_id, links.c.score, links.c.is_bridge)
                             .where(links.c.run_id == run["id"]), c)
            cu = pd.read_sql(select(customers.c.customer_id, customers.c.record_count)
                             .where(customers.c.run_id == run["id"]).where(customers.c.record_count > 1), c)
        g = lk.groupby("customer_id").agg(links=("score", "size"), weakest=("score", "min"),
                                          bridges=("is_bridge", "sum"))
        out = cu.set_index("customer_id").join(g, how="left").fillna({"links": 0, "bridges": 0})
        n = out.record_count
        out["density"] = (out.links / (n * (n - 1) / 2)).round(3)
        return out

    @app.get("/customers", dependencies=[read])
    def list_customers(order: str = "size", limit: int = Query(20, ge=0, le=200)):
        """Customers of two or more records to inspect: the largest, those with the weakest link,
        the most bridge links (a single link holding parts together), the sparsest links, or at random."""
        need_run()
        st = cached("link_stats", link_stats)
        if order == "size":
            pick = st.sort_values(["record_count", "links"], ascending=[False, True])
        elif order == "weakest":
            pick = st.sort_values("weakest")
        elif order == "bridges":
            pick = st.sort_values(["bridges", "record_count"], ascending=False)
        elif order == "sparse":
            pick = st[st.record_count >= 3].sort_values("density")
        elif order == "random":
            pick = st.sample(min(limit, len(st)))
        else:
            raise HTTPException(400, "order must be size, weakest, bridges, sparse or random")
        pick = pick.head(limit)
        return {"customers_with_several_records": int(len(st)),
                "customers": [{"customer_id": str(cid), "records": int(r.record_count), "links": int(r.links),
                               "weakest": None if r.weakest != r.weakest else round(float(r.weakest), 4),
                               "bridges": int(r.bridges), "density": None if r.density != r.density else float(r.density)}
                              for cid, r in pick.iterrows()]}

    @app.get("/customers/{customer_id}/graph", dependencies=[read])
    def graph(customer_id: int):
        run = need_run()
        m = matcher()
        now, batch = membership(customer_id)
        if not now:
            raise HTTPException(404, "customer not found")
        if len(now) > 200:
            raise HTTPException(413, "a customer of more than 200 records is too large to draw")
        members = set(now)
        with eng.connect() as c:
            olds = {r for (r,) in c.execute(select(record_customer.c.customer_id).where(record_customer.c.run_id == run["id"])
                                            .where(record_customer.c.source.in_({k.partition(":")[0] for k in now}))
                                            .where(record_customer.c.record_id.in_([k.partition(":")[2] for k in now])))}
            edges = c.execute(select(links).where(links.c.run_id == run["id"])
                              .where(links.c.customer_id.in_(list(olds) or [-1]))).mappings().all()
        key = lambda e, x: f"{e['source_' + x]}:{e['record_' + x]}"  # noqa: E731
        edges = [e for e in edges if key(e, "a") in members and key(e, "b") in members]
        return {"customer_id": str(customer_id),
                "nodes": [brief(m, k) | {"provisional": k not in batch} for k in now],
                "edges": [{"a": f"{e['source_a']}:{e['record_a']}", "b": f"{e['source_b']}:{e['record_b']}",
                           "score": e["score"], "is_bridge": bool(e["is_bridge"]), "evidence": e["evidence"]}
                          for e in edges]}

    @app.get("/links", dependencies=[read])
    def browse_links(min: float = 0.0, max: float = 1.0, limit: int = Query(20, ge=0, le=200)):
        """A random sample of the run's accepted links scoring between min and max - to look at
        what the model accepts at each level of confidence."""
        run = need_run()
        m = matcher()
        q = select(links).where(links.c.run_id == run["id"]).where(links.c.score >= min).where(links.c.score <= max)
        with eng.connect() as c:
            total = c.execute(select(func.count()).select_from(q.subquery())).scalar()
            rows = c.execute(q.order_by(func.random()).limit(limit)).mappings().all()
        return {"total": int(total), "threshold": m.model["threshold"],
                "links": [{"a": brief(m, f"{r['source_a']}:{r['record_a']}"),
                           "b": brief(m, f"{r['source_b']}:{r['record_b']}"),
                           "score": r["score"], "customer_id": str(r["customer_id"]), "is_bridge": bool(r["is_bridge"]),
                           "evidence": r["evidence"]} for r in rows]}

    def unlinkables(m, sample=2000):
        """Splink's unlinkables chart: each record scored against an exact copy of itself. A record
        whose copy scores at or below the threshold has too little information ever to be linked
        (a first name alone, say). Returns the share at each threshold, from a random sample."""
        n = len(m.rec)
        if not n:
            return None
        idx = np.random.default_rng(0).choice(n, min(sample, n), replace=False)
        part = m.rec.iloc[np.sort(idx)].reset_index(drop=True)
        mini = pd.concat([part, part], ignore_index=True)
        k = len(part)
        try:
            X = m._features(mini, np.column_stack([np.arange(k), np.arange(k) + k]))
        except NotImplementedError:
            return None
        p = small_proba(m.model["forest"], X)
        grid = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
        return {"sample": int(k), "threshold": m.model["threshold"],
                "share_at_threshold": round(100 * float((p <= m.model["threshold"]).mean()), 2),
                "curve": {f"{t:g}": round(100 * float((p <= t).mean()), 2) for t in grid}}

    def quality_of(run):
        art = Path(run["artifact_uri"])
        read_json = lambda f: json.loads((art / f).read_text()) if (art / f).exists() else {}  # noqa: E731
        summary, monitor, blocking = read_json("summary.json"), read_json("monitor.json"), read_json("blocking_report.json")
        m = matcher()
        forest = m.model["forest"]
        names = m.feature_names()
        importance = []
        if hasattr(forest, "feature_importances_"):
            importance = sorted(({"feature": names[k] if k < len(names) else f"feature {k}", "importance": round(float(v), 4)}
                                 for k, v in enumerate(forest.feature_importances_)), key=lambda e: -e["importance"])
        rec = m.rec
        fields = [f for f in ("first_name", "last_name", "date_of_birth", "email", "phone", "addr_street", "postcode")
                  if f in rec.columns]
        fill = {src: {f: round(100 * float((g[f] != "").mean()), 1) for f in fields}
                for src, g in rec.groupby("source", sort=True)}
        common, profile = {}, {}
        for f in fields:
            filled = rec[f][rec[f] != ""]
            vc = filled.value_counts()
            common[f] = [{"value": str(v), "records": int(n)} for v, n in vc.head(8).items()]
            profile[f] = {"filled": int(len(filled)), "distinct": int(len(vc)),
                          "top_value_share": round(100 * float(vc.iloc[0]) / len(filled), 2) if len(vc) else None,
                          "singletons": round(100 * float((vc == 1).sum()) / len(vc), 1) if len(vc) else None}
        # term frequency: what agreeing on a value is worth - the rarity feature, ln(records / records with it)
        n_rec = len(rec)
        rarity = {}
        for f in ("first_name", "last_name"):
            vc = rec[f][rec[f] != ""].value_counts()
            if len(vc):
                show = list(vc.head(6).items()) + list(vc.iloc[[len(vc) // 2]].items()) + list(vc.tail(2).items())
                rarity[f] = [{"value": str(v), "records": int(c), "rarity": round(float(np.log(n_rec / c)), 2)}
                             for v, c in show]
        unlinkable = unlinkables(m)
        contacts = []
        if (art / "shared_and_junk_contacts.csv").exists():
            cj = pd.read_csv(art / "shared_and_junk_contacts.csv", dtype=str, keep_default_na=False)
            contacts = cj.head(30).to_dict("records")
        return {"run": run["id"], "threshold": m.model["threshold"],
                "estimated": summary.get("cross_fitted_closure_aware"), "threshold_sweep": summary.get("threshold_sweep"),
                "score_histogram": monitor.get("score_histogram"), "pairs_scored": monitor.get("pairs_scored"),
                "match_rate_by_source": monitor.get("match_rate_by_source"),
                "records_by_source": rec.source.value_counts().to_dict(),
                "fill_rate_by_source": fill, "common_values": common, "profile": profile, "name_rarity": rarity,
                "unlinkables": unlinkable, "feature_importance": importance,
                "blocking": blocking, "shared_and_junk_contacts": contacts,
                "labelled_pairs": summary.get("labelled_pairs"), "warnings": summary.get("warnings", [])}

    @app.get("/quality", dependencies=[read])
    def quality():
        need_run()
        return cached("quality", quality_of)

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return HTMLResponse(page)

    @app.get("/overrides", dependencies=[read])
    def list_overrides(status: str | None = None, decision: str | None = None, by: str | None = None,
                       q: str | None = None, format: str = "json",  # noqa: A002
                       limit: int = Query(1000, ge=1, le=100_000), offset: int = Query(0, ge=0)):
        sel = select(overrides).order_by(overrides.c.id.desc()).limit(limit).offset(offset)
        if status:
            sel = sel.where(overrides.c.status == status)
        if decision:
            sel = sel.where(overrides.c.decision == decision)
        if by:
            sel = sel.where(overrides.c.created_by == by)
        if q:
            sel = sel.where(overrides.c.a.contains(q) | overrides.c.b.contains(q))
        with eng.connect() as c:
            rows = [dict(r) for r in c.execute(sel).mappings()]
        if format == "csv":
            buf = _io.StringIO()
            safe = lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r") else v  # noqa: E731
            pd.DataFrame([{k: safe(v) for k, v in r.items()} for r in rows],
                         columns=[col.name for col in overrides.columns]).to_csv(buf, index=False)
            return Response(buf.getvalue(), media_type="text/csv",
                            headers={"Content-Disposition": "attachment; filename=decisions.csv"})
        return rows

    @app.post("/overrides")
    def add_override(req: OverrideRequest, who=steward):
        return write(lambda m: m.decide_override(req.a, req.b, req.decision, actor(who), req.note,
                                                 dry_run=req.dry_run, propose=four_eyes))

    @app.post("/overrides/{override_id}/approve")
    def approve(override_id: int, req: ActionRequest, who=steward):
        return write(lambda m: m.approve(override_id, actor(who), req.note, dry_run=req.dry_run))

    @app.post("/overrides/{override_id}/reject")
    def reject(override_id: int, req: ActionRequest, who=steward):
        if req.dry_run:
            return {"id": override_id, "preview": True, "applied": "would reject the proposal"}
        return write(lambda m: m.revoke(override_id, actor(who), req.note, rejected=True))

    @app.delete("/overrides/{override_id}")
    def revoke(override_id: int, who=steward):
        return write(lambda m: m.revoke(override_id, actor(who), propose=four_eyes))

    class SplitRequest(ActionRequest):
        record_key: str

    @app.post("/split")
    def split(req: SplitRequest, who=steward):
        """Split a record (source:record_id - in the body, so any record id works) out of its customer."""
        return write(lambda m: m.split_out(req.record_key, actor(who), req.note, req.dry_run,
                                           propose=four_eyes))

    @app.post("/customers/{customer_id}/merge")
    def merge(customer_id: int, req: MergeRequest, who=steward):
        return write(lambda m: m.merge_customers(customer_id, req.other, actor(who), req.note,
                                                 req.dry_run, propose=four_eyes))

    class PossibleRequest(BaseModel):
        source: str
        fields: dict

    @app.post("/possible", dependencies=[read])
    def possible(req: PossibleRequest):
        """Who might this person be: details in the standard field names (first_name, last_name,
        date_of_birth, email, phone, address...), mapped to the source's own columns and scored
        by the real-time matcher. Nothing is saved, so the read role is enough."""
        m = matcher()
        src = next((x for x in m.cfg["sources"] if x["name"] == req.source), None)
        if src is None:
            raise HTTPException(400, f"unknown source {req.source!r}")
        cols = src["columns"]
        raw = {cols[k]: str(v) for k, v in req.fields.items() if k in cols and k != "record_id" and str(v).strip()}
        if not raw:
            raise HTTPException(400, "give at least one detail the source has")
        return m.match(req.source, "possible-match-lookup", raw, write=False)

    @app.get("/compare", dependencies=[read])
    def compare(a: str, b: str):
        """Any two records side by side: the score and its breakdown, whether blocking would
        compare them (why not matched), the decision on them if any."""
        m = matcher()
        try:
            out = m.compare(a, b)
        except KeyError as e:
            raise HTTPException(404, str(e).strip("'")) from None
        x, y = sorted((a, b))
        with eng.connect() as c:
            d = c.execute(select(overrides).where(overrides.c.a == x).where(overrides.c.b == y)).mappings().first()
        return out | {"a_record": brief(m, a), "b_record": brief(m, b), "decision": dict(d) if d else None}

    @app.post("/pins")
    def pin(req: PinRequest, who=steward):
        """Pin a golden-record value: it beats survivorship now and in every later run, for the
        customer holding this record."""
        if req.field not in GOLDEN and not req.field.replace("_", "").isalnum():
            raise HTTPException(400, "unknown field")
        if matcher()._index_of(req.record_key) is None:
            raise HTTPException(404, "record not found")
        with eng.begin() as c:
            c.execute(golden_pins.delete().where(golden_pins.c.record_key == req.record_key)
                      .where(golden_pins.c.field == req.field))
            c.execute(insert(golden_pins).values(record_key=req.record_key, field=req.field, value=req.value,
                                                 created_at=db.now(), created_by=actor(who), note=req.note))
            c.execute(insert(steward_log).values(at=db.now(), actor=actor(who), action="pinned",
                                                 a=req.record_key, detail=json.dumps({"field": req.field, "value": req.value}),
                                                 note=req.note))
        return {"pinned": req.field, "value": req.value}

    @app.delete("/pins/{pin_id}")
    def unpin(pin_id: int, who=steward):
        with eng.begin() as c:
            row = c.execute(select(golden_pins).where(golden_pins.c.id == pin_id)).mappings().first()
            if row is None:
                raise HTTPException(404, "pin not found")
            c.execute(golden_pins.delete().where(golden_pins.c.id == pin_id))
            c.execute(insert(steward_log).values(at=db.now(), actor=actor(who), action="unpinned",
                                                 a=row["record_key"], detail=json.dumps({"field": row["field"],
                                                                                         "value": row["value"]})))
        return {"unpinned": row["field"]}

    @app.get("/stewards", dependencies=[read])
    def stewards(days: int = Query(30, ge=1, le=3650)):
        """Steward activity: actions per steward, how often their decisions overturned the
        model (same on a pair it scored below the threshold, different on one above), and
        decisions per day."""
        since = db.now() - timedelta(days=days)
        with eng.connect() as c:
            log = pd.read_sql(select(steward_log.c.at, steward_log.c.actor, steward_log.c.action, steward_log.c.detail)
                              .where(steward_log.c.at >= since), c)
            pending = c.execute(select(func.count()).select_from(overrides).where(overrides.c.status == "proposed")).scalar()
        th = matcher().model["threshold"]
        out = []
        if len(log):
            log["actor"] = log.actor.fillna("(no key)")
            det = [(_json(d) or {}) for d in log.detail]
            log["decision"] = [d.get("decision") for d in det]
            log["model_score"] = [d.get("model_score") for d in det]
            for actor, g in log.groupby("actor"):
                dec = g[g.action.isin(["decided", "proposed"])]
                scored = dec[dec.model_score.notna()]
                over = ((scored.decision == "same") & (scored.model_score <= th)) | \
                       ((scored.decision == "different") & (scored.model_score > th))
                out.append({"steward": actor, "decisions": int(len(dec)), "same": int((dec.decision == "same").sum()),
                            "different": int((dec.decision == "different").sum()),
                            "overturned_model": round(100 * float(over.mean()), 1) if len(scored) else None,
                            "approved": int((g.action == "approved").sum()), "revoked": int((g.action == "revoked").sum()),
                            "splits": int((g.action == "split").sum()), "pins": int((g.action == "pinned").sum()),
                            "last": str(g["at"].max())})
            log["day"] = pd.to_datetime(log["at"], utc=True).dt.strftime("%Y-%m-%d")
            per_day = log[log.action.isin(["decided", "proposed"])].groupby("day").size().to_dict()
        else:
            per_day = {}
        return {"days": days, "four_eyes": four_eyes, "awaiting_approval": int(pending), "stewards": out,
                "decisions_per_day": per_day}

    @app.post("/match", dependencies=[match_role])
    def match(req: MatchRequest):
        try:
            return write(lambda m: m.match(req.source, req.record_id, req.fields, req.write))
        except HTTPException as e:
            if e.status_code == 404:                     # an unknown source is the caller's mistake
                raise HTTPException(400, e.detail) from None
            raise

    return app
