"""Health checks for an operator: is the service doing its job, and has anything changed sharply?

Each check is ok, warn or fail, with a sentence saying why:

  run           a published run exists; the latest run did not fail
  run_age       the current run is newer than CUSTMATCH_MAX_RUN_AGE_HOURS (default 36)
  customers     the customer count moved by less than CUSTMATCH_MAX_CUSTOMER_CHANGE % (default 5)
                since the previous run - a sharp jump means a wave of merges or splits worth a look
  review        the review queue is under CUSTMATCH_MAX_REVIEW (default 1000) and not growing fast
  loads         no load failed in the last day, and none has been running for over two hours
  realtime      real-time records waiting for a batch run are younger than the run age limit

GET /status, `custmatch-server check` (exit 0 ok, 1 warn, 2 fail) and the scheduler's alerts
(CUSTMATCH_ALERT_WEBHOOK) all use these.
"""
import json
import os
import urllib.request
from datetime import timedelta
from pathlib import Path

from sqlalchemy import func, select

from . import db
from .schema import loads, realtime_records, runs


def _env(name, default):
    return float(os.environ.get(name, default))


def _summary_file(run):
    path = Path(run["artifact_uri"]) / "summary.json" if run and run["artifact_uri"] else None
    return json.loads(path.read_text()) if path and path.exists() else {}


def _aware(t):
    from datetime import timezone
    return t if t is None or t.tzinfo else t.replace(tzinfo=timezone.utc)


def checks(eng):
    now = db.now()
    max_age = timedelta(hours=_env("CUSTMATCH_MAX_RUN_AGE_HOURS", 36))
    out = []

    def add(name, status, detail):
        out.append({"check": name, "status": status, "detail": detail})

    with eng.connect() as c:
        cur = db.current(c)
        latest = c.execute(select(runs).order_by(runs.c.id.desc()).limit(1)).mappings().first()
        published = c.execute(select(runs).where(runs.c.status.in_(("published", "retired")))
                              .order_by(runs.c.id.desc()).limit(2)).mappings().all()
        failed_loads = c.execute(select(func.count()).select_from(loads).where(loads.c.status == "failed")
                                 .where(loads.c.started_at >= now - timedelta(days=1))).scalar()
        stuck_loads = c.execute(select(func.count()).select_from(loads).where(loads.c.status == "running")
                                .where(loads.c.started_at < now - timedelta(hours=2))).scalar()
        oldest_pending = c.execute(select(func.min(realtime_records.c.created_at))
                                   .where(realtime_records.c.reconciled_run.is_(None))).scalar()
    # the run
    if cur is None:
        add("run", "fail", "no published run: load records and run a batch")
    elif latest["status"] == "failed":
        add("run", "fail", f"run {latest['id']} failed: {str(latest['error'])[:200]}")
    elif latest["status"] == "running" and _aware(latest["created_at"]) < now - timedelta(hours=6):
        add("run", "warn", f"run {latest['id']} has been running for over six hours")
    else:
        add("run", "ok", f"run {cur['id']} is current")
    if cur is not None:
        age = now - _aware(cur["published_at"])
        add("run_age", "warn" if age > max_age else "ok",
            f"the current run is {age.total_seconds() / 3600:.1f} hours old (limit {max_age.total_seconds() / 3600:g})")
    # customers: a sharp change since the previous run
    if len(published) == 2:
        s_new, s_old = (db.as_json(r["summary"]) for r in published)
        a, b = s_old.get("customers"), s_new.get("customers")
        if a and b is not None:
            change = 100 * (b - a) / a
            limit = _env("CUSTMATCH_MAX_CUSTOMER_CHANGE", 5)
            records = (s_new.get("records") or 0) - (s_old.get("records") or 0)
            add("customers", "warn" if abs(change) > limit else "ok",
                f"customers {a:,} -> {b:,} ({change:+.1f}%) with {records:+,} records since run {published[1]['id']}"
                + (f"; over the {limit:g}% limit - look at the run's merges and splits" if abs(change) > limit else ""))
    # the review queue
    if cur is not None:
        now_q = _summary_file(cur).get("review_pairs")
        prev_q = _summary_file(published[1]).get("review_pairs") if len(published) == 2 else None
        limit = _env("CUSTMATCH_MAX_REVIEW", 1000)
        if now_q is not None:
            growing = prev_q is not None and now_q > 1.5 * prev_q + 20
            add("review", "warn" if now_q > limit or growing else "ok",
                f"{now_q:,} pairs to review" + (f" (was {prev_q:,})" if prev_q is not None else "")
                + (" - growing fast" if growing else "") + (f"; over the {limit:g} limit" if now_q > limit else ""))
    # loads
    if failed_loads or stuck_loads:
        add("loads", "warn", f"{failed_loads} load(s) failed in the last day; {stuck_loads} running for over two hours")
    else:
        add("loads", "ok", "no failed or stuck loads")
    # real-time records waiting for a batch run
    if oldest_pending is not None:
        wait = now - _aware(oldest_pending)
        add("realtime", "warn" if wait > max_age else "ok",
            f"the oldest real-time record has waited {wait.total_seconds() / 3600:.1f} hours for a batch run")
    return out


def worst(results):
    order = {"ok": 0, "warn": 1, "fail": 2}
    return max((r["status"] for r in results), key=order.get, default="ok")


def alert(results, title, url=None):
    """POST the checks that are not ok to the webhook (Slack-compatible: {"text": ...}, plus the
    checks as JSON). Returns True if sent. Never raises: an alert failing must not fail a run."""
    url = url or os.environ.get("CUSTMATCH_ALERT_WEBHOOK")
    bad = [r for r in results if r["status"] != "ok"]
    if not url or not bad:
        return False
    text = title + "\n" + "\n".join(f"[{r['status']}] {r['check']}: {r['detail']}" for r in bad)
    body = json.dumps({"text": text, "checks": results}).encode()
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).close()
        return True
    except Exception:  # noqa: BLE001
        return False
