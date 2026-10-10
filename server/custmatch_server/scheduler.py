"""The scheduler: loads what arrives in an inbox folder, runs the batch at set times, checks health,
and alerts.

  custmatch-server schedule CONFIG --labels LABELS --inbox DIR [--batch-at 02:00 ...] [--poll 60] [--once]

Inbox: drop a source's file as <source>.<full|incremental>.<csv|parquet>, or with anything after
the mode (web.incremental.2026-10-05.csv). Files are loaded oldest first; a loaded file moves to
DIR/done/, one that fails to load to DIR/failed/ (with an alert). A file still being written is
skipped until its size has stopped changing for one poll.

Batch: at each --batch-at time (UTC, HH:MM), once a day; with --once, now. After a batch run the
health checks run (health.py), and anything not ok is posted to CUSTMATCH_ALERT_WEBHOOK.

One scheduler at a time: each pass takes the "scheduler" lease, so a second copy (a restart that
overlapped, or two replicas) waits rather than loading files twice. --once does one pass - load
the inbox, run a batch, check - and exits with the checks' status (0 ok, 1 warn, 2 fail): for an
external scheduler (cron, a Kubernetes CronJob, Airflow) instead of this loop.
"""
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from . import db, health
from .loader import load_file
from .worker import run_batch

LEASE = "scheduler"
NAME = re.compile(r"^(?P<source>[A-Za-z0-9_\-]+)\.(?P<mode>full|incremental)(\..*)?\.(csv|parquet|pq)$")


class Scheduler:
    def __init__(self, eng, config_path, labels_path, inbox, data_dir, batch_at=(), log=print):
        self.eng, self.config_path, self.labels_path = eng, Path(config_path), Path(labels_path)
        self.inbox, self.data_dir, self.log = Path(inbox), Path(data_dir), log
        self.batch_at = sorted(batch_at)
        self.holder = db.holder_name()
        self.sizes = {}                                   # file -> size at the last poll
        self.last_batch_day = {}                          # "HH:MM" -> the date it last ran
        self.cfg = json.loads(self.config_path.read_text())

    # ---------------------------------------------------------------- the inbox
    def _named_files(self):
        """Inbox files with a recognised name, oldest first."""
        if not self.inbox.exists():
            return []
        return [f for f in sorted(self.inbox.iterdir(), key=lambda p: p.stat().st_mtime)
                if f.is_file() and NAME.match(f.name)]

    def ready_files(self):
        """Inbox files with a recognised name whose size has not changed since the last poll."""
        out, sizes = [], {}
        for f in self._named_files():
            sizes[f] = f.stat().st_size
            if self.sizes.get(f) == sizes[f]:
                out.append(f)
        self.sizes = sizes
        return out

    def load_inbox(self, wait_for_stable=True):
        files = self.ready_files() if wait_for_stable else self._named_files()
        loaded = []
        for f in files:
            m = NAME.match(f.name)
            dest = self.inbox / "done"
            try:
                r = load_file(self.eng, self.cfg, m["source"], f, mode=m["mode"])
                self.log(f"loaded {f.name}: {json.dumps(r)}")
                loaded.append(r)
            except Exception as e:  # noqa: BLE001 - a bad file must not stop the scheduler
                dest = self.inbox / "failed"
                self.log(f"failed to load {f.name}: {type(e).__name__}: {e}")
                health.alert([{"check": "loads", "status": "fail", "detail": f"{f.name}: {type(e).__name__}: {str(e)[:200]}"}],
                             "custmatch: a load failed")
            dest.mkdir(exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            shutil.move(str(f), str(dest / f"{stamp}-{f.name}"))
            self.sizes.pop(f, None)
        return loaded

    # ---------------------------------------------------------------- the batch run
    def batch_due(self, now=None):
        """A scheduled time has passed today and no run has started since it. Asked of the
        database, not this process's memory: a restart after 02:00 does not run the 02:00 batch
        again, and a scheduler that was down at 02:00 catches up when it starts."""
        from sqlalchemy import func, select

        from .schema import runs
        now = now or datetime.now(timezone.utc)
        with self.eng.connect() as c:
            last = c.execute(select(func.max(runs.c.created_at))).scalar()
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        for at in self.batch_at:
            hh, mm = map(int, at.split(":"))
            slot = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if now >= slot and (last is None or last < slot) and self.last_batch_day.get(at) != now.date():
                self.last_batch_day[at] = now.date()       # (and once per process, if the run never starts)
                return True
        return False

    def batch(self):
        try:
            run_id, summary = run_batch(self.eng, self.config_path, self.labels_path, self.data_dir, log=self.log)
            title = f"custmatch: run {run_id} published ({summary['customers']:,} customers)"
        except Exception as e:  # noqa: BLE001
            self.log(f"batch run failed: {type(e).__name__}: {e}")
            title = "custmatch: a batch run failed"
        results = health.checks(self.eng)
        status = health.worst(results)
        self.log(f"health: {status}; " + "; ".join(f"{r['check']} {r['status']}" for r in results))
        health.alert(results, title)
        return status

    # ---------------------------------------------------------------- one pass, and the loop
    def once(self):
        """Load the inbox (every recognised file, now), run a batch, check: one pass."""
        if not db.take_lease(self.eng, LEASE, self.holder, seconds=3600):
            self.log("another scheduler is running and holds the lease: nothing done (stop it, or let it do the work)")
            return "warn"
        try:
            self.load_inbox(wait_for_stable=False)
            return self.batch()
        finally:
            db.release_lease(self.eng, LEASE, self.holder)

    def loop(self, poll=60):
        """Until stopped. A stop (SIGTERM from docker stop or Kubernetes, or Ctrl-C) gives the lease
        up, so a --once pass or a new scheduler can start straight away."""
        import signal

        def stop(*_):
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, stop)
        self.log(f"scheduler {self.holder}: inbox {self.inbox}, batch at {', '.join(self.batch_at) or 'never'} UTC")
        try:
            while True:
                if db.take_lease(self.eng, LEASE, self.holder, seconds=max(3 * poll, 300)):
                    self.load_inbox()
                    if self.batch_due():
                        self.batch()
                time.sleep(poll)
        finally:
            db.release_lease(self.eng, LEASE, self.holder)
            self.log("scheduler stopped; lease released")
