"""Operating the service: the scheduler (inbox loads, scheduled batch runs, one scheduler at a
time), the health checks and the alert webhook. SQLite; skipped where the server's dependencies
are absent."""
import importlib.util
import json
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples/demo"
pytestmark = pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("fastapi", "sqlalchemy")),
                                reason="pip install ./server (fastapi, sqlalchemy) to run the service tests")


@pytest.fixture()
def server_path():
    sys.path.insert(0, str(ROOT / "server"))


@pytest.fixture()
def webhook():
    """A local webhook that records what is posted to it."""
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/hook", got
    srv.shutdown()


def test_no_run_is_a_failed_check(server_path, tmp_path):
    from custmatch_server import db, health
    eng = db.engine_from_env(f"sqlite:///{tmp_path / 'h.db'}")
    results = health.checks(eng)
    assert health.worst(results) == "fail" and results[0]["check"] == "run"


def test_scheduler_pass_loads_the_inbox_runs_a_batch_and_alerts(server_path, tmp_path, webhook, monkeypatch):
    from custmatch_server import db, health
    from custmatch_server.scheduler import Scheduler
    url, got = webhook
    monkeypatch.setenv("CUSTMATCH_ALERT_WEBHOOK", url)
    monkeypatch.setenv("CUSTMATCH_MAX_REVIEW", "1")          # the demo's review queue is over this: a warning
    eng = db.engine_from_env(f"sqlite:///{tmp_path / 's.db'}")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in ("web", "app", "store"):
        (inbox / f"{name}.full.csv").write_bytes((DEMO / f"{name}.csv").read_bytes())
    (inbox / "nope.full.csv").write_text("record_id,first_name\n1,x\n")     # not a configured source
    (inbox / "notes.txt").write_text("ignored")
    sch = Scheduler(eng, DEMO / "config.json", DEMO / "labels.csv", inbox, tmp_path / "data", log=lambda *_: None)
    status = sch.once()
    done = sorted(p.name.split("-", 1)[1] for p in (inbox / "done").iterdir())
    assert done == ["app.full.csv", "store.full.csv", "web.full.csv"]
    assert [p.name.split("-", 1)[1] for p in (inbox / "failed").iterdir()] == ["nope.full.csv"]
    assert (inbox / "notes.txt").exists()
    with eng.connect() as c:
        assert db.current(c) is not None
    assert status == "warn"
    checks = {r["check"]: r["status"] for r in health.checks(eng)}
    assert checks["run"] == "ok" and checks["review"] == "warn"
    texts = [g["text"] for g in got]
    assert any("nope.full.csv" in t for t in texts), texts                  # the failed load
    assert any("published" in t and "review" in t for t in texts), texts    # the run, and the warning


def test_one_scheduler_at_a_time(server_path, tmp_path):
    from custmatch_server import db
    from custmatch_server.scheduler import Scheduler
    eng = db.engine_from_env(f"sqlite:///{tmp_path / 'l.db'}")
    first = Scheduler(eng, DEMO / "config.json", DEMO / "labels.csv", tmp_path / "in", tmp_path / "d", log=lambda *_: None)
    second = Scheduler(eng, DEMO / "config.json", DEMO / "labels.csv", tmp_path / "in", tmp_path / "d", log=lambda *_: None)
    assert db.take_lease(eng, "scheduler", first.holder, seconds=600)
    assert second.once() == "warn"                                    # waits: nothing loaded, no run
    with eng.connect() as c:
        assert db.current(c) is None
    db.release_lease(eng, "scheduler", first.holder)                  # the first stops (SIGTERM): no wait
    assert db.take_lease(eng, "scheduler", second.holder, seconds=600)


def test_batch_times_and_files_still_being_written(server_path, tmp_path):
    from custmatch_server import db
    from custmatch_server.scheduler import Scheduler
    eng = db.engine_from_env(f"sqlite:///{tmp_path / 't.db'}")
    inbox = tmp_path / "in"
    inbox.mkdir()
    sch = Scheduler(eng, DEMO / "config.json", DEMO / "labels.csv", inbox, tmp_path / "d", batch_at=["02:00"],
                    log=lambda *_: None)
    t = lambda h, m, d=5: datetime(2026, 10, d, h, m, tzinfo=timezone.utc)  # noqa: E731
    assert not sch.batch_due(t(1, 59))
    assert sch.batch_due(t(2, 0)) and not sch.batch_due(t(2, 30))       # once a day
    assert sch.batch_due(t(2, 1, d=6))
    # a run that started after today's slot (in the database: another process, or before a restart)
    from custmatch_server.schema import runs
    from sqlalchemy import insert
    with eng.begin() as c:
        c.execute(insert(runs).values(status="published", created_at=t(2, 5, d=7)))
    fresh = Scheduler(eng, DEMO / "config.json", DEMO / "labels.csv", inbox, tmp_path / "d", batch_at=["02:00"],
                      log=lambda *_: None)
    assert not fresh.batch_due(t(9, 0, d=7))                           # restarted after the run: not again
    assert fresh.batch_due(t(2, 0, d=8))                               # the next day's slot
    f = inbox / "web.incremental.2026-10-05.csv"
    pd.DataFrame({"record_id": ["1"]}).to_csv(f, index=False)
    assert sch.ready_files() == []                                      # first sight: maybe still being written
    assert sch.ready_files() == [f]                                     # same size a poll later: ready


def test_a_forest_catboost_run_serves_scores_and_importances(server_path, tmp_path):
    """The opt-in ensemble through the service's model paths: real-time scoring, the importances
    the dashboard shows, and no per-feature breakdown (that is the forest's alone)."""
    pytest.importorskip("catboost")
    import numpy as np
    from custmatch_server import worker
    from custmatch_server.matcher import small_proba

    from custmatch import pipeline
    from custmatch.models import load_model, p_match
    cfg = json.loads((DEMO / "config.json").read_text()) | {"matcher_model": "forest_catboost"}
    for s in cfg["sources"]:
        s["path"] = str(DEMO / s["path"])
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    pipeline.run(tmp_path / "config.json", DEMO / "labels.csv", tmp_path / "out", log=lambda *_: None)
    state = tmp_path / "out" / "state"
    model = load_model(state)["forest"]
    X = np.random.default_rng(0).random((5, model.models[0].n_features_in_))
    assert np.allclose(small_proba(model, X), p_match(model, X))
    imp = worker._importance(state)
    assert imp and abs(sum(imp.values()) - 1) < 0.01
    assert not hasattr(model, "estimators_")              # the dashboard shows no waterfall for it
