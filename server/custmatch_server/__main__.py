"""custmatch-server load CONFIG SOURCE FILE [--full]     load one source's records into the database
custmatch-server batch CONFIG --labels LABELS          a batch run over the loaded records, published
custmatch-server serve [--host 0.0.0.0] [--port 8000]  the API
custmatch-server key create NAME --role ROLE          an API key (shown once); roles read, match, steward, admin
custmatch-server key revoke NAME | key list
custmatch-server schedule CONFIG --labels LABELS --inbox DIR [--batch-at HH:MM ...] [--poll 60] [--once]
                                                      load the inbox, run batches on a schedule, alert (scheduler.py)
custmatch-server check                                the health checks; exit 0 ok, 1 warn, 2 fail (health.py)

A load is incremental (adds and changes records) unless --full, which also marks records missing
from the file as deleted. The database comes from CUSTMATCH_DB_URL (default sqlite:///custmatch.db;
for Postgres postgresql+psycopg://user:password@host/db); run files go under CUSTMATCH_DATA_DIR
(default ./data).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from . import db


def main(argv=None):
    ap = argparse.ArgumentParser(prog="custmatch-server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ld = sub.add_parser("load")
    ld.add_argument("config")
    ld.add_argument("source")
    ld.add_argument("file")
    ld.add_argument("--full", action="store_true")
    b = sub.add_parser("batch")
    b.add_argument("config")
    b.add_argument("--labels", required=True)
    k = sub.add_parser("key")
    k.add_argument("action", choices=["create", "revoke", "list"])
    k.add_argument("name", nargs="?")
    k.add_argument("--role", default="read")
    sc = sub.add_parser("schedule")
    sc.add_argument("config")
    sc.add_argument("--labels", required=True)
    sc.add_argument("--inbox", required=True)
    sc.add_argument("--batch-at", action="append", default=[], help="UTC HH:MM, once a day; repeat for more")
    sc.add_argument("--poll", type=int, default=60)
    sc.add_argument("--once", action="store_true", help="load the inbox, run a batch, check, exit")
    sub.add_parser("check")
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    eng = db.engine_from_env()
    if a.cmd == "load":
        from .loader import load_file
        cfg = json.loads(Path(a.config).read_text())
        print(json.dumps(load_file(eng, cfg, a.source, a.file, mode="full" if a.full else "incremental")))
    elif a.cmd == "key":
        from . import auth
        if a.action == "list":
            for r in auth.list_keys(eng):
                print(r["name"], r["role"], r["prefix"] + "...", "revoked" if r["revoked_at"] else "live")
        elif not a.name:
            ap.error("key create/revoke needs a NAME")
        elif a.action == "create":
            print(auth.create_key(eng, a.name, a.role))
        else:
            print(f"revoked {auth.revoke(eng, a.name)} key(s) of {a.name}")
    elif a.cmd == "schedule":
        from .scheduler import Scheduler
        sch = Scheduler(eng, a.config, a.labels, a.inbox, os.environ.get("CUSTMATCH_DATA_DIR", "data"), a.batch_at)
        if a.once:
            return {"ok": 0, "warn": 1, "fail": 2}[sch.once()]
        sch.loop(a.poll)
    elif a.cmd == "check":
        from . import health
        results = health.checks(eng)
        for r in results:
            print(f"{r['status']:5} {r['check']:10} {r['detail']}")
        return {"ok": 0, "warn": 1, "fail": 2}[health.worst(results)]
    elif a.cmd == "batch":
        from .worker import run_batch
        run_batch(eng, a.config, a.labels, os.environ.get("CUSTMATCH_DATA_DIR", "data"))
    else:
        import uvicorn
        from .api import create_app
        if os.environ.get("CUSTMATCH_AUTH", "on").lower() == "off" and a.host not in ("127.0.0.1", "localhost", "::1"):
            ap.error("CUSTMATCH_AUTH=off answers anyone who can reach the API: serve it on 127.0.0.1 only")
        uvicorn.run(create_app(eng), host=a.host, port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
