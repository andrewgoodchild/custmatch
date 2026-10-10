"""Replay test: real-time matching one record at a time, against a batch run over the same records.

A batch run over part of the demo; the rest arrives through the matcher one record at a time (in a
shuffled order); then a batch run over everything. Measured: how closely the real-time grouping
agrees with that batch run (pairwise F1 of one partition against the other), and each one's
pairwise F1 against the demo's true people (person_id). The real-time path decides each record
once, against what it has seen so far, so it should trail the batch run - by how much is the point.

    python server/replay_test.py [--first 0.5] [--seed 0]     (SQLite, a temporary database)
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from custmatch import io, labels  # noqa: E402
from custmatch.core import partition_scores  # noqa: E402
from custmatch_server import db  # noqa: E402
from custmatch_server.loader import load_file  # noqa: E402
from custmatch_server.matcher import Matcher  # noqa: E402
from custmatch_server.worker import run_batch  # noqa: E402

DEMO = ROOT / "examples/demo"


def pair_f1(pred, truth):
    """Pairwise F1 (as a percentage) of a partition against a reference partition."""
    truth = np.asarray(truth)
    _, n = np.unique(truth, return_counts=True)
    return partition_scores(np.asarray(pred), truth, float(np.sum(n * (n - 1) / 2)))["f1"]


def _pr(pred, truth):
    """Pairwise precision and recall (percentages) against the true people."""
    truth = np.asarray(truth)
    _, n = np.unique(truth, return_counts=True)
    s = partition_scores(np.asarray(pred), truth, float(np.sum(n * (n - 1) / 2)))
    return {"precision": s["precision"], "recall": s["recall"]}


def _labels(config, out, truth, seed, n=400):
    labels.sample(config, n, out, seed=seed)
    lab = pd.read_csv(out, dtype=str, keep_default_na=False)
    lab["label"] = [int(truth[x] == truth[y]) for x, y in zip(lab.a, lab.b)]
    lab.to_csv(out, index=False)


def replay(first=0.5, seed=0, merge=None, coherence=None, log=lambda *_: None):
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = json.loads((DEMO / "config.json").read_text())
        rng = np.random.default_rng(seed)
        later, truth = [], {}
        for s in cfg["sources"]:
            d = pd.read_csv(DEMO / s["path"], dtype=str, keep_default_na=False)
            rid = s["columns"]["record_id"]
            truth.update({f"{s['name']}:{r}": p for r, p in zip(d[rid], d.person_id)})
            now = rng.random(len(d)) < first
            d[now].to_csv(tmp / s["path"], index=False)
            later += [(s["name"], row[rid], row) for row in d[~now].to_dict("records")]
            d.to_csv(tmp / f"all_{s['path']}", index=False)
        order = rng.permutation(len(later))
        later = [later[k] for k in order]
        (tmp / "config.json").write_text(json.dumps(cfg))
        keys_now = set(io.key(io.load(tmp / "config.json")[0]))
        # labels for the records each batch run holds, answered from the true people
        _labels(tmp / "config.json", tmp / "labels.csv", truth, seed)
        all_cfg = dict(cfg, sources=[dict(s, path=f"all_{s['path']}") for s in cfg["sources"]])
        (tmp / "config_all.json").write_text(json.dumps(all_cfg))
        _labels(tmp / "config_all.json", tmp / "labels_all.csv", truth, seed)

        eng = db.engine_from_env(f"sqlite:///{tmp / 'replay.db'}")
        for s in cfg["sources"]:
            load_file(eng, cfg, s["name"], tmp / s["path"], mode="full")
        run_batch(eng, tmp / "config.json", tmp / "labels.csv", tmp / "data", log=log)
        with eng.connect() as c:
            m = Matcher(eng, db.current(c))
        if merge:
            m.merge = merge
        if coherence is not None:
            m.coherence = coherence
        for source, rid, row in later:                       # one at a time, as they arrive
            m.match(source, rid, row, write=True)
        realtime = {m._key(i): m._customer(i) for i in range(len(m))}

        # the batch run over everything (the realtime records are already in the database; the
        # files are loaded again to be sure every record is live)
        for s in cfg["sources"]:
            load_file(eng, cfg, s["name"], tmp / f"all_{s['path']}", mode="full")
        _, summary = run_batch(eng, tmp / "config.json", tmp / "labels_all.csv", tmp / "data", log=log)
        with eng.connect() as c:
            art = Path(db.current(c)["artifact_uri"])
        cl = pd.read_csv(art / "clusters.csv", dtype=str)
        batch = dict(zip(cl.source + ":" + cl.record_id, cl.customer_id))
        keys = sorted(batch)
        assert set(keys) == set(realtime), "the two runs hold different records"
        rt = [realtime[k] for k in keys]
        bt = [batch[k] for k in keys]
        tr = [truth[k] for k in keys]
        arrived = [k for k in keys if k not in keys_now]
        out = {"records": len(keys), "in_first_batch": len(keys_now), "arrived_one_at_a_time": len(arrived),
               "agreement_f1": pair_f1(rt, bt),
               "realtime_f1_vs_truth": pair_f1(rt, tr), "batch_f1_vs_truth": pair_f1(bt, tr),
               **{f"realtime_{k}": v for k, v in _pr(rt, tr).items()},
               "customers_realtime": len(set(rt)), "customers_batch": len(set(bt)),
               "customers_true": len(set(tr))}
        eng.dispose()
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def all_runs(seeds=(0, 1, 2, 3, 4)):
    """results/server_replay.json: each merge policy, and "strong" without the coherence check."""
    import time

    from custmatch_server.matcher import COHERENCE
    out = {"what": "server/replay_test.py --all on the demo: a batch run over half the records, the other "
                   "half through the real-time matcher one at a time, then a batch run over everything; "
                   "pairwise F1", "runs": []}
    for merge, coherence in (("strong", COHERENCE), ("strong", 0), ("strong", 0.51), ("always", COHERENCE),
                             ("never", COHERENCE)):
        for seed in seeds:
            t0 = time.time()
            r = replay(seed=seed, merge=merge, coherence=coherence)
            out["runs"].append({"merge": merge, "coherence": coherence, "seed": seed, **r,
                                "seconds": round(time.time() - t0, 1)})
            print(merge, coherence, seed, r["realtime_f1_vs_truth"], r["batch_f1_vs_truth"], flush=True)
        runs = [x for x in out["runs"] if x["merge"] == merge and x["coherence"] == coherence]
        out[f"mean_{merge}_coherence_{coherence}"] = {
            k: round(sum(x[k] for x in runs) / len(runs), 2)
            for k in ("agreement_f1", "realtime_f1_vs_truth", "realtime_precision", "realtime_recall",
                      "batch_f1_vs_truth")}
    (ROOT / "results/server_replay.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", choices=["always", "strong", "never"])
    ap.add_argument("--all", action="store_true", help="every policy over five seeds: results/server_replay.json")
    a = ap.parse_args()
    if a.all:
        all_runs()
    else:
        print(json.dumps(replay(a.first, a.seed, a.merge), indent=1))
