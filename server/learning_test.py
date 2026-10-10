"""Do stewards' decisions improve the model when they also train it?

The demo with a small random label set; a steward (simulated from the demo's true people) answers
every pair in the review queue, round after round, each round followed by a batch run. Two ways:
decisions only override their own pairs, or they also train the next run's model
(worker.run_batch(decisions_as_labels=...)). Measured each round: clustered pairwise F1 against
the true people, and the queue's size.

    python server/learning_test.py [--labels 200] [--rounds 3] [--seeds 3]
Writes results/server_learning.json.
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
from custmatch import labels  # noqa: E402
from custmatch.core import partition_scores  # noqa: E402
from custmatch_server import db  # noqa: E402
from custmatch_server.loader import load_file  # noqa: E402
from custmatch_server.matcher import Matcher  # noqa: E402
from custmatch_server.worker import run_batch  # noqa: E402

DEMO = ROOT / "examples/demo"


def f1_vs_truth(result_dir, truth):
    cl = pd.read_csv(Path(result_dir) / "clusters.csv", dtype=str)
    t = np.array([truth[k] for k in cl.source + ":" + cl.record_id])
    _, n = np.unique(t, return_counts=True)
    return partition_scores(cl.customer_id.to_numpy(), t, float(np.sum(n * (n - 1) / 2)))["f1"]


def trial(n_labels, rounds, seed, learn):
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = json.loads((DEMO / "config.json").read_text())
        truth = {}
        for s in cfg["sources"]:
            d = pd.read_csv(DEMO / s["path"], dtype=str, keep_default_na=False)
            truth.update({f"{s['name']}:{r}": p for r, p in zip(d[s["columns"]["record_id"]], d.person_id)})
        labels.sample(DEMO / "config.json", n_labels, tmp / "labels.csv", seed=seed)
        lab = pd.read_csv(tmp / "labels.csv", dtype=str, keep_default_na=False)
        lab["label"] = [int(truth[a] == truth[b]) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(tmp / "labels.csv", index=False)
        eng = db.engine_from_env(f"sqlite:///{tmp / 'learn.db'}")
        for s in cfg["sources"]:
            load_file(eng, cfg, s["name"], DEMO / s["path"], mode="full")
        history = []
        for r in range(rounds + 1):
            run_batch(eng, DEMO / "config.json", tmp / "labels.csv", tmp / "data", log=lambda *_: None,
                      decisions_as_labels=learn)
            with eng.connect() as c:
                run = db.current(c)
            art = Path(run["artifact_uri"])
            review = pd.read_csv(art / "review.csv", dtype=str, keep_default_na=False)
            history.append({"round": r, "f1": f1_vs_truth(art, truth), "review_queue": len(review)})
            if r == rounds:
                break
            m = Matcher(eng, run, holder="learning-test")   # the steward answers the whole queue
            for a, b in zip(review.a, review.b):
                m.decide_override(a, b, "same" if truth[a] == truth[b] else "different", "simulated steward")
            history[-1]["decided"] = len(review)
        eng.dispose()
        return history
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    out = {"what": __doc__.split("\n\n")[1].replace("\n", " "), "labels": a.labels, "runs": []}
    for seed in range(a.seeds):
        for learn in (False, True):
            h = trial(a.labels, a.rounds, seed, learn)
            out["runs"].append({"seed": seed, "decisions_train_the_model": learn, "rounds": h})
            print(seed, "learn" if learn else "override only", [(x["f1"], x["review_queue"]) for x in h], flush=True)
    for learn in (False, True):
        rs = [r["rounds"] for r in out["runs"] if r["decisions_train_the_model"] == learn]
        out[f"mean_f1_by_round_{'learn' if learn else 'override_only'}"] = [
            round(float(np.mean([h[k]["f1"] for h in rs])), 2) for k in range(a.rounds + 1)]
    (ROOT / "results/server_learning.json").write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k.startswith("mean")})


if __name__ == "__main__":
    main()
