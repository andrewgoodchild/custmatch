"""Validate the custmatch tool end to end on the synthetic customer file, as a user would run it.

For each labelling budget: `custmatch sample` draws candidate pairs, the ground truth fills
their labels (standing in for a person), `custmatch run` matches, and the output clusters are
scored against the true identities and households - on all records, and on the records of
people who appear in no labelled pair (no leakage from training or threshold choice).
`--jev` also runs the Jev route: 300 truth-labelled check pairs gate Jev, which labels 2,000.
`--active` reaches 500 and 1,000 labels by active learning after 300 random ones.

Needs data/customer from customer_data.py. Writes results/custmatch_validation.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import labels, pipeline
from custmatch.core import partition_scores

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_labelling  # noqa: E402  (the LLM labelling researched in report chapter 4)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/synthetic/config.json"
DATA = ROOT / "data/customer"


def truth():
    t = pd.concat([pd.read_parquet(DATA / f"source_{s}.parquet").assign(source=s) for s in ("crm", "billing")])
    return dict(zip(t.source + ":" + t.record_id.astype(str), zip(t.simulant_id, t.household_id)))


def fill(path, gold):
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    lab["label"] = [int(gold[a][0] == gold[b][0]) for a, b in zip(lab.a, lab.b)]
    lab.to_csv(path, index=False)
    return lab


def score_clusters(out_dir, gold, labelled):
    """score() for outputs without a summary.json of the batch kind (incremental add)."""
    return {k: v for k, v in score(out_dir, gold, labelled, summary=False).items()}


def score(out_dir, gold, labelled, summary=True):
    c = pd.read_csv(out_dir / "clusters.csv", dtype=str)
    k = (c.source + ":" + c.record_id).to_numpy()
    sid = np.array([gold[x][0] for x in k]); hh = np.array([gold[x][1] for x in k])
    seen = set()
    for a, b in zip(labelled.a, labelled.b):
        seen.update((gold[a][0], gold[b][0]))
    unseen = np.array([s not in seen for s in sid])

    def n_true(ids):
        _, n = np.unique(ids, return_counts=True)
        return float(np.sum(n * (n - 1) / 2))
    return {
        "all_records": partition_scores(c.customer_id.to_numpy(), sid, n_true(sid)),
        "unlabelled_people_only": partition_scores(c.customer_id.to_numpy()[unseen], sid[unseen], n_true(sid[unseen])),
        "households": partition_scores(c.household_id.to_numpy(), hh, n_true(hh)),
        **({"summary": json.loads((out_dir / "summary.json").read_text())} if summary else {})}


def main():
    gold = truth()
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for n in (500, 1000, 3000):
            lp = tmp / f"labels_{n}.csv"
            labels.sample(CONFIG, n, lp, seed=1)
            lab = fill(lp, gold)
            pipeline.run(CONFIG, lp, tmp / f"out_{n}")
            res[f"hand_{n}"] = score(tmp / f"out_{n}", gold, lab)
            r = res[f"hand_{n}"]
            print(f"{n:5d} hand labels: clustered F1 all {r['all_records']['f1']:.2f}, unlabelled people "
                  f"{r['unlabelled_people_only']['f1']:.2f}, households {r['households']['f1']:.2f}, "
                  f"threshold {r['summary']['threshold']}", flush=True)
        if "--active" in sys.argv:
            # same totals as the random budgets, but after 300 random labels the rest are
            # chosen 100 at a time by uncertainty, retraining each round
            for total in (500, 1000):
                lp = tmp / f"active_{total}.csv"
                labels.sample(CONFIG, 300, lp, seed=1)
                lab = fill(lp, gold)
                while len(lab) < total:
                    nxt = tmp / "next.csv"
                    labels.sample_active(CONFIG, lp, min(100, total - len(lab)), nxt)
                    lab = pd.concat([lab, fill(nxt, gold)], ignore_index=True)
                    lab.to_csv(lp, index=False)
                pipeline.run(CONFIG, lp, tmp / f"out_active_{total}", log=lambda *_: None)
                res[f"active_{total}"] = score(tmp / f"out_active_{total}", gold, lab)
                r = res[f"active_{total}"]
                print(f"{total:5d} labels (300 random + {total - 300} active): clustered F1 all "
                      f"{r['all_records']['f1']:.2f}, unlabelled people {r['unlabelled_people_only']['f1']:.2f}, "
                      f"households {r['households']['f1']:.2f}, threshold {r['summary']['threshold']}", flush=True)
        if "--active-seeds" in sys.argv:
            # is the 1,000-label gain from active learning stable across label draws?
            rows = []
            for seed in (2, 3, 4):
                rp, ap = tmp / f"r{seed}.csv", tmp / f"a{seed}.csv"
                labels.sample(CONFIG, 1000, rp, seed=seed)
                rl = fill(rp, gold)
                pipeline.run(CONFIG, rp, tmp / f"or{seed}", log=lambda *_: None)
                labels.sample(CONFIG, 300, ap, seed=seed)
                al = fill(ap, gold)
                while len(al) < 1000:
                    nxt = tmp / "next.csv"
                    labels.sample_active(CONFIG, ap, 100, nxt, seed=seed)
                    al = pd.concat([al, fill(nxt, gold)], ignore_index=True)
                    al.to_csv(ap, index=False)
                pipeline.run(CONFIG, ap, tmp / f"oa{seed}", log=lambda *_: None)
                r = score(tmp / f"or{seed}", gold, rl)["all_records"]["f1"]
                a_ = score(tmp / f"oa{seed}", gold, al)["all_records"]["f1"]
                rows.append({"seed": seed, "random_1000": r, "active_1000": a_})
                print(f"seed {seed}: 1,000 random {r:.2f}  vs  300 random + 700 active {a_:.2f}", flush=True)
            res["active_vs_random_1000_seeds"] = rows
        if "--jev" in sys.argv:
            check = tmp / "check.csv"
            labels.sample(CONFIG, 300, check, seed=2)
            fill(check, gold)
            gate = jev_labelling.label(CONFIG, 2000, check, tmp / "jev_labels.csv", max_cost=0.30)
            lab = pd.read_csv(tmp / "jev_labels.csv", dtype=str, keep_default_na=False)
            jl = lab[lab.by == "jev"]
            gate["jev_label_accuracy_vs_truth"] = round(100 * float(np.mean(
                [int(gold[a][0] == gold[b][0]) == int(y) for a, b, y in zip(jl.a, jl.b, jl.label)])), 2)
            pipeline.run(CONFIG, tmp / "jev_labels.csv", tmp / "out_jev")
            res["jev_2000_gated_on_300"] = score(tmp / "out_jev", gold, lab) | {"gate": gate}
            r = res["jev_2000_gated_on_300"]
            print(f"Jev route: gate {gate}\n  clustered F1 all {r['all_records']['f1']:.2f}, unlabelled people "
                  f"{r['unlabelled_people_only']['f1']:.2f}, households {r['households']['f1']:.2f}", flush=True)
    path = ROOT / "results/custmatch_validation.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | res, indent=1, default=float))


if __name__ == "__main__":
    main()
