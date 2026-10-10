"""Compare threshold rules offline on what scripts/threshold_capture.py saved.

custmatch picks the threshold with the best estimated clustered F1: precision from the
in-cluster check (coherence), recall from the held-out random labels. With few random matches
the recall estimate is blind across a range of thresholds - with 8 matches every threshold from
0.25 to 0.85 read 100% - while the precision estimate creeps up, so the pick drifts to the strict
end and real recall is lost (NCVR, 200,000 people, draw 3: chosen 0.85, true recall 75%).

Rules compared, each scored by the true clustered F1 at the threshold it picks and its regret
against the best threshold in hindsight:

  current        the best estimate; ties to the stricter threshold
  middle_<c>     every threshold whose estimate is within c x (100 / random matches) F1 points of
                 the best - what the labels cannot tell apart, c matches' worth - and the middle
                 one of those
  lowest_<c>     the same set, its least strict threshold

The development cases (NCVR, the Australian file, the synthetic file) were used to choose the
rule; the public benchmarks (febrl3, febrl4, fake_1000, historical_50k; random and active
labels, three draws) were captured afterwards and held out to check it.

    python scripts/threshold_rules.py

Writes results/threshold_rules.json.
"""
import json
from pathlib import Path

import numpy as np

from custmatch.core import components, partition_scores

ROOT = Path(__file__).resolve().parents[1]
CAP = ROOT / "data/threshold_capture"
OUT = ROOT / "results/threshold_rules.json"


def estimates(d):
    """custmatch's estimated F1 per threshold, as choose_threshold computes it."""
    tp, fp, fn, coh = d["tp"].astype(float), d["fp"].astype(float), d["fn"].astype(float), d["coherence"]
    r = np.where(tp + fn > 0, 100 * tp / np.maximum(tp + fn, 1), 0.0)
    p = 100 * coh
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-9), 0.0)


def true_f1(d):
    pairs, prob, pid = d["pairs"], d["prob"], d["pid"]
    _, n = np.unique(pid, return_counts=True)
    tot = float(np.sum(n * (n - 1) / 2))
    return np.array([partition_scores(components(len(pid), pairs, prob > t), pid, tot)["f1"] for t in d["grid"]])


def rule(name, grid, est, k):
    best = est.max()
    if name == "current":
        return grid[max(i for i in range(len(grid)) if est[i] == best)]
    kind, c = name.split("_")
    tol = float(c) * 100.0 / max(k, 1)
    near = [i for i in range(len(grid)) if est[i] >= best - tol]
    if kind == "middle":
        return grid[near[len(near) // 2]]
    return grid[near[0]]


RULES = ["current", "middle_0.5", "middle_1", "middle_2", "lowest_0.5", "lowest_1"]


def main():
    res = {}
    for f in sorted(CAP.glob("*.npz")):
        d = np.load(f)
        grid, est, tf = d["grid"], estimates(d), true_f1(d)
        k = int(d["tp"][0] + d["fn"][0])
        row = {"random_matches": k, "best_possible": float(tf.max()), "best_threshold": float(grid[tf.argmax()]),
               "chosen_in_run": float(d["chosen"])}
        for r in RULES:
            t = rule(r, grid, est, k)
            row[r] = {"threshold": float(t), "f1": float(tf[list(grid).index(t)]), "regret": float(tf.max() - tf[list(grid).index(t)])}
        res[f.stem] = row
        print(f.stem, k, {r: (row[r]["threshold"], round(row[r]["f1"], 2)) for r in RULES}, "best", row["best_threshold"], round(row["best_possible"], 2))
    # development cases chose the rule (middle_0.5); the public benchmarks were held out to check it
    groups = {"development": {k: v for k, v in res.items() if not k.startswith("bench_")},
              "held_out_public_benchmarks": {k: v for k, v in res.items() if k.startswith("bench_")}}
    summary = {g: {"cases": len(cs), **{r: {"mean_regret": round(float(np.mean([v[r]["regret"] for v in cs.values()])), 3),
                                          "max_regret": round(float(np.max([v[r]["regret"] for v in cs.values()])), 2),
                                          "worse_than_current": int(sum(v[r]["regret"] > v["current"]["regret"] + 1e-9 for v in cs.values())),
                                          "better_than_current": int(sum(v[r]["regret"] < v["current"]["regret"] - 1e-9 for v in cs.values()))}
                                      for r in RULES}}
               for g, cs in groups.items() if cs}
    print(json.dumps(summary, indent=1))
    OUT.write_text(json.dumps({"cases": res, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
