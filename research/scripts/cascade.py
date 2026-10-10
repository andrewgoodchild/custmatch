"""RF → Ditto cascade on person records: how much of Ditto's clustering quality survives when
Ditto only scores the pairs the forest is unsure about?

Chapter 2 argues for a cascade because Ditto scores ~25 pairs/s against the forest's instant.
Every pair gets the forest's score; a pair goes to Ditto only when lo < rf <= hi. Below lo it is
rejected outright, above hi it is accepted outright (hi = 1 means never: the forest only
rejects). Edges are then clustered by connected components exactly as in
`person_head_to_head.py`, and optionally by the bridge-aware rule from `kuzu_graph.py`.

Uses the per-pair score dumps from chapter 2 - no training, no inference:
research/results/person_scores_rf.npz, research/results/person_scores_ditto.npz.

Tuning is honest in one respect the chapter 2 sweep is not: the cascade's parameters are
chosen on one half of the test people and scored on the other (split by person, so no cluster
straddles the halves), then the halves swap. The full-test grid is reported alongside so the
optimism of picking on test is visible.
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from person_head_to_head import cluster_at, cluster_scores  # noqa: E402
from kuzu_graph import bridge_aware, labels_from_keep  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/person_em"
DITTO_PAIRS_PER_SEC = 25  # measured in chapter 2: 152k pairs in ~100 min

LOS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]           # rf <= lo: reject without Ditto
HIS = [0.9, 0.95, 0.98, 0.99, 0.995, 1.0]         # rf > hi: accept without Ditto
BRIDGE_GRID = [(b, m) for b in (0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999)
               for m in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995) if m > b]
THS = [0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]  # edge threshold on the result


def combine(rf, dt, lo, hi):
    """The cascade's edge score, and which pairs Ditto had to score."""
    routed = (rf > lo) & (rf <= hi)
    s = np.where(rf > hi, 1.0, 0.0)
    s[routed] = dt[routed]
    return s, routed


def n_true_pairs(clusters):
    _, sizes = np.unique(clusters, return_counts=True)
    return int(np.sum(sizes * (sizes - 1) // 2))


def score(s, th, te, ids, clusters, n_true):
    labels = cluster_at(s, te["left"], te["right"], ids, th)
    return cluster_scores(labels, clusters, n_true)


def grid(rf, dt, te, ids, clusters, n_true):
    rows = []
    for lo in LOS:
        for hi in HIS:
            s, routed = combine(rf, dt, lo, hi)
            for th in THS:
                c = score(s, th, te, ids, clusters, n_true)
                rows.append({"lo": lo, "hi": hi, "threshold": th,
                             "routed_frac": float(routed.mean()), **c})
    return pd.DataFrame(rows)


def restrict(te, meta, keep_ids):
    """The sub-problem induced by a subset of people: their records and the pairs among them."""
    keep = np.isin(meta["ids"], keep_ids)
    ids, clusters = meta["ids"][keep], meta["clusters"][keep]
    m = np.isin(te["left"], ids) & np.isin(te["right"], ids)
    return {k: v[m] for k, v in te.items()}, ids, clusters, m


def frontier(df, budgets):
    """Best clustered F1 achievable while routing at most each budget fraction to Ditto."""
    out = []
    for b in budgets:
        ok = df[df.routed_frac <= b + 1e-9]
        if len(ok):
            r = ok.loc[ok.f1.idxmax()]
            out.append({"budget": b, **{k: (float(r[k]) if k != "clusters" else int(r[k]))
                                        for k in ("lo", "hi", "threshold", "routed_frac",
                                                  "precision", "recall", "f1",
                                                  "adjusted_rand", "clusters",
                                                  "largest_cluster")}})
    return out


def main():
    te = dict(np.load(DATA / "test.npz"))
    meta = dict(np.load(DATA / "test_meta.npz", allow_pickle=True))
    rf = np.load(ROOT / "research/results/person_scores_rf.npz")["prob"]
    dt = np.load(ROOT / "research/results/person_scores_ditto.npz")["prob"]
    ids, clusters = meta["ids"], meta["clusters"]
    n_true = int(meta["n_true_pairs"])
    assert n_true == n_true_pairs(clusters), "true-pair count no longer matches the clusters"
    n_pairs = len(rf)

    out = {"test_pairs": n_pairs, "ditto_pairs_per_sec": DITTO_PAIRS_PER_SEC}

    # --- the two single-model references, on the same finer threshold grid
    for name, p in (("random_forest_only", rf), ("ditto_only", dt)):
        best = max((score(p, th, te, ids, clusters, n_true) | {"threshold": th}
                    for th in THS), key=lambda c: c["f1"])
        out[name] = best
        print(f"{name:20s} best clustered F1 {best['f1']:.2f} @ {best['threshold']}", flush=True)

    # --- full-test grid (picked on test, like chapter 2: optimistic)
    df = grid(rf, dt, te, ids, clusters, n_true)
    budgets = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0]
    out["frontier_picked_on_test"] = frontier(df, budgets)
    print("\nfrontier (picked on test):")
    for r in out["frontier_picked_on_test"]:
        print(f"  <= {r['budget']:.0%} to Ditto: F1 {r['f1']:.2f}  ARI {r['adjusted_rand']:.3f}"
              f"  (lo {r['lo']}, hi {r['hi']}, th {r['threshold']}, routed {r['routed_frac']:.1%})")

    # --- split-half: choose on one half of the people, score on the other
    people = np.unique(clusters)
    half_a = np.array([zlib.crc32(str(p).encode()) % 2 == 0 for p in people])
    halves = {}
    for name, members in (("A", people[half_a]), ("B", people[~half_a])):
        keep_ids = ids[np.isin(clusters, members)]
        h_te, h_ids, h_cl, m = restrict(te, meta, keep_ids)
        halves[name] = (h_te, h_ids, h_cl, m, n_true_pairs(h_cl))
    split = []
    for tune, held in (("A", "B"), ("B", "A")):
        t_te, t_ids, t_cl, t_m, t_n = halves[tune]
        h_te, h_ids, h_cl, h_m, h_n = halves[held]
        t_df = grid(rf[t_m], dt[t_m], t_te, t_ids, t_cl, t_n)
        # Ditto alone gets the same treatment: its threshold is chosen on the tuning half too
        d_th = max(THS, key=lambda th: score(dt[t_m], th, t_te, t_ids, t_cl, t_n)["f1"])
        d_held = score(dt[h_m], d_th, h_te, h_ids, h_cl, h_n)["f1"]
        picks = [(f"budget_{b:.0%}", frontier(t_df, [b])[0]) for b in (0.2, 0.3)]
        r_only = t_df[t_df.hi == 1.0]
        picks.append(("forest_rejects_only", r_only.loc[r_only.f1.idxmax()].to_dict()))
        for label, pick in picks:
            s, routed = combine(rf[h_m], dt[h_m], pick["lo"], pick["hi"])
            c = score(s, pick["threshold"], h_te, h_ids, h_cl, h_n)
            split.append({"tuned_on": tune, "scored_on": held, "config": label,
                          "lo": pick["lo"], "hi": pick["hi"], "threshold": pick["threshold"],
                          "routed_frac": float(routed.mean()), "f1": c["f1"],
                          "adjusted_rand": c["adjusted_rand"],
                          "ditto_only_f1": d_held, "ditto_only_threshold": d_th})
    out["split_half"] = split
    print("\nsplit-half (tuned on one half of the people, scored on the other):")
    for r in split:
        print(f"  tune {r['tuned_on']} -> score {r['scored_on']}, {r['config']:19s}: "
              f"F1 {r['f1']:.2f} routing {r['routed_frac']:5.1%} (lo {r['lo']}, hi {r['hi']}, "
              f"th {r['threshold']})  vs Ditto alone {r['ditto_only_f1']:.2f}")

    # --- bridge-aware clustering (kuzu_graph.py's rule) against plain thresholding, both tuned
    # on the same fine grid. Chapter 2 originally compared bridges tuned up to 0.99 against a
    # plain threshold capped at 0.95, which credited the bridge rule with a threshold gain.
    edges = pd.DataFrame({"src": te["left"], "dst": te["right"]})
    configs = [("random_forest_only", None, None), ("ditto_only", -1.0, 1.0),
               *[(f"cascade_budget_{r['budget']:.0%}", r["lo"], r["hi"])
                 for r in out["frontier_picked_on_test"] if r["budget"] in (0.3, 0.5)]]
    bridge = []
    for label, lo, hi in configs:
        if lo is None:
            s, routed = rf, np.zeros(len(rf), bool)
        else:
            s, routed = combine(rf, dt, lo, hi)
        edges["s"] = s
        plain = max((score(s, th, te, ids, clusters, n_true) | {"threshold": th}
                     for th in THS), key=lambda c: c["f1"])
        best = None
        for base, bmin in BRIDGE_GRID:
            keep, removed = bridge_aware(edges, "s", low=base, bridge_min=bmin)
            c = cluster_scores(labels_from_keep(edges, keep, ids), clusters, n_true)
            if best is None or c["f1"] > best["f1"]:
                best = c | {"base": base, "bridge_min": bmin, "removed": removed}
        bridge.append({"config": label, "lo": lo, "hi": hi,
                       "routed_frac": float(routed.mean()),
                       "plain_f1": plain["f1"], "plain_threshold": plain["threshold"],
                       "bridge": best})
        print(f"{label:22s} routed {routed.mean():5.1%}: plain F1 {plain['f1']:.2f} "
              f"@{plain['threshold']:g}  bridge-aware F1 {best['f1']:.2f} (base {best['base']}, "
              f"bridges >= {best['bridge_min']}, removed {best['removed']})", flush=True)
    out["bridge_aware"] = bridge

    json.dump(out, open(ROOT / "research/results/cascade.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
