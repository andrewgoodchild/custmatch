"""RF vs Ditto on person records, pairwise and after a clustering pass.

Pairwise scores are compared against ALL true pairs in the test split, so blocking misses count
against recall. The clustering pass takes edges above a threshold, forms connected components
(transitive closure - what a customer file actually needs to produce golden records) and scores
the induced partition:

  pairwise P/R/F1 of the implied within-cluster pairs, computed from cluster-overlap counts
  rather than enumerating pairs, and the adjusted Rand index over the held-out records.

Run `person_head_to_head.py rf` first (trains the RF and writes its scores), then with no
argument once results/person_scores_ditto.npz exists.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/person_em"
# The fine tail matters: Ditto's scores saturate near 1, and a grid that stopped at 0.95 put its
# best clustered F1 at 88.1 when the real optimum (0.998) is 94.7.
THRESHOLDS = np.concatenate([np.round(np.arange(0.05, 0.96, 0.05), 2),
                             [0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]])


def pair_prf(y, pred, n_true_all):
    """Precision/recall/F1 where recall is against every true pair, including blocked-out ones."""
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_true_all
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0,
            "predicted_pairs": tp + fp}


def cluster_scores(labels_pred, labels_true, n_true_all):
    """Pairwise metrics of a partition, from cluster-overlap counts (no pair enumeration)."""
    def n2(c):
        c = np.asarray(c, dtype=np.float64)
        return float(np.sum(c * (c - 1) / 2))

    _, pred_idx = np.unique(labels_pred, return_inverse=True)
    _, true_idx = np.unique(labels_true, return_inverse=True)
    pred_sizes = np.bincount(pred_idx)
    m = coo_matrix((np.ones(len(pred_idx)), (pred_idx, true_idx))).tocsr()
    tp = n2(m.data)
    predicted = n2(pred_sizes)
    p = tp / predicted if predicted else 0.0
    r = tp / n_true_all
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0,
            "clusters": int(len(pred_sizes)), "largest_cluster": int(pred_sizes.max()),
            "implied_pairs": int(predicted),
            "adjusted_rand": round(adjusted_rand_score(labels_true, labels_pred), 4)}


def cluster_at(prob, left, right, ids, th):
    """Connected components over edges above th, restricted to the held-out records."""
    pos = {v: i for i, v in enumerate(ids)}
    keep = prob > th
    r = np.array([pos[v] for v in left[keep]])
    c = np.array([pos[v] for v in right[keep]])
    g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(ids), len(ids)))
    _, labels = connected_components(g, directed=False)
    return labels


def evaluate(name, prob, th_valid, te, meta, out):
    y = te["y"].astype(bool)
    n_true_all = int(meta["n_true_pairs"])
    ids, true_clusters = meta["ids"], meta["clusters"]

    res = {"model": name, "threshold_from_valid": float(th_valid),
           "pairwise": pair_prf(y, prob > th_valid, n_true_all)}

    sweep = []
    for th in THRESHOLDS:
        labels = cluster_at(prob, te["left"], te["right"], ids, th)
        sweep.append({"threshold": float(th),
                      "pairwise": pair_prf(y, prob > th, n_true_all),
                      "clustered": cluster_scores(labels, true_clusters, n_true_all)})
    res["sweep"] = sweep
    res["clustered_at_valid_threshold"] = next(
        (s["clustered"] for s in sweep if abs(s["threshold"] - th_valid) < 1e-6), None)
    res["best_pairwise"] = max(sweep, key=lambda s: s["pairwise"]["f1"])
    res["best_clustered"] = max(sweep, key=lambda s: s["clustered"]["f1"])

    print(f"\n=== {name}")
    print(f"  pairwise @valid-threshold {th_valid:.2f}: {res['pairwise']}")
    print(f"  best pairwise  @{res['best_pairwise']['threshold']:g}: "
          f"{res['best_pairwise']['pairwise']}")
    print(f"  best clustered @{res['best_clustered']['threshold']:g}: "
          f"{res['best_clustered']['clustered']}")
    out[name] = res


def main():
    te = dict(np.load(DATA / "test.npz"))
    meta = dict(np.load(DATA / "test_meta.npz", allow_pickle=True))
    out_path = ROOT / "results/person_head_to_head.json"
    out = json.load(open(out_path)) if out_path.exists() else {}
    out["dataset"] = {"test_records": int(len(meta["ids"])),
                      "test_candidate_pairs": int(len(te["y"])),
                      "test_true_pairs": int(meta["n_true_pairs"]),
                      "true_clusters": int(len(set(meta["clusters"].tolist())))}

    if "rf" in sys.argv or not (ROOT / "results/person_scores_rf.npz").exists():
        tr, va = dict(np.load(DATA / "train.npz")), dict(np.load(DATA / "valid.npz"))
        rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4,
                                    class_weight="balanced_subsample").fit(tr["X"], tr["y"])
        p_va = rf.predict_proba(va["X"])[:, 1]
        th, f1 = max(((t, f1_score(va["y"], p_va > t)) for t in THRESHOLDS), key=lambda z: z[1])
        print(f"RF valid_f1={100 * f1:.2f} at threshold {th:.2f}", flush=True)
        np.savez_compressed(ROOT / "results/person_scores_rf.npz",
                            prob=rf.predict_proba(te["X"])[:, 1], threshold=th, valid_f1=f1)

    rf_s = dict(np.load(ROOT / "results/person_scores_rf.npz"))
    evaluate("random_forest", rf_s["prob"], float(rf_s["threshold"]), te, meta, out)

    ditto_path = ROOT / "results/person_scores_ditto.npz"
    if ditto_path.exists():
        d = dict(np.load(ditto_path))
        evaluate("ditto_roberta", d["prob"], float(d["threshold"]), te, meta, out)
    else:
        print("\n(ditto scores not present yet - rerun once person_ditto.py has finished)")

    json.dump(out, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
