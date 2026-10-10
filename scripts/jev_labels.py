"""Can an LLM replace the hand labels? Train the forest on Jev's labels instead of the truth.

Chapter 2 showed a zero-shot model is not precise enough to sit in the clustering path. The
use the 2025-26 literature supports instead (Steiner & Bizer 2026) is labelling: have the LLM
label the training pairs, then train the supervised matcher on those labels. Jev labels every
training pair with its best setup from `jev_variants.py` (written criteria + structured fields);
nothing about Jev is tuned on true labels. The forest - same features, same hyperparameters -
is trained two ways:

  hard   label = P(match) > 0.5
  soft   each pair twice, as a match with weight p and a non-match with weight 1 - p

and scored exactly as its hand-labelled counterpart, on the same test pairs, clustered, with
the threshold tuned on one half of the test people and scored on the other.

  jev_labels.py historical    person_em training pairs -> 152,772-pair test split (chapter 2)
  jev_labels.py customer      customer pipeline training pairs -> its 14,997 test pairs (chapter 5)
"""
import json
import sys
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_variants import ask, body  # noqa: E402
from llm_zeroshot_openrouter import LEDGER, OUT, api_key, spent  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
THS = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                               [0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]]), 4)
BUDGET = 7.80
_lock = threading.Lock()


def label(name, records_a, records_b):
    """P(match) from Jev for every training pair; resumes from the answers file."""
    path = OUT / f"answers_jevlabels_{name}.jsonl"
    done = {r["k"]: r["p"] for r in map(json.loads, open(path))} if path.exists() else {}
    state = {"total": spent()}
    print(f"{name}: {len(records_a):,} training pairs, {len(done):,} already labelled; "
          f"ledger ${state['total']:.4f} of ${BUDGET:.2f}", flush=True)
    if state["total"] >= BUDGET:
        raise SystemExit("budget already spent - not starting")
    client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key()}"})
    fh = open(path, "a")

    def one(k):
        if state["total"] >= BUDGET:
            return
        d = ask(client, body("structured_criteria", {"structured": (records_a[k], records_b[k])}))
        with _lock:
            state["total"] += d["usage"]["cost"]
            done[k] = d["answers"]["match"]["noul"]
            fh.write(json.dumps({"k": int(k), "p": done[k]}) + "\n")
            fh.flush()
            with open(LEDGER, "a") as led:
                led.write(json.dumps({"model": "typesafe/jev-1.13:labels", "dataset": name,
                                      "i": int(k), "cost": d["usage"]["cost"]}) + "\n")

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, [k for k in range(len(records_a)) if k not in done]))
    fh.close()
    if state["total"] >= BUDGET:
        raise SystemExit(f"budget ${BUDGET:.2f} reached - stopping; rerun resumes")
    print(f"{name}: labelled, ledger ${state['total']:.4f}", flush=True)
    return np.array([done[k] for k in range(len(records_a))])


N_HAND = 300  # hand-labelled pairs allowed for choosing Jev's labelling cut-off


def cutoff_from_sample(y_true, p, n=N_HAND, seed=0):
    """Jev's cut-off chosen to maximise label F1 on n randomly chosen hand-labelled pairs."""
    from sklearn.metrics import f1_score
    k = np.random.default_rng(seed).choice(len(p), n, replace=False)
    return float(max(np.round(np.arange(0.05, 0.96, 0.05), 2),
                     key=lambda t: f1_score(y_true[k], p[k] > t)))


def forests(X, y_true, p):
    kw = dict(n_estimators=300, random_state=0, n_jobs=4, class_weight="balanced_subsample")
    hard = RandomForestClassifier(**kw).fit(X, p > 0.5)
    cut = cutoff_from_sample(y_true, p)
    print(f"  cut-off chosen on {N_HAND} hand-labelled pairs: {cut}", flush=True)
    tuned = RandomForestClassifier(**kw).fit(X, p > cut)
    soft = RandomForestClassifier(**kw).fit(
        np.vstack([X, X]), np.r_[np.ones(len(X), bool), np.zeros(len(X), bool)],
        sample_weight=np.r_[p, 1 - p])
    true = RandomForestClassifier(**kw).fit(X, y_true)
    return {"forest_true_labels": true, "forest_jev_hard_labels": hard,
            "forest_jev_soft_labels": soft, f"forest_jev_labels_cutoff_{cut}_from_{N_HAND}": tuned}


def label_quality(y, p, cut=0.5):
    pred = p > cut
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    return {"pairs": int(len(y)), "positives": int(y.sum()),
            "precision": round(100 * tp / (tp + fp), 2), "recall": round(100 * tp / (tp + fn), 2),
            "accuracy": round(100 * float((pred == y).mean()), 2), "cutoff": cut}


def split_half(score_fn, name):
    """score_fn(half, th) -> clustered scores; tune th on one half, score the other."""
    f1s = []
    for tune, held in (("A", "B"), ("B", "A")):
        th = max(THS, key=lambda t: score_fn(tune, t)["f1"])
        f1s.append(score_fn(held, th)["f1"])
    print(f"  {name:26s} held-out F1 {f1s[0]:.2f} / {f1s[1]:.2f}  mean {np.mean(f1s):.2f}", flush=True)
    return {"held_out_f1": f1s, "mean": round(float(np.mean(f1s)), 2)}


def historical():
    from cascade import restrict, n_true_pairs
    from jev_cascade import records, structured
    from person_head_to_head import cluster_at, cluster_scores
    tr = dict(np.load(ROOT / "data/person_em/train.npz"))
    te = dict(np.load(ROOT / "data/person_em/test.npz"))
    meta = dict(np.load(ROOT / "data/person_em/test_meta.npz", allow_pickle=True))
    rows = records()
    p = label("historical", [structured(rows[int(a)]) for a in tr["left"]],
              [structured(rows[int(b)]) for b in tr["right"]])
    y = tr["y"].astype(bool)
    out = {"labels": label_quality(y, p), "split_half": {},
           "labels_at_sample_cutoff": label_quality(y, p, cutoff_from_sample(y, p))}
    print(f"  Jev labels vs truth: {out['labels']}\n  at the sample cut-off: "
          f"{out['labels_at_sample_cutoff']}", flush=True)
    ids, clusters = meta["ids"], meta["clusters"]
    people = np.unique(clusters)
    half_a = np.array([zlib.crc32(str(q).encode()) % 2 == 0 for q in people])
    halves = {}
    for h, members in (("A", people[half_a]), ("B", people[~half_a])):
        h_te, h_ids, h_cl, m = restrict(te, meta, ids[np.isin(clusters, members)])
        halves[h] = (h_te, h_ids, h_cl, m, n_true_pairs(h_cl))
    for name, model in forests(tr["X"], y, p).items():
        s = model.predict_proba(te["X"])[:, 1]

        def score_fn(h, th, s=s):
            h_te, h_ids, h_cl, m, n = halves[h]
            return cluster_scores(cluster_at(s[m], h_te["left"], h_te["right"], h_ids, th), h_cl, n)
        out["split_half"][name] = split_half(score_fn, name)
    return out


def customer():
    from customer_ditto import cluster_f1
    from customer_pipeline import prepare
    P = prepare(aware=True)
    df, pairs, X, y, tr, te, in_train = (P[k] for k in ("df", "pairs", "X", "y", "tr", "te", "in_train"))
    df = df.copy()
    df["address"] = (df.addr_number.astype(str) + " " + df.addr_street.astype(str) + " "
                     + df.addr_unit.astype(str)).str.split().str.join(" ")
    keys = ["first_name", "middle_initial", "last_name", "date_of_birth", "ssn", "address", "email", "phone"]
    rows = [{k: str(r[k]) for k in keys if str(r[k]).strip() and str(r[k]) != "nan"}
            for r in df.to_dict("records")]
    idx = np.flatnonzero(tr)
    p = label("customer", [rows[a] for a in pairs[idx, 0]], [rows[b] for b in pairs[idx, 1]])
    out = {"labels": label_quality(y[idx], p), "split_half": {},
           "labels_at_sample_cutoff": label_quality(y[idx], p, cutoff_from_sample(y[idx], p))}
    print(f"  Jev labels vs truth: {out['labels']}", flush=True)
    sid = df.simulant_id.to_numpy()
    half_a = np.array([zlib.crc32(str(q).encode()) % 2 == 0 for q in sid])
    test_records = ~in_train
    halves = {h: test_records & (half_a if h == "A" else ~half_a) for h in "AB"}
    in_half = {h: te & halves[h][pairs[:, 0]] & halves[h][pairs[:, 1]] for h in "AB"}
    for name, model in forests(X[idx], y[idx], p).items():
        s = model.predict_proba(X)[:, 1]
        out["split_half"][name] = split_half(
            lambda h, th, s=s: cluster_f1(P, s, in_half[h], halves[h], th), name)
    return out


def main():
    which = sys.argv[1:] or ["customer", "historical"]
    path = ROOT / "results/jev_labels.json"
    res = json.load(open(path)) if path.exists() else {}
    for name in which:
        res[name] = {"customer": customer, "historical": historical}[name]()
        json.dump(res, open(path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
