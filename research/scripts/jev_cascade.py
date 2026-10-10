"""Forest → Jev cascade on the full person-record test split: no deep-learning training.

Chapter 2's best pipeline scores every blocked pair with a random forest, vetoes the pairs it
is confident are non-matches (rf <= 0.2), and lets Ditto - a RoBERTa model fine-tuned for ~30
minutes and scored at ~25 pairs/s - judge the rest. This replaces Ditto with Jev, TypeSafe's
decision model, called zero-shot with the setup that won in `jev_variants.py` (written
criteria + structured fields). The forest trains in seconds on a CPU; Jev needs no training.

Same 152,772 pairs, same clustering (connected components) and scoring as
`person_head_to_head.py` / `cascade.py`, and the same split-half protocol: every threshold is
chosen on one half of the test people and scored on the other.

  jev_cascade.py --pilot 1000     score a random sample of the band, report cost and AUC
  jev_cascade.py                  score the whole band (resumes), then evaluate
  jev_cascade.py --evaluate       evaluate only
"""
import json
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import restrict, n_true_pairs  # noqa: E402
from jev_variants import ask, body  # noqa: E402
from llm_zeroshot_openrouter import LEDGER, OUT, api_key, spent  # noqa: E402
from person_head_to_head import cluster_at, cluster_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/person_em"
VETO = 0.2  # chapter 2: the forest may reject at rf <= 0.2, never accept
FIELDS = {"first_name": "first_name", "surname": "surname", "date_of_birth": "dob",
          "birth_place": "birth_place", "postcode": "postcode_fake",
          "occupation": "occupation", "gender": "gender"}
ANSWERS = OUT / "answers_jevcascade_historical50k.jsonl"
# Ditto's scores saturate near 1 (its optimum is 0.998), so the grid must reach 0.9995
THS = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                               [0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]]), 4)
_lock = threading.Lock()


def records():
    from splink import splink_datasets
    df = splink_datasets.historical_50k.reset_index(drop=True)
    for c in set(FIELDS.values()):
        df[c] = df[c].fillna("").astype(str).str.lower().str.strip()
    return df.to_dict("index")


def structured(r):
    return {k: r[v] for k, v in FIELDS.items() if r[v]}


def load():
    te = dict(np.load(DATA / "test.npz"))
    meta = dict(np.load(DATA / "test_meta.npz", allow_pickle=True))
    rf = np.load(ROOT / "research/results/person_scores_rf.npz")["prob"]
    dt = np.load(ROOT / "research/results/person_scores_ditto.npz")["prob"]
    return te, meta, rf, dt


def score_band(ks, rows, te, budget):
    client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key()}"})
    state = {"total": spent()}
    print(f"ledger so far ${state['total']:.4f} of ${budget:.2f}", flush=True)
    if state["total"] >= budget:
        raise SystemExit("budget already spent - not starting")
    done = {json.loads(l)["k"] for l in open(ANSWERS)} if ANSWERS.exists() else set()
    todo = [k for k in ks if k not in done]
    fh = open(ANSWERS, "a")
    t0 = time.time()

    def one(k):
        if state["total"] >= budget:
            return
        a, b = int(te["left"][k]), int(te["right"][k])
        d = ask(client, body("structured_criteria",
                             {"structured": (structured(rows[a]), structured(rows[b]))}))
        with _lock:
            state["total"] += d["usage"]["cost"]
            fh.write(json.dumps({"k": int(k), "p": d["answers"]["match"]["noul"]}) + "\n")
            fh.flush()
            with open(LEDGER, "a") as led:
                led.write(json.dumps({"model": "typesafe/jev-1.13:cascade", "dataset": "person_em_test",
                                      "i": int(k), "cost": d["usage"]["cost"]}) + "\n")
            n = len(done) + 1
            done.add(k)
            if n % 5000 == 0:
                print(f"  {n:,}/{len(ks):,} scored, ledger ${state['total']:.3f}, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(one, todo))
    fh.close()
    print(f"scored {len(todo):,} pairs, ledger ${state['total']:.4f}", flush=True)
    if state["total"] >= budget:
        raise SystemExit(f"budget ${budget:.2f} reached - stopping; rerun resumes")


def jev_scores(n):
    p = np.full(n, np.nan)
    for r in map(json.loads, open(ANSWERS)):
        p[r["k"]] = r["p"]
    return p


def best_threshold(s, te, ids, clusters, n_true):
    res = [(th, cluster_scores(cluster_at(s, te["left"], te["right"], ids, th), clusters, n_true))
           for th in THS]
    return max(res, key=lambda z: z[1]["f1"])


def evaluate(te, meta, rf, dt):
    n = len(rf)
    pj = jev_scores(n)
    routed = rf > VETO
    assert not np.isnan(pj[routed]).any(), "band not fully scored yet"
    ids, clusters, n_true = meta["ids"], meta["clusters"], int(meta["n_true_pairs"])
    y = te["y"].astype(bool)
    configs = {
        "forest_only": rf,
        "ditto_only": dt,
        "forest_veto_then_ditto": np.where(routed, dt, 0.0),
        "forest_veto_then_jev": np.where(routed, pj, 0.0),
        # no deep learning: in the pilot the forest and Jev err on different pairs (corr 0.50),
        # and their mean ranks the band better than either alone
        "forest_veto_then_mean_forest_jev": np.where(routed, (rf + pj) / 2, 0.0),
        "forest_veto_then_mean_jev_ditto": np.where(routed, (pj + dt) / 2, 0.0),
    }
    out = {"pairs": n, "routed_to_llm": int(routed.sum()), "routed_frac": float(routed.mean()),
           "band_auc": {"jev": round(roc_auc_score(y[routed], pj[routed]), 4),
                        "ditto": round(roc_auc_score(y[routed], dt[routed]), 4),
                        "forest": round(roc_auc_score(y[routed], rf[routed]), 4)}}
    print(json.dumps(out), flush=True)

    # picked on the full test set (as chapter 2 reports), for reference
    full = {}
    for name, s in configs.items():
        th, c = best_threshold(s, te, ids, clusters, n_true)
        full[name] = c | {"threshold": float(th)}
        print(f"{name:34s} full-test best F1 {c['f1']:.2f} @ {th}  ARI {c['adjusted_rand']:.3f}",
              flush=True)
    out["picked_on_test"] = full

    # split-half by person: tune the threshold on one half, score the other (cascade.py's split)
    people = np.unique(clusters)
    half_a = np.array([zlib.crc32(str(p).encode()) % 2 == 0 for p in people])
    halves = {}
    for name, members in (("A", people[half_a]), ("B", people[~half_a])):
        h_te, h_ids, h_cl, m = restrict(te, meta, ids[np.isin(clusters, members)])
        halves[name] = (h_te, h_ids, h_cl, m, n_true_pairs(h_cl))
    split = {}
    for name, s in configs.items():
        f1s = []
        for tune, held in (("A", "B"), ("B", "A")):
            t_te, t_ids, t_cl, t_m, t_n = halves[tune]
            h_te, h_ids, h_cl, h_m, h_n = halves[held]
            th, _ = best_threshold(s[t_m], t_te, t_ids, t_cl, t_n)
            f1s.append(cluster_scores(cluster_at(s[h_m], h_te["left"], h_te["right"], h_ids, th),
                                      h_cl, h_n)["f1"])
        split[name] = {"held_out_f1": f1s, "mean": round(float(np.mean(f1s)), 2)}
        print(f"{name:34s} split-half held-out F1 {f1s[0]:.2f} / {f1s[1]:.2f}  "
              f"mean {np.mean(f1s):.2f}", flush=True)
    out["split_half"] = split
    json.dump(out, open(ROOT / "research/results/jev_cascade.json", "w"), indent=1, default=float)


def main():
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 7.80
    te, meta, rf, dt = load()
    band = np.flatnonzero(rf > VETO)
    if "--evaluate" not in sys.argv:
        rows = records()
        assert all(rows[i]["cluster"] == c for i, c in zip(meta["ids"], meta["clusters"]))
        if "--pilot" in sys.argv:
            n = int(sys.argv[sys.argv.index("--pilot") + 1])
            ks = np.random.default_rng(0).choice(band, n, replace=False)
            score_band(ks, rows, te, budget)
            p = jev_scores(len(rf))
            m = ~np.isnan(p)
            y = te["y"].astype(bool)
            print(f"pilot on {m.sum()} band pairs: AUC jev {roc_auc_score(y[m], p[m]):.4f}  "
                  f"ditto {roc_auc_score(y[m], dt[m]):.4f}  forest {roc_auc_score(y[m], rf[m]):.4f}")
            return
        print(f"band: {len(band):,} of {len(rf):,} pairs ({len(band) / len(rf):.1%})", flush=True)
        score_band(band, rows, te, budget)
    evaluate(te, meta, rf, dt)


if __name__ == "__main__":
    main()
