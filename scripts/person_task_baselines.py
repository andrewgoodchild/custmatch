"""Score the SAME person-matching pairs the LLM saw, with classical matchers.

Rebuilds the historical50k-person-hard task (same seed, same pairs), then evaluates:
  - a deterministic rule (high name similarity AND same date of birth)
  - a random forest on similarity features, trained on records from disjoint clusters

so the LLM number and the classical numbers are directly comparable.
"""
import json
import random
from pathlib import Path

import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "external/MatchGPT/LLMForEM/tasks/historical50k-person-hard_domain-complex-force.json"
FIELDS = ["full_name", "dob", "birth_place", "postcode_fake", "occupation"]


def feats(a, b):
    return [
        JaroWinkler.similarity(a["first_name"], b["first_name"]),
        JaroWinkler.similarity(a["surname"], b["surname"]),
        JaroWinkler.similarity(a["full_name"], b["full_name"]),
        Levenshtein.normalized_similarity(a["full_name"], b["full_name"]),
        float(a["dob"] == b["dob"]),
        Levenshtein.normalized_similarity(a["dob"], b["dob"]),
        float(a["birth_place"] == b["birth_place"]),
        JaroWinkler.similarity(a["birth_place"], b["birth_place"]),
        float(a["postcode_fake"] == b["postcode_fake"]),
        float(a["postcode_fake"][:4] == b["postcode_fake"][:4]),
        float(a["gender"] == b["gender"]),
        float(a["occupation"] == b["occupation"] and a["occupation"] != ""),
    ]


def prf(truth, pred):
    tp = sum(t and p for t, p in zip(truth, pred))
    fp = sum((not t) and p for t, p in zip(truth, pred))
    fn = sum(t and (not p) for t, p in zip(truth, pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0,
            "predicted_matches": int(tp + fp),
            "accuracy": round(100 * sum(t == p for t, p in zip(truth, pred)) / len(truth), 2)}


def rebuild_pairs():
    """The exact pairs of make_person_llm_task.py (same seed and order), with their rows."""
    import random
    from splink import splink_datasets
    df = splink_datasets.historical_50k.reset_index(drop=True)
    for c in FIELDS + ["first_name", "surname", "gender"]:
        df[c] = df[c].fillna("").astype(str).str.lower().str.strip()
    rows = df.to_dict("records")

    # rebuild exactly the pairs from make_person_llm_task.py (same seed and order)
    rng = random.Random(42)
    by_cluster = {}
    for i, r in enumerate(rows):
        by_cluster.setdefault(r["cluster"], []).append(i)
    pos = []
    for ids in by_cluster.values():
        if len(ids) > 1:
            a, b = rng.sample(ids, 2)
            if rows[a]["full_name"] != rows[b]["full_name"]:
                pos.append((a, b))
    rng.shuffle(pos)
    pos = pos[:250]
    by_surname = {}
    for i, r in enumerate(rows):
        by_surname.setdefault(r["surname"], []).append(i)
    neg = []
    surnames = [s for s, ids in by_surname.items() if len(ids) > 1 and s]
    rng.shuffle(surnames)
    for s in surnames:
        ids = by_surname[s]
        for _ in range(3):
            a, b = rng.sample(ids, 2)
            if rows[a]["cluster"] != rows[b]["cluster"]:
                if JaroWinkler.similarity(rows[a]["full_name"], rows[b]["full_name"]) > 0.75:
                    neg.append((a, b))
                    break
        if len(neg) >= 1000:
            break
    pairs = [(a, b, 1) for a, b in pos] + [(a, b, 0) for a, b in neg]
    return rows, pairs, by_cluster, by_surname, surnames


def main():
    rows, pairs, by_cluster, by_surname, surnames = rebuild_pairs()
    task = json.load(open(TASK))
    assert len(pairs) == len(task["examples"]), (len(pairs), len(task["examples"]))

    X = np.array([feats(rows[a], rows[b]) for a, b, _ in pairs], dtype=np.float32)
    y = [lab for _, _, lab in pairs]
    eval_clusters = {rows[a]["cluster"] for a, _, _ in pairs} | {rows[b]["cluster"] for _, b, _ in pairs}

    res = {"n": len(pairs), "positives": sum(y),
           "llm_llama31_8b": json.load(open(
               ROOT / "results/llm/result_historical50k-person-hard_domain-complex-force"
                      "_Meta-Llama-3.1-8B-Instruct-4bit.json"))}

    rule = [int(JaroWinkler.similarity(rows[a]["full_name"], rows[b]["full_name"]) > 0.9
                and rows[a]["dob"] == rows[b]["dob"]) for a, b, _ in pairs]
    res["rule_name090_and_dob"] = prf(y, rule)

    # training pairs from clusters that do not appear in the evaluation set
    rng2 = random.Random(7)
    train = []
    other = [c for c in by_cluster if c not in eval_clusters]
    for c in other:
        ids = by_cluster[c]
        if len(ids) > 1:
            a, b = rng2.sample(ids, 2)
            train.append((a, b, 1))
    for s in surnames:
        ids = [i for i in by_surname[s] if rows[i]["cluster"] not in eval_clusters]
        if len(ids) > 1:
            a, b = rng2.sample(ids, 2)
            if rows[a]["cluster"] != rows[b]["cluster"]:
                train.append((a, b, 0))
    Xtr = np.array([feats(rows[a], rows[b]) for a, b, _ in train], dtype=np.float32)
    ytr = [lab for _, _, lab in train]
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4,
                                class_weight="balanced_subsample").fit(Xtr, ytr)
    res["rf_trained_on_disjoint_clusters"] = prf(y, rf.predict(X)) | {
        "train_pairs": len(train), "train_positives": int(sum(ytr))}

    json.dump(res, open(ROOT / "results/person_task_comparison.json", "w"), indent=1, default=int)
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) else
              {kk: v[kk] for kk in ("precision", "recall", "f1", "accuracy") if kk in v}, flush=True)


if __name__ == "__main__":
    main()
