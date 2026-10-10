"""A hard person-matching task on SYNTHETIC customers, where no model can know the people.

The historical50k-person-hard task is built from real historical figures, and a model with
enough world knowledge can use that: GPT-6 Luna's explanations cite who people were and when
they died. That is knowledge a matcher on a real customer file does not have. This task is
built the same way from the two-source pseudopeople customer file (`customer_data.py`), whose
people are simulated:

  positives  the same simulant, with the two records' names written differently
  negatives  different simulants sharing a surname with name similarity > 0.75 - mostly
             household members, the realistic hard case

Same size and positive rate as the historical task (250 + 410), same prompt, written in the
MatchGPT task format so `llm_zeroshot_*.py` run it unchanged. A random forest trained on
pairs from a disjoint set of simulants scores the identical pairs, so the LLM rows and the
classical row are directly comparable.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.ensemble import RandomForestClassifier

from person_task_baselines import prf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/customer"
TASKS = ROOT / "external/MatchGPT/LLMForEM/tasks"
NAME = "customer-person-hard"
N_POS, N_NEG, SEED = 250, 410, 42
PREFIX = ("Do the two person records refer to the same real-world person? "
          "Answer with 'Yes' if they do and 'No' if they do not.\n")


def load():
    crm = pd.read_parquet(DATA / "source_crm.parquet")
    bill = pd.read_parquet(DATA / "source_billing.parquet")
    df = pd.concat([crm, bill], ignore_index=True)
    for c in ["first_name", "middle_initial", "last_name", "date_of_birth", "addr_number",
              "addr_street", "addr_unit", "email", "phone"]:
        df[c] = df[c].fillna("").astype(str).str.lower().str.strip()
    df["full_name"] = (df.first_name + " " + df.middle_initial + " " + df.last_name).str.split().str.join(" ")
    df["address"] = (df.addr_number + " " + df.addr_street + " " + df.addr_unit).str.split().str.join(" ")
    return df.to_dict("records")


def serialise(r):
    parts = [r["full_name"], r["date_of_birth"], r["address"], r["email"], r["phone"]]
    return " ".join(p for p in parts if p)


def feats(a, b):
    both = lambda f: bool(a[f]) and bool(b[f])
    return [
        JaroWinkler.similarity(a["first_name"], b["first_name"]),
        JaroWinkler.similarity(a["last_name"], b["last_name"]),
        JaroWinkler.similarity(a["full_name"], b["full_name"]),
        Levenshtein.normalized_similarity(a["full_name"], b["full_name"]),
        float(a["middle_initial"] == b["middle_initial"]),
        float(both("date_of_birth") and a["date_of_birth"] == b["date_of_birth"]),
        float(both("date_of_birth")),
        Levenshtein.normalized_similarity(a["address"], b["address"]),
        float(a["address"] == b["address"]),
        float(both("email") and a["email"] == b["email"]),
        float(both("phone") and a["phone"] == b["phone"]),
    ]


def sample_pairs(rows, members, rng, n_pos, n_neg):
    """Hard pairs among the records of `members` (a set of simulant ids)."""
    idx = [i for i, r in enumerate(rows) if r["simulant_id"] in members]
    by_sim, by_surname = {}, {}
    for i in idx:
        by_sim.setdefault(rows[i]["simulant_id"], []).append(i)
        by_surname.setdefault(rows[i]["last_name"], []).append(i)
    pos = []
    for ids in by_sim.values():
        if len(ids) > 1:
            a, b = rng.sample(ids, 2)
            if rows[a]["full_name"] != rows[b]["full_name"]:
                pos.append((a, b))
    rng.shuffle(pos)
    neg = []
    surnames = [s for s, ids in by_surname.items() if len(ids) > 1 and s]
    rng.shuffle(surnames)
    for s in surnames:
        ids = by_surname[s]
        for _ in range(3):
            a, b = rng.sample(ids, 2)
            if rows[a]["simulant_id"] != rows[b]["simulant_id"] and \
                    JaroWinkler.similarity(rows[a]["full_name"], rows[b]["full_name"]) > 0.75:
                neg.append((a, b))
                break
        if len(neg) >= n_neg:
            break
    return pos[:n_pos], neg[:n_neg]


def build():
    """Rows, the test pairs in task order (a, b, label), the training simulants, and the rng."""
    rows = load()
    rng = random.Random(SEED)
    sims = sorted({r["simulant_id"] for r in rows})
    rng.shuffle(sims)
    half = len(sims) // 2
    test_sims, train_sims = set(sims[:half]), set(sims[half:])

    pos, neg = sample_pairs(rows, test_sims, rng, N_POS, N_NEG)
    assert len(pos) == N_POS and len(neg) == N_NEG, (len(pos), len(neg))
    pairs = [(a, b, 1) for a, b in pos] + [(a, b, 0) for a, b in neg]
    rng.shuffle(pairs)
    return rows, pairs, train_sims


def main():
    rows, pairs, train_sims = build()

    task = {"name": f"Entity matching - {NAME}",
            "description": "Person matching on synthetic two-source customer records (pseudopeople): "
                           "positives with differing names; negatives sharing a surname with "
                           "name similarity > 0.75, mostly household members.",
            "task_prefix": PREFIX, "metrics": ["multiple_choice_grade"],
            "examples": [{"input": f"Record 1: '{serialise(rows[a])}'\nRecord 2: '{serialise(rows[b])}'",
                          "target_scores": {"Yes": y, "No": 1 - y}} for a, b, y in pairs]}
    path = TASKS / f"{NAME}_domain-complex-force.json"
    json.dump(task, open(path, "w"), indent=1)
    print(f"wrote {path}: {len(pairs)} pairs, {N_POS} positive")

    # random forest trained on hard pairs from the other half of the simulants
    tr_pos, tr_neg = sample_pairs(rows, train_sims, random.Random(SEED + 1), 5000, 5000)
    X = np.array([feats(rows[a], rows[b]) for a, b in tr_pos + tr_neg])
    y = np.array([1] * len(tr_pos) + [0] * len(tr_neg))
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4,
                                class_weight="balanced").fit(X, y)
    Xt = np.array([feats(rows[a], rows[b]) for a, b, _ in pairs])
    truth = [bool(t) for _, _, t in pairs]
    res = {"task": NAME, "train_pairs": {"pos": len(tr_pos), "neg": len(tr_neg)},
           "random_forest": prf(truth, list(rf.predict_proba(Xt)[:, 1] > 0.5))}
    # no name+DOB rule here: billing records carry no DOB, so it could almost never fire
    json.dump(res, open(ROOT / "results/customer_person_task_baselines.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
