"""Build a person-record entity-matching dataset from historical_50k.

Produces, over CLUSTER-DISJOINT splits (no person appears in two splits):
  data/person_em/{train,valid,test}.txt   Ditto format: <serialised A>\\t<serialised B>\\t<label>
  data/person_em/{train,valid,test}.npz   the same pairs as similarity features, for the RF
  data/person_em/test_meta.npz            record ids + true cluster ids, for the clustering pass

Train/valid are downsampled (all positives, negatives capped) so a transformer can be trained
on this laptop; the test split keeps EVERY candidate pair among held-out records, because the
clustering pass needs the full candidate graph.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/person_em"
FIELDS = ["first_name", "surname", "dob", "birth_place", "postcode_fake", "occupation", "gender"]
BLOCKS = [("first_name", "surname"), ("surname", "dob"), ("first_name", "dob"),
          ("postcode_fake",), ("birth_place", "surname"), ("dob", "birth_place"),
          ("surname", "occupation"), ("first_name", "birth_place"),
          ("surname_init3", "first_init3")]
MAX_BLOCK = 150
TRAIN_CAP, VALID_CAP = 12_000, 3_000
SEED = 13


def serialise(r):
    return " ".join(f"COL {f} VAL {r[f]}" for f in FIELDS if str(r[f]).strip())


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


def candidate_pairs(df, idx_subset):
    """Blocked candidate pairs restricted to a subset of row indices."""
    sub = df.loc[sorted(idx_subset)]
    keys = set()
    for cols in BLOCKS:
        s2 = sub.dropna(subset=list(cols))
        for _, g in s2.groupby(list(cols)):
            ids = sorted(g.index.values.tolist())
            if 1 < len(ids) <= MAX_BLOCK:
                keys.update(combinations(ids, 2))
    return sorted(keys)


def write_split(name, pairs, rows, cap, rng):
    y = np.array([rows[a]["cluster"] == rows[b]["cluster"] for a, b in pairs])
    keep = np.arange(len(pairs))
    if cap and len(pairs) > cap:
        # uniform subsample: keeps the natural positive rate of the blocked candidate set,
        # so train/valid match the distribution the test split is scored on
        keep = np.sort(rng.choice(keep, cap, replace=False))
    pairs = [pairs[i] for i in keep]
    y = y[keep]
    with open(OUT / f"{name}.txt", "w", encoding="utf-8") as f:
        for (a, b), lab in zip(pairs, y):
            f.write(f"{serialise(rows[a])}\t{serialise(rows[b])}\t{int(lab)}\n")
    X = np.array([feats(rows[a], rows[b]) for a, b in pairs], dtype=np.float32)
    np.savez_compressed(OUT / f"{name}.npz", X=X, y=y,
                        left=np.array([a for a, _ in pairs]),
                        right=np.array([b for _, b in pairs]))
    print(f"{name}: {len(pairs):,} pairs, {int(y.sum()):,} positive "
          f"({100 * y.mean():.1f}%)", flush=True)
    return pairs, y


def main():
    from splink import splink_datasets
    df = splink_datasets.historical_50k.reset_index(drop=True)
    for c in ["first_name", "surname", "full_name", "dob", "birth_place",
              "postcode_fake", "gender", "occupation"]:
        df[c] = df[c].fillna("").astype(str).str.lower().str.strip()
    df["surname_init3"] = df.surname.str[:3]
    df["first_init3"] = df.first_name.str[:3]
    rows = df.to_dict("index")

    rng = np.random.default_rng(SEED)
    clusters = np.array(sorted(df.cluster.unique()))
    rng.shuffle(clusters)
    n = len(clusters)
    split_of = {}
    for c in clusters[: int(0.60 * n)]:
        split_of[c] = "train"
    for c in clusters[int(0.60 * n): int(0.75 * n)]:
        split_of[c] = "valid"
    for c in clusters[int(0.75 * n):]:
        split_of[c] = "test"
    df["split"] = df.cluster.map(split_of)

    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"clusters": {k: int((df.split == k).sum()) for k in ("train", "valid", "test")}}
    for name, cap in (("train", TRAIN_CAP), ("valid", VALID_CAP), ("test", None)):
        idx = set(df.index[df.split == name])
        pairs = candidate_pairs(df, idx)
        # blocking ceiling within this split
        truth = set()
        for _, g in df[df.split == name].groupby("cluster"):
            truth.update(combinations(sorted(g.index.values.tolist()), 2))
        recall = len(set(pairs) & truth) / len(truth) if truth else 0.0
        print(f"{name}: records={len(idx):,} candidates={len(pairs):,} "
              f"true_pairs={len(truth):,} blocking_recall={100 * recall:.2f}%", flush=True)
        kept, y = write_split(name, pairs, rows, cap, rng)
        stats[name] = {"records": len(idx), "candidates": len(pairs),
                       "true_pairs": len(truth), "blocking_recall": round(100 * recall, 2),
                       "written_pairs": len(kept), "written_positives": int(y.sum())}
        if name == "test":
            ids = np.array(sorted(idx))
            np.savez_compressed(OUT / "test_meta.npz", ids=ids,
                                clusters=np.array([rows[i]["cluster"] for i in ids]),
                                n_true_pairs=len(truth))
    json.dump(stats, open(ROOT / "results/person_em_dataset.json", "w"), indent=1)


if __name__ == "__main__":
    main()
