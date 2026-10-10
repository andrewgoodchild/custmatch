"""Harder person-record benchmark: Splink's historical_50k (50,578 records, 5,156 people).

Real historical people from Wikidata with injected errors; `cluster` is the ground truth.
Deduplication, scored as pairwise precision / recall / F1 over all true within-cluster pairs
(blocking misses count against recall).

Methods: deterministic rules; Splink 4 (unsupervised Fellegi-Sunter); random forest on
similarity features (trained on a disjoint half of the clusters).

Pairs are held as int32 row-index arrays keyed by i*N+j, which keeps ~10M candidate pairs
inside a few hundred MB.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

BLOCKS = [("first_name", "surname"), ("surname", "dob"), ("first_name", "dob"),
          ("postcode_fake",), ("birth_place", "surname"), ("dob", "birth_place"),
          ("surname", "occupation"), ("first_name", "birth_place"),
          ("surname_init3", "first_init3")]
MAX_BLOCK = 150  # skip pathological blocks, as any real pipeline would


def pair_keys(idx_lists, n):
    """Encode sorted (i, j) index pairs as unique int64 keys."""
    out = []
    for ids in idx_lists:
        if 1 < len(ids) <= MAX_BLOCK:
            a = np.array(sorted(ids), dtype=np.int64)
            i, j = np.triu_indices(len(a), k=1)
            out.append(a[i] * n + a[j])
    if not out:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(out))


def true_keys(df, n):
    return pair_keys([g.index.values for _, g in df.groupby("cluster")], n)


def candidate_keys(df, n):
    keys = []
    for cols in BLOCKS:
        sub = df.dropna(subset=list(cols))
        keys.append(pair_keys([g.index.values for _, g in sub.groupby(list(cols))], n))
    return np.unique(np.concatenate(keys))


def score(pred, truth):
    pred, truth = np.asarray(pred), np.asarray(truth)
    tp = np.intersect1d(pred, truth, assume_unique=True).size
    p = tp / pred.size if pred.size else 0.0
    r = tp / truth.size if truth.size else 0.0
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0,
            "predicted": int(pred.size), "true_positives": int(tp)}


def sim_features(keys, n, cols):
    i, j = (keys // n).astype(np.int64), (keys % n).astype(np.int64)
    fn, sn, full, dob, bp, pc, gen, occ = cols
    f = np.zeros((len(keys), 12), dtype=np.float32)
    for k, (x, y) in enumerate(zip(i, j)):
        f[k] = [
            JaroWinkler.similarity(fn[x], fn[y]),
            JaroWinkler.similarity(sn[x], sn[y]),
            JaroWinkler.similarity(full[x], full[y]),
            Levenshtein.normalized_similarity(full[x], full[y]),
            dob[x] == dob[y],
            Levenshtein.normalized_similarity(dob[x], dob[y]),
            bp[x] == bp[y],
            JaroWinkler.similarity(bp[x], bp[y]),
            pc[x] == pc[y],
            pc[x][:4] == pc[y][:4],
            gen[x] == gen[y],
            occ[x] == occ[y] and occ[x] != "",
        ]
    return f


def run_rules(df, n):
    keys = []
    for cols in (["first_name", "surname", "dob"], ["full_name", "dob"],
                 ["first_name", "surname", "postcode_fake"]):
        sub = df.dropna(subset=cols)
        keys.append(pair_keys([g.index.values for _, g in sub.groupby(cols)], n))
    return np.unique(np.concatenate(keys))


def run_splink(df, n, id_to_idx):
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[
            block_on("first_name", "surname"), block_on("surname", "dob"),
            block_on("first_name", "dob"), block_on("postcode_fake"),
            block_on("birth_place", "surname"), block_on("surname", "occupation"),
            block_on("substr(surname, 1, 3)", "substr(first_name, 1, 3)"),
        ],
        comparisons=[
            cl.ForenameSurnameComparison("first_name", "surname"),
            cl.DateOfBirthComparison("dob", input_is_string=True, invalid_dates_as_null=True),
            cl.LevenshteinAtThresholds("postcode_fake", [1, 2]),
            cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
            cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
        ],
        retain_intermediate_calculation_columns=False,
    )
    linker = Linker(df, settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("first_name", "surname", "dob")], recall=0.7)
    linker.training.estimate_u_using_random_sampling(max_pairs=2e6)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("dob"))
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("first_name", "surname"))
    preds = linker.inference.predict(threshold_match_probability=0.5).as_pandas_dataframe(
        limit=5_000_000)
    li = preds.unique_id_l.map(id_to_idx).to_numpy(dtype=np.int64)
    ri = preds.unique_id_r.map(id_to_idx).to_numpy(dtype=np.int64)
    lo, hi = np.minimum(li, ri), np.maximum(li, ri)
    return np.unique(lo * n + hi)


def main():
    from splink import splink_datasets
    df = splink_datasets.historical_50k.reset_index(drop=True)
    for c in ["first_name", "surname", "full_name", "dob", "birth_place",
              "postcode_fake", "gender", "occupation"]:
        df[c] = df[c].fillna("").astype(str).str.lower().str.strip()
    df["surname_init3"] = df.surname.str[:3]
    df["first_init3"] = df.first_name.str[:3]
    n = len(df)
    id_to_idx = {u: i for i, u in enumerate(df.unique_id)}

    truth = true_keys(df, n)
    cands = candidate_keys(df, n)
    ceiling = np.intersect1d(cands, truth, assume_unique=True).size / truth.size
    print(f"records={n:,} clusters={df.cluster.nunique():,} true_pairs={truth.size:,} "
          f"candidates={cands.size:,} blocking_recall={100 * ceiling:.2f}%", flush=True)

    res = {"dataset": "historical_50k (50,578 records, 5,156 people)",
           "true_pairs": int(truth.size), "candidate_pairs": int(cands.size),
           "blocking_recall": round(100 * ceiling, 2)}
    res["rules"] = score(run_rules(df, n), truth); print("rules", res["rules"], flush=True)
    res["splink"] = score(run_splink(df, n, id_to_idx), truth)
    print("splink", res["splink"], flush=True)

    cols = [df[c].to_numpy() for c in ["first_name", "surname", "full_name", "dob",
                                       "birth_place", "postcode_fake", "gender", "occupation"]]
    X = sim_features(cands, n, cols)
    cl_arr = df.cluster.to_numpy()
    ci, cj = cl_arr[cands // n], cl_arr[cands % n]
    y = ci == cj
    clusters = sorted(df.cluster.unique())
    train_cl = set(clusters[: len(clusters) // 2])
    in_train = np.fromiter(((a in train_cl and b in train_cl) for a, b in zip(ci, cj)),
                           dtype=bool, count=len(ci))
    # subsample the training pairs: the full set is millions of rows and mostly easy negatives
    tr_idx = np.flatnonzero(in_train)
    if tr_idx.size > 400_000:
        rng = np.random.default_rng(0)
        pos_idx = tr_idx[y[tr_idx]]
        neg_idx = rng.choice(tr_idx[~y[tr_idx]], 400_000 - pos_idx.size, replace=False)
        tr_idx = np.concatenate([pos_idx, neg_idx])
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=4,
                                class_weight="balanced_subsample").fit(X[tr_idx], y[tr_idx])
    test_keys = cands[~in_train]
    pred_keys = test_keys[rf.predict(X[~in_train])]
    t_ci, t_cj = cl_arr[truth // n], cl_arr[truth % n]
    test_truth = truth[~np.fromiter(((a in train_cl and b in train_cl)
                                     for a, b in zip(t_ci, t_cj)), dtype=bool, count=len(t_ci))]
    res["rf_supervised_heldout_clusters"] = score(pred_keys, test_truth) | {
        "note": "held-out half of the clusters"}
    res["rf_within_candidates"] = score(
        pred_keys, np.intersect1d(test_truth, test_keys, assume_unique=True)) | {
        "note": "same predictions, scored only against true pairs blocking retrieved"}
    print("rf", res["rf_supervised_heldout_clusters"], flush=True)
    print("rf_within_candidates", res["rf_within_candidates"], flush=True)
    json.dump(res, open(ROOT / "results/historical50k_dedupe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
