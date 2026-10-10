"""Person/customer record linkage on FEBRL (synthetic Australian person records).

FEBRL4: link file A (5,000) to file B (5,000); 5,000 true links (rec-N-org <-> rec-N-dup-0).
Methods, all scored as pairwise precision / recall / F1 against ALL true links
(so blocking misses count against recall):
  1. Deterministic rules (exact soc_sec_id, or exact given_name+surname+dob)
  2. recordlinkage ECM (unsupervised Fellegi-Sunter, binary features)
  3. Splink 4 (Fellegi-Sunter with fuzzy levels, unsupervised EM + u from random sampling)
  4. Random forest on similarity features, trained on FEBRL3 (a different file), tested on FEBRL4
"""
import json
import warnings
from pathlib import Path

import numpy as np
import recordlinkage as rl
from recordlinkage.datasets import load_febrl3, load_febrl4
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
FIELDS = ["given_name", "surname", "street_number", "address_1", "address_2",
          "suburb", "postcode", "state", "date_of_birth", "soc_sec_id"]


def clean(df):
    df = df.copy()
    for c in FIELDS:
        df[c] = df[c].astype("string").str.strip().str.lower()
    return df


def entity(rec_id):  # rec-1070-org / rec-1070-dup-0 -> 1070
    return rec_id.split("-")[1]


def score(pred_pairs, true_pairs):
    pred, true = set(pred_pairs), set(true_pairs)
    tp = len(pred & true)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(true)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2), "f1": round(100 * f, 2),
            "predicted": len(pred), "true_positives": tp}


def candidates(a, b=None):
    """Union of simple blocking keys (standard multi-pass blocking)."""
    idx = []
    for keys in (["given_name"], ["surname"], ["date_of_birth"], ["postcode"], ["soc_sec_id"]):
        ix = rl.Index()
        ix.block(keys)
        idx.append(ix.index(a) if b is None else ix.index(a, b))
    out = idx[0]
    for i in idx[1:]:
        out = out.union(i)
    return out


def compare_features(pairs, a, b):
    c = rl.Compare(n_jobs=4)
    c.string("given_name", "given_name", method="jarowinkler", label="given_name")
    c.string("surname", "surname", method="jarowinkler", label="surname")
    c.string("address_1", "address_1", method="damerau_levenshtein", label="address_1")
    c.string("address_2", "address_2", method="damerau_levenshtein", label="address_2")
    c.string("suburb", "suburb", method="damerau_levenshtein", label="suburb")
    c.exact("street_number", "street_number", label="street_number")
    c.exact("postcode", "postcode", label="postcode")
    c.exact("state", "state", label="state")
    c.string("date_of_birth", "date_of_birth", method="levenshtein", label="date_of_birth")
    c.string("soc_sec_id", "soc_sec_id", method="levenshtein", label="soc_sec_id")
    c.exact("given_name", "surname", label="swap_gs")  # transposed first/last name
    return c.compute(pairs, a, b).fillna(0)


def run_rules(a, b):
    m1 = a.reset_index().merge(b.reset_index(), on="soc_sec_id", suffixes=("_a", "_b"))
    m2 = a.reset_index().merge(b.reset_index(), on=["given_name", "surname", "date_of_birth"],
                               suffixes=("_a", "_b"))
    return set(zip(m1.rec_id_a, m1.rec_id_b)) | set(zip(m2.rec_id_a, m2.rec_id_b))


def run_ecm(feats):
    binary = (feats >= 0.85).astype(int)
    ecm = rl.ECMClassifier(binarize=None)
    matches = ecm.fit_predict(binary)
    return set(matches)


def run_splink(a, b):
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    da, db = a.reset_index(), b.reset_index()
    settings = SettingsCreator(
        link_type="link_only",
        unique_id_column_name="rec_id",
        blocking_rules_to_generate_predictions=[
            block_on("given_name"), block_on("surname"), block_on("date_of_birth"),
            block_on("postcode"), block_on("soc_sec_id"),
        ],
        comparisons=[
            cl.NameComparison("given_name"),
            cl.NameComparison("surname"),
            cl.DateOfBirthComparison("date_of_birth", input_is_string=True,
                                     datetime_format="%Y%m%d", invalid_dates_as_null=True),
            cl.DamerauLevenshteinAtThresholds("soc_sec_id", [1, 2]),
            cl.ExactMatch("street_number"),
            cl.JaroWinklerAtThresholds("address_1", [0.9, 0.8]),
            cl.JaroWinklerAtThresholds("address_2", [0.9, 0.8]),
            cl.JaroWinklerAtThresholds("suburb", [0.9, 0.8]),
            cl.PostcodeComparison("postcode"),
            cl.ExactMatch("state"),
        ],
    )
    linker = Linker([da, db], settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match(
        [block_on("given_name", "surname", "date_of_birth"), block_on("soc_sec_id")], recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=5e6)
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("date_of_birth"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("given_name", "surname"))
    preds = linker.inference.predict(threshold_match_probability=0.5).as_pandas_dataframe()
    # Splink orders each pair by id; restore (A, B) orientation
    out = set()
    for l, r in zip(preds.rec_id_l, preds.rec_id_r):
        out.add((l, r) if l in a.index else (r, l))
    return out


def run_rf(a4, b4, pairs4, feats4):
    d3 = clean(load_febrl3())
    pairs3 = candidates(d3)
    f3 = compare_features(pairs3, d3, d3)
    y3 = np.array([entity(i) == entity(j) for i, j in pairs3])
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4).fit(f3.values, y3)
    pred = rf.predict(feats4.values)
    return set(p for p, m in zip(pairs4, pred) if m), int(y3.sum()), len(pairs3)


def main():
    a, b, links = load_febrl4(return_links=True)
    a, b = clean(a), clean(b)
    truth = set(links)
    pairs = candidates(a, b)
    feats = compare_features(pairs, a, b)
    blocking_recall = len(set(pairs) & truth) / len(truth)
    print(f"candidates={len(pairs):,}  blocking recall={100 * blocking_recall:.2f}%", flush=True)

    res = {"dataset": "FEBRL4 (5,000 x 5,000; 5,000 true links)",
           "candidate_pairs": len(pairs), "blocking_recall": round(100 * blocking_recall, 2)}
    res["rules"] = score(run_rules(a, b), truth); print("rules", res["rules"], flush=True)
    res["ecm"] = score(run_ecm(feats), truth); print("ecm", res["ecm"], flush=True)
    res["splink"] = score(run_splink(a, b), truth); print("splink", res["splink"], flush=True)
    rf_pred, n_pos3, n_pairs3 = run_rf(a, b, pairs, feats)
    res["rf_trained_on_febrl3"] = score(rf_pred, truth) | {"train_pairs": n_pairs3, "train_pos": n_pos3}
    print("rf", res["rf_trained_on_febrl3"], flush=True)
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / "febrl4_person_matching.json", "w"), indent=1)


if __name__ == "__main__":
    main()
