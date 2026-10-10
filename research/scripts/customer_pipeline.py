"""A customer-matching pipeline that handles fake contacts and households explicitly.

Stages
  1. profile contacts   - classify every email/phone value as personal / household / junk,
                          unsupervised (value frequency + co-field diversity + syntax)
  2. block              - multi-pass, with hub suppression; junk contacts never used as keys
  3. features           - similarity features; contact agreement weighted by how rare the
                          value is, junk contacts treated as missing
  4. match              - supervised classifier over the candidate pairs
  5. cluster            - connected components with the threshold tuned on the CLUSTERING
                          metric, and merges required to have non-contact evidence
  6. households         - a separate graph layer: records that share a contact/address but
                          are not the same person

`--naive` runs the same pipeline with the pathology handling switched off, which is the
ablation: contacts trusted as strong keys, no junk detection, threshold tuned pairwise.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

pd.set_option("future.infer_string", False)

from custmatch.core import (  # noqa: E402  the stages live in the package
    THRESHOLDS, block, components, features, household_partition, pair_scores,
    partition_scores, profile_contacts)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/customer"

def prepare(aware=True):
    """Stages 1-3 and the train/test split by person - shared with customer_ditto.py."""
    a = pd.read_parquet(DATA / "source_crm.parquet")
    b = pd.read_parquet(DATA / "source_billing.parquet")
    df = pd.concat([a, b], ignore_index=True)
    for col in ("date_of_birth", "ssn", "employer_name"):
        if col not in df:
            df[col] = ""
    df["date_of_birth"] = df.date_of_birth.fillna("").astype(str)
    df["ssn"] = df.ssn.fillna("").astype(str)
    n = len(df)

    profiles = {f: profile_contacts(df, f, aware) for f in ("email", "phone")}
    pairs, hubs = block(df, profiles, aware)
    X = features(df, pairs, profiles, aware)

    sid = df.simulant_id.to_numpy()
    y = sid[pairs[:, 0]] == sid[pairs[:, 1]]
    # every true pair, including ones blocking never proposed
    sizes = df.groupby("simulant_id").size().to_numpy()
    n_true_all = float(np.sum(sizes * (sizes - 1) / 2))

    people = np.array(sorted(df.simulant_id.unique()))
    rng = np.random.default_rng(0)
    rng.shuffle(people)
    train_people = set(people[: len(people) // 2])
    in_train = np.array([s in train_people for s in sid])
    tr = in_train[pairs[:, 0]] & in_train[pairs[:, 1]]
    te = ~in_train[pairs[:, 0]] & ~in_train[pairs[:, 1]]
    return dict(df=df, n=n, profiles=profiles, pairs=pairs, hubs=hubs, X=X, y=y,
                n_true_all=n_true_all, in_train=in_train, tr=tr, te=te)


def run(aware=True):
    P = prepare(aware)
    df, n, profiles, pairs, hubs, X, y = (P[k] for k in ("df", "n", "profiles", "pairs", "hubs", "X", "y"))
    n_true_all, in_train, tr, te = P["n_true_all"], P["in_train"], P["tr"], P["te"]

    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4,
                                class_weight="balanced_subsample").fit(X[tr], y[tr])
    prob = rf.predict_proba(X)[:, 1]

    test_sizes = df[~in_train].groupby("simulant_id").size().to_numpy()
    n_true_test = float(np.sum(test_sizes * (test_sizes - 1) / 2))
    true_labels = df.simulant_id.to_numpy()

    sweep = []
    for th in THRESHOLDS:
        labels = components(n, pairs, te & (prob > th))
        mask = ~in_train
        sweep.append({
            "threshold": float(th),
            "pairwise": pair_scores(y[te], prob[te] > th, n_true_test),
            "clustered": partition_scores(labels[mask], true_labels[mask], n_true_test),
        })
    best_pair = max(sweep, key=lambda s: s["pairwise"]["f1"])
    best_clu = max(sweep, key=lambda s: s["clustered"]["f1"])
    chosen = best_pair if not aware else best_clu   # the ablation: which metric you tune on

    # persist the identity clusters so downstream stages (golden records) can use them
    if aware:
        out_labels = components(n, pairs, prob > chosen["threshold"])
        pd.DataFrame({"record_id": df.record_id.to_numpy(),
                      "simulant_id": df.simulant_id.to_numpy(),
                      "cluster": out_labels}).to_parquet(
            ROOT / "research/results/customer_clusters.parquet", index=False)

    # ---- stage 6: households (aware runs only)
    hh = None
    if aware:
        th = chosen["threshold"]
        identity_labels = components(n, pairs, prob > th)
        hh_labels, big = household_partition(df, profiles, identity_labels)
        hh_sizes = df.groupby("household_id").size().to_numpy()
        hh = partition_scores(hh_labels, df.household_id.to_numpy(),
                              float(np.sum(hh_sizes * (hh_sizes - 1) / 2)))
        hh["group_quarters_sized"] = int(big)

    # ---- contact classification quality.
    # Gold for "household" is OBSERVED sharing - a value carried by 2+ distinct people in the
    # data. A value the generator labelled household but that only one person ever used is
    # indistinguishable from a personal one, and scoring against intent would be unfair.
    cls_eval = {}
    if aware:
        truth = pd.read_parquet(DATA / "contact_truth.parquet")
        junk_values = set(truth.loc[truth.contact_class == "junk", "email"]) | \
                      set(truth.loc[truth.contact_class == "junk", "phone"])
        for field in ("email", "phone"):
            sub = df[df[field] != ""]
            people_per_value = sub.groupby(field).simulant_id.nunique()
            gold = pd.DataFrame({
                "value": people_per_value.index,
                "shared": (people_per_value > 1).to_numpy(),
            })
            gold["is_junk"] = gold.value.isin(junk_values)
            gold["pred"] = [profiles[field].get(v, ("personal", 1.0))[0] for v in gold.value]
            for target, mask in (("junk", gold.is_junk),
                                 ("household", gold.shared & ~gold.is_junk)):
                got = gold.pred == target
                tp, fp, fn = int((mask & got).sum()), int((~mask & got).sum()), int((mask & ~got).sum())
                pr = tp / (tp + fp) if tp + fp else 0.0
                rc = tp / (tp + fn) if tp + fn else 0.0
                cls_eval[f"{field}_{target}"] = {
                    "precision": round(100 * pr, 2), "recall": round(100 * rc, 2),
                    "f1": round(200 * pr * rc / (pr + rc), 2) if pr + rc else 0.0,
                    "gold_values": int(mask.sum())}

    return {"mode": "pathology_aware" if aware else "naive",
            "records": n, "candidate_pairs": int(len(pairs)), "hubs_suppressed": hubs,
            "blocking_recall": round(100 * float(y.sum()) / n_true_all, 2),
            "tuned_on": "clustered_f1" if aware else "pairwise_f1",
            "chosen_threshold": chosen["threshold"],
            "pairwise": chosen["pairwise"], "clustered": chosen["clustered"],
            "best_possible_clustered": best_clu["clustered"],
            "households": hh, "contact_detection": cls_eval, "sweep": sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive", action="store_true")
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()
    out = {}
    for aware in ([True, False] if args.both else [not args.naive]):
        r = run(aware)
        out[r["mode"]] = r
        print(f"\n=== {r['mode']} (tuned on {r['tuned_on']}, threshold {r['chosen_threshold']})")
        print(f"  candidates={r['candidate_pairs']:,} blocking_recall={r['blocking_recall']}% "
              f"hubs_suppressed={r['hubs_suppressed']}")
        print(f"  pairwise : {r['pairwise']}")
        print(f"  clustered: {r['clustered']}")
        if r["households"]:
            print(f"  households: {r['households']}")
        if r["contact_detection"]:
            for k, v in r["contact_detection"].items():
                print(f"  detect {k:18s}: {v}")
    path = ROOT / "research/results/customer_pipeline.json"
    if path.exists() and not args.both:
        out = {**json.load(open(path)), **out}
    json.dump(out, open(path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
