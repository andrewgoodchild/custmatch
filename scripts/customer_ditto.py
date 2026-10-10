"""Ditto on customer-style records: does the transformer beat the forest here too?

Ditto was only ever run on historical_50k (chapter 2), where it beats the forest by 11 clustered
F1. The customer pipeline (`customer_pipeline.py`) uses a random forest and reaches 97.8. This
asks whether Ditto helps on records that carry addresses, emails and phones.

Same candidate pairs, same people split, same clustering and scoring as the pipeline - taken
from `customer_pipeline.prepare()`. Ditto sees the fields the forest's features use (names,
middle initial, address, email, phone, DOB, SSN), with contacts the pipeline's unsupervised
profiler flags as junk blanked - the forest's features already treat those as missing, so the
matcher is the only difference. Ditto trains on 85% of the pipeline's training people and
picks its epoch on the other 15%.

  customer_ditto.py build      write data/customer_em/{train,valid,test}.txt
  person_ditto.py --data-dir data/customer_em --out customer --max-len 160
  customer_ditto.py evaluate   forest vs Ditto vs forest-veto cascade, clustered
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from customer_pipeline import components, partition_scores, prepare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/customer_em"
THS = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                               [0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]]), 4)
VETO = 0.2


def serialise(r, profiles):
    addr = " ".join(str(x) for x in (r["addr_number"], r["addr_street"], r["addr_unit"]) if str(x).strip())
    vals = {"first_name": r["first_name"], "middle_initial": r["middle_initial"],
            "last_name": r["last_name"], "address": addr, "date_of_birth": r["date_of_birth"],
            "ssn": r["ssn"]}
    for f in ("email", "phone"):
        v = r[f]
        vals[f] = "" if (v and profiles[f].get(v, ("personal",))[0] == "junk") else v
    return " ".join(f"COL {k} VAL {v}" for k, v in vals.items() if str(v).strip())


def split(P):
    """The pipeline's train people, re-split 85/15 into Ditto train and valid."""
    df, pairs, in_train = P["df"], P["pairs"], P["in_train"]
    train_people = np.array(sorted(df.simulant_id[in_train].unique()))
    np.random.default_rng(1).shuffle(train_people)
    valid_people = set(train_people[: int(0.15 * len(train_people))])
    in_valid = df.simulant_id.isin(valid_people).to_numpy()
    va = P["tr"] & in_valid[pairs[:, 0]] & in_valid[pairs[:, 1]]
    tr = P["tr"] & ~in_valid[pairs[:, 0]] & ~in_valid[pairs[:, 1]]
    return tr, va


def build():
    P = prepare(aware=True)
    df, pairs, y = P["df"], P["pairs"], P["y"]
    rows = df.to_dict("records")
    ser = [serialise(r, P["profiles"]) for r in rows]
    tr, va = split(P)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, mask in (("train", tr), ("valid", va), ("test", P["te"])):
        with open(OUT / f"{name}.txt", "w") as f:
            for (a, b), lab in zip(pairs[mask], y[mask]):
                f.write(f"{ser[a]}\t{ser[b]}\t{int(lab)}\n")
        print(f"{name}: {int(mask.sum()):,} pairs, {int(y[mask].sum()):,} positive", flush=True)
    np.save(OUT / "test_pair_index.npy", np.flatnonzero(P["te"]))


def cluster_f1(P, prob, mask_pairs, records, th):
    n, pairs, y_true = P["n"], P["pairs"], P["df"].simulant_id.to_numpy()
    sizes = P["df"][records].groupby("simulant_id").size().to_numpy()
    n_true = float(np.sum(sizes * (sizes - 1) / 2))
    labels = components(n, pairs, mask_pairs & (prob > th))
    return partition_scores(labels[records], y_true[records], n_true)


def evaluate():
    P = prepare(aware=True)
    df, pairs, X, y, te, in_train = P["df"], P["pairs"], P["X"], P["y"], P["te"], P["in_train"]
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4,
                                class_weight="balanced_subsample").fit(X[P["tr"]], y[P["tr"]])
    p_rf = rf.predict_proba(X)[:, 1]
    d = np.load(ROOT / "results/customer_scores_ditto.npz")
    idx = np.load(OUT / "test_pair_index.npy")
    assert (d["y"].astype(bool) == y[idx]).all(), "Ditto test file is out of step with the pipeline"
    p_dt = np.zeros(len(pairs))
    p_dt[idx] = d["prob"]
    configs = {"forest_only": p_rf, "ditto_only": p_dt,
               "forest_veto_then_ditto": np.where(p_rf > VETO, p_dt, 0.0)}

    test_records = ~in_train
    out = {"test_pairs": int(te.sum()), "test_records": int(test_records.sum()),
           "routed_frac": float((p_rf[te] > VETO).mean()), "picked_on_test": {}, "split_half": {}}
    for name, s in configs.items():
        best = max(((th, cluster_f1(P, s, te, test_records, th)) for th in THS), key=lambda z: z[1]["f1"])
        out["picked_on_test"][name] = best[1] | {"threshold": float(best[0])}
        print(f"{name:24s} picked on test: F1 {best[1]['f1']:.2f} @ {best[0]}  "
              f"ARI {best[1]['adjusted_rand']:.4f}", flush=True)

    # split-half by person: choose the threshold on one half of the test people, score the other
    sid = df.simulant_id.to_numpy()
    half_a = np.array([zlib.crc32(str(s).encode()) % 2 == 0 for s in sid])
    halves = {h: test_records & (half_a if h == "A" else ~half_a) for h in "AB"}
    in_half = {h: te & halves[h][pairs[:, 0]] & halves[h][pairs[:, 1]] for h in "AB"}
    for name, s in configs.items():
        f1s = []
        for tune, held in (("A", "B"), ("B", "A")):
            th = max(THS, key=lambda t: cluster_f1(P, s, in_half[tune], halves[tune], t)["f1"])
            f1s.append(cluster_f1(P, s, in_half[held], halves[held], th)["f1"])
        out["split_half"][name] = {"held_out_f1": f1s, "mean": round(float(np.mean(f1s)), 2)}
        print(f"{name:24s} split-half held-out F1 {f1s[0]:.2f} / {f1s[1]:.2f}  mean {np.mean(f1s):.2f}",
              flush=True)
    out["ditto_train"] = json.load(open(ROOT / "results/customer_ditto_train.json"))
    json.dump(out, open(ROOT / "results/customer_ditto.json", "w"), indent=1, default=float)


def golden():
    """Chapter 7's question for the better matcher: does it change the merged records?

    Golden records built from the test records under three clusterings - the forest, forest
    veto then Ditto (each at its picked threshold from `evaluate`), and perfect linkage - and
    scored against pseudopeople's noise-free truth with golden_record.py's own functions."""
    from golden_record import FIELDS, build_golden, score, truth_table
    P = prepare(aware=True)
    df, pairs, X, y, te, in_train = P["df"], P["pairs"], P["X"], P["y"], P["te"], P["in_train"]
    res = json.load(open(ROOT / "results/customer_ditto.json"))
    rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=4,
                                class_weight="balanced_subsample").fit(X[P["tr"]], y[P["tr"]])
    p_rf = rf.predict_proba(X)[:, 1]
    d = np.load(ROOT / "results/customer_scores_ditto.npz")
    p_dt = np.zeros(len(pairs))
    p_dt[np.load(OUT / "test_pair_index.npy")] = d["prob"]
    cascade = np.where(p_rf > VETO, p_dt, 0.0)

    test = ~in_train
    sub = df[test].reset_index(drop=True)
    for f in FIELDS:
        sub[f] = sub[f].fillna("").astype(str).str.lower().str.strip()
    labelings = {}
    for name, s in (("forest_only", p_rf), ("forest_veto_then_ditto", cascade)):
        th = res["picked_on_test"][name]["threshold"]
        labelings[name] = components(P["n"], pairs, te & (s > th))[test]
    labelings["perfect_linkage"] = np.unique(sub.simulant_id.to_numpy(), return_inverse=True)[1]
    truth = truth_table()
    out = {}
    for name, labels in labelings.items():
        out[name] = {st: score(sub, labels, build_golden(sub, labels, st), truth)
                     for st in ("first_seen", "most_complete")}
        print(name, {st: r["all_fields_correct"] for st, r in out[name].items()},
              out[name]["most_complete"]["per_field"], flush=True)
    res["golden_records_on_test"] = out
    json.dump(res, open(ROOT / "results/customer_ditto.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "evaluate": evaluate, "golden": golden}[sys.argv[1]]()
