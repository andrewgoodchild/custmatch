"""custmatch on the sparse Australian customer file (research/scripts/au_customer_data.py), as a user
would run it: `custmatch sample` draws 1,000 pairs, the truth labels them, `custmatch run`
matches, and the clusters are scored against true people and households.

Scored on all records, on records with no surname (the sparse ones), and on households.
Each variant is a config override, run over three label draws (seeds): with ~4% of
candidate pairs matching, 1,000 random labels hold ~45 matches and one draw is noisy.

    python research/scripts/au_validate.py [variant ...]      # default: all variants

Writes research/results/au_validation.json (merging with variants already there).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import labels, pipeline
from custmatch.core import partition_scores

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "examples/au/config.json"
DATA = ROOT / "data/au_customer"
OUT = ROOT / "research/results/au_validation.json"
SOURCES = ("web", "app", "store")
SEEDS = [1, 2, 3]
OFF = {"phone": False, "dates": False, "email": False, "nicknames": False, "email_user": False}
# each adds one normalisation step (custmatch/country.py) to the one before
VARIANTS = {"baseline": {},
            "phone": {"country": "AU", "normalise": OFF | {"phone": True}},
            "phone_dates": {"country": "AU", "normalise": OFF | {"phone": True, "dates": True}},
            "phone_dates_email": {"country": "AU", "normalise": OFF | {"phone": True, "dates": True,
                                                                      "email": True}},
            "plus_nicknames": {"country": "AU", "normalise": {"email_user": False}},
            "country_au": {"country": "AU"},
            "baseline_active": {"_active": True},
            "country_au_active": {"country": "AU", "_active": True},
            "country_au_active_twin_guard": {"country": "AU", "_active": True, "twin_guard": True},
            # the opt-in match weight (custmatch/matchweight.py): does it cost sign-ups here?
            "country_au_match_weight": {"country": "AU", "match_weight_feature": True},
            "country_au_active_match_weight": {"country": "AU", "_active": True, "match_weight_feature": True},
            # names in Chinese / Arabic script (--data au_customer_script): transliteration off vs on
            "country_au_active_no_script": {"country": "AU", "normalise": {"script": False}, "_active": True},
            # leave-one-out with active learning, where runs are stable: what each step is worth
            **{f"active_without_{step}": {"country": "AU", "normalise": {step: False}, "_active": True}
               for step in ("phone", "dates", "email", "nicknames", "local_nicknames",
                            "email_user", "postcode")}}


SCRIPT_PEOPLE = set()     # people with a record in Chinese or Arabic script (--data au_customer_script)


def truth():
    t = pd.concat([pd.read_parquet(DATA / f"source_{s}.parquet").assign(source=s) for s in SOURCES])
    k = t.source + ":" + t.record_id.astype(str)
    flags = t if "script" in t.columns else None
    if flags is None and (ROOT / "data/au_customer_script/source_store.parquet").exists():
        flags = pd.read_parquet(ROOT / "data/au_customer_script/source_store.parquet")   # same people, Latin file
    if flags is not None:
        SCRIPT_PEOPLE.update(flags.person_id[flags.script.fillna(False).astype(bool)])
    return dict(zip(k, zip(t.person_id, t.household_id))), set(k[t.last_name.str.strip() == ""])


def fill(path, gold):
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    lab["label"] = [int(gold[a][0] == gold[b][0]) for a, b in zip(lab.a, lab.b)]
    lab.to_csv(path, index=False)
    return lab


def n_true(ids):
    _, n = np.unique(ids, return_counts=True)
    return float(np.sum(n * (n - 1) / 2))


def touching(cust, pid, sparse):
    """Pairwise precision / recall / F1 over the pairs with at least one record that has no
    surname - a web or app sign-up joining (or kept from) anyone. Counts all pairs of a group
    minus the pairs among its full records."""
    def pairs(*keys):
        d = pd.DataFrame({f"k{n}": k for n, k in enumerate(keys)} | {"full": ~sparse})
        g = d.groupby([f"k{n}" for n in range(len(keys))]).full.agg(["size", "sum"])
        return int((g["size"] * (g["size"] - 1) // 2 - g["sum"] * (g["sum"] - 1) // 2).sum())
    tp, true, pred = pairs(cust, pid), pairs(pid), pairs(cust)
    p, r = (tp / pred if pred else 1.0), (tp / true if true else 1.0)
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0, "true_pairs": true}


def score(out_dir, gold, no_surname):
    c = pd.read_csv(out_dir / "clusters.csv", dtype=str)
    k = (c.source + ":" + c.record_id).to_numpy()
    pid = np.array([gold[x][0] for x in k])
    hh = np.array([gold[x][1] for x in k])
    cust = c.customer_id.to_numpy()
    sparse = np.array([x in no_surname for x in k])
    return {"all_records": partition_scores(cust, pid, n_true(pid)),
            "no_surname_records": partition_scores(cust[sparse], pid[sparse], n_true(pid[sparse])),
            "pairs_with_a_sign_up": touching(cust, pid, sparse),
            "households": partition_scores(c.household_id.to_numpy(), hh, n_true(hh)),
            **({"native_script_people": partition_scores(cust[np.isin(pid, list(SCRIPT_PEOPLE))],
                                                         pid[np.isin(pid, list(SCRIPT_PEOPLE))],
                                                         n_true(pid[np.isin(pid, list(SCRIPT_PEOPLE))]))}
               if SCRIPT_PEOPLE else {}),
            "summary": {k: v for k, v in json.loads((out_dir / "summary.json").read_text()).items()
                        if k in ("records", "customers", "threshold", "largest_customer",
                                 "candidate_pairs", "junk_contact_values", "shared_contact_values")}}


def ceilings(config_path, gold):
    """What any matcher could reach with this blocking: accept every true candidate pair and
    close transitively. And households from the TRUE identities - the limit of householding
    when half the people have no address on any record."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from custmatch import blocking, io
    from custmatch import households as H
    from custmatch.core import profile_contacts
    df, cfg = io.load(config_path)
    k = io.key(df)
    pid = np.array([gold[x][0] for x in k])
    hh = np.array([gold[x][1] for x in k])
    prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
    pairs, *_ = blocking.candidates_for(df, prof, cfg)
    tp = pairs[pid[pairs[:, 0]] == pid[pairs[:, 1]]]
    g = coo_matrix((np.ones(len(tp)), (tp[:, 0], tp[:, 1])), shape=(len(df), len(df)))
    comp = connected_components(g, directed=False)[1]
    cust = {f: H.profile_by_customer(df, f, pid) for f in ("email", "phone")}
    households = H.assign(df, pid, cust)[0]
    has_addr = pd.Series((df.addr_street != "").to_numpy()).groupby(pid).any()
    return {"candidate_pairs": int(len(pairs)),
            "blocking_pair_recall": round(100 * len(tp) / n_true(pid), 2),
            "blocking_and_closure": partition_scores(comp, pid, n_true(pid)),
            "households_from_true_identities": partition_scores(households, hh, n_true(hh)),
            "households_from_true_identities_postcode_join": partition_scores(
                H.assign(df, pid, cust, postcode_join=True)[0], hh, n_true(hh)),
            "people_with_an_address": round(100 * float(has_addr.mean()), 1)}


def main():
    global DATA, OUT
    args = sys.argv[1:]
    data_name = "au_customer"
    if "--data" in args:          # e.g. --data au_customer_script: another build of the file
        data_name = args.pop(args.index("--data") + 1)
        args.remove("--data")
        DATA, OUT = ROOT / "data" / data_name, ROOT / f"research/results/au_validation_{data_name.removeprefix('au_customer_')}.json"
    wanted = args or list(VARIANTS) + ["ceilings"]
    if "ceilings" in wanted:
        wanted.remove("ceilings")
        gold, _ = truth()
        cfg = json.loads(CONFIG.read_text()) | {"country": "AU"}
        for s in cfg["sources"]:
            s["path"] = str((CONFIG.parent / s["path"]).resolve())
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "c.json").write_text(json.dumps(cfg))
            c = ceilings(Path(tmp) / "c.json", gold)
        res = json.loads(OUT.read_text()) if OUT.exists() else {}
        res["ceilings_country_au"] = c
        OUT.write_text(json.dumps(res, indent=1, default=float))
        print("ceilings", json.dumps(c))
    gold, no_surname = truth()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = json.loads(CONFIG.read_text())
        for s in base["sources"]:
            s["path"] = str((CONFIG.parent / s["path"].replace("/au_customer/", f"/{data_name}/")).resolve())
        for name in wanted:
            spec = VARIANTS[name]
            cfg = json.loads(json.dumps(base)) | {k: v for k, v in spec.items() if not k.startswith("_")}
            cp = tmp / f"{name}.json"
            cp.write_text(json.dumps(cfg))
            runs = []
            for seed in SEEDS:
                lp = tmp / f"{name}_{seed}_labels.csv"
                # 1,000 random labels, or 300 random then 700 chosen by active learning 100 at a time
                labels.sample(cp, 300 if spec.get("_active") else 1000, lp, seed=seed)
                lab = fill(lp, gold)
                while len(lab) < 1000:
                    nxt = tmp / "next.csv"
                    labels.sample_active(cp, lp, 100, nxt, seed=seed)
                    lab = pd.concat([lab, fill(nxt, gold)], ignore_index=True)
                    lab.to_csv(lp, index=False)
                out = tmp / f"{name}_{seed}"
                pipeline.run(cp, lp, out, log=lambda *_: None)
                runs.append(score(out, gold, no_surname) | {"labelled_matches": int(lab.label.astype(int).sum())})
            res[name] = {"seeds": SEEDS, "runs": runs, "mean": {
                m: {k: round(float(np.mean([r[m][k] for r in runs])), 2) for k in ("precision", "recall", "f1")}
                for m in ("all_records", "no_surname_records", "pairs_with_a_sign_up", "households",
                          "native_script_people") if m in runs[0]},
                "f1_sd": round(float(np.std([r["all_records"]["f1"] for r in runs])), 2)}
            r = res[name]["mean"]
            print(f"{name:22s} clustered F1 {r['all_records']['f1']:6.2f} +- {res[name]['f1_sd']:5.2f} "
                  f"(P {r['all_records']['precision']:.1f} R {r['all_records']['recall']:.1f})  "
                  f"no-surname {r['no_surname_records']['f1']:6.2f}  households {r['households']['f1']:6.2f}  "
                  + (f"native-script people {r['native_script_people']['f1']:6.2f}  " if "native_script_people" in r else "")
                  + f"labelled matches {[x['labelled_matches'] for x in runs]}", flush=True)
            OUT.write_text(json.dumps(res, indent=1, default=float))

if __name__ == "__main__":
    main()
