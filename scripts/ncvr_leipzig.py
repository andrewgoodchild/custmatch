"""custmatch on the Leipzig NCVR benchmark (Saeedi et al.): real North Carolina voter names in
five sources, with duplicates corrupted by the GeCo generator (20% of records, 2 changes each).
Four fields only: given name, surname, suburb (town) and postcode. `recid` is the person.

    python scripts/ncvr_leipzig.py [--people 20000 ...] [--seeds 1,2,3]

Download (CC BY 4.0, 74 MB) into the gitignored data/ncvr_leipzig/:
    https://dbs.uni-leipzig.de/files/datasets/saeedi/5Party-ocp20.tar.gz
and unpack it there. It holds real people's names: never commit it.

For each sample size, a random set of people with all their records across the five sources;
300 random + 700 active labels from the truth; clustered F1 against the true people, and the
ceiling: every true pair blocking proposes, accepted and closed transitively.
With --splink, Splink 4 (Fellegi-Sunter, MIT) runs on each sample too: the same canonical
records and blocking keys, name comparisons with term-frequency adjustment, town by
Jaro-Winkler, postcode by edit distance, m and u trained unsupervised (EM). Three cuts, each
clustered by connected components: Splink's conventional p > 0.99; the match weight whose
clusters score best on the same 1,000 labels custmatch used; and the best weight in hindsight
against the truth (Splink's ceiling, not achievable in practice).
Writes results/ncvr_leipzig.json.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import blocking, io, labels, pipeline
from custmatch.core import partition_scores, profile_contacts

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ncvr_leipzig/5Party-ocp20"
OUT = ROOT / "results/ncvr_leipzig.json"
CONFIG = {"country": "US",
          "extra_fields": [{"name": "town", "compare": "similarity"}, {"name": "postcode", "compare": "edit"}],
          "extra_blocking": [["postcode"]]}
# postcode alone is a block of thousands; with a name's first letter it catches records whose
# two names were both corrupted (the most common miss) - docs/report/01-data.md "NCVR"
CONFIG_KEYS = dict(CONFIG, extra_blocking=[["postcode"], ["postcode", "first_name:first1"],
                                           ["postcode", "last_name:first1"], ["town", "last_name:soundex"]])
LSH = {"fields": ["first_name", "last_name"], "with": "town", "bands": 16, "rows": 3}
VARIANTS = {"default_keys": CONFIG, "postcode_initial_keys": CONFIG_KEYS,
            # from anonlink: one record per source per person; similarity (LSH) blocking
            "postcode_initial_keys_unique_sources": dict(CONFIG_KEYS, unique_sources=True),
            "lsh_blocking": dict(CONFIG, lsh_blocking=LSH),
            "lsh_blocking_unique_sources": dict(CONFIG, lsh_blocking=LSH, unique_sources=True)}


def load():
    frames = []
    for k in range(5):
        d = pd.read_csv(DATA / f"ncvr_numrec_1000000_modrec_2_ocp_20_myp_{k}_nump_5.csv", dtype=str,
                        keep_default_na=False)
        frames.append(d.assign(source=f"p{k}"))
    return pd.concat(frames, ignore_index=True)


def sample(allrec, people, seed):
    ids = allrec.recid.unique()
    keep = set(np.random.default_rng(seed).choice(ids, people, replace=False))
    return allrec[allrec.recid.isin(keep)]


def n_true(ids):
    _, n = np.unique(ids, return_counts=True)
    return float(np.sum(n * (n - 1) / 2))


def splink_scores(df, base, labelled=None):
    """(pairs, match weights) from Splink, with blocking mirroring custmatch's keys."""
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    d = df[["first_name", "last_name", "town", "postcode"]].replace("", None)
    d["unique_id"] = np.arange(len(d))
    d["fi"], d["li"] = df.first_name.str[:1].replace("", None), df.last_name.str[:1].replace("", None)
    d["f3"] = df.first_name.str[:3].replace("", None)
    rules = [block_on("last_name", "f3")]
    if len(base["extra_blocking"]) > 1:
        # town + surname initial: DuckDB has no soundex, and the initial is a superset of
        # custmatch's town + soundex(surname) key (a soundex code starts with the initial)
        rules += [block_on("postcode", "fi"), block_on("postcode", "li"), block_on("town", "li")]
    settings = SettingsCreator(
        link_type="dedupe_only", blocking_rules_to_generate_predictions=rules,
        comparisons=[cl.NameComparison("first_name").configure(term_frequency_adjustments=True),
                     cl.NameComparison("last_name").configure(term_frequency_adjustments=True),
                     cl.JaroWinklerAtThresholds("town", [0.9, 0.8]),
                     cl.LevenshteinAtThresholds("postcode", [1, 2])],
        retain_intermediate_calculation_columns=False)
    linker = Linker(d, settings, db_api=DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match([block_on("first_name", "last_name", "postcode")],
                                                                  recall=0.8)
    linker.training.estimate_u_using_random_sampling(max_pairs=5e6)
    if labelled is None:
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("postcode"))
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("first_name", "last_name"))
    else:                                      # Splink's supervised option: m from labelled pairs
        i, j, y = labelled
        lab = pd.DataFrame({"unique_id_l": i, "source_dataset_l": "d", "unique_id_r": j,
                            "source_dataset_r": "d", "clerical_match_score": np.asarray(y, float)})
        linker.table_management.register_table(lab, "labels", overwrite=True)
        linker.training.estimate_m_from_pairwise_labels("labels")
    # pairs below p = 0.0001 (match weight about -13) are left in Splink: no useful threshold is
    # that low (the best found were -9 and above), and returning all of them ran out of memory
    p = linker.inference.predict(threshold_match_probability=1e-4).as_pandas_dataframe()
    pairs = np.c_[np.minimum(p.unique_id_l, p.unique_id_r), np.maximum(p.unique_id_l, p.unique_id_r)]
    return pairs.astype(np.int64), p.match_weight.to_numpy()


def splink_results(df, base, pid, lab, keys):
    import time
    pos = {k: i for i, k in enumerate(keys)}
    li = np.array([pos[a] for a in lab.a])
    lj = np.array([pos[b] for b in lab.b])
    ly = lab.label.astype(int).to_numpy()
    out = {}
    rnd = (lab.by != "active").to_numpy() if "by" in lab.columns else np.ones(len(lab), bool)
    for name, labelled in (("unsupervised", None), ("m_from_labels", (li, lj, ly)),
                           # only the random labels: the 700 active ones were chosen by custmatch's model
                           ("m_from_random_labels", (li[rnd], lj[rnd], ly[rnd]))):
        t0 = time.time()
        pairs, w = splink_scores(df, base, labelled)
        out[name] = _cuts(df, pairs, w, pid, li, lj, ly) | {"seconds": round(time.time() - t0, 1)}
    return out


def _cuts(df, pairs, w, pid, li, lj, ly):
    from custmatch.core import components

    def at(cut):
        cc = components(len(df), pairs, w > cut)
        return cc, partition_scores(cc, pid, n_true(pid))
    best_lab, best_true = None, None
    for cut in np.arange(-13, 30, 1.0):
        cc, sc = at(cut)
        same = cc[li] == cc[lj]
        tp, fp, fn = int((same & (ly == 1)).sum()), int((same & (ly == 0)).sum()), int((~same & (ly == 1)).sum())
        lf1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if best_lab is None or lf1 > best_lab[0]:
            best_lab = (lf1, cut, sc)
        if best_true is None or sc["f1"] > best_true[1]["f1"]:
            best_true = (cut, sc)
    return {"p_0.99": at(np.log2(0.99 / 0.01))[1],
            "threshold_from_labels": best_lab[2] | {"match_weight": float(best_lab[1])},
            "best_in_hindsight": best_true[1] | {"match_weight": float(best_true[0])}}


def run(sub, tmp, seed, base=CONFIG, splink=False):
    cfg = dict(base, sources=[])
    for s, g in sub.groupby("source"):
        g = g.reset_index(drop=True).assign(rid=lambda d: d.recid + "_" + d.index.astype(str))
        g.to_parquet(tmp / f"{s}.parquet", index=False)
        cfg["sources"].append({"name": s, "path": f"{s}.parquet", "columns": {
            "record_id": "rid", "first_name": "givenname", "last_name": "surname",
            "town": "suburb", "postcode": "postcode"}})
    cp = tmp / "config.json"
    cp.write_text(json.dumps(cfg))
    person = lambda k: k.split(":")[1].rsplit("_", 1)[0]  # noqa: E731

    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(person(a) == person(b)) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(path, index=False)
        return lab
    lp = tmp / "labels.csv"
    labels.sample(cp, 300, lp, seed=seed)
    lab = fill(lp)
    while len(lab) < 1000:
        labels.sample_active(cp, lp, 100, tmp / "next.csv", seed=seed)
        lab = pd.concat([lab, fill(tmp / "next.csv")], ignore_index=True)
        lab.to_csv(lp, index=False)
    s = pipeline.run(cp, lp, tmp / "out", seed=seed, log=lambda *_: None)
    c = pd.read_csv(tmp / "out" / "clusters.csv", dtype=str)
    pid = np.array([person(k) for k in c.source + ":" + c.record_id])
    res = {"records": int(len(c)), "people": int(len(set(pid))), "true_pairs": int(n_true(pid)),
           "clustered": partition_scores(c.customer_id.to_numpy(), pid, n_true(pid)),
           "threshold": s["threshold"], "candidate_pairs": s["candidate_pairs"],
           "seconds": round(sum(s["stage_seconds"].values()), 1)}
    # the ceiling: every true candidate pair accepted and closed
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    df, cfg2 = io.load(cp)
    k = io.key(df)
    p2 = np.array([person(x) for x in k])
    pairs, *_ = blocking.candidates_for(df, {f: profile_contacts(df, f) for f in ("email", "phone")}, cfg2)
    tp = pairs[p2[pairs[:, 0]] == p2[pairs[:, 1]]]
    g = coo_matrix((np.ones(len(tp)), (tp[:, 0], tp[:, 1])), shape=(len(df), len(df)))
    res["ceiling"] = partition_scores(connected_components(g, directed=False)[1], p2, n_true(p2))
    if splink:
        res["splink"] = splink_results(df, base, p2, pd.read_csv(lp, dtype=str, keep_default_na=False), k)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", default="20000")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--splink", action="store_true")
    a = ap.parse_args()
    allrec = load()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for people in [int(x) for x in a.people.split(",")]:
        for variant in a.variants.split(","):
            key = f"{variant}/people_{people}"
            runs = [r for r in res.get(key, {}).get("runs", []) if r.get("seed", 1) not in {int(x) for x in a.seeds.split(",")}]
            for seed in [int(x) for x in a.seeds.split(",")]:
                with tempfile.TemporaryDirectory() as tmp:
                    r = run(sample(allrec, people, seed), Path(tmp), seed, VARIANTS[variant], a.splink)
                prev = next((x for x in res.get(key, {}).get("runs", []) if x.get("seed") == seed), None)
                if not a.splink and prev is not None and "splink" in prev:
                    r["splink"] = prev["splink"]          # Splink does not depend on custmatch's threshold
                runs.append(r | {"seed": seed})
                print(variant, people, seed, {k: r[k] for k in ("records", "clustered", "ceiling", "candidate_pairs",
                                                               "seconds", "splink") if k in r}, flush=True)
            runs.sort(key=lambda r: r["seed"])
            res[key] = {
                "runs": runs, "mean_f1": round(float(np.mean([r["clustered"]["f1"] for r in runs])), 2),
                "mean_ceiling_f1": round(float(np.mean([r["ceiling"]["f1"] for r in runs])), 2)}
            # merge with the file as it is now: another run may have written other variants
            latest = json.loads(OUT.read_text()) if OUT.exists() else {}
            OUT.write_text(json.dumps(latest | {key: res[key]}, indent=1, default=float))


if __name__ == "__main__":
    main()
