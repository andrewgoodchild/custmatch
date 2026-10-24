"""Blocking alone on the Leipzig NCVR benchmark as the file grows: how much of the truth the
candidate pairs can reach, at 200,000 people up to all 3,500,840 (5 million records).

The full run fell from 95.63 clustered F1 at 200,000 people to 84.53 at 3.5 million
(docs/report/06-blocking-and-scale.md, "Larger files"). This measures the blocking half of that
without labels or a model, so it runs on a laptop: for each size and blocking mode, the
candidate pairs, the share of true pairs among them (pair completeness), and the ceiling - every
true candidate pair accepted and closed transitively, scored as clustered F1 against the people.

    python research/scripts/ncvr_blocking_scale.py [--people 200000,1000000,2000000,all]
                                                   [--modes core,fixed,adaptive]

Same records and config as ncvr_leipzig.py's "postcode_initial_keys" variant. Writes
research/results/ncvr_blocking_scale.json.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncvr_leipzig as N  # noqa: E402

from custmatch import blocking  # noqa: E402
from custmatch.core import partition_scores, profile_contacts  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/ncvr_blocking_scale.json"
MODES = {"core": {}, "fixed": {"blocking": "fixed"}, "adaptive": {"blocking": "adaptive"}}


def frame(sub):
    """The records as io.load gives them for ncvr_leipzig.run's config, without the parquet trip."""
    from custmatch import io
    import tempfile
    cfg = dict(N.CONFIG_KEYS, sources=[])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for s, g in sub.groupby("source"):
            g = g.reset_index(drop=True).assign(rid=lambda d: d.recid + "_" + d.index.astype(str))
            g.to_parquet(tmp / f"{s}.parquet", index=False)
            cfg["sources"].append({"name": s, "path": f"{s}.parquet", "columns": {
                "record_id": "rid", "first_name": "givenname", "last_name": "surname",
                "town": "suburb", "postcode": "postcode"}})
        (tmp / "config.json").write_text(json.dumps(cfg))
        df, cfg = io.load(tmp / "config.json")
        keys = io.key(df)
    pid = np.array([k.split(":")[1].rsplit("_", 1)[0] for k in keys])
    return df, cfg, pid


def measure(df, cfg, pid, overrides):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
    t0 = time.time()
    pairs, hubs, report, _ = blocking.candidates_for(df, profiles, dict(cfg, **overrides))
    secs = time.time() - t0
    true = pid[pairs[:, 0]] == pid[pairs[:, 1]]
    tp = pairs[true]
    total = N.n_true(pid)
    g = coo_matrix((np.ones(len(tp)), (tp[:, 0], tp[:, 1])), shape=(len(df), len(df)))
    ceiling = partition_scores(connected_components(g, directed=False)[1], pid, total)
    out = {"candidate_pairs": int(len(pairs)), "true_pairs_found": int(true.sum()),
           "pair_completeness": round(100 * float(true.sum()) / total, 2),
           "pairs_per_record": round(len(pairs) / len(df), 2), "ceiling": ceiling,
           "hubs_dropped": int(hubs), "seconds": round(secs, 1)}
    if report.get("mode") == "core":
        out["records_only_in_skipped_blocks"] = report["records_only_in_skipped_blocks"]
        out["skipped_blocks"] = report["skipped_blocks_with_values"]
    else:
        out["oversized_blocks_split"] = sum(r["oversized_blocks_split"] for r in report["rules"])
        out["max_block"] = report.get("max_block")
        if report.get("adaptive"):
            out["adaptive"] = report["adaptive"]
    return out


def splink_datasets(modes):
    """The same measurement on Splink's four datasets (splink_datasets_benchmark.py's configs),
    where blocks rarely outgrow max_block: adaptive splitting should change little there."""
    import tempfile
    import splink_datasets_benchmark as B
    from custmatch import io
    out = {}
    for name, spec in B.datasets().items():
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = {"sources": [], "extra_fields": spec["extra_fields"], "extra_blocking": spec["extra_blocking"]}
            truth_of = {}
            for src, f in spec["frames"].items():
                f.fillna("").astype(str).to_csv(tmp / f"{src}.csv", index=False)
                cfg["sources"].append({"name": src, "path": f"{src}.csv", "columns": spec["columns"]})
                truth_of.update(zip(src + ":" + f[spec["id_col"]].astype(str), f[spec["truth_col"]].astype(str)))
            (tmp / "config.json").write_text(json.dumps(cfg))
            df, cfg = io.load(tmp / "config.json")
            pid = np.array([truth_of[k] for k in io.key(df)])
        for mode in modes:
            r = measure(df, cfg, pid, MODES[mode])
            out[f"splink_datasets/{name}/{mode}"] = r
            print(name, mode, r["candidate_pairs"], r["pair_completeness"], "ceiling F1", r["ceiling"]["f1"], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", default="200000,1000000,2000000,all")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--splink-datasets", action="store_true", help="Splink's four datasets instead of NCVR")
    a = ap.parse_args()
    if a.splink_datasets:
        res = splink_datasets(a.modes.split(","))
        OUT.write_text(json.dumps((json.loads(OUT.read_text()) if OUT.exists() else {}) | res, indent=1, default=float))
        return
    allrec = N.load()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for people in a.people.split(","):
        sub = allrec if people == "all" else N.sample(allrec, int(people), a.seed)
        df, cfg, pid = frame(sub)
        for mode in a.modes.split(","):
            r = measure(df, cfg, pid, MODES[mode])
            r |= {"records": int(len(df)), "people": int(len(set(pid))), "true_pairs": int(N.n_true(pid)),
                  "peak_memory_gb_so_far": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3, 2)}
            res[f"{mode}/people_{people}"] = r
            print(mode, people, {k: r[k] for k in ("records", "candidate_pairs", "pair_completeness",
                                                    "seconds")}, "ceiling F1", r["ceiling"]["f1"], flush=True)
            latest = json.loads(OUT.read_text()) if OUT.exists() else {}
            OUT.write_text(json.dumps(latest | {f"{mode}/people_{people}": r}, indent=1, default=float))


if __name__ == "__main__":
    main()
