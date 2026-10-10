"""Why historical_50k scores lower: how much each blocking change lifts the ceiling.

The ceiling is the clustered F1 with a perfect matcher: every true candidate pair accepted,
none false, then connected components. On historical_50k (about 10 records per person, half of
them missing the surname or birth date, names given as "tom baron chudleigh") the default keys
leave a ceiling near 93, and the matcher reaches about 87.

    python research/scripts/historical_blocking.py                 the ceilings
    python research/scripts/historical_blocking.py --run V1,V2     also custmatch end to end (rich features;
                                                          1,000 random labels, and 300 random + 700
                                                          active; label seeds 1, 2, 3)
Writes research/results/historical_blocking.json.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import beat_splink as BS  # noqa: E402
import splink_datasets_benchmark as B  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.blocking import candidates_for  # noqa: E402
from custmatch.core import components, partition_scores, profile_contacts  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/historical_blocking.json"
VARIANTS = {
    "default": {"lsh_blocking": None},              # the exact keys alone
    "lsh_names_birth_place": {"lsh_blocking": {"fields": ["first_name", "last_name"], "with": "birth_place"}},
    "gb": {"country": "GB", "lsh_blocking": None},
    "gb_lsh_names_birth_place": {"country": "GB",
                                 "lsh_blocking": {"fields": ["first_name", "last_name"], "with": "birth_place"}},
}


def run(cp, truth, tmp, tag, seed, active):
    """custmatch as a user runs it, labels from the truth; clustered F1 and the threshold."""
    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(truth[a] == truth[b]) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(path, index=False)
        return lab
    lp = tmp / f"{tag}.csv"
    labels.sample(cp, 300 if active else 1000, lp, seed=seed)
    lab = fill(lp)
    while active and len(lab) < 1000:
        labels.sample_active(cp, lp, 100, tmp / "next.csv", seed=seed)
        lab = pd.concat([lab, fill(tmp / "next.csv")], ignore_index=True)
        lab.to_csv(lp, index=False)
    t0 = time.time()
    s = pipeline.run(cp, lp, tmp / f"{tag}_out", seed=seed, log=lambda *_: None)
    c = pd.read_csv(tmp / f"{tag}_out" / "clusters.csv", dtype=str)
    t = np.array([truth[f"{a}:{b}"] for a, b in zip(c.source, c.record_id)])
    return {"f1": B.clustered_f1(c.customer_id.to_numpy(), t)["f1"], "threshold": s["threshold"],
            "seconds": round(time.time() - t0, 1)}


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    runs = sys.argv[sys.argv.index("--run") + 1].split(",") if "--run" in sys.argv else []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config, truth = BS.setup("historical_50k", tmp)
        for name, over in VARIANTS.items():
            if runs and name not in runs:
                continue
            cfg = json.loads(Path(config).read_text())
            for s in cfg["sources"]:
                s["path"] = str((Path(config).parent / s["path"]).resolve())
            cfg.update({"rich_features": True} | {k: v for k, v in over.items() if not k.startswith("_")})
            cp = tmp / f"{name}.json"
            cp.write_text(json.dumps(cfg))
            df, cfg2 = io.load(cp)
            pid = np.array([truth[k] for k in io.key(df)])
            prof = {f: profile_contacts(df, f) for f in ("email", "phone")}
            t0 = time.time()
            pairs, hubs, _, _ = candidates_for(df, prof, cfg2)
            _, n = np.unique(pid, return_counts=True)
            tot = float(np.sum(n * (n - 1) / 2))
            true = pid[pairs[:, 0]] == pid[pairs[:, 1]]
            ceil = partition_scores(components(len(df), pairs, true), pid, tot)
            res[name] = res.get(name, {}) | {
                "candidates": int(len(pairs)), "true_candidates": int(true.sum()),
                "true_pairs_found_directly_pct": round(100 * true.sum() / tot, 2),
                "ceiling": ceil, "seconds": round(time.time() - t0, 1)}
            print(name, res[name], flush=True)
            if name in runs:
                for active in (False, True):
                    for seed in (1, 2, 3):
                        tag = f"{name}_{'active' if active else 'random'}_{seed}"
                        r = run(cp, truth, tmp, tag, seed, active)
                        res[name].setdefault("runs", {})[tag.removeprefix(name + "_")] = r
                        print(tag, r, flush=True)
                        OUT.write_text(json.dumps(res, indent=1))
                f = [v["f1"] for v in res[name]["runs"].values()]
                res[name]["mean_f1"] = round(float(np.mean(f)), 2)
            OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
