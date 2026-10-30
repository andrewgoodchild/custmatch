"""Do the cross-fitting's fold models need 300 trees? Each fold model only scores the candidate
pairs to choose the threshold; the final model, which scores them for the clusters, keeps 300.

For each dataset and label draw the labels are drawn once - 1,000 random, and 300 random + 700
active, drawn stratified as `custmatch sample` does by default - and the run is repeated with
"fold_trees" 300 (the default), 100 and 50 on the same labels, so only the fold models differ.
Datasets: the five small benchmarks of beat_splink.py, three draws each, and the NCVR folder of
ncvr_memory.py (a million people, its saved labels).

  python research/scripts/fold_trees.py [--seeds 1,2,3] [--ncvr data/ncvr_memory_1m] [DATASET ...]
Writes research/results/fold_trees.json.
"""
import argparse
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
from custmatch.core import partition_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/fold_trees.json"
TREES = (300, 100, 50)


def runs(config, lab, tmp, tag, score):
    """The run at each fold_trees on the same labels: clustered F1, threshold, forest seconds."""
    out = {}
    cfg = json.loads(Path(config).read_text())
    for trees in TREES:
        cp = Path(config).with_name(f"{tag}_{trees}.json")
        cp.write_text(json.dumps(cfg | {"fold_trees": trees}))
        s = pipeline.run(cp, lab, tmp / f"{tag}_{trees}", seed=1, log=lambda *_: None)
        forest = next(v for k, v in s["stage_seconds"].items() if k.startswith("forest"))
        out[str(trees)] = {"f1": score(tmp / f"{tag}_{trees}"), "threshold": s["threshold"],
                           "forest_seconds": forest}
        print(f"  {tag:28s} fold_trees {trees:3d}: F1 {out[str(trees)]['f1']:6.2f}  threshold "
              f"{s['threshold']:.3f}  forest {forest:6.1f} s", flush=True)
    return out


def small(name, seed, tmp):
    config, truth = BS.setup(name, tmp)
    config = BS.with_overrides(config, {}, tmp, f"{name}_s{seed}")

    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(truth[a] == truth[b]) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(path, index=False)
        return lab

    def score(out_dir):
        c = pd.read_csv(out_dir / "clusters.csv", dtype=str)
        t = np.array([truth[f"{s}:{r}"] for s, r in zip(c.source, c.record_id)])
        return B.clustered_f1(c.customer_id.to_numpy(), t)["f1"]
    df, cfg = io.load(config)
    rp = tmp / f"{name}_s{seed}_random.csv"
    labels.sample(config, min(1000, len(labels.candidates(df, cfg))), rp, seed=seed)
    fill(rp)
    ap = tmp / f"{name}_s{seed}_active.csv"
    labels.sample(config, 300, ap, seed=seed)
    al = fill(ap)
    while len(al) < 1000:
        labels.sample_active(config, ap, 100, tmp / "next.csv", seed=seed)
        al = pd.concat([al, fill(tmp / "next.csv")], ignore_index=True)
        al.to_csv(ap, index=False)
    return {"random_1000": runs(config, rp, tmp, f"{name}_s{seed}_random", score),
            "active_300_700": runs(config, ap, tmp, f"{name}_s{seed}_active", score)}


def ncvr(work, tmp):
    def score(out_dir):
        c = pd.read_csv(out_dir / "clusters.csv", dtype=str)
        pid = np.array([k.split(":")[1].rsplit("_", 1)[0] for k in c.source + ":" + c.record_id])
        _, sz = np.unique(pid, return_counts=True)
        return partition_scores(c.customer_id.to_numpy(), pid, float((sz * (sz - 1) / 2).sum()))["f1"]
    cfg = json.loads((work / "config.json").read_text())
    cp = work / "config_fold.json"                    # beside the sources: their relative paths hold
    cp.write_text(json.dumps(cfg))
    return {"active_300_700": runs(cp, work / "labels.csv", tmp, "ncvr", score)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=BS.DATASETS)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--ncvr", help="an ncvr_memory.py folder (config.json, sources, labels.csv)")
    a = ap.parse_args()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in a.datasets:
            for seed in map(int, a.seeds.split(",")):
                out.setdefault(name, {})[f"seed{seed}"] = small(name, seed, tmp)
                OUT.write_text(json.dumps(out, indent=1))
        if a.ncvr:
            out["ncvr_1m"] = ncvr(Path(a.ncvr), tmp)
            OUT.write_text(json.dumps(out, indent=1))
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
