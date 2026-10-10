"""Which pairwise classifier? The random forest against newer options, everything else fixed.

Models (custmatch/models.py): forest (default), hgb (histogram gradient boosting, the LightGBM
algorithm), catboost, auto (forest, catboost and tabpfn each cross-fitted; the best estimate
kept - does the tool's own estimate pick the right one?), and tabpfn (TabPFN v2, a tabular foundation model; one ensemble member,
on Apple's MPS, to keep historical_50k to about an hour).

Datasets: the five benchmark sets (research/scripts/beat_splink.py setup, rich features on) and the
sparse Australian file with "country": "AU". For each dataset and seed, one file of 1,000
random labels and one of 300 random + 700 active (chosen by the forest) is built ONCE and
every model trains on exactly those labels, so differences are the model's.

    python research/scripts/matcher_bakeoff.py [--models forest,hgb] [--datasets febrl3,au] [--seeds 1,2,3]

Writes research/results/matcher_bakeoff.json, resuming: runs already there are skipped.
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
import au_validate as AU  # noqa: E402
import beat_splink as BS  # noqa: E402
import splink_datasets_benchmark as B  # noqa: E402
from custmatch import labels, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/matcher_bakeoff.json"
DATASETS = ["febrl3", "febrl4", "fake_1000", "synthetic", "au", "historical_50k"]
MODELS = ["forest"]
# hgb, catboost, tabpfn and auto were removed from custmatch in 0.1.0 after research/scripts/modal_models.py;
# their results here (and in docs/guide.md) stay as recorded


def setup(name, tmp):
    if name == "au":
        gold, _ = AU.truth()
        cfg = json.loads(AU.CONFIG.read_text()) | {"country": "AU"}
        for s in cfg["sources"]:
            s["path"] = str((AU.CONFIG.parent / s["path"]).resolve())
        (tmp / "au.json").write_text(json.dumps(cfg))
        return tmp / "au.json", {k: str(v[0]) for k, v in gold.items()}
    return BS.setup(name, tmp)


def label_files(config, truth, tmp, tag, seed):
    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(truth[a] == truth[b]) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(path, index=False)
        return lab
    rp = tmp / f"{tag}_random.csv"
    labels.sample(config, 1000, rp, seed=seed)
    fill(rp)
    ap = tmp / f"{tag}_active.csv"
    labels.sample(config, 300, ap, seed=seed)
    al = fill(ap)
    while len(al) < 1000:
        labels.sample_active(config, ap, 100, tmp / "next.csv", seed=seed)
        al = pd.concat([al, fill(tmp / "next.csv")], ignore_index=True)
        al.to_csv(ap, index=False)
    return {"random_1000": rp, "active_300_700": ap}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="1,2,3")
    # a config variant stored under its own name, e.g. --tag forest_cc --override '{"clustering": "correlation"}'
    ap.add_argument("--tag", default="")
    ap.add_argument("--override", default="{}")
    a = ap.parse_args()
    models, datasets = a.models.split(","), a.datasets.split(",")
    override = json.loads(a.override)
    name_of = {m: (a.tag or m) if len(models) == 1 else f"{m}_{a.tag}" if a.tag else m for m in models}
    seeds = [int(s) for s in a.seeds.split(",")]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for ds in datasets:
            config, truth = setup(ds, tmp)
            for seed in seeds:
                todo = [m for m in models for lt in ("random_1000", "active_300_700")
                        if f"{lt}/seed{seed}" not in res.get(ds, {}).get(name_of[m], {})]
                if not todo:
                    continue
                files = label_files(config, truth, tmp, f"{ds}_{seed}", seed)
                for m in models:
                    cfg = json.loads(Path(config).read_text())
                    for s in cfg["sources"]:
                        s["path"] = str((Path(config).parent / s["path"]).resolve())
                    cfg |= {"matcher_model": m} | override
                    cp = tmp / f"{ds}_{m}.json"
                    cp.write_text(json.dumps(cfg))
                    for lt, lp in files.items():
                        key = f"{lt}/seed{seed}"
                        if key in res.get(ds, {}).get(name_of[m], {}):
                            continue
                        t0 = time.time()
                        out = tmp / f"{ds}_{m}_{lt}_{seed}"
                        pipeline.run(cp, lp, out, log=lambda *_: None)
                        c = pd.read_csv(out / "clusters.csv", dtype=str)
                        t = np.array([truth[f"{s}:{r}"] for s, r in zip(c.source, c.record_id)])
                        f1 = B.clustered_f1(c.customer_id.to_numpy(), t)["f1"]
                        res.setdefault(ds, {}).setdefault(name_of[m], {})[key] = {
                            "f1": f1, "seconds": round(time.time() - t0, 1)}
                        print(f"{ds:15s} {name_of[m]:9s} {lt:15s} seed {seed}  clustered F1 {f1:6.2f}  "
                              f"{time.time() - t0:6.1f}s", flush=True)
                        OUT.write_text(json.dumps(res, indent=1))
    summary(res)


def summary(res):
    for ds, by_model in res.items():
        for m, runs in by_model.items():
            for lt in ("random_1000", "active_300_700"):
                f = [v["f1"] for k, v in runs.items() if k.startswith(lt)]
                if f:
                    print(f"{ds:15s} {m:9s} {lt:15s} mean {np.mean(f):6.2f} +- {np.std(f):4.2f} (n={len(f)})")


if __name__ == "__main__":
    main()
