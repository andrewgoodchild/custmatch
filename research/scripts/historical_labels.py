"""historical_50k: does the matcher improve with more labels? A learning curve.

At 1,000 labels every blocking or feature change tried (research/scripts/historical_blocking.py) lands
between 86 and 88 clustered F1, under a ceiling of 93 to 97. This runs custmatch with 1,000,
3,000 and 6,000 random labels (seeds 1 and 2), default keys and similarity blocking, and records
the chosen threshold's F1 and the best threshold's in hindsight, so the model and the threshold
choice can be told apart.

    python research/scripts/historical_labels.py
Writes research/results/historical_labels.json.
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
from custmatch.core import components, partition_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/historical_labels.json"
VARIANTS = {"default": {"lsh_blocking": None},     # the exact keys alone
            "lsh_names_birth_place": {"lsh_blocking": {"fields": ["first_name", "last_name"], "with": "birth_place"}}}
SIZES = (1000, 3000, 6000)
GRID = np.round(np.arange(0.3, 0.99, 0.05), 2)
CAP = {}


def _hook():
    orig = pipeline.cluster_coherence

    def coh(n, pairs, score, t, predict, seed=0, k=2000):
        CAP["pairs"], CAP["prob"] = pairs, score   # the final model's scores on every candidate
        return orig(n, pairs, score, t, predict, seed, k)
    pipeline.cluster_coherence = coh


def main():
    _hook()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        config, truth = BS.setup("historical_50k", tmp)
        for name, over in VARIANTS.items():
            cfg = json.loads(Path(config).read_text())
            for s in cfg["sources"]:
                s["path"] = str((Path(config).parent / s["path"]).resolve())
            cp = tmp / f"{name}.json"
            cp.write_text(json.dumps(cfg | {"rich_features": True} | over))
            df, _ = io.load(cp)
            pid = np.array([truth[k] for k in io.key(df)])
            _, cnt = np.unique(pid, return_counts=True)
            tot = float(np.sum(cnt * (cnt - 1) / 2))
            for n in SIZES:
                for seed in (1, 2):
                    tag = f"{name}_{n}_{seed}"
                    if tag in res:
                        continue
                    lp = tmp / f"{tag}.csv"
                    labels.sample(cp, n, lp, seed=seed, stratified=False)
                    lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
                    lab["label"] = [int(truth[a] == truth[b]) for a, b in zip(lab.a, lab.b)]
                    lab.to_csv(lp, index=False)
                    t0 = time.time()
                    s = pipeline.run(cp, lp, tmp / f"{tag}_out", seed=seed, log=lambda *_: None)
                    c = pd.read_csv(tmp / f"{tag}_out" / "clusters.csv", dtype=str)
                    t = np.array([truth[f"{a}:{b}"] for a, b in zip(c.source, c.record_id)])
                    by_t = {float(g): partition_scores(components(len(df), CAP["pairs"], CAP["prob"] > g), pid, tot)
                            for g in GRID}
                    best = max(by_t, key=lambda g: by_t[g]["f1"])
                    res[tag] = {"labels": n, "random_matches": int(lab.label.sum()),
                                "f1": B.clustered_f1(c.customer_id.to_numpy(), t)["f1"], "threshold": s["threshold"],
                                "best_threshold": best, "best_in_hindsight": by_t[best],
                                "seconds": round(time.time() - t0, 1)}
                    print(tag, res[tag], flush=True)
                    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
