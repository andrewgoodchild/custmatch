"""Incremental matching vs a full rerun, on the synthetic customer file.

10% of records (random, from both sources) are held out as "new". `run` on the other 90% with
1,000 hand labels, then `add` the new 10% on top; compare with one batch `run` over all
records using the same labels. Writes research/results/custmatch_incremental.json.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custmatch_validate as V  # noqa: E402
from custmatch import incremental, labels, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def split_config(tmp, frac_new=0.10, seed=0):
    cfg = json.load(open(V.CONFIG))
    old_cfg, new_cfg = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
    rng = np.random.default_rng(seed)
    for s_old, s_new in zip(old_cfg["sources"], new_cfg["sources"]):
        d = pd.read_parquet((V.CONFIG.parent / s_old["path"]).resolve())
        is_new = rng.random(len(d)) < frac_new
        d[~is_new].to_parquet(tmp / f"old_{s_old['name']}.parquet", index=False)
        d[is_new].to_parquet(tmp / f"new_{s_old['name']}.parquet", index=False)
        s_old["path"], s_new["path"] = str(tmp / f"old_{s_old['name']}.parquet"), str(tmp / f"new_{s_old['name']}.parquet")
    (tmp / "old.json").write_text(json.dumps(old_cfg))
    (tmp / "new.json").write_text(json.dumps(new_cfg))
    return tmp / "old.json", tmp / "new.json"


def main():
    gold = V.truth()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        old_cfg, new_cfg = split_config(tmp)
        lp = tmp / "labels.csv"
        labels.sample(old_cfg, 1000, lp, seed=1)
        lab = V.fill(lp, gold)
        t0 = time.time()
        pipeline.run(old_cfg, lp, tmp / "batch_90", log=lambda *_: None)
        t_old = time.time() - t0
        t0 = time.time()
        s = incremental.add(tmp / "batch_90", new_cfg, tmp / "incremental")
        t_add = time.time() - t0
        out["incremental"] = V.score_clusters(tmp / "incremental", gold, lab) | {
            "add_seconds": round(t_add, 2), "batch_90_seconds": round(t_old, 2), "summary": s}
        t0 = time.time()
        pipeline.run(V.CONFIG, lp, tmp / "batch_all", log=lambda *_: None)
        out["full_rerun"] = V.score_clusters(tmp / "batch_all", gold, lab) | {
            "seconds": round(time.time() - t0, 2)}
    for k, r in out.items():
        print(f"{k:12s} clustered F1 {r['all_records']['f1']:.2f}, households {r['households']['f1']:.2f}", flush=True)
    print(f"add: {out['incremental']['summary']}", flush=True)
    print(f"time: add {out['incremental']['add_seconds']}s vs full rerun {out['full_rerun']['seconds']}s")
    (ROOT / "research/results/custmatch_incremental.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
