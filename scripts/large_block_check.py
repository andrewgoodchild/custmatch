"""The large-block check (docs/guide.md "How it decides"): blocks skipped as hubs that the model
judges to be one entity are compared after all. On by default; this compares it with
"check_large_blocks": false on the six person benchmarks and dedupe's patent applicants, on the
same labels (1,000 random, seed 1), and records which blocks each run kept.

    python scripts/large_block_check.py            (the patent file needs data/dedupe/, as
                                                    home_data_benchmark.py describes)
Writes results/large_block_check.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import home_data_benchmark as HD  # noqa: E402
import matcher_bakeoff as MB  # noqa: E402
import splink_datasets_benchmark as B  # noqa: E402

from custmatch import labels, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/large_block_check.json"
DATASETS = ["febrl3", "febrl4", "fake_1000", "synthetic", "au", "historical_50k", "patents"]


def setup(ds, tmp):
    if ds != "patents":
        return MB.setup(ds, tmp)
    spec = HD.dedupe_datasets()["dedupe_patent_example"]
    frame = spec["frames"]["patents"]
    frame.fillna("").astype(str).to_csv(tmp / "patents.csv", index=False)
    cfg = {"sources": [{"name": "patents", "path": "patents.csv", "columns": spec["columns"]}],
           "extra_fields": spec["extra_fields"], "extra_blocking": spec["extra_blocking"]}
    (tmp / "patents.json").write_text(json.dumps(cfg))
    return tmp / "patents.json", dict(zip("patents:" + frame.person_id.astype(str), frame.leuven_id.astype(str)))


def main():
    res = {}
    for ds in sys.argv[1:] or DATASETS:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config, truth = setup(ds, tmp)
            lp = tmp / "labels.csv"
            labels.sample(config, 1000, lp, seed=1)
            lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
            lab["label"] = [int(truth[a] == truth[b]) for a, b in zip(lab.a, lab.b)]
            lab.to_csv(lp, index=False)
            cfg = json.loads(Path(config).read_text())
            for s in cfg["sources"]:
                s["path"] = str((Path(config).parent / s["path"]).resolve())
            res[ds] = {}
            for tag, extra in (("off", {"check_large_blocks": False}), ("on", {})):
                cp = tmp / f"{tag}.json"
                cp.write_text(json.dumps(cfg | extra))
                pipeline.run(cp, lp, tmp / tag, log=lambda *_: None)
                c = pd.read_csv(tmp / tag / "clusters.csv", dtype=str)
                f1 = B.clustered_f1(c.customer_id.to_numpy(),
                                    np.array([truth[f"{a}:{b}"] for a, b in zip(c.source, c.record_id)]))
                rep = json.loads((tmp / tag / "blocking_report.json").read_text())
                checked = rep.get("large_blocks_checked", [])
                res[ds][tag] = f1 | {"blocks_checked": len(checked), "blocks_kept": sum(r["kept"] for r in checked),
                                     "pairs_added": rep.get("large_block_pairs_added", 0)}
            print(ds, {t: (r["f1"], r["blocks_kept"], r["blocks_checked"]) for t, r in res[ds].items()}, flush=True)
            OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
