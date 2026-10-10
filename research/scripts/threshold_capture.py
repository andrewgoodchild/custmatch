"""Capture what custmatch sees when it chooses its threshold, and the truth, so threshold rules
can be compared offline (research/scripts/threshold_rules.py) without rerunning the pipeline per rule.

For each case it runs custmatch as a user would (labels from the truth: random, or random then
active) and saves to data/threshold_capture/<case>_<seed>.npz:
  pairs, prob      every candidate pair and the final model's match probability
  grid, tp, fp, fn the cross-fitted closure-aware counts on the held-out random labels per threshold
  coherence        the in-cluster precision estimate per threshold
  pid              the true person of every record, in the pipeline's record order
  chosen           the threshold custmatch chose

    python research/scripts/threshold_capture.py CASE [CASE ...] [--seeds 1,2,3]
    cases: ncvr20k ncvr200k au synthetic synthetic_active, and the public benchmarks as
    bench_<febrl3|febrl4|historical_50k|fake_1000>[_active] (held out: not used to choose a rule)

Data stays in the gitignored data/ folder.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from custmatch import labels, pipeline  # noqa: E402
from custmatch.core import components  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/threshold_capture"
CAP = {}


def _install_hooks():
    """Wrap choose_threshold and cluster_coherence to record their inputs."""
    orig_choose, orig_coh = pipeline.choose_threshold, pipeline.cluster_coherence

    def coh(n, pairs, score, t, predict, seed=0, k=2000):
        CAP.setdefault("pairs", pairs)
        CAP.setdefault("prob", score)
        v = orig_coh(n, pairs, score, t, predict, seed, k)
        CAP.setdefault("coherence", {})[float(t)] = float(v)
        return v

    def choose(n, pairs, lab_pairs, ly, fold_scores, grid, coherence=None, min_precision=None):
        th, m = orig_choose(n, pairs, lab_pairs, ly, fold_scores, grid, coherence, min_precision)
        counts = {float(t): np.zeros(3, dtype=int) for t in grid}
        for s, held in fold_scores:
            hi, hj, hy = lab_pairs[held, 0], lab_pairs[held, 1], ly[held]
            for t in grid:
                cc = components(n, pairs, s > t)
                same = cc[hi] == cc[hj]
                counts[float(t)] += [(same & hy).sum(), (same & ~hy).sum(), (~same & hy).sum()]
        CAP.update(counts=counts, chosen=float(th), n=n)
        return th, m
    pipeline.choose_threshold, pipeline.cluster_coherence = choose, coh


def _save(name, out_dir, pid_of):
    c = pd.read_csv(Path(out_dir) / "clusters.csv", dtype=str)
    pid = np.array([pid_of[k] for k in c.source + ":" + c.record_id])
    grid = np.array(sorted(CAP["counts"]))
    cnt = np.array([CAP["counts"][t] for t in grid])
    OUT.mkdir(parents=True, exist_ok=True)
    _, pid_codes = np.unique(pid.astype(str), return_inverse=True)
    np.savez_compressed(OUT / f"{name}.npz", pairs=np.asarray(CAP["pairs"], np.int32), prob=np.asarray(CAP["prob"], np.float32),
                        grid=grid, tp=cnt[:, 0], fp=cnt[:, 1], fn=cnt[:, 2],
                        coherence=np.array([CAP["coherence"].get(float(t), np.nan) for t in grid]),
                        pid=pid_codes, chosen=CAP["chosen"])
    print(name, "chosen", CAP["chosen"], "random matches", int(cnt[0, 0] + cnt[0, 2]), flush=True)
    CAP.clear()


def _label(cp, lp, gold_pid, seed, active):
    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(gold_pid[a] == gold_pid[b]) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(path, index=False)
        return lab
    labels.sample(cp, 300 if active else 1000, lp, seed=seed)
    lab = fill(lp)
    while active and len(lab) < 1000:
        labels.sample_active(cp, lp, 100, lp.parent / "next.csv", seed=seed)
        lab = pd.concat([lab, fill(lp.parent / "next.csv")], ignore_index=True)
        lab.to_csv(lp, index=False)


def case_ncvr(people, seed, tmp):
    import ncvr_leipzig as N
    sub = N.sample(N.load(), people, seed)
    N.run(sub, tmp, seed, N.VARIANTS["postcode_initial_keys"])
    pid_of = {}
    for s, g in sub.groupby("source"):
        g = g.reset_index(drop=True)
        pid_of.update({f"{s}:{r}_{i}": r for i, r in enumerate(g.recid)})
    return tmp / "out", pid_of


def case_config(config, gold_pid, seed, tmp, active, extra=None):
    cfg = json.loads(Path(config).read_text()) | (extra or {})
    for s in cfg["sources"]:
        s["path"] = str((Path(config).parent / s["path"]).resolve())
    cp = tmp / "c.json"
    cp.write_text(json.dumps(cfg))
    lp = tmp / "labels.csv"
    _label(cp, lp, gold_pid, seed, active)
    pipeline.run(cp, lp, tmp / "out", seed=seed, log=lambda *_: None)
    return tmp / "out", gold_pid


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = [int(x) for x in sys.argv[sys.argv.index("--seeds") + 1].split(",")] if "--seeds" in sys.argv else [1, 2, 3]
    args = [a for a in args if not a.replace(",", "").isdigit()]
    _install_hooks()
    for case in args:
        for seed in seeds:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                if case.startswith("ncvr"):
                    out, pid_of = case_ncvr(20000 if case == "ncvr20k" else 200000, seed, tmp)
                elif case == "au":
                    import au_validate as AU
                    gold, _ = AU.truth()
                    out, pid_of = case_config(AU.CONFIG, {k: v[0] for k, v in gold.items()}, seed, tmp, True, {"country": "AU"})
                elif case.startswith("bench_"):
                    import beat_splink as BS
                    name = case.removeprefix("bench_").removesuffix("_active")
                    config, truth = BS.setup(name, tmp)
                    out, pid_of = case_config(config, truth, seed, tmp, case.endswith("_active"), {"rich_features": True})
                else:
                    import custmatch_validate as V
                    gold = V.truth()
                    out, pid_of = case_config(V.CONFIG, {k: v[0] for k, v in gold.items()}, seed, tmp, case == "synthetic_active")
                _save(f"{case}_{seed}", out, pid_of)


if __name__ == "__main__":
    main()
