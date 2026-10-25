"""The full NCVR benchmark - 5,000,000 records of 3,500,840 real people - through custmatch on
Modal, resumable.

The run labels as a user would: 300 random pairs, then 700 chosen by active learning, 100 at a
time, then the full run and its score against the truth. Every step saves its output to a Modal
volume and is skipped when it is already there, so a preempted container (Modal restarts it with
the same input) carries on from the last finished step instead of starting again. Each step's
time and peak memory are kept in progress.json on the volume.

    modal run research/scripts/modal_ncvr_full.py                          # the default config, start or resume
    modal run research/scripts/modal_ncvr_full.py --variants fixed_blocking
    modal run research/scripts/modal_ncvr_full.py --fresh                  # discard saved progress first
    python research/scripts/modal_ncvr_full.py --local adaptive_blocking   # on this machine instead,
                                                                           # progress under data/ncvr_full_local/

Variants (VARIANTS) change the config and the number of random labels; each keeps its own
progress on the volume. Writes research/results/scale_large.json under "ncvr_all" (default) or
"ncvr_all_<variant>".
"""
import json
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/scale_large.json"
WORK = Path("/vol/work")
# config changes and random-label counts. At 3.5 million people 300 random labels held only 11
# matches, too few to judge the threshold, and blocks over max_block were dropped: these try
# 1,000 random labels with oversized blocks split on a further key ("fixed") rather than dropped.
# (max_block 200 was not run: 3.7 times the candidate pairs at 20,000 people projects to about
# 250 million at full scale, beyond the machine's memory.)
VARIANTS = {"default": ({}, 300),
            # the default again after the memory redesign (numpy blocking, chunked features and
            # predictions, a leaner active-learning sampler): same config, its own progress
            "default_memory_redesign": ({}, 300),
            "fixed_blocking": ({"blocking": "fixed"}, 1000),
            # oversized blocks split on three fields at once, not one: blocking alone lifts pair
            # completeness from 93.4% to 96.7% here (research/results/ncvr_blocking_scale.json)
            "adaptive_blocking": ({"blocking": "adaptive"}, 1000),
            # nothing set (None drops the key): blocking switches to adaptive by itself and the
            # random labels are drawn stratified (labels.sample's default; the other variants plainly)
            "current_default": ({"blocking": None}, 300)}

app = modal.App("custmatch-ncvr-full")
vol = modal.Volume.from_name("custmatch-ncvr-full", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "pandas", "scipy", "scikit-learn", "rapidfuzz", "httpx", "certifi", "pyarrow",
                      "phonenumbers", "anyascii", "networkx")
         .env({"PYTHONPATH": "/repo:/repo/research/scripts"}))
for d in ("custmatch", "research/scripts"):
    image = image.add_local_dir(ROOT / d, f"/repo/{d}", ignore=["**/__pycache__"])
image = image.add_local_dir(ROOT / "data/ncvr_leipzig", "/repo/data/ncvr_leipzig")


def _progress():
    p = _work() / "progress.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _step(name, fn):
    """Run fn unless progress.json says it finished; record its time and peak memory; commit."""
    import resource
    import sys
    per_gb = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2     # ru_maxrss: bytes on macOS, KiB on Linux
    prog = _progress()
    if name in prog:
        print(f"skip {name} (done)", flush=True)
        return
    t0 = time.time()
    fn()
    prog = _progress()
    prog[name] = {"seconds": round(time.time() - t0, 1),
                  "peak_memory_gb_so_far": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / per_gb, 2)}
    (_work() / "progress.json").write_text(json.dumps(prog, indent=1))
    _CURRENT["commit"]()
    print(f"done {name}: {prog[name]}", flush=True)


_CURRENT = {"work": WORK, "commit": lambda: vol.commit()}


def _work():
    return _CURRENT["work"]


@app.function(image=image, cpu=16, memory=131072, timeout=16 * 3600, volumes={"/vol": vol})
def full(variant="default", fresh=False):
    work = Path("/vol/work") if variant == "default" else Path(f"/vol/{variant}")
    return _run(variant, fresh, work) | {"machine": "Modal, 16 cores, 128 GiB"}


def _run(variant, fresh, work):
    import shutil
    import numpy as np
    import pandas as pd
    import ncvr_leipzig as N
    from custmatch import labels, pipeline
    from custmatch.core import partition_scores
    overrides, n_random = VARIANTS[variant]
    WORK = work
    _CURRENT["work"] = WORK
    if fresh and WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)
    cp, lp, seed = WORK / "config.json", WORK / "labels.csv", 1
    person = lambda k: k.split(":")[1].rsplit("_", 1)[0]  # noqa: E731

    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(person(a) == person(b)) for a, b in zip(lab.a, lab.b)]
        return lab

    def prepare():                      # the records and config, as ncvr_leipzig.run builds them
        allrec = N.load()
        cfg = dict(N.VARIANTS["postcode_initial_keys"], sources=[], **overrides)
        cfg = {k: v for k, v in cfg.items() if v is not None}
        for s, g in allrec.groupby("source"):
            g = g.reset_index(drop=True).assign(rid=lambda d: d.recid + "_" + d.index.astype(str))
            g.to_parquet(WORK / f"{s}.parquet", index=False)
            cfg["sources"].append({"name": s, "path": f"{s}.parquet", "columns": {
                "record_id": "rid", "first_name": "givenname", "last_name": "surname",
                "town": "suburb", "postcode": "postcode"}})
        cp.write_text(json.dumps(cfg))

    def random_labels():
        labels.sample(cp, n_random, lp, seed=seed, stratified=variant == "current_default")
        fill(lp).to_csv(lp, index=False)

    def active_round():
        labels.sample_active(cp, lp, 100, WORK / "next.csv", seed=seed)
        lab = pd.concat([pd.read_csv(lp, dtype=str, keep_default_na=False), fill(WORK / "next.csv")],
                        ignore_index=True)
        lab.to_csv(lp, index=False)

    def final_run():
        s = pipeline.run(cp, lp, WORK / "out", seed=seed, log=print)
        c = pd.read_csv(WORK / "out" / "clusters.csv", dtype=str)
        pid = np.array([person(k) for k in c.source + ":" + c.record_id])
        res = {"records": int(len(c)), "people": int(len(set(pid))), "true_pairs": int(N.n_true(pid)),
               "clustered": partition_scores(c.customer_id.to_numpy(), pid, N.n_true(pid)),
               "threshold": s["threshold"], "candidate_pairs": s["candidate_pairs"],
               "stage_seconds": s["stage_seconds"], "warnings": s["warnings"],
               "random_matches": int(pd.read_csv(lp).query("by == 'random'").label.sum()),
               "config_changes": overrides}
        (WORK / "result.json").write_text(json.dumps(res, indent=1, default=float))

    _step("prepare", prepare)
    _step(f"random_{n_random}", random_labels)
    for k in range(7):
        _step(f"active_{k + 1}", active_round)
    _step("final_run", final_run)
    return {"result": json.loads((WORK / "result.json").read_text()), "progress": _progress()}


def _save(name, out):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["ncvr_all" if name == "default" else f"ncvr_all_{name}"] = out
    OUT.write_text(json.dumps(res, indent=1, default=float))
    r = out["result"]
    print(f"{name}: records {r['records']:,}  clustered F1 {r['clustered']['f1']}  "
          f"total {sum(v['seconds'] for v in out['progress'].values()):,.0f} s", flush=True)


@app.local_entrypoint()
def main(variants: str = "default", fresh: bool = False):
    names = variants.split(",")
    for name, out in zip(names, full.starmap([(v, fresh) for v in names])):
        _save(name, out)


if __name__ == "__main__":
    import argparse
    import os
    import platform
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, choices=sorted(VARIANTS), help="the variant to run here")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    sys.path[:0] = [str(ROOT), str(ROOT / "research/scripts")]
    _CURRENT["commit"] = lambda: None
    out = _run(a.local, a.fresh, ROOT / "data/ncvr_full_local" / a.local)
    gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    _save(a.local, out | {"machine": f"{platform.machine()} laptop, {os.cpu_count()} cores, {gb:.0f} GiB"})
