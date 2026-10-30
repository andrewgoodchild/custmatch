"""Peak memory and time of a full custmatch run on NCVR, by blocking engine.

    python research/scripts/ncvr_memory.py --people 1000000 --work data/ncvr_memory_1m [--engine duckdb]
    python research/scripts/ncvr_memory.py --work data/ncvr_full_local/current_default [--engine duckdb]
    python research/scripts/ncvr_memory.py --work data/ncvr_memory_1m --engine splink

--work holds the run's config.json, sources and labels.csv. With --people and no labels there yet,
it is built first: that many people with all their records, the "current_default" config of
ncvr_leipzig.py, 300 random labels drawn stratified, then 700 by active learning (as a user
would label). The full five-million-record folder is the one research/scripts/modal_ncvr_full.py
leaves. Every later call reuses the folder, so runs differ only in the code and engine.

--engine splink runs Splink on DuckDB instead, set up as ncvr_leipzig.py's comparison: blocking
rules mirroring the config's keys (not its adaptive splits), name comparisons with term-frequency
adjustment, m from the same labels. Its time is Splink's training and prediction, not loading the
records or choosing the threshold; it is scored at the threshold that does best on the labels.

Peak memory is the process's lifetime maximum physical footprint on macOS (proc_pid_rusage, what
Activity Monitor shows): resident memory reads low under memory pressure, as macOS compresses
pages, and high after a stage frees memory that the allocator keeps for reuse. Elsewhere it is
the peak resident memory (getrusage). Writes research/results/ncvr_memory.json.
"""
import argparse
import ctypes
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncvr_leipzig as nv  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.core import partition_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/ncvr_memory.json"


def peak_gb():
    """The process's peak memory so far, in GB."""
    if sys.platform == "darwin":
        u64 = ctypes.c_uint64
        names = ("user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
                 "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time",
                 "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins",
                 "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten", "cpu_time_qos_default",
                 "cpu_time_qos_maintenance", "cpu_time_qos_background", "cpu_time_qos_utility", "cpu_time_qos_legacy",
                 "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive", "billed_system_time",
                 "serviced_system_time", "logical_writes", "lifetime_max_phys_footprint")

        class RUsageInfoV4(ctypes.Structure):             # <sys/resource.h>, RUSAGE_INFO_V4 (first fields)
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, u64) for n in names] + [("rest", u64 * 6)]
        r = RUsageInfoV4()
        if ctypes.CDLL("/usr/lib/libSystem.B.dylib").proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)) == 0:
            return r.lifetime_max_phys_footprint / 2**30
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20     # KB on Linux


def build(work, people):
    """Sources, config and 300 stratified random + 700 active labels for `people` NCVR people."""
    person = lambda k: k.split(":")[1].rsplit("_", 1)[0]  # noqa: E731
    sub = nv.sample(nv.load(), people, 1)
    cfg = dict(nv.VARIANTS["current_default"], sources=[])
    for s, g in sub.groupby("source"):
        g = g.reset_index(drop=True).assign(rid=lambda d: d.recid + "_" + d.index.astype(str))
        g.to_parquet(work / f"{s}.parquet", index=False)
        cfg["sources"].append({"name": s, "path": f"{s}.parquet", "columns": {
            "record_id": "rid", "first_name": "givenname", "last_name": "surname",
            "town": "suburb", "postcode": "postcode"}})
    (work / "config.json").write_text(json.dumps(cfg))

    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(person(a) == person(b)) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(path, index=False)
        return lab
    lp = work / "labels.csv"
    labels.sample(work / "config.json", 300, lp, seed=1, stratified=True)
    lab = fill(lp)
    while len(lab) < 1000:
        labels.sample_active(work / "config.json", lp, 100, work / "next.csv", seed=1)
        lab = pd.concat([lab, fill(work / "next.csv")], ignore_index=True)
        lab.to_csv(lp, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="folder with config.json, sources and labels.csv")
    ap.add_argument("--people", type=int, help="build the folder from this many NCVR people if it has no labels")
    ap.add_argument("--engine", choices=["numpy", "duckdb", "splink"], default="numpy")
    ap.add_argument("--splink-memory", help="DuckDB's memory limit for Splink, such as 8GB (default 80%% of RAM)")
    ap.add_argument("--tag", default="", help="kept as KEY/TAG beside the plain runs")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "labels.csv").exists():
        build(work, a.people)
    if a.engine == "splink":
        return splink(work, a.tag, a.splink_memory)
    cfg = json.loads((work / "config.json").read_text()) | {"blocking_engine": a.engine}
    cp = work / f"config_{a.engine}.json"                # beside the sources: their relative paths hold
    cp.write_text(json.dumps(cfg))
    stages, t0 = [], time.time()
    s = pipeline.run(cp, work / "labels.csv", work / f"out_{a.engine}", seed=1,
                     log=lambda m: stages.append({"stage": m.split("] ", 1)[1].removesuffix(" done"),
                                                  "seconds": round(time.time() - t0, 1),
                                                  "peak_gb_so_far": round(peak_gb(), 2)}))
    seconds = round(time.time() - t0, 1)
    c = pd.read_csv(work / f"out_{a.engine}" / "clusters.csv", dtype=str)
    pid = np.array([k.split(":")[1].rsplit("_", 1)[0] for k in c.source + ":" + c.record_id])
    _, sz = np.unique(pid, return_counts=True)
    res = {"records": int(len(c)), "candidate_pairs": s["candidate_pairs"], "engine": a.engine,
           "clustered": partition_scores(c.customer_id.to_numpy(), pid, float((sz * (sz - 1) / 2).sum())),
           "peak_memory_gb": round(peak_gb(), 2), "seconds": seconds, "stage_seconds": s["stage_seconds"],
           "peak_by_stage": stages, "machine": f"{os.cpu_count()} cores, {sys.platform}"}
    save(f"records_{len(c)}/{a.engine}" + (f"/{a.tag}" if a.tag else ""), res)


def save(key, res):
    latest = json.loads(OUT.read_text()) if OUT.exists() else {}
    OUT.write_text(json.dumps(latest | {key: res}, indent=1, default=float))
    print(key, {k: res[k] for k in ("candidate_pairs", "peak_memory_gb", "seconds")}, res["clustered"]["f1"])


def splink(work, tag, memory_limit=None):
    """Splink on DuckDB, m from the folder's labels, scored at the threshold the labels choose."""
    import splink as sp
    df, cfg = io.load(work / "config.json")
    keys = io.key(df)
    pid = np.array([k.split(":")[1].rsplit("_", 1)[0] for k in keys])
    lab = pd.read_csv(work / "labels.csv", dtype=str, keep_default_na=False)
    pos = {k: i for i, k in enumerate(keys)}
    labelled = (np.array([pos[x] for x in lab.a]), np.array([pos[x] for x in lab.b]), lab.label.astype(int).to_numpy())
    loaded, t0 = round(peak_gb(), 2), time.time()
    predicted = []
    import tempfile
    with tempfile.TemporaryDirectory(dir=work) as spill:
        pairs, w = nv.splink_scores(df, cfg, labelled, after_predict=lambda: predicted.append(round(peak_gb(), 2)),
                                    memory_limit=memory_limit, spill=spill)
    seconds = round(time.time() - t0, 1)
    peak = round(peak_gb(), 2)              # with the predictions fetched, before the threshold search
    cuts = nv._cuts(df, pairs, w, pid, *labelled)
    res = {"records": int(len(df)), "candidate_pairs": int(len(pairs)), "engine": f"splink {sp.__version__}",
           "clustered": cuts["threshold_from_labels"], "p_0.99": cuts["p_0.99"],
           "best_in_hindsight": cuts["best_in_hindsight"], "peak_memory_gb": peak,
           "duckdb_memory_limit": memory_limit or "default (80% of RAM)", "peak_after_loading_gb": loaded, "peak_after_predict_gb": predicted[0], "seconds": seconds, "machine": f"{os.cpu_count()} cores, {sys.platform}",
           "note": "candidate_pairs: Splink's predictions above p = 0.0001, not every pair it compared"}
    save(f"records_{len(df)}/splink" + (f"/{tag}" if tag else ""), res)


if __name__ == "__main__":
    main()
