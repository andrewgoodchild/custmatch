"""Head to head at scale: custmatch against Splink 5 on the same records, labels and machine.

The synthetic customer file copied K times (custmatch_scale.py: copy-specific names and contacts,
so copies never match each other), cleaned once by custmatch's loader so both tools see the same
records. 1,000 labelled pairs drawn by custmatch's sampler and answered from the true people.

  custmatch   pipeline.run: training (cross-fitted forest), threshold, clustering, households,
              golden records, outputs
  splink_em   Splink 5 as its docs recommend: blocking mirroring custmatch's keys, name and email
              comparisons with term-frequency adjustment, u by random sampling, m by EM; predict;
              the threshold chosen by clustering on all the labels; Splink's own clustering at it
  splink_m_from_labels  the same with m estimated from half the labels, the threshold from the other half

Each tool runs in its own process (Splink in its own environment, .venv-splink5:
pip install "splink>=5" pandas pyarrow; dedupe in .venv-dedupe: pip install dedupe "BTrees<6",
which dedupe 3.0.3 needs), timed per step, its peak memory sampled. Scored as
clustered pairwise F1 against the true people on all records.

    python research/scripts/head_to_head.py --copies 50          (about a million records)
Writes research/results/head_to_head.json.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLINK_PY = ROOT / ".venv-splink5/bin/python"
DEDUPE_PY = ROOT / ".venv-dedupe/bin/python"
OUT = ROOT / "research/results/head_to_head.json"


class Stopped(Exception):
    """A tool went over the memory or time limit and was stopped: (reason, seconds, peak GB)."""


LIMIT_GB, LIMIT_SECONDS = 10, 3600         # a 16 GB laptop: past 10 GB the machine itself is at risk


def run_measured(cmd):
    """Run a child process; (stdout, wall seconds, peak resident memory of it and its children, GB).
    Stopped (with its children) past LIMIT_GB or LIMIT_SECONDS: raises Stopped."""
    import psutil
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    proc, peak, done, why = psutil.Process(p.pid), [0], threading.Event(), []
    t0 = time.time()

    def sample():
        while not done.is_set():
            try:
                kids = proc.children(recursive=True)
                rss = proc.memory_info().rss + sum(c.memory_info().rss for c in kids)
                peak[0] = max(peak[0], rss)
                over = (f"over {LIMIT_GB} GB" if rss > LIMIT_GB * 2 ** 30 else
                        f"over {LIMIT_SECONDS} s" if time.time() - t0 > LIMIT_SECONDS else None)
                if over:
                    why.append(over)
                    for c in kids + [proc]:
                        c.kill()
                    return
            except psutil.Error:
                pass
            time.sleep(0.1)
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    out, err = p.communicate()
    done.set()
    t.join()
    secs, gb = round(time.time() - t0, 1), round(peak[0] / 2 ** 30, 2)
    if why:
        raise Stopped(why[0], secs, gb)
    if p.returncode:
        raise RuntimeError(err[-3000:])
    return out, secs, gb


# ------------------------------------------------------------------ the inputs, built once
def prepare(copies, work):
    sys.path.insert(0, str(ROOT / "research/scripts"))
    import custmatch_scale as CS
    import custmatch_validate as V
    import pandas as pd
    from custmatch import io, labels, pipeline
    cfg = CS.scaled(copies, work)
    df, _ = io.load(cfg)
    gold = {f"{k.split(':', 1)[0]}:{k.split(':', 1)[1]}-x{c}": f"{s}-x{c}"
            for c in range(copies) for k, (s, _) in V.truth().items()}
    lp = work / "labels.csv"
    labels.sample(cfg, 1000, lp, seed=1, stratified=False)
    lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
    lab["label"] = [int(gold[a] == gold[b]) for a, b in zip(lab.a, lab.b)]
    lab.to_csv(lp, index=False)
    li, lj, ly = pipeline.read_labels(lp, df)
    recs = df[list(io.CANONICAL)].copy()
    recs.insert(0, "unique_id", range(len(df)))
    recs["truth"] = [gold.get(k, k) for k in io.key(df)]
    recs.to_parquet(work / "records.parquet", index=False)
    pd.DataFrame({"i": li, "j": lj, "y": ly.astype(int)}).to_parquet(work / "labels.parquet", index=False)
    return cfg, lp, len(df)


def f1_of(truth, clusters):
    import numpy as np
    from custmatch.core import partition_scores
    _, n = np.unique(truth, return_counts=True)
    return partition_scores(np.asarray(clusters), np.asarray(truth), float(np.sum(n * (n - 1) / 2)))


# ------------------------------------------------------------------ custmatch, in its own process
def custmatch_child(cfg, labels_path, out):
    from custmatch import pipeline
    s = pipeline.run(cfg, labels_path, out, log=lambda *a: None)
    print(json.dumps({"stage_seconds": s["stage_seconds"], "threshold": s["threshold"]}))


# ------------------------------------------------------------------ Splink 5, in its own environment
SPLINK_CHILD = r'''
import json, sys, time
import numpy as np, pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on
work, mode, step = sys.argv[1], sys.argv[2], sys.argv[3]
recs = pd.read_parquet(f"{work}/records.parquet").drop(columns=["truth"])
for c in recs.columns:
    if c != "unique_id":
        recs[c] = recs[c].replace("", None)
lab = pd.read_parquet(f"{work}/labels.parquet")
capped = mode.endswith("_capped")
base_mode = mode.removesuffix("_capped")
if capped:      # a value on more than 60 records does not block (custmatch's cap): placeholders like a@a.com
    for c in ("email", "phone"):
        n = recs[c].map(recs[c].value_counts())
        recs[f"{c}_block"] = recs[c].where(n <= 60)
half = np.random.default_rng(0).permutation(len(lab)) % 2 == 0
settings = SettingsCreator(
    link_type="dedupe_only",
    blocking_rules_to_generate_predictions=[
        block_on("last_name", "substr(first_name, 1, 3)"), block_on("last_name", "addr_street"),
        block_on("addr_number", "addr_street"), block_on("first_name", "addr_street"),
        block_on("email_block" if capped else "email"), block_on("phone_block" if capped else "phone")],
    comparisons=[
        cl.NameComparison("first_name").configure(term_frequency_adjustments=True),
        cl.NameComparison("last_name").configure(term_frequency_adjustments=True),
        cl.ExactMatch("middle_initial"),
        cl.DateOfBirthComparison("date_of_birth", input_is_string=True, datetime_format="%m/%d/%Y",
                                 invalid_dates_as_null=True),
        cl.ExactMatch("addr_number"),
        cl.JaroWinklerAtThresholds("addr_street", [0.9]),
        cl.EmailComparison("email").configure(term_frequency_adjustments=True),
        cl.ExactMatch("phone").configure(term_frequency_adjustments=True),
        cl.ExactMatch("ssn"),
    ],
    retain_intermediate_calculation_columns=False)
t = {}
t0 = time.time()
db_api = DuckDBAPI()
linker = Linker(db_api.register(recs), settings)
linker.training.estimate_probability_two_random_records_match(
    [block_on("first_name", "last_name", "addr_number")], recall=0.8)
linker.training.estimate_u_using_random_sampling(max_pairs=1e7)
if base_mode == "em":
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("addr_number", "addr_street"))
    linker.training.estimate_parameters_using_expectation_maximisation(block_on("first_name", "last_name"))
else:
    l = lab[half]
    pairs = pd.DataFrame({"unique_id_l": np.minimum(l.i, l.j), "unique_id_r": np.maximum(l.i, l.j),
                          "clerical_match_score": l.y.astype(float)})
    linker.training.estimate_m_from_pairwise_labels(db_api.register(pairs))
t["train"] = round(time.time() - t0, 1)
t0 = time.time()
p = linker.inference.predict(threshold_match_probability=0.001)
t["predict"] = round(time.time() - t0, 1)
# for scoring only, outside the timing: DuckDB writes the pairs (converting millions of rows to
# pandas took longer than Splink's prediction itself)
sql = db_api._execute_sql_against_backend
sql(f"COPY (SELECT unique_id_l, unique_id_r, match_weight FROM {p.physical_name}) "
    f"TO '{work}/splink_{mode}_pairs.parquet' (FORMAT PARQUET)")
n_pairs = sql(f"SELECT count(*) FROM {p.physical_name}").fetchone()[0]
if step == "cluster":
    th = float(sys.argv[4])
    t0 = time.time()
    cl_ = linker.clustering.cluster_pairwise_predictions_at_threshold(p, threshold_match_probability=th)
    t["cluster"] = round(time.time() - t0, 1)
    sql(f"COPY (SELECT unique_id, cluster_id FROM {cl_.physical_name}) "
        f"TO '{work}/splink_{mode}_clusters.parquet' (FORMAT PARQUET)")
print(json.dumps({"seconds": t, "pairs": int(n_pairs)}))
'''


def splink_run(work, mode):
    """Train and predict (timed), choose the threshold on labels, then Splink clusters at it (timed)."""
    import numpy as np
    import pandas as pd
    from custmatch import pipeline
    from custmatch.core import components
    child = Path(work) / "splink_child.py"
    child.write_text(SPLINK_CHILD)
    out, wall, peak = run_measured([str(SPLINK_PY), str(child), str(work), mode, "predict"])
    first = json.loads(out.strip().splitlines()[-1])
    recs = pd.read_parquet(Path(work) / "records.parquet", columns=["truth"])
    lab = pd.read_parquet(Path(work) / "labels.parquet")
    p = pd.read_parquet(Path(work) / f"splink_{mode}_pairs.parquet")
    n = len(recs)
    pairs = np.c_[np.minimum(p.unique_id_l, p.unique_id_r), np.maximum(p.unique_id_l, p.unique_id_r)].astype(np.int32)
    w = p.match_weight.to_numpy()
    li, lj = np.minimum(lab.i, lab.j).to_numpy(), np.maximum(lab.i, lab.j).to_numpy()
    tune = np.ones(len(lab), bool) if mode.startswith("em") else ~(np.random.default_rng(0).permutation(len(lab)) % 2 == 0)
    grid = np.arange(-5, 81, 1.0)              # wide: Splink's weights reach far past 40
    th, _ = pipeline.choose_threshold(n, pairs, np.c_[li, lj], lab.y.to_numpy().astype(bool), [(w, tune)], grid)
    prob = float(1 / (1 + 2 ** -th))
    # the whole Splink job again, now ending with its own clustering at the chosen threshold
    out, wall2, peak2 = run_measured([str(SPLINK_PY), str(child), str(work), mode, "cluster", repr(prob)])
    second = json.loads(out.strip().splitlines()[-1])
    cl_ = pd.read_parquet(Path(work) / f"splink_{mode}_clusters.parquet").sort_values("unique_id")
    ours = components(n, pairs, w > th)                 # the same clusters by connected components
    return {"seconds": second["seconds"], "splink_seconds": round(sum(second["seconds"].values()), 1),
            "wall_seconds_with_loading_and_export": wall2, "peak_memory_gb": max(peak, peak2),
            "candidate_pairs_above_0.001": first["pairs"], "threshold_match_weight": float(th),
            "threshold_match_probability": prob,
            "clustered": f1_of(recs.truth.to_numpy(), cl_.cluster_id.to_numpy()),
            "clustered_check_connected_components": f1_of(recs.truth.to_numpy(), ours)["f1"]}


# ------------------------------------------------------------------ dedupe, in its own environment
DEDUPE_CHILD = r'''
import io, json, os, sys, time
import pandas as pd
import dedupe
from dedupe import variables as v


def main():
    work = sys.argv[1]
    recs = pd.read_parquet(f"{work}/records.parquet").drop(columns=["truth"])
    cols = ["first_name", "middle_initial", "last_name", "date_of_birth", "addr_number", "addr_street",
            "addr_unit", "email", "phone", "ssn"]
    data = {int(u): {c: (x or None) for c, x in zip(cols, row)}
            for u, row in zip(recs.unique_id, recs[cols].itertuples(index=False, name=None))}
    del recs
    lab = pd.read_parquet(f"{work}/labels.parquet")
    fields = [v.String("first_name", has_missing=True), v.Exact("middle_initial", has_missing=True),
              v.String("last_name", has_missing=True), v.ShortString("date_of_birth", has_missing=True),
              v.ShortString("addr_number", has_missing=True), v.String("addr_street", has_missing=True),
              v.ShortString("addr_unit", has_missing=True), v.String("email", has_missing=True),
              v.ShortString("phone", has_missing=True), v.ShortString("ssn", has_missing=True)]
    t = {}
    t0 = time.time()
    deduper = dedupe.Dedupe(fields, num_cores=os.cpu_count())
    pairs = [(data[int(i)], data[int(j)]) for i, j in zip(lab.i, lab.j)]
    training = io.StringIO()       # labelled pairs go in as a training file, so they join dedupe's index
    dedupe.serializer.write_training({"match": [p for p, y in zip(pairs, lab.y) if y],
                                      "distinct": [p for p, y in zip(pairs, lab.y) if not y]}, training)
    training.seek(0)
    deduper.prepare_training(data, training_file=training)     # plus its own sample, for learning blocking rules
    deduper.train()
    t["train"] = round(time.time() - t0, 1)
    t0 = time.time()
    clusters = deduper.partition(data, threshold=0.5)      # dedupe's documented default
    t["predict_and_cluster"] = round(time.time() - t0, 1)
    cid = {u: k for k, (ids, _) in enumerate(clusters) for u in ids}
    pd.DataFrame({"unique_id": list(data), "cluster_id": [cid.get(u, -1 - u) for u in data]}).to_parquet(
        f"{work}/dedupe_clusters.parquet", index=False)
    print(json.dumps({"seconds": t}))


# macOS starts worker processes by re-running this script: without the guard each of
# dedupe's workers would rerun the whole job
if __name__ == "__main__":
    main()
'''


def dedupe_run(work):
    """dedupe 3 trained on the same 1,000 labelled pairs (no interactive labelling), partitioned at 0.5."""
    import pandas as pd
    child = Path(work) / "dedupe_child.py"
    child.write_text(DEDUPE_CHILD)
    try:
        out, wall, peak = run_measured([str(DEDUPE_PY), str(child), str(work)])
    except Stopped as e:
        return {"stopped": e.args[0], "seconds_until_stopped": e.args[1], "peak_memory_gb": e.args[2]}
    r = json.loads(out.strip().splitlines()[-1])
    recs = pd.read_parquet(Path(work) / "records.parquet", columns=["truth"])
    cl_ = pd.read_parquet(Path(work) / "dedupe_clusters.parquet").sort_values("unique_id")
    ver = "import importlib.metadata as m; print(m.version('dedupe'))"
    return {"seconds": r["seconds"], "dedupe_seconds": round(sum(r["seconds"].values()), 1),
            "wall_seconds_with_loading_and_export": wall, "peak_memory_gb": peak, "threshold": 0.5,
            "version": subprocess.run([str(DEDUPE_PY), "-c", ver], capture_output=True, text=True).stdout.strip(),
            "clustered": f1_of(recs.truth.to_numpy(), cl_.cluster_id.to_numpy())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=50)
    ap.add_argument("--child", nargs="*")
    ap.add_argument("--work", help="reuse inputs prepared there (same copies and seed)")
    ap.add_argument("--splink-only", action="store_true", help="keep the stored custmatch result; rerun Splink")
    ap.add_argument("--dedupe", action="store_true", help="add dedupe to the stored results (.venv-dedupe: pip install dedupe)")
    a = ap.parse_args()
    if a.child:                                          # (the custmatch child process)
        custmatch_child(*a.child)
        return
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg, lp, n = prepare(a.copies, work)
    res = {"records": n, "copies": a.copies, "prepare_seconds": round(time.time() - t0, 1),
           "machine": "one laptop (Apple silicon), both tools, one at a time"}
    allres = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.splink_only or a.dedupe:
        res = allres[f"copies_{a.copies}"]
    print(f"{n:,} records prepared", flush=True)
    if a.dedupe:
        res["dedupe"] = r = dedupe_run(work)
        print("dedupe", r.get("stopped") or (r["dedupe_seconds"], "s", r["peak_memory_gb"], "GB", r["clustered"]["f1"]),
              flush=True)
        allres[f"copies_{a.copies}"] = res
        OUT.write_text(json.dumps(allres, indent=1, default=float))
        return
    if a.splink_only:
        return splink_all(work, res, allres, a)
    out, wall, peak = run_measured([sys.executable, __file__, "--child", str(cfg), str(lp), str(work / "cm_out")])
    s = json.loads(out.strip().splitlines()[-1])
    import pandas as pd
    cl = pd.read_csv(work / "cm_out/clusters.csv", dtype=str)
    recs = pd.read_parquet(work / "records.parquet", columns=["truth"])
    from custmatch import io
    rec_keys = io.key(io.load(cfg)[0])
    truth = dict(zip(rec_keys, recs.truth))
    res["custmatch"] = {"wall_seconds": wall, "peak_memory_gb": peak, "stage_seconds": s["stage_seconds"],
                        "clustered": f1_of([truth[k] for k in cl.source + ":" + cl.record_id], cl.customer_id)}
    print("custmatch", wall, "s", peak, "GB", res["custmatch"]["clustered"]["f1"], flush=True)
    splink_all(work, res, allres, a)


def splink_all(work, res, allres, a):
    for mode in ("em", "m_from_labels", "em_capped", "m_from_labels_capped"):
        r = splink_run(work, mode)
        res[f"splink_{mode}"] = r
        print(f"splink_{mode}", r["splink_seconds"], "s", r["peak_memory_gb"], "GB", r["clustered"]["f1"],
              "pairs", r["candidate_pairs_above_0.001"], flush=True)
        allres[f"copies_{a.copies}"] = res
        OUT.write_text(json.dumps(allres, indent=1, default=float))
    ver = "import splink, duckdb; print(splink.__version__, duckdb.__version__)"
    res["splink_version"] = subprocess.run([str(SPLINK_PY), "-c", ver], capture_output=True, text=True).stdout.split()
    allres[f"copies_{a.copies}"] = res
    OUT.write_text(json.dumps(allres, indent=1, default=float))


if __name__ == "__main__":
    main()
