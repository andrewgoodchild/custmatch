"""Each tool on every tool's home data: custmatch, Splink and dedupe on Splink's and dedupe's own datasets.

  Splink's datasets   fake_1000, febrl3, febrl4, historical_50k (splink_datasets_benchmark.py runs
                      custmatch and Splink on them with Splink's documented settings; this adds dedupe)
  dedupe's datasets   csv_example (3,337 Chicago early-childhood sites, 1,162 true sites) and
                      patent_example (2,379 patent applicants, 102 true organisations), from
                      github.com/dedupeio/dedupe-examples, in data/dedupe/ (gitignored)

dedupe's datasets hold organisations, not people: custmatch, a tool for people, reads the
organisation's name as a surname (out of its domain), and Splink has no documented settings for
them, so it gets plain string comparisons on the same fields. dedupe runs with its examples'
settings verbatim on its own data, and with fields mirroring Splink's comparisons on Splink's.

All three learn from the same labelled pairs (custmatch's sampler, seed 1, answered from the
truth; at most 1,000), dedupe without its interactive console. Each is scored as clustered pairwise
F1 against the truth on all records. dedupe clusters at its documented threshold, 0.5; Splink at
the match weight the labels choose.

    python scripts/home_data_benchmark.py [datasets] [--only=custmatch,splink,dedupe] [--dedupe-run=N]
dedupe runs in .venv-dedupe: pip install dedupe "BTrees<6" unidecode (dedupe 3.0.3 needs BTrees<6).
Writes results/home_data_benchmark.json.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import io, labels, pipeline
from custmatch.core import components

sys.path.insert(0, str(Path(__file__).resolve().parent))
import head_to_head as H  # noqa: E402
import splink_datasets_benchmark as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEDUPE_PY = ROOT / ".venv-dedupe/bin/python"
DATA = ROOT / "data/dedupe"
OUT = ROOT / "results/home_data_benchmark.json"


# ------------------------------------------------------------------ dedupe's datasets, as custmatch specs
def dedupe_datasets():
    sites = pd.read_csv(DATA / "csv_example_input_with_true_ids.csv", dtype=str, keep_default_na=False)
    sites = sites.rename(columns={"True Id": "true_id", "Site name": "site_name", "Address": "address",
                                  "Zip": "zip", "Phone": "phone"})
    for c in ("site_name", "address"):
        sites[c] = sites[c].str.replace("\n", " ")
    pat = pd.read_csv(DATA / "patstat_input.csv", dtype=str, keep_default_na=False)
    ref = pd.read_csv(DATA / "patstat_reference.csv", dtype=str, keep_default_na=False)
    pat["leuven_id"] = pat.person_id.map(dict(zip(ref.person_id, ref.leuven_id)))
    pat["latlng"] = np.where((pat.Lat == "0.0") & (pat.Lng == "0.0"), "", pat.Lat + "," + pat.Lng)
    pat["coauthor"] = pat.Coauthor.str.replace("**", " ", regex=False)
    pat["klass"] = pat.Class.str.replace("**", " ", regex=False)
    sites["no_first_name"] = pat["no_first_name"] = ""       # custmatch requires the column; organisations have none
    return {
        "dedupe_csv_example": {
            "frames": {"sites": sites}, "truth_col": "true_id", "id_col": "Id",
            "columns": {"record_id": "Id", "first_name": "no_first_name", "last_name": "site_name",
                        "addr_street": "address", "phone": "phone",
                        "zip": "zip"},
            "extra_fields": [{"name": "zip", "compare": "exact"}],
            "extra_blocking": [["zip"], ["last_name"]]},
        "dedupe_patent_example": {
            "frames": {"patents": pat}, "truth_col": "leuven_id", "id_col": "person_id",
            "columns": {"record_id": "person_id", "first_name": "no_first_name", "last_name": "Name", "latlng": "latlng",
                        "coauthor": "coauthor", "klass": "klass"},
            "extra_fields": [{"name": "latlng", "compare": "exact"}, {"name": "coauthor", "compare": "similarity"},
                             {"name": "klass", "compare": "similarity"}],
            "extra_blocking": [["last_name"], ["latlng"]],
            # what custmatch's warning says to do here: 1,019 records shared only blocks over max_block
            "custmatch_variants": {"max_block_300": {"max_block": 300}, "fixed_blocking": {"blocking": "fixed"}}},
    }


# ------------------------------------------------------------------ dedupe, in its own environment
DEDUPE_CHILD = r'''
import io, json, os, re, sys, time
import pandas as pd
import dedupe
from dedupe import variables as v
from unidecode import unidecode


def main():
    work, name = sys.argv[1], sys.argv[2]
    frames = {s: pd.read_parquet(f"{work}/{name}_{s}.parquet") for s in json.loads(sys.argv[3])}
    lab = pd.read_parquet(f"{work}/{name}_labels.parquet")


    def pre(x):              # csv_example's preProcess, verbatim in effect
        x = unidecode(x or "")
        x = re.sub("  +", " ", re.sub("\n", " ", x)).strip().strip('"').strip("'").lower().strip()
        return x or None


    def records(frame, fields):
        return {k: {f: pre(row[f]) for f in fields} for k, row in zip(frame.key, frame.to_dict("records"))}


    STR = lambda f: v.String(f, has_missing=True)          # noqa: E731
    SHORT = lambda f: v.ShortString(f, has_missing=True)   # noqa: E731
    EXACT = lambda f: v.Exact(f, has_missing=True)         # noqa: E731
    # fields mirroring Splink's documented comparisons for its datasets
    MIRROR = {
        "fake_1000": [STR("first_name"), STR("surname"), SHORT("dob"), EXACT("city"), STR("email")],
        "febrl3": [STR("given_name"), STR("surname"), SHORT("date_of_birth"), SHORT("soc_sec_id"),
                   SHORT("street_number"), EXACT("postcode"), STR("address_1")],
        "historical_50k": [STR("first_name"), STR("surname"), SHORT("dob"), STR("birth_place"),
                           SHORT("postcode_fake"), STR("occupation"), EXACT("gender")],
    }
    MIRROR["febrl4"] = MIRROR["febrl3"]
    cols = lambda fs: sorted({f.field for f in fs})        # noqa: E731

    if name == "dedupe_csv_example":                       # csv_example.py, its fields verbatim
        fields = [v.String("site_name"), v.String("address"), v.Exact("zip", has_missing=True),
                  v.String("phone", has_missing=True)]
        data = [records(f, cols(fields)) for f in frames.values()]
    elif name == "dedupe_patent_example":                  # patent_example.py, its fields verbatim
        f = next(iter(frames.values()))
        d = {}
        for k, row in zip(f.key, f.to_dict("records")):
            row = {c: str(x).lower() for c, x in row.items()}
            d[k] = {"Name": row["Name"] or None,
                    "LatLong": None if row["Lat"] == row["Lng"] == "0.0" else (float(row["Lat"]), float(row["Lng"])),
                    "Class": tuple(sorted(row["Class"].split("**"))) if row["Class"] else None,
                    "Coauthor": tuple(sorted(a for a in row["Coauthor"].split("**") if a != "none"))}
        fields = [v.String("Name", name="name_string", has_missing=True), v.LatLong("LatLong", has_missing=True),
                  v.Set("Class", corpus=[r["Class"] for r in d.values()], has_missing=True),
                  v.Set("Coauthor", corpus=[r["Coauthor"] for r in d.values()], has_missing=True),
                  v.Text("Name", name="name_text", corpus=[r["Name"] for r in d.values()], has_missing=True),
                  v.Interaction("name_string", "name_text")]
        data = [d]
    else:
        fields = MIRROR[name]
        data = [records(f, cols(fields)) for f in frames.values()]

    link = len(data) == 2
    t = {}
    t0 = time.time()
    model = (dedupe.RecordLink if link else dedupe.Dedupe)(fields, num_cores=os.cpu_count())
    if link:
        side = {k: 0 if k in data[0] else 1 for k in list(data[0]) + list(data[1])}
        rec = lambda a, b: (data[0][a], data[1][b]) if side[a] == 0 else (data[0][b], data[1][a])   # noqa: E731
    else:
        rec = lambda a, b: (data[0][a], data[0][b])        # noqa: E731
    pairs = [(rec(a, b), y) for a, b, y in zip(lab.a, lab.b, lab.label)
             if not link or (a in data[0]) != (b in data[0])]
    training = io.StringIO()       # labelled pairs go in as a training file, so they join dedupe's index
    dedupe.serializer.write_training({"match": [p for p, y in pairs if y], "distinct": [p for p, y in pairs if not y]},
                                     training)
    training.seek(0)
    model.prepare_training(*data, training_file=training)
    model.train()
    t["train"] = round(time.time() - t0, 1)
    t0 = time.time()
    if link:                                               # febrl4 is one-to-one, as RecordLink's join assumes
        groups = [ids for ids, _ in model.join(data[0], data[1], threshold=0.5, constraint="one-to-one")]
    else:
        groups = [ids for ids, _ in model.partition(data[0], threshold=0.5)]
    t["predict_and_cluster"] = round(time.time() - t0, 1)
    cid = {k: i for i, ids in enumerate(groups) for k in ids}
    keys = [k for d in data for k in d]
    pd.DataFrame({"key": keys, "cluster": [str(cid[k]) if k in cid else "s:" + k for k in keys]}).to_parquet(
        f"{work}/{name}_dedupe.parquet", index=False)
    print(json.dumps({"seconds": t, "labels_used": len(pairs)}))


# macOS starts worker processes by re-running this script: without the guard each of
# dedupe's workers would rerun the whole job
if __name__ == "__main__":
    main()
'''


def run_dedupe(name, spec, tmp, lab, keys):
    """dedupe on the dataset; its clusters, keyed like custmatch's records ("source:id")."""
    for src, frame in spec["frames"].items():
        f = frame.copy()
        f.insert(0, "key", src + ":" + f[spec["id_col"]].astype(str))
        f.fillna("").astype(str).to_parquet(tmp / f"{name}_{src}.parquet", index=False)   # pandas 3 keeps NaN in astype(str)
    lab[["a", "b", "label"]].astype({"label": int}).to_parquet(tmp / f"{name}_labels.parquet", index=False)
    child = tmp / "dedupe_child.py"
    child.write_text(DEDUPE_CHILD)
    out, wall, peak = H.run_measured([str(DEDUPE_PY), str(child), str(tmp), name, json.dumps(list(spec["frames"]))])
    r = json.loads(out.strip().splitlines()[-1]) | {"peak_memory_gb": peak}
    c = pd.read_parquet(tmp / f"{name}_dedupe.parquet")
    cl = dict(zip(c.key, c.cluster))
    return np.array([cl[k] for k in keys]), r


# ------------------------------------------------------------------ Splink on dedupe's datasets
def splink_plain(name, frame):
    """No documented settings exist: string comparisons on dedupe's fields, u by sampling, m by EM."""
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on
    d = frame.copy()
    d["unique_id"] = d.index.astype(int)
    if name == "dedupe_csv_example":
        d = d[["unique_id", "site_name", "address", "zip", "phone"]].replace("", None)
        comps = [cl.JaroWinklerAtThresholds("site_name", [0.9, 0.8]).configure(term_frequency_adjustments=True),
                 cl.JaroWinklerAtThresholds("address", [0.9, 0.8]), cl.ExactMatch("zip"), cl.ExactMatch("phone")]
        rules = [block_on("zip"), block_on("phone"), block_on("substr(site_name, 1, 5)"),
                 block_on("substr(address, 1, 6)")]
        em = [block_on("zip"), block_on("phone")]
    else:
        d = d.assign(name=d.Name.str.lower(), lat=d.Lat.astype(float), lng=d.Lng.astype(float),
                     klass=d.Class.str.split(r"\*\*", regex=True),
                     coauthor=d.Coauthor.str.lower().str.split(r"\*\*", regex=True))
        d.loc[(d.lat == 0) & (d.lng == 0), ["lat", "lng"]] = None
        d = d[["unique_id", "name", "lat", "lng", "klass", "coauthor"]].replace("", None)
        comps = [cl.JaroWinklerAtThresholds("name", [0.9, 0.8]).configure(term_frequency_adjustments=True),
                 cl.DistanceInKMAtThresholds("lat", "lng", [1, 10]),
                 cl.ArrayIntersectAtSizes("klass", [2, 1]), cl.ArrayIntersectAtSizes("coauthor", [1])]
        rules = [block_on("substr(name, 1, 4)"), block_on("lat", "lng")]
        em = [block_on("substr(name, 1, 4)"), block_on("lat", "lng")]
    linker = Linker(d, SettingsCreator(link_type="dedupe_only", blocking_rules_to_generate_predictions=rules,
                                       comparisons=comps), db_api=DuckDBAPI())
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6)
    for r in em:
        linker.training.estimate_parameters_using_expectation_maximisation(r)
    p = linker.inference.predict(threshold_match_probability=0.001).as_pandas_dataframe()
    return p.unique_id_l.to_numpy(), p.unique_id_r.to_numpy(), p.match_weight.to_numpy()


# ------------------------------------------------------------------ one dataset
def run(name, spec, tmp, tools, out, dedupe_key="dedupe"):
    """tools: which of custmatch, splink, dedupe to run (Splink only on dedupe's datasets: on its
    own, splink_datasets_benchmark.py runs its documented settings). out: the stored results, kept
    for the tools not run."""
    cfg = {"sources": [], "extra_fields": spec["extra_fields"], "extra_blocking": spec["extra_blocking"]}
    for src, frame in spec["frames"].items():
        path = tmp / f"{name}_{src}.csv"
        frame.fillna("").astype(str).to_csv(path, index=False)
        cfg["sources"].append({"name": src, "path": path.name, "columns": spec["columns"]})
    config = tmp / f"{name}.json"
    config.write_text(json.dumps(cfg))
    df, _ = io.load(config)
    truth_of = {}
    for src, frame in spec["frames"].items():
        truth_of.update(dict(zip(src + ":" + frame[spec["id_col"]].astype(str), frame[spec["truth_col"]].astype(str))))
    keys = io.key(df)
    truth = np.array([truth_of[k] for k in keys])
    lp = tmp / f"{name}_labels.csv"
    labels.sample(config, min(1000, len(labels.candidates(df, cfg))), lp, seed=1)
    lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
    lab["label"] = [int(truth_of[a] == truth_of[b]) for a, b in zip(lab.a, lab.b)]
    lab.to_csv(lp, index=False)
    out["labels"] = int(len(lab))
    out["records"], out["true_entities"] = int(len(df)), int(len(set(truth)))

    if "custmatch" in tools:                             # current defaults
        t0 = time.time()
        pipeline.run(config, lp, tmp / f"{name}_out", log=lambda *_: None)
        c = pd.read_csv(tmp / f"{name}_out/clusters.csv", dtype=str)
        cm = dict(zip(c.source + ":" + c.record_id, c.customer_id))
        out["custmatch_1000"] = S.clustered_f1(np.array([cm[k] for k in keys]), truth) | {
            "seconds": round(time.time() - t0, 1)}
        for tag, extra in spec.get("custmatch_variants", {}).items():     # labels drawn under that config
            vcfg = tmp / f"{name}_{tag}.json"
            vcfg.write_text(json.dumps(cfg | extra))
            vl = tmp / f"{name}_{tag}_labels.csv"
            labels.sample(vcfg, min(1000, len(labels.candidates(df, cfg | extra))), vl, seed=1)
            v = pd.read_csv(vl, dtype=str, keep_default_na=False)
            v["label"] = [int(truth_of[a] == truth_of[b]) for a, b in zip(v.a, v.b)]
            v.to_csv(vl, index=False)
            pipeline.run(vcfg, vl, tmp / f"{name}_{tag}_out", log=lambda *_: None)
            c = pd.read_csv(tmp / f"{name}_{tag}_out/clusters.csv", dtype=str)
            cm = dict(zip(c.source + ":" + c.record_id, c.customer_id))
            out[f"custmatch_1000_{tag}"] = S.clustered_f1(np.array([cm[k] for k in keys]), truth) | {"config": extra}
    if "splink" in tools and name.startswith("dedupe_"):
        frame = next(iter(spec["frames"].values()))
        t0 = time.time()
        i, j, w = splink_plain(name, frame)
        secs = round(time.time() - t0, 1)
        src = next(iter(spec["frames"]))
        idx = {k: n for n, k in enumerate(keys)}
        ids = frame[spec["id_col"]].astype(str).to_numpy()
        a = np.array([idx[f"{src}:{ids[x]}"] for x in i])
        b = np.array([idx[f"{src}:{ids[x]}"] for x in j])
        pairs = np.c_[np.minimum(a, b), np.maximum(a, b)].astype(np.int32)
        li, lj, ly = pipeline.read_labels(lp, df)
        lab_pairs = np.c_[np.minimum(li, lj), np.maximum(li, lj)]
        n = len(df)
        th, _ = pipeline.choose_threshold(n, pairs, lab_pairs, ly, [(w, np.ones(len(ly), bool))], S.WEIGHTS)
        out["splink_from_labels"] = S.clustered_f1(components(n, pairs, w > th), truth) | {
            "match_weight": th, "seconds": secs}
        best = max(((t, S.clustered_f1(components(n, pairs, w > t), truth)) for t in S.WEIGHTS),
                   key=lambda z: z[1]["f1"])
        out["splink_ceiling"] = best[1] | {"match_weight": float(best[0])}

    if "dedupe" not in tools:
        return out
    t0 = time.time()
    try:
        clusters, r = run_dedupe(name, spec, tmp, lab, keys)
        out[dedupe_key] = S.clustered_f1(clusters, truth) | {
            "seconds": r["seconds"], "labels_used": r["labels_used"], "threshold": 0.5,
            "peak_memory_gb": r["peak_memory_gb"], "wall_seconds": round(time.time() - t0, 1)}
    except H.Stopped as e:
        out[dedupe_key] = {"stopped": e.args[0], "seconds_until_stopped": e.args[1], "peak_memory_gb": e.args[2]}
    return out


def main():
    specs = S.datasets() | dedupe_datasets()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = [a[len("--only="):] for a in sys.argv[1:] if a.startswith("--only=")]
    tools = set(only[0].split(",")) if only else {"custmatch", "splink", "dedupe"}
    # dedupe samples at random with no seed: --dedupe-run=2 stores a repeat as "dedupe_run2"
    rerun = [a[len("--dedupe-run="):] for a in sys.argv[1:] if a.startswith("--dedupe-run=")]
    which = args or list(specs)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    ver = "import importlib.metadata as m; print(m.version('dedupe'))"
    res["dedupe_version"] = subprocess.run([str(DEDUPE_PY), "-c", ver], capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        for name in which:
            key = f"dedupe_run{rerun[0]}" if rerun else "dedupe"
            res[name] = run(name, specs[name], Path(tmp), tools, res.get(name, {}), key)
            f1s = {k: v.get("f1", v.get("stopped")) if isinstance(v, dict) else v for k, v in res[name].items()}
            print(name, json.dumps(f1s), flush=True)
            OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
