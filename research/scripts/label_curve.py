"""How many labels does custmatch need, and how far does it get with none?

For each dataset: custmatch's defaults trained on 50, 100, 200, 300, 500 and 1,000 labelled pairs,
drawn two ways - all at random, or 100 at random then batches of 100 chosen by active learning
(the dashboard's and `custmatch sample --active`'s workflow) - over three seeds. Labels are
answered from the truth. Below 200 labels the pipeline normally refuses to run (MIN_LABELS); here
the floor is lowered to see whether it is in the right place.

With no labels at all: custmatch's own Fellegi-Sunter match weight (custmatch/matchweight.py,
m by EM on the candidate pairs, no labels) scores every candidate pair, and the clusters are
taken at label-free cut-offs - the top share of pairs EM estimates to be matches ("top_lambda"),
and fixed weights - and at the best weight in hindsight (an upper bound no real run can know).

Datasets: Splink's four (fake_1000, febrl3, febrl4, historical_50k), the Leipzig NCVR file
(20,000 people, postcode and initial keys; data/ncvr_leipzig, see ncvr_leipzig.py) and the
synthetic Australian customer file (data/au_customer, customer_data.py). Scored as clustered F1.

    python research/scripts/label_curve.py [datasets] [--seeds=1,2,3]

Writes research/results/label_curve.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from custmatch import blocking, io, labels, matchweight, pipeline  # noqa: E402
from custmatch.core import components, profile_contacts  # noqa: E402

import splink_datasets_benchmark as S  # noqa: E402

OUT = ROOT / "research/results/label_curve.json"
SIZES = (50, 100, 200, 300, 500, 1000)
FIXED_WEIGHTS = (0, 5, 10, 15, 20)


def splink_dataset(name, spec, tmp):
    cfg = {"sources": [], "extra_fields": spec["extra_fields"], "extra_blocking": spec["extra_blocking"]}
    truth = {}
    for src, frame in spec["frames"].items():
        frame.fillna("").astype(str).to_csv(tmp / f"{src}.csv", index=False)
        cfg["sources"].append({"name": src, "path": f"{src}.csv", "columns": spec["columns"]})
        truth.update(dict(zip(src + ":" + frame[spec["id_col"]].astype(str), frame[spec["truth_col"]].astype(str))))
    (tmp / "config.json").write_text(json.dumps(cfg))
    return tmp / "config.json", truth


def ncvr(tmp):
    import ncvr_leipzig as N
    sub = N.sample(N.load(), 20_000, 1)
    cfg, truth = dict(N.CONFIG_KEYS, sources=[]), {}
    for s, g in sub.groupby("source"):
        g = g.reset_index(drop=True).assign(rid=lambda d: d.recid + "_" + d.index.astype(str))
        g.to_parquet(tmp / f"{s}.parquet", index=False)
        cfg["sources"].append({"name": s, "path": f"{s}.parquet", "columns": {
            "record_id": "rid", "first_name": "givenname", "last_name": "surname",
            "town": "suburb", "postcode": "postcode"}})
        truth.update(dict(zip(s + ":" + g.rid, g.recid)))
    (tmp / "config.json").write_text(json.dumps(cfg))
    return tmp / "config.json", truth


def au(tmp):
    import au_validate as A
    gold, _ = A.truth()
    cfg = json.loads(A.CONFIG.read_text()) | {"country": "AU"}
    for s in cfg["sources"]:
        s["path"] = str((A.CONFIG.parent / s["path"]).resolve())
    (tmp / "config.json").write_text(json.dumps(cfg))
    return tmp / "config.json", {k: str(v[0]) for k, v in gold.items()}


def datasets():
    specs = S.datasets()
    out = {name: (lambda tmp, name=name: splink_dataset(name, specs[name], tmp)) for name in specs}
    if (ROOT / "data/ncvr_leipzig").exists():
        out["ncvr_20k"] = ncvr
    if (ROOT / "data/au_customer").exists():
        out["au_customer"] = au
    return out


def fill(path, truth):
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    lab["label"] = [int(truth[a] == truth[b]) for a, b in zip(lab.a, lab.b)]
    lab.to_csv(path, index=False)
    return lab


def draw(cp, truth, n, how, seed, tmp):
    lp = tmp / "labels.csv"
    first = n if how == "random" else min(n, 100)
    labels.sample(cp, first, lp, seed=seed)
    lab = fill(lp, truth)
    while len(lab) < n:
        labels.sample_active(cp, lp, min(100, n - len(lab)), tmp / "next.csv", seed=seed)
        nxt = fill(tmp / "next.csv", truth)
        if nxt.empty:
            break
        lab = pd.concat([lab, nxt], ignore_index=True)
        lab.to_csv(lp, index=False)
    return lp, lab


def score(clusters_csv, truth):
    c = pd.read_csv(clusters_csv, dtype=str)
    keys = c.source + ":" + c.record_id
    return S.clustered_f1(c.customer_id.to_numpy(), np.array([truth[k] for k in keys]))


def zero_label(cp, truth):
    df, cfg = io.load(cp)
    keys = io.key(df)
    t = np.array([truth[k] for k in keys])
    profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
    pairs, *_ = blocking.candidates_for(df, profiles, cfg)
    model = matchweight.fit(df, profiles, pairs, cfg)
    w = matchweight.load_scorer(df, profiles, model).weights(pairs)
    n = len(df)

    def at(keep):
        return S.clustered_f1(components(n, pairs, keep), t)
    res = {"lambda": round(model["lambda"], 4), "candidate_pairs": int(len(pairs))}
    k = int(round(model["lambda"] * len(pairs)))
    order = np.argsort(-w)
    keep = np.zeros(len(pairs), bool)
    keep[order[:k]] = True
    res["top_lambda"] = at(keep) | {"pairs_accepted": k}
    for f in FIXED_WEIGHTS:
        res[f"weight_{f}"] = at(w >= f)
    grid = np.quantile(w, np.linspace(0.5, 0.999, 60))
    best = max(((at(w >= g)["f1"], g) for g in grid))
    res["best_in_hindsight"] = {"f1": best[0], "weight": round(float(best[1]), 2)}
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = [int(s) for a in sys.argv[1:] if a.startswith("--seeds=") for s in a[8:].split(",")] or [1, 2, 3]
    pipeline.MIN_LABELS = 20                     # measure below the product's floor
    data = datasets()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args or list(data):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cp, truth = data[name](tmp)
            r = res.setdefault(name, {})
            r["zero_label"] = zero_label(cp, truth)
            print(name, "zero labels", json.dumps({k: v["f1"] if isinstance(v, dict) else v
                                                    for k, v in r["zero_label"].items()}), flush=True)
            for how in ("random", "active"):
                for n in SIZES:
                    if how == "active" and n <= 100:
                        continue                 # the first 100 are random either way
                    runs = []
                    for seed in seeds:
                        lp, lab = draw(cp, truth, n, how, seed, tmp)
                        try:
                            pipeline.run(cp, lp, tmp / "out", seed=seed, log=lambda *_: None)
                            runs.append(score(tmp / "out/clusters.csv", truth)["f1"])
                        except io.ConfigError as e:          # e.g. no matches among the labels
                            runs.append(None)
                            print(name, how, n, seed, "no run:", e, flush=True)
                    ok = [x for x in runs if x is not None]
                    r[f"{how}_{n}"] = {"f1_by_seed": runs, "mean": round(float(np.mean(ok)), 2) if ok else None}
                    print(name, how, n, r[f"{how}_{n}"], flush=True)
                    OUT.write_text(json.dumps(res, indent=1, default=float))
        OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
