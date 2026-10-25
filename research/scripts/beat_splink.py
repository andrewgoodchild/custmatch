"""What does it take for custmatch to beat Splink everywhere? Each variant on all five datasets.

Datasets: Splink's febrl3, febrl4, historical_50k and fake_1000 (tailored config, as Splink's
tutorial is) plus the synthetic customer file. For each variant (a config change) custmatch
runs twice - 1,000 random labels, and 300 random + 700 active - with the same label seeds as
the earlier benchmarks, and is scored as clustered F1 against the truth. Splink's figures are
its official-settings results from research/results/splink_datasets_benchmark.json and
research/results/splink_vs_custmatch.json.

  beat_splink.py VARIANT [DATASET ...]      variants: base, rich, ...
  beat_splink.py VARIANT --seed 2 [DATASET ...]   another label draw, kept as "VARIANT/seed2"
Writes research/results/beat_splink.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custmatch_validate as V  # noqa: E402
import jev_labelling  # noqa: E402  (the LLM labelling researched in report chapter 4)
import splink_datasets_benchmark as B  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["febrl3", "febrl4", "historical_50k", "fake_1000", "synthetic"]
VARIANTS = {                         # config overrides; "_mix" / "_jev" change the labelling
    "base": {"rich_features": False},        # rich features were off by default when these ran
    "rich": {"rich_features": True},
    "base_mix": {"rich_features": False, "_mix": 0.5},
    # splink, rich_splink, rich_splink_mix, jev_rich_splink and rich_splink_v5 used the Splink
    # feature, removed in 0.1.0; their results stay in research/results/beat_splink.json
    # blocking lessons: fixed rules with oversized blocks split (Zingg), and rules learned from
    # the labelled matches (dedupe); max_block left at its default 60 on purpose
    "rich_split": {"blocking": "fixed", "max_block": 60},
    # oversized blocks split on three fields at once (research/scripts/ncvr_blocking_scale.py),
    # and rich_split again on the same code, since its saved figures predate 0.1.2
    "rich_adaptive": {"blocking": "adaptive", "max_block": 60},
    "rich_split_0_1_2": {"blocking": "fixed", "max_block": 60},
    # random labels drawn stratified by fields agreeing, weighted back (labels.stratified_pick)
    "rich_stratified": {"rich_features": True, "_stratified": True},
    "rich_rerun": {"rich_features": True},       # rich on current code: rich_stratified's baseline
    "rich_adaptive_stratified": {"blocking": "adaptive", "max_block": 60, "_stratified": True},
    "rich_learned": {"blocking": "learned", "max_block": 60},
    "rich_learned99": {"blocking": "learned", "max_block": 60, "blocking_recall": 0.99},
    # active learning that also looks where blocking missed (dedupe), feeding learned blocking
    "rich_learned_disagree": {"blocking": "learned", "max_block": 60, "_strategy": "disagreement"},
    # re-cut customers over 30 records at stricter thresholds (dedupe); 30 is a fixed guess
    "rich_recut30": {"max_cluster": 30},
}


def splink_reference():
    b = json.loads((ROOT / "research/results/splink_datasets_benchmark.json").read_text())
    s = json.loads((ROOT / "research/results/splink_vs_custmatch.json").read_text())
    return {"febrl3": b["febrl3"]["splink_ceiling"]["f1"],          # no threshold documented
            "febrl4": b["febrl4"]["splink_documented"]["f1"],
            "historical_50k": b["historical_50k"]["splink_documented"]["f1"],
            "fake_1000": b["fake_1000"]["splink_documented_loose_blocking"]["f1"],
            "synthetic": max(s["splink_unsupervised"]["ceiling_threshold_picked_on_truth"]["f1"],
                             s["splink_m_from_500_labels"]["threshold_from_labels"]["f1"])}


# historical_50k: similarity blocking on the names within a birth place, added in 0.1.0. Half its
# records lack a surname or birth date, so exact keys left about 2,450 of 5,156 people split
# whatever the matcher did (ceiling 93.3); this lifts the ceiling to 96.8 for 13% more pairs
# (research/scripts/historical_blocking.py). Results recorded before it was added used the keys alone.
HISTORICAL_LSH = {"fields": ["first_name", "last_name"], "with": "birth_place"}


def setup(name, tmp):
    """(config path, truth dict source:id -> entity) for a dataset."""
    if name == "synthetic":
        gold = V.truth()
        return V.CONFIG, {k: v[0] for k, v in gold.items()}
    spec = B.datasets()[name]
    cfg = {"sources": [], "extra_fields": spec["extra_fields"], "extra_blocking": spec["extra_blocking"]}
    if name == "fake_1000":   # the tailored config, matching Splink's tutorial-07 setup
        cfg.update(extra_blocking=[["first_name"], ["city"], ["email"], ["date_of_birth"]], max_block=200,
                   extra_fields=spec["extra_fields"] + [{"name": "date_of_birth", "compare": "edit"},
                                                        {"name": "email", "compare": "similarity"}])
    if name == "historical_50k":
        cfg["lsh_blocking"] = HISTORICAL_LSH
    truth = {}
    for src, frame in spec["frames"].items():
        path = tmp / f"{name}_{src}.csv"
        frame.fillna("").astype(str).to_csv(path, index=False)
        cfg["sources"].append({"name": src, "path": path.name, "columns": spec["columns"]})
        truth.update(dict(zip(src + ":" + frame[spec["id_col"]].astype(str), frame[spec["truth_col"]].astype(str))))
    (tmp / f"{name}.json").write_text(json.dumps(cfg))
    return tmp / f"{name}.json", truth


def with_overrides(config, overrides, tmp, tag):
    cfg = json.loads(Path(config).read_text())
    for s in cfg["sources"]:                      # keep paths valid from the new location
        s["path"] = str((Path(config).parent / s["path"]).resolve())
    cfg.update({k: v for k, v in overrides.items() if not k.startswith("_")})
    if overrides.get("blocking"):             # the blocker modes take exact keys only, so
        cfg.pop("lsh_blocking", None)         # historical_50k runs them without similarity blocking
    out = tmp / f"{tag}.json"
    out.write_text(json.dumps(cfg))
    return out


def evaluate(config, truth, tmp, tag, mix=0.0, jev=False, strategy="uncertainty", stratified=False, seed=1):
    def fill(path):
        lab = pd.read_csv(path, dtype=str, keep_default_na=False)
        lab["label"] = [int(truth[a] == truth[b]) for a, b in zip(lab.a, lab.b)]
        lab.to_csv(path, index=False)
        return lab

    def score(out_dir):
        c = pd.read_csv(out_dir / "clusters.csv", dtype=str)
        t = np.array([truth[f"{s}:{r}"] for s, r in zip(c.source, c.record_id)])
        return B.clustered_f1(c.customer_id.to_numpy(), t)["f1"]

    res = {}
    df, cfg = io.load(config)
    if jev:
        # no hand labels beyond a 100-pair check: Jev labels 900 random candidate pairs, gated
        check = tmp / f"{tag}_check.csv"
        labels.sample(config, 100, check, seed=7, stratified=False)
        fill(check)
        gate = jev_labelling.label(config, 900, check, tmp / f"{tag}_jev.csv", min_accuracy=0.0, max_cost=0.10)
        jl = pd.read_csv(tmp / f"{tag}_jev.csv", dtype=str, keep_default_na=False)
        jl.loc[jl.by == "jev", "by"] = "random"      # Jev's pairs are a random draw: they may choose the threshold
        jl.to_csv(tmp / f"{tag}_jev.csv", index=False)
        pipeline.run(config, tmp / f"{tag}_jev.csv", tmp / f"{tag}_j", log=lambda *_: None)
        return {"jev_900_plus_100_hand": score(tmp / f"{tag}_j"), "gate": gate}
    n = min(1000, len(labels.candidates(df, cfg)))
    lp = tmp / f"{tag}_r.csv"
    labels.sample(config, n, lp, seed=seed, stratified=stratified)
    fill(lp)
    import time
    t0 = time.time()
    summ = pipeline.run(config, lp, tmp / f"{tag}_r", log=lambda *_: None)
    res["random_1000"] = score(tmp / f"{tag}_r")
    res["candidate_pairs"] = summ["candidate_pairs"]
    res["seconds"] = round(time.time() - t0, 1)
    ap = tmp / f"{tag}_a.csv"
    labels.sample(config, 300, ap, seed=seed, stratified=stratified)
    al = fill(ap)
    while len(al) < 1000:
        labels.sample_active(config, ap, 100, tmp / "next.csv", mix=mix, strategy=strategy)
        al = pd.concat([al, fill(tmp / "next.csv")], ignore_index=True)
        al.to_csv(ap, index=False)
    pipeline.run(config, ap, tmp / f"{tag}_a", log=lambda *_: None)
    res["active_300_700"] = score(tmp / f"{tag}_a")
    return res


def main():
    args = sys.argv[1:]
    seed = 1
    if "--seed" in args:
        k = args.index("--seed")
        seed = int(args[k + 1])
        del args[k:k + 2]
    variant, which = args[0], args[1:] or DATASETS
    key = variant if seed == 1 else f"{variant}/seed{seed}"
    path = ROOT / "research/results/beat_splink.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out["splink_reference"] = splink_reference()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in which:
            config, truth = setup(name, tmp)
            config = with_overrides(config, VARIANTS[variant], tmp, f"{name}_{variant}")
            ov = VARIANTS[variant]
            r = evaluate(config, truth, tmp, f"{name}_{variant}", mix=ov.get("_mix", 0.0), jev=ov.get("_jev", False),
                         strategy=ov.get("_strategy", "uncertainty"), stratified=ov.get("_stratified", False), seed=seed)
            out.setdefault(key, {})[name] = r
            shown = "  ".join(f"{k} {v:6.2f}" for k, v in r.items() if isinstance(v, float))
            print(f"{key:16s} {name:15s} {shown}   | Splink {out['splink_reference'][name]:6.2f}", flush=True)
            path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
