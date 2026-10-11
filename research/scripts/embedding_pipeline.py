"""Does embedding blocking raise the match score, not only the ceiling?

research/scripts/embedding_blocking.py found that each record's 5 or 10 nearest neighbours by
character 2-4-gram TF-IDF vectors of its names and place lift the blocking ceiling above LSH on
historical_50k and NCVR for a similar number of extra pairs. A higher ceiling only helps if the
forest calls the extra pairs right, so here the whole pipeline is run - labels drawn and the
forest trained over the same candidate pairs - with four blockings:

  keys          the dataset's keys alone
  keys_lsh      the keys plus LSH on the names within the place (lsh_blocking)
  keys_emb5     the keys plus the embedding pass, 5 neighbours
  keys_emb10    the keys plus the embedding pass, 10 neighbours
  keys_ann10    the same 10 neighbours found approximately (pynndescent), as a product would

The embedding pass wraps blocking.candidates_for, so label sampling, active learning and the
pipeline all see the same pairs. Labels: 100 at random then 900 by active learning, answered
from the truth, seeds 1-10. Datasets as in embedding_blocking.py. Scored as clustered F1.

    python research/scripts/embedding_pipeline.py [historical_50k] [ncvr_20k] [--seeds=1,2,3] [--variants=keys,keys_ann10]
                                                 # default: seeds 1-10, every variant
    python research/scripts/embedding_pipeline.py --timing     # each blocking timed alone

--timing: the run times above come from a long batch, where other load on the machine shows; this
draws seed 1's labels once per blocking and times the pipeline three times on them, keeping the
median, into "timing" in the same file.

Writes research/results/embedding_pipeline.json, with the pipeline's own stage times per run (the
embedding pass is built once per dataset, while the first labels are drawn, and cached: its build
time is in embedding_blocking.json, and the pipeline's blocking stage here shows only the union).
"""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from custmatch import blocking, io, pipeline  # noqa: E402

import embedding_blocking as E  # noqa: E402
import label_curve as LC  # noqa: E402
import splink_datasets_benchmark as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/embedding_pipeline.json"
LABELS = 1000
_original = blocking.candidates_for
_cache = {}
EMB = {"k": None, "place": None, "method": "exact"}


def knn_for(df):
    sig = hashlib.sha1("\n".join(io.key(df)).encode()).hexdigest()
    key = (sig, EMB["k"], EMB["place"], EMB["method"])
    if key not in _cache:
        vec = E.char_vectors(E.texts(df, EMB["place"])["names_place"])
        p = (E.ann_pairs if EMB["method"] == "ann" else E.knn_pairs)(vec, EMB["k"])
        p = p[p[:, 0] != p[:, 1]]
        _cache[key] = E.from_set(np.unique(E.as_set(p))).astype(np.int32)
    return _cache[key]


def with_embeddings(df, profiles, cfg, labelled_pos=None, rules=None):
    pairs, hubs, report, used = _original(df, profiles, cfg, labelled_pos, rules)
    if EMB["k"]:
        extra = knn_for(df)
        pairs = E.from_set(np.union1d(E.as_set(pairs), E.as_set(extra))).astype(np.int32)
        report = dict(report, candidate_pairs=int(len(pairs)), embedding_pairs=int(len(extra)))
    return pairs, hubs, report, used


blocking.candidates_for = with_embeddings


def embedding(k):
    """A variant's embedding pass: None, a number of neighbours, or (neighbours, "ann")."""
    k, method = k if isinstance(k, tuple) else (k, "exact")
    return {"k": k, "method": method}


def variants(name):
    lsh = {"lsh_blocking": {"fields": ["first_name", "last_name"], "with": E.PLACE[name]}}
    return {"keys": ({}, None), "keys_lsh": (lsh, None), "keys_emb5": ({}, 5), "keys_emb10": ({}, 10),
            "keys_ann10": ({}, (10, "ann"))}


def timing(name, loader):
    out = {}
    for variant, (extra, k) in variants(name).items():
        EMB.update(embedding(k), place=E.PLACE[name])
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cp, truth = loader(tmp)
            cp.write_text(json.dumps(json.loads(cp.read_text()) | extra))
            lp, _ = LC.draw(cp, truth, LABELS, "active", 1, tmp)
            runs = [pipeline.run(cp, lp, tmp / "out", seed=1, log=lambda *_: None)["stage_seconds"] for _ in range(3)]
        total = sorted(sum(r.values()) for r in runs)[1]
        blk = sorted(next(v for n, v in r.items() if n.startswith("blocking")) for r in runs)[1]
        out[variant] = {"run_seconds": round(total, 1), "blocking_seconds": round(blk, 2)}
        print(name, variant, out[variant], flush=True)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = [v for a in sys.argv[1:] if a.startswith("--variants=") for v in a[11:].split(",")]
    seeds = [int(s) for a in sys.argv[1:] if a.startswith("--seeds=") for s in a[8:].split(",")] or list(range(1, 11))
    loaders = {"historical_50k": lambda tmp: LC.splink_dataset("historical_50k", S.datasets()["historical_50k"], tmp),
               "ncvr_20k": E.ncvr_default}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "--timing" in sys.argv:
        for name in args or list(loaders):
            res.setdefault("timing", {})[name] = timing(name, loaders[name])
            OUT.write_text(json.dumps(res, indent=1, default=float))
        return
    for name in args or list(loaders):
        r = res.setdefault(name, {})
        for variant, (extra, k) in variants(name).items():
            if only and variant not in only:
                continue
            EMB.update(embedding(k), place=E.PLACE[name])
            runs = []
            for seed in seeds:
                with tempfile.TemporaryDirectory() as d:
                    tmp = Path(d)
                    cp, truth = loaders[name](tmp)
                    cp.write_text(json.dumps(json.loads(cp.read_text()) | extra))
                    t0 = time.time()
                    lp, _ = LC.draw(cp, truth, LABELS, "active", seed, tmp)
                    s = pipeline.run(cp, lp, tmp / "out", seed=seed, log=lambda *_: None)
                    sc = LC.score(tmp / "out/clusters.csv", truth)
                    runs.append({"seed": seed, "f1": sc["f1"], "precision": sc.get("precision"),
                                 "recall": sc.get("recall"), "candidate_pairs": s["candidate_pairs"],
                                 "stage_seconds": s["stage_seconds"],
                                 "seconds": round(time.time() - t0, 1)})
                    print(name, variant, runs[-1], flush=True)
            r[variant] = {"runs": runs, "mean_f1": round(float(np.mean([x["f1"] for x in runs])), 2)}
            print(name, variant, "mean", r[variant]["mean_f1"], flush=True)
            OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
