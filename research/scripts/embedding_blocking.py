"""Do embeddings find true pairs that soundex keys and similarity blocking miss?

Blocking decides which pairs are ever compared; a pair it never proposes cannot be matched. On
name-only and dense files that, not the matcher, is the limit (docs/report/06-blocking-and-scale.md).
custmatch's keys are exact values, soundex codes and MinHash LSH on the names' letter pairs
("lsh_blocking"). This adds an embedding pass - each record's nearest neighbours by the cosine of
its embedding - to see whether it raises the ceiling (every true candidate pair accepted and
joined up), and at what cost in extra pairs, against LSH.

Embeddings, built from the names alone or the names with a place (town, birth place):

  char       character 2-4-gram TF-IDF vectors (no model, no download)
  neural     a small sentence model (all-MiniLM-L6-v2), only with --neural and the
             sentence-transformers package installed

Baselines: the dataset's default keys; the keys plus LSH on the names. Datasets: historical_50k
(Splink's, dense genealogy records) and the Leipzig NCVR file, 20,000 people with the default
keys (name-only, corrupted values). Each embedding is tried at k = 5, 10 and 20 neighbours, and the names with the
place at 10 found approximately (ann_pairs; the first call compiles pynndescent, so its time is
an overestimate).

    python research/scripts/embedding_blocking.py [historical_50k] [ncvr_20k] [--neural]

Writes research/results/embedding_blocking.json.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from custmatch import blocking, io  # noqa: E402
from custmatch.core import profile_contacts  # noqa: E402

import label_curve as LC  # noqa: E402
import splink_datasets_benchmark as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/embedding_blocking.json"
KS = (5, 10, 20)
PLACE = {"historical_50k": "birth_place", "ncvr_20k": "town"}


def ncvr_default(tmp):
    import ncvr_leipzig as N
    cp, truth = LC.ncvr(tmp)
    cfg = json.loads(cp.read_text())
    cp.write_text(json.dumps(dict(N.CONFIG, sources=cfg["sources"])))   # the default keys
    return cp, truth


def ceiling(n, pairs, t):
    tp = pairs[t[pairs[:, 0]] == t[pairs[:, 1]]]
    g = coo_matrix((np.ones(len(tp)), (tp[:, 0], tp[:, 1])), shape=(n, n))
    return S.clustered_f1(connected_components(g, directed=False)[1], t)["f1"]


def as_set(pairs):
    a, b = np.minimum(pairs[:, 0], pairs[:, 1]).astype(np.int64), np.maximum(pairs[:, 0], pairs[:, 1]).astype(np.int64)
    return np.unique(a * 10_000_000 + b)


def from_set(codes):
    return np.c_[codes // 10_000_000, codes % 10_000_000]


def knn_pairs(vectors, k, chunk=1000):
    """Each record's k nearest neighbours by cosine (vectors L2-normalised), as pairs."""
    from scipy import sparse
    n, out = vectors.shape[0], []
    dense = not sparse.issparse(vectors)
    for s in range(0, n, chunk):
        sim = vectors[s:s + chunk] @ vectors.T
        sim = np.asarray(sim.todense()) if not dense else sim
        for r in range(sim.shape[0]):
            sim[r, s + r] = -1                      # not itself
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        out.append(np.c_[np.repeat(np.arange(s, s + sim.shape[0]), k), top.ravel()])
    return np.vstack(out)


def ann_pairs(vectors, k, graph=16, seed=0):
    """Each record's k nearest neighbours by cosine, approximately: an NN-descent graph of `graph`
    neighbours (pynndescent), its k nearest kept. Seconds where knn_pairs takes minutes."""
    from pynndescent import NNDescent
    nb, _ = NNDescent(vectors, metric="cosine", n_neighbors=graph, random_state=seed, n_jobs=-1,
                      low_memory=True).neighbor_graph
    nb = nb[:, :k + 1]                          # the record itself is usually first
    p = np.c_[np.repeat(np.arange(len(nb)), nb.shape[1]), nb.ravel()]
    return p[(p[:, 0] != p[:, 1]) & (p[:, 1] >= 0)]


def texts(df, place):
    name = (df.first_name.fillna("") + " " + df.last_name.fillna("")).str.strip()
    return {"names": name.tolist(),
            "names_place": (name + " | " + df[place].fillna("").astype(str)).tolist() if place in df else None}


def char_vectors(strings):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit_transform(strings)


def neural_vectors(strings):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    return m.encode(strings, batch_size=256, normalize_embeddings=True, show_progress_bar=False)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    neural = "--neural" in sys.argv
    loaders = {"historical_50k": lambda tmp: LC.splink_dataset("historical_50k", S.datasets()["historical_50k"], tmp),
               "ncvr_20k": ncvr_default}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in args or list(loaders):
        with tempfile.TemporaryDirectory() as d:
            cp, truth = loaders[name](Path(d))
            df, cfg = io.load(cp)
            t = np.array([truth[k] for k in io.key(df)])
            n = len(df)
            profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
            place = PLACE[name]
            r = res.setdefault(name, {})
            r["records"] = n
            base = {}
            for label, extra in (("keys", {}), ("keys_lsh", {"lsh_blocking": {"fields": ["first_name", "last_name"], "with": place}})):
                t0 = time.time()
                pairs, *_ = blocking.candidates_for(df, profiles, cfg | extra)
                base[label] = as_set(pairs)
                r[label] = {"pairs": int(len(pairs)), "ceiling": ceiling(n, pairs, t), "seconds": round(time.time() - t0, 1)}
                print(name, label, r[label], flush=True)
            kinds = {"char": char_vectors} | ({"neural": neural_vectors} if neural else {})
            for kind, build in kinds.items():
                for field, strings in texts(df, place).items():
                    if strings is None:
                        continue
                    t0 = time.time()
                    vec = build(strings)
                    built = time.time() - t0
                    for k in KS:
                        t1 = time.time()
                        emb = as_set(knn_pairs(vec, k))
                        secs = built + time.time() - t1
                        for b in ("keys", "keys_lsh"):
                            union = np.union1d(base[b], emb)
                            added = from_set(np.setdiff1d(union, base[b]))
                            new_true = int((t[added[:, 0]] == t[added[:, 1]]).sum())
                            key = f"{b}+{kind}_{field}_k{k}"
                            r[key] = {"pairs": int(len(union)), "added_pairs": int(len(added)), "added_true_pairs": new_true,
                                      "ceiling": ceiling(n, from_set(union), t), "embedding_seconds": round(secs, 1)}
                            print(name, key, r[key], flush=True)
                    OUT.write_text(json.dumps(res, indent=1, default=float))
            # the same 10 neighbours of names and place, found approximately (as a product would)
            t0 = time.time()
            emb = as_set(ann_pairs(char_vectors(texts(df, place)["names_place"]), 10))
            secs = time.time() - t0
            for b in ("keys", "keys_lsh"):
                union = np.union1d(base[b], emb)
                r[f"{b}+ann_names_place_k10"] = {"pairs": int(len(union)), "ceiling": ceiling(n, from_set(union), t),
                                                 "embedding_seconds": round(secs, 1)}
                print(name, f"{b}+ann_names_place_k10", r[f"{b}+ann_names_place_k10"], flush=True)
            OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
