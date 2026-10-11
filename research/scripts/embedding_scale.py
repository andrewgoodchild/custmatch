"""How does embedding blocking scale, against LSH? Time and memory on the Leipzig NCVR file.

research/scripts/embedding_pipeline.py found 10 nearest neighbours by character n-gram vectors
of the names and town beat LSH, and that an approximate search (pynndescent) finds them in
seconds where comparing every record with every other takes minutes. Here, on 100,000 records
up to the whole file (5 million), each size in a fresh process:

  vectors     character 2-4-gram TF-IDF of "first last | town"
  ann         each record's 10 nearest neighbours, approximately (embedding_blocking.ann_pairs)
  keys        custmatch's candidate pairs with the default keys
  keys_lsh    the same plus LSH on the names within the town

with the peak memory of the process. Blocking runs up to --blocking-max records (default
1,000,000); custmatch_scale.py and modal_scale.py time the whole pipeline at larger sizes.

    python research/scripts/embedding_scale.py [--sizes=100000,500000,1000000,2000000,5000000]

Writes research/results/embedding_scale.json.
"""
import json
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/embedding_scale.json"


def peak_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1 << 30 if sys.platform == "darwin" else 1 << 20), 2)   # bytes on macOS, KiB on Linux


def one(n, blocking_max):
    import ncvr_leipzig as N
    import embedding_blocking as E
    from custmatch import blocking, io
    from custmatch.core import profile_contacts
    allrec = N.load()
    sub = allrec.iloc[np.sort(np.random.default_rng(0).choice(len(allrec), n, replace=False))] if n < len(allrec) else allrec
    out = {"records": int(len(sub))}
    text = (sub.givenname + " " + sub.surname).str.strip() + " | " + sub.suburb
    t = time.time()
    vec = E.char_vectors(text.tolist())
    out["vectors_seconds"] = round(time.time() - t, 1)
    E.ann_pairs(vec[:2000], 10)                   # compile pynndescent's numba code outside the timing
    t = time.time()
    pairs = E.ann_pairs(vec, 10)
    out["ann_seconds"] = round(time.time() - t, 1)
    out["ann_pairs"] = int(len(E.as_set(pairs)))
    out["peak_gb_after_ann"] = peak_gb()
    del vec, pairs
    if n <= blocking_max:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            cfg = dict(N.CONFIG, sources=[])
            for s, g in sub.groupby("source"):
                g = g.reset_index(drop=True).assign(rid=lambda x: x.recid + "_" + x.index.astype(str))
                g.to_parquet(tmp / f"{s}.parquet", index=False)
                cfg["sources"].append({"name": s, "path": f"{s}.parquet", "columns": {
                    "record_id": "rid", "first_name": "givenname", "last_name": "surname",
                    "town": "suburb", "postcode": "postcode"}})
            (tmp / "c.json").write_text(json.dumps(cfg))
            df, cfg = io.load(tmp / "c.json")
        profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
        for label, extra in (("keys", {}), ("keys_lsh", {"lsh_blocking": {"fields": ["first_name", "last_name"],
                                                                           "with": "town"}})):
            t = time.time()
            p, *_ = blocking.candidates_for(df, profiles, cfg | extra)
            out[f"{label}_seconds"] = round(time.time() - t, 1)
            out[f"{label}_pairs"] = int(len(p))
        out["peak_gb"] = peak_gb()
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print(json.dumps(one(int(sys.argv[2]), int(sys.argv[3]))))
        return
    sizes = [int(x) for a in sys.argv[1:] if a.startswith("--sizes=") for x in a[8:].split(",")] \
        or [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    bmax = next((int(a[15:]) for a in sys.argv[1:] if a.startswith("--blocking-max=")), 1_000_000)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for n in sizes:
        r = subprocess.run([sys.executable, __file__, "--one", str(n), str(bmax)], capture_output=True, text=True)
        if r.returncode:
            print(n, "failed:", r.stderr[-2000:], flush=True)
            res[str(n)] = {"failed": r.stderr[-300:]}
        else:
            res[str(n)] = json.loads(r.stdout.strip().splitlines()[-1])
            print(n, res[str(n)], flush=True)
        OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
