"""How does custmatch scale? Time each stage on K disjoint copies of the synthetic file.

Copy k gets a copy-specific suffix on surname, street, email and phone, so copies never match
one another and every block, household and junk pattern is reproduced K times - the structure
of a larger customer file, with known timing. Syntactic junk values (test@test.com,
1111111111) stay identical across copies, as placeholders recur in a real file.

Writes research/results/custmatch_scale.json.
"""
import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custmatch_validate as V  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.core import block, features, profile_contacts, looks_fake  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def scaled(k, tmp):
    """A config over k disjoint copies of the synthetic sources."""
    for name in ("crm", "billing"):
        base = pd.read_parquet(ROOT / f"data/customer/source_{name}.parquet").astype(str)
        parts = []
        for c in range(k):
            d = base.copy()
            tag = f"x{c}"
            d["record_id"] = d.record_id + "-" + tag
            d["last_name"] = d.last_name + tag
            d["addr_street"] = d.addr_street + " " + tag
            d["email"] = [e if (not e or looks_fake(e)) else f"{tag}.{e}" for e in d.email]
            d["phone"] = [p if (not p or looks_fake(p)) else f"{c:03d}{p}" for p in d.phone]
            parts.append(d)
        pd.concat(parts, ignore_index=True).to_parquet(tmp / f"{name}.parquet", index=False)
        cfg = json.load(open(V.CONFIG))
    for s in cfg["sources"]:
        s["path"] = str(tmp / f"{s['name']}.parquet")
    (tmp / "config.json").write_text(json.dumps(cfg))
    return tmp / "config.json"


def main():
    # optional blocking mode as the first argument: fixed | learned
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else None
    args = sys.argv[2:] if mode else sys.argv[1:]
    sizes = [int(x) for x in args] or [1, 5, 25]
    out = []
    for k in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = scaled(k, tmp)
            if mode:
                c = json.loads(cfg.read_text()) | {"blocking": mode}
                cfg.write_text(json.dumps(c))
            t = {}
            t0 = time.time(); df, _ = io.load(cfg); t["load"] = time.time() - t0
            t0 = time.time(); prof = {f: profile_contacts(df, f) for f in ("email", "phone")}; t["profile"] = time.time() - t0
            t0 = time.time(); pairs, _ = block(df, prof); t["block"] = time.time() - t0
            t0 = time.time(); features(df, pairs, prof); t["features"] = time.time() - t0
            lp = tmp / "labels.csv"
            labels.sample(cfg, 1000, lp, seed=1)
            lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
            gold = {}
            for c in range(k):
                for key, (s, h) in V.truth().items():
                    src, rid = key.split(":", 1)
                    gold[f"{src}:{rid}-x{c}"] = (f"{s}-x{c}", f"{h}-x{c}")
            lab["label"] = [int(gold[a][0] == gold[b][0]) for a, b in zip(lab.a, lab.b)]
            lab.to_csv(lp, index=False)
            t0 = time.time()
            pipeline.run(cfg, lp, tmp / "out", log=lambda *_: None)
            t["full_run"] = time.time() - t0
            r = V.score(tmp / "out", gold, lab)
            summ = json.loads((tmp / "out" / "summary.json").read_text())
            row = {"copies": k, "records": len(df), "blocking": mode or "core",
                   "run_candidate_pairs": summ["candidate_pairs"], "stage_seconds": summ.get("stage_seconds"),
                   "candidate_pairs": int(len(pairs)),
                   **{f"{s}_seconds": round(v, 2) for s, v in t.items()},
                   "clustered_f1": r["all_records"]["f1"]}
            out.append(row)
            print(json.dumps(row), flush=True)
    name = f"research/results/custmatch_scale_{mode}.json" if mode else "research/results/custmatch_scale.json"
    (ROOT / name).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
