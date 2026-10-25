"""Record dates: does knowing when each record was written help?

Builds data/au_customer_dated from the sparse Australian file (research/scripts/au_customer_data.py),
giving every record a date and three things that happen over time:

  moves          30% of people move once: their records after the move carry a new address
                 (store) and postcode (web)
  name changes   15% of people with a surname change it once (marriage): later records carry
                 the new one
  recycled       20% of mobiles are reissued: a stranger (a new person, own household) gets the
  mobiles        number 1 to 4 years after its owner's last record, and has two records with it

Then runs custmatch as au_validate.py does (300 random labels plus 700 by active learning,
labelled from the truth; three label draws) with and without "record_dates": true, and scores:
all records (clustered F1), pairs of one person across a move or a name change (share kept
together: recall), and owner-stranger pairs sharing a recycled mobile (share wrongly together).

    python research/scripts/record_dates_research.py [--rebuild]

Writes research/results/record_dates.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import labels, pipeline
from custmatch.core import partition_scores

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/au_customer"
DATA = ROOT / "data/au_customer_dated"
CONFIG = ROOT / "examples/au/config.json"
OUT = ROOT / "research/results/record_dates.json"
SOURCES = ("web", "app", "store")
SEEDS = [1, 2, 3]
MOVE, RENAME, RECYCLE = 0.30, 0.15, 0.20
VARIANTS = {"without_dates": {}, "record_dates": {"record_dates": True}}


def build(seed=0):
    rng = np.random.default_rng(seed)
    frames = {s: pd.read_parquet(SRC / f"source_{s}.parquet").assign(record_id=lambda d: d.record_id.astype(str))
              for s in SOURCES}
    allp = pd.concat([f[["person_id"]] for f in frames.values()]).person_id.unique()
    start = dict(zip(allp, rng.uniform(2012, 2020, len(allp))))
    for s, f in frames.items():
        f["_t"] = [start[p] + rng.uniform(0, 6) for p in f.person_id]
    # events, each at a time inside the person's own span so records fall on both sides
    span = pd.concat([f[["person_id", "_t"]] for f in frames.values()]).groupby("person_id")._t.agg(["min", "max", "count"])
    multi = span[span["count"] >= 2]
    move_at = {p: rng.uniform(r["min"], r["max"]) for p, r in multi.iterrows() if rng.random() < MOVE}
    surnamed = set(pd.concat([f[f.last_name.str.strip() != ""].person_id for f in frames.values()]))
    rename_at = {p: rng.uniform(r["min"], r["max"]) for p, r in multi.iterrows()
                 if p in surnamed and rng.random() < RENAME}
    streets = frames["store"].address[frames["store"].address != ""].str.replace(r"^[\w/, ]*?\d+\w?\s+", "", regex=True)
    streets = streets[streets.str.len() > 5].tolist()
    postcodes = frames["web"].postcode[frames["web"].postcode != ""].tolist()
    surnames = pd.concat([f.last_name for f in frames.values()])
    surnames = surnames[surnames.str.strip() != ""].tolist()
    firsts = pd.concat([f.first_name for f in frames.values()])
    firsts = firsts[firsts.str.strip() != ""].tolist()
    new_addr = {p: f"{rng.integers(1, 300)} {streets[rng.integers(len(streets))]}" for p in move_at}
    new_pc = {p: postcodes[rng.integers(len(postcodes))] for p in move_at}
    new_ln = {p: surnames[rng.integers(len(surnames))] for p in rename_at}
    for s, f in frames.items():
        moved = np.array([p in move_at and t > move_at[p] for p, t in zip(f.person_id, f._t)])
        if "address" in f:
            f.loc[moved & (f.address != ""), "address"] = [new_addr[p] for p in f.person_id[moved & (f.address != "")]]
        if "postcode" in f:
            f.loc[moved & (f.postcode != ""), "postcode"] = [new_pc[p] for p in f.person_id[moved & (f.postcode != "")]]
        f["_moved"] = moved
        renamed = np.array([p in rename_at and t > rename_at[p] for p, t in zip(f.person_id, f._t)])
        f.loc[renamed & (f.last_name != ""), "last_name"] = [new_ln[p] for p in f.person_id[renamed & (f.last_name != "")]]
        f["_renamed"] = renamed
    # recycled mobiles: strangers with an owner's number, after the owner's last record
    owners = pd.concat([f[f.mobile.str.replace(r"\D", "", regex=True).str.len() >= 9][["person_id", "mobile"]]
                        for f in frames.values()]).drop_duplicates("person_id")
    d = owners.mobile.str.replace(r"\D", "", regex=True)
    owners = owners[[len(set(x[-8:])) > 2 for x in d]]             # not 0400000000 and the like
    next_pid = int(max(f.person_id.max() for f in frames.values())) + 1
    next_hh = int(max(f.household_id.max() for f in frames.values())) + 1
    extra = {"app": [], "web": []}
    for _, o in owners.iterrows():
        if rng.random() >= RECYCLE:
            continue
        first, last = firsts[rng.integers(len(firsts))], surnames[rng.integers(len(surnames))]
        t0 = span.loc[o.person_id, "max"] + rng.uniform(1, 4)
        for k in range(2):
            s = ("app", "web")[k]
            rec = {"record_id": f"rc{next_pid}_{k}", "first_name": first, "last_name": last,
                   "email": f"{first}.{last}{rng.integers(10, 99)}@gmail.com".lower().replace(" ", ""),
                   "mobile": o.mobile, "date_of_birth": "", "person_id": next_pid, "household_id": next_hh,
                   "_t": t0 + rng.uniform(0, 1), "_moved": False, "_renamed": False,
                   "_recycled_from": o.person_id}
            if s == "web":
                rec["postcode"] = postcodes[rng.integers(len(postcodes))]
            extra[s].append(rec)
        next_pid += 1
        next_hh += 1
    DATA.mkdir(exist_ok=True)
    for s, f in frames.items():
        if s in extra and extra[s]:
            f = pd.concat([f, pd.DataFrame(extra[s])], ignore_index=True)
        f["_recycled_from"] = f.get("_recycled_from", pd.Series(np.nan, index=f.index))
        when = pd.to_datetime((f._t - 1970) * 365.25, unit="D", origin="unix")
        f["updated_at"] = when.dt.strftime("%Y-%m-%d")
        f.to_parquet(DATA / f"source_{s}.parquet", index=False)
    stats = {"people_moved": len(move_at), "people_renamed": len(rename_at),
             "recycled_mobiles": int(sum(len(v) for v in extra.values()) // 2)}
    (DATA / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def truth():
    t = pd.concat([pd.read_parquet(DATA / f"source_{s}.parquet").assign(source=s) for s in SOURCES], ignore_index=True)
    t["key"] = t.source + ":" + t.record_id.astype(str)
    return t


def fill(path, pid):
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    lab["label"] = [int(pid[a] == pid[b]) for a, b in zip(lab.a, lab.b)]
    lab.to_csv(path, index=False)
    return lab


def n_true(ids):
    _, n = np.unique(ids, return_counts=True)
    return float(np.sum(n * (n - 1) / 2))


def score(out_dir, t):
    c = pd.read_csv(out_dir / "clusters.csv", dtype=str)
    cust = dict(zip(c.source + ":" + c.record_id, c.customer_id))
    k = (c.source + ":" + c.record_id).to_numpy()
    pid_of = dict(zip(t.key, t.person_id))
    pid = np.array([pid_of[x] for x in k])
    res = {"all_records": partition_scores(c.customer_id.to_numpy(), pid, n_true(pid))}
    # one person's records on both sides of a move / a name change: kept together?
    for flag, name in (("_moved", "across_a_move"), ("_renamed", "across_a_name_change")):
        g = t[t.person_id.isin(t.person_id[t[flag]])]
        pairs = [(a, b) for _, r in g.groupby("person_id") for a in r.key[r[flag]] for b in r.key[~r[flag]]]
        res[name] = {"pairs": len(pairs),
                     "kept_together_pct": round(100 * np.mean([cust[a] == cust[b] for a, b in pairs]), 2) if pairs else None}
    # owner records holding the mobile vs the stranger's: wrongly together?
    rc = t[t._recycled_from.notna()]
    digits = t.mobile.str.replace(r"\D", "", regex=True).str[-9:]
    pairs = [(a, b) for _, r in rc.iterrows()
             for a in t.key[(t.person_id == r._recycled_from) & (digits == digits[r.name])] for b in [r.key]]
    res["recycled_mobile_owner_vs_stranger"] = {
        "pairs": len(pairs), "wrongly_together_pct": round(100 * np.mean([cust[a] == cust[b] for a, b in pairs]), 2)}
    return res


def main():
    if "--data-seeds" in sys.argv:                # one draw on each of several generated files
        seeds = [int(x) for x in sys.argv[sys.argv.index("--data-seeds") + 1].split(",")]
        res = json.loads(OUT.read_text()) if OUT.exists() else {}
        for ds in seeds:
            build(ds)
            out = run_variants(label_seeds=[1])
            res[f"data_seed_{ds}"] = out
            OUT.write_text(json.dumps(res, indent=1, default=float))
            print("data seed", ds, {v: out[v]["runs"][0]["all_records"]["f1"] for v in VARIANTS}, flush=True)
        return
    if "--rebuild" in sys.argv or not (DATA / "stats.json").exists():
        print("built", build())
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["data"] = json.loads((DATA / "stats.json").read_text())
    res |= run_variants(SEEDS)
    OUT.write_text(json.dumps(res, indent=1, default=float))


def run_variants(label_seeds):
    t = truth()
    pid = dict(zip(t.key, t.person_id))
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = json.loads(CONFIG.read_text()) | {"country": "AU"}
        for s in base["sources"]:
            s["path"] = str(DATA / f"source_{s['name']}.parquet")
            s["columns"]["updated_at"] = "updated_at"
        for name, spec in VARIANTS.items():
            runs = []
            cp = tmp / f"{name}.json"
            cp.write_text(json.dumps(base | spec))
            for seed in label_seeds:
                lp = tmp / f"{name}_{seed}.csv"
                labels.sample(cp, 300, lp, seed=seed, stratified=False)
                lab = fill(lp, pid)
                while len(lab) < 1000:
                    nxt = tmp / "next.csv"
                    labels.sample_active(cp, lp, 100, nxt, seed=seed)
                    lab = pd.concat([lab, fill(nxt, pid)], ignore_index=True)
                    lab.to_csv(lp, index=False)
                out = tmp / f"{name}_{seed}"
                pipeline.run(cp, lp, out, seed=seed, log=lambda *_: None)
                runs.append(score(out, t))
            res[name] = {"runs": runs}
    return res


if __name__ == "__main__":
    main()
