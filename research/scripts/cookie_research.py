"""Cookie stitching on the sparse Australian file: how much do marketing events help, and what
does the shared-device guard protect?

Generated events (data/au_customer/events.parquet): every person browses on one or two
devices (a cookie each); on each device they log in or fill a form a few times, exposing an
email or a mobile taken from one of their own records, as written there. In 15% of households
with two or more people, a shared laptop is logged into by two of them. Most events carry no
identifier and are dropped by stitching anyway, so only identified ones are written.
Call-centre agents act for customers from their own browsers: 10 heavy agents each enter the
details of 40 random customers, and 100 light agents each serve two - one customer's email,
another's mobile, the case a per-identifier limit cannot see.

Variants, each run as au_validate.py does (300 random + 700 active labels, three draws):
  no_cookies                  the file alone
  cookies                     + identity_events, the default guard (one identity per cookie)
  cookies_unprotected         + identity_events, the guard off (cookie_max_identities 99)

    python research/scripts/cookie_research.py

Writes research/results/cookies.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import au_validate as AU  # noqa: E402
from custmatch import labels, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EVENTS = AU.DATA / "events.parquet"
OUT = ROOT / "research/results/cookies.json"
SHARED = 0.15
EVENTS_SPEC = {"path": str(EVENTS), "cookie": "cookie", "identifiers": {"email": "email", "phone": "mobile"}}
VARIANTS = {"no_cookies": {},
            "cookies": {"identity_events": [EVENTS_SPEC]},
            "cookies_no_name_check": {"identity_events": [EVENTS_SPEC], "cookie_name_check": False},
            "cookies_unprotected": {"identity_events": [EVENTS_SPEC], "cookie_max_identities": 99,
                                    "cookie_name_check": False, "cookie_max_records": 10**6}}
HEAVY_AGENTS, LIGHT_AGENTS = 10, 100


def build(seed=0):
    rng = np.random.default_rng(seed)
    t = pd.concat([pd.read_parquet(AU.DATA / f"source_{s}.parquet") for s in AU.SOURCES], ignore_index=True)
    rows, n = [], 0
    ids = {p: [("email", e) for e in g.email if e] + [("mobile", m) for m in g.mobile if m]
           for p, g in t.groupby("person_id")}
    for p, seen in ids.items():
        if not seen:
            continue
        for _ in range(rng.integers(1, 3)):                     # one or two devices
            n += 1
            for _ in range(rng.integers(1, 5)):                 # logins and forms on it
                kind, v = seen[rng.integers(len(seen))]
                rows.append({"cookie": f"ck{n}", "email": v if kind == "email" else "",
                             "mobile": v if kind == "mobile" else "", "_person": p, "_shared": False})
    shared = 0
    for hh, g in t.groupby("household_id"):
        people = [p for p in g.person_id.unique() if ids.get(p)]
        if len(people) < 2 or rng.random() >= SHARED:
            continue
        n += 1
        shared += 1
        for p in rng.choice(people, 2, replace=False):
            kind, v = ids[p][rng.integers(len(ids[p]))]
            rows.append({"cookie": f"ck{n}", "email": v if kind == "email" else "",
                         "mobile": v if kind == "mobile" else "", "_person": p, "_shared": True})
    people = [p for p in ids if ids[p]]
    for k in range(HEAVY_AGENTS + LIGHT_AGENTS):
        heavy = k < HEAVY_AGENTS
        served = rng.choice(len(people), 40 if heavy else 2, replace=False)
        for x, i in enumerate(served):
            p = people[i]
            kinds = [v for v in ids[p] if v[0] == ("email" if x % 2 == 0 else "mobile")] or ids[p]
            kind, v = kinds[rng.integers(len(kinds))]
            rows.append({"cookie": f"agent{k}", "email": v if kind == "email" else "",
                         "mobile": v if kind == "mobile" else "", "_person": p, "_shared": True})
    ev = pd.DataFrame(rows)
    ev.to_parquet(EVENTS, index=False)
    return {"cookies": int(ev.cookie.nunique()), "identified_events": int(len(ev)), "shared_laptops": shared,
            "heavy_agents": HEAVY_AGENTS, "light_agents": LIGHT_AGENTS}


def main():
    stats = build()
    print("events", stats)
    gold, no_surname = AU.truth()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["events"] = stats
    base = json.loads(AU.CONFIG.read_text()) | {"country": "AU"}
    for s in base["sources"]:
        s["path"] = str((AU.CONFIG.parent / s["path"]).resolve())
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, spec in VARIANTS.items():
            cp = tmp / f"{name}.json"
            cp.write_text(json.dumps(base | spec))
            runs = []
            for seed in AU.SEEDS:
                lp = tmp / f"{name}_{seed}.csv"
                labels.sample(cp, 300, lp, seed=seed, stratified=False)
                lab = AU.fill(lp, gold)
                while len(lab) < 1000:
                    nxt = tmp / "next.csv"
                    labels.sample_active(cp, lp, 100, nxt, seed=seed)
                    lab = pd.concat([lab, AU.fill(nxt, gold)], ignore_index=True)
                    lab.to_csv(lp, index=False)
                out = tmp / f"{name}_{seed}"
                s = pipeline.run(cp, lp, out, seed=seed, log=lambda *_: None)
                sc = AU.score(out, gold, no_surname)
                runs.append({"all_records": sc["all_records"], "no_surname_records": sc["no_surname_records"],
                             "cookies": s.get("cookies", {})})
                print(name, seed, sc["all_records"], s.get("cookies", {}), flush=True)
            res[name] = {"runs": runs, "mean": {k: round(float(np.mean([r[k]["f1"] for r in runs])), 2)
                                                for k in ("all_records", "no_surname_records")},
                         "precision": round(float(np.mean([r["all_records"]["precision"] for r in runs])), 2)}
            OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
