"""Deterministic CDP identity resolution (Segment-style) vs custmatch, at high and low sign-up rates.

Many sites let visitors buy and browse without an account: sign-up is friction. A CDP then
knows a visitor only by cookie until they log in or type an email or phone into a form. This
compares, on the sparse Australian file (scripts/au_customer_data.py) and generated web events:

  segment_raw         Segment's documented identity resolution, reimplemented: profiles merge
                      on any shared identifier value, event by event; value limits user_id 1,
                      others 5; on a limit, the incoming event's lowest-priority identifier is
                      dropped and the match retried (priority user_id, email, then
                      alphabetical); values case-insensitive; the recommended blocked values
                      (zeros and dashes, -1, null, anonymous). Identifiers as the systems wrote
                      them ("+61 412 345 678" and "0412345678" differ).
  segment_normalised  the same, on identifiers cleaned first as Segment advises customers to do
                      (custmatch's email and phone normalisation).
  segment_tuned       well configured: cleaned identifiers, junk values blocked (test emails,
                      0400 000 000 and the like - custmatch's junk check - and every email or
                      phone held by more than 10 records, as an admin reviewing the most common
                      values would: noemail@noemail.com, na@na.com, a store's own address), the tightest
                      limits, one email and one phone per profile, so a shared device or a
                      call-centre browser cannot chain people through them.
  custmatch           user IDs as trusted IDs, the events as identity_events, and the fuzzy
                      matcher on names, birth dates, emails and phones; 300 random + 700
                      active labels.

Scenarios (per person): high sign-up - 70% have an account; low - 15%. Account holders' web and
app records carry their user_id (90%), store records sometimes (20%, a loyalty link). Web and
app records also carry the cookie of the device they were made on, as a CDP sees them: one of
the person's own (80%), the family's shared laptop (30% of records where there is one), or
none (cleared cookies). Both methods get it: Segment as the record's anonymous_id, custmatch
as a record field that a stitched cookie reaches. Every
person browses on one or two devices; account holders log in on most sessions, everyone fills
forms (checkout, newsletter) on some. Shared family laptops and call-centre agents as in
scripts/cookie_research.py. Scored as clustered F1 against the true people, on all records and
on records with no email, phone or user_id (the ones only a name links).

    python scripts/cdp_comparison.py [--scenarios high,low] [--seeds 1,2,3]

Writes results/cdp_comparison.json. Data goes to the gitignored data/au_customer_cdp/.
"""
import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import au_validate as AU  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.core import partition_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/au_customer_cdp"
OUT = ROOT / "results/cdp_comparison.json"
SCENARIOS = {"high": 0.70, "low": 0.15}
LIMITS = {"user_id": 1}                                  # every other identifier: 5
PRIORITY = ["user_id", "email", "anonymous_id", "phone"]   # user_id, email, then alphabetical
BLOCKED = re.compile(r"^[0-]*$")


def build(rate, seed=0):
    """Records with user_ids, and identified web events, for one sign-up rate."""
    rng = np.random.default_rng(seed)
    frames = {s: pd.read_parquet(AU.DATA / f"source_{s}.parquet") for s in AU.SOURCES}
    people = pd.concat([f.person_id for f in frames.values()]).unique()
    account = {p: rng.random() < rate for p in people}
    for s, f in frames.items():
        p_link = 0.9 if s in ("web", "app") else 0.2
        f["user_id"] = [f"u{p}" if account[p] and rng.random() < p_link else "" for p in f.person_id]
    ids = defaultdict(list)
    for f in frames.values():
        for p, e, m in zip(f.person_id, f.email, f.mobile):
            ids[p] += [("email", e)] * bool(e) + [("phone", m)] * bool(m)
    rows, n = [], 0
    devices, shared = defaultdict(list), {}
    for p in people:
        for _ in range(rng.integers(1, 3)):                  # devices
            n += 1
            devices[p].append(f"ck{n}")
            for _ in range(rng.integers(1, 5)):              # sessions
                if account[p] and rng.random() < 0.7:
                    rows.append({"cookie": f"ck{n}", "user_id": f"u{p}", "email": "", "phone": ""})
                if ids[p] and rng.random() < 0.25:            # a checkout or newsletter form
                    kind, v = ids[p][rng.integers(len(ids[p]))]
                    rows.append({"cookie": f"ck{n}", "user_id": "", "email": v if kind == "email" else "",
                                 "phone": v if kind == "phone" else ""})
    hh = pd.concat([f[["person_id", "household_id"]] for f in frames.values()]).drop_duplicates()
    for _, g in hh.groupby("household_id"):                   # shared family laptops
        ppl = [p for p in g.person_id if ids[p] or account[p]]
        if len(ppl) >= 2 and rng.random() < 0.15:
            n += 1
            for p in g.person_id:
                shared[p] = f"ck{n}"
            for p in rng.choice(ppl, 2, replace=False):
                if account[p]:
                    rows.append({"cookie": f"ck{n}", "user_id": f"u{p}", "email": "", "phone": ""})
                else:
                    kind, v = ids[p][rng.integers(len(ids[p]))]
                    rows.append({"cookie": f"ck{n}", "user_id": "", "email": v if kind == "email" else "",
                                 "phone": v if kind == "phone" else ""})
    contact = [p for p in people if ids[p]]
    for k in range(110):                                     # call-centre agents
        for x, i in enumerate(rng.choice(len(contact), 40 if k < 10 else 2, replace=False)):
            p = contact[i]
            kinds = [v for v in ids[p] if v[0] == ("email" if x % 2 == 0 else "phone")] or ids[p]
            kind, v = kinds[rng.integers(len(kinds))]
            rows.append({"cookie": f"agent{k}", "user_id": "", "email": v if kind == "email" else "",
                         "phone": v if kind == "phone" else ""})
    for s, f in frames.items():                               # the device each record was made on
        def made_on(p):
            if s == "store":
                return ""
            if p in shared and rng.random() < 0.3:
                return shared[p]
            return devices[p][rng.integers(len(devices[p]))] if rng.random() < 0.8 else ""
        f["cookie"] = [made_on(p) for p in f.person_id]
    DATA.mkdir(exist_ok=True)
    tag = f"{int(rate * 100)}"
    for s, f in frames.items():
        f.to_parquet(DATA / f"source_{s}_{tag}.parquet", index=False)
    ev = pd.DataFrame(rows)
    ev.to_parquet(DATA / f"events_{tag}.parquet", index=False)
    return tag, {"accounts_pct": round(100 * float(np.mean(list(account.values()))), 1), "events": int(len(ev)),
                 "cookies": int(ev.cookie.nunique()),
                 "records_with_cookie_pct": round(100 * float(np.mean(np.concatenate(
                     [(f.cookie != "").to_numpy() for f in frames.values()]))), 1),
                 "records_with_user_id_pct": round(100 * float(np.mean(np.concatenate(
                     [(f.user_id != "").to_numpy() for f in frames.values()]))), 1)}


# ------------------------------------------------------------------ Segment-style resolution
def segment(events, limits=None, blocked=None):
    """events: list of (record index or None, {type: value}) in arrival order. Returns a profile
    id per record index."""
    owner = {}                                  # (type, value) -> profile
    prof_ids = {}                               # profile -> {type: set(values)}
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    limits = limits or LIMITS
    record_profile, nxt = {}, 0
    for rec, ident in events:
        ident = {t: v.strip().lower() for t, v in ident.items() if v and v.strip()}
        ident = {t: v for t, v in ident.items() if not (BLOCKED.match(v) or v in ("-1", "null", "anonymous")
                                                        or (blocked is not None and blocked(t, v)))}
        while True:
            matched = {find(owner[(t, v)]) for t, v in ident.items() if (t, v) in owner}
            merged = defaultdict(set)
            for m in matched:
                for t, vs in prof_ids[m].items():
                    merged[t] |= vs
            for t, v in ident.items():
                merged[t].add(v)
            over = [t for t, vs in merged.items() if len(vs) > limits.get(t, 5)]
            if not over:
                break
            # demote the incoming event's lowest-priority identifier and search again
            drop = max(ident, key=lambda t: PRIORITY.index(t) if t in PRIORITY else len(PRIORITY))
            ident.pop(drop)
            if not ident:
                matched, merged = set(), defaultdict(set)
                break
        if matched:
            root = min(matched)
            for m in matched:
                parent[m] = root
            prof_ids[root] = merged
        else:
            root = nxt
            nxt += 1
            parent[root] = root
            prof_ids[root] = defaultdict(set, {t: {v} for t, v in ident.items()})
        for t, v in ident.items():
            owner[(t, v)] = root
        if rec is not None:
            record_profile[rec] = root
    return {r: find(p) for r, p in record_profile.items()}


def segment_run(df, ev, order_seed, normalised, tuned=False):
    """df: canonical records (io.load), with raw identifiers alongside; ev: identified events."""
    if normalised:
        em, ph = df.email, df.phone
        ev_em, ev_ph = ev.email_n, ev.phone_n
    else:
        em, ph = df.email_raw, df.phone_raw
        ev_em, ev_ph = ev.email, ev.phone
    stream = [(i, {"email": e, "phone": p, "user_id": u, "anonymous_id": c})
              for i, (e, p, u, c) in enumerate(zip(em, ph, df.user_id, df.cookie))]
    stream += [(None, {"anonymous_id": c, "email": e, "phone": p, "user_id": u})
               for c, e, p, u in zip(ev.cookie, ev_em, ev_ph, ev.user_id)]
    order = np.random.default_rng(order_seed).permutation(len(stream))
    if tuned:
        from custmatch.core import looks_fake
        common = {("email", v) for v, c in em.value_counts().items() if v and c > 10} | \
                 {("phone", v) for v, c in ph.value_counts().items() if v and c > 10}
        prof = segment([stream[k] for k in order], limits={"user_id": 1, "email": 1, "phone": 1},
                       blocked=lambda t, v: t in ("email", "phone") and (looks_fake(v) or (t, v) in common))
    else:
        prof = segment([stream[k] for k in order])
    return np.array([prof[i] for i in range(len(df))])


def score(cust, pid, sparse):
    return {"all_records": partition_scores(cust, pid, AU.n_true(pid)),
            "records_without_email_phone_or_user_id": partition_scores(cust[sparse], pid[sparse], AU.n_true(pid[sparse]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="high,low")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--segment-only", action="store_true", help="rescore Segment; keep custmatch's runs")
    a = ap.parse_args()
    gold, _ = AU.truth()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    base = json.loads(AU.CONFIG.read_text()) | {"country": "AU"}
    for name in a.scenarios.split(","):
        tag, stats = build(SCENARIOS[name])
        print(name, stats, flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = json.loads(json.dumps(base))
            for s in cfg["sources"]:
                s["path"] = str(DATA / f"source_{s['name']}_{tag}.parquet")
                s["columns"]["user_id"] = "user_id"
                s["columns"]["cookie"] = "cookie"
            cfg["extra_fields"] = cfg.get("extra_fields", []) + [{"name": "user_id", "compare": "exact"},
                                                                 {"name": "cookie", "compare": "exact"}]
            # the CDP sees each record's cookie as an event of its own; so does custmatch
            ev0 = pd.read_parquet(DATA / f"events_{tag}.parquet")
            rec_ck = pd.concat([pd.read_parquet(DATA / f"source_{s}_{tag}.parquet").cookie for s in AU.SOURCES])
            rec_ck = rec_ck[rec_ck != ""].drop_duplicates()
            pd.concat([ev0, pd.DataFrame({"cookie": rec_ck, "user_id": "", "email": "", "phone": ""})],
                      ignore_index=True).to_parquet(tmp / "events_all.parquet", index=False)
            cfg["trusted_ids"] = ["user_id"]
            cfg["identity_events"] = [{"path": str(tmp / "events_all.parquet"), "cookie": "cookie",
                                       "identifiers": {"email": "email", "phone": "phone", "user_id": "user_id",
                                                       "cookie": "cookie"}}]
            cp = tmp / "c.json"
            cp.write_text(json.dumps(cfg))
            df, cfg2 = io.load(cp)
            raw = pd.concat([pd.read_parquet(DATA / f"source_{s}_{tag}.parquet").assign(source=s) for s in AU.SOURCES])
            rk = dict(zip(raw.source + ":" + raw.record_id.astype(str), zip(raw.email, raw.mobile)))
            keys = io.key(df)
            df["email_raw"] = [rk[k][0] for k in keys]
            df["phone_raw"] = [rk[k][1] for k in keys]
            pid = np.array([gold[k][0] for k in keys])
            sparse = ((df.email == "") & (df.phone == "") & (df.user_id == "")).to_numpy()
            ev = pd.read_parquet(DATA / f"events_{tag}.parquet")
            from custmatch import identity
            ev["email_n"] = identity._normalise(ev.email, "email", cfg2)
            ev["phone_n"] = identity._normalise(ev.phone, "phone", cfg2)
            runs = []
            for seed in [int(x) for x in a.seeds.split(",")]:
                r = {"seed": seed}
                for norm in (False, True):
                    r["segment_normalised" if norm else "segment_raw"] = score(segment_run(df, ev, seed, norm), pid, sparse)
                r["segment_tuned"] = score(segment_run(df, ev, seed, True, tuned=True), pid, sparse)
                old = {x["seed"]: x for x in res.get(name, {}).get("runs", [])}
                if a.segment_only and seed in old:
                    r["custmatch"] = old[seed]["custmatch"]
                    runs.append(r)
                    print(name, seed, {m: r[m]["all_records"]["f1"] for m in r if m != "seed"}, flush=True)
                    continue
                lp = tmp / f"l{seed}.csv"
                labels.sample(cp, 300, lp, seed=seed)
                lab = AU.fill(lp, gold)
                while len(lab) < 1000:
                    labels.sample_active(cp, lp, 100, tmp / "next.csv", seed=seed)
                    lab = pd.concat([lab, AU.fill(tmp / "next.csv", gold)], ignore_index=True)
                    lab.to_csv(lp, index=False)
                pipeline.run(cp, lp, tmp / f"o{seed}", seed=seed, log=lambda *_: None)
                c = pd.read_csv(tmp / f"o{seed}" / "clusters.csv", dtype=str)
                cm = dict(zip(c.source + ":" + c.record_id, c.customer_id))
                r["custmatch"] = score(np.array([cm[k] for k in keys]), pid, sparse)
                runs.append(r)
                print(name, seed, {m: (r[m]["all_records"]["precision"], r[m]["all_records"]["recall"], r[m]["all_records"]["f1"])
                                   for m in ("segment_raw", "segment_normalised", "segment_tuned", "custmatch")},
                      flush=True)
            res[name] = {"data": stats, "sparse_records_pct": round(100 * float(sparse.mean()), 1), "runs": runs}
            latest = json.loads(OUT.read_text()) if OUT.exists() else {}
            OUT.write_text(json.dumps(latest | {name: res[name]}, indent=1, default=float))


if __name__ == "__main__":
    main()
