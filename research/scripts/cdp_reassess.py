"""Reassessing the CDP comparison (research/scripts/cdp_comparison.py) with the questions a marketer asks.

F1 weighs a wrong merge the same as a missed one. In marketing a wrong merge is worse: it sends
one person's offers, or details, to someone else - often an ex-partner. So, on the same data:

  precision, recall, F1 and F0.5 (precision weighted twice as heavily as recall)
  false merges split into same-household (couples, families) and strangers
  wrong-person contacts: records whose customer's chosen email (the commonest real one on its
                         records) belongs to nobody that record's person ever used
  records in mixed customers: share of records whose customer holds someone else's record

and three fairness checks on the earlier comparison:

  Segment tuned properly  a grid of value limits (1, 2, 5 emails and phones) and blocking of
                          common values (held by more than 5, 10 or 20 records, or none),
                          scored against the truth - generous, as no CDP admin has the truth
  custmatch made stricter its accepted links replayed at higher thresholds, to compare recall
                          at the same precision as Segment rather than at each one's own F1
  custmatch with worse labels  5% of labels wrong, and 300 random labels only (no active)

    python research/scripts/cdp_reassess.py [--scenarios high,low] [--seeds 1,2,3]

Writes research/results/cdp_reassess.json.
"""
import argparse
import itertools
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import au_validate as AU  # noqa: E402
import cdp_comparison as C  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.core import components, looks_fake  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/cdp_reassess.json"
STRICTER = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def metrics(cust, pid, hh, email, owners):
    cust = np.asarray(cust)
    _, c = np.unique(cust, return_inverse=True)
    pair = lambda keys: sum(n * (n - 1) // 2 for n in Counter(keys).values())  # noqa: E731
    together = pair(c)
    same_person = pair(zip(c, pid))
    same_hh = pair(zip(c, hh))
    true_pairs = pair(pid)
    p = same_person / together if together else 1.0
    r = same_person / true_pairs if true_pairs else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    f05 = 1.25 * p * r / (0.25 * p + r) if p + r else 0.0
    # mixed customers, and the email each customer would be contacted at
    people_in = defaultdict(set)
    emails_in = defaultdict(Counter)
    for k, (ci, pi, e) in enumerate(zip(c, pid, email)):
        people_in[ci].add(pi)
        if e:
            emails_in[ci][e] += 1
    mixed = float(np.mean([len(people_in[ci]) > 1 for ci in c]))
    golden = {ci: cnt.most_common(1)[0][0] for ci, cnt in emails_in.items()}
    reached = [(pi in owners[golden[ci]]) for ci, pi in zip(c, pid) if ci in golden]
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2), "f1": round(100 * f1, 2),
            "f0_5": round(100 * f05, 2),
            "false_pairs_same_household": int(same_hh - same_person), "false_pairs_strangers": int(together - same_hh),
            "records_in_mixed_customers_pct": round(100 * mixed, 2),
            "wrong_person_email_pct": round(100 * (1 - float(np.mean(reached))), 3) if reached else None,
            "largest": int(max(Counter(c).values()))}


def setup(name, tmp, transform=None):
    """The scenario's data and custmatch config, as cdp_comparison builds them. transform(frames)
    may change the records (a dict of source -> DataFrame) before they are read."""
    tag, stats = C.build(C.SCENARIOS[name])
    if transform is not None:
        frames = {s: pd.read_parquet(C.DATA / f"source_{s}_{tag}.parquet") for s in AU.SOURCES}
        stats = stats | transform(frames)
        for s, f in frames.items():
            f.to_parquet(C.DATA / f"source_{s}_{tag}.parquet", index=False)
    cfg = json.loads(AU.CONFIG.read_text()) | {"country": "AU"}
    for s in cfg["sources"]:
        s["path"] = str(C.DATA / f"source_{s['name']}_{tag}.parquet")
        s["columns"]["user_id"] = "user_id"
        s["columns"]["cookie"] = "cookie"
    cfg["extra_fields"] = cfg.get("extra_fields", []) + [{"name": "user_id", "compare": "exact"},
                                                         {"name": "cookie", "compare": "exact"}]
    cfg["trusted_ids"] = ["user_id"]
    ev0 = pd.read_parquet(C.DATA / f"events_{tag}.parquet")
    rec_ck = pd.concat([pd.read_parquet(C.DATA / f"source_{s}_{tag}.parquet").cookie for s in AU.SOURCES])
    rec_ck = rec_ck[rec_ck != ""].drop_duplicates()
    pd.concat([ev0, pd.DataFrame({"cookie": rec_ck, "user_id": "", "email": "", "phone": ""})],
              ignore_index=True).to_parquet(tmp / "events_all.parquet", index=False)
    cfg["identity_events"] = [{"path": str(tmp / "events_all.parquet"), "cookie": "cookie",
                               "identifiers": {"email": "email", "phone": "phone", "user_id": "user_id", "cookie": "cookie"}}]
    cp = tmp / "c.json"
    cp.write_text(json.dumps(cfg))
    df, cfg2 = io.load(cp)
    raw = pd.concat([pd.read_parquet(C.DATA / f"source_{s}_{tag}.parquet").assign(source=s) for s in AU.SOURCES])
    rk = dict(zip(raw.source + ":" + raw.record_id.astype(str), zip(raw.email, raw.mobile)))
    keys = io.key(df)
    df["email_raw"] = [rk[k][0] for k in keys]
    df["phone_raw"] = [rk[k][1] for k in keys]
    from custmatch import identity
    ev = pd.read_parquet(C.DATA / f"events_{tag}.parquet")
    ev["email_n"] = identity._normalise(ev.email, "email", cfg2)
    ev["phone_n"] = identity._normalise(ev.phone, "phone", cfg2)
    return cp, df, keys, ev, stats


def segment_grid(df, ev, seed, pid, hh, email, owners):
    from custmatch.core import looks_fake as junk
    em, ph = df.email, df.phone
    out = {}
    for lim, block, ck in itertools.product((1, 2, 5), (3, 5, 10, 20, None), (5, 1)):
        if ck == 1 and lim != 1:
            continue                               # the cookie limit is tried with the strictest grid only
        common = set() if block is None else (
            {("email", v) for v, c in em.value_counts().items() if v and c > block} |
            {("phone", v) for v, c in ph.value_counts().items() if v and c > block})
        stream = [(i, {"email": e, "phone": p, "user_id": u, "anonymous_id": c})
                  for i, (e, p, u, c) in enumerate(zip(em, ph, df.user_id, df.cookie))]
        stream += [(None, {"anonymous_id": c, "email": e, "phone": p, "user_id": u})
                   for c, e, p, u in zip(ev.cookie, ev.email_n, ev.phone_n, ev.user_id)]
        order = np.random.default_rng(seed).permutation(len(stream))
        prof = C.segment([stream[k] for k in order], limits={"user_id": 1, "email": lim, "phone": lim, "anonymous_id": ck},
                         blocked=lambda t, v: t in ("email", "phone") and (junk(v) or (t, v) in common))
        out[f"limit{lim}_block{block}" + ("_cookie1" if ck == 1 else "")] = metrics(
            [prof[i] for i in range(len(df))], pid, hh, email, owners)
    return out


def custmatch_runs(cp, keys, tmp, seed, gold, pid, hh, email, owners):
    out = {}
    for variant in ("clean_1000", "noisy5_1000", "random_300", "min_precision_097"):
        lp = tmp / f"l_{variant}_{seed}.csv"
        labels.sample(cp, 300, lp, seed=seed)
        lab = AU.fill(lp, gold)
        while variant != "random_300" and len(lab) < 1000:   # min_precision: the same labels as clean
            labels.sample_active(cp, lp, 100, tmp / "next.csv", seed=seed)
            lab = pd.concat([lab, AU.fill(tmp / "next.csv", gold)], ignore_index=True)
            lab.to_csv(lp, index=False)
        if variant == "noisy5_1000":                     # a labeller who gets 1 in 20 wrong
            flip = np.random.default_rng(seed).random(len(lab)) < 0.05
            lab["label"] = np.where(flip, 1 - lab.label.astype(int), lab.label.astype(int))
            lab.to_csv(lp, index=False)
        o = tmp / f"o_{variant}_{seed}"
        run_cfg = cp
        if variant == "min_precision_097":         # as a marketer would run it: the setting, not a hand-picked threshold
            run_cfg = tmp / "min_precision.json"
            run_cfg.write_text(json.dumps(json.loads(cp.read_text()) | {"min_precision": 0.97}))
        s = pipeline.run(run_cfg, lp, o, seed=seed, log=lambda *_: None)
        c = pd.read_csv(o / "clusters.csv", dtype=str)
        cm = dict(zip(c.source + ":" + c.record_id, c.customer_id))
        res = {"threshold": s["threshold"], "at_its_threshold": metrics([cm[k] for k in keys], pid, hh, email, owners)}
        if variant == "clean_1000":
            links = pd.read_csv(o / "links.csv")
            pos = {k: i for i, k in enumerate(keys)}
            pairs = np.c_[links.a.map(pos), links.b.map(pos)]
            for t in STRICTER:
                if t > s["threshold"]:
                    cc = components(len(keys), pairs, links.score.to_numpy() > t)
                    res[f"stricter_{t}"] = metrics(cc, pid, hh, email, owners)
        out[variant] = res
        print(variant, seed, res["at_its_threshold"], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="high,low")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--data", default="au_customer", help="au_customer_noshared: no family contacts")
    a = ap.parse_args()
    if a.data != "au_customer":
        AU.DATA = ROOT / "data" / a.data
        C.DATA = ROOT / f"data/{a.data}_cdp"
    suffix = "" if a.data == "au_customer" else "_" + a.data.removeprefix("au_customer_")
    gold, _ = AU.truth()
    for name in a.scenarios.split(","):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cp, df, keys, ev, stats = setup(name, tmp)
            pid = np.array([gold[k][0] for k in keys])
            hh = np.array([gold[k][1] for k in keys])
            # the email a customer would be contacted at: real ones only (no junk, no value on
            # more than 10 records - a placeholder), and who actually uses each
            counts = df.email.value_counts()
            email = np.array([e if e and not looks_fake(e) and counts[e] <= 10 else "" for e in df.email])
            owners = defaultdict(set)
            for e, p in zip(email, pid):
                if e:
                    owners[e].add(p)
            runs = []
            for seed in [int(x) for x in a.seeds.split(",")]:
                r = {"seed": seed, "segment": segment_grid(df, ev, seed, pid, hh, email, owners)}
                r["custmatch"] = custmatch_runs(cp, keys, tmp, seed, gold, pid, hh, email, owners)
                runs.append(r)
            latest = json.loads(OUT.read_text()) if OUT.exists() else {}
            OUT.write_text(json.dumps(latest | {name + suffix: {"data": stats, "runs": runs}}, indent=1, default=float))


if __name__ == "__main__":
    main()
