"""Couples who separate: are an ex-partner's details used to contact the one who left?

The commonest complaint about matching in marketing: a couple shared a family email and a home;
they separate; one moves out, and their new records are matched to the old household - so offers,
statements or personal details go to the ex. This builds that case on the CDP scenario data
(research/scripts/cdp_comparison.py, many sign-ups) and measures it.

Couples: the two people of every two-person household with records (most are couples), and the
two adults sharing a family email in larger ones. Half of them separate. The one who leaves has half their records made after the split:
a new address (store) and postcode (web), and - since habits outlast relationships - the family
email kept on half of those records, their own email on the rest. After the split the family
email and the old address are the ex's.

For the leaver's post-split records, on each method's customers:
  merged_with_ex       share whose customer holds a record of the ex
  email_reaches_ex     share whose contact email (the customer's commonest real one; with
                       contact safety, the record's own certain group's) is the ex's - split
                       into "own": the record itself carries the family email, and "matched":
                       the record does not, and matching brought the ex's email in
  post_to_ex_address   share whose contact address (the commonest street address) is the ex's home
and overall precision and recall, so a method cannot win by matching nothing.

Every record carries a date (updated_at): before the split 2018-2022, after it 2023-2025. A CDP
profile is last-write-wins - its email and address are the newest seen - so the "_newest"
variants choose contact details by date: the CDP's newest values, and custmatch with
"most_recent" survivorship. Without dates both choose the commonest value.

    python research/scripts/separation_research.py [--seeds 1,2,3]

Writes research/results/separation.json.
"""
import argparse
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
import cdp_reassess as R  # noqa: E402
from custmatch import labels, pipeline  # noqa: E402
from custmatch.core import components, looks_fake  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/separation.json"
SPLIT = {}            # filled by separate(): leaver person -> ex, and post-split record keys


def separate(frames, seed=0):
    rng = np.random.default_rng(seed)
    truth = pd.read_parquet(AU.DATA / "contact_truth.parquet")
    family = set(truth.value[(truth.contact_class == "household") & truth.value.str.contains("family", regex=False)])
    allrec = pd.concat([f.assign(source=s) for s, f in frames.items()], ignore_index=True)
    holders = allrec[allrec.email.isin(family)].groupby("email").person_id.unique()
    fam_of = {}
    for fam, ppl in holders.items():
        if len(ppl) == 2:
            fam_of[frozenset(ppl)] = fam
    sizes = allrec.groupby("household_id").person_id.unique()
    couples = {frozenset(p) for p in sizes if len(p) == 2} | set(fam_of)
    streets = frames["store"].address[frames["store"].address != ""].str.replace(r"^[\w/, ]*?\d+\w?\s+", "", regex=True)
    streets = streets[streets.str.len() > 5].tolist()
    postcodes = frames["web"].postcode[frames["web"].postcode != ""].tolist()
    leavers, post = {}, set()
    for pair in sorted(couples, key=lambda c: sorted(c)):
        ppl = sorted(pair)
        if rng.random() >= 0.5:
            continue
        fam = fam_of.get(pair, "")
        leaver, ex = (ppl[0], ppl[1]) if rng.random() < 0.5 else (ppl[1], ppl[0])
        leavers[int(leaver)] = {"ex": int(ex), "family_email": fam}
        own = [e for e in allrec.email[(allrec.person_id == leaver) & (allrec.email != fam)] if e]
        new_email = own[0] if own else f"leaver{leaver}@gmail.com"
        new_addr = f"{rng.integers(1, 300)} {streets[rng.integers(len(streets))]}"
        new_pc = postcodes[rng.integers(len(postcodes))]
        for s, f in frames.items():
            idx = np.flatnonzero((f.person_id == leaver).to_numpy())
            for i in idx[rng.random(len(idx)) < 0.5]:          # made after the split
                post.add(f"{s}:{f.record_id.iloc[i]}")
                if "address" in f and f.address.iloc[i]:
                    f.iloc[i, f.columns.get_loc("address")] = new_addr
                if "postcode" in f and f.postcode.iloc[i]:
                    f.iloc[i, f.columns.get_loc("postcode")] = new_pc
                if "cookie" in f and f.cookie.iloc[i]:   # made on the leaver's own new device
                    f.iloc[i, f.columns.get_loc("cookie")] = f"moved{leaver}"
                if fam and f.email.iloc[i] == fam and rng.random() >= 0.5:
                    f.iloc[i, f.columns.get_loc("email")] = new_email
    for s, f in frames.items():                       # a date on every record; post-split ones newest
        keys = s + ":" + f.record_id.astype(str)
        years = np.where(keys.isin(post), rng.uniform(2023, 2026, len(f)), rng.uniform(2018, 2023, len(f)))
        f["updated_at"] = pd.to_datetime((years - 1970) * 365.25, unit="D", origin="unix").strftime("%Y-%m-%d")
    SPLIT.update({"leavers": leavers, "post": post})
    return {"couples": len(couples), "couples_sharing_a_family_email": len(fam_of), "separated": len(leavers),
            "separated_with_family_email": sum(bool(v["family_email"]) for v in leavers.values()),
            "post_split_records": len(post)}


def outcomes(cust, df, keys, pid, contact_email=None, newest=False):
    """The three separation measures plus precision and recall, for one set of customers."""
    cust = np.asarray(cust)
    leavers = SPLIT["leavers"]
    ex_of = {l: v["ex"] for l, v in leavers.items()}
    fam_of = {l: v["family_email"] for l, v in leavers.items()}
    members = defaultdict(set)
    for c, p in zip(cust, pid):
        members[c].add(p)
    counts = df.email.value_counts()
    real = np.array([e if e and not looks_fake(e) and counts[e] <= 10 else "" for e in df.email])
    addr = (df.addr_number + " " + df.addr_street).str.strip().to_numpy()
    when = df.updated_at.to_numpy()

    def choose(values):
        """per customer: the commonest value, or (newest) the value on its newest record"""
        out = {}
        for c, v, t in zip(cust, values, when):
            if v:
                out.setdefault(c, []).append((t, v))
        if newest:
            return {c: max(vs)[1] for c, vs in out.items()}
        return {c: Counter(v for _, v in vs).most_common(1)[0][0] for c, vs in out.items()}
    golden_email, golden_addr = choose(real), choose(addr)
    # the ex's emails (the family one included) and home addresses
    ex_emails, ex_addrs = defaultdict(set), defaultdict(set)
    for k, p, e, a in zip(keys, pid, real, addr):
        for l, ex in ex_of.items():
            if p == ex:
                if e:
                    ex_emails[l].add(e)
                if a:
                    ex_addrs[l].add(a)
    for l, fam in fam_of.items():
        if fam:
            ex_emails[l].add(fam)
    merged = email_ex = email_own = post_ex = n = 0
    for i, (k, p, c) in enumerate(zip(keys, pid, cust)):
        if k not in SPLIT["post"]:
            continue
        n += 1
        merged += ex_of[p] in members[c]
        e = contact_email[i] if contact_email is not None else golden_email.get(c, "")
        if bool(e) and e in ex_emails[p]:
            if real[i] in ex_emails[p]:
                email_own += 1                     # the record itself carries the family email
            else:
                email_ex += 1                      # matching brought the ex's email in
        a = golden_addr.get(c, "")
        post_ex += bool(a) and a in ex_addrs[p]
    m = R.metrics(cust, pid, pid, real, defaultdict(set))
    return {"post_split_records": n, "merged_with_ex_pct": round(100 * merged / n, 2),
            "email_reaches_ex_matched_pct": round(100 * email_ex / n, 2),
            "email_reaches_ex_own_record_pct": round(100 * email_own / n, 2),
            "post_to_ex_address_pct": round(100 * post_ex / n, 2),
            "precision": m["precision"], "recall": m["recall"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--data", default="au_customer", help="another generated population, e.g. au_customer_s1")
    a = ap.parse_args()
    if a.data != "au_customer":
        AU.DATA = ROOT / "data" / a.data
        C.DATA = ROOT / f"data/{a.data}_cdp"
    gold, _ = AU.truth()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cp, df, keys, ev, stats = R.setup("high", tmp, transform=separate)
        cfg0 = json.loads(cp.read_text())
        for src in cfg0["sources"]:
            src["columns"]["updated_at"] = "updated_at"
        cp.write_text(json.dumps(cfg0))
        from custmatch import io as cio
        df2, _ = cio.load(cp)
        assert (cio.key(df2) == keys).all()
        df["updated_at"] = df2.updated_at.to_numpy()
        pid = np.array([gold[k][0] for k in keys])
        print("data", stats, flush=True)
        runs = []
        for seed in [int(x) for x in a.seeds.split(",")]:
            r = {"seed": seed}
            # the tuned CDP: one email and one phone per profile, values on more than 10 records blocked
            em, ph = df.email, df.phone
            common = {("email", v) for v, c in em.value_counts().items() if v and c > 10} | \
                     {("phone", v) for v, c in ph.value_counts().items() if v and c > 10}
            stream = [(i, {"email": e, "phone": p, "user_id": u, "anonymous_id": c})
                      for i, (e, p, u, c) in enumerate(zip(em, ph, df.user_id, df.cookie))]
            stream += [(None, {"anonymous_id": c, "email": e, "phone": p, "user_id": u})
                       for c, e, p, u in zip(ev.cookie, ev.email_n, ev.phone_n, ev.user_id)]
            order = np.random.default_rng(seed).permutation(len(stream))
            prof = C.segment([stream[k] for k in order], limits={"user_id": 1, "email": 1, "phone": 1},
                             blocked=lambda t, v: t in ("email", "phone") and (looks_fake(v) or (t, v) in common))
            r["cdp_tuned"] = outcomes([prof[i] for i in range(len(df))], df, keys, pid)
            r["cdp_tuned_newest"] = outcomes([prof[i] for i in range(len(df))], df, keys, pid, newest=True)
            # custmatch: 300 random + 700 active labels
            lp = tmp / f"l{seed}.csv"
            labels.sample(cp, 300, lp, seed=seed, stratified=False)
            lab = AU.fill(lp, gold)
            while len(lab) < 1000:
                labels.sample_active(cp, lp, 100, tmp / "next.csv", seed=seed)
                lab = pd.concat([lab, AU.fill(tmp / "next.csv", gold)], ignore_index=True)
                lab.to_csv(lp, index=False)
            cfg = json.loads(cp.read_text())
            newest = {"survivorship": {"default": "most_complete", "email": "most_recent", "phone": "most_recent",
                                       "address": "most_recent"}}
            for name, extra in (("custmatch", {}), ("custmatch_contact_safety", {"contact_confidence": 0.95}),
                                ("custmatch_min_precision", {"min_precision": 0.97, "contact_confidence": 0.95}),
                                ("custmatch_newest", newest | {"contact_confidence": 0.95}),
                                ("custmatch_newest_min_precision", newest | {"contact_confidence": 0.95, "min_precision": 0.97})):
                c2 = tmp / f"{name}.json"
                c2.write_text(json.dumps(cfg | extra))
                o = tmp / f"o_{name}_{seed}"
                s = pipeline.run(c2, lp, o, seed=seed, log=lambda *_: None)
                cl = pd.read_csv(o / "clusters.csv", dtype=str)
                cm = dict(zip(cl.source + ":" + cl.record_id, cl.customer_id))
                contact = None
                if extra.get("contact_confidence"):
                    ct = pd.read_csv(o / "contacts.csv", dtype=str, keep_default_na=False)
                    ce = dict(zip(ct.source + ":" + ct.record_id, ct.email))
                    contact = np.array([ce[k] for k in keys])
                r[name] = outcomes([cm[k] for k in keys], df, keys, pid, contact,
                                   newest="survivorship" in extra) | {"threshold": s["threshold"]}
                if name == "custmatch":                   # its own links, replayed at 0.8
                    links = pd.read_csv(o / "links.csv")
                    pos = {k: i for i, k in enumerate(keys)}
                    pairs = np.c_[links.a.map(pos), links.b.map(pos)]
                    cc = components(len(keys), pairs, links.score.to_numpy() > 0.8)
                    r["custmatch_threshold_0.8"] = outcomes(cc, df, keys, pid)
            runs.append(r)
            print(seed, {k: v for k, v in r.items() if k != "seed"}, flush=True)
            latest = json.loads(OUT.read_text()) if OUT.exists() else {}
            latest = latest if "runs" not in latest else {"au_customer": latest}     # older single-file layout
            OUT.write_text(json.dumps(latest | {a.data: {"data": stats, "runs": runs}}, indent=1, default=float))


if __name__ == "__main__":
    main()
