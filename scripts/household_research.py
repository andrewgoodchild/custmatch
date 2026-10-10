"""Householding and shared-contact ideas from the research, each tested against ground truth.

Research summary (docs/guide.md): the Census Bureau treats a household as a standardised address
unit; industry rules (patents US6985926, US11860909B2) also merge on a shared phone or email
plus a surname across addresses and cap groups at 7; the literature's gains come from
standardised and fuzzy addresses and group moves. Two weaknesses in custmatch itself:
household-contact detection counts distinct first names across RECORDS (one person's typo'd
records look like two people), and households key on the exact raw address string.

On the synthetic customer file (household_id and each contact value's class are known):

  contacts   C0  current profiling (records)
             C1  profiling over identity clusters: distinct customers, not distinct first names
  households H0  current: modal exact address
             H1  canonical address (suffixes, directions, unit designators, whitespace)
             H2  H1 + fuzzy street within the same house number (Jaro-Winkler >= 0.9)
             H3  H2 + a missing unit matches the only unit seen at that address
             H4  H3 + merge address households sharing a household-class contact AND a surname,
                 never above 7 customers (US Bank's cap), never through junk

Each is run with custmatch's predicted customers (1,000 random labels) and with the true
identities, so matching errors and householding errors can be told apart.
Writes results/household_research.json.
"""
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custmatch_validate as V  # noqa: E402
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.core import partition_scores, profile_contacts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = {"avenue": "ave", "av": "ave", "street": "st", "str": "st", "drive": "dr", "drv": "dr",
          "road": "rd", "court": "ct", "circle": "cir", "lane": "ln", "place": "pl",
          "boulevard": "blvd", "parkway": "pkwy", "highway": "hwy", "terrace": "ter", "trail": "trl",
          "way": "wy", "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w",
          "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw"}
UNIT_WORDS = {"apt", "apartment", "unit", "ste", "suite", "no", "flt", "flat", "#", "rm", "room", "bldg"}
CAP = 7


def canon_street(s):
    return " ".join(SUFFIX.get(t, t) for t in re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def canon_unit(u):
    toks = [t for t in re.sub(r"[^\w\s]", " ", str(u).lower()).split() if t not in UNIT_WORDS]
    return "".join(toks).lstrip("0")


def canon_number(x):
    return str(x).strip().lstrip("0")


def n2(sizes):
    sizes = np.asarray(sizes, dtype=np.float64)
    return float(np.sum(sizes * (sizes - 1) / 2))


# ------------------------------------------------------------------ contacts
def profile_by_cluster(df, field, identity):
    """C1: the current rules, but counting distinct CUSTOMERS where they counted first names,
    and diversities per customer rather than per record."""
    from custmatch.core import (HOUSEHOLD_DIVERSITY, JUNK_DIVERSITY, JUNK_MIN_RECORDS,
                                looks_fake)
    sub = df[df[field] != ""].assign(_c=identity[(df[field] != "").to_numpy()])
    prof = {}
    for val, g in sub.groupby(field):
        n = len(g)
        customers = g._c.nunique()
        per = g.groupby("_c").agg(ln=("last_name", "first"), ad=("addr_street", "first"))
        surn = per.ln.nunique() / customers
        addr = per.ad.nunique() / customers
        if looks_fake(val) or (n >= JUNK_MIN_RECORDS and (g.last_name.nunique() / n) > JUNK_DIVERSITY
                                   and ((g.addr_number + " " + g.addr_street).nunique() / n) > JUNK_DIVERSITY):
            prof[val] = ("junk", 0.0)
        elif customers >= 2 and (surn <= HOUSEHOLD_DIVERSITY or addr <= 0.5):
            prof[val] = ("household", 0.35)
        else:
            prof[val] = ("personal", 1.0 / (1.0 + np.log1p(n - 1)))
    return prof


def contact_quality(df, profiles, junk_values):
    out = {}
    for field in ("email", "phone"):
        sub = df[df[field] != ""]
        people = sub.groupby(field).simulant_id.nunique()
        gold = pd.DataFrame({"v": people.index, "shared": (people > 1).to_numpy()})
        gold["junk"] = gold.v.isin(junk_values)
        gold["pred"] = [profiles[field].get(v, ("personal",))[0] for v in gold.v]
        for target, mask in (("junk", gold.junk), ("household", gold.shared & ~gold.junk)):
            got = gold.pred == target
            tp, fp, fn = int((mask & got).sum()), int((~mask & got).sum()), int((mask & ~got).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            out[f"{field}_{target}"] = {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
                                        "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0}
    return out


# ------------------------------------------------------------------ households
def address_keys(df, level):
    num = df.addr_number.map(canon_number).to_numpy()
    raw_st, raw_un = df.addr_street.astype(str).to_numpy(), df.addr_unit.astype(str).to_numpy()
    if level == 0:
        return np.array([f"{a}|{b}|{c}" if a else "" for a, b, c in
                         zip(df.addr_number.astype(str), raw_st, raw_un)], dtype=object)
    st = np.array([canon_street(s) for s in raw_st], dtype=object)
    un = np.array([canon_unit(u) for u in raw_un], dtype=object)
    if level >= 2:      # fuzzy street within the same house number
        by_num = defaultdict(set)
        for a, s in zip(num, st):
            if a and s:
                by_num[a].add(s)
        rep = {}
        for a, streets in by_num.items():
            streets = sorted(streets)
            parent = {s: s for s in streets}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for x in range(len(streets)):
                for y in range(x + 1, len(streets)):
                    if JaroWinkler.similarity(streets[x], streets[y]) >= 0.9:
                        parent[find(streets[y])] = find(streets[x])
            for s in streets:
                rep[(a, s)] = find(s)
        st = np.array([rep.get((a, s), s) for a, s in zip(num, st)], dtype=object)
    if level >= 3:      # a missing unit matches the only unit seen at that number + street
        units = defaultdict(set)
        for a, s, u in zip(num, st, un):
            if a and u:
                units[(a, s)].add(u)
        un = np.array([u if u else (next(iter(units[(a, s)])) if len(units[(a, s)]) == 1 else "")
                       for a, s, u in zip(num, st, un)], dtype=object)
    return np.array([f"{a}|{s}|{u}" if a else "" for a, s, u in zip(num, st, un)], dtype=object)


def households(df, identity, profiles, level, merge_contacts=False):
    key = address_keys(df, min(level, 3))
    by_c = defaultdict(list)
    for i, c in enumerate(identity):
        by_c[c].append(i)
    hh_contact = [next((f"{f}:{v}" for f, v in (("email", e), ("phone", p))
                        if v and profiles[f].get(v, ("",))[0] == "household"), "")
                  for e, p in zip(df.email.to_numpy(), df.phone.to_numpy())]
    home = {}
    for c, idx in by_c.items():
        ks = [key[i] for i in idx if key[i]]
        if ks:
            home[c] = "addr:" + Counter(ks).most_common(1)[0][0]
        else:
            cs = [hh_contact[i] for i in idx if hh_contact[i]]
            home[c] = ("hh:" + cs[0]) if cs else f"own:{c}"
    if merge_contacts:
        # union address households whose customers share a household-class contact and a surname
        parent = {h: h for h in set(home.values())}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        size = Counter(home.values())
        surname = {c: Counter(df.last_name.to_numpy()[idx]).most_common(1)[0][0] for c, idx in by_c.items()}
        groups = defaultdict(set)
        for c, idx in by_c.items():
            for i in idx:
                if hh_contact[i]:
                    groups[hh_contact[i]].add(c)
        for cs in groups.values():
            cs = sorted(cs)
            for a in range(len(cs)):
                for b in range(a + 1, len(cs)):
                    x, y = cs[a], cs[b]
                    if surname[x] != surname[y]:
                        continue
                    hx, hy = find(home[x]), find(home[y])
                    if hx != hy and size[hx] + size[hy] <= CAP:
                        parent[hy] = hx
                        size[hx] += size[hy]
        home = {c: find(h) for c, h in home.items()}
    lab = np.array([home[c] for c in identity])
    return np.unique(lab, return_inverse=True)[1]


def main():
    gold = V.truth()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lp = tmp / "l.csv"
        labels.sample(V.CONFIG, 1000, lp, seed=1)
        V.fill(lp, gold)
        pipeline.run(V.CONFIG, lp, tmp / "o", log=lambda *_: None)
        c = pd.read_csv(tmp / "o" / "clusters.csv", dtype=str)
    df, _ = io.load(V.CONFIG)
    keys = io.key(df)
    pred = dict(zip(c.source + ":" + c.record_id, c.customer_id.astype(int)))
    df["simulant_id"] = [gold[k][0] for k in keys]
    df["household_id"] = [gold[k][1] for k in keys]
    identities = {"predicted": np.array([pred[k] for k in keys]),
                  "true": pd.factorize(df.simulant_id)[0]}
    truth = pd.read_parquet(ROOT / "data/customer/contact_truth.parquet")
    junk_values = set(truth.loc[truth.contact_class == "junk", "email"]) | \
        set(truth.loc[truth.contact_class == "junk", "phone"])
    hh_true = df.household_id.to_numpy()
    n_true = n2(np.unique(hh_true, return_counts=True)[1])

    for id_name, identity in identities.items():
        profiles = {"C0": {f: profile_contacts(df, f) for f in ("email", "phone")},
                    "C1": {f: profile_by_cluster(df, f, identity) for f in ("email", "phone")}}
        res = {"contacts": {k: contact_quality(df, p, junk_values) for k, p in profiles.items()}}
        for pname, prof in profiles.items():
            for level, merge, tag in ((0, False, "H0"), (1, False, "H1"), (2, False, "H2"),
                                      (3, False, "H3"), (3, True, "H4")):
                s = partition_scores(households(df, identity, prof, level, merge), hh_true, n_true)
                res[f"{tag}_{pname}"] = s
                print(f"{id_name:9s} {tag} {pname}: household F1 {s['f1']:6.2f} "
                      f"P {s['precision']:6.2f} R {s['recall']:6.2f} largest {s['largest_cluster']}", flush=True)
        for pname, q in res["contacts"].items():
            print(f"{id_name:9s} contacts {pname}: " + ", ".join(
                f"{k} P {v['precision']} R {v['recall']}" for k, v in q.items()), flush=True)
        out[id_name] = res
    (ROOT / "results/household_research.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
