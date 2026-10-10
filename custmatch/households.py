"""Households and shared-contact classes, after identity clustering (docs/report/05-customer-data.md "Households").

Two changes from the research (scripts/household_research.py, synthetic file, predicted
customers):

  contacts   classified over CUSTOMERS, not records. The first profiler counted distinct first
             names across records, so one person with a typo'd name in two sources looked like
             two people and their personal email became "household": precision 57.9%. Counting
             distinct customers after matching: 84.5% (100% with perfect matching), recall ~96%.
  households standardised addresses (street suffixes, directions, unit designators), streets
             matched fuzzily within the same house number (Jaro-Winkler >= 0.9), a missing unit
             matching the only unit at that address, and address households merged when their
             customers share a household-class contact and a surname (never above 7 customers,
             never through junk) - the industry rule in patents US6985926 / US11860909B2.
             Household F1 90.05 -> 91.28, recall 87.97 -> 90.14, above the 89.5% that exact
             addresses allow.

Discarding very large address groups as institutions (US Bank discards groups over 7) is not
done: on this benchmark group quarters are real households and it cut F1 to 22.5.
"""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist

from .core import (FAMILY_MAX, HOUSEHOLD_DIVERSITY, JUNK_DIVERSITY, JUNK_MIN_RECORDS, OTHER_CONTACT, PATTERN_MIN_RECORDS,
                   contact_junk, name_and_address_diversity, patterned_junk,
                   syntactic_junk_mask)

SUFFIX = {"avenue": "ave", "av": "ave", "street": "st", "str": "st", "drive": "dr", "drv": "dr",
          "road": "rd", "court": "ct", "circle": "cir", "lane": "ln", "place": "pl",
          "boulevard": "blvd", "parkway": "pkwy", "highway": "hwy", "terrace": "ter", "trail": "trl",
          "way": "wy", "square": "sq", "north": "n", "south": "s", "east": "e", "west": "w",
          "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw"}
UNIT_WORDS = {"apt", "apartment", "unit", "ste", "suite", "no", "flt", "flat", "#", "rm", "room", "bldg"}
MERGE_CAP = 7


def canon_street(s):
    return " ".join(SUFFIX.get(t, t) for t in re.sub(r"[^\w\s]", " ", str(s).lower()).split())


def canon_unit(u):
    return "".join(t for t in re.sub(r"[^\w\s]", " ", str(u).lower()).split() if t not in UNIT_WORDS).lstrip("0")


def profile_by_customer(df, field, customers):
    """{value: (class, weight)} like core.profile_contacts, counting distinct customers.
    One grouped pass (a loop per value took 56 s for 20,000 emails)."""
    has = (df[field] != "").to_numpy()
    if not has.any():
        return {}
    ad = (df.addr_number + " " + df.addr_street).str.strip()
    sub = pd.DataFrame({"v": df[field][has].to_numpy(), "c": np.asarray(customers)[has],
                        "fn": df.first_name[has].to_numpy(),
                        "ln": df.last_name[has].where(df.last_name[has] != "").to_numpy(),   # blank = unknown
                        "ad": ad[has].where(ad[has] != "").to_numpy()})
    g = sub.groupby("v")
    rec = pd.DataFrame({"n": g.size(), "fn": g.fn.nunique(), "ln": g.ln.nunique(), "ln_n": g.ln.count(),
                        "ad": g.ad.nunique(), "ad_n": g.ad.count()})
    per = sub.groupby(["v", "c"]).agg(fn=("fn", "first"), ln=("ln", "first"), ad=("ad", "first")).groupby(level=0)
    cus = pd.DataFrame({"k": per.size(), "fn": per.fn.nunique(), "ln": per.ln.nunique(), "ln_n": per.ln.count(),
                        "ad": per.ad.nunique(), "ad_n": per.ad.count()}).reindex(rec.index)
    div, adiv = name_and_address_diversity(rec.n, rec.fn, rec.ln, rec.ln_n, rec.ad, rec.ad_n)
    cdiv, cadiv = name_and_address_diversity(cus.k, cus.fn, cus.ln, cus.ln_n, cus.ad, cus.ad_n)
    n, k = rec.n.to_numpy(), cus.k.to_numpy()
    syntax = syntactic_junk_mask(rec.index)
    junk = syntax | ((n >= JUNK_MIN_RECORDS) & (div > JUNK_DIVERSITY) & ~(adiv <= JUNK_DIVERSITY))
    other = np.zeros(len(n), bool)
    if field in OTHER_CONTACT and OTHER_CONTACT[field] in df:
        args = (field, df[OTHER_CONTACT[field]][has].to_numpy(), df[field][has].to_numpy(), n,
                np.asarray(div), rec.ln_n.to_numpy(), np.asarray(adiv))
        junk |= contact_junk(*args)
        other = contact_junk(*args, min_records=PATTERN_MIN_RECORDS)
    junk |= patterned_junk(rec.index, n, np.asarray(div), np.asarray(adiv), other)
    household = ~junk & (k >= 2) & ((cdiv <= HOUSEHOLD_DIVERSITY) | (cadiv <= 0.5))
    weight = np.where(junk, 0.0, np.where(household, 0.35, 1.0 / (1.0 + np.log1p(n - 1))))
    cls = np.where(junk, "junk", np.where(household, "household", "personal"))
    return {v: (str(c), float(w)) for v, c, w in zip(rec.index, cls, weight)}


_DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw"}


def _markers(street):
    """Direction words: 12 n main st and 12 s main st are different streets. Street types are
    NOT compared: on the synthetic file the same household's street type varies through noise,
    and guarding types cost 0.9 household F1 (recall 90.15 -> 88.51) for no precision - so
    12 park rd and 12 park dr still merge."""
    return tuple(sorted(t for t in street.split() if t in _DIRECTIONS))


def address_keys(df):
    num = df.addr_number.astype(str).str.strip().str.lstrip("0").to_numpy()
    st = _map_unique(df.addr_street.astype(str), canon_street)       # each distinct value cleaned once
    un = _map_unique(df.addr_unit.astype(str), canon_unit)
    # fuzzy street within one house number: streets merge only with the same first letter and
    # direction words, so only those are compared - in groups, with rapidfuzz's matrix. Comparing
    # every pair of a number's streets grew with the square of the file (number 12's streets
    # grow with it). Pairs are merged in the same order as before, so the result is identical.
    pairs = pd.DataFrame({"a": num, "s": st})
    pairs = pairs[(pairs.a != "") & (pairs.s != "")].drop_duplicates()
    rep = {}
    for a, streets in pairs.groupby("a", sort=False).s:
        streets = sorted(streets)
        groups = defaultdict(list)                  # (first letter, direction words) -> positions
        for x, s in enumerate(streets):
            groups[(s[0], _markers(s))].append(x)
        root = list(range(len(streets)))
        for xs in groups.values():
            if len(xs) > 1:
                for x, r in zip(xs, _typo_roots([streets[x] for x in xs])):
                    root[x] = xs[r]
        for x, s in enumerate(streets):
            rep[(a, s)] = streets[root[x]]
    st = np.array([rep.get((a, s), s) for a, s in zip(num, st)], dtype=object)
    units = defaultdict(set)
    for a, s, u in zip(num, st, un):
        if a and u:
            units[(a, s)].add(u)
    un = [u if u else (next(iter(units[(a, s)])) if len(units[(a, s)]) == 1 else "")
          for a, s, u in zip(num, st, un)]
    return np.array([f"{a}|{s}|{u}" if a else "" for a, s, u in zip(num, st, un)], dtype=object)


def _typo_roots(streets):
    """For streets sorted alphabetically, the representative of each: streets at Jaro-Winkler
    0.9 or more are one street (a typo), joined transitively, and each group's representative is
    the one the original union-find chose - it merged pairs (x, y), x < y, in order, each time
    putting y's group under x's current root. Processing x in order, every group x touches joins
    x's group, which keeps its root: the same result, with numpy steps per street instead of a
    Python step per similar pair (a group of a thousand near-identical streets had half a million).
    Each street is compared with the later ones a row at a time, so memory stays O(streets)."""
    g = len(streets)
    lab = np.arange(g)                              # the group each street is in
    root = np.arange(g)                             # each group's representative
    # float64: cdist's default float32 rounds 0.8999999999999999 up to 0.9, which the original
    # (JaroWinkler.similarity, a Python float) does not reach
    full = cdist(streets, streets, scorer=JaroWinkler.similarity, dtype=np.float64, workers=-1) >= 0.9 \
        if g <= 2000 else None                     # (bounded memory: a row at a time above that)
    rows = np.flatnonzero(np.triu(full, k=1).any(axis=1)) if full is not None else range(g - 1)
    for x in rows:
        if full is not None:
            nb = np.flatnonzero(full[x, x + 1:]) + x + 1
        else:
            sims = cdist([streets[x]], streets[x + 1:], scorer=JaroWinkler.similarity, dtype=np.float64, workers=-1)[0]
            nb = np.flatnonzero(sims >= 0.9) + x + 1
        if not len(nb):
            continue
        keep = lab[x]
        comps = np.unique(lab[nb])
        if len(comps) == 1 and comps[0] == keep:
            continue                                # already one group: the original's no-op unions
        r = root[keep]
        lab[np.isin(lab, comps)] = keep
        root[keep] = r
    return root[lab].tolist()


def _map_unique(series, fn):
    """fn applied to each distinct value once, mapped back to every record."""
    codes, uniq = pd.factorize(series)
    done = np.array([fn(v) for v in uniq], dtype=object)
    return done[codes] if len(codes) else np.zeros(0, dtype=object)


def assign(df, customers, profiles, postcode_join=False):
    """Household label per record, and how many households exceed FAMILY_MAX records.
    postcode_join ("household_postcode_join"): see below - measured +1.38 household F1 but -5.01
    precision on the sparse Australian file (same surname, same postcode is often relatives
    nearby), so off by default."""
    customers = np.asarray(customers)
    key = address_keys(df)
    by_c = defaultdict(list)
    for i, c in enumerate(customers):
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
    parent = {h: h for h in set(home.values())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    size = Counter(home.values())
    ln = df.last_name.to_numpy()
    # a customer's most common NON-blank surname: two customers with no surname share nothing
    surname = {c: next((s for s, _ in Counter(ln[idx]).most_common() if s), "") for c, idx in by_c.items()}
    groups = defaultdict(set)
    for c, idx in by_c.items():
        for i in idx:
            if hh_contact[i]:
                groups[hh_contact[i]].add(c)
    for cs in groups.values():
        cs = sorted(cs)
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                if not surname[cs[a]] or surname[cs[a]] != surname[cs[b]]:
                    continue
                hx, hy = find(home[cs[a]]), find(home[cs[b]])
                if hx != hy and size[hx] + size[hy] <= MERGE_CAP:
                    parent[hy] = hx
                    size[hx] += size[hy]
    if postcode_join and "postcode" in df.columns:
        # a customer with no address joins the one address household with its surname in its
        # postcode - web sign-ups give a postcode, store addresses end in one (country.py)
        pc = df.postcode.to_numpy()
        postcode = {c: next((p for p, _ in Counter(pc[idx]).most_common() if p), "") for c, idx in by_c.items()}
        at = defaultdict(set)
        for c, h in home.items():
            if h.startswith("addr:") and surname[c] and postcode[c]:
                at[(surname[c], postcode[c])].add(find(h))
        for c, h in home.items():
            if h.startswith("addr:") or not surname[c] or not postcode[c]:
                continue
            found = {find(x) for x in at.get((surname[c], postcode[c]), ())}
            hx, hy = (next(iter(found)), find(h)) if len(found) == 1 else (None, None)
            if hx is not None and hx != hy and size[hx] + size[hy] <= MERGE_CAP:
                parent[hy] = hx
                size[hx] += size[hy]
    lab = np.array([find(home[c]) for c in customers])
    out = np.unique(lab, return_inverse=True)[1]
    return out, int(np.sum(np.bincount(out) > FAMILY_MAX))
