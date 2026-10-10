"""A Fellegi-Sunter match weight as one more feature for the forest - no labels, no Splink.

Each field of a pair is put in a comparison level (missing, disagree, close, ..., exact). Two
probabilities per level: u, how often two random records land there (counted on random record
pairs), and m, how often two records of one person do (fitted by expectation maximisation over
all candidate pairs, without labels). A pair's weight is the sum over fields of log2(m / u).
An exact agreement on a value is weighed by that value's own frequency (term frequency): two
records sharing "zebedee" are stronger evidence than two sharing "smith", for names, contacts,
postcodes or any exact extra field.

The forest gets the weight as one more input and learns from your labels how far to trust it:
the weight carries what 500,000 unlabelled candidate pairs say about each field, where the
forest alone sees only your 1,000 labels. Opt-in: "match_weight_feature": true (it costs life-event
cases such as a move or a marriage; docs/guide.md). The
weights are a function of the fitted m and u and the records' value counts, so any pair can be
scored later - by the in-cluster check and by `custmatch add`.
"""
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from .core import parse_dates
from .schema import extra_fields

RANDOM_PAIRS = 1_000_000
EM_PAIRS = 400_000
EM_ROUNDS = 25
# floor on every fitted m: a field disagreement costs at most log2(M_FLOOR / u) - about 5.6 bits.
# Without it EM learned from files where nobody moves that a true match "never" disagrees on
# address or phone (m ~ 0.0005, -11 to -20 bits), and a customer who moved, married or changed
# phone was outvoted by one field (docs/report/05-customer-data.md "Scenarios: a scorecard of hard cases").
M_FLOOR = 0.02
# Fields a person legitimately changes - they move, change phone or email, marry - count only
# when they agree: a different address is no evidence of a different person (Fellegi-Sunter
# practice for time-varying fields; MDM systems treat them as many-valued). Birth date, first
# name and other extra fields keep their full disagreement weight.
CHANGEABLE = {"last_name", "addr_number", "addr_street", "email", "phone",
              "postcode", "city", "suburb", "state", "town", "occupation"}


def _fields(df, cfg):
    """(column, kind, term frequency?) for every comparison the records support."""
    out = []
    for c in ("first_name", "last_name"):
        out.append((c, "name", True))
    out += [("date_of_birth", "date", False), ("middle_initial", "exact", False),
            ("addr_number", "exact", False), ("addr_street", "street", False),
            ("email", "exact", True), ("phone", "exact", True), ("ssn", "exact", True)]
    done = {c for c, _, _ in out}
    for name, compare in extra_fields(cfg):
        if name in done or compare not in ("exact", "similarity", "edit"):
            continue
        out.append((name, {"exact": "exact", "similarity": "similar", "edit": "edit"}[compare], True))
    return [(c, k, tf) for c, k, tf in out if c in df.columns and (df[c] != "").sum() > 1]


def _values(df, profiles):
    """Field values with junk contacts blanked (they are not evidence) and dates normalised."""
    v = {c: df[c].fillna("").astype(str).to_numpy() for c in df.columns}
    for f in ("email", "phone"):
        if f in v:
            junk = np.array([profiles[f].get(x, ("personal",))[0] == "junk" for x in v[f]])
            v[f] = np.where(junk, "", v[f])
    ymd = parse_dates(v.get("date_of_birth", np.full(len(df), "")))
    v["_date"] = np.array([f"{y:04d}{m:02d}{d:02d}" if y >= 0 else "" for y, m, d in ymd])
    v["_year"] = np.array([y for y, _, _ in ymd])
    if "first_roots" in df.columns:
        v["_roots"] = df.first_roots.fillna("").to_numpy()
    return v


def levels(v, field, kind, i, j):
    """Comparison level per pair: -1 missing, 0 disagree, higher = closer (top = exact)."""
    col = "_date" if kind == "date" else field
    a, b = v[col][i], v[col][j]
    miss = (a == "") | (b == "")
    exact = (a == b) & ~miss
    lev = np.zeros(len(i), np.int8)
    if kind == "name":                      # 3 exact, 2 nickname or JW >= 0.92, 1 JW >= 0.8
        jw = cpdist(list(a), list(b), scorer=JaroWinkler.normalized_similarity, workers=-1) if len(i) else np.zeros(0)
        lev = np.where(jw >= 0.8, 1, lev)
        lev = np.where(jw >= 0.92, 2, lev)
        if field == "first_name" and "_roots" in v:
            ra, rb = v["_roots"][i], v["_roots"][j]
            share = np.array([bool(set(x.split("|")) & set(y.split("|"))) if x and y else False
                              for x, y in zip(ra, rb)])
            lev = np.where(share, np.maximum(lev, 2), lev)
        lev = np.where(exact, 3, lev)
        # a bare initial ("l") is not a rare name: agreeing initials are no evidence either way
        # (missing), a different initial is a disagreement
        initial = (np.char.str_len(a.astype(str)) == 1) | (np.char.str_len(b.astype(str)) == 1)
        same_initial = np.array([x[:1] == y[:1] for x, y in zip(a, b)], dtype=bool) if len(i) else np.zeros(0, bool)
        lev = np.where(initial & ~miss, np.where(same_initial, -1, 0), lev)
    elif kind == "date":                    # 3 exact, 2 one edit, 1 same year (or a year apart)
        ed = cpdist(list(a), list(b), scorer=Levenshtein.distance, workers=-1) if len(i) else np.zeros(0)
        lev = np.where(np.abs(v["_year"][i] - v["_year"][j]) <= 1, 1, lev)
        lev = np.where(ed <= 1, 2, lev)
        lev = np.where(exact, 3, lev)
    elif kind in ("street", "similar"):     # 3 exact, 2 JW >= 0.9, 1 JW >= 0.7
        jw = cpdist(list(a), list(b), scorer=JaroWinkler.normalized_similarity, workers=-1) if len(i) else np.zeros(0)
        lev = np.where(jw >= 0.7, 1, lev)
        lev = np.where(jw >= 0.9, 2, lev)
        lev = np.where(exact, 3, lev)
    elif kind == "edit":                    # 3 exact, 2 one edit, 1 two edits
        ed = cpdist(list(a), list(b), scorer=Levenshtein.distance, workers=-1) if len(i) else np.zeros(0)
        lev = np.where(ed <= 2, 1, lev)
        lev = np.where(ed <= 1, 2, lev)
        lev = np.where(exact, 3, lev)
    else:                                   # exact only: 1 agree, 0 disagree
        lev = exact.astype(np.int8)
    return np.where(miss, -1, lev).astype(np.int8)


def _top(kind):
    return 1 if kind == "exact" else 3


class Scorer:
    """Match weights for any (i, j) pairs of these records, from fitted m and u."""

    def __init__(self, df, profiles, model):
        self.model = model
        self.model_json = model
        self.v = _values(df, profiles)
        self.freq = {}
        for f in model["fields"]:
            if f["tf"]:                     # P(a random record with this field set has value x)
                col = self.v[f["column"]]
                s = pd.Series(col[col != ""])
                vc = s.value_counts()
                self.freq[f["column"]] = (dict(vc / max(len(s), 1)))

    def field_weights(self, pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        out = []
        for f in self.model["fields"]:
            lev = levels(self.v, f["column"], f["kind"], i, j)
            m, u = np.array(f["m"]), np.array(f["u"])
            w = np.log2(m / u)[np.maximum(lev, 0)]
            if f["tf"]:
                top = lev == _top(f["kind"])
                if top.any():
                    fr = self.freq[f["column"]]
                    tf_u = np.array([fr.get(x, 1e-6) for x in self.v[f["column"]][i[top]]])
                    w = w.astype(float)
                    w[top] = np.log2(m[-1] / np.maximum(tf_u, 1e-6))
            if f["column"] in CHANGEABLE:
                w = np.maximum(w, 0.0)
            out.append(np.where(lev < 0, 0.0, w))
        return np.column_stack(out) if out else np.zeros((len(pairs), 0))

    def weights(self, pairs):
        if not len(pairs):
            return np.zeros(0, np.float32)
        return self.field_weights(pairs).sum(axis=1).astype(np.float32)


def fit(df, profiles, pairs, cfg, seed=0):
    """Fit u on random record pairs and m by EM on the candidate pairs; returns the model."""
    rng = np.random.default_rng(seed)
    n = len(df)
    v = _values(df, profiles)
    fields = _fields(df, cfg)
    k = min(RANDOM_PAIRS, n * (n - 1) // 2)
    ri, rj = rng.integers(0, n, k), rng.integers(0, n, k)
    keep = ri != rj
    ri, rj = ri[keep], rj[keep]
    sub = pairs if len(pairs) <= EM_PAIRS else pairs[rng.choice(len(pairs), EM_PAIRS, replace=False)]
    ci, cj = sub[:, 0], sub[:, 1]
    model, L = [], []
    for col, kind, tf in fields:
        nlev = _top(kind) + 1
        rl = levels(v, col, kind, ri, rj)
        ok = rl >= 0
        u = (np.bincount(rl[ok], minlength=nlev) + 1.0) / (ok.sum() + nlev)   # smoothed
        m = np.linspace(0.05, 1.0, nlev) ** 2
        m = m / m.sum()
        model.append({"column": col, "kind": kind, "tf": tf, "u": u.tolist(), "m": m.tolist()})
        L.append(levels(v, col, kind, ci, cj))
    L = np.column_stack(L) if L else np.zeros((len(sub), 0), np.int8)
    # EM fits both classes on the candidate pairs: blocking has already made non-matching
    # candidates agree far more often than random records do, so a non-match class fixed at the
    # random-pair u would leave the match class to absorb every candidate (it did: lambda -> 1).
    # The random-pair u is used only in the final weights.
    nu = [np.array(spec["u"]) for spec in model]
    lam = 0.1
    for _ in range(EM_ROUNDS):
        logit = np.full(len(sub), np.log(lam / (1 - lam)))
        for f, spec in enumerate(model):
            lev = L[:, f]
            r = np.log(np.array(spec["m"]) / nu[f])
            logit += np.where(lev < 0, 0.0, r[np.maximum(lev, 0)])
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        lam = float(np.clip(p.mean(), 1e-4, 1 - 1e-4))
        for f, spec in enumerate(model):
            lev = L[:, f]
            ok = lev >= 0
            nlev = len(spec["m"])
            m = np.bincount(lev[ok], weights=p[ok], minlength=nlev) + 1e-3
            m = np.maximum(m / m.sum(), M_FLOOR)
            spec["m"] = (m / m.sum()).tolist()
            q = np.bincount(lev[ok], weights=1 - p[ok], minlength=nlev) + 1e-3
            nu[f] = q / q.sum()
    return {"fields": model, "lambda": lam}


def weights(df, profiles, pair_sets, cfg, seed=0, return_scorer=False):
    """The match weight of each (i, j) row of each array in pair_sets; pair_sets[0] must be the
    candidate pairs (EM is fitted on them)."""
    scorer = Scorer(df, profiles, fit(df, profiles, pair_sets[0], cfg, seed))
    out = [scorer.weights(p) for p in pair_sets]
    return (out, scorer) if return_scorer else out


def load_scorer(df, profiles, model):
    return Scorer(df, profiles, model)
