"""Training labels for the forest: candidate pairs drawn for a person to label, at random or by
active learning."""
import json

import numpy as np
import pandas as pd

from . import io, schema
from .core import profile_contacts

SHOWN = ["first_name", "middle_initial", "last_name", "date_of_birth", "ssn", "email", "phone"]


def candidates(df, cfg=None):
    """Candidate pairs before any labels exist: learned blocking falls back to fixed rules."""
    from .blocking import candidates_for
    profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
    return candidates_for(df, profiles, cfg or {})[0]


def record(r, fields=None):
    """What a labeller sees (fields=None: every field): the non-empty fields, address joined."""
    out = {k: r[k] for k in SHOWN if r[k]}
    addr = " ".join(x for x in (r["addr_number"], r["addr_street"], r["addr_unit"]) if x)
    if addr:
        out["address"] = addr
    for c in r.keys():                 # alias values ("aliases"): "last_name@maiden_name"
        if "@" in c and r[c]:
            out[c.replace("@", " (") + ")"] = r[c]
    return out if fields is None else {k: v for k, v in out.items() if k in fields}


AGREE_FIELDS = ["first_name", "last_name", "date_of_birth", "email", "phone", "addr_number", "addr_street"]
AGREE_CHUNK = 5_000_000
# random labels are drawn stratified from this many candidate pairs up: on NCVR from 200,000
# people (8.9 million pairs) a plain draw held a handful of matches, and at a million people
# with 300 random labels stratifying lifted 80.84 to 94.85. The five small benchmarks (under a
# million pairs) gained too with 1,000 random labels, on one draw each, and were level with
# active ones; they draw plainly until more draws confirm it (docs/report/06-blocking-and-scale.md)
STRATIFY_FROM = 2_000_000


def agreement(df, pairs, cfg=None):
    """Which fields each candidate pair agrees on exactly (both filled), as a bit pattern: the
    strata of a stratified draw. Not a count of agreeing fields - on NCVR at a million people
    pairs agreeing on three of four fields were 0.01% matches (relatives: surname, town and
    postcode) and pairs agreeing on two were 1.25% (movers)."""
    cols = [c for c in AGREE_FIELDS + [f for f, _ in schema.extra_fields(cfg or {})] if c in df.columns]
    out = np.zeros(len(pairs), np.int32)
    for bit, c in enumerate(cols):
        v = df[c].astype(str).to_numpy()
        codes = pd.factorize(v)[0].astype(np.int32)
        codes[v == ""] = -1
        for s in range(0, len(pairs), AGREE_CHUNK):
            a, b = codes[pairs[s:s + AGREE_CHUNK, 0]], codes[pairs[s:s + AGREE_CHUNK, 1]]
            out[s:s + AGREE_CHUNK] |= ((a == b) & (a >= 0)).astype(np.int32) << bit
    return np.unique(out, return_inverse=True)[1].astype(np.int32)       # patterns as 0..k-1


def stratified_pick(score, n, rng):
    """n pair indices: half uniformly at random, half from the strata (agreement patterns), each
    stratum's share the square root of its size - every kind of pair is seen, not only the
    commonest. Returns the picks and each one's weight - the candidate pairs it stands for (1 /
    its chance of being drawn) - so estimates from the labels stay unbiased while far more of
    them are matches: with millions of candidate pairs, matches are a fraction of a percent of
    them and a plain random draw holds a handful."""
    N, n_u = len(score), n - n // 2
    size = np.bincount(score)
    share = np.sqrt(size)
    m = np.minimum(size, np.floor((n // 2) * share / share.sum()).astype(np.int64))
    m[np.argmax(share)] += min(n // 2 - m.sum(), size[np.argmax(share)] - m[np.argmax(share)])
    pick = [rng.choice(N, min(n_u, N), replace=False)]
    for s in np.flatnonzero(m):
        pick.append(rng.choice(np.flatnonzero(score == s), m[s], replace=False))
    pick = np.unique(np.concatenate(pick))
    # inclusion probability of a pair in stratum s, from the uniform half and its stratum's draw
    p_in = 1 - (1 - min(n_u, N) / N) * (1 - m / np.maximum(size, 1))
    return pick, 1 / p_in[score[pick]]


def sample(config_path, n, out_path, seed=0, stratified=None):
    """Random candidate pairs, both records side by side, `label` left blank to fill 1/0.
    stratified: half of them drawn towards pairs that agree on more fields (stratified_pick),
    each with the `weight` that keeps the threshold's estimates unbiased; None (the default)
    stratifies from STRATIFY_FROM candidate pairs up."""
    df, cfg = io.load(config_path)
    pairs = candidates(df, cfg)
    rng = np.random.default_rng(seed)
    if stratified is None:
        stratified = len(pairs) >= STRATIFY_FROM
    if stratified and len(pairs) > n:
        pick, weight = stratified_pick(agreement(df, pairs, cfg), n, rng)
    else:
        pick = np.sort(rng.choice(len(pairs), min(n, len(pairs)), replace=False))
        weight = np.full(len(pick), len(pairs) / max(len(pick), 1))
    out = _pairs_for_labelling(df, [(pairs[pick], "random")])
    out["weight"] = np.round(weight, 3)
    out.to_csv(out_path, index=False)
    return len(out)


def _pairs_for_labelling(df, groups):
    """Rows to label - both records side by side - for [(pairs array, by), ...]. Only the records
    shown are turned into dicts (the whole table as dicts cost gigabytes at 5 million records)."""
    keys = io.key(df)
    need = np.unique(np.concatenate([np.asarray(p, dtype=np.int64).ravel() for p, _ in groups] + [np.zeros(0, np.int64)]))
    rows = dict(zip(need.tolist(), df.iloc[need].to_dict("records")))
    return pd.DataFrame([{"a": keys[i], "b": keys[j], "label": "", "by": by,
                          "record_a": json.dumps(record(rows[i])), "record_b": json.dumps(record(rows[j]))}
                         for p, by in groups for i, j in np.asarray(p, dtype=np.int64).reshape(-1, 2).tolist()],
                        columns=["a", "b", "label", "by", "record_a", "record_b"])


LOOSE_RULES = [("last_name:soundex",), ("first_name:soundex", "date_of_birth:year"),
               ("last_name:first3", "date_of_birth:year"), ("first_name:first3", "last_name:first3")]


def _off_block_pairs(df, profiles, candidates, limit=50000, seed=0):
    """Plausible pairs that blocking did NOT propose, from a few loose rules (dedupe's
    DisagreementLearner looks exactly here: pairs the model calls matches but blocking missed)."""
    from .blocking import Keys, generate
    loose, _ = generate(df, Keys(df, profiles), LOOSE_RULES, max_block=60)
    if not len(loose):
        return loose
    have = pd.MultiIndex.from_arrays([candidates[:, 0], candidates[:, 1]])
    off = loose[~pd.MultiIndex.from_arrays([loose[:, 0], loose[:, 1]]).isin(have)]
    if len(off) > limit:
        off = off[np.sort(np.random.default_rng(seed).choice(len(off), limit, replace=False))]
    return off


def sample_active(config_path, labels_path, n, out_path, seed=0, mix=0.0, strategy="uncertainty"):
    """The n candidate pairs the current forest is least sure about, for hand-labelling.

    Trains on the labels so far and picks unlabelled candidate pairs with P(match) nearest 0.5.
    They are marked `by` = "active": `run` trains on them but chooses its threshold only on
    randomly drawn pairs, since hard pairs alone misjudge how the clusters come out.
    mix: the share of each batch drawn at random instead (marked "random") - on dense files
    pure uncertainty sampling starved the threshold of representative pairs.
    strategy="disagreement" (dedupe): half the active pairs are the ones the forest most
    confidently calls matches among plausible pairs blocking did NOT propose; with learned
    blocking, those labels teach the blocker what it is missing."""
    from sklearn.ensemble import RandomForestClassifier
    from .core import features
    from .models import p_match
    from .pipeline import read_labels
    df, cfg = io.load(config_path)
    profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
    from .blocking import candidates_for
    pairs = candidates_for(df, profiles, cfg)[0]
    fopts = schema.feature_options_for(df, cfg)
    li, lj, ly = read_labels(labels_path, df)
    lab = np.c_[np.minimum(li, lj), np.maximum(li, lj)].astype(np.int32)
    # pairs as int64 codes, so "already labelled" is one vectorised lookup, not a tuple per pair
    n_rec = np.int64(len(df))
    codes = pairs[:, 0].astype(np.int64) * n_rec + pairs[:, 1]
    done = np.unique(lab[:, 0].astype(np.int64) * n_rec + lab[:, 1])
    fresh = ~np.isin(codes, done)
    if ly.all() or not ly.any():
        # nothing to be uncertain between yet: another random batch, marked as such
        pool = np.flatnonzero(fresh)
        pick = np.sort(np.random.default_rng(seed + len(done)).choice(pool, min(n, len(pool)), replace=False))
        out = _pairs_for_labelling(df, [(pairs[pick], "random")])
        out["weight"] = round(len(pool) / max(len(pick), 1), 3)
        out.to_csv(out_path, index=False)
        return len(pick)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1,
                                class_weight="balanced_subsample").fit(features(df, lab, profiles, **fopts), ly)
    # scored a chunk at a time: the full feature matrix of every candidate pair is never held
    from .core import FEATURE_CHUNK
    p = np.concatenate([p_match(rf, features(df, pairs[s:s + FEATURE_CHUNK], profiles, **fopts))
                        for s in range(0, len(pairs), FEATURE_CHUNK)]) if len(pairs) else np.zeros(0)
    n_random = int(round(n * mix))
    off_rows = np.zeros((0, 2), np.int64)
    if strategy == "disagreement":
        off = _off_block_pairs(df, profiles, pairs, seed=seed)
        if len(off):
            po = p_match(rf, features(df, off, profiles, **fopts))
            want = (n - n_random) // 2
            off_done = np.isin(off[:, 0].astype(np.int64) * n_rec + off[:, 1], done)
            cand = [k for k in np.argsort(-po) if po[k] > 0.5 and not off_done[k]][:want]
            off_rows = off[cand]
    by_uncertainty = np.argsort(np.abs(p - 0.5))
    order = by_uncertainty[fresh[by_uncertainty]][:n - n_random - len(off_rows)]
    taken = fresh.copy()
    taken[order] = False
    if len(off_rows):
        taken &= ~np.isin(codes, off_rows[:, 0].astype(np.int64) * n_rec + off_rows[:, 1])
    pool = np.flatnonzero(taken)
    rand = np.random.default_rng(seed + len(done)).choice(pool, min(n_random, len(pool)), replace=False) \
        if n_random and len(pool) else np.zeros(0, np.int64)
    out = _pairs_for_labelling(df, [(pairs[order], "active"), (pairs[rand], "random"), (off_rows, "active")])
    out["weight"] = np.where(out.by == "random", round(len(pool) / max(len(rand), 1), 3), np.nan)
    out.to_csv(out_path, index=False)
    return len(order) + len(rand) + len(off_rows)


