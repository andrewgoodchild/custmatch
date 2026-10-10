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
    return out if fields is None else {k: v for k, v in out.items() if k in fields}


def sample(config_path, n, out_path, seed=0):
    """Random candidate pairs, both records side by side, `label` left blank to fill 1/0."""
    df, cfg = io.load(config_path)
    pairs = candidates(df, cfg)
    pick = np.random.default_rng(seed).choice(len(pairs), min(n, len(pairs)), replace=False)
    out = _pairs_for_labelling(df, [(pairs[np.sort(pick)], "random")])
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
        _pairs_for_labelling(df, [(pairs[pick], "random")]).to_csv(out_path, index=False)
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
    _pairs_for_labelling(df, [(pairs[order], "active"), (pairs[rand], "random"), (off_rows, "active")]
                         ).to_csv(out_path, index=False)
    return len(order) + len(rand) + len(off_rows)


