"""How good is this run on YOUR records? A stratified audit sample, labelled by a person, gives
precision, recall and F1 of the run's decisions with confidence intervals - where the benchmark
figures in the docs, measured on other people's data, do not transfer.

  custmatch audit sample --state results/ --n 300 --out audit.csv
      pairs to label, drawn from every score band - most candidate pairs score near 0 and
      would fill a plain random sample - in the labelling format of `custmatch sample`
  custmatch audit estimate --state results/ audit.csv
      once `label` is filled (1 same person, 0 not): precision, recall and F1, each with a 95%
      interval

Each run saves an audit pool (state/audit_pool.parquet). Its strata are the score bands
(BANDS) split into the pairs the run linked - put in one customer, after clustering, overrides
and the cannot-link rules: what the run delivered - and the pairs it did not. Whether a pair is
linked is known for every candidate pair; only whether it is a match is not. For each stratum the
pool holds a uniform random sample of up to POOL_PER_STRATUM pairs, with the stratum's size and
the sum of its pairs' scores. The audit sample is drawn from the pool, about equally from every
stratum.

The matches in a stratum are estimated by the difference estimator of survey sampling: the sum
of the run's scores over all its pairs, corrected by the labelled pairs' mean residual (label
minus score) times the stratum's size. It is unbiased however well or badly calibrated the scores
are - the labels correct them - and the better the scores, the narrower the interval. The 95%
interval resamples the residuals within each stratum. Two simpler intervals were measured and
are not used (research/scripts/audit_check.py): resampling the labels themselves, which gives a
stratum whose labelled pairs all agree no uncertainty at all, and a Beta interval per stratum,
whose long tail in big strata with no match seen dragged the intervals off the truth.

(The weighting and the bands follow jlink - keltokhy/jlink, src/jlink/audit.py.)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = (0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0)
POOL_PER_STRATUM = 5000
POOL, TOTALS = "audit_pool.parquet", "audit_strata.json"


def band_of(score):
    b = np.searchsorted(BANDS, score, side="right")
    if not np.ndim(b):
        return np.clip(b - 1, 0, len(BANDS) - 2)
    np.clip(b, 1, len(BANDS) - 1, out=b)           # in place, one byte a pair: 192M pairs at 5M records
    b -= 1
    return b.astype(np.uint8)


def stratum(band, linked):
    return f"{int(band)}{'L' if linked else 'U'}"


def save_pool(state_dir, keys, pairs, score, customer, seed=0):
    """The run's audit pool: per stratum (score band x linked or not), up to POOL_PER_STRATUM
    candidate pairs at random, and the stratum's size. keys: 'source:record_id' per record;
    pairs, score: every scored candidate pair; customer: each record's customer."""
    from .core import pair_slices
    rng = np.random.default_rng(seed)
    score = np.asarray(score)
    customer = np.asarray(customer)
    if len(customer) and customer.max() < 2 ** 31:
        customer = customer.astype(np.int32)      # half the memory of the gathers below
    # a slice of pairs at a time: all 5M NCVR records have 192M pairs, most of them in one
    # stratum, whose indices and scores as whole arrays were 3 GB
    slices = pair_slices(len(score))
    band, linked = np.empty(len(score), np.uint8), np.empty(len(score), bool)
    for lo, hi in slices:
        band[lo:hi] = band_of(score[lo:hi])
        linked[lo:hi] = customer[pairs[lo:hi, 0]] == customer[pairs[lo:hi, 1]]
    keep, totals, score_sum = [], {}, {}
    for b in range(len(BANDS) - 1):
        for li in (True, False):
            m = (band == b) & (linked == li)
            per_slice = np.array([np.count_nonzero(m[lo:hi]) for lo, hi in slices], np.int64)
            count = int(per_slice.sum())
            if not count:
                continue
            first = np.r_[0, np.cumsum(per_slice)]       # rank of each slice's first pair in the stratum
            totals[stratum(b, li)] = count
            score_sum[stratum(b, li)] = float(_masked_sum(score, m, slices, first, 0, count))
            ranks = np.arange(count) if count <= POOL_PER_STRATUM else \
                np.sort(rng.choice(count, POOL_PER_STRATUM, replace=False))   # as rng.choice of the indices
            keep.append(_at_ranks(m, slices, first, ranks))
            del m
    keep = np.concatenate(keep) if keep else np.zeros(0, np.int64)
    pd.DataFrame({"a": keys[pairs[keep, 0]], "b": keys[pairs[keep, 1]], "score": score[keep],
                  "stratum": [stratum(b, li) for b, li in zip(band[keep], linked[keep])]}
                 ).to_parquet(Path(state_dir) / POOL, index=False)
    (Path(state_dir) / TOTALS).write_text(json.dumps({"bands": list(BANDS), "pairs": totals, "score_sum": score_sum}))


SUM_LEAF = 1 << 22


def _masked(score, m, slices, first, r0, r1):
    """score[m][r0:r1], from the slices that hold those ranks."""
    k = int(np.searchsorted(first, r0, side="right")) - 1
    parts = []
    while r0 < r1:
        lo, hi = slices[k]
        a, b = r0 - first[k], min(r1, first[k + 1]) - first[k]
        parts.append(score[lo:hi][m[lo:hi]][a:b])
        r0 = first[k] + b
        k += 1
    return np.concatenate(parts) if parts else score[:0]


def _masked_sum(score, m, slices, first, r0, r1):
    """score[m][r0:r1].sum() to the bit, without the copy: numpy sums by halving (pairwise, cut
    at a multiple of 8), so the halves are summed the same way, each from its slices."""
    n = r1 - r0
    if n <= SUM_LEAF:
        return _masked(score, m, slices, first, r0, r1).sum()
    half = n // 2
    half -= half % 8
    return _masked_sum(score, m, slices, first, r0, r0 + half) + _masked_sum(score, m, slices, first, r0 + half, r1)


def _at_ranks(m, slices, first, ranks):
    """np.flatnonzero(m)[ranks] for sorted ranks, a slice at a time."""
    out = []
    for k, (lo, hi) in enumerate(slices):
        r = ranks[(ranks >= first[k]) & (ranks < first[k + 1])] - first[k]
        if len(r):
            out.append(lo + np.flatnonzero(m[lo:hi])[r])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def _state(state_dir):
    from .incremental import STATE
    from .io import ConfigError
    d = Path(state_dir) / STATE
    if not (d / POOL).exists():
        raise ConfigError(f"no audit pool in {state_dir}: `custmatch run` saves one (not `add`, and not older "
                          "versions); rerun to audit")
    return d, pd.read_parquet(d / POOL), json.loads((d / TOTALS).read_text())


def sample(state_dir, n, out_path, seed=0):
    """About n pairs from the pool, as equally as the strata allow, for labelling."""
    from .labels import record
    d, pool, meta = _state(state_dir)
    strata = list(meta["pairs"])
    want = {s: 0 for s in strata}
    room = {s: int((pool.stratum == s).sum()) for s in strata}
    left = n
    while left > 0:                                    # equal shares; a stratum that runs out passes on its spare
        open_ = [s for s in strata if want[s] < room[s]]
        if not open_:
            break
        share = max(left // len(open_), 1)
        for s in open_:
            take = min(share, room[s] - want[s], left)
            want[s] += take
            left -= take
    rng = np.random.default_rng(seed)
    picked = pd.concat([pool[pool.stratum == s].iloc[np.sort(rng.choice(room[s], want[s], replace=False))]
                        for s in strata if want[s]], ignore_index=True)
    recs = pd.read_parquet(d / "records.parquet")
    by_key = dict(zip(recs.source + ":" + recs.record_id, range(len(recs))))
    need = sorted({by_key[k] for k in pd.concat([picked.a, picked.b])})
    rows = dict(zip(need, recs.iloc[need].to_dict("records")))
    out = pd.DataFrame({"a": picked.a, "b": picked.b, "label": "", "by": "audit:" + picked.stratum,
                        "record_a": [json.dumps(record(rows[by_key[k]])) for k in picked.a],
                        "record_b": [json.dumps(record(rows[by_key[k]])) for k in picked.b]})
    out.to_csv(out_path, index=False)
    return len(out)


def estimate(state_dir, labels_path, draws=4000, seed=0):
    """Precision, recall and F1 of the run's links, with 95% intervals, from an audit sample whose
    `label` column is filled."""
    from .io import ConfigError
    from .pipeline import LABEL_NO, LABEL_YES
    _, pool, meta = _state(state_dir)
    totals, score_sum = meta["pairs"], meta["score_sum"]
    lab = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    lab = lab[lab.by.str.startswith("audit:")]
    word = lab.label.str.strip().str.lower()
    bad = ~(word.isin(LABEL_YES | LABEL_NO) | (word == ""))
    if bad.any():
        raise ConfigError(f"labels must be 1 or 0 (or blank to skip): {lab.label[bad].iloc[0]!r} in {labels_path}")
    lab = lab[word != ""].assign(match=word[word != ""].isin(LABEL_YES).to_numpy(), stratum=lambda t: t.by.str[6:])
    lab = lab.merge(pool[["a", "b", "score"]], on=["a", "b"], how="left")
    if lab.score.isna().any() or not set(lab.stratum) <= set(totals):
        raise ConfigError(f"{labels_path} was not drawn from this run's audit pool (custmatch audit sample)")
    strata = list(totals)
    size = np.array([totals[s] for s in strata], float)
    base = np.array([score_sum[s] for s in strata])
    linked = np.array([s.endswith("L") for s in strata])
    resid = [lab.match[lab.stratum == s].to_numpy(float) - lab.score[lab.stratum == s].to_numpy() for s in strata]

    def scores(mean_resid):                            # draws x strata (or strata): the difference estimator
        matches = np.clip(base + size * mean_resid, 0, size)
        tp = (matches * linked).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = tp / (size * linked).sum()
            r = tp / matches.sum(axis=-1)
            return p, r, 2 * p * r / (p + r)

    point = scores(np.array([r.mean() if len(r) else 0.0 for r in resid]))
    rng = np.random.default_rng(seed)
    boot = np.column_stack([r[rng.integers(0, len(r), (draws, len(r)))].mean(axis=1) if len(r) else np.zeros(draws)
                            for r in resid])
    reps = np.column_stack(scores(boot))
    lo, hi = np.nanpercentile(reps, 2.5, axis=0), np.nanpercentile(reps, 97.5, axis=0)
    out = {name: {"estimate": round(100 * float(point[i]), 2),
                  "interval_95": [round(100 * float(lo[i]), 2), round(100 * float(hi[i]), 2)]}
           for i, name in enumerate(("precision", "recall", "f1"))}
    out["labelled_pairs"] = int(len(lab))
    out["by_stratum"] = [{"scores": f"{BANDS[int(s[:-1])]:.2f}-{BANDS[int(s[:-1]) + 1]:.2f}",
                          "linked": s.endswith("L"), "candidate_pairs": totals[s], "labelled": int(len(r)),
                          "matches": int(lab.match[lab.stratum == s].sum())}
                         for s, r in zip(strata, resid)]
    missing = [s for s, r in zip(strata, resid) if not len(r)]
    out["warnings"] = ([f"no labelled pair from {len(missing)} stratum(s) holding {sum(totals[s] for s in missing):,} "
                        "candidate pairs: their matches are the run's scores alone, uncorrected; label pairs from "
                        "every stratum (custmatch audit sample draws from all)"] if missing else [])
    out["note"] = ("recall is among candidate pairs: a true match blocking never proposed is not counted "
                   "(custmatch blocking reports what blocking misses)")
    return out
