"""Run the measured customer-matching pipeline on your records.

  records -> contact profiling -> blocking -> features -> random forest (trained on your
  labelled pairs) -> threshold chosen AFTER clustering -> customers -> households -> golden
  records

The threshold rule is custmatch's central design choice (docs/report/02-whole-customers.md): a threshold that is
good pairwise can chain thousands of records into one customer. So the threshold is chosen by
closure-aware F1 - a held-out labelled pair counts as matched only if its two records end up in
the same cluster - cross-fitted over 4 folds so every labelled pair helps choose it (a single
25% hold-out was too noisy to separate thresholds); thresholds the random labels cannot tell
apart count as tied and the middle one is chosen (docs/guide.md "How it decides").
The final forest is then trained on all labelled pairs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import explain, identity, incremental, io, monitor, schema
from .core import components, features, profile_contacts, relay_records
from .golden import build_golden

THRESHOLDS = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05), [0.97, 0.98, 0.99]]), 3)
GOLDEN_FIELDS = ["first_name", "middle_initial", "last_name", "date_of_birth", "addr_number",
                 "addr_street", "addr_unit", "email", "phone"]
MIN_LABELS = 200
MIN_RANDOM_MATCHES = 30     # random labelled matches below which the threshold choice is noisy
# a two-field blocking key degraded to one field (blocking report "pairs_with_a_blank_part"): warn
# when those pairs are at least this share of the candidate pairs. On an address book with no
# streets, first name + street gave 660 of 1,057 pairs
BLANK_KEY_SHARE, BLANK_KEY_PAIRS = 0.25, 100
RECOMMENDED_LABELS = 500   # custmatch_validate, rich features: 500 labels 97.38, 1,000 97.32 clustered F1
FOLDS = 4


LABEL_YES = {"1", "1.0", "true", "t", "yes", "y", "match", "same"}
LABEL_NO = {"0", "0.0", "false", "f", "no", "n", "nonmatch", "non-match", "different"}

def read_labels(path, df, with_active=False, with_weight=False):
    """Labelled pairs as (row i, row j, label); labels name records as 'source:record_id'.
    with_active also returns a mask of pairs chosen by active learning (`by` == "active");
    with_weight also each pair's `weight`, the candidate pairs a randomly drawn label stands for
    (custmatch sample writes it; labels without one count equally - the median of the others')."""
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("a", "b", "label"):
        if c not in lab.columns:
            raise io.ConfigError(f"labels file needs columns a, b, label (got {list(lab.columns)})")
    lab = lab[lab.label.str.strip() != ""]
    pos = {k: i for i, k in enumerate(io.key(df))}
    unknown = sorted(set(lab.a) - set(pos) | set(lab.b) - set(pos))
    if unknown:
        raise io.ConfigError(f"{len(unknown)} labelled record(s) not in the input, e.g. {unknown[:3]}")
    v = lab.label.str.strip().str.lower()
    # spreadsheets and pandas write labels back as 1.0 / True: read them, and refuse anything else
    # rather than let an unrecognised "match" silently train the model as a non-match
    bad = sorted(set(v) - LABEL_YES - LABEL_NO)
    if bad:
        raise io.ConfigError(f"unrecognised label value(s) {bad[:5]}: use 1 / 0 (also read: "
                             f"{', '.join(sorted(LABEL_YES))} and {', '.join(sorted(LABEL_NO))})")
    y = v.isin(LABEL_YES).to_numpy()
    i, j = np.array([pos[a] for a in lab.a]), np.array([pos[b] for b in lab.b])
    if not with_active:
        return i, j, y
    active = (lab.by == "active").to_numpy() if "by" in lab.columns else np.zeros(len(y), bool)
    if not with_weight:
        return i, j, y, active
    w = pd.to_numeric(lab["weight"], errors="coerce").to_numpy(dtype=float, copy=True) if "weight" in lab.columns \
        else np.full(len(y), np.nan)
    w[active] = np.nan
    have = np.isfinite(w) & (w > 0)
    w = np.where(have, w, np.median(w[have]) if have.any() else 1.0)
    return i, j, y, active, w


def _scores(tp, fp, fn):
    return {"precision": round(100 * tp / (tp + fp), 2) if tp + fp else 0.0,
            "recall": round(100 * tp / (tp + fn), 2) if tp + fn else 0.0,
            "f1": round(200 * tp / (2 * tp + fp + fn), 2) if tp else 0.0}


def cluster_coherence(n, pairs, score, t, predict, seed=0, k=2000):
    """Estimated precision of the clusters at threshold t, without labels: sample record pairs
    INSIDE the predicted clusters (weighted by the pairs each cluster implies, so big clusters
    count for what they claim) and return the share the model itself calls a match.

    Labelled pairs are drawn from blocking candidates, so they cannot see the false pairs that
    chaining creates between records that were never candidates. On historical_50k (~10 records
    per person) that hid a 242-record false cluster behind a 91.9 labelled-pair F1."""
    cc = components(n, pairs, score > t)
    sizes = np.bincount(cc)
    implied = sizes * (sizes - 1) / 2
    if implied.sum() == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    order = np.argsort(cc, kind="stable")
    start = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    c = rng.choice(len(sizes), k, p=implied / implied.sum())
    a = rng.integers(0, sizes[c])
    b = rng.integers(0, sizes[c] - 1)
    b = b + (b >= a)
    i, j = order[start[c] + a], order[start[c] + b]
    return float(np.mean(predict(np.c_[np.minimum(i, j), np.maximum(i, j)].astype(np.int32)) > 0.5))


def _redact_keys(report):
    """The blocking report without key values (a block's key is a name, email or phone)."""
    if isinstance(report, dict):
        return {k: ("(redacted)" if k == "key" else _redact_keys(v)) for k, v in report.items()}
    if isinstance(report, list):
        return [_redact_keys(v) for v in report]
    return report


def choose_threshold(n, pairs, lab_pairs, ly, fold_scores, grid, coherence=None, min_precision=None,
                     rule="middle", weight=None):
    """Closure-aware threshold from cross-fitted scores: fold_scores = [(candidate scores from a
    model that did not see these labels, held-out mask)]. Ties go to the stricter threshold.
    coherence(t) -> estimated cluster precision; when given it replaces the labelled pairs'
    precision, which cannot see chaining between records that were never candidates.
    weight: the candidate pairs each labelled pair stands for (a stratified draw); None counts
    every pair once."""
    w = np.ones(len(ly)) if weight is None else np.asarray(weight, dtype=float)
    counts = {t: np.zeros(3) for t in grid}
    for s, held in fold_scores:
        hi, hj, hy, hw = lab_pairs[held, 0], lab_pairs[held, 1], ly[held], w[held]
        for t in grid:
            cc = components(n, pairs, s > t)
            same = cc[hi] == cc[hj]
            counts[t] += [hw[same & hy].sum(), hw[same & ~hy].sum(), hw[~same & hy].sum()]
    sweep = {t: _scores(*c) for t, c in counts.items()}
    if coherence is not None:
        for t, v in sweep.items():
            p, r = 100 * coherence(t), v["recall"]
            v.update(labelled_pair_precision=v["precision"], precision=round(p, 2),
                     f1=round(2 * p * r / (p + r), 2) if p + r else 0.0)
    pool = sweep
    target = {}
    if min_precision is not None:
        # "min_precision": the most complete threshold whose estimated precision meets the
        # target; when none does, the most precise one, and the summary warns
        ok = {t: v for t, v in sweep.items() if v["precision"] >= 100 * float(min_precision)}
        if ok:
            pool = ok
        else:
            top = max(v["precision"] for v in sweep.values())
            pool = {t: v for t, v in sweep.items() if v["precision"] == top}
        target = {"min_precision": float(min_precision), "min_precision_met": bool(ok)}
    best = max(v["f1"] for v in pool.values())
    if rule == "strictest":                        # the rule before 0.1.0: ties to the stricter threshold
        th = max(t for t, v in pool.items() if v["f1"] == best)
    else:
        # thresholds the random labels cannot tell apart - within half a held-out match's worth of
        # recall of the best - count as tied, and the middle one is chosen. With few random matches
        # the recall estimate is flat across a wide range while the precision estimate creeps up,
        # and the strictest of them lost real recall: NCVR, 200,000 people, 8 random matches,
        # 0.85 chosen against a best of 0.6 (research/scripts/threshold_rules.py)
        # held-out matches, counted as a sample of that many equal ones would be (Kish's
        # effective size): equal weights give their number
        mw = np.concatenate([w[held & ly] for _, held in fold_scores])
        matches = mw.sum() ** 2 / (mw ** 2).sum() if len(mw) else 0.0
        tol = 50.0 / max(matches, 1)
        near = sorted(t for t, v in pool.items() if v["f1"] >= best - tol)
        th = near[len(near) // 2]
    curve = {f"{t:g}": {k: sweep[t][k] for k in ("precision", "recall", "f1")} for t in sorted(sweep)}
    return float(th), sweep[th] | {"pairs": int(len(ly)), "folds": len(fold_scores)} | target | {"sweep": curve}


def recut_large(n, pairs, score, th, customers, max_cluster, grid):
    """dedupe's second line of defence against chaining: a customer larger than max_cluster is
    re-clustered on its own links at the next stricter threshold, repeatedly, until every
    piece fits (or the grid runs out). Returns (labels, number of customers re-cut)."""
    labels = customers.copy()
    sizes = np.bincount(labels)
    big = np.flatnonzero(sizes > max_cluster)
    next_label = labels.max() + 1
    for c in big:
        members = np.flatnonzero(labels == c)
        inside = np.isin(pairs[:, 0], members) & np.isin(pairs[:, 1], members)
        sub_pairs, sub_score = pairs[inside], score[inside]
        local = {m: k for k, m in enumerate(members)}
        lp = np.array([[local[a], local[b]] for a, b in sub_pairs], dtype=np.int32).reshape(-1, 2)
        stricter = [g for g in grid if g > th]
        if not stricter:
            continue            # already at the strictest threshold: nothing to re-cut with
        for t in stricter:
            cc = components(len(members), lp, sub_score > t)
            if np.bincount(cc).max() <= max_cluster:
                break
        labels[members] = cc + next_label
        next_label += cc.max() + 1
    return labels, int(len(big))


def confidence(n, pairs, score, labels):
    """Per record: the mean score of its links inside its customer; for a record on its own,
    one minus its strongest candidate link (how sure we are it has no match)."""
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    tot, cnt, best = np.zeros(n), np.zeros(n), np.zeros(n)
    for col in (0, 1):
        np.add.at(tot, pairs[same, col], score[same])
        np.add.at(cnt, pairs[same, col], 1)
        np.maximum.at(best, pairs[~same, col], score[~same])
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), 1 - best)


def run(config_path, labels_path, out_dir, seed=0, log=print, model_dir=None, previous_dir=None,
        previous_extra=None, issued=-1):
    """model_dir: reuse that run's trained model, threshold and
    blocking rules instead of training (labels_path may then be None) - rescoring nightly with a
    fixed model. previous_dir: keep that run's customer IDs where records overlap and extend its
    history.csv (stewardship.carry_ids). previous_extra: {record key: customer ID} given since
    that run (a service's real-time matches), kept like the run's own; issued: the highest
    customer ID given out anywhere else, so new customers start above it."""
    import time
    from datetime import datetime, timezone
    t_start, stages = time.time(), {}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def stage(name):
        now = time.time()
        stages[name] = round(now - t_start - sum(stages.values()), 2)
        log(f"[{now - t_start:7.1f}s] {name} done")
    df, cfg = io.load(config_path)
    if cfg.get("overrides"):                       # saved with the state: `add` finds it from anywhere
        ov = cfg["overrides"]
        resolve = lambda p: str((Path(config_path).parent / p).resolve())  # noqa: E731
        cfg["overrides"] = resolve(ov) if isinstance(ov, str) else [resolve(p) for p in ov]
    identity.resolve_paths(cfg, Path(config_path).parent)
    if cfg.get("country"):
        # saved with the state: `add` must build the same name features whatever its batch holds
        cfg["_name_skeleton"] = "name_skeleton" in df.columns
    n = len(df)
    suspect_flags, profanity_list = {}, True
    if cfg.get("suspect_records", True):          # for review only: matching is not affected
        from . import suspect
        if "suspect" in df.attrs:
            suspect_flags = df.attrs["suspect"]
        else:
            words = suspect.profanity_words()
            profanity_list = words is not None
            suspect_flags = suspect.flag(df, words)
    fake_name = fake_addr = fake_dob = np.zeros(n, bool)
    if cfg.get("placeholders"):
        # placeholder names / addresses / dates ("test test", "123 main st", 1900-01-01) are not
        # evidence: blank them
        from . import placeholders
        fake_dob = placeholders.detect_dates(df)
        df, fake_name, fake_addr = placeholders.blank(df)
    profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
    fopts = schema.feature_options_for(df, cfg)
    saved = None
    if model_dir is not None:
        from .models import load_model
        saved = load_model(Path(model_dir) / incremental.STATE)
    if labels_path is not None:
        li, lj, ly, active, lw = read_labels(labels_path, df, with_active=True, with_weight=True)
    else:
        li = lj = np.zeros(0, np.int64)
        ly = active = np.zeros(0, bool)
        lw = np.zeros(0)
    if saved is None and (len(ly) < MIN_LABELS or ly.all() or not ly.any()):
        raise io.ConfigError(f"need at least {MIN_LABELS} labelled pairs with both matches and "
                             f"non-matches (got {len(ly)}, {int(ly.sum())} matches)")
    lab_pairs = np.c_[np.minimum(li, lj), np.maximum(li, lj)].astype(np.int32)
    from . import blocking, stewardship
    stage("load and profile")
    if saved is not None and saved.get("blocking") and not cfg.get("blocking"):
        # the reused model's blocker, chosen or switched to
        cfg.update(blocking=saved["blocking"], _blocking_switched=saved.get("_blocking_switched", False))
    pairs, hubs, block_report, rules = blocking.candidates_for(
        df, profiles, cfg, lab_pairs[ly], rules=saved.get("rules") if saved else None)
    if block_report.get("switched_from_core"):
        # saved with the state, so `add` blocks the same way and still checks the large blocks
        cfg.update(blocking="adaptive", _blocking_switched=True)
    stage(f"blocking ({len(pairs):,} candidate pairs)")
    if not len(pairs):
        raise io.ConfigError("blocking produced no candidate pairs: no two records share a name, "
                             "address, contact or extra blocking key - check the column mapping")

    from .models import factory, p_match
    X_lab = features(df, lab_pairs, profiles, **fopts)
    X_cand = features(df, pairs, profiles, **fopts)
    weight_scorer = None                         # "match_weight_feature": custmatch/matchweight.py
    # opt-in ("match_weight_feature": true); a reused model (--model) keeps whatever its run did
    uses_weight = bool(cfg.get("match_weight_feature")) if saved is None else bool(saved.get("match_weight_model"))
    if uses_weight and saved is not None:
        from .matchweight import load_scorer
        weight_scorer = load_scorer(df, profiles, saved["match_weight_model"])
        X_cand = np.column_stack([X_cand, weight_scorer.weights(pairs)])
        uses_weight = False                      # scored above; no fitting below
    if uses_weight:
        from . import matchweight
        (w_cand, w_lab), weight_scorer = matchweight.weights(df, profiles, [pairs, lab_pairs], cfg, seed,
                                                             return_scorer=True)
        X_cand, X_lab = np.column_stack([X_cand, w_cand]), np.column_stack([X_lab, w_lab])
    # pairs chosen by active learning are deliberately unrepresentative (the hard ones): they
    # train every fold but never choose the threshold, which must be judged on random pairs
    folds = np.where(active, -1, np.random.default_rng(seed).permutation(len(ly)) % FOLDS)

    def fit_matcher(make):
        """Cross-fitted fold scores, the final model, and the closure-aware threshold."""
        fold_scores = [(p_match(make().fit(X_lab[folds != k], ly[folds != k]), X_cand), folds == k)
                       for k in range(FOLDS)]
        final = make().fit(X_lab, ly)
        prob = p_match(final, X_cand)
        # the coherence check scores pairs inside predicted customers, most of which were never
        # candidates; the match weight scores any pair, so the full model judges them

        def predict(p):
            X = features(df, p, profiles, **fopts)
            if weight_scorer is not None:
                X = np.column_stack([X, weight_scorer.weights(p)])
            return p_match(final, X)
        th, held_out = choose_threshold(n, pairs, lab_pairs, ly, fold_scores, THRESHOLDS,
                                        coherence=lambda t: cluster_coherence(n, pairs, prob, t, predict, seed),
                                        min_precision=cfg.get("min_precision"),
                                        rule=cfg.get("threshold_rule", "middle"), weight=lw)
        return {"final": final, "prob": prob, "predict": predict, "th": th, "held_out": held_out}

    if saved is None:                             # the saved model is reused, not refitted
        model_choice = cfg.get("matcher_model", "forest")
        try:
            make = factory(cfg, seed)
        except ValueError as e:                   # an unknown or removed model: say so before fitting
            raise io.ConfigError(str(e)) from None
        fit = fit_matcher(make)
        stage(f"{model_choice}: cross-fitted clustered F1 {fit['held_out']['f1']}")
        final_forest, prob, predict, th, held_out = (fit[k] for k in ("final", "prob", "predict", "th", "held_out"))
    else:
        final_forest, th = saved["forest"], float(saved["threshold"])
        model_choice = saved.get("matcher_model", "forest")
        prob = p_match(final_forest, X_cand)

        def predict(p):
            X = features(df, p, profiles, **fopts)
            if weight_scorer is not None:
                X = np.column_stack([X, weight_scorer.weights(p)])
            return p_match(final_forest, X)
        held_out = {}

    stage("threshold (clustered, in-cluster precision check)")
    score = prob
    skipped = block_report.pop("_skipped", [])
    if blocking.large_block_check_on(cfg) and skipped:
        # oversized blocks that are one real entity (an organisation) are compared after all
        extra, rows = blocking.check_large_blocks(skipped, predict, th, budget=max(1_000_000, len(pairs)), seed=seed)
        if len(extra):
            have = pairs[:, 0].astype(np.int64) * n + pairs[:, 1]
            extra = extra[~np.isin(extra[:, 0].astype(np.int64) * n + extra[:, 1], have)]
        if len(extra):
            X_extra = features(df, extra, profiles, **fopts)
            if weight_scorer is not None:
                X_extra = np.column_stack([X_extra, weight_scorer.weights(extra)])
            pairs, score = np.vstack([pairs, extra]), np.concatenate([score, p_match(final_forest, X_extra)])
            X_cand = np.vstack([X_cand, X_extra])
        block_report["large_blocks_checked"] = rows
        block_report["large_block_pairs_added"] = int(len(extra))
        stage(f"large blocks checked: {sum(r['kept'] for r in rows)} of {len(rows)} kept, {len(extra):,} pairs added")
    # decisions people made, and identifiers that settle a match, override the model
    keys = io.key(df)
    must, cannot = stewardship.load_overrides(cfg, Path(config_path).parent, keys)
    trusted = stewardship.trusted_pairs(df, cfg)
    # cookies and device IDs seen with a login (docs/guide.md "Cookies and device IDs")
    cookie_pairs, cookie_table, cookie_counts = identity.stitch(df, cfg, profiles=profiles)
    if len(cookie_pairs):
        trusted = np.unique(np.vstack([trusted, cookie_pairs]), axis=0)
    if len(trusted):
        must = np.unique(np.vstack([must, trusted]), axis=0)
    if len(must) or len(cannot):
        n_before = len(pairs)
        pairs, score = stewardship.apply_decisions(pairs, score, must, cannot)
        if len(pairs) > n_before:                  # must-links blocking did not propose
            extra = features(df, pairs[n_before:], profiles, **fopts)
            if X_cand.shape[1] > extra.shape[1]:
                extra = np.column_stack([extra, np.zeros((len(extra), X_cand.shape[1] - extra.shape[1]))])
            X_cand = np.vstack([X_cand, extra])
    # stability as records arrive (docs/guide.md "Stable matches"): keep the previous decision
    # unless evidence moves by "stability_margin"; hold records between two people
    # measured (research/scripts/order_dependence.py): 0.05 halved partner changes and removed the
    # back-and-forth at no accuracy cost, so it is on whenever a previous run is given
    if previous_dir is not None and cfg.get("stability_margin", 0.05):
        prev_state = Path(previous_dir) / incremental.STATE
        if (prev_state / "links.parquet").exists():
            prev_recs = pd.read_parquet(prev_state / "records.parquet", columns=["source", "record_id"])
            score = stewardship.hysteresis(pairs, score, keys, pd.read_parquet(prev_state / "links.parquet"),
                                           io.key(prev_recs), float(cfg.get("stability_margin", 0.05)))
    twins = np.zeros(len(pairs), dtype=bool)
    if cfg.get("twin_guard"):                       # twins are kept apart; a steward's "same" still wins
        twins = stewardship.twin_pairs(df, pairs, hashed="first_name" in cfg.get("hash_fields", []))
        if len(must):
            forced = set(map(tuple, must.tolist()))
            twins &= np.array([(int(a), int(b)) not in forced for a, b in pairs])
        score = np.where(twins, np.minimum(score, th * 0.999), score)
    if len(must):                                  # nothing after the decisions may undo a "same"
        midx = {(int(a), int(b)): k for k, (a, b) in enumerate(pairs)}
        score = score.astype(float).copy()
        score[[midx[(int(a), int(b))] for a, b in must if (int(a), int(b)) in midx]] = 1.0
    from .core import clear_caches
    clear_caches()                                  # scoring is done: free the per-run feature caches
    customers = components(n, pairs, score > th)
    recut = 0
    if cfg.get("max_cluster"):
        grid = THRESHOLDS
        customers, recut = recut_large(n, pairs, score, th, customers, int(cfg["max_cluster"]), grid)
    customers = stewardship.join_must(customers, must)
    # duplicate-free sources: two of a source's records in one customer are two people
    same_source = stewardship.unique_source_pairs(df, customers, stewardship.unique_sources(cfg, df))
    split_by_source = int(len(same_source))
    cannot_all = np.vstack([cannot, same_source]) if len(same_source) else cannot
    customers, cut = stewardship.enforce_cannot(n, pairs, score, th, customers, cannot_all)
    # households and the published contact classes use customer-level profiles (after matching):
    # counting customers, not first names, lifted household-contact precision from 58% to 84%
    from . import households as hh_mod
    profiles = {f: hh_mod.profile_by_customer(df, f, customers) for f in ("email", "phone")}
    households, _ = hh_mod.assign(df, customers, profiles, postcode_join=bool(cfg.get("household_postcode_join")))
    rank = {s: k for k, s in enumerate(cfg.get("source_priority", []))}
    fields = [f for f in cfg.get("golden_fields", GOLDEN_FIELDS) if f in df.columns]
    golden = build_golden(df, customers, cfg.get("survivorship", "most_complete"), fields, rank)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if previous_dir is not None:
        prev = pd.read_csv(Path(previous_dir) / "clusters.csv", dtype=str)
        previous = dict(zip(prev.source + ":" + prev.record_id, prev.customer_id.astype(int)))
        previous.update(previous_extra or {})
        cust_id, events = stewardship.carry_ids(keys, customers, previous,
                                                max(stewardship.highest_issued(previous_dir), int(issued)))
    else:
        _, cust_id = np.unique(customers, return_inverse=True)
        events = stewardship.carry_ids(keys, customers, {})[1]
    stewardship.write_history(out, events, previous_dir)
    band = float(cfg.get("review_band", 0.1))
    stewardship.review_pairs(pairs, score, th, band, keys, X_cand, df, profiles).to_csv(out / "review.csv", index=False)
    _, hh_id = np.unique(households, return_inverse=True)
    pd.DataFrame({"source": df.source, "record_id": df.record_id,
                  "customer_id": cust_id, "household_id": hh_id,
                  "confidence": np.round(confidence(n, pairs, score, customers), 4)}
                 ).to_csv(out / "clusters.csv", index=False)
    identity.write_cookies(out, cookie_table, cust_id, minimal=bool(cfg.get("minimal_outputs")))
    if cfg.get("contact_confidence") is not None:
        from .golden import safe_contacts
        strong = (score >= float(cfg["contact_confidence"])) & (customers[pairs[:, 0]] == customers[pairs[:, 1]])
        per_customer, per_record = safe_contacts(df, customers, pairs[strong], cfg.get("survivorship", "most_complete"),
                                                 rank, classes=profiles)
        for lab, vals in per_customer.items():
            golden[lab].update({f: v for f, v in vals.items() if f in golden[lab]})
        if not cfg.get("minimal_outputs"):
            pd.DataFrame({"source": df.source, "record_id": df.record_id, "customer_id": cust_id,
                          **per_record}).to_csv(out / "contacts.csv", index=False)
    lab_to_id = dict(zip(customers.tolist(), cust_id.tolist()))
    g = pd.DataFrame([{"customer_id": lab_to_id[lab], **vals}
                      for lab, vals in golden.items()]).sort_values("customer_id")
    minimal = bool(cfg.get("minimal_outputs"))        # no file that repeats personal data
    if not minimal:
        g.to_csv(out / "golden_records.csv", index=False)
    linked = score > th
    state_links = pd.DataFrame({"a": keys[pairs[linked, 0]], "b": keys[pairs[linked, 1]],
                                "score": score[linked].astype(float)})
    same_customer = cust_id[pairs[linked, 0]] == cust_id[pairs[linked, 1]]
    incremental.save_state(out, df, cust_id, final_forest, th, cfg, rules,
                           match_weight_model=weight_scorer.model_json if weight_scorer is not None else None,
                           matcher_model=model_choice, links=state_links[same_customer])
    from . import audit                           # pairs a person can label to audit this run (custmatch audit)
    audit.save_pool(out / incremental.STATE, keys, pairs, score, cust_id, seed)
    if cfg.get("minimal_outputs") or cfg.get("hash_fields"):
        block_report = _redact_keys(block_report)  # the largest blocks' keys are names and contacts
    (out / "blocking_report.json").write_text(json.dumps(block_report, indent=1, default=str))
    if (fake_name.any() or fake_addr.any() or fake_dob.any()) and not minimal:
        raw, _ = io.load(config_path)
        flagged = fake_name | fake_addr | fake_dob
        pd.DataFrame({"source": raw.source[flagged], "record_id": raw.record_id[flagged],
                      "fake_name": fake_name[flagged], "fake_address": fake_addr[flagged],
                      "fake_date_of_birth": fake_dob[flagged],
                      "name": (raw.first_name + " " + raw.last_name)[flagged],
                      "address": (raw.addr_number + " " + raw.addr_street)[flagged],
                      "date_of_birth": raw.date_of_birth[flagged]}
                     ).to_csv(out / "placeholders.csv", index=False)
    if cfg.get("suspect_records", True) and not minimal:
        from . import suspect
        suspect.table(df, suspect_flags).to_csv(out / "suspect_records.csv", index=False)
    link_table = explain.links(df, profiles, pairs, X_cand, score, th, io.key(df), cust_id)
    link_table.to_csv(out / "links.csv", index=False)
    if not minimal:
        pd.DataFrame([{"field": f, "value": v, "class": c, "weight": round(w, 3)}
                      for f, prof in profiles.items() for v, (c, w) in prof.items() if c != "personal"]
                     ).to_csv(out / "shared_and_junk_contacts.csv", index=False)

    sizes = np.unique(cust_id, return_counts=True)[1]       # carried IDs are sparse
    summary = {
        "records": n, "sources": sorted(df.source.unique().tolist()),
        "candidate_pairs": int(len(pairs)), "hub_blocks_skipped": int(hubs),
        "labelled_pairs": {"total": int(len(ly)), "matches": int(ly.sum()),
                           "active": int(active.sum())},
        "matcher_model": model_choice,
        "model_reused_from": str(model_dir) if model_dir is not None else None,
        "overrides": {"must_link": int(len(must)), "cannot_link": int(len(cannot)),
                      "trusted_id_links": int(len(trusted)), "links_cut_to_keep_cannot_links_apart": int(cut)},
        "review_pairs": int(np.sum(np.abs(score - th) < band)),
        **({"cookies": cookie_counts} if cookie_counts else {}),
        **({"same_source_pairs_split": split_by_source} if cfg.get("unique_sources") else {}),
        "twin_pairs_kept_apart": int((twins & (score > th * 0.99)).sum()) if cfg.get("twin_guard") else 0,
        "threshold": float(th),
        "cross_fitted_closure_aware": {k: v for k, v in held_out.items() if k != "sweep"},
        # estimated precision, recall and F1 at every threshold tried - the threshold chart
        "threshold_sweep": held_out.get("sweep"),
        "customers": int(len(sizes)), "largest_customer": int(sizes.max()),
        "households": int(len(np.unique(hh_id))),
        "links": int(len(link_table)), "bridge_links": int(link_table.is_bridge.sum()),
        "customers_recut": recut,
        "placeholder_names": int(fake_name.sum()), "placeholder_addresses": int(fake_addr.sum()),
        "placeholder_dates_of_birth": int(fake_dob.sum()),
        "stage_seconds": stages,
        "junk_contact_values": sum(c == "junk" for p in profiles.values() for c, _ in p.values()),
        "shared_contact_values": sum(c == "household" for p in profiles.values() for c, _ in p.values()),
        "warnings": [],
        "suspect_records": len(suspect_flags),
    }
    if saved is None and not (ly & (folds >= 0)).any():
        summary["warnings"].append("no randomly drawn labelled pair is a match, so the threshold could "
                                   "not be judged and fell back to the strictest; label a random batch "
                                   "(custmatch sample) as well as active ones")
    random_matches = int(ly[~active].sum()) if len(ly) else 0
    if saved is None and len(ly) and not active.any():
        summary["warnings"].append(
            "every label was drawn at random: add actively chosen pairs (custmatch sample --active). With "
            "300 random labels and none chosen actively, custmatch was less precise than well-configured "
            "exact-identifier matching (docs/report/08-against-a-cdp.md \"Reassessed for marketing\")")
    if saved is None and len(ly) and 0 < random_matches < MIN_RANDOM_MATCHES:
        summary["warnings"].append(
            f"only {random_matches} of your randomly drawn labels are matches, and the threshold is judged on "
            f"those: label more random pairs (custmatch sample) until at least {MIN_RANDOM_MATCHES} are matches")
    bad = io.unknown_keys(cfg)
    if bad:
        summary["warnings"].append("config keys custmatch does not use, so they had no effect: " + ", ".join(bad))
    if not profanity_list:
        summary["warnings"].append("the profanity list could not be downloaded, so suspect_records.csv does not "
                                   "check for profanity; copy it into the data folder to check offline")
    if cfg.get("twin_guard") and "first_name" in cfg.get("hash_fields", []):
        summary["warnings"].append("twin_guard is off: hashed first names cannot be told apart from typos")
    if held_out and held_out.get("min_precision_met") is False:
        summary["warnings"].append(
            f"no threshold reached the min_precision of {100 * held_out['min_precision']:.0f}% (best estimate "
            f"{held_out['precision']}%); the most precise one was used - label more pairs, or lower the target")
    if saved is None and len(ly) < RECOMMENDED_LABELS:
        summary["warnings"].append(f"{len(ly)} labelled pairs; label at least {RECOMMENDED_LABELS} "
                                   "(more for dense files)")
    in_skipped = block_report.get("records_only_in_skipped_blocks", 0)
    if in_skipped >= 0.01 * n:
        eg = ", ".join(f"{b['key']} ({b['records']})" for b in block_report["largest_skipped_blocks"][:3])
        summary["warnings"].append(
            f"{in_skipped:,} records were never compared with any other: every block they share is larger "
            f"than max_block, so it was skipped. Largest: {eg}. See blocking_report.json; raise "
            "\"max_block\" if these values are legitimately common, check them for placeholders, or "
            "split them instead (\"blocking\": \"adaptive\")")
    relay, found = relay_records(df)
    k = int((relay & ((df.first_name != "") | (df.last_name != "")).to_numpy()).sum())
    if k:
        eg = "" if cfg.get("minimal_outputs") else f" (e.g. {', '.join(found[:3])})"
        summary["warnings"].append(
            (f"{k:,} records have" if k > 1 else "1 record has") +
            f" a no-reply or notification address and a person's name{eg}: "
            "usually a service that wrote on someone's behalf, filed under their name, not that person's contact. "
            "The address is ignored as evidence, but the name can still link the record to that person; remove "
            "such records if they are not people")
    for row in block_report.get("pairs_by_key", []):
        part = row.get("pairs_with_a_blank_part", 0)
        if part >= max(BLANK_KEY_PAIRS, BLANK_KEY_SHARE * summary["candidate_pairs"]):
            summary["warnings"].append(
                f"\"{row['key']}\" was blank in one part for many records, so it blocked on the other part "
                f"alone (first name + street is first name alone where there is no street): it gave {part:,} "
                f"pairs, against {summary['candidate_pairs']:,} candidate pairs in all. Those pairs share one field and are "
                "mostly different people, so random labels drawn from them hold few matches - label more, or "
                "fill in the missing field. See pairs_by_key in blocking_report.json")
    switched = block_report.get("switched_from_core")
    if switched:
        summary["warnings"].append(
            f"blocking switched to \"adaptive\": with the default keys {switched['records_only_in_skipped_blocks']:,} "
            f"records ({100 * switched['share']:.1f}%) were only in blocks larger than max_block and would never "
            "have been compared, so oversized blocks were split instead (set \"blocking\": \"core\" to keep "
            "the default keys and skip them)")
    if sizes.max() > 20:
        summary["warnings"].append(f"largest customer has {int(sizes.max())} records - check it "
                                   "for chaining (clusters.csv)")
    stages["households, golden records, outputs"] = round(time.time() - t_start - sum(stages.values()), 2)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    monitor.write(out, monitor.snapshot(df, cust_id, score, th))
    from . import manifest                        # last: it fingerprints every file written above
    manifest.write(out, manifest.build("run", config_path, json.loads(Path(config_path).read_text()), df, seed,
                                       labels_path, started, out, model=model_dir, previous=previous_dir))
    return summary
