"""custmatch on BPID (Wang et al., EMNLP 2024 Industry Track): a benchmark for personal identity
deduplication from AWS. Synthetic profiles - SSA first names, Census surnames, USDOT addresses -
with variations written by trained annotators and by program, and three annotators' unanimous
match / no-match on each labelled pair. Someone else wrote its errors, which is the point.

    python research/scripts/bpid_benchmark.py [--variant shipped] [--skip-blocking] [--tag NAME]

Download (Apache 2.0, 58 MB) into the gitignored data/bpid/ and unzip it there:
    https://zenodo.org/records/13932202
It is synthetic: no real people.

Two files, two tasks, the paper's own measures:
  matching_dataset.jsonl  10,000 labelled pairs (4,333 matches). Pairwise precision, recall, F1.
                          The paper gives no split; here 5-fold cross-validation, stratified,
                          each fold's threshold chosen on its training folds by inner cross-fitting.
                          Every profile is in exactly one pair, so folds share no records.
  blocking_dataset.jsonl  1,000,000 unlabelled profiles holding all 20,000 of the pairs'. Recall
                          of the 4,333 matching pairs among custmatch's candidate pairs, and how
                          many candidates (the paper's best: 0.711 at 10M).
The pairs' features are computed with contacts profiled over all million profiles, as a run
would. There is no ground truth for the million, so no clustered F1.

Each profile is a full name, lists of emails, phones and addresses, and a date of birth in one of
dozens of forms. The adapter only reshapes: the name's first word is the first name, its last the
surname, a middle word's initial the middle initial; the first email, phone and address are the
fields, further emails and phones their aliases; further addresses are dropped (aliases cover
surnames, emails and phones). The date of birth goes in as written.
Writes research/results/bpid.json.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import blocking, io, schema
from custmatch.core import features, profile_contacts
from custmatch.models import factory, p_match

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/bpid/data_release"
OUT = ROOT / "research/results/bpid.json"
N_ALIAS = 3            # 3 emails and 3 phones cover every profile
GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
PAPER = {"random_forest": 0.629, "ditto": 0.752, "sudowoodo": 0.788, "claude3_sonnet": 0.658,
         "llama3_70b": 0.729, "mistral_large": 0.604, "gpt4_turbo": 0.687,
         "blocking_best_recall": 0.711, "blocking_best_candidates": 10_000_000}
VARIANTS = {
    # what a user would write: the columns, the aliases, the country
    "shipped": {"country": "US"},
    # the date of birth also compared by edit distance, as the guide suggests for messy dates
    "dob_edit": {"country": "US", "extra_fields": [{"name": "date_of_birth", "compare": "edit"}]},
}


def canon(p):
    return json.dumps(p, sort_keys=True)


def to_row(i, p):
    words = p["fullname"].split()
    row = {"id": str(i), "first_name": words[0] if words else "",
           "middle": words[1][:1] if len(words) > 2 else "",
           "last_name": words[-1] if len(words) > 1 else "",
           "dob": p["dob"], "address": p["addr"][0] if p["addr"] else ""}
    for f, vals in (("email", p["email"]), ("phone", p["phone"])):
        for k in range(N_ALIAS):
            row[f if k == 0 else f"{f}_{k + 1}"] = vals[k] if k < len(vals) else ""
    return row


def build(tmp):
    """The million profiles as one source file; the labelled pairs as row indices into it."""
    csv = tmp / "bpid.csv"
    where = {}
    rows = []
    with open(DATA / "blocking_dataset.jsonl") as fh:
        for i, line in enumerate(fh):
            p = json.loads(line)
            where[canon(p)] = i
            rows.append(to_row(i, p))
    pd.DataFrame(rows).to_csv(csv, index=False)
    pi, pj, y = [], [], []
    with open(DATA / "matching_dataset.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            pi.append(where[canon(d["profile1"])])
            pj.append(where[canon(d["profile2"])])
            y.append(d["match"] == "True")
    a, b = np.array(pi), np.array(pj)
    return csv, np.c_[np.minimum(a, b), np.maximum(a, b)].astype(np.int32), np.array(y)


def config(csv, extra):
    cols = {"record_id": "id", "first_name": "first_name", "middle_initial": "middle", "last_name": "last_name",
            "date_of_birth": "dob", "address": "address", "email": "email", "phone": "phone"}
    alias = {f: [f"{f}_{k}" for k in range(2, N_ALIAS + 1)] for f in ("email", "phone")}
    for names in alias.values():
        cols.update({n: n for n in names})
    return {"sources": [{"name": "bpid", "path": csv.name, "columns": cols}], "aliases": alias} | extra


def scores(y, pred):
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4)}


def best_threshold(make, X, y, seed):
    """The pairwise-F1 threshold on out-of-fold scores - chosen without the test fold."""
    folds = np.random.default_rng(seed).permutation(len(y)) % 4
    oof = np.zeros(len(y))
    for k in range(4):
        oof[folds == k] = p_match(make().fit(X[folds != k], y[folds != k]), X[folds == k])
    return float(max(GRID, key=lambda t: scores(y, oof > t)["f1"]))


def cross_validate(make, X, y, n_train, seed):
    """5 stratified folds; train on n_train pairs drawn at random from the other four."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)
    for cls in (True, False):
        idx = np.flatnonzero(y == cls)
        fold[rng.permutation(idx)] = np.arange(len(idx)) % 5
    pred, prob = np.zeros(len(y), bool), np.zeros(len(y))
    for k in range(5):
        train = np.flatnonzero(fold != k)
        if n_train < len(train):
            train = rng.choice(train, n_train, replace=False)
        th = best_threshold(make, X[train], y[train], seed)
        prob[fold == k] = p_match(make().fit(X[train], y[train]), X[fold == k])
        pred[fold == k] = prob[fold == k] > th
    return scores(y, pred) | {"train_pairs": int(min(n_train, len(y) * 4 // 5))}


def _digits(p):
    return re.sub(r"\D", "", p.lower().split("x")[0])          # "400 8908778788 x 2922" -> no extension


def error_kinds(a, b):
    """What a wrongly judged pair has in common, from the raw profiles."""
    na, nb = set(a["fullname"].split()), set(b["fullname"].split())
    pa, pb = {_digits(p) for p in a["phone"]}, {_digits(p) for p in b["phone"]}
    return {"shared_email_or_phone": bool(set(a["email"]) & set(b["email"]) or pa & pb),
            "shared_contact_names_differ": bool((set(a["email"]) & set(b["email"]) or pa & pb) and na and nb and na != nb),
            "a_name_blank": not na or not nb,
            "phone_agrees_only_in_last_10_digits": bool({p[-10:] for p in pa if len(p) >= 10}
                                                        & {p[-10:] for p in pb if len(p) >= 10}) and not pa & pb,
            "shared_address_not_both_first": bool(set(a["addr"]) & set(b["addr"]))
                                             and not (a["addr"][:1] and a["addr"][:1] == b["addr"][:1])}


def errors(X, y, lab, cfg, seed=1):
    """Out-of-fold errors on one draw, 8,000 training pairs a fold, and what they have in common."""
    fold = np.random.default_rng(seed).permutation(len(y)) % 5
    oof = np.zeros(len(y))
    make = factory(cfg, seed)
    for k in range(5):
        oof[fold == k] = p_match(make().fit(X[fold != k], y[fold != k]), X[fold == k])
    th = float(max(GRID, key=lambda t: scores(y, oof > t)["f1"]))
    want = set(lab.ravel().tolist())
    raw = {}
    with open(DATA / "blocking_dataset.jsonl") as fh:
        for i, line in enumerate(fh):
            if i in want:
                raw[i] = json.loads(line)
    out = {}
    for name, idx in (("false_matches", np.flatnonzero((oof > th) & ~y)), ("missed_matches", np.flatnonzero((oof <= th) & y))):
        kinds = [error_kinds(raw[lab[k][0]], raw[lab[k][1]]) for k in idx]
        out[name] = {"count": len(idx)} | {k: round(float(np.mean([x[k] for x in kinds])), 4) for k in kinds[0]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="shipped", choices=sorted(VARIANTS))
    ap.add_argument("--skip-blocking", action="store_true")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--tag", help="the result's name (default: the variant's)")
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res["paper"] = PAPER
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t0 = time.time()
        csv, lab, y = build(tmp)
        cfg_path = tmp / "config.json"
        cfg_path.write_text(json.dumps(config(csv, VARIANTS[a.variant])))
        df, cfg = io.load(cfg_path)
        profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
        X = features(df, lab, profiles, **schema.feature_options_for(df, cfg))
        out = {"load_seconds": round(time.time() - t0, 1), "records": len(df), "features": int(X.shape[1])}
        print(f"{a.variant}: {len(df):,} records, {X.shape[1]} features, {out['load_seconds']}s", flush=True)
        seeds = [int(s) for s in a.seeds.split(",")]
        for n_train in (300, 1000, 8000):
            runs = [cross_validate(factory(cfg, s), X, y, n_train, s) for s in seeds]
            out[f"matching_{n_train}"] = {k: round(float(np.mean([r[k] for r in runs])), 4)
                                          for k in ("precision", "recall", "f1")} | {
                "f1_per_seed": [r["f1"] for r in runs], "train_pairs": runs[0]["train_pairs"]}
            print(f"  {n_train} labels: {out[f'matching_{n_train}']}", flush=True)
        out["errors"] = errors(X, y, lab, cfg)
        print(f"  errors: {out['errors']}", flush=True)
        if not a.skip_blocking:
            t0 = time.time()
            pairs, _, report, _ = blocking.candidates_for(df, profiles, cfg)
            have = pairs[:, 0].astype(np.int64) * len(df) + pairs[:, 1]
            pos = lab[y]
            found = np.isin(pos[:, 0].astype(np.int64) * len(df) + pos[:, 1], have)
            out["blocking"] = {"recall": round(float(found.mean()), 4), "candidate_pairs": len(pairs),
                               "seconds": round(time.time() - t0, 1),
                               "blocks_skipped": len(report.get("_skipped", []))}
            print(f"  blocking: {out['blocking']}", flush=True)
        res[a.tag or a.variant] = out
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
