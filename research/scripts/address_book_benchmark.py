"""custmatch on the synthetic address books (research/scripts/address_book_data.py).

For each generated book (seeds 1-3), dirty (as generated) and clean (relay records removed,
the switchboard number blanked), and each labelling budget:
  random_300        300 random candidate pairs, drawn plainly
  stratified_300    300 drawn stratified (custmatch sample --stratified)
  random150_active150  150 random, then 150 chosen by active learning, 100 then 50
  all               every candidate pair labelled: the upper bound
labels come from the truth and the run is scored on its clusters: pairwise precision, recall
and F1 over records, the people with several records put wholly in one customer, the people
with several names among them (a nickname, a title, a maiden name) found, and the false
links - pairs put together that are different people - by the trap either record carries.

    python research/scripts/address_book_benchmark.py [--seeds 1,2,3]
    python research/scripts/address_book_benchmark.py --draws 10   # random_300 against stratified_300,
                                                                   # label draws 1-10 on each book
    python research/scripts/address_book_benchmark.py --cards 5    # Profile B, the address-book cards:
                                                                   # CARD_BUDGETS, label draws 1-5 per book

Generates the books first if they are missing. Writes research/results/address_book.json.
"""
import argparse
import itertools
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from custmatch import io, labels, pipeline  # noqa: E402
from custmatch.core import partition_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BOOKS = ROOT / "data/address_book"
OUT = ROOT / "research/results/address_book.json"
BUDGETS = ["random_300", "stratified_300", "random150_active150", "all"]
# Profile B: duplicates are 0.3% of the candidate pairs, so 150 random labels can hold none
CARD_BUDGETS = ["random_300", "stratified_300", "random150_active150", "stratified300_active150", "all"]


def fill(path, person):
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    lab["label"] = [int(person[a] == person[b]) for a, b in zip(lab.a, lab.b)]
    lab.to_csv(path, index=False)
    return lab


def label(config, budget, tmp, person, seed):
    lp = tmp / f"{budget}.csv"
    if budget == "all":                           # n beyond the candidate pairs: every one of them
        labels.sample(config, 10 ** 9, lp, seed=seed, stratified=False)
        fill(lp, person)
        return lp
    n = 150 if budget.startswith("random150") else 300     # random_300, stratified300_active150, ...
    labels.sample(config, n, lp, seed=seed, stratified=budget.startswith("stratified"))
    lab = fill(lp, person)
    for batch in ((100, 50) if "active" in budget else ()):
        labels.sample_active(config, lp, batch, tmp / "next.csv", seed=seed)
        lab = pd.concat([lab, fill(tmp / "next.csv", person)], ignore_index=True)
        lab.to_csv(lp, index=False)
    return lp


def score(out_dir, truth, recs):
    c = pd.read_csv(out_dir / "clusters.csv", dtype=str)
    c["key"] = c.source + ":" + c.record_id
    c = c.merge(truth, on="key")
    pid = c.person_id.to_numpy()
    _, sizes = np.unique(pid, return_counts=True)
    s = partition_scores(c.customer_id.to_numpy(), pid, float(np.sum(sizes * (sizes - 1) / 2)))
    names = recs.assign(full=(recs["first"] + " " + recs["last"]).str.strip().str.lower()).merge(truth, on="key")
    by_person = c.groupby("person_id").customer_id.nunique()
    multi = c.person_id.value_counts()
    multi = multi[multi > 1].index
    multiname = names.groupby("person_id").full.nunique()
    multiname = multiname[multiname > 1].index
    false = dict.fromkeys(sorted(set(truth.trap) - {""}) + ["none"], 0)
    for _, g in c.groupby("customer_id"):
        if g.person_id.nunique() > 1:
            for x, y in itertools.combinations(g.to_dict("records"), 2):
                if x["person_id"] != y["person_id"]:
                    for t in ({x["trap"], y["trap"]} - {""}) or {"none"}:
                        false[t] += 1
    return {"precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
            "people_with_several_records_found": f"{int((by_person[multi] == 1).sum())} of {len(multi)}",
            "people_with_several_names_found": f"{int((by_person[multiname] == 1).sum())} of {len(multiname)}",
            "false_links_by_trap": false}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--draws", type=int, help="only the two 300-label budgets, label draws 1..N per book")
    ap.add_argument("--cards", type=int, help="Profile B (cards), label draws 1..N per book and budget")
    ap.add_argument("--budgets", default=",".join(CARD_BUDGETS), help="with --cards: which budgets")
    ap.add_argument("--tag", default="", help="with --cards: kept as .../drawN/TAG, beside the plain runs")
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if a.cards:
        return cards(a, res)
    for seed in map(int, a.seeds.split(",")):
        book = BOOKS / f"s{seed}"
        if not (book / "records.csv").exists():
            subprocess.run([sys.executable, str(Path(__file__).parent / "address_book_data.py"), "--seed", str(seed)], check=True)
        truth = pd.read_csv(book / "truth.csv", dtype=str, keep_default_na=False)
        truth["key"] = "book:" + truth.record_id
        person = dict(zip(truth.key, truth.person_id))
        for version in ("records", "records_clean"):
            recs = pd.read_csv(book / f"{version}.csv", dtype=str, keep_default_na=False).assign(key=lambda d: "book:" + d.record_id)
            config = book / f"{version}.json"
            runs = ([(b, d) for b in BUDGETS[:2] for d in range(1, a.draws + 1)] if a.draws
                    else [(b, None) for b in BUDGETS])
            for budget, draw in runs:
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    lseed = draw or seed
                    lp = label(config, budget, tmp, person, lseed)
                    lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
                    summ = pipeline.run(config, lp, tmp / "out", seed=lseed, log=lambda *_: None)
                    r = score(tmp / "out", truth, recs) | {
                        "candidate_pairs": summ["candidate_pairs"], "labels": len(lab),
                        "random_matches": int(lab[lab.get("by", "random") != "active"].label.astype(int).sum()),
                        "warnings": summ["warnings"]}
                key = f"s{seed}/{'dirty' if version == 'records' else 'clean'}/{budget}" + (f"/draw{draw}" if draw else "")
                res[key] = r
                OUT.write_text(json.dumps(res, indent=1, default=float))
                print(f"{key:38s} P {r['precision']:6.2f} R {r['recall']:6.2f} F1 {r['f1']:6.2f}  "
                      f"several records {r['people_with_several_records_found']:9s} names {r['people_with_several_names_found']:8s} "
                      f"random matches {r['random_matches']:3d}  false {r['false_links_by_trap']}", flush=True)


def cards(a, res):
    """Profile B: every CARD_BUDGETS budget on label draws 1..N of each book's cards. A run whose
    labels hold no match records the error it stops with."""
    for seed in map(int, a.seeds.split(",")):
        book = BOOKS / f"s{seed}"
        if not (book / "cards.csv").exists():
            subprocess.run([sys.executable, str(Path(__file__).parent / "address_book_data.py"), "--seed", str(seed)], check=True)
        truth = pd.read_csv(book / "cards_truth.csv", dtype=str, keep_default_na=False).rename(columns={"kind": "trap"})
        truth["key"] = "cards:" + truth.card_id
        person = dict(zip(truth.key, truth.person_id))
        recs = pd.read_csv(book / "cards.csv", dtype=str, keep_default_na=False).assign(key=lambda d: "cards:" + d.card_id)
        for budget in a.budgets.split(","):
            for draw in range(1, (1 if budget == "all" else a.cards) + 1):
                key = f"s{seed}/cards/{budget}/draw{draw}" + (f"/{a.tag}" if a.tag else "")
                with tempfile.TemporaryDirectory() as tmp:
                    tmp = Path(tmp)
                    lp = label(book / "cards.json", budget, tmp, person, draw)
                    lab = pd.read_csv(lp, dtype=str, keep_default_na=False)
                    random_matches = int(lab[lab.get("by", "random") != "active"].label.astype(int).sum())
                    try:
                        summ = pipeline.run(book / "cards.json", lp, tmp / "out", seed=draw, log=lambda *_: None)
                    except io.ConfigError as e:
                        res[key] = {"error": str(e), "labels": len(lab), "matches": int(lab.label.astype(int).sum())}
                        print(f"{key:38s} stopped: {e}", flush=True)
                        continue
                    r = score(tmp / "out", truth, recs)
                dups = r.pop("people_with_several_records_found")
                res[key] = r | {"duplicates_found": dups, "candidate_pairs": summ["candidate_pairs"], "labels": len(lab),
                                "random_matches": random_matches, "warnings": summ["warnings"]}
                OUT.write_text(json.dumps(res, indent=1, default=float))
                print(f"{key:38s} P {r['precision']:6.2f} R {r['recall']:6.2f} F1 {r['f1']:6.2f}  duplicates {dups:8s} "
                      f"random matches {random_matches:2d}  false {dict((k, v) for k, v in r['false_links_by_trap'].items() if v)}",
                      flush=True)


if __name__ == "__main__":
    main()
