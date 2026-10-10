"""LLM labelling, as researched in docs/report/04-labels.md - not part of custmatch.

custmatch once had a `label` command: TypeSafe's Jev decision model, called through OpenRouter,
labelled candidate pairs after passing a check on pairs a person had labelled. It was tried on
synthetic data to see what the hype was about, and dropped: it added little over active learning
with hand labels, and sending customer records to a third party is a security risk. The code is
kept here, unchanged, for the research scripts whose results the report cites
(scripts/custmatch_validate.py, scripts/beat_splink.py). Needs httpx, and OPENROUTER_API_KEY in
the environment or a .env file.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import io
from custmatch.labels import candidates, record

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
COST_PER_PAIR = 0.000027  # measured on customer-style pairs; the run tracks the real cost
# identical to scripts/jev_variants.py, where these were measured (a test checks it)
QUESTION = "Do the two person records refer to the same real-world person?"
PLACEHOLDERS = ("Placeholder contact values such as test@test.com, none@none.com, xxx@xxx.com, "
                "0000000000 or 1111111111 are not evidence either way.")
CRITERIA = {
    "true": "Both records describe the same individual. Records of one person often differ "
            "through clerical error: nicknames, initials or misspellings in names, a missing or "
            "abbreviated field, a date of birth with a digit or year wrong, or an address, email "
            "or phone that changed or was entered differently.",
    "false": "The records describe different individuals, such as relatives or household "
             "members who share a surname, address or contact details, or people whose first "
             "names or dates of birth clearly conflict.",
}
# what leaves the machine when an LLM labels: never an SSN unless the config's "llm_fields" says so
LLM_FIELDS = ["first_name", "middle_initial", "last_name", "date_of_birth", "email", "phone", "address"]



class GateFailed(RuntimeError):
    pass



def llm_record(r, cfg):
    """The record as sent to OpenRouter: LLM_FIELDS, or the config's "llm_fields"."""
    return record(r, cfg.get("llm_fields", LLM_FIELDS))



def _api_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = Path(".env")
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY is not set (environment or ./.env)")


def jev(record_pairs, max_cost, workers=16):
    """P(match) per (record_a, record_b) dict pair; None where a pair was not labelled. Stops
    starting new requests once max_cost is spent (in-flight ones may add a few cents). A pair
    whose request fails stays None instead of aborting the batch and losing what was paid for."""
    import httpx
    client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {_api_key()}"})
    p = [None] * len(record_pairs)
    spent = {"usd": 0.0, "failed": 0}
    lock = threading.Lock()

    def one(k):
        try:
            _one(k)
        except Exception:
            with lock:
                spent["failed"] += 1

    def _one(k):
        with lock:
            if spent["usd"] >= max_cost:
                return
        a, b = record_pairs[k]
        body = {"model": MODEL, "state": {"record_1": a, "record_2": b},
                "questions": {"match": {"type": "noul", "instructions": f"{QUESTION} {PLACEHOLDERS}",
                                        "criteria": CRITERIA}}}
        for attempt in range(6):
            r = client.post(URL, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            d = r.json()
            with lock:
                p[k] = float(d["answers"]["match"]["noul"])
                spent["usd"] += float((d.get("usage") or {}).get("cost", 0.0))
            return
        r.raise_for_status()

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, range(len(record_pairs))))
    if spent["failed"]:
        print(f"custmatch: {spent['failed']} request(s) failed; those pairs are left unlabelled", file=sys.stderr)
    return p, spent["usd"]


def label(config_path, n, check_path, out_path, min_accuracy=98.0, max_cost=1.0, seed=0,
          judge=jev):
    """Gate Jev on your hand-labelled pairs, then label n more candidate pairs.

    Writes a labels file for `run`: your hand labels plus Jev's, with a `by` column.
    `judge` is injectable for tests; it takes record pairs and max_cost, returns (p, cost)."""
    from .pipeline import read_labels
    df, cfg = io.load(config_path)
    rows, keys = df.to_dict("records"), io.key(df)
    hi, hj, hy = read_labels(check_path, df)
    if len(hy) < 100:
        raise GateFailed(f"the check needs at least 100 hand-labelled pairs (got {len(hy)})")
    pairs = candidates(df, cfg)
    todo = min(n, len(pairs))
    estimate = (len(hy) + todo) * COST_PER_PAIR
    if estimate > max_cost:
        raise GateFailed(f"estimated ${estimate:.2f} exceeds max_cost ${max_cost:.2f}")

    p_check, cost = judge([(llm_record(rows[i], cfg), llm_record(rows[j], cfg)) for i, j in zip(hi, hj)], max_cost)
    pc = np.array([x if x is not None else np.nan for x in p_check])
    if np.isnan(pc).any():
        raise GateFailed("the check pairs were not all labelled (cost limit?)")
    pred = pc > 0.5
    tp, fp, fn = int((pred & hy).sum()), int((pred & ~hy).sum()), int((~pred & hy).sum())
    gate = {"check_pairs": int(len(hy)), "accuracy": round(100 * float((pred == hy).mean()), 2),
            "precision": round(100 * tp / (tp + fp), 2) if tp + fp else 0.0,
            "recall": round(100 * tp / (tp + fn), 2) if tp + fn else 0.0,
            "min_accuracy": min_accuracy, "cost_usd": round(cost, 4)}
    if gate["accuracy"] < min_accuracy:
        raise GateFailed(
            f"Jev labelled {gate['accuracy']}% of your {len(hy)} check pairs correctly "
            f"(precision {gate['precision']}, recall {gate['recall']}); below {min_accuracy}% a "
            "forest trained on its labels degrades badly (docs/report/04-labels.md: 91% cost 10.6 "
            f"clustered F1). Hand-label instead: custmatch sample. Gate: {json.dumps(gate)}")

    gate["warnings"] = []
    if gate["precision"] < 95:
        # false match labels teach the forest to merge people; in validation a gate that passed
        # at 98.0% accuracy but 91.8% precision cost 1.4 clustered F1 against 1,000 hand labels
        gate["warnings"].append(f"Jev's precision on your check pairs is {gate['precision']}%; "
                                "expect the forest to merge more people than with hand labels")
    checked = {(min(i, j), max(i, j)) for i, j in zip(hi, hj)}
    pool = np.array([k for k, (i, j) in enumerate(pairs) if (i, j) not in checked], dtype=int)
    pick = np.sort(np.random.default_rng(seed).choice(pool, min(todo, len(pool)), replace=False)) \
        if len(pool) else np.zeros(0, dtype=int)  # every candidate already hand-labelled
    p_new, cost2 = judge([(llm_record(rows[i], cfg), llm_record(rows[j], cfg)) for i, j in pairs[pick]],
                         max_cost - cost)
    out = [{"a": keys[i], "b": keys[j], "label": int(y), "p_match": "", "by": "hand"}
           for i, j, y in zip(hi, hj, hy)]
    out += [{"a": keys[i], "b": keys[j], "label": int(p > 0.5), "p_match": round(p, 4), "by": "jev"}
            for (i, j), p in zip(pairs[pick], p_new) if p is not None]
    pd.DataFrame(out).to_csv(out_path, index=False)
    gate.update(labelled_by_jev=sum(r["by"] == "jev" for r in out),
                total_cost_usd=round(cost + cost2, 4))
    return gate
