"""Which of Jev's options improve person matching? One controlled variant at a time.

`jev_decisions.py` asked Jev the bare task question over two serialised strings. Jev's API
offers more (https://openrouter.ai/docs/guides/community/jev): `criteria` guidance for each
answer, structured JSON `state`, `choice` questions over several labelled options, and several
questions per call. Each variant below changes one of those, on the two person tasks:

  criteria             bare strings, plus written guidance for yes and for no
  structured           records as JSON objects with named fields, bare question
  structured_criteria  both
  choice3              (customer only) same person / same household / different, structured
  multi                (customer only) one call: match, same household, and "is this record's
                       contact a placeholder" for each record, structured + criteria

The criteria text is fixed below and was written from domain knowledge before any variant was
run - it is not tuned on these pairs. Every variant is reported. Cut-offs are tuned on one
half of the pairs and scored on the other (the split used for jev-1.13 in chapter 10).
"""
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from jev_decisions import MODEL, URL
from llm_zeroshot_openrouter import LEDGER, OUT, api_key, spent

ROOT = Path(__file__).resolve().parents[2]
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
CHOICE = {
    "same_person": CRITERIA["true"],
    "same_household": "Different individuals who live together: relatives or housemates who "
                      "share a surname, an address or contact details.",
    "different": "Different individuals who do not live together.",
}
_lock = threading.Lock()


# --- the two tasks, as structured records ------------------------------------------------

def historical():
    from person_task_baselines import FIELDS, rebuild_pairs
    rows, pairs, *_ = rebuild_pairs()
    task = json.load(open(ROOT / "external/MatchGPT/LLMForEM/tasks/"
                               "historical50k-person-hard_domain-complex-force.json"))
    ser = lambda r: " ".join(str(r[f]) for f in FIELDS if str(r[f]).strip())
    by_input = {f"Record 1: '{ser(rows[a])}'\nRecord 2: '{ser(rows[b])}'": (a, b) for a, b, _ in pairs}
    fields = {"first_name": "first_name", "surname": "surname", "date_of_birth": "dob",
              "birth_place": "birth_place", "postcode": "postcode_fake", "occupation": "occupation"}
    struct = lambda r: {k: r[v] for k, v in fields.items() if str(r[v]).strip()}
    out = []
    for e in task["examples"]:  # task order, so the half split matches jev_decisions
        a, b = by_input[e["input"]]
        r1, r2 = [l.split(": ", 1)[1] for l in e["input"].split("\n")]  # as jev_decisions sends them
        out.append({"strings": (r1, r2), "structured": (struct(rows[a]), struct(rows[b])),
                    "match": e["target_scores"]["Yes"]})
    return out


def customer():
    import customer_person_llm_task as C
    rows, pairs, _ = C.build()
    keys = ["first_name", "middle_initial", "last_name", "date_of_birth", "address", "email", "phone"]
    struct = lambda r: {k: r[k] for k in keys if str(r[k]).strip()}
    return [{"strings": (f"'{C.serialise(rows[a])}'", f"'{C.serialise(rows[b])}'"),
             "structured": (struct(rows[a]), struct(rows[b])), "match": y,
             "household": int(rows[a]["household_id"] == rows[b]["household_id"]),
             "junk": (int(rows[a]["contact_class"] == "junk"), int(rows[b]["contact_class"] == "junk"))}
            for a, b, y in pairs]


# --- the variants ------------------------------------------------------------------------

def body(variant, pair):
    structured = variant != "criteria"
    r1, r2 = pair["structured"] if structured else pair["strings"]
    state = {"record_1": r1, "record_2": r2}
    q = {}
    if variant in ("criteria", "structured_criteria", "multi"):
        q["match"] = {"type": "noul", "instructions": f"{QUESTION} {PLACEHOLDERS}", "criteria": CRITERIA}
    elif variant == "structured":
        q["match"] = {"type": "noul", "instructions": QUESTION}
    elif variant == "choice3":
        q["relation"] = {"type": "choice", "instructions": "How are the people described by "
                         f"`record_1` and `record_2` related? {PLACEHOLDERS}", "criteria": CHOICE}
    if variant == "multi":
        q["household"] = {"type": "noul", "instructions": "Do `record_1` and `record_2` describe "
                          "people in the same household (including the same person twice)?"}
        for k in ("record_1", "record_2"):
            q[f"junk_{k}"] = {"type": "noul", "instructions": f"Is the email or phone in `{k}` a "
                              "placeholder or obviously fake value rather than a real contact?"}
    return {"model": MODEL, "state": state, "questions": q}


def ask(client, payload):
    for attempt in range(6):
        r = client.post(URL, json=payload)
        if (r.status_code == 429 or r.status_code >= 500) or (r.is_success and "answers" not in r.json()):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(r.text[:200])


def run(client, task_name, variant, pairs, budget, state):
    path = OUT / f"answers_jevvar_{task_name}_{variant}.jsonl"
    done = {json.loads(l)["i"] for l in open(path)} if path.exists() else set()
    fh = open(path, "a")

    def one(i):
        if state["total"] >= budget:
            return
        d = ask(client, body(variant, pairs[i]))
        with _lock:
            state["total"] += d["usage"]["cost"]
            fh.write(json.dumps({"i": i, "answers": d["answers"], "snapshot": d.get("model")}) + "\n")
            fh.flush()
            with open(LEDGER, "a") as led:
                led.write(json.dumps({"model": f"{MODEL}:{variant}", "dataset": task_name, "i": i,
                                      "cost": d["usage"]["cost"],
                                      "prompt_tokens": d["usage"].get("input_tokens")}) + "\n")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, [i for i in range(len(pairs)) if i not in done]))
    fh.close()
    if state["total"] >= budget:
        raise SystemExit(f"budget ${budget:.2f} reached - stopping; rerun resumes")
    return {r["i"]: r["answers"] for r in map(json.loads, open(path))}


# --- scoring -----------------------------------------------------------------------------

THS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def halves(n):
    idx = list(range(n))
    random.Random(0).shuffle(idx)
    return idx[::2], idx[1::2]


def binary(y, p):
    y, p = np.asarray(y), np.asarray(p)
    a, b = halves(len(y))
    held = []
    for tune, test in ((a, b), (b, a)):
        t = max(THS, key=lambda t: f1_score(y[tune], p[tune] > t))
        held.append(100 * f1_score(y[test], p[test] > t))
    return {"f1_at_0.5": round(100 * f1_score(y, p > 0.5), 2), "f1_tuned_held_out": round(float(np.mean(held)), 2),
            "auc": round(roc_auc_score(y, p), 4), "brier": round(float(np.mean((p - y) ** 2)), 4)}


def score(variant, pairs, ans):
    ans = [ans[i] for i in range(len(pairs))]
    if variant == "choice3":
        pr = [a["relation"]["probabilities"] for a in ans]
        res = {"match": binary([p["match"] for p in pairs], [x["same_person"] for x in pr])}
        res["household"] = binary([p["household"] for p in pairs],
                                  [x["same_person"] + x["same_household"] for x in pr])
        return res
    res = {"match": binary([p["match"] for p in pairs], [a["match"]["noul"] for a in ans])}
    if variant == "multi":
        res["household"] = binary([p["household"] for p in pairs], [a["household"]["noul"] for a in ans])
        y = [j for p in pairs for j in p["junk"]]
        pj = [a[f"junk_{k}"]["noul"] for a in ans for k in ("record_1", "record_2")]
        res["junk_contact_per_record"] = binary(y, pj)
    return res


def main():
    budget = float(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 5.80
    client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key()}"})
    state = {"total": spent()}
    print(f"ledger so far ${state['total']:.4f} of ${budget:.2f}", flush=True)
    if state["total"] >= budget:
        raise SystemExit("budget already spent - not starting")
    plan = {"historical50k-person-hard": (historical, ["criteria", "structured", "structured_criteria"]),
            "customer-person-hard": (customer, ["criteria", "structured", "structured_criteria",
                                                "choice3", "multi"])}
    results = {}
    for task_name, (loader, variants) in plan.items():
        pairs = loader()
        base = [json.loads(l) for l in open(OUT / f"answers_{task_name}_domain-complex-force_jev-1.13.jsonl")]
        base = {r["i"]: r["p_yes"] for r in base}
        results[task_name] = {"baseline": {"match": binary([p["match"] for p in pairs],
                                                           [base[i] for i in range(len(pairs))])}}
        for v in variants:
            before = state["total"]
            ans = run(client, task_name, v, pairs, budget, state)
            results[task_name][v] = score(v, pairs, ans) | {"cost_usd": round(state["total"] - before, 4)}
            print(task_name, v, json.dumps(results[task_name][v]), flush=True)
    json.dump(results, open(ROOT / "research/results/jev_variants.json", "w"), indent=1)


if __name__ == "__main__":
    main()
