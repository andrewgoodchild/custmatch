"""Entity matching with Jev, TypeSafe's decision model, through OpenRouter's Decisions API.

Jev is not a text model. It takes application state plus typed questions and returns typed
answers with probabilities (https://openrouter.ai/docs/guides/community/jev). A match is one
yes/no ("noul") question, so each pair gets P(same entity) instead of a parsed "Yes"/"No".

To stay comparable with `llm_zeroshot_*.py`, the question is the MatchGPT task's own prompt
sentence (minus the "Answer with 'Yes'..." instruction, which a typed answer makes redundant),
the state is the two serialised records, and no criteria or examples are added. F1 is reported
at P > 0.5; the per-pair probabilities are kept so thresholds can be studied separately.

Spend goes to the same ledger as `llm_zeroshot_openrouter.py`, under the same --budget.
The model is pinned (typesafe/jev-1.13); the response names the dated snapshot.

Usage: jev_decisions.py [--datasets ...] [--limit N] [--budget USD]
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from llm_zeroshot_mlx import DATASETS, OUT, TASKS, prf
from llm_zeroshot_openrouter import LEDGER, api_key, ledger_cost, spent

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
TAG = "jev-1.13"
_lock = threading.Lock()


def question(task_prefix):
    return task_prefix.split(" Answer with")[0].strip()


def ask(client, instructions, pair_input):
    r1, r2 = [line.split(": ", 1)[1] for line in pair_input.split("\n")]
    body = {"model": MODEL, "state": {"record_1": r1, "record_2": r2},
            "questions": {"match": {"type": "noul", "instructions": instructions}}}
    for attempt in range(6):
        r = client.post(URL, json=body)
        if (r.status_code == 429 or r.status_code >= 500) or (r.is_success and "answers" not in r.json()):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        d = r.json()
        return float(d["answers"]["match"]["noul"]), d["usage"], d.get("model")
    raise RuntimeError(f"no usable response after retries: {r.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*",
                    default=DATASETS + ["historical50k-person-hard", "customer-person-hard"])
    ap.add_argument("--prompt", default="domain-complex-force")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=float, default=5.80, help="USD, across all runs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {api_key()}"})
    total = spent()
    print(f"ledger so far ${total:.4f} of ${args.budget:.2f}", flush=True)
    if total >= args.budget:
        raise SystemExit("budget already spent - not starting")

    for ds in args.datasets:
        task = json.load(open(TASKS / f"{ds}_{args.prompt}.json"))
        examples = task["examples"][: args.limit]
        instructions = question(task["task_prefix"])
        path = OUT / f"answers_{ds}_{args.prompt}_{TAG}.jsonl"
        done = {json.loads(l)["i"] for l in open(path)} if path.exists() else set()
        todo = [i for i in range(len(examples)) if i not in done]
        fh = open(path, "a")
        stop = threading.Event()

        def one(i):
            nonlocal total
            if stop.is_set():
                return
            p, usage, snapshot = ask(client, instructions, examples[i]["input"])
            with _lock:
                total += float(usage.get("cost", 0.0))
                fh.write(json.dumps({"i": i, "p_yes": p, "snapshot": snapshot,
                                     "target": examples[i]["target_scores"]["Yes"]}) + "\n")
                fh.flush()
                with open(LEDGER, "a") as led:
                    led.write(json.dumps({"model": MODEL, "dataset": ds, "i": i,
                                          "cost": float(usage.get("cost", 0.0)),
                                          "prompt_tokens": usage.get("input_tokens"),
                                          "completion_tokens": usage.get("output_tokens")}) + "\n")
                if total >= args.budget:
                    stop.set()

        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(one, todo))
        fh.close()
        print(f"{ds}: {len(todo)} calls, ledger ${total:.4f}, {time.time() - t0:.0f}s", flush=True)
        if stop.is_set():
            raise SystemExit(f"budget ${args.budget:.2f} reached - stopping; rerun resumes")

        rows = sorted((json.loads(l) for l in open(path)), key=lambda r: r["i"])
        rows = [r for r in rows if r["i"] < len(examples)]
        truth = [bool(r["target"]) for r in rows]
        from jev_variants import binary  # AUC and a cut-off tuned on one half, scored on the other
        res = prf(truth, [r["p_yes"] > 0.5 for r in rows]) | binary(
            [r["target"] for r in rows], [r["p_yes"] for r in rows]) | {
            "dataset": ds, "prompt": args.prompt, "model": MODEL,
            "snapshots": sorted({r["snapshot"] for r in rows}), "threshold": 0.5,
            "cost_usd": round(ledger_cost(MODEL, ds), 4)}
        if args.limit is None:
            json.dump(res, open(OUT / f"result_{ds}_{args.prompt}_{TAG}.json", "w"), indent=1)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
