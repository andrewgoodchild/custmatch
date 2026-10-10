"""Zero-shot entity matching with hosted GPT models via OpenRouter, replaying the MatchGPT setup.

Same protocol as `llm_zeroshot_mlx.py` - the authors' downsampled test sets, prompt templates,
temperature 0, and their answer parsing (imported, not copied) - so the only thing that changes
is the model. Models match the authors' results workbook: `openai/gpt-4o-2024-08-06` is pinned;
gpt-4o-mini is requested as `openai/gpt-4o-mini`, because the pinned
`openai/gpt-4o-mini-2024-07-18` is served only by OpenAI directly, which an account requiring
zero data retention cannot use. That loses nothing - 2024-07-18 is gpt-4o-mini's only release -
and both then run on Azure.

Every call's actual cost, as reported by OpenRouter, is appended to a ledger shared by all runs
(results/llm/openrouter_spend.jsonl). A run refuses to start, and stops mid-run, before the
ledger total would pass --budget.

Reads OPENROUTER_API_KEY from the environment or from .env at the repo root.

Usage: llm_zeroshot_openrouter.py --model openai/gpt-4o-mini-2024-07-18 [--datasets ...] [--limit N]
"""
import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx

from llm_zeroshot_mlx import DATASETS, OUT, ROOT, TASKS, parse_answer, prf

URL = "https://openrouter.ai/api/v1/chat/completions"
LEDGER = OUT / "openrouter_spend.jsonl"
_lock = threading.Lock()


def api_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("OPENROUTER_API_KEY not set and not in .env")


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["cost"] for l in open(LEDGER))


def ask(client, model, content, max_tokens, temperature):
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}], "usage": {"include": True}}
    if temperature is not None:
        body["temperature"] = temperature
    for attempt in range(6):
        r = client.post(URL, json=body)
        if (r.status_code == 429 or r.status_code >= 500) or (r.is_success and "choices" not in r.json()):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        d = r.json()
        # a router (e.g. typesafe/jev-router) reports the model that actually answered
        return (d["choices"][0]["message"]["content"] or "", d.get("usage", {}),
                d.get("model"), d.get("provider"))
    r.raise_for_status()
    raise RuntimeError(f"no usable response after retries: {r.text[:200]}")


def ledger_cost(model, ds):
    if not LEDGER.exists():
        return 0.0
    return sum(r["cost"] for r in map(json.loads, open(LEDGER))
               if r["model"] == model and r["dataset"] == ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="domain-complex-force")
    ap.add_argument("--datasets", nargs="*", default=DATASETS + ["historical50k-person-hard"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=12,
                    help="raise to ~2000 for reasoning models: thinking counts against it")
    ap.add_argument("--no-temperature", action="store_true",
                    help="omit temperature, for models that do not support it (reasoning models)")
    ap.add_argument("--budget", type=float, default=2.80, help="USD, across all runs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]
    temperature = None if args.no_temperature else 0
    client = httpx.Client(timeout=300, headers={"Authorization": f"Bearer {api_key()}"})
    total = spent()
    print(f"ledger so far ${total:.4f} of ${args.budget:.2f}", flush=True)
    if total >= args.budget:
        raise SystemExit("budget already spent - not starting")

    for ds in args.datasets:
        task = json.load(open(TASKS / f"{ds}_{args.prompt}.json"))
        examples = task["examples"][: args.limit]
        answers_path = OUT / f"answers_{ds}_{args.prompt}_{tag}.jsonl"
        done = {json.loads(l)["i"] for l in open(answers_path)} if answers_path.exists() else set()
        todo = [i for i in range(len(examples)) if i not in done]
        fh = open(answers_path, "a")
        t0, cost_ds, stop = time.time(), 0.0, threading.Event()

        def one(i):
            nonlocal total, cost_ds
            if stop.is_set():
                return
            reply, usage, answered_by, provider = ask(
                client, args.model, task["task_prefix"] + examples[i]["input"],
                args.max_tokens, temperature)
            c = float(usage.get("cost", 0.0))
            with _lock:
                total += c
                cost_ds += c
                fh.write(json.dumps({"i": i, "reply": reply,
                                     "target": examples[i]["target_scores"]["Yes"],
                                     "answered_by": answered_by, "provider": provider}) + "\n")
                fh.flush()
                with open(LEDGER, "a") as led:
                    led.write(json.dumps({"model": args.model, "dataset": ds, "i": i, "cost": c,
                                          "prompt_tokens": usage.get("prompt_tokens"),
                                          "completion_tokens": usage.get("completion_tokens"),
                                          "reasoning_tokens": (usage.get("completion_tokens_details")
                                                               or {}).get("reasoning_tokens"),
                                          "answered_by": answered_by})
                              + "\n")
                if total >= args.budget:
                    stop.set()

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(one, todo))
        fh.close()
        print(f"{ds}: {len(todo)} calls, ${cost_ds:.4f}, ledger ${total:.4f}, "
              f"{time.time() - t0:.0f}s", flush=True)
        if stop.is_set():
            raise SystemExit(f"budget ${args.budget:.2f} reached - stopping; rerun resumes")

        rows = sorted((json.loads(l) for l in open(answers_path)), key=lambda r: r["i"])
        rows = [r for r in rows if r["i"] < len(examples)]
        preds, unclear = zip(*[parse_answer(r["reply"]) for r in rows])
        res = prf([r["target"] for r in rows], list(preds)) | {
            "dataset": ds, "prompt": args.prompt, "model": args.model,
            "unclear_or_long_answers": sum(unclear),
            "empty_answers": sum(not r["reply"].strip() for r in rows),
            "cost_usd": round(ledger_cost(args.model, ds), 4),
            "temperature": temperature, "max_tokens": args.max_tokens}
        by = Counter(r.get("answered_by") for r in rows)
        if len(by) > 1 or next(iter(by)) != args.model:
            res["answered_by"] = dict(by.most_common())
        if args.limit is None:
            json.dump(res, open(OUT / f"result_{ds}_{args.prompt}_{tag}.json", "w"), indent=1)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
