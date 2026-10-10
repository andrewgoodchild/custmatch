"""Zero-shot entity matching with a local LLM (MLX), replaying the MatchGPT setup.

Reproduces the protocol of Peeters, Steiner & Bizer, "Entity Matching using Large Language
Models" (EDBT 2025): their downsampled test sets and prompt templates, temperature 0,
an answer counts as a match if the lowercased, de-punctuated reply contains "yes".

Usage: llm_zeroshot_mlx.py --model <mlx-repo> --prompt domain-complex-force [--limit N]
"""
import argparse
import json
import string
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "external/MatchGPT/LLMForEM/tasks"
OUT = ROOT / "research/results/llm"
DATASETS = ["abt-buy-sampled-gs", "walmart-amazon-sampled-gs", "amazon-google-sampled-gs",
            "dblp-scholar-sampled-gs", "wdcproducts-80cc-seen-sampled-250-gs-2"]


def parse_answer(text):
    """MatchGPT's parsing: strip punctuation, lowercase, look for yes/no."""
    p = text.strip().translate(str.maketrans("", "", string.punctuation)).lower()
    if "yes" in p:
        return 1, p not in ("yes", "no")
    if "no" in p:
        return 0, p not in ("yes", "no")
    return 0, True  # unclear -> counted as non-match


def prf(truth, pred):
    tp = sum(t and p for t, p in zip(truth, pred))
    fp = sum((not t) and p for t, p in zip(truth, pred))
    fn = sum(t and (not p) for t, p in zip(truth, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = sum(t == p for t, p in zip(truth, pred)) / len(truth)
    return {"precision": round(100 * prec, 2), "recall": round(100 * rec, 2),
            "f1": round(100 * f1, 2), "accuracy": round(100 * acc, 2),
            "n": len(truth), "predicted_matches": tp + fp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    ap.add_argument("--prompt", default="domain-complex-force")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=12)
    args = ap.parse_args()

    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)  # greedy = temperature 0
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.model.split("/")[-1]

    for ds in args.datasets:
        task = json.load(open(TASKS / f"{ds}_{args.prompt}.json"))
        examples = task["examples"][: args.limit]
        answers_path = OUT / f"answers_{ds}_{args.prompt}_{tag}.jsonl"
        done = []
        if answers_path.exists():
            done = [json.loads(l) for l in open(answers_path)]
        fh = open(answers_path, "a")
        t0 = time.time()
        for i, ex in enumerate(examples):
            if i < len(done):
                continue
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": task["task_prefix"] + ex["input"]}],
                add_generation_prompt=True)
            reply = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                             sampler=sampler, verbose=False)
            fh.write(json.dumps({"i": i, "reply": reply,
                                 "target": ex["target_scores"]["Yes"]}) + "\n")
            fh.flush()
            if i % 100 == 0:
                rate = (i - len(done) + 1) / max(time.time() - t0, 1e-9)
                print(f"{ds} {i}/{len(examples)} {rate:.2f} ex/s", flush=True)
        fh.close()

        rows = [json.loads(l) for l in open(answers_path)][: len(examples)]
        preds, unclear = zip(*[parse_answer(r["reply"]) for r in rows])
        res = prf([r["target"] for r in rows], list(preds)) | {
            "dataset": ds, "prompt": args.prompt, "model": args.model,
            "unclear_or_long_answers": sum(unclear),
            "seconds": round(time.time() - t0, 1)}
        json.dump(res, open(OUT / f"result_{ds}_{args.prompt}_{tag}.json", "w"), indent=1)
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
