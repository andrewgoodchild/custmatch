"""Could a local LLM label pairs, or judge them in the review queue? Accuracy on easy and hard pairs.

The hosted labeller (Jev, docs/report/04-labels.md) helped only where its labels were near-perfect
(98.9% on the customer file: within half a point of hand labels; 91.3% on historical_50k: about
ten points lost), and sent records to a third party. A model running on the laptop keeps the
records local; the question left is accuracy, above all on the pairs that matter.

For each of the two datasets the Jev experiment used, the training pairs are scored by the
forest out of fold (4 folds) and two samples drawn:

  random   500 training pairs at random - what labelling from scratch would ask
  hard     the 500 pairs the forest is least sure of (P nearest 0.5) - what active learning and
           the review queue put in front of a steward

Each pair goes to a local model through Ollama (http://localhost:11434, temperature 0, no
reasoning trace) with Jev's question, placeholder note and written criteria, and the records as
JSON; it answers same person or not, with a confidence and a one-line reason. Scored against the
truth: accuracy, precision and recall on each sample, and the forest's own accuracy on the same
pairs for comparison. Nothing leaves the machine.

    python research/scripts/local_llm_labels.py [customer] [historical] [--model=qwen3:8b] [--n=500]

Writes research/results/local_llm_labels.json (answers in research/results/llm/, gitignored).
"""
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_labelling import CRITERIA, PLACEHOLDERS, QUESTION  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/local_llm_labels.json"
ANSWERS = ROOT / "research/results/llm"
OLLAMA = "http://localhost:11434/api/chat"
SYSTEM = (f"You compare two person records. {QUESTION} {PLACEHOLDERS}\n"
          f"Answer true when: {CRITERIA['true']}\nAnswer false when: {CRITERIA['false']}\n"
          "Reply in JSON; keep the reason to 12 words or fewer.")
SCHEMA = {"type": "object", "properties": {"same_person": {"type": "boolean"},
                                           "confidence": {"type": "number"},
                                           "reason": {"type": "string"}},
          "required": ["same_person", "confidence", "reason"]}


def customer():
    from customer_pipeline import prepare
    P = prepare(aware=True)
    df, pairs, X, y, tr = (P[k] for k in ("df", "pairs", "X", "y", "tr"))
    df = df.copy()
    df["address"] = (df.addr_number.astype(str) + " " + df.addr_street.astype(str) + " "
                     + df.addr_unit.astype(str)).str.split().str.join(" ")
    keys = ["first_name", "middle_initial", "last_name", "date_of_birth", "address", "email", "phone"]
    rows = [{k: str(r[k]) for k in keys if str(r[k]).strip() and str(r[k]) != "nan"} for r in df.to_dict("records")]
    idx = np.flatnonzero(tr)
    return X[idx], y[idx].astype(bool), [(rows[a], rows[b]) for a, b in pairs[idx]]


def historical():
    from jev_cascade import records, structured
    tr = dict(np.load(ROOT / "data/person_em/train.npz"))
    rows = records()
    pairs = [(structured(rows[int(a)]), structured(rows[int(b)])) for a, b in zip(tr["left"], tr["right"])]
    return tr["X"], tr["y"].astype(bool), pairs


def samples(X, y, n, seed=0):
    p = cross_val_predict(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
                          X, y, cv=4, method="predict_proba")[:, 1]
    rng = np.random.default_rng(seed)
    rand = rng.choice(len(y), n, replace=False)
    hard = np.argsort(np.abs(p - 0.5))[:n]
    return {"random": rand, "hard": hard}, p


def ask(client, model, a, b):
    msg = f"record_1: {json.dumps(a)}\nrecord_2: {json.dumps(b)}\nDo they refer to the same real-world person?"
    for attempt in range(4):
        try:
            r = client.post(OLLAMA, json={"model": model, "stream": False, "think": False, "format": SCHEMA,
                                          "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 120},
                                          "messages": [{"role": "system", "content": SYSTEM},
                                                       {"role": "user", "content": msg}]})
            r.raise_for_status()
            return json.loads(r.json()["message"]["content"])
        except (httpx.HTTPError, json.JSONDecodeError, KeyError):
            time.sleep(2 ** attempt)
    return None


def quality(y, pred):
    ok = pred is not None
    y, pred = np.asarray(y), np.asarray(pred)
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    return {"pairs": int(len(y)), "matches": int(y.sum()), "accuracy": round(100 * float((pred == y).mean()), 2),
            "precision": round(100 * tp / max(tp + fp, 1), 2), "recall": round(100 * tp / max(tp + fn, 1), 2)} if ok else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    model, n = opt.get("model", "qwen3:8b"), int(opt.get("n", 500))
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    ANSWERS.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=300) as client:
        for name in args or ["customer", "historical"]:
            X, y, pairs = {"customer": customer, "historical": historical}[name]()
            picks, p_forest = samples(X, y, n)
            path = ANSWERS / f"answers_local_{model.replace(':', '_')}_{name}.jsonl"
            done = {json.loads(l)["i"]: json.loads(l) for l in open(path)} if path.exists() else {}
            r = res.setdefault(model, {}).setdefault(name, {})
            t0, asked = time.time(), 0
            with open(path, "a") as fh:
                for i in sorted(set(np.concatenate(list(picks.values())).tolist())):
                    if i in done:
                        continue
                    a = ask(client, model, *pairs[i])
                    done[i] = {"i": i, "answer": a}
                    fh.write(json.dumps(done[i]) + "\n")
                    fh.flush()
                    asked += 1
                    if asked % 50 == 0:
                        print(f"{name}: {asked} asked, {(time.time() - t0) / asked:.2f} s a pair", flush=True)
            if asked:
                r["seconds_per_pair"] = round((time.time() - t0) / asked, 2)
            for kind, ix in picks.items():
                answered = [i for i in ix if done[i]["answer"] is not None]
                pred = np.array([bool(done[i]["answer"]["same_person"]) for i in answered])
                r[kind] = {"llm": quality(y[answered], pred),
                           "forest": quality(y[ix], p_forest[ix] > 0.5),
                           "unanswered": int(len(ix) - len(answered))}
                conf = np.array([float(done[i]["answer"].get("confidence", 0)) for i in answered])
                sure = conf >= 0.9
                r[kind]["llm_confident"] = quality(y[answered][sure], pred[sure]) | {"share": round(100 * float(sure.mean()), 1)}
                print(name, kind, json.dumps(r[kind]), flush=True)
            OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
