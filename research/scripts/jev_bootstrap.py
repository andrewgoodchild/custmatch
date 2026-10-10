"""Paired bootstrap for the Jev variant comparisons quoted in docs/report/10-reproductions.md.

Resamples the same test pairs 2,000 times and reports the change against the baseline call (F1 at
the 0.5 cut-off, and AUC) with a 95% percentile interval. Reads the per-pair answers in
research/results/llm/ (gitignored: rebuild with research/scripts/jev_decisions.py and research/scripts/jev_variants.py; no
API calls here). Writes research/results/jev_bootstrap.json.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
LLM = ROOT / "research/results/llm"
TASKS = ["historical50k-person-hard", "customer-person-hard"]
VARIANTS = ["criteria", "structured", "structured_criteria"]


def answers(path, key):
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["i"]] = r
    return out


def f1(y, p):
    pred = p > 0.5
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    return 200 * tp / (2 * tp + fp + fn) if tp else 0.0


def main(n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for task in TASKS:
        base = answers(LLM / f"answers_{task}_domain-complex-force_jev-1.13.jsonl", "p_yes")
        for v in VARIANTS:
            path = LLM / f"answers_jevvar_{task}_{v}.jsonl"
            if not path.exists():
                continue
            var = answers(path, "answers")
            ids = sorted(set(base) & set(var))
            y = np.array([base[i]["target"] == 1 for i in ids])
            pb = np.array([base[i]["p_yes"] for i in ids])
            pv = np.array([var[i]["answers"]["match"]["noul"] for i in ids])
            d_f1, d_auc = [], []
            for _ in range(n_boot):
                k = rng.integers(0, len(ids), len(ids))
                if y[k].all() or not y[k].any():
                    continue
                d_f1.append(f1(y[k], pv[k]) - f1(y[k], pb[k]))
                d_auc.append(roc_auc_score(y[k], pv[k]) - roc_auc_score(y[k], pb[k]))
            res[f"{task}/{v}"] = {
                "pairs": len(ids),
                "f1_at_0.5_change": round(f1(y, pv) - f1(y, pb), 2),
                "f1_interval": [round(float(np.percentile(d_f1, q)), 2) for q in (2.5, 97.5)],
                "auc_change": round(roc_auc_score(y, pv) - roc_auc_score(y, pb), 4),
                "auc_interval": [round(float(np.percentile(d_auc, q)), 4) for q in (2.5, 97.5)]}
            print(f"{task:28s} {v:20s}", res[f"{task}/{v}"], flush=True)
    (ROOT / "research/results/jev_bootstrap.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
