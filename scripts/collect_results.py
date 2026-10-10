"""Collect Ditto training logs and LLM result files into one comparison table."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DITTO_PAPER = {  # Li et al., PVLDB 2021, Table 5
    "Structured_Beer": 94.37, "Structured_Fodors-Zagats": 100.0,
    "Structured_iTunes-Amazon": 97.06, "Structured_Amazon-Google": 75.58,
    "Structured_Walmart-Amazon": 86.76, "Structured_DBLP-ACM": 98.99,
    "Structured_DBLP-GoogleScholar": 95.60, "Textual_Abt-Buy": 89.33,
}
# Peeters, Steiner & Bizer, EDBT 2025, Table 2 (domain-complex-force) and Table 4 (best prompt)
LLM_PAPER = {
    "abt-buy-sampled-gs": {"gpt-4o-mini": 90.95, "gpt-4-0613": 95.15, "llama-3.1-70b_best": 89.84,
                           "ditto_same_testset": 91.31, "tailormatch_llama31_8b": 56.57},
    "walmart-amazon-sampled-gs": {"gpt-4o-mini": 86.36, "gpt-4-0613": 89.00,
                                  "llama-3.1-70b_best": 84.85, "ditto_same_testset": 86.39,
                                  "tailormatch_llama31_8b": 42.04},
    "amazon-google-sampled-gs": {"gpt-4o-mini": 70.98, "gpt-4-0613": 75.61,
                                 "llama-3.1-70b_best": 73.99, "ditto_same_testset": 80.07,
                                 "tailormatch_llama31_8b": 49.16},
    "dblp-scholar-sampled-gs": {"gpt-4o-mini": 86.11, "gpt-4-0613": 88.44,
                                "llama-3.1-70b_best": 86.32, "ditto_same_testset": 94.31,
                                "tailormatch_llama31_8b": 67.69},
    "wdcproducts-80cc-seen-sampled-250-gs-2": {"gpt-4o-mini": 80.84, "gpt-4-0613": 88.35,
                                               "llama-3.1-70b_best": 83.67,
                                               "ditto_same_testset": 84.90,
                                               "tailormatch_llama31_8b": 53.36},
}


def ditto_rows():
    rows = []
    for log in sorted((ROOT / "results/ditto").glob("*.log")):
        epochs = re.findall(r"epoch (\d+): dev_f1=([\d.]+), f1=([\d.]+), best_f1=([\d.]+)",
                            log.read_text())
        if not epochs:
            continue
        last = epochs[-1]
        rows.append({"task": log.stem, "epochs_done": int(last[0]),
                     "best_test_f1": round(100 * float(last[3]), 2),
                     "final_dev_f1": round(100 * float(last[1]), 2),
                     "paper_f1": DITTO_PAPER.get(log.stem)})
    return rows


def llm_rows():
    rows = []
    for f in sorted((ROOT / "results/llm").glob("result_*.json")):
        r = json.load(open(f))
        r["reference"] = LLM_PAPER.get(r["dataset"], {})
        rows.append(r)
    return rows


if __name__ == "__main__":
    out = {"ditto": ditto_rows(), "llm": llm_rows()}
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "results/summary.json", "w"), indent=1)
