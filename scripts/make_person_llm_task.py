"""Build an LLM matching task from person records, in the MatchGPT task format.

Takes the hardest candidate pairs from historical_50k (high string similarity, mixed labels),
samples a test set with a realistic positive rate, and writes a task JSON that
`llm_zeroshot_mlx.py` can consume - so the same prompt and parsing used for the product
benchmarks is applied to customer-style person records.
"""
import json
import random
from pathlib import Path

from rapidfuzz.distance import JaroWinkler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external/MatchGPT/LLMForEM/tasks"
FIELDS = ["full_name", "dob", "birth_place", "postcode_fake", "occupation"]
N_POS, N_NEG, SEED = 250, 1000, 42

PREFIX = ("Do the two person records refer to the same real-world person? "
          "Answer with 'Yes' if they do and 'No' if they do not.\n")


def serialise(row):
    return " ".join(str(row[f]) for f in FIELDS if str(row[f]).strip())


def main():
    from splink import splink_datasets
    df = splink_datasets.historical_50k.reset_index(drop=True)
    for c in FIELDS + ["first_name", "surname"]:
        df[c] = df[c].fillna("").astype(str).str.lower().str.strip()

    rng = random.Random(SEED)
    rows = df.to_dict("records")
    by_cluster = {}
    for i, r in enumerate(rows):
        by_cluster.setdefault(r["cluster"], []).append(i)

    # positives: same person, but the two records differ textually
    pos = []
    for ids in by_cluster.values():
        if len(ids) > 1:
            a, b = rng.sample(ids, 2)
            if rows[a]["full_name"] != rows[b]["full_name"]:
                pos.append((a, b))
    rng.shuffle(pos)
    pos = pos[:N_POS]

    # negatives: different people who share a surname or a birth year -> genuinely hard
    by_surname = {}
    for i, r in enumerate(rows):
        by_surname.setdefault(r["surname"], []).append(i)
    neg = []
    surnames = [s for s, ids in by_surname.items() if len(ids) > 1 and s]
    rng.shuffle(surnames)
    for s in surnames:
        ids = by_surname[s]
        for _ in range(3):
            a, b = rng.sample(ids, 2)
            if rows[a]["cluster"] != rows[b]["cluster"]:
                sim = JaroWinkler.similarity(rows[a]["full_name"], rows[b]["full_name"])
                if sim > 0.75:
                    neg.append((a, b))
                    break
        if len(neg) >= N_NEG:
            break
    neg = neg[:N_NEG]

    examples = [{"input": f"Record 1: '{serialise(rows[a])}'\nRecord 2: '{serialise(rows[b])}'",
                 "target_scores": {"Yes": int(lab), "No": int(not lab)}}
                for lab, pairs in ((1, pos), (0, neg)) for a, b in pairs]
    rng.shuffle(examples)

    task = {"name": "Entity matching - historical50k-person-hard",
            "description": "Person record matching on historical_50k: hard pairs "
                           "(positives with differing names; negatives sharing a surname "
                           "and with high name similarity).",
            "task_prefix": PREFIX,
            "metrics": ["multiple_choice_grade"],
            "examples": examples}
    path = OUT / "historical50k-person-hard_domain-complex-force.json"
    json.dump(task, open(path, "w"), indent=1)
    print(f"wrote {path} with {len(examples)} examples "
          f"({sum(e['target_scores']['Yes'] for e in examples)} positive)")


if __name__ == "__main__":
    main()
