"""How many of fake_1000's true pairs the default keys compare, with and without date of birth
alone as a key ("dob_key", schema.with_dob_key): the share of true pairs among the candidates,
and the share any matcher could reach by accepting every true candidate and closing transitively.

    python research/scripts/dob_key_blocking.py
Writes research/results/dob_key.json. The matching results are in home_data_benchmark.json
(fake_1000, historical_50k, febrl3, febrl4), au_validation*.json (the generated Australian files
at four sizes), beat_splink.json (rich_dob_key) and bpid.json (all_addresses_dob_key).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import splink_datasets_benchmark as S  # noqa: E402
from custmatch import blocking, io  # noqa: E402
from custmatch.core import components, profile_contacts  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/results/dob_key.json"


def pairs_within(labels):
    _, c = np.unique(labels, return_counts=True)
    return int((c * (c - 1) // 2).sum())


def main():
    spec = S.datasets()["fake_1000"]
    fake = spec["frames"]["fake"]
    truth_of = dict(zip("fake:" + fake.unique_id.astype(str), fake.cluster.astype(str)))
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake.fillna("").astype(str).to_csv(tmp / "fake.csv", index=False)
        base = {"sources": [{"name": "fake", "path": "fake.csv", "columns": spec["columns"]}],
                "extra_fields": spec["extra_fields"], "extra_blocking": spec["extra_blocking"]}
        for tag, on in (("without", False), ("with", True)):
            p = tmp / "config.json"
            p.write_text(json.dumps(base | {"dob_key": on}))
            df, cfg = io.load(p)
            pid = np.array([truth_of[k] for k in io.key(df)])
            profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
            pairs, *_ = blocking.candidates_for(df, profiles, cfg)
            true = pairs[pid[pairs[:, 0]] == pid[pairs[:, 1]]]
            comp = components(len(df), true, np.ones(len(true), bool))
            # pairs of one person in one component: what accepting every true candidate reaches
            reach = pairs_within([f"{a}|{b}" for a, b in zip(pid, comp)])
            n_true = pairs_within(pid)
            out[tag] = {"candidate_pairs": int(len(pairs)),
                        "true_pairs_compared_pct": round(100 * len(true) / n_true, 1),
                        "reachable_after_closure_pct": round(100 * reach / n_true, 1)}
            if not on:                                   # what the default keys miss
                have = set(map(tuple, pairs.tolist()))
                missed = [(a, b) for g in np.unique(pid) for ix in [np.flatnonzero(pid == g)]
                          for k, a in enumerate(ix) for b in ix[k + 1:] if (a, b) not in have]
                fn, ln = df.first_name.to_numpy(), df.last_name.to_numpy()
                out["missed_by_default_keys"] = {
                    "true_pairs": len(missed),
                    "both_names_differ": sum(bool(fn[a] != fn[b] and ln[a] != ln[b]) for a, b in missed),
                    "a_name_blank": sum(not (fn[a] and fn[b] and ln[a] and ln[b]) for a, b in missed)}
            print(tag, out[tag], flush=True)
    OUT.write_text(json.dumps({"fake_1000": out}, indent=1))


if __name__ == "__main__":
    main()
