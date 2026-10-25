"""Does `custmatch audit` tell the truth? Its estimates and 95% intervals against the known truth.

One run on the synthetic customer file (1,000 random labels, seed 1, as the other synthetic
benchmarks). The truth over its candidate pairs is known: which pairs are one person, which the
run linked. Then 100 audit samples of 300 pairs each (seeds 0-99), labelled from the truth, and
for each: the estimates, the intervals, and whether each interval holds the true value. A 95%
interval should hold it about 95 times in 100. For comparison, the same samples with the two
intervals custmatch does not use: the labels resampled within their strata, and a Beta interval
per stratum.

    python research/scripts/audit_check.py

Writes research/results/audit_check.json.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import custmatch_validate as V  # noqa: E402
from custmatch import audit, io, labels, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SAMPLES, N = 100, 300


def other_intervals(state, path, draws=2000, seed=0):
    """The two intervals custmatch does not use, on the same sample: the labels resampled within
    each stratum (no scores), and a Beta(m + 0.5, n - m + 0.5) match rate per stratum."""
    _, _, meta = audit._state(state)
    lab = pd.read_csv(path, dtype=str, keep_default_na=False)
    lab = lab.assign(stratum=lab.by.str[6:], match=lab.label == "1")
    groups = [(meta["pairs"][s], g.match.to_numpy(float), s.endswith("L")) for s, g in lab.groupby("stratum")]
    size = np.array([t for t, _, _ in groups], float)
    linked = np.array([li for _, _, li in groups])
    rng = np.random.default_rng(seed)

    def prf(q):
        tp = (size * q * linked).sum(axis=-1)
        p, r = tp / (size * linked).sum(), tp / (size * q).sum(axis=-1)
        return np.column_stack([p, r, 2 * p * r / (p + r)]) * 100
    plain = prf(np.column_stack([m[rng.integers(0, len(m), (draws, len(m)))].mean(axis=1) for _, m, _ in groups]))
    n = np.array([len(m) for _, m, _ in groups], float)
    k = np.array([m.sum() for _, m, _ in groups])
    beta = prf(rng.beta(k + 0.5, n - k + 0.5, size=(draws, len(groups))))
    return {name: (np.nanpercentile(x, 2.5, axis=0), np.nanpercentile(x, 97.5, axis=0))
            for name, x in (("plain_bootstrap", plain), ("beta", beta))}


def main():
    gold = V.truth()
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = json.loads(V.CONFIG.read_text())
        for s in cfg["sources"]:
            s["path"] = str((V.CONFIG.parent / s["path"]).resolve())
        (tmp / "c.json").write_text(json.dumps(cfg))
        labels.sample(tmp / "c.json", 1000, tmp / "l.csv", seed=1, stratified=False)
        V.fill(tmp / "l.csv", gold)
        pipeline.run(tmp / "c.json", tmp / "l.csv", tmp / "run", log=lambda *_: None)

        # the truth over the run's candidate pairs (its audit pool's totals count the same pairs)
        df, cfg = io.load(tmp / "c.json")
        keys = io.key(df)
        pairs = labels.candidates(df, cfg)
        c = pd.read_csv(tmp / "run" / "clusters.csv", dtype=str)
        cust = dict(zip(c.source + ":" + c.record_id, c.customer_id))
        a, b = keys[pairs[:, 0]], keys[pairs[:, 1]]
        match = np.array([gold[x][0] == gold[y][0] for x, y in zip(a, b)])
        linked = np.array([cust[x] == cust[y] for x, y in zip(a, b)])
        tp = (match & linked).sum()
        p, r = tp / linked.sum(), tp / match.sum()
        truth = {"precision": 100 * p, "recall": 100 * r, "f1": 100 * 2 * p * r / (p + r)}
        out["candidate_pairs"] = int(len(pairs))
        out["truth"] = {k: round(v, 2) for k, v in truth.items()}
        print("truth", out["truth"], "over", len(pairs), "candidate pairs", flush=True)

        rows = []
        for seed in range(SAMPLES):
            path = tmp / f"a{seed}.csv"
            audit.sample(tmp / "run", N, path, seed=seed)
            V.fill(path, gold)
            e = audit.estimate(tmp / "run", path, seed=seed)
            other = other_intervals(tmp / "run", path, seed=seed)
            rows.append({k: (e[k]["estimate"], *e[k]["interval_95"], *(v[j][i] for v in other.values() for j in (0, 1)))
                         for i, k in enumerate(("precision", "recall", "f1"))})
        out["samples"], out["pairs_per_sample"] = SAMPLES, N
        for k in ("precision", "recall", "f1"):
            est = np.array([row[k] for row in rows])
            covered = (est[:, 1] <= truth[k]) & (truth[k] <= est[:, 2])
            out[k] = {"mean_estimate": round(float(est[:, 0].mean()), 2),
                      "sd_estimate": round(float(est[:, 0].std()), 2),
                      "mean_interval_width": round(float((est[:, 2] - est[:, 1]).mean()), 2),
                      "intervals_holding_the_truth": int(covered.sum()),
                      "not_used": {name: {"holding_the_truth": int(((est[:, c] <= truth[k]) & (truth[k] <= est[:, c + 1])).sum()),
                                          "mean_width": round(float((est[:, c + 1] - est[:, c]).mean()), 2)}
                                   for name, c in (("plain_bootstrap", 3), ("beta", 5))}}
            print(k, out[k], flush=True)
    (ROOT / "research/results/audit_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
