"""Monitoring across runs: each `run` and `add` writes monitor.json, the same measures every time;
`custmatch monitor` lines several outputs up and flags drift between consecutive ones.

Measures: how many records are matched (in a customer with others), customer sizes, how full
each field is, per source, and the spread of match scores. Drift checks, consecutive runs:

  match rate      moved by more than MATCH_RATE_POINTS percentage points
  field fill      a field's fill rate moved by more than FILL_POINTS points (a feed stopped
                  sending a column, or a mapping broke)
  scores          the score distribution moved: population stability index above PSI_ALERT
                  (0.1-0.2 is the usual "watch", above 0.2 "act")
  largest         the largest customer grew by more than LARGEST_GROWTH times (chaining)
  threshold       the threshold changed (a retrained model)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BINS = np.linspace(0, 1, 11)
MATCH_RATE_POINTS = 5.0
FILL_POINTS = 10.0
PSI_ALERT = 0.2
LARGEST_GROWTH = 2.0
SKIP = {"source", "record_id"}


def snapshot(df, customer, score, th):
    """The measures for one output. customer: an ID per record; score: the candidate pairs' scores."""
    customer = np.asarray(customer)
    _, inv, counts = np.unique(customer, return_inverse=True, return_counts=True)
    matched = counts[inv] > 1
    fields = [c for c in df.columns if c not in SKIP and not c.startswith("_")
              and (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))]
    fill = {c: round(100 * float((df[c].astype(str) != "").mean()), 2) for c in fields}
    by_source = {s: round(100 * float(matched[(df.source == s).to_numpy()].mean()), 2)
                 for s in sorted(df.source.unique())}
    hist = np.histogram(np.clip(score, 0, 1), BINS)[0] if len(score) else np.zeros(len(BINS) - 1, int)
    return {"records": int(len(df)), "customers": int(len(counts)),
            "match_rate": round(100 * float(matched.mean()), 2) if len(df) else 0.0,
            "match_rate_by_source": by_source,
            "mean_customer_size": round(float(counts.mean()), 3) if len(counts) else 0.0,
            "largest_customer": int(counts.max()) if len(counts) else 0,
            "threshold": float(th),
            "pairs_scored": int(len(score)),
            "pairs_near_threshold": int(np.sum(np.abs(np.asarray(score) - th) < 0.1)),
            "score_histogram": hist.tolist(),
            "fill_rate": fill}


def write(out, snap):
    (Path(out) / "monitor.json").write_text(json.dumps(snap, indent=1))


def psi(a, b):
    """Population stability index between two histograms (counts over the same bins)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    p, q = (a + 0.5) / (a + 0.5).sum(), (b + 0.5) / (b + 0.5).sum()   # +0.5: empty bins
    return float(np.sum((q - p) * np.log(q / p)))


def compare(dirs):
    """A row per output (in the order given) and the drift alerts between consecutive ones."""
    rows, alerts, prev = [], [], None
    for d in dirs:
        path = Path(d) / "monitor.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - outputs from before monitoring have none")
        m = json.loads(path.read_text())
        row = {"run": str(d), "records": m["records"], "customers": m["customers"],
               "match_rate": m["match_rate"], "largest_customer": m["largest_customer"],
               "threshold": m["threshold"], "pairs_near_threshold": m["pairs_near_threshold"],
               "score_psi": None}
        if prev is not None:
            row["score_psi"] = round(psi(prev["score_histogram"], m["score_histogram"]), 4)
            where = f"{rows[-1]['run']} -> {d}"
            if abs(m["match_rate"] - prev["match_rate"]) > MATCH_RATE_POINTS:
                alerts.append(f"{where}: match rate {prev['match_rate']}% -> {m['match_rate']}%")
            for s, v in m["match_rate_by_source"].items():
                was = prev["match_rate_by_source"].get(s)
                if was is not None and abs(v - was) > MATCH_RATE_POINTS:
                    alerts.append(f"{where}: match rate of source {s} {was}% -> {v}%")
            for f, v in m["fill_rate"].items():
                was = prev["fill_rate"].get(f)
                if was is not None and abs(v - was) > FILL_POINTS:
                    alerts.append(f"{where}: {f} filled on {was}% -> {v}% of records")
            if row["score_psi"] > PSI_ALERT:
                alerts.append(f"{where}: score distribution shifted (PSI {row['score_psi']})")
            if prev["largest_customer"] and m["largest_customer"] > LARGEST_GROWTH * prev["largest_customer"] \
                    and m["largest_customer"] > 5:
                alerts.append(f"{where}: largest customer {prev['largest_customer']} -> "
                              f"{m['largest_customer']} records - check for chaining")
            if m["threshold"] != prev["threshold"]:
                alerts.append(f"{where}: threshold {prev['threshold']:.3f} -> {m['threshold']:.3f} "
                              "(a retrained model; expect IDs to move)")
        rows.append(row)
        prev = m
    return pd.DataFrame(rows), alerts
