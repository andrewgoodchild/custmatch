"""Golden-record test set: does the MERGED record say the right thing?

Linkage benchmarks only score which records belong together. That leaves the second half of
customer matching untested - given a cluster of noisy records, what should the surviving
"golden" record contain? pseudopeople can generate the same population with noise disabled,
which gives the true value of every field for every person, so survivorship can be scored
field by field.

Reported two ways:
  predicted clusters - end to end, so linkage errors and survivorship errors are mixed
  oracle clusters    - perfect linkage handed in, isolating survivorship on its own
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("future.infer_string", False)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/customer"

FIELDS = ["first_name", "last_name", "addr_number", "addr_street"]
from custmatch.golden import STRATEGIES  # noqa: E402
from custmatch.golden import build_golden as _build_golden, survive as _survive  # noqa: E402
SOURCE_RANK = {"crm": 0, "billing": 1}          # census is the tidier source for names


def truth_table():
    """Noise-free values, one row per person."""
    import pseudopeople as psp
    census = psp.generate_decennial_census(year=2020, config=psp.NO_NOISE)
    w2 = psp.generate_taxes_w2_and_1099(year=2020, config=psp.NO_NOISE)
    a = census.rename(columns={"street_number": "addr_number", "street_name": "addr_street"})
    b = w2.rename(columns={"mailing_address_street_number": "addr_number",
                           "mailing_address_street_name": "addr_street"})
    cols = ["simulant_id"] + FIELDS
    truth = pd.concat([a[cols], b[cols]]).dropna(subset=["simulant_id"])
    for f in FIELDS:
        truth[f] = truth[f].fillna("").astype(str).str.lower().str.strip()
    # a person's true values are identical across sources apart from mailing vs residential
    # address, so take the first census row where available
    return truth.drop_duplicates("simulant_id").set_index("simulant_id")


def survive(values, sources, strategy):
    return _survive(values, sources, strategy, SOURCE_RANK)


def build_golden(df, labels, strategy):
    return _build_golden(df, labels, strategy, FIELDS, SOURCE_RANK)


def score(df, labels, golden, truth):
    """A cluster is scored against the true values of its majority person."""
    per_field = {f: [0, 0] for f in FIELDS}
    perfect = total = 0
    sid = df.simulant_id.to_numpy()
    order = np.argsort(labels, kind="stable")
    lab_sorted = labels[order]
    for idx in np.split(order, np.flatnonzero(np.diff(lab_sorted)) + 1):
        people = Counter(sid[idx])
        person = people.most_common(1)[0][0]
        if person not in truth.index:
            continue
        want = truth.loc[person]
        got = golden[labels[idx[0]]]
        ok_all = True
        for f in FIELDS:
            per_field[f][1] += 1
            if str(got[f]).strip() == str(want[f]).strip():
                per_field[f][0] += 1
            else:
                ok_all = False
        perfect += ok_all
        total += 1
    return {"per_field": {f: round(100 * c / n, 2) for f, (c, n) in per_field.items()},
            "all_fields_correct": round(100 * perfect / total, 2), "clusters_scored": total}


def main():
    a = pd.read_parquet(DATA / "source_crm.parquet")
    b = pd.read_parquet(DATA / "source_billing.parquet")
    df = pd.concat([a, b], ignore_index=True)
    for f in FIELDS:
        df[f] = df[f].fillna("").astype(str).str.lower().str.strip()

    clusters = pd.read_parquet(ROOT / "research/results/customer_clusters.parquet")
    assert list(clusters.record_id) == list(df.record_id), "cluster file is out of sync"
    predicted = clusters.cluster.to_numpy()
    _, oracle = np.unique(df.simulant_id.to_numpy(), return_inverse=True)

    truth = truth_table()
    print(f"golden-record truth for {len(truth):,} people; "
          f"{len(df):,} records, {len(set(predicted)):,} predicted clusters", flush=True)

    # how wrong are the raw records to begin with?
    baseline = {}
    for f in FIELDS:
        have = df.simulant_id.isin(truth.index)
        sub = df[have]
        correct = (sub[f].to_numpy() ==
                   truth.loc[sub.simulant_id, f].to_numpy()).mean()
        baseline[f] = round(100 * correct, 2)
    print(f"raw source records correct per field: {baseline}", flush=True)

    # Ceiling: can ANY record in the cluster supply the true value? If not, no survivorship
    # rule can recover it - the information simply is not in the data.
    ceiling = {}
    sid_arr = df.simulant_id.to_numpy()
    for name, labels in (("predicted", predicted), ("oracle", oracle)):
        hits = {f: [0, 0] for f in FIELDS}
        order = np.argsort(labels, kind="stable")
        for idx in np.split(order, np.flatnonzero(np.diff(labels[order])) + 1):
            person = Counter(sid_arr[idx]).most_common(1)[0][0]
            if person not in truth.index:
                continue
            want = truth.loc[person]
            for f in FIELDS:
                hits[f][1] += 1
                if str(want[f]).strip() in {str(v).strip() for v in df[f].to_numpy()[idx]}:
                    hits[f][0] += 1
        ceiling[name] = {f: round(100 * c / n, 2) for f, (c, n) in hits.items()}
    print(f"\nceiling - true value present in at least one record of the cluster:")
    for k, v in ceiling.items():
        print(f"  {k:10s} {v}", flush=True)

    out = {"baseline_raw_records": baseline, "value_available_ceiling": ceiling,
           "note_address_truth": "truth uses the census (residential) address, so rules that "
                                 "prefer the CRM source are favoured on address fields; the "
                                 "name fields are the clean comparison",
           "note_first_seen": "first_seen and source_priority coincide because CRM rows are "
                              "concatenated first",
           "predicted": {}, "oracle": {}}
    for name, labels in (("predicted", predicted), ("oracle", oracle)):
        for strategy in STRATEGIES:
            golden = build_golden(df, labels, strategy)
            out[name][strategy] = score(df, labels, golden, truth)
        best = max(out[name], key=lambda s: out[name][s]["all_fields_correct"])
        print(f"\n{name} clusters:")
        for s in STRATEGIES:
            r = out[name][s]
            marker = " <-- best" if s == best else ""
            print(f"  {s:16s} all-fields {r['all_fields_correct']:6.2f}%  "
                  f"per-field {r['per_field']}{marker}", flush=True)

    json.dump(out, open(ROOT / "research/results/golden_record.json", "w"), indent=1)


if __name__ == "__main__":
    main()
