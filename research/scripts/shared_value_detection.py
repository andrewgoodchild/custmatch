"""Can shared-household values and junk/placeholder values be detected without ground truth?

Both look the same at first glance - one value carried by many records - but they separate on
the DIVERSITY of the records carrying it:

  household  : shared by a few records that agree on other fields (same surname, same place)
  junk/fake  : shared by many records that agree on nothing (test@test.com, 555-1234, 000000)
  identifier : shared by ~1-2 records, high agreement (a real SSN, a real personal email)

The detector below uses only value frequency and co-field diversity - no labels. Ground truth
is then used ONLY to check whether the flags were right.

Also injects synthetic fake emails/phones into FEBRL, since neither public dataset has those
fields, to check the detector fires on the real thing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def profile(df, field, entity_col, co_fields):
    """Per-value: how many records, how many distinct entities, and how diverse the co-fields."""
    rows = []
    for val, g in df.groupby(field):
        if not str(val).strip():
            continue
        rows.append({
            "value": val,
            "records": len(g),
            "entities": g[entity_col].nunique(),
            **{f"distinct_{c}": g[c].nunique() for c in co_fields},
        })
    p = pd.DataFrame(rows)
    # diversity: distinct surnames per record sharing the value (1.0 = everyone different)
    p["diversity"] = p[f"distinct_{co_fields[0]}"] / p["records"]
    return p.sort_values("records", ascending=False)


def pair_purity(df, field, entity_col, values):
    """Of the record pairs that share this value, what fraction are the same entity?"""
    out = {}
    for v in values:
        g = df[df[field] == v]
        ents = g[entity_col].value_counts().to_numpy()
        same = float(np.sum(ents * (ents - 1) / 2))
        total = len(g) * (len(g) - 1) / 2
        out[v] = round(100 * same / total, 2) if total else None
    return out


def main():
    res = {}
    from splink import splink_datasets
    h = splink_datasets.historical_50k.reset_index(drop=True)
    for c in ["surname", "first_name", "dob", "postcode_fake", "birth_place", "occupation"]:
        h[c] = h[c].fillna("").astype(str).str.lower().str.strip()

    print("=== historical_50k: postcode_fake (the closest thing here to a shared address)")
    p = profile(h, "postcode_fake", "cluster", ["surname", "dob"])
    print(f"values={len(p):,}  median records/value={p.records.median():.0f}  "
          f"max={p.records.max():,}")
    top = p.head(8)
    purity = pair_purity(h, "postcode_fake", "cluster", top.value.tolist())
    print(f"{'value':>12}{'recs':>7}{'entities':>9}{'surnames':>9}{'diversity':>10}{'same-entity %':>14}")
    for _, r in top.iterrows():
        print(f"{r.value:>12}{r.records:>7}{r.entities:>9}{r.distinct_surname:>9}"
              f"{r.diversity:>10.2f}{purity[r.value]:>14}")
    res["historical_50k_postcode"] = {
        "values": int(len(p)), "max_records_per_value": int(p.records.max()),
        "share_of_values_with_1_entity": round(100 * float((p.entities == 1).mean()), 2),
        "top": top.assign(same_entity_pct=[purity[v] for v in top.value]).to_dict("records"),
    }

    # FEBRL: address as a household-ish key, soc_sec_id as a personal identifier
    from recordlinkage.datasets import load_febrl4
    a, b, _ = load_febrl4(return_links=True)
    f = pd.concat([a, b]).reset_index()
    for c in ["given_name", "surname", "street_number", "address_1", "suburb",
              "postcode", "date_of_birth", "soc_sec_id"]:
        f[c] = f[c].astype(str).str.lower().str.strip()
    f["entity"] = f.rec_id.str.split("-").str[1]
    f["address"] = f.street_number + " " + f.address_1 + " " + f.suburb

    print("\n=== FEBRL4: address vs soc_sec_id")
    for field in ["address", "soc_sec_id"]:
        p = profile(f, field, "entity", ["surname", "date_of_birth"])
        multi = p[p.records > 1]
        shared_diff_entity = p[p.entities > 1]
        print(f"{field:>12}: values={len(p):,}  shared_by>1_record={len(multi):,}  "
              f"shared_across_different_people={len(shared_diff_entity):,}")
        res[f"febrl_{field}"] = {"values": int(len(p)), "shared_values": int(len(multi)),
                                 "shared_across_entities": int(len(shared_diff_entity))}

    # inject junk values, since neither dataset ships email/phone
    print("\n=== injected fake email / phone into FEBRL (neither dataset has these fields)")
    rng = np.random.default_rng(0)
    n = len(f)
    email = np.array([f"user{i}@mail.com" for i in range(n)], dtype=object)
    phone = np.array([f"04{rng.integers(10**7, 10**8 - 1)}" for _ in range(n)], dtype=object)
    junk_email, junk_phone = "test@test.com", "0000000000"
    # 4% of records get the placeholder, spread over unrelated people
    idx = rng.choice(n, int(0.04 * n), replace=False)
    email[idx], phone[idx] = junk_email, junk_phone
    # 1.5% get a genuine household share: same email as another record with the same surname
    hh = rng.choice(np.setdiff1d(np.arange(n), idx), int(0.015 * n), replace=False)
    for i in range(0, len(hh) - 1, 2):
        email[hh[i + 1]] = email[hh[i]]
        f.loc[f.index[hh[i + 1]], "surname"] = f.loc[f.index[hh[i]], "surname"]
    f["email"], f["phone"] = email, phone

    for field in ["email", "phone"]:
        p = profile(f, field, "entity", ["surname", "date_of_birth"])
        flagged = p[(p.records >= 5) & (p.diversity > 0.5)]
        household = p[(p.records.between(2, 4)) & (p.diversity <= 0.5)]
        print(f"{field:>6}: junk flagged={len(flagged)} "
              f"(values: {list(flagged.value)[:3]}, records={int(flagged.records.sum())}) | "
              f"household-like={len(household)}")
        res[f"injected_{field}"] = {
            "junk_flagged": list(map(str, flagged.value)),
            "junk_records": int(flagged.records.sum()),
            "household_like_values": int(len(household)),
            "detector": "records>=5 and surname-diversity>0.5 -> junk; "
                        "2-4 records and diversity<=0.5 -> household",
        }
        if len(flagged):
            pur = pair_purity(f, field, "entity", flagged.value.tolist()[:3])
            print(f"        same-entity % among pairs sharing a flagged value: {pur}")
            res[f"injected_{field}"]["same_entity_pct"] = pur

    json.dump(res, open(ROOT / "research/results/shared_value_detection.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
