"""Build a two-source customer-matching dataset with labelled contact pathologies.

Sources come from pseudopeople (synthetic US administrative records, safe to share):
  A "CRM"     = Decennial Census 2020: names, DOB, residential address, household_id
  B "Billing" = W-2/1099 2020:        names, SSN, mailing address, employer

They deliberately do not share a strong key - no DOB in B, no SSN in A - which is the usual
MDM situation. Ground truth: `simulant_id` for identity, `household_id` for households.

On top we inject email and phone with a known mix:
  personal   - unique to the person
  household  - shared by everyone in the household   (the householding problem)
  junk       - placeholder values many people type   (disguised missing data)
  missing    - blank

Every injected value's true class is written out, so detection can be scored.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("future.infer_string", False)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/customer"
SEED = 7

# placeholder values people actually type into forms
JUNK_EMAILS = ["test@test.com", "asdf@asdf.com", "none@none.com", "noreply@example.com",
               "a@a.com", "test@example.com", "no@email.com", "xxx@xxx.com"]
JUNK_PHONES = ["0000000000", "1234567890", "5555555555", "1111111111", "9999999999",
               "0123456789", "5551234567"]
MIX = {"personal": 0.50, "household": 0.25, "junk": 0.15, "missing": 0.10}
CROSS_SOURCE_DRIFT = 0.10   # chance the same person's contact differs between sources


def _slug(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def assign_contacts(people, rng):
    """people: DataFrame with simulant_id, household_id, first_name, last_name."""
    classes = rng.choice(list(MIX), size=len(people), p=list(MIX.values()))
    hh_email, hh_phone = {}, {}
    email, phone, truth = [], [], []
    for (sid, hid, fn, ln), cls in zip(
            people[["simulant_id", "household_id", "first_name", "last_name"]].to_numpy(), classes):
        if cls == "personal":
            e = f"{_slug(fn)}.{_slug(ln)}{rng.integers(1, 9999)}@mail.com"
            p = f"206{rng.integers(1000000, 9999999)}"
        elif cls == "household":
            if hid not in hh_email:
                hh_email[hid] = f"the{_slug(ln)}family{rng.integers(1, 999)}@mail.com"
                hh_phone[hid] = f"206{rng.integers(1000000, 9999999)}"
            e, p = hh_email[hid], hh_phone[hid]
        elif cls == "junk":
            e, p = rng.choice(JUNK_EMAILS), rng.choice(JUNK_PHONES)
        else:
            e = p = ""
        email.append(e)
        phone.append(p)
        truth.append(cls)
    return pd.DataFrame({"simulant_id": people.simulant_id.to_numpy(), "email": email,
                         "phone": phone, "contact_class": truth})


def attach(df, contacts, rng):
    """Attach contacts to a source, with some drift so contacts are not a perfect key."""
    m = df.merge(contacts, on="simulant_id", how="left")
    drift = rng.random(len(m)) < CROSS_SOURCE_DRIFT
    has_email = m.email.fillna("").ne("")
    idx = m.index[drift & has_email]
    # a drifted contact becomes a different personal value (changed address/provider)
    m.loc[idx, "email"] = [f"user{rng.integers(10**5, 10**6)}@mail.com" for _ in idx]
    m.loc[idx, "phone"] = [f"425{rng.integers(1000000, 9999999)}" for _ in idx]
    m["email"] = m.email.fillna("").str.lower().str.strip()
    m["phone"] = m.phone.fillna("").astype(str).str.replace(r"\D", "", regex=True)
    return m


def main():
    import pseudopeople as psp
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    census = psp.generate_decennial_census(year=2020)
    w2 = psp.generate_taxes_w2_and_1099(year=2020)
    print(f"census={len(census):,}  w2={len(w2):,}", flush=True)

    people = (pd.concat([census[["simulant_id", "household_id", "first_name", "last_name"]],
                         w2[["simulant_id", "household_id", "first_name", "last_name"]]])
              .dropna(subset=["simulant_id"])
              .drop_duplicates("simulant_id")
              .reset_index(drop=True))
    contacts = assign_contacts(people, rng)

    a = census.rename(columns={"street_number": "addr_number", "street_name": "addr_street",
                               "unit_number": "addr_unit"})
    a = a[["simulant_id", "household_id", "first_name", "middle_initial", "last_name",
           "date_of_birth", "addr_number", "addr_street", "addr_unit", "sex"]].copy()
    a["source"] = "crm"
    a["record_id"] = ["A" + str(i) for i in range(len(a))]

    b = w2.rename(columns={"mailing_address_street_number": "addr_number",
                           "mailing_address_street_name": "addr_street",
                           "mailing_address_unit_number": "addr_unit"})
    b = b[["simulant_id", "household_id", "first_name", "middle_initial", "last_name",
           "ssn", "addr_number", "addr_street", "addr_unit", "employer_name"]].copy()
    b["source"] = "billing"
    b["record_id"] = ["B" + str(i) for i in range(len(b))]

    a, b = attach(a, contacts, rng), attach(b, contacts, rng)
    for df in (a, b):
        for c in ["first_name", "last_name", "addr_street", "addr_unit", "middle_initial"]:
            df[c] = df[c].fillna("").astype(str).str.lower().str.strip()
        df["addr_number"] = df.addr_number.fillna("").astype(str).str.strip()

    a.to_parquet(OUT / "source_crm.parquet")
    b.to_parquet(OUT / "source_billing.parquet")
    contacts.to_parquet(OUT / "contact_truth.parquet")

    stats = {
        "crm_records": len(a), "billing_records": len(b),
        "people": int(contacts.simulant_id.nunique()),
        "shared_people": int(len(set(a.simulant_id) & set(b.simulant_id))),
        "households": int(pd.concat([a.household_id, b.household_id]).nunique()),
        "contact_mix": contacts.contact_class.value_counts().to_dict(),
        "distinct_emails": int(pd.concat([a.email, b.email]).replace("", np.nan).nunique()),
        "junk_email_records": int(pd.concat([a.email, b.email]).isin(JUNK_EMAILS).sum()),
    }
    print(json.dumps(stats, indent=1), flush=True)
    json.dump(stats, open(ROOT / "results/customer_dataset.json", "w"), indent=1, default=int)


if __name__ == "__main__":
    main()
