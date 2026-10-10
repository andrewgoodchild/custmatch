"""Build examples/demo/: a small, shippable slice of the synthetic Australian file, so a new user
can run custmatch straight after installing it.

    custmatch run examples/demo/config.json --labels examples/demo/labels.csv --out demo_out/

The records are synthetic (scripts/au_customer_data.py: names drawn from public-domain US Census
and Social Security frequency lists, everything else generated). Their emails are moved to the
reserved .example domain (RFC 2606) - bigpond.com becomes bigpond.example - so no generated
address can be someone's real inbox; placeholder domains (test.com, noemail.com) stay, as the
junk they stand for. person_id and household_id are
the truth, kept as unmapped columns so you can score the output; labels.csv holds 600 candidate
pairs labelled from that truth, standing in for a person's labels: 300 drawn at random, then 300
chosen by active learning, 100 at a time - the labelling the README recommends.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/au_customer"
OUT = ROOT / "examples/demo"
HOUSEHOLDS = 500
PLACEHOLDER_DOMAINS = {"test.com", "none.com", "noemail.com", "a.com", "na.com", "asdf.com", "example.com"}


def fictional(email):
    """A provider domain on the reserved .example domain: gmail.com -> gmail.example,
    yahoo.com.au -> yahoo.au.example. Placeholder domains are kept."""
    if "@" not in email:
        return email
    user, dom = email.rsplit("@", 1)
    if dom.lower() in PLACEHOLDER_DOMAINS:
        return email
    parts = dom.split(".")
    country = [parts[-1]] if len(parts) > 2 and len(parts[-1]) == 2 else []
    return f"{user}@{'.'.join([parts[0]] + country + ['example'])}"


def main():
    sys.path.insert(0, str(ROOT))
    from custmatch import labels
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "examples/au/config.json").read_text())
    for s in cfg["sources"]:
        d = pd.read_parquet(SRC / f"source_{s['name']}.parquet")
        d = d[d.household_id < HOUSEHOLDS]
        d["email"] = d["email"].fillna("").map(fictional)
        d.to_csv(OUT / f"{s['name']}.csv", index=False)
        s["path"] = f"{s['name']}.csv"
    cfg["country"] = "AU"
    (OUT / "config.json").write_text(json.dumps(cfg, indent=1))
    frames = {s["name"]: pd.read_csv(OUT / s["path"], dtype=str, keep_default_na=False) for s in cfg["sources"]}
    truth = {f"{n}:{r}": p for n, f in frames.items() for r, p in zip(f.record_id, f.person_id)}

    def fill(path):
        d = pd.read_csv(path, dtype=str, keep_default_na=False)
        d["label"] = [int(truth[a] == truth[b]) for a, b in zip(d.a, d.b)]
        return d[["a", "b", "label", "by"]]
    labels.sample(OUT / "config.json", 300, OUT / "labels.csv", seed=1)
    lab = fill(OUT / "labels.csv")
    lab.to_csv(OUT / "labels.csv", index=False)
    while len(lab) < 600:
        labels.sample_active(OUT / "config.json", OUT / "labels.csv", 100, OUT / "next.csv", seed=1)
        lab = pd.concat([lab, fill(OUT / "next.csv")], ignore_index=True)
        lab.to_csv(OUT / "labels.csv", index=False)
    (OUT / "next.csv").unlink()
    print(f"{sum(len(f) for f in frames.values())} records, {len(lab)} labelled pairs "
          f"({int(lab.label.sum())} matches) in {OUT}")


if __name__ == "__main__":
    main()
