"""Order dependence: do records keep swapping partners as new records arrive?

Records of a slice of the sparse Australian file (couples who share a surname, an address, a
family email) arrive in batches in random order. After each batch a method assigns every record
that has arrived to a customer. For each record present in consecutive steps we compare its
partners among the records that were already there:

  churn    its partners changed (a merge that only adds partners is not counted)
  returns  its partners return to a set it had before, after having been different
  swap     it moved from a cluster holding one member of a household to one holding another
           member of the same household - the couple case

and score the final clusters (clustered F1). Methods are the tool's options (see --methods).
Labels: 1,000 candidate pairs of the whole slice labelled from the truth, of which each step uses
those whose two records have arrived - what a team labelling as it goes would have.

    python scripts/order_dependence.py [--methods rerun,add,...] [--seeds 1,2,3]

Writes results/order_dependence.json.
"""
import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from custmatch import incremental, labels, pipeline  # noqa: E402
from custmatch.core import partition_scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/au_customer"
OUT = ROOT / "results/order_dependence.json"
SOURCES = ("web", "app", "store")
HOUSEHOLDS = 1200
STEPS = 20
METHODS = {                          # config overrides per method; "_mode" picks how steps run
    "rerun": {"_mode": "rerun"},             # a full `custmatch run` after every batch, retrained
    "add": {"_mode": "add"},                 # run once when labels allow, then `custmatch add`
    "rerun_fixed": {"_mode": "rerun", "_fixed": True},          # rerun, one model trained up front
    # hysteresis is on by default with a previous run; off here, to isolate carrying IDs over
    "rerun_fixed_ids": {"_mode": "rerun", "_fixed": True, "_previous": True, "stability_margin": 0},   # + IDs carried
    "add_fixed": {"_mode": "add", "_fixed": True},              # add, the same fixed model
    # candidate fixes, on the rerun with a fixed model and carried IDs
    "rerun_hyst05": {"_mode": "rerun", "_fixed": True, "_previous": True, "stability_margin": 0.05},
    "rerun_hyst10": {"_mode": "rerun", "_fixed": True, "_previous": True, "stability_margin": 0.10},
}


def load_slice(tmp, couples=False, seed=0):
    frames = {}
    for s in SOURCES:
        d = pd.read_parquet(DATA / f"source_{s}.parquet")
        frames[s] = d[d.household_id < HOUSEHOLDS].reset_index(drop=True)
    if couples:
        frames["store"] = pd.concat([frames["store"], couple_records(frames, seed)], ignore_index=True)
    return frames


def couple_records(frames, seed, per_couple=2):
    """Records that could belong to either member of a couple - the case where
    order dependence shows as a record swapping between partners: first initial
    only, the surname and household address both share, no contact. Each belongs to one of the
    two, at random (person_id says which). Couples: households with exactly two people whose
    store records share a surname and an address."""
    rng = np.random.default_rng(seed)
    st = frames["store"]
    rows = []
    for hh, g in st[(st.last_name != "") & (st.address != "")].groupby("household_id"):
        people = g.groupby("person_id").first()
        if len(people) != 2 or people.last_name.str.lower().nunique() != 1:
            continue
        for _ in range(per_couple):
            owner = people.index[rng.integers(2)]
            r = people.loc[owner]
            rows.append({"record_id": f"amb{len(rows):05d}", "first_name": str(r.first_name)[:1],
                         "last_name": r.last_name, "email": "", "mobile": "", "date_of_birth": "",
                         "person_id": owner, "household_id": hh, "address": g.address.iloc[0]})
    return pd.DataFrame(rows)


def config(tmp, frames, extra, tag):
    cfg = json.loads((ROOT / "examples/au/config.json").read_text()) | {"country": "AU"}
    for s in cfg["sources"]:
        path = tmp / f"{tag}_{s['name']}.parquet"
        frames[s["name"]].to_parquet(path, index=False)
        s["path"] = str(path)
    cfg |= {k: v for k, v in extra.items() if not k.startswith("_")}
    p = tmp / f"{tag}.json"
    p.write_text(json.dumps(cfg))
    return p


def partners(clusters):
    """record key -> frozenset of the other records in its customer."""
    by = defaultdict(set)
    for k, c in clusters.items():
        by[c].add(k)
    return {k: frozenset(by[c] - {k}) for k, c in clusters.items()}


def run_method(name, spec, frames, keys_order, all_labels, truth, tmp, seed, model_dir=None):
    batches = np.array_split(np.array(keys_order), STEPS)
    arrived = set()
    snapshots, id_maps = [], []
    state = None
    fixed = model_dir if spec.get("_fixed") else None
    for step, batch in enumerate(batches):
        new = set(batch)
        tag = f"{name}_{seed}_{step}"
        if spec["_mode"] == "rerun" or state is None:
            arrived |= new
            sub = {s: f[(s + ":" + f.record_id.astype(str)).isin(arrived)] for s, f in frames.items()}
            cp = config(tmp, sub, spec, tag)
            lab = all_labels[all_labels.a.isin(arrived) & all_labels.b.isin(arrived)]
            lp = tmp / f"{tag}_labels.csv"
            lab.to_csv(lp, index=False)
            out = tmp / tag
            if fixed is None and (lab.label.astype(int).sum() < 5 or len(lab) < pipeline.MIN_LABELS):
                clusters = {k: k for k in arrived}                  # too few labels yet: all apart
                state = None
            else:
                prev = state if spec.get("_previous") else None
                pipeline.run(cp, None if fixed else lp, out, log=lambda *_: None, model_dir=fixed,
                             previous_dir=prev)
                state = out
                c = pd.read_csv(out / "clusters.csv", dtype=str)
                clusters = dict(zip(c.source + ":" + c.record_id, c.customer_id))
        else:
            sub = {s: f[(s + ":" + f.record_id.astype(str)).isin(new)] for s, f in frames.items()}
            cp = config(tmp, sub, spec, tag)
            out = tmp / tag
            incremental.add(state, cp, out)
            state = out
            arrived |= new
            c = pd.read_csv(out / "clusters.csv", dtype=str)
            clusters = dict(zip(c.source + ":" + c.record_id, c.customer_id))
        snapshots.append(partners(clusters))
        id_maps.append(clusters)
    return snapshots, id_maps


def stability(snapshots, truth_person, truth_household):
    """churn, returns and swap counts over consecutive snapshots."""
    churn = returns = swap = 0
    seen_sets = defaultdict(list)
    for t in range(1, len(snapshots)):
        prev, cur = snapshots[t - 1], snapshots[t]
        for k, before in prev.items():
            after = cur[k] & set(prev)                # partners among records already there
            if after == before:
                continue
            if before <= after:                       # only gained partners: a merge
                continue
            churn += 1
            hist = seen_sets[k]
            if after in hist and after != before:
                returns += 1
            hist.append(before)
            # swap: lost a same-household, different-person partner and gained another
            lost = {x for x in before - after if truth_household[x] == truth_household[k]}
            gained = {x for x in after - before if truth_household[x] == truth_household[k]}
            if lost and gained and {truth_person[x] for x in lost} != {truth_person[x] for x in gained}:
                swap += 1
    return {"churn": churn, "returns": returns, "swap": swap}


def couple_outcomes(snapshots, truth_person, truth_household):
    """For the couple records: how often they moved between the two partners, and where they
    ended - with their own person, with the partner, with both (the couple merged), or alone."""
    amb = [k for k in snapshots[-1] if k.split(":")[1].startswith("amb")]
    moves = 0
    final = {"own_person": 0, "partner": 0, "both": 0, "alone": 0}
    for k in amb:
        owner, hh = truth_person[k], truth_household[k]
        prev = None
        for snap in snapshots:
            if k not in snap:
                continue
            with_whom = frozenset(truth_person[x] for x in snap[k]
                                  if truth_household[x] == hh and not x.split(":")[1].startswith("amb"))
            if prev is not None and with_whom != prev and prev and with_whom:
                moves += 1
            prev = with_whom
        others = {truth_person[x] for x in snapshots[-1][k] if not x.split(":")[1].startswith("amb")}
        if not others:
            final["alone"] += 1
        elif others == {owner}:
            final["own_person"] += 1
        elif owner in others:
            final["both"] += 1
        else:
            final["partner"] += 1
    return {"couple_records": len(amb), "couple_record_moves": moves, "couple_final": final}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--couples", action="store_true", help="add records that could be either of a couple")
    a = ap.parse_args()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frames = load_slice(tmp, couples=a.couples)
        truth_person, truth_household = {}, {}
        for s, f in frames.items():
            k = s + ":" + f.record_id.astype(str)
            truth_person.update(zip(k, f.person_id))
            truth_household.update(zip(k, f.household_id))
        full_cfg = config(tmp, frames, {}, "full")
        all_labels_path = tmp / "all_labels.csv"
        labels.sample(full_cfg, 1000, all_labels_path, seed=1)
        all_labels = pd.read_csv(all_labels_path, dtype=str, keep_default_na=False)
        all_labels["label"] = [int(truth_person[x] == truth_person[y]) for x, y in zip(all_labels.a, all_labels.b)]
        keys = sorted(truth_person)
        # the fixed model: trained once on every label over the whole slice
        model_dir = tmp / "fixed_model"
        all_labels.to_csv(tmp / "fixed_labels.csv", index=False)
        pipeline.run(full_cfg, tmp / "fixed_labels.csv", model_dir, log=lambda *_: None)
        for seed in [int(s) for s in a.seeds.split(",")]:
            order = list(np.random.default_rng(seed).permutation(keys))
            for name in a.methods.split(","):
                key = f"{name}/seed{seed}" + ("/couples" if a.couples else "")
                if key in res:
                    continue
                snaps, ids = run_method(name, METHODS[name], frames, order, all_labels, truth_person, tmp, seed,
                                        model_dir=model_dir)
                id_changes = sum(1 for t in range(1, len(ids)) for k, v in ids[t - 1].items()
                                 if k in ids[t] and ids[t][k] != v)
                final = snaps[-1]
                ks = list(final)
                cust = {}
                for k in ks:                           # a customer label from the partner sets
                    cust[k] = min(final[k] | {k})
                pid = np.array([truth_person[k] for k in ks])
                _, n = np.unique(pid, return_counts=True)
                f1 = partition_scores(np.array([cust[k] for k in ks]), pid, float(np.sum(n * (n - 1) / 2)))["f1"]
                res[key] = stability(snaps, truth_person, truth_household) | {"final_f1": f1, "id_changes": id_changes}
                if a.couples:
                    res[key] |= couple_outcomes(snaps, truth_person, truth_household)
                print(f"{name:10s} seed {seed}: {res[key]}", flush=True)
                OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
