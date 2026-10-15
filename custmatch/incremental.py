"""Match new records against existing customers without a full rerun.

`run` leaves its state in <out>/state/: the records, the trained forest and its threshold.
`add` loads new records, scores only the candidate pairs that involve a new record, and
applies the accepted links on top of the existing customers:

  a new record linked to one customer   joins it
  linked to none                        becomes a new customer
  linked to two or more customers       merges them - logged in changes.csv for review

Existing customers are never re-decided, so their ids stay stable. Contact profiles are
recomputed over old + new records (a value can turn into junk as it spreads) but decisions
already made are not revisited: rerun `run` periodically to re-decide everything.
"""
import json
from collections import defaultdict
from datetime import datetime, timezone
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from . import explain, identity, io, monitor, schema
from .core import features, profile_contacts
from .golden import build_golden
from .models import load_model

STATE = "state"


# settings the trained model and its blocking depend on: an add must use the saved run's
MODEL_KEYS = ("country", "normalise", "extra_fields", "record_dates", "rich_features", "match_weight_feature",
              "blocking", "extra_blocking", "max_block", "hash_fields", "hash_salt_env", "placeholders", "aliases")
# settings an add's config may change: how links are accepted and what is written
ADD_OPTIONS = ("twin_guard", "unique_sources", "contact_confidence", "cookie_max_identities", "cookie_max_records", "cookie_name_check", "max_cluster", "minimal_outputs", "survivorship", "golden_fields",
               "source_priority", "household_postcode_join", "trusted_ids", "trusted_max")


def save_state(out_dir, df, customers, forest, threshold, cfg, rules=None, match_weight_model=None,
               matcher_model="forest", links=None):
    """links: DataFrame a, b, score of the accepted links (record keys) - what `add` needs to
    re-split a customer when records are deleted or changed."""
    d = Path(out_dir) / STATE
    d.mkdir(parents=True, exist_ok=True)
    df.assign(_customer=customers).to_parquet(d / "records.parquet", index=False)
    with open(d / "model.pkl", "wb") as f:
        pickle.dump({"forest": forest, "threshold": float(threshold), "rules": rules, "match_weight_model": match_weight_model, "matcher_model": matcher_model}, f)
    (d / "config.json").write_text(json.dumps(cfg))
    if links is not None:
        links.to_parquet(d / "links.parquet", index=False)


class _UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def add(state_dir, new_config, out_dir, delete=None):
    """Match new records against a saved run; a record whose key already exists is an UPDATE
    (its old version is replaced and it is matched again); `delete` is a CSV of source, record_id
    to remove. Customers that lose records are re-split on their saved links, so a deletion or
    a change can divide a customer. IDs follow stewardship.carry_ids; every change goes to
    history.csv."""
    from . import manifest, stewardship
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state = Path(state_dir) / STATE
    if not state.exists():
        raise io.ConfigError(f"no saved state in {state_dir}: run `custmatch run` there first")
    model = load_model(state)
    old = pd.read_parquet(state / "records.parquet")
    cfg = json.loads((state / "config.json").read_text())
    # the new records are read with the SAVED run's settings - country normalisation, extra
    # fields, name features - and only the new config's sources; normalising them differently
    # from the records they are matched against broke blocking and the model's feature set
    if new_config is None:                         # nothing new: a deletion only (custmatch forget)
        new_cfg_path, new_cfg = Path(state_dir), {}
        new = old.drop(columns=["_customer"]).iloc[:0].copy()
    else:
        new_cfg_path = Path(new_config)
        new_cfg = json.loads(new_cfg_path.read_text())
        fixed = sorted(k for k in MODEL_KEYS if k in new_cfg and new_cfg[k] != cfg.get(k))
        if fixed:
            raise io.ConfigError(f"{fixed} are fixed by the trained model in {state_dir}; leave them out "
                                 "of the add config, or rerun `custmatch run` to change them")
        # decision and output settings may change between batches
        cfg |= {k: v for k, v in new_cfg.items() if k in ADD_OPTIONS}
        new, _ = io.load_config(cfg | {"sources": new_cfg["sources"]}, new_cfg_path.parent)
    if cfg.get("placeholders") and len(new):
        from . import placeholders
        new = placeholders.blank(new)[0]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    old_keys = io.key(old)
    deleted = set()
    if delete is not None:
        d = pd.read_csv(delete, dtype=str, keep_default_na=False)
        deleted = set(d.source + ":" + d.record_id)
        unknown = deleted - set(old_keys)
        if unknown:
            raise io.ConfigError(f"{len(unknown)} record(s) to delete are not in the saved run, "
                                 f"e.g. {sorted(unknown)[:3]}")
    updated = set(io.key(new)) & set(old_keys)
    old_customer_all = old.pop("_customer").to_numpy()
    keep = ~np.isin(old_keys, list(deleted | updated))
    events = [[now, "record_deleted", int(c), "", 1, k] for k, c in zip(old_keys, old_customer_all) if k in deleted]
    events += [[now, "record_updated", int(c), "", 1, k] for k, c in zip(old_keys, old_customer_all) if k in updated]
    old, old_customer = old[keep].reset_index(drop=True), old_customer_all[keep]

    df = pd.concat([old, new], ignore_index=True)
    # a column one side lacks (a state saved before `updated_at` existed) is blank, not NaN:
    # "nan" would sort after every date and win "most_recent"
    obj = df.columns[df.dtypes == object]
    df[obj] = df[obj].fillna("")
    n_old, n = len(old), len(df)
    keys = io.key(df)
    pos = {k: i for i, k in enumerate(keys)}
    profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
    from .blocking import candidates_for
    from .blocking import saved_run_large_blocks
    from .models import p_match
    fopts = schema.feature_options_for(df, cfg)
    scorer = None
    if model.get("match_weight_model"):
        from .matchweight import load_scorer
        scorer = load_scorer(df, profiles, model["match_weight_model"])

    def featurise(p):
        X = features(df, p, profiles, **fopts)
        return X if scorer is None else np.column_stack([X, scorer.weights(p)])
    th = model["threshold"]
    pairs, _, report, _ = candidates_for(df, profiles, cfg, rules=model.get("rules"))
    pairs = saved_run_large_blocks(cfg, report, lambda p: p_match(model["forest"], featurise(p)), th, pairs)
    pairs = pairs[(pairs[:, 0] >= n_old) | (pairs[:, 1] >= n_old)]   # only pairs touching new records
    X = featurise(pairs)
    score = p_match(model["forest"], X)

    # decisions: the saved run's overrides file (and the new config's, if it names one) and
    # trusted identifiers
    if new_cfg.get("overrides"):                   # kept with the state, for every later add
        mine = new_cfg["overrides"]
        mine = [mine] if isinstance(mine, str) else list(mine)
        saved = cfg.get("overrides") or []
        saved = [saved] if isinstance(saved, str) else list(saved)
        cfg["overrides"] = list(dict.fromkeys(saved + [str((new_cfg_path.parent / m).resolve()) for m in mine]))
    must, cannot = stewardship.load_overrides(cfg, Path(state_dir), keys)
    trusted = stewardship.trusted_pairs(df, cfg)
    # cookies: the saved run's event files and the new config's, over old and new records
    if new_cfg.get("identity_events"):
        both = cfg.get("identity_events", []) + identity.resolve_paths(dict(new_cfg), new_cfg_path.parent)["identity_events"]
        cfg["identity_events"] = [json.loads(s) for s in dict.fromkeys(json.dumps(e, sort_keys=True) for e in both)]
    cookie_pairs, cookie_table, cookie_counts = identity.stitch(df, cfg, profiles=profiles)
    if len(cookie_pairs):
        trusted = np.unique(np.vstack([trusted, cookie_pairs]), axis=0)
    cannot_by = defaultdict(set)
    for a, b in cannot:
        cannot_by[int(a)].add(int(b))
        cannot_by[int(b)].add(int(a))

    uf = _UnionFind(n)
    members = defaultdict(set)
    for i in range(n):
        members[i].add(i)

    uniq = stewardship.unique_sources(cfg, df)
    src = df.source.to_numpy()

    def join(i, j):
        ri, rj = uf.find(i), uf.find(j)
        if ri == rj:
            return True
        for rec in members[ri]:
            if any(uf.find(o) == rj for o in cannot_by.get(rec, ())):
                return False
        if uniq and ({src[r] for r in members[ri]} & {src[r] for r in members[rj]} & uniq):
            return False                           # a duplicate-free source would hold two records
        uf.union(i, j)
        r = uf.find(i)
        members[r] = members[ri] | members[rj]
        return True

    # the saved customers, re-formed from their links so removed records can split them
    saved_links = state / "links.parquet"
    if saved_links.exists():
        sl = pd.read_parquet(saved_links)
    else:
        # a state from before links were saved: its customers as chains of links, so they are
        # kept now AND saved for the next `add` (else that one would break them apart)
        first_of, chain = {}, []
        for k, c in zip(keys[:n_old], old_customer):
            if c in first_of:
                chain.append((first_of[c], k))
            else:
                first_of[c] = k
        sl = pd.DataFrame(chain, columns=["a", "b"]).assign(score=1.0)
    # a link to a record that was updated belongs to its old version: it is matched afresh
    sl = sl[~sl.a.isin(updated) & ~sl.b.isin(updated)]
    for a, b in zip(sl.a, sl.b):
        if a in pos and b in pos and pos[a] < n_old and pos[b] < n_old:
            join(pos[a], pos[b])
    # every must-link and trusted-ID link, between old records too
    for a, b in list(must) + list(trusted):
        join(int(a), int(b))
    # new links, strongest first; with "max_cluster" a link that would grow a customer past it
    # is refused (and listed) rather than chaining customers together without limit
    cap = int(cfg["max_cluster"]) if cfg.get("max_cluster") else None
    order = np.flatnonzero(score > th)
    order = order[np.argsort(-score[order], kind="stable")]
    refused, new_links = [], []
    for k in order:
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        if uf.find(i) == uf.find(j):
            new_links.append(k)
            continue
        if cap and len(members[uf.find(i)]) + len(members[uf.find(j)]) > cap:
            refused.append({"a": keys[i], "b": keys[j], "score": round(float(score[k]), 4), "reason": "max_cluster"})
            continue
        if not join(i, j):
            refused.append({"a": keys[i], "b": keys[j], "score": round(float(score[k]), 4), "reason": "cannot-link"})
            continue
        new_links.append(k)
    roots = np.array([uf.find(i) for i in range(n)])
    # an updated record votes for its old customer's ID too, so it keeps it if it lands there
    previous = {k: int(c) for k, c in zip(old_keys, old_customer_all) if k not in deleted}
    customer, id_events = stewardship.carry_ids(keys, roots, previous, stewardship.highest_issued(state_dir))
    events += id_events
    merges = [{"merged_customer_ids": e[3], "into_customer_id": e[2]} for e in id_events if e[1] == "merged"]
    fresh = {int(customer[i]) for i in range(n_old, n)} - set(previous.values())
    from . import households as hh_mod
    hh_profiles = {f: hh_mod.profile_by_customer(df, f, customer) for f in ("email", "phone")}
    households, _ = hh_mod.assign(df, customer, hh_profiles, postcode_join=bool(cfg.get("household_postcode_join")))
    rank = {s: k for k, s in enumerate(cfg.get("source_priority", []))}
    from .pipeline import GOLDEN_FIELDS
    fields = [f for f in cfg.get("golden_fields", GOLDEN_FIELDS) if f in df.columns]
    golden = build_golden(df, customer, cfg.get("survivorship", "most_complete"), fields, rank)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = io.key(df)
    _, hh_id = np.unique(households, return_inverse=True)
    pd.DataFrame({"source": df.source, "record_id": df.record_id, "customer_id": customer,
                  "household_id": hh_id}).to_csv(out / "clusters.csv", index=False)
    joined = set(customer[:n_old])
    changes = [{"record": keys[i], "customer_id": int(customer[i]),
                "action": ("updated" if keys[i] in updated else
                           "joined existing customer" if customer[i] in joined else "new customer")}
               for i in range(n_old, n)]
    changes += [{"record": k, "customer_id": "", "action": "deleted"} for k in sorted(deleted)]
    pd.DataFrame(changes).to_csv(out / "changes.csv", index=False)
    identity.write_cookies(out, cookie_table, customer, minimal=bool(cfg.get("minimal_outputs")))
    pd.DataFrame(merges, columns=["merged_customer_ids", "into_customer_id"]).to_csv(
        out / "merges.csv", index=False)
    pd.DataFrame(refused, columns=["a", "b", "score", "reason"]).to_csv(out / "refused_links.csv", index=False)
    stewardship.write_history(out, events, state_dir)
    explain.links(df, profiles, pairs, X, score, th, keys, customer).to_csv(out / "links.csv", index=False)
    kept_links = sl[sl.a.isin(pos) & sl.b.isin(pos)]
    added = pd.DataFrame({"a": keys[pairs[new_links, 0]], "b": keys[pairs[new_links, 1]], "score": score[new_links]})
    forced = np.array([*must, *trusted], dtype=int).reshape(-1, 2)
    forced = pd.DataFrame({"a": keys[forced[:, 0]], "b": keys[forced[:, 1]], "score": 1.0})
    all_links = pd.concat([x for x in (kept_links, added, forced) if len(x)] or [kept_links], ignore_index=True)
    same = np.array([customer[pos[a]] == customer[pos[b]] for a, b in zip(all_links.a, all_links.b)], dtype=bool)
    if cfg.get("contact_confidence") is not None:          # contacts only through certain links
        from .golden import safe_contacts
        sure = all_links[same & (all_links.score.astype(float) >= float(cfg["contact_confidence"])).to_numpy()]
        sp = np.c_[sure.a.map(pos).to_numpy(), sure.b.map(pos).to_numpy()]
        per_customer, per_record = safe_contacts(df, customer, sp, cfg.get("survivorship", "most_complete"), rank,
                                                 classes=hh_profiles)
        for c, vals in per_customer.items():
            golden[c].update({f: v for f, v in vals.items() if f in golden[c]})
        if not cfg.get("minimal_outputs"):
            pd.DataFrame({"source": df.source, "record_id": df.record_id, "customer_id": customer,
                          **per_record}).to_csv(out / "contacts.csv", index=False)
    if not cfg.get("minimal_outputs"):
        pd.DataFrame([{"customer_id": c, **v} for c, v in golden.items()]).sort_values(
            "customer_id").to_csv(out / "golden_records.csv", index=False)
    save_state(out, df, customer, model["forest"], th, cfg, model.get("rules"),
               model.get("match_weight_model"), model.get("matcher_model", "forest"), all_links[same])
    summary = {"existing_records": n_old, "new_records": n - n_old - len(updated),
               "updated_records": len(updated), "deleted_records": len(deleted),
               "pairs_scored": int(len(pairs)), "links_accepted": int(len(new_links)),
               "joined_existing": sum(c["action"] == "joined existing customer" for c in changes),
               "new_customers": len(fresh), "merges_of_existing_customers": len(merges),
               "links_refused_by_max_cluster": sum(r["reason"] == "max_cluster" for r in refused),
               "links_refused_by_cannot_link": sum(r["reason"] == "cannot-link" for r in refused),
               "threshold": model["threshold"],
               **({"cookies": cookie_counts} if cookie_counts else {}),
               "warnings": ([f"{len(merges)} existing customers were merged by new records - review "
                             "merges.csv"] if merges else []) +
                           ([f"config keys custmatch does not use, so they had no effect: {', '.join(io.unknown_keys(new_cfg))}"]
                            if io.unknown_keys(new_cfg) else [])}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    monitor.write(out, monitor.snapshot(df, customer, score, th))
    manifest.write(out, manifest.build("add" if new_config is not None else "forget",
                                       new_config, new_cfg, new, 0, None, started, out,
                                       state=state_dir, delete=delete))
    return summary
