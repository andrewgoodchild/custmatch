"""Golden-record survivorship, as measured in docs/report/07-golden-records.md.

For each cluster and field, pick the surviving value. On a two-source file the modal rules
(`most_common`, `most_complete`) win on names; `first_seen` / `source_priority` depend on source
order. Chapter 7 also shows the ceiling: a value no record in the cluster holds cannot be
recovered by any rule.
"""
from collections import Counter

import numpy as np
import pandas as pd

STRATEGIES = ["first_seen", "most_common", "longest", "source_priority", "most_complete", "most_recent"]
# fields that survive together, from one record: a golden address never mixes records
GROUPS = {"address": ["addr_number", "addr_street", "addr_unit"]}


def check(rules, fields=()):
    """A config "survivorship" value -> a list of problems (empty when valid)."""
    def bad(rule, where):
        if isinstance(rule, dict):
            if not isinstance(rule.get("prefer"), list) or not rule["prefer"]:
                return [f"{where}: {{\"prefer\": [...]}} needs a list of sources"]
            if set(rule) - {"prefer", "then"}:
                return [f"{where}: unknown keys {sorted(set(rule) - {'prefer', 'then'})}"]
            return bad(rule.get("then", "most_complete"), where + " then")
        return [] if rule in STRATEGIES else [f"{where}: unknown rule {rule!r}; use one of {STRATEGIES} "
                                              "or {\"prefer\": [sources], \"then\": rule}"]
    if not isinstance(rules, dict):
        return bad(rules, "survivorship")
    known = set(fields) | set(GROUPS) | {"default"}
    problems = [f"survivorship: {f!r} is not a golden field ({sorted(known)})" for f in rules if fields and f not in known]
    return problems + [p for f, r in rules.items() for p in bad(r, f"survivorship.{f}")]


def survive(values, sources, strategy, source_rank=None, dates=None):
    """Pick the surviving value for one field from one cluster's records. strategy may also be
    {"prefer": [source, ...], "then": rule}: the first listed source that has a value, else the
    `then` rule (default most_complete)."""
    source_rank = source_rank or {}
    if isinstance(strategy, dict):
        order = {s: k for k, s in enumerate(strategy.get("prefer", []))}
        if any(str(v).strip() and s in order for v, s in zip(values, sources)):
            return survive(values, sources, "source_priority", order)
        return survive(values, sources, strategy.get("then", "most_complete"), source_rank, dates)
    if strategy == "most_recent":
        dated = [(d, v) for v, d in zip(values, dates if dates is not None else [""] * len(values))
                 if str(v).strip() and str(d).strip()]
        if dated:
            return max(dated, key=lambda t: t[0])[1]
        strategy = "most_complete"                 # no dates: fall back
    pairs = [(v, s) for v, s in zip(values, sources) if str(v).strip()]
    if not pairs:
        return ""
    vals = [v for v, _ in pairs]
    if strategy == "first_seen":
        return vals[0]
    if strategy == "most_common":
        return Counter(vals).most_common(1)[0][0]
    if strategy == "longest":
        return max(vals, key=len)
    if strategy == "source_priority":
        return min(pairs, key=lambda p: source_rank.get(p[1], 9))[0]
    if strategy == "most_complete":
        # modal value, tie-broken by the most trusted source then by length
        counts = Counter(vals)
        top = max(counts.values())
        cands = [(v, s) for v, s in pairs if counts[v] == top]
        return min(cands, key=lambda p: (source_rank.get(p[1], 9), -len(p[0])))[0]
    raise ValueError(strategy)


def build_golden(df, labels, strategy, fields, source_rank=None, table=False):
    """{cluster label: {field: surviving value}}. strategy: one strategy for every field, or a
    config "survivorship" dict - {"default": "most_complete", "email": "most_recent",
    "address": "most_recent", "phone": {"prefer": ["app", "store"]}} - where "address" covers
    the address parts, chosen together from one record. most_recent reads `updated_at`.
    table: the same as a DataFrame instead (golden_table)."""
    if not isinstance(strategy, dict):
        # one rule for every field: as {"default": rule, "address": rule}, so most_recent sees the
        # dates and an address is never assembled from parts of different records
        strategy = {"default": strategy, "address": strategy}
    if isinstance(strategy, dict):
        return _build_by_field(df, labels, strategy, fields, source_rank, table)
    out = {}
    src = df.source.to_numpy()
    arrays = {f: df[f].to_numpy() for f in fields}
    order = np.argsort(labels, kind="stable")
    lab_sorted = labels[order]
    bounds = np.flatnonzero(np.diff(lab_sorted)) + 1
    for idx in np.split(order, bounds):
        if len(idx) == 1:       # one record: every strategy keeps its own values
            k = idx[0]
            out[labels[k]] = {f: (arrays[f][k] if str(arrays[f][k]).strip() else "") for f in fields}
            continue
        out[labels[idx[0]]] = {f: survive(arrays[f][idx], src[idx], strategy, source_rank)
                               for f in fields}
    return out


def _most_complete(labels, values, src, source_rank):
    """survive(..., "most_complete") for every cluster at once: the cluster's most common
    non-blank value; ties to the most trusted source, then the longest value, then the first
    record. Returns a Series: cluster label -> winning row, absent when the cluster has no value
    (as a dict, millions of entries a field). The per-cluster loop took 29 s of a million-record
    run."""
    as_str = pd.Series(np.asarray(values, dtype=object), dtype=object).astype(str)
    rows = np.flatnonzero((as_str.str.strip() != "").to_numpy(dtype=bool))
    if not len(rows):
        return pd.Series(np.zeros(0, np.int64), dtype=np.int64)
    v = as_str.iloc[rows]
    f = pd.DataFrame({"lab": np.asarray(labels)[rows], "v": v.to_numpy(dtype=object), "row": rows,
                      "rank": pd.Series(np.asarray(src)[rows], dtype=object).map(source_rank).fillna(9).to_numpy(),
                      "len": v.str.len().to_numpy()})
    f["cnt"] = f.groupby(["lab", "v"], sort=False).v.transform("size")
    f = f.sort_values(["lab", "cnt", "rank", "len", "row"], ascending=[True, False, True, False, True], kind="stable")
    first = f.drop_duplicates("lab")
    return pd.Series(first.row.to_numpy(np.int64), index=first.lab.to_numpy())


def golden_table(df, labels, strategy, fields, source_rank=None):
    """build_golden as a DataFrame: a row per cluster, indexed by its label, in the same order and
    with the same columns as its dicts. Built column by column: a dict per customer was 2 GB on
    five million NCVR records, and the table made from those dicts 4 GB more."""
    return build_golden(df, labels, strategy, fields, source_rank, table=True)


def set_golden(table, per_customer):
    """Each {label: {field: value}} written into golden_table's rows, as dict.update would, for
    the fields the table has."""
    updates = {}
    for row, vals in zip(table.index.get_indexer(list(per_customer)).tolist(), per_customer.values()):
        for f, v in vals.items():
            if f in table.columns:
                updates.setdefault(f, ([], []))
                updates[f][0].append(row)
                updates[f][1].append(v)
    for f, (rows, vals) in updates.items():
        col = table[f].to_numpy(dtype=object, copy=True)
        col[rows] = vals
        table[f] = col.tolist()


def _build_by_field(df, labels, rules, fields, source_rank, table=False):
    default = rules.get("default", "most_complete")
    source_rank = source_rank or {}
    src = df.source.to_numpy()
    dates = df["updated_at"].to_numpy() if "updated_at" in df.columns else np.full(len(df), "", dtype=object)
    arrays = {f: df[f].to_numpy(dtype=object) for f in fields}
    grouped = {f: g for g, members in GROUPS.items() if g in rules for f in members if f in fields}
    labels = np.asarray(labels)
    # the default rule, most_complete, for every cluster at once; other rules cluster by cluster
    fast = {f: _most_complete(labels, arrays[f], src, source_rank)
            for f in fields if f not in grouped and rules.get(f, default) == "most_complete"}
    group_fast = {}
    for g, members in GROUPS.items():
        members = [m for m in members if m in fields]
        if g in rules and members and rules[g] == "most_complete":
            parts = [pd.Series(arrays[m], dtype=object).map(str) for m in members]
            joined = parts[0].str.cat(parts[1:], sep=" ").str.strip().to_numpy(dtype=object) if len(df) \
                else np.zeros(0, dtype=object)        # " ".join(str(...)).strip(), column by column
            win = _most_complete(labels, joined, src, source_rank)
            # the first record (in record order) of its cluster holding the winning joined value
            w = win.reindex(labels).to_numpy()
            ok = ~np.isnan(w)
            ok[ok] = joined[ok] == joined[w[ok].astype(np.int64)]
            at = np.flatnonzero(ok)
            group_fast[g] = (members, pd.Series(at, dtype=np.int64).groupby(labels[at]).first())
    slow_groups = [g for g, members in GROUPS.items()
                   if g in rules and g not in group_fast and any(m in fields for m in members)]
    if not slow_groups and all(f in fast or f in grouped for f in fields):
        # every field by the vectorised rule: no loop over customers at all
        labs = pd.unique(labels)

        def pick(values, rows):                     # each customer's winning value, "" when none
            r = pd.Series(rows, dtype="float64").reindex(labs).to_numpy()
            have = ~np.isnan(r)
            out = np.full(len(labs), "", dtype=object)
            out[have] = values[r[have].astype(np.int64)]
            return out.tolist()
        cols = {}
        for f in fields:
            if f in fast:
                cols[f] = pick(arrays[f], fast[f])
        for members, rows in group_fast.values():
            for m in members:
                cols[m] = pick(arrays[m], rows)
        # the loop's key order: grouped fields (address) first, then the rest in field order
        order = [m for members, _ in group_fast.values() for m in members] + \
                [f for f in fields if f not in grouped and f in cols]
        if table:
            return pd.DataFrame({f: cols.pop(f) for f in order}, index=labs)
        return {lab: dict(zip(order, vals)) for lab, vals in zip(labs, zip(*(cols[f] for f in order)))}
    out = {}
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    for idx in np.split(order, bounds):
        lab = labels[idx[0]]
        rec = {}
        for members, rows in group_fast.values():
            k = rows.get(lab)
            for m in members:
                rec[m] = arrays[m][k] if k is not None else ""
        for g, members in GROUPS.items():
            members = [m for m in members if m in fields]
            if g not in rules or not members or g in group_fast:
                continue
            # the whole group from one record: the record the rule picks on its joined value
            joined = np.array([" ".join(str(arrays[m][k]) for m in members).strip() for k in idx], dtype=object)
            pick = survive(joined, src[idx], rules[g], source_rank, dates[idx])
            k = next((k for k, v in zip(idx, joined) if v == pick), None)
            for m in members:
                rec[m] = arrays[m][k] if k is not None else ""
        for f in fields:
            if f in grouped:
                continue
            if f in fast:
                k = fast[f].get(lab)
                rec[f] = arrays[f][k] if k is not None else ""
                continue
            rec[f] = survive(arrays[f][idx], src[idx], rules.get(f, default), source_rank, dates[idx])
        out[lab] = rec
    return pd.DataFrame(list(out.values()), index=list(out)) if table else out


# ------------------------------------------------------------------ contact safety
CONTACT_FIELDS = ("email", "phone")


def safe_contacts(df, customers, strong_pairs, rules, source_rank=None, classes=None):
    """Contact details that are safe to use: chosen only among records joined by links at or
    above the contact confidence ("contact_confidence") - a record's own, and those certainly the
    same person. A weaker link may join records into one customer for analysis, but never decides
    whose email or phone is used to reach them (docs/guide.md "Contact safety").

    classes: {field: {value: class}} (custmatch's contact profiles). A value a household shares -
    a family email, a home phone - is used only when the group has no personal one: after a
    separation the family email reaches the ex. `{field}_shared` says when it was the only one.

    Returns ({customer label: {field: value}} from the customer's largest certain group, and
    {field: per-record value array} - the contact to use when acting on that record)."""
    from .core import components
    fields = [f for f in CONTACT_FIELDS if f in df.columns]
    customers = np.asarray(customers)
    n = len(df)
    pairs = np.asarray(strong_pairs, dtype=np.int64).reshape(-1, 2)
    core = components(n, pairs, np.ones(len(pairs), dtype=bool))
    classes = classes or {}
    personal_df, usable_df = df.copy(), df.copy()
    for f in fields:                               # personal values first: shared ones blanked
        cls = classes.get(f, {})
        personal_df[f] = [v if cls.get(v, ("personal",))[0] == "personal" else "" for v in df[f]]
        usable_df[f] = [v if cls.get(v, ("personal",))[0] != "junk" else "" for v in df[f]]   # never junk
    by_core = build_golden(personal_df, core, rules, fields, source_rank)
    any_core = build_golden(usable_df, core, rules, fields, source_rank)
    per_record = {}
    for f in fields:
        chosen = [by_core[c][f] or any_core[c][f] for c in core]
        per_record[f] = np.array(chosen, dtype=object)
        per_record[f + "_shared"] = np.array([bool(v) and not by_core[c][f] for v, c in zip(chosen, core)])
    by_core = {c: {f: by_core[c][f] or any_core[c][f] for f in fields} for c in any_core}
    biggest = {}
    for cust, c in zip(customers, core):
        biggest.setdefault(cust, {}).setdefault(c, 0)
        biggest[cust][c] += 1
    per_customer = {cust: by_core[min(cs, key=lambda c: (-cs[c], c))] for cust, cs in biggest.items()}
    return per_customer, per_record
