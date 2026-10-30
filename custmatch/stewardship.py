"""Running custmatch over time: decisions people make, IDs that stay put, and a record of changes.

  overrides     a CSV of `a, b, decision` - `same` (must-link) or `different` (cannot-link) - by
                record key (`source:record_id`), from a data steward. `run` and `add` both honour
                it: a must-link joins the two records whatever the model says; a cannot-link
                keeps them apart even when other links would chain them together, by cutting the
                weakest link on the strongest path between them - that is an unmerge. Config:
                "overrides": "overrides.csv" (relative to the config).
  trusted_ids   columns whose equal values mean the same person (a loyalty or customer number):
                "trusted_ids": ["loyalty_id"]. They become must-links; a cannot-link still wins,
                and a value shared by more than `trusted_max` records (default 5) is ignored as
                a placeholder.
  review band   pairs whose score lies within "review_band" (default 0.1) of the threshold, on
                either side, are written to review.csv for a person to decide; their answers go
                back in as overrides.
  carry IDs     `run --previous DIR` gives each new customer the ID of the previous run's
                customer it shares most records with (Zingg does the same), so a full rerun
                keeps IDs; customers that merged, split or appeared are listed in history.csv.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_COLUMNS = ["time", "event", "customer_id", "other_customer_ids", "records", "detail"]


# ------------------------------------------------------------------ overrides and trusted IDs
def load_overrides(cfg, base_dir, keys):
    """(must, cannot) index pairs from the config's "overrides" file - or files: `add` appends each
    batch's own, so a decision made with one batch holds for every later one."""
    from . import io
    paths = cfg.get("overrides")
    if not paths:
        return np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64)
    frames = []
    for path in ([paths] if isinstance(paths, str) else paths):
        path = Path(path)
        if not path.is_absolute():
            path = Path(base_dir) / path
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    o = pd.concat(frames, ignore_index=True)
    pos = {k: i for i, k in enumerate(keys)}
    d = o.decision.str.strip().str.lower()
    bad = sorted(set(d) - {"same", "different"})
    if bad:
        raise io.ConfigError(f"overrides: decision must be same or different, not {bad[:3]}")
    known = o.a.isin(pos) & o.b.isin(pos)
    # an override about a record not in this run (deleted, or in another batch) simply waits
    o, d = o[known], d[known]
    pairs = np.array([[pos[a], pos[b]] for a, b in zip(o.a, o.b)], dtype=np.int64).reshape(-1, 2)
    pairs = np.sort(pairs, axis=1)
    return pairs[(d == "same").to_numpy()], pairs[(d == "different").to_numpy()]


def trusted_pairs(df, cfg):
    """Must-link pairs from records sharing a trusted identifier value."""
    out = set()
    limit = int(cfg.get("trusted_max", 5))
    for col in cfg.get("trusted_ids", []):
        if col not in df.columns:
            continue
        v = df[col].astype(str).str.strip()
        for idx in v[v != ""].groupby(v[v != ""]).groups.values():
            idx = sorted(idx)
            if 2 <= len(idx) <= limit:
                out.update((idx[0], j) for j in idx[1:])      # a chain is enough: closure joins them
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2)


def apply_decisions(pairs, score, must, cannot):
    """Candidate pairs and scores with must-links added at 1.0 and cannot-links set to 0.0."""
    if not len(must) and not len(cannot):
        return pairs, score
    idx = {(int(a), int(b)): k for k, (a, b) in enumerate(pairs)}
    score = score.astype(float).copy()
    extra = []
    for a, b in must:
        k = idx.get((int(a), int(b)))
        if k is None:
            extra.append((a, b))
        else:
            score[k] = 1.0
    for a, b in cannot:
        k = idx.get((int(a), int(b)))
        if k is not None:
            score[k] = 0.0
    if extra:
        pairs = np.vstack([pairs, np.array(extra, dtype=pairs.dtype)])
        score = np.concatenate([score, np.ones(len(extra))])
    return pairs, score


def enforce_cannot(n, pairs, score, th, labels, cannot):
    """Split customers that hold a cannot-link pair: while the two records are connected, cut
    the weakest link on the strongest path between them. Returns (labels, links cut)."""
    import networkx as nx
    if not len(cannot):
        return labels, 0
    labels = labels.copy()
    cut = 0
    linked = score > th
    by_label = defaultdict(list)
    for a, b in cannot:
        if labels[a] == labels[b]:
            by_label[labels[a]].append((int(a), int(b)))
    if not by_label:
        return labels, 0
    next_label = labels.max() + 1
    for lab, bad in by_label.items():
        members = set(np.flatnonzero(labels == lab).tolist())
        g = nx.Graph()
        g.add_nodes_from(members)
        for (a, b), s in zip(pairs[linked], score[linked]):
            if a in members and b in members:
                g.add_edge(int(a), int(b), weight=float(s))
        for a, b in bad:
            while nx.has_path(g, a, b):
                tree = nx.maximum_spanning_tree(g.subgraph(nx.node_connected_component(g, a)))
                path = nx.shortest_path(tree, a, b)
                edges = list(zip(path, path[1:]))
                u, v = min(edges, key=lambda e: g[e[0]][e[1]]["weight"])
                g.remove_edge(u, v)
                cut += 1
        for comp in nx.connected_components(g):
            comp = sorted(comp)
            labels[comp] = next_label
            next_label += 1
    return labels, cut


# ------------------------------------------------------------------ the review band
def review_pairs(pairs, score, th, band, keys, X=None, df=None, profiles=None):
    """Pairs within `band` of the threshold, most uncertain first. X: the pairs' features, or a
    function giving the features of the pairs at some row indices."""
    near = np.flatnonzero(np.abs(score - th) < band)
    rows = pd.DataFrame({"a": keys[pairs[near, 0]], "b": keys[pairs[near, 1]],
                         "score": np.round(score[near].astype(float), 4),
                         "linked": score[near] > th, "decision": ""})
    if X is not None and len(near):
        from .explain import _both_streets, _contacts, evidence
        i, j = pairs[near, 0], pairs[near, 1]
        ce, cp = _contacts(df, profiles, "email", i, j), _contacts(df, profiles, "phone", i, j)
        Xn = X(near) if callable(X) else X[near]
        rows["evidence"] = [evidence(x, a_, b_, s) for x, a_, b_, s in zip(Xn, ce, cp, _both_streets(df, i, j))]
    return rows.iloc[np.argsort(np.abs(rows.score.to_numpy() - th), kind="stable")].reset_index(drop=True)


# ------------------------------------------------------------------ IDs and their history
def highest_issued(out_dir):
    """The highest customer ID a previous output ever used - in its clusters or its history - so a
    retired ID (merged away, erased) is never given to someone else. -1 when none."""
    out_dir = Path(out_dir)
    best = -1
    if (out_dir / "clusters.csv").exists():
        c = pd.read_csv(out_dir / "clusters.csv", usecols=["customer_id"])
        if len(c):
            best = max(best, int(c.customer_id.max()))
    if (out_dir / "history.csv").exists():
        h = pd.read_csv(out_dir / "history.csv", dtype=str, keep_default_na=False)
        ids = [int(x) for col in ("customer_id", "other_customer_ids") for v in h.get(col, []) for x in str(v).split(";")
               if x.strip().lstrip("-").isdigit()]
        best = max([best] + ids)
    return best


def join_must(labels, must):
    """Labels with every must-link pair in one customer: a steward's "same" holds after the steps
    that can split customers (the size re-cut)."""
    if not len(must):
        return labels
    labels = labels.copy()
    for a, b in must:
        la, lb = labels[a], labels[b]
        if la != lb:
            labels[labels == lb] = la
    return labels


def carry_ids(keys, labels, previous, issued=-1):
    """Customer IDs for `labels`, reusing the previous run's where records overlap.

    previous: {record key: previous customer id}. A previous ID goes to the new customer holding
    most of its records (ties: the larger customer, then the smaller label); a new customer that
    wins several previous IDs keeps the one it holds most records of (ties: the smaller ID) and
    the others are recorded as merged into it; a customer that wins none gets a fresh ID.
    issued: the highest ID ever used before (highest_issued); fresh IDs start above it.
    Returns (ids per record, history events)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    labels = np.asarray(labels)
    if not previous and np.issubdtype(labels.dtype, np.integer):
        # a first run: every customer is new, numbered in label order (what the loop below does,
        # without a vote per customer - a million of them at a million records)
        uniq, inv, size = np.unique(labels, return_inverse=True, return_counts=True)
        first = max(int(issued), -1) + 1
        events = [[now, "new", first + k, "", c, ""] for k, c in enumerate(size.tolist())]
        return (first + inv.reshape(-1)).astype(np.int64), events
    members = defaultdict(list)
    for k, lab in zip(keys, labels):
        members[lab].append(k)
    votes = {lab: Counter(previous[k] for k in ks if k in previous) for lab, ks in members.items()}
    holders = defaultdict(list)                                # previous id -> [new labels holding it]
    for lab, c in votes.items():
        for pid in c:
            holders[pid].append(lab)
    winner = {pid: max(labs, key=lambda lab: (votes[lab][pid], len(members[lab]), -int(lab) if str(lab).lstrip("-").isdigit() else 0))
              for pid, labs in holders.items()}
    next_id = max([int(p) for p in previous.values()] + [int(issued), -1]) + 1
    won_by = defaultdict(list)                                 # new label -> previous ids it won
    for pid, lab in winner.items():
        won_by[lab].append(pid)
    ids, events = {}, []
    for lab in sorted(members, key=lambda x: (str(type(x)), x)):
        won = sorted(won_by.get(lab, ()), key=lambda pid: (-votes[lab][pid], int(pid)))
        if won:
            ids[lab] = int(won[0])
            if len(won) > 1:
                events.append([now, "merged", ids[lab], ";".join(map(str, sorted(won))), len(members[lab]),
                               "customers joined by new evidence"])
        else:
            ids[lab] = next_id
            next_id += 1
            events.append([now, "new", ids[lab], "", len(members[lab]),
                           "records split from " + ";".join(map(str, sorted(votes[lab]))) if votes[lab] else ""])
    for pid, labs in holders.items():
        if len(labs) > 1:
            events.append([now, "split", int(pid), ";".join(str(ids[lab]) for lab in labs),
                           sum(votes[lab][pid] for lab in labs), "customer divided"])
    return np.array([ids[lab] for lab in labels], dtype=np.int64), events


def write_history(out_dir, events, previous_dir=None):
    """history.csv: the previous run's history plus this run's events."""
    out = Path(out_dir) / "history.csv"
    old = pd.DataFrame(columns=HISTORY_COLUMNS)
    if previous_dir and (Path(previous_dir) / "history.csv").exists():
        old = pd.read_csv(Path(previous_dir) / "history.csv", dtype=str, keep_default_na=False)
    new = pd.DataFrame(events, columns=HISTORY_COLUMNS)
    pd.concat([old, new], ignore_index=True).to_csv(out, index=False)


# ------------------------------------------------------------------ stability as records arrive
def hysteresis(pairs, score, keys, previous_links, previous_keys, margin):
    """Scores nudged toward the previous run's decision: a pair linked last time gets +margin, a
    pair whose records were both there and were NOT linked gets -margin. A pair near the
    threshold then keeps its decision unless the evidence moves by more than the margin - the
    usual cure for a decision that flips back and forth on small changes."""
    if not margin or previous_links is None:
        return score
    linked = set(zip(previous_links.a, previous_links.b)) | set(zip(previous_links.b, previous_links.a))
    before = set(previous_keys)
    a, b = keys[pairs[:, 0]], keys[pairs[:, 1]]
    was = np.array([(x, y) in linked for x, y in zip(a, b)], dtype=bool)
    both = np.array([x in before and y in before for x, y in zip(a, b)], dtype=bool)
    return np.clip(score + margin * was - margin * (both & ~was), 0.0, 1.0)


# ------------------------------------------------------------------ twins
def twin_pairs(df, pairs, hashed=False):
    """Candidate pairs that look like twins, not one person: the same surname and full date of
    birth, and first names that are both written out (3+ letters), clearly different (Jaro-Winkler
    below 0.8) and not nicknames of each other (their nickname roots do not meet). Everything
    else a twin shares - address, phone, family email - says "same person", and a model trained
    on ordinary labels, where twins are rare, links them (docs/report/05-customer-data.md "Scenarios")."""
    from rapidfuzz.distance import JaroWinkler
    from rapidfuzz.process import cpdist
    if not len(pairs) or hashed:
        # hashed first names cannot be told apart from typos or initials: the guard would split
        # Jonathan from Jonathon, so it is off (pipeline warns)
        return np.zeros(len(pairs), dtype=bool)
    from anyascii import anyascii
    i, j = pairs[:, 0], pairs[:, 1]
    fn = np.array([anyascii(x).lower() for x in df.first_name.astype(str)], dtype=object)
    ln = df.last_name.astype(str).str.lower().to_numpy()
    dob = df.date_of_birth.astype(str).to_numpy()
    cand = (dob[i] != "") & (dob[i] == dob[j]) & (ln[i] != "") & (ln[i] == ln[j]) \
        & (np.char.str_len(fn[i].astype(str)) >= 3) & (np.char.str_len(fn[j].astype(str)) >= 3) & (fn[i] != fn[j])
    out = np.zeros(len(pairs), dtype=bool)
    k = np.flatnonzero(cand)
    if not len(k):
        return out
    sim = cpdist(list(fn[i[k]]), list(fn[j[k]]), scorer=JaroWinkler.normalized_similarity, workers=-1)
    differ = sim < 0.8
    if "first_roots" in df.columns:
        roots = df.first_roots.astype(str).to_numpy()
        differ &= np.array([not (set(roots[a].split("|")) & set(roots[b].split("|"))) for a, b in zip(i[k], j[k])])
    out[k] = differ
    return out


# ------------------------------------------------------------------ duplicate-free sources
def unique_sources(cfg, df):
    """The sources declared duplicate-free ("unique_sources": a list, or true for all)."""
    u = cfg.get("unique_sources")
    if not u:
        return set()
    return set(df.source.unique()) if u is True else set(u)


def unique_source_pairs(df, labels, sources):
    """Cannot-link pairs inside each customer: two records from one duplicate-free source are
    two people (a voter roll, a loyalty scheme, one account per person), so the customer is cut
    between them (enforce_cannot) - the constraint anonlink's solver and Leipzig's CLIP use."""
    if not sources:
        return np.zeros((0, 2), dtype=np.int64)
    src = df.source.to_numpy()
    keep = np.flatnonzero(np.isin(src, list(sources)))
    out = []
    g = pd.DataFrame({"lab": np.asarray(labels)[keep], "src": src[keep], "i": keep})
    for _, grp in g[g.duplicated(["lab", "src"], keep=False)].groupby(["lab", "src"]):
        idx = grp.i.tolist()
        out += [(idx[x], idx[y]) for x in range(len(idx)) for y in range(x + 1, len(idx))]
    return np.array(out, dtype=np.int64).reshape(-1, 2)
