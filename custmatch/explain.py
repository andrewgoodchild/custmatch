"""Why were two records merged? One row per accepted link, in words a reviewer can check.

Each link carries its score, the evidence the matcher saw (name similarity, shared address,
date of birth, SSN, and whether a shared email or phone is personal, a household's, or junk),
and whether it is a BRIDGE: a link whose removal would split its customer in two.

Rows are ordered weakest first, because that is where the false merges are: on the synthetic
customer file the weakest 10% of links held 86% of the false ones (160 of 187). Ordering by
bridges found 44 for the same effort - with about two records per customer most links are
bridges. On files with many records per customer (docs/report/02-whole-customers.md), bridges matter more,
so the flag is kept.
"""
import networkx as nx
import numpy as np
import pandas as pd

# column order of core.features
F_FIRST, F_LAST, F_NUM, F_STREET, F_UNIT = 0, 1, 4, 5, 6
F_DOB, F_SSN = 11, 12


def evidence(x, contact_email="", contact_phone="", street=True):
    """street: both records have a street. Two blank streets score 1.0 as a feature (blank agrees
    with blank, deliberately), which is not evidence of an address, so it is not called one."""
    parts = [f"names {x[F_FIRST]:.2f}/{x[F_LAST]:.2f}"]
    if street and x[F_NUM] == 1 and x[F_STREET] > 0.95:
        parts.append("same address" + (" and unit" if x[F_UNIT] == 1 else ""))
    elif street and x[F_STREET] > 0.95:
        parts.append("same street")
    if x[F_DOB] == 1:
        parts.append("same DOB")
    elif x[F_DOB] == 0:
        parts.append("DOB differs")
    if x[F_SSN] == 1:
        parts.append("same SSN")
    elif x[F_SSN] == 0:
        parts.append("SSN differs")
    parts += [c for c in (contact_email, contact_phone) if c]
    return "; ".join(parts)


def evidence_many(X, contact_email, contact_phone, street):
    """evidence() for many pairs at once - the same text, built column by column (a call per
    link took 4.5 s of a million-record run)."""
    n = len(X)
    if not n:
        return np.zeros(0, dtype=object)
    fmt = lambda col: np.char.mod("%.2f", X[:, col].astype(float))  # noqa: E731  (as f"{x:.2f}")
    parts = [np.char.add(np.char.add(np.char.add("names ", fmt(F_FIRST)), "/"), fmt(F_LAST))]
    street = np.asarray(street, dtype=bool)
    st = X[:, F_STREET] > 0.95
    addr = street & (X[:, F_NUM] == 1) & st
    parts.append(np.where(addr, np.where(X[:, F_UNIT] == 1, "same address and unit", "same address"),
                          np.where(street & st, "same street", "")))
    parts.append(np.where(X[:, F_DOB] == 1, "same DOB", np.where(X[:, F_DOB] == 0, "DOB differs", "")))
    parts.append(np.where(X[:, F_SSN] == 1, "same SSN", np.where(X[:, F_SSN] == 0, "SSN differs", "")))
    parts += [np.asarray(contact_email, dtype=str), np.asarray(contact_phone, dtype=str)]
    # "; "-joined, skipping empty parts (zip over lists: faster than a 2-D object array or numpy's
    # string functions, measured)
    return np.array(["; ".join(filter(None, row)) for row in zip(*(np.asarray(q).tolist() for q in parts))],
                    dtype=object)


def _both_streets(df, i, j):
    st = df.addr_street.to_numpy()
    return (st[np.asarray(i)] != "") & (st[np.asarray(j)] != "")


def _contacts(df, profiles, field, i, j):
    """'same email' / 'same email (household: little weight)' / '' for each pair, vectorised."""
    v = df[field].to_numpy(dtype=object)
    vi = v[np.asarray(i)]
    # classes only for the records in these pairs (not the whole file), through one mapping
    classes = pd.Series({k: t[0] for k, t in profiles[field].items()}, dtype=object)
    ci = pd.Series(vi, dtype=object).map(classes).fillna("personal").to_numpy(dtype=object)
    ci = np.where(vi == "", "", ci)                 # (pandas may hand back a read-only array)
    same = (vi != "") & (vi == v[np.asarray(j)])
    label = np.where(ci == "personal", f"same {field}",
                     np.char.add(np.char.add(f"same {field} (", ci.astype("U")), ": little weight)"))
    return np.where(same, label, "")


def _bridges(n_nodes, edges):
    """Bridge flag per edge. A customer of two records is one link, trivially a bridge; only
    larger customers need the graph search."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    flag = np.ones(len(edges), bool)
    if not len(edges):
        return flag
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
    _, comp = connected_components(g, directed=False)
    size = np.bincount(comp)
    # a component whose links form a tree (nodes - 1 edges) has every link a bridge; only those
    # with a cycle need the graph search (networkx's was 6 s of a million-record run)
    n_edges = np.bincount(comp[edges[:, 0]], minlength=len(size))
    cyclic = n_edges > size - 1
    big = (size[comp[edges[:, 0]]] > 2) & cyclic[comp[edges[:, 0]]]
    if big.any():
        gx = nx.Graph()
        gx.add_edges_from(map(tuple, edges[big].tolist()))
        br = {frozenset(e) for e in nx.bridges(gx)}
        flag[big] = [frozenset(e) in br for e in map(tuple, edges[big].tolist())]
    return flag


def links(df, profiles, pairs, X, score, threshold, keys, customer_id):
    """DataFrame of accepted links: a, b, customer_id, score, is_bridge, evidence. X: the pairs'
    features, or a function giving the features of the pairs at some row indices."""
    keep = np.flatnonzero(score > threshold)
    e = pairs[keep].astype(np.int64)
    i, j = e[:, 0], e[:, 1]
    ce, cp = _contacts(df, profiles, "email", i, j), _contacts(df, profiles, "phone", i, j)
    out = pd.DataFrame({
        "a": keys[i], "b": keys[j], "customer_id": np.asarray(customer_id)[i].astype(int),
        "score": np.round(score[keep].astype(float), 4),
        "is_bridge": _bridges(len(df), e),
        "evidence": evidence_many(X(keep) if callable(X) else X[keep], ce, cp, _both_streets(df, i, j))})
    return out.sort_values("score", kind="stable").reset_index(drop=True)  # review weakest first


def _block_keys(df, i, profiles):
    """The default blocker's keys for record i (core.block), with the fields that make them."""
    r = df.iloc[i]
    keys = [(("nm", r.last_name, r.first_name[:3]), "surname + first 3 letters of first name"),
            (("ns", r.last_name, r.addr_street), "surname + street"),
            (("ad", r.addr_number, r.addr_street), "house number + street"),
            (("fn", r.first_name, r.addr_street), "first name + street")]
    for f in ("email", "phone"):
        v = r[f]
        if v and profiles[f].get(v, ("personal",))[0] != "junk":
            keys.append(((f, v), f))
    return keys


def why(state_dir, a, b):
    """Why two records were or were not matched, from a run's or add's output folder: whether
    blocking ever compared them (and which shared keys were skipped as oversized blocks), the
    evidence and score against the threshold, overrides, and the customers they ended up in."""
    import json
    from pathlib import Path
    from . import io, schema, stewardship
    from .blocking import candidates_for, saved_run_large_blocks
    from .core import MAX_BLOCK, features, profile_contacts
    from .models import load_model, p_match
    state = Path(state_dir) / "state"
    df = pd.read_parquet(state / "records.parquet")
    customer = df.pop("_customer").to_numpy()
    cfg = json.loads((state / "config.json").read_text())
    model = load_model(state)
    keys = io.key(df)
    pos = {k: i for i, k in enumerate(keys)}
    for k in (a, b):
        if k not in pos:
            raise io.ConfigError(f"{k} is not in {state_dir} (keys look like source:record_id)")
    i, j = sorted((pos[a], pos[b]))
    profiles = {f: profile_contacts(df, f) for f in ("email", "phone")}
    fopts = schema.feature_options_for(df, cfg)
    scorer = None
    if model.get("match_weight_model"):
        from .matchweight import load_scorer
        scorer = load_scorer(df, profiles, model["match_weight_model"])

    def featurise(p):
        f = features(df, p, profiles, **fopts)
        return f if scorer is None else np.column_stack([f, scorer.weights(p)])
    pairs, _, report, _ = candidates_for(df, profiles, cfg, rules=model.get("rules"))
    pairs = saved_run_large_blocks(cfg, report, lambda p: p_match(model["forest"], featurise(p)),
                                   float(model["threshold"]), pairs)
    compared = bool(len(pairs)) and bool(((pairs[:, 0] == i) & (pairs[:, 1] == j)).any())
    limit = schema.max_block(cfg) or MAX_BLOCK
    shared = []
    ki, kj = dict(_block_keys(df, i, profiles)), dict(_block_keys(df, j, profiles))
    common = [k for k in ki if k in kj and all(k[1:])]
    if common:
        cols = {"nm": (df.last_name, df.first_name.str[:3]), "ns": (df.last_name, df.addr_street),
                "ad": (df.addr_number, df.addr_street), "fn": (df.first_name, df.addr_street)}
        sizes = {}
        for key in common:
            if key[0] in cols:
                x, y = cols[key[0]]
                sizes[key] = int(((x == key[1]) & (y == key[2])).sum())
            else:
                sizes[key] = int((df[key[0]] == key[1]).sum())
        for key in common:
            shared.append({"key": ki[key], "block_size": sizes[key], "skipped_as_too_big": sizes[key] > limit})
    x = featurise(np.array([[i, j]]))
    score = float(p_match(model["forest"], x)[0])
    th = float(model["threshold"])
    must, cannot = stewardship.load_overrides(cfg, state_dir, keys)
    decision = ("same (override)" if any((m == [i, j]).all() for m in must) else
                "different (override)" if any((c == [i, j]).all() for c in cannot) else "")
    ce = _contacts(df, profiles, "email", np.array([i]), np.array([j]))[0]
    cp = _contacts(df, profiles, "phone", np.array([i]), np.array([j]))[0]
    same_customer = bool(customer[i] == customer[j])
    if decision:
        reason = f"a steward's decision: {decision}"
    elif not compared:
        reason = ("never compared: " + ("every key they share was skipped as an oversized block"
                                        if shared and all(s["skipped_as_too_big"] for s in shared)
                                        else "they share no blocking key"))
    elif score > th and same_customer:
        reason = f"matched directly: score {score:.3f} above the threshold {th:.3f}"
    elif score > th:
        reason = (f"scored {score:.3f}, above the threshold {th:.3f}, but the link was cut - by a "
                  "cannot-link override, the max_cluster size cap or re-clustering")
    else:
        reason = f"compared but scored {score:.3f}, below the threshold {th:.3f}"
    if same_customer and score <= th and not decision:
        reason += "; they are in the same customer through other records (a chain of links)"
    return {"a": a, "b": b, "same_customer": same_customer,
            "customer_ids": [int(customer[pos[a]]), int(customer[pos[b]])],
            "reason": reason, "compared_by_blocking": compared, "shared_blocking_keys": shared,
            "score": round(score, 4), "threshold": th, "override": decision,
            "evidence": evidence(x[0], ce, cp, bool(_both_streets(df, [i], [j])[0]))}
