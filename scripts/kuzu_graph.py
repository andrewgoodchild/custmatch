"""Port the match graph into Kuzu, with edge provenance, and cluster with graph queries.

Why bother, when scipy's connected_components already does the clustering: the graph store
buys three things scipy cannot.

  1. Provenance - every edge carries the features and score that created it, so "why were
     these two records merged?" is a query, not a re-run.
  2. Clustering as a query - Kuzu's `algo` extension runs weakly-connected-components over a
     filtered projection, so the threshold becomes a WHERE clause.
  3. Structure-aware clustering - a cluster held together by a single weak edge (a BRIDGE) is
     exactly the percolation failure that collapsed 8,200 records into one blob. Bridges are a
     graph property, so we can find them and refuse to merge across them.

Runs on the person-record split from `person_em_build.py`, where plain connected components
scored 0.45 clustered F1 at the pairwise-optimal threshold.
"""
import json
import shutil
from pathlib import Path

import kuzu
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_rand_score

pd.set_option("future.infer_string", False)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/person_em"
DB_PATH = ROOT / "data/kuzu/match_graph"
STAGE = ROOT / "data/kuzu/stage"

FEATURES = ["first_name_sim", "surname_sim", "full_name_sim", "full_name_lev", "dob_exact",
            "dob_sim", "birthplace_exact", "birthplace_sim", "postcode_exact",
            "postcode_prefix", "gender_exact", "occupation_exact"]
LOW, BRIDGE_MIN, MIN_COMPONENT = 0.30, 0.90, 5


def partition_scores(pred, true, n_true_all):
    def n2(c):
        c = np.asarray(c, dtype=np.float64)
        return float(np.sum(c * (c - 1) / 2))
    _, pi = np.unique(pred, return_inverse=True)
    _, ti = np.unique(true, return_inverse=True)
    sizes = np.bincount(pi)
    m = coo_matrix((np.ones(len(pi)), (pi, ti))).tocsr()
    tp, total = n2(m.data), n2(sizes)
    p = tp / total if total else 0.0
    r = tp / n_true_all
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0,
            "clusters": int(len(sizes)), "largest_cluster": int(sizes.max()),
            "adjusted_rand": round(adjusted_rand_score(true, pred), 4)}


def build_db():
    from splink import splink_datasets
    src = splink_datasets.historical_50k.reset_index(drop=True)
    for c in ["first_name", "surname", "full_name", "dob", "birth_place",
              "postcode_fake", "occupation"]:
        src[c] = src[c].fillna("").astype(str).str.lower().str.strip()

    te = dict(np.load(DATA / "test.npz"))
    meta = dict(np.load(DATA / "test_meta.npz", allow_pickle=True))
    rf = dict(np.load(ROOT / "results/person_scores_rf.npz"))
    ditto = dict(np.load(ROOT / "results/person_scores_ditto.npz"))
    ids = meta["ids"]

    nodes = src.loc[ids, ["first_name", "surname", "dob", "birth_place", "postcode_fake",
                          "occupation"]].copy()
    nodes.insert(0, "rec_id", ids.astype(np.int64))
    nodes.insert(1, "person", meta["clusters"].astype(str))

    edges = pd.DataFrame({
        "src": te["left"].astype(np.int64), "dst": te["right"].astype(np.int64),
        "rf_prob": rf["prob"].astype(float), "ditto_prob": ditto["prob"].astype(float),
        "same_person": te["y"].astype(bool),
    })
    for k, name in enumerate(FEATURES):
        edges[name] = te["X"][:, k].astype(float)

    STAGE.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(STAGE / "nodes.parquet", index=False)
    edges.to_parquet(STAGE / "edges.parquet", index=False)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Kuzu may store the database as a file or a directory, plus .wal/.lock siblings
    for leftover in DB_PATH.parent.glob(DB_PATH.name + "*"):
        shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() else leftover.unlink()
    conn = kuzu.Connection(kuzu.Database(str(DB_PATH)))
    conn.execute("INSTALL algo; LOAD algo;")
    conn.execute("""CREATE NODE TABLE Record(
        rec_id INT64 PRIMARY KEY, person STRING, first_name STRING, surname STRING,
        dob STRING, birth_place STRING, postcode_fake STRING, occupation STRING)""")
    feat_cols = ", ".join(f"{f} DOUBLE" for f in FEATURES)
    conn.execute(f"""CREATE REL TABLE CANDIDATE(FROM Record TO Record,
        rf_prob DOUBLE, ditto_prob DOUBLE, same_person BOOLEAN, is_bridge BOOLEAN,
        {feat_cols})""")
    conn.execute(f"COPY Record FROM '{STAGE / 'nodes.parquet'}'")
    edges["is_bridge"] = False
    cols = ["src", "dst", "rf_prob", "ditto_prob", "same_person", "is_bridge"] + FEATURES
    edges[cols].to_parquet(STAGE / "edges.parquet", index=False)
    conn.execute(f"COPY CANDIDATE FROM '{STAGE / 'edges.parquet'}'")
    print(f"loaded {len(nodes):,} records and {len(edges):,} candidate edges into Kuzu",
          flush=True)
    return conn, nodes, edges, meta


def cluster_in_kuzu(conn, name, edge_filter, node_filter="n.rec_id >= 0"):
    """Weakly connected components over a filtered projection - the threshold is a WHERE."""
    conn.execute(f"CALL PROJECT_GRAPH('{name}', "
                 f"{{'Record': '{node_filter}'}}, {{'CANDIDATE': '{edge_filter}'}})")
    df = conn.execute(
        f"CALL weakly_connected_components('{name}') RETURN node.rec_id AS rec_id, group_id"
    ).get_as_df()
    return df.set_index("rec_id").group_id


def bridge_aware(edges, prob_col, low=LOW, bridge_min=BRIDGE_MIN, rounds=6):
    """Drop weak bridges: an edge whose removal splits a component is a single point of
    failure, so it has to clear a much higher bar than an edge inside a dense group."""
    keep = edges[prob_col].to_numpy() > low
    removed_total = 0
    for _ in range(rounds):
        g = nx.Graph()
        g.add_nodes_from(pd.unique(edges[["src", "dst"]].to_numpy().ravel()))
        sub = edges[keep]
        g.add_edges_from(zip(sub.src, sub.dst))
        comp_size = {}
        for comp in nx.connected_components(g):
            for node in comp:
                comp_size[node] = len(comp)
        weak = set()
        for u, v in nx.bridges(g):
            if comp_size[u] > MIN_COMPONENT:
                weak.add((u, v))
                weak.add((v, u))
        if not weak:
            break
        drop = np.array([(s, d) in weak for s, d in zip(edges.src, edges.dst)])
        drop &= keep & (edges[prob_col].to_numpy() < bridge_min)
        if not drop.any():
            break
        keep &= ~drop
        removed_total += int(drop.sum())
    return keep, removed_total


def labels_from_keep(edges, keep, ids):
    g = nx.Graph()
    g.add_nodes_from(ids)
    sub = edges[keep]
    g.add_edges_from(zip(sub.src, sub.dst))
    lab = {}
    for i, comp in enumerate(nx.connected_components(g)):
        for node in comp:
            lab[node] = i
    return np.array([lab[i] for i in ids])


def main():
    conn, nodes, edges, meta = build_db()
    ids, true = meta["ids"], meta["clusters"]
    n_true_all = float(meta["n_true_pairs"])
    out = {}

    # --- clustering as a Kuzu query, at two thresholds
    for th in (0.30, 0.95):
        s = cluster_in_kuzu(conn, f"G{int(th * 100)}", f"r.rf_prob > {th}")
        lab = s.reindex(ids).to_numpy()
        out[f"kuzu_wcc_rf_{th}"] = partition_scores(lab, true, n_true_all)
        print(f"Kuzu WCC, rf_prob > {th}: {out[f'kuzu_wcc_rf_{th}']}", flush=True)

    # --- structure-aware: refuse to merge across weak bridges
    for col in ("rf_prob", "ditto_prob"):
        keep, removed = bridge_aware(edges, col)
        lab = labels_from_keep(edges, keep, ids)
        r = partition_scores(lab, true, n_true_all)
        r["bridges_removed"] = removed
        out[f"bridge_aware_{col}"] = r
        print(f"bridge-aware ({col}, keep>{LOW}, bridges must clear {BRIDGE_MIN}): "
              f"removed {removed:,} -> {r}", flush=True)
        # write the flags back so the graph records why an edge was cut
        if col == "rf_prob":
            flags = pd.DataFrame({"src": edges.src, "dst": edges.dst,
                                  "is_bridge": (edges.rf_prob > LOW) & ~keep})
            flags.to_parquet(STAGE / "bridges.parquet", index=False)
            conn.execute(
                f"LOAD FROM '{STAGE / 'bridges.parquet'}' WITH * "
                "MATCH (a:Record)-[e:CANDIDATE]->(b:Record) "
                "WHERE a.rec_id = src AND b.rec_id = dst SET e.is_bridge = is_bridge")
            print(f"flagged {int(flags.is_bridge.sum()):,} edges as cut bridges in the graph",
                  flush=True)

    # --- provenance: what actually holds a wrong merge together
    q = conn.execute("""
        MATCH (a:Record)-[e:CANDIDATE]->(b:Record)
        WHERE e.rf_prob > 0.30 AND e.same_person = False AND e.is_bridge = True
        RETURN a.first_name AS a_first, a.surname AS a_last, a.dob AS a_dob,
               b.first_name AS b_first, b.surname AS b_last, b.dob AS b_dob,
               e.rf_prob AS prob, e.surname_sim AS surname_sim, e.dob_exact AS dob_exact,
               e.postcode_exact AS postcode_exact, e.occupation_exact AS occ
        ORDER BY e.rf_prob DESC LIMIT 5""").get_as_df()
    print("\nfalse merges that were bridges (the single edges joining two groups):")
    print(q.to_string(index=False), flush=True)
    out["example_false_bridges"] = q.to_dict("records")

    # --- hub query: which records attract the most candidate edges
    hubs = conn.execute("""
        MATCH (a:Record)-[e:CANDIDATE]->(b:Record) WHERE e.rf_prob > 0.30
        RETURN a.surname AS surname, count(*) AS edges
        ORDER BY edges DESC LIMIT 5""").get_as_df()
    print("\ntop surnames by accepted-edge count (hub check):")
    print(hubs.to_string(index=False), flush=True)
    out["hub_surnames"] = hubs.to_dict("records")

    # --- the hub query above says blank surnames and the token "baronet" dominate the
    # accepted edges. Suppressing those nodes is the fix the graph store diagnosed.
    hub_tokens = ["", "baronet"]
    node_filter = " AND ".join(f'n.surname <> "{t}"' for t in hub_tokens)
    s = cluster_in_kuzu(conn, "Ghub", "r.rf_prob > 0.30", node_filter)
    lab = s.reindex(ids)
    # nodes excluded from the projection are their own cluster (a review queue)
    lab = lab.fillna(pd.Series(-np.arange(1, len(ids) + 1), index=ids)).to_numpy()
    out["hub_suppressed_rf_0.30"] = partition_scores(lab, true, n_true_all)
    print(f"\nhub-suppressed (blank/baronet surnames excluded), rf_prob > 0.30: "
          f"{out['hub_suppressed_rf_0.30']}", flush=True)

    keep, removed = bridge_aware(edges, "rf_prob")
    hub_nodes = set(nodes.loc[nodes.surname.isin(hub_tokens), "rec_id"])
    keep &= ~(edges.src.isin(hub_nodes) | edges.dst.isin(hub_nodes)).to_numpy()
    lab = labels_from_keep(edges, keep, ids)
    r = partition_scores(lab, true, n_true_all)
    r["bridges_removed"] = removed
    out["hub_suppressed_plus_bridges_rf"] = r
    print(f"hub-suppressed + bridge-aware, rf_prob > 0.30: {r}", flush=True)

    json.dump(out, open(ROOT / "results/kuzu_graph.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
