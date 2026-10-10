"""PoC: real-time matching against a saved custmatch run, and its gotchas (docs/report/12-the-service.md, "The service").

Setup: the sparse Australian demo (examples/demo), 10% of records held back as "new". A model is
trained on the other 90% (`custmatch run`); the held-back records then arrive one at a time.

  1. Baseline: `custmatch add` for one record - it recomputes contact profiles and blocking over
     the whole table on every call.
  2. An online matcher: blocking keys in an in-memory index, features for the new record's
     candidates only, the saved forest and threshold, union-find over customers. Latency per record.
  3. Equivalence: are the online candidates and feature rows the ones a batch run computes for the
     same records? (Name rarity and contact classes are file-wide statistics.)
  4. Consistency: after all arrivals, do the customers equal a batch run over all records with the
     same model (`custmatch run --model`)?
  5. Concurrency: two records of one new person arriving together, with and without one writer.

    python research/poc/realtime.py
"""
import json
import shutil
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from custmatch import incremental, io, pipeline, schema
from custmatch.core import block, features, profile_contacts, record_arrays

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples/demo"


def split_demo(tmp):
    """The demo's sources split 90 / 10: base files and new-record files, same config shape."""
    cfg = json.loads((DEMO / "config.json").read_text())
    base, new = json.loads(json.dumps(cfg)), json.loads(json.dumps(cfg))
    rng = np.random.default_rng(0)
    for b, nw in zip(base["sources"], new["sources"]):
        d = pd.read_csv(DEMO / b["path"], dtype=str, keep_default_na=False)
        hold = rng.random(len(d)) < 0.10
        d[~hold].to_csv(tmp / f"base_{b['path']}", index=False)
        d[hold].to_csv(tmp / f"new_{b['path']}", index=False)
        b["path"], nw["path"] = f"base_{b['path']}", f"new_{nw['path']}"
    full = json.loads(json.dumps(cfg))
    for s in full["sources"]:
        s["path"] = str((DEMO / s["path"]).resolve())
    for name, c in (("base.json", base), ("new.json", new), ("full.json", full)):
        (tmp / name).write_text(json.dumps(c))
    lab = pd.read_csv(DEMO / "labels.csv", dtype=str, keep_default_na=False)
    base_keys = set(io.key(io.load(tmp / "base.json")[0]))
    lab[lab.a.isin(base_keys) & lab.b.isin(base_keys)].to_csv(tmp / "labels.csv", index=False)


# ------------------------------------------------------------------ the online matcher
def record_keys(df, profiles, extra_keys):
    """(key, record) for every blocking key core.block uses - re-implemented per record, as an
    online index must. Gotcha: this must agree with core.block exactly (checked below)."""
    from custmatch import country
    from custmatch.blocking import Keys
    out = list(country.blocking_keys(df, profiles))
    kk = Keys(df)
    for m, cols in enumerate(extra_keys):
        vals = [kk.get(c if ":" in c else f"{c}:exact") for c in cols]
        out += [(("x", m) + tuple(v), i) for i, v in enumerate(zip(*vals)) if all(v)]
    for i, r in enumerate(df.itertuples(index=False)):
        out += [(("nm", r.last_name, r.first_name[:3]), i), (("ns", r.last_name, r.addr_street), i),
                (("ad", r.addr_number, r.addr_street), i), (("fn", r.first_name, r.addr_street), i)]
        for field in ("email", "phone"):
            v = getattr(r, field)
            if v and profiles[field].get(v, ("personal",))[0] != "junk":
                out.append(((field, v), i))
    return out


class OnlineMatcher:
    def __init__(self, state_dir, lock=True):
        st = Path(state_dir) / "state"
        self.model = pd.read_pickle(st / "model.pkl")
        self.cfg = json.loads((st / "config.json").read_text())
        rec = pd.read_parquet(st / "records.parquet")
        self.customer = list(rec.pop("_customer").astype(int))
        self.df = rec
        self.cols = list(rec.columns)
        self.rows = rec.to_dict("records")              # one dict per record: append is O(1)
        self.profiles = {f: profile_contacts(rec, f) for f in ("email", "phone")}
        self.extra = schema.extra_keys(self.cfg)
        self.index = defaultdict(list)
        for k, i in record_keys(rec, self.profiles, self.extra):
            self.index[k].append(i)
        self.counts = {f: Counter(rec[f]) for f in ("first_name", "last_name")}
        self.max_block = schema.max_block(self.cfg) or 60
        self.fopts = schema.feature_options(self.cfg)
        self.lock = threading.Lock() if lock else None
        self.next_customer = max(self.customer) + 1

    def _candidates(self, row):
        one = pd.DataFrame([row], columns=self.cols)
        cand = set()
        for k, _ in record_keys(one, self.profiles, self.extra):
            idx = self.index.get(k, [])
            if 0 < len(idx) < self.max_block:           # the new record makes it len + 1 <= max_block
                cand.update(idx)
        return sorted(cand)

    def _score(self, row, cand):
        """Features for (candidate, new) on a frame of just those records, with the file-wide
        name counts put back - record_arrays would otherwise count names within the mini frame."""
        mini = pd.DataFrame([self.rows[i] for i in cand] + [row], columns=self.cols)
        pairs = np.array([[k, len(cand)] for k in range(len(cand))])
        rec = record_arrays(mini)
        for f in ("first_name", "last_name"):
            c = self.counts[f]
            rec[f"{f}_count"] = np.array([c[v] + (v == row[f]) for v in mini[f]], dtype=float)
        rec["n_records"] = len(self.rows) + 1           # rarity is relative to the file, not the frame
        X = features(mini, pairs, self.profiles, **(self.fopts | {"records": rec}))
        from custmatch.models import p_match
        return p_match(self.model["forest"], X), X

    def add(self, row, pause=0.0):
        if self.lock:
            with self.lock:
                return self._add(row, pause)
        return self._add(row, pause)

    def _add(self, row, pause):
        cand = self._candidates(row)
        p, _ = self._score(row, cand) if cand else (np.zeros(0), None)
        time.sleep(pause)                               # widen the race window (test 5)
        linked = sorted({self.customer[c] for c, s in zip(cand, p) if s > self.model["threshold"]})
        i = len(self.rows)
        self.rows.append(row)
        if not linked:
            cid = self.next_customer
            self.next_customer += 1
        else:                                           # merge every customer the record links
            cid = linked[0]
            if len(linked) > 1:
                gone = set(linked[1:])
                self.customer = [cid if c in gone else c for c in self.customer]
        self.customer.append(cid)
        one = pd.DataFrame([row], columns=self.cols)
        for k, _ in record_keys(one, self.profiles, self.extra):
            self.index[k].append(i)
        for f in ("first_name", "last_name"):
            self.counts[f][row[f]] += 1
        return cid


def agreement(labels_a, labels_b):
    """Share of records whose customer holds exactly the same records in both clusterings."""
    def groups(lab):
        g = defaultdict(set)
        for i, c in enumerate(lab):
            g[c].add(i)
        return [frozenset(g[c]) for c in lab]
    a, b = groups(labels_a), groups(labels_b)
    return float(np.mean([x == y for x, y in zip(a, b)]))


def main():
    out = {}
    tmp = Path(tempfile.mkdtemp())
    try:
        split_demo(tmp)
        pipeline.run(tmp / "base.json", tmp / "labels.csv", tmp / "base_run", log=lambda *_: None)
        new_df, _ = io.load(tmp / "new.json")
        new_rows = new_df.to_dict("records")
        out["records"] = {"base": len(pd.read_parquet(tmp / "base_run/state/records.parquet")), "new": len(new_rows)}

        # 1. custmatch add, one record
        one_cfg = json.loads((tmp / "new.json").read_text())
        src = one_cfg["sources"][0]
        pd.read_csv(tmp / src["path"], dtype=str, keep_default_na=False).head(1).to_csv(tmp / "one.csv", index=False)
        one_cfg["sources"] = [dict(src, path="one.csv")]
        (tmp / "one.json").write_text(json.dumps(one_cfg))
        t0 = time.time()
        incremental.add(tmp / "base_run", tmp / "one.json", tmp / "add_one")
        out["1_add_one_record_seconds"] = round(time.time() - t0, 2)

        # 2. online matcher: build, then every new record one at a time
        t0 = time.time()
        om = OnlineMatcher(tmp / "base_run")
        build = time.time() - t0
        lat = []
        for row in new_rows:
            s = time.perf_counter()
            om.add(row)
            lat.append(time.perf_counter() - s)
        out["2_online"] = {"index_build_seconds": round(build, 2), "keys_indexed": len(om.index),
                           "latency_ms_p50": round(1000 * np.percentile(lat, 50), 1),
                           "latency_ms_p95": round(1000 * np.percentile(lat, 95), 1),
                           "latency_ms_max": round(1000 * max(lat), 1)}

        # 3. equivalence with batch for the same records (base + all new)
        full = pd.DataFrame(om.rows, columns=om.cols)
        prof_full = {f: profile_contacts(full, f) for f in ("email", "phone")}
        batch_pairs, _ = block(full, prof_full, extra_keys=om.extra, max_block=om.max_block)
        n_base = out["records"]["base"]
        touching = {tuple(p) for p in batch_pairs.tolist() if p[1] >= n_base and p[0] < p[1]}
        om2 = OnlineMatcher(tmp / "base_run")
        online_pairs, feat_diff, prof_changed = set(), [], 0
        for k, row in enumerate(new_rows):
            cand = om2._candidates(row)
            i = n_base + k
            online_pairs |= {(c, i) for c in cand}
            if cand:
                _, X_on = om2._score(row, cand)
                X_b = features(full, np.array([[c, i] for c in cand]), prof_full,
                               **schema.feature_options_for(full, om2.cfg))
                feat_diff.append(float(np.abs(X_on - X_b).max()))
            om2.add(row)
        for f in ("email", "phone"):
            prof_changed += sum(prof_full[f].get(v, ("personal",))[0] != om2.profiles[f].get(v, ("personal",))[0]
                                for v in set(full[f]) if v)
        out["3_equivalence"] = {
            "batch_pairs_touching_new": len(touching), "online_candidate_pairs": len(online_pairs),
            "missing_online": len(touching - online_pairs), "extra_online": len(online_pairs - touching),
            "max_feature_difference": round(max(feat_diff), 4),
            "pairs_with_a_feature_difference": int(sum(d > 1e-6 for d in feat_diff)),
            "contact_values_classed_differently_once_new_records_arrive": prof_changed}

        # 4. consistency with a batch run over all records, same model
        full_cfg = tmp / "full.json"
        pipeline.run(full_cfg, None, tmp / "batch_all", log=lambda *_: None, model_dir=tmp / "base_run")
        c = pd.read_csv(tmp / "batch_all/clusters.csv", dtype=str)
        pos = dict(zip(c.source + ":" + c.record_id, c.customer_id))
        keys = io.key(full)
        batch_labels = [pos[k] for k in keys]
        incremental.add(tmp / "base_run", tmp / "new.json", tmp / "add_all")
        a = pd.read_csv(tmp / "add_all/clusters.csv", dtype=str)
        apos = dict(zip(a.source + ":" + a.record_id, a.customer_id))
        out["4_consistency_with_batch"] = {
            "online_one_at_a_time": round(agreement(om.customer, batch_labels), 4),
            "custmatch_add_one_batch": round(agreement([apos[k] for k in keys], batch_labels), 4)}

        # 5. two records of one new person at once
        twin = dict(new_rows[0], first_name="zephyrine", last_name="quillfeather", email="zq@nowhere.example",
                    phone="", addr_number="", addr_street="", first_roots="zephyrine", email_user="zq")
        res = {}
        for locked in (False, True):
            m = OnlineMatcher(tmp / "base_run", lock=locked)
            a_row = dict(twin, record_id="rt-a")
            b_row = dict(twin, record_id="rt-b")
            got = {}
            ts = [threading.Thread(target=lambda r=r, k=k: got.__setitem__(k, m.add(r, pause=0.2)))
                  for k, r in (("a", a_row), ("b", b_row))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            res["one writer (lock)" if locked else "no lock"] = "same customer" if got["a"] == got["b"] \
                else "TWO customers for one person"
        out["5_concurrent_arrivals"] = res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
