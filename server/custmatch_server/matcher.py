"""Real-time matching against the current run (server/README.md, "How it works"; measured in docs/report/12-the-service.md, "The service").

One writer per database: a process may add records only while it holds the lease; lookups need no
lease. The matcher starts from the current run's saved state (records, model, threshold), replays
the API records the run does not include yet, and keeps a blocking index in memory.

The index is compact: the run's records stay one columnar DataFrame, and every blocking key - the
same keys the batch run blocks on (core.blocking_families) - is a 64-bit hash in a sorted numpy
array next to its record's row number, searched with searchsorted: 16 bytes an entry. Records
added since the start go into small dict overlays. (The first version kept a dict per record and a
dict of key tuples: 1.4 GB and 24 s to start at a million records.)

The batch run's rules, online: a steward's overrides ("same" joins or merges, "different" keeps
apart; splitting a customer that already holds both waits for the batch run), trusted IDs (a
shared loyalty number joins), the twin guard, and duplicate-free sources (a record never joins or
merges a customer holding another record of its source). A "different", a duplicate-free source
and a merge's own conflicts win over a "same", as in the batch run. Left to the batch run: cookie
stitching, max_cluster re-cuts, and opt-in rule blocking ("blocking": "fixed" or "learned"; the
matcher uses the default keys).

Merges are written to realtime_merges and replayed at start, so a restart keeps them."""
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import insert, select

from custmatch import io, schema, stewardship
from custmatch.core import _hash_strings, blocking_hashes, features, profile_contacts, record_arrays
from custmatch.models import load_model, p_match

from . import db, loader
from .schema import overrides, realtime_merges, realtime_moves, realtime_records, source_records, steward_log

LEASE = "matcher"


class StaleMatcher(Exception):
    """Another process wrote while this one did not hold the lease: its memory is out of date.
    The API rebuilds the matcher and retries."""
# a record joins a customer only if it matches at least this share of the customer's records
# (_incoherent): measured against "mean" and "veto" rules, server/replay_test.py --all
COHERENCE = 0.5
MERGE, MERGE_SCORE = "strong", 0.9    # how a record linking to several customers merges them (_plan;
                                      # server/replay_test.py compares the policies)


class HashIndex:
    """hash -> values: a sorted uint64 array with its values, plus a dict for later additions."""

    def __init__(self, hashes, values=None):
        order = np.argsort(hashes, kind="stable")
        self.h = np.ascontiguousarray(hashes[order])
        self.v = None if values is None else np.ascontiguousarray(values[order])
        self.extra = defaultdict(list)

    def get(self, h):
        lo, hi = np.searchsorted(self.h, h, "left"), np.searchsorted(self.h, h, "right")
        return self.v[lo:hi].tolist() + self.extra.get(int(h), [])

    def count(self, h):
        return int(np.searchsorted(self.h, h, "right") - np.searchsorted(self.h, h, "left")) + len(self.extra.get(int(h), ()))

    def add(self, h, v=None):
        self.extra[int(h)].append(v)

    def __len__(self):
        return len(self.h) + sum(map(len, self.extra.values()))


def small_proba(model, X):
    """P(match) for the few dozen pairs of one request. A random forest's predict_proba spends
    ~14 ms a call on its thread pool and per-tree checks at this size; each tree's compiled
    predict, averaged here, gives the same probabilities in about a millisecond. Other models,
    and forests with other classes, go through the library's p_match."""
    from sklearn.ensemble import RandomForestClassifier
    if not isinstance(model, RandomForestClassifier) or list(model.classes_) != [0, 1] or not len(X):
        return p_match(model, X)
    X = np.ascontiguousarray(X, dtype=np.float32)
    total = np.zeros(len(X))
    for tree in model.estimators_:
        v = tree.tree_.predict(X)                       # per-leaf class weights (fractions since sklearn 1.4)
        v = v[:, 0, :] if v.ndim == 3 else v
        total += v[:, 1] / v.sum(axis=1)
    return total / len(model.estimators_)


def forest_contributions(model, x):
    """Each feature's share of a random forest's P(match) for one pair - the forest's waterfall.
    In every tree the prediction moves from the root's match rate to the leaf's, one split at a
    time; each step is credited to the split's feature, and the trees are averaged. Returns
    (base rate, contributions per feature): base + contributions.sum() is the forest's score."""
    x = np.asarray(x, dtype=np.float32).ravel()
    total, base = np.zeros(len(x)), 0.0
    for est in model.estimators_:
        t = est.tree_
        v = t.value[:, 0, :] if t.value.ndim == 3 else t.value
        p1 = v[:, 1] / v.sum(axis=1)
        node = 0
        base += p1[0]
        while t.children_left[node] != -1:
            f = t.feature[node]
            nxt = t.children_left[node] if x[f] <= t.threshold[node] else t.children_right[node]
            total[f] += p1[nxt] - p1[node]
            node = nxt
    n = len(model.estimators_)
    return base / n, total / n


def _one_chunk(df):
    """Each Arrow-backed column as a single chunk: taking a few rows from a column of several
    chunks costs time in proportion to the column (~7 ms a request at a million records)."""
    import pyarrow as pa
    for c in df.columns:
        arr = df[c].array
        if hasattr(arr, "_pa_array") and arr._pa_array.num_chunks > 1:
            df[c] = pd.Series(type(arr)(pa.chunked_array([arr._pa_array.combine_chunks()])), index=df.index, name=c)
    return df


def _record_hash(source, record_id):
    return _hash_strings([f"{source}:{record_id}"])[0]


class Matcher:
    def __init__(self, eng, run, holder=None):
        """holder: the lease holder's name - one per process, so a process that reloads its
        matcher after a batch run keeps its lease (a new name would wait out the old lease)."""
        self.eng, self.run_id = eng, run["id"]
        result = Path(run["artifact_uri"])
        data_dir = os.environ.get("CUSTMATCH_DATA_DIR")
        if data_dir and not result.resolve().is_relative_to(Path(data_dir).resolve()):
            # the model is a pickle: loading one from elsewhere would run whatever it holds
            raise PermissionError(f"run {run['id']}'s files are outside CUSTMATCH_DATA_DIR: not loaded")
        st = result / "state"
        self.model = load_model(st)
        self.cfg = json.loads((st / "config.json").read_text())
        self.base_dir = result.parent / "inputs"
        rec = _one_chunk(pd.read_parquet(st / "records.parquet"))
        self.cust = rec.pop("_customer").to_numpy(dtype=np.int64, copy=True)   # writable: merges change it
        self.cust_added = []
        self.rec, self.added = rec, []
        self.cols = list(rec.columns)
        n = len(rec)
        self.profiles = {f: profile_contacts(rec, f) for f in ("email", "phone")}
        self.extra = schema.extra_keys(self.cfg)
        self.lsh = self.cfg.get("lsh_blocking")
        h, r = blocking_hashes(rec, self.profiles, self.extra, self.lsh)
        self.index = HashIndex(h, r)
        self.by_key = HashIndex(_hash_strings((rec.source + ":" + rec.record_id).to_numpy(dtype=object)),
                                np.arange(n, dtype=np.int64))
        self.counts = {f: HashIndex(_hash_strings(rec[f].to_numpy(dtype=object))) for f in ("first_name", "last_name")}
        self.max_block = schema.max_block(self.cfg) or 60
        self.fopts = schema.feature_options(self.cfg)
        self.lock = threading.Lock()
        self.holder = holder or db.holder_name()
        # new customers start above every ID the run's history ever used (merged-away ones too);
        # pending real-time IDs raise it further as they are replayed (_add, _set_customer)
        self.next_customer = max(int(self.cust.max()) if n else -1, stewardship.highest_issued(result)) + 1
        self.seen = self._watermark()                    # what this process has seen of others' writes
        # values the batch run blanked as placeholders
        self.fake_names, self.fake_addrs, self.fake_dobs = set(), set(), set()
        if self.cfg.get("placeholders") and (result / "placeholders.csv").exists():
            ph = pd.read_csv(result / "placeholders.csv", dtype=str, keep_default_na=False)
            self.fake_names = set(ph.name[ph.fake_name == "True"].str.strip())
            self.fake_addrs = set(ph.address[ph.fake_address == "True"].str.strip())
            if "fake_date_of_birth" in ph.columns:       # older runs did not check dates
                self.fake_dobs = set(ph.date_of_birth[ph.fake_date_of_birth == "True"].str.strip())
        self.merge, self.merge_score = MERGE, MERGE_SCORE
        self.coherence = COHERENCE
        self.lease_until = 0.0                           # renew the lease only when half its time is left
        # the batch run's rules (stewardship.py), online
        self.unique = stewardship.unique_sources(self.cfg, rec)
        self.twin = bool(self.cfg.get("twin_guard")) and "first_name" not in self.cfg.get("hash_fields", [])
        self.trusted_max = int(self.cfg.get("trusted_max", 5))
        self.trusted = {}
        for col in self.cfg.get("trusted_ids", []):
            if col in rec.columns:
                v = rec[col].astype(str).str.strip().to_numpy(dtype=object)
                has = np.flatnonzero(v != "")
                self.trusted[col] = HashIndex(_hash_strings(v[has]), has.astype(np.int64))
        self.must, self.cannot = defaultdict(set), defaultdict(set)     # record key -> keys
        for path in db.as_list(self.cfg.get("overrides")):
            if Path(path).exists():
                for a, b, d in pd.read_csv(path, dtype=str, keep_default_na=False)[["a", "b", "decision"]].values:
                    self._remember(a, b, d.strip().lower())
        with eng.connect() as c:
            for a, b, d in c.execute(select(overrides.c.a, overrides.c.b, overrides.c.decision)
                                     .where(overrides.c.status == "applied")       # not proposals awaiting approval
                                     .order_by(overrides.c.id)):
                self._remember(a, b, d)
        with eng.connect() as c:                         # API records the run does not include yet, in order
            pending = c.execute(select(realtime_records.c.source, realtime_records.c.record_id,
                                       realtime_records.c.customer_id, source_records.c.fields)
                                .join(source_records, (source_records.c.source == realtime_records.c.source)
                                      & (source_records.c.record_id == realtime_records.c.record_id))
                                .where(realtime_records.c.reconciled_run.is_(None))
                                .order_by(realtime_records.c.seq)).mappings().all()
            merges = c.execute(select(realtime_merges.c.created_at, realtime_merges.c.into_customer,
                                      realtime_merges.c.from_customer)
                               .where(realtime_merges.c.reconciled_run.is_(None))).all()
            moves = c.execute(select(realtime_moves.c.created_at, realtime_moves.c.source, realtime_moves.c.record_id,
                                     realtime_moves.c.to_customer)
                              .where(realtime_moves.c.reconciled_run.is_(None))).all()
        for r in pending:
            if self._index_of(f"{r['source']}:{r['record_id']}") is not None:
                continue                                 # already in the run (it read the record before reconciling)
            fields = db.as_json(r["fields"])
            self._add(self._normalise(r["source"], fields), int(r["customer_id"]))
        # then the merges and splits, in the order they happened
        events = [(t, "merge", (into, gone)) for t, into, gone in merges] + \
                 [(t, "move", (src, rid, to)) for t, src, rid, to in moves]
        for _, kind, args in sorted(events, key=lambda e: (e[0], e[1])):
            if kind == "merge":
                self._merge(int(args[0]), {int(args[1])})
            else:
                i = self._index_of(f"{args[0]}:{args[1]}")
                if i is not None:
                    self._set_customer(i, int(args[2]))

    # ---------------------------------------------------------------- records
    def __len__(self):
        return len(self.rec) + len(self.added)

    def _rows(self, idx):
        """Records by row number, as one DataFrame in that order."""
        n = len(self.rec)
        base = [i for i in idx if i < n]
        frame = self.rec.iloc[base] if base else self.rec.iloc[:0]
        if len(base) == len(idx):
            return frame.reset_index(drop=True)
        rows = {i: r for i, r in zip(base, frame.to_dict("records"))}
        rows.update({i: self.added[i - n] for i in idx if i >= n})
        return pd.DataFrame([rows[i] for i in idx], columns=self.cols)

    def _customer(self, i):
        n = len(self.rec)
        return int(self.cust[i]) if i < n else self.cust_added[i - n]

    def _key(self, i):
        n = len(self.rec)
        r = self.rec.iloc[i] if i < n else self.added[i - n]
        return f"{r['source']}:{r['record_id']}"

    # ---------------------------------------------------------------- records in, scored
    def _normalise(self, source, fields):
        """One record as the batch run would clean it: the library's own loader, from memory."""
        src = self._source(source)
        frame = pd.DataFrame([{c: str(fields.get(c, "")) for c in src["columns"].values()}])
        df, _ = io.load_config(self.cfg | {"sources": [src]}, self.base_dir, data={source: frame},
                               whole_config=False)
        if self.cfg.get("placeholders"):                 # as the batch run does: patterns, and its flagged values
            from custmatch import placeholders
            from custmatch.country import refresh_names
            df, _, _ = placeholders.blank(df)
            r = df.iloc[0]
            if f"{r.first_name} {r.last_name}".strip() in self.fake_names:
                df.loc[:, ["first_name", "middle_initial", "last_name"]] = ""
                df = refresh_names(df)
            if f"{r.addr_number} {r.addr_street}".strip() in self.fake_addrs:
                df.loc[:, ["addr_number", "addr_street", "addr_unit"]] = ""
            if r.date_of_birth and r.date_of_birth in self.fake_dobs:
                df.loc[:, "date_of_birth"] = ""
        return {c: (df.iloc[0][c] if c in df.columns else "") for c in self.cols}

    def _hashes(self, row):
        h, _ = blocking_hashes(pd.DataFrame([row], columns=self.cols), self.profiles, self.extra, self.lsh)
        return np.unique(h)

    def _candidates(self, row):
        cand = set()
        for h in self._hashes(row):
            idx = self.index.get(h)
            if 0 < len(idx) < self.max_block:
                cand.update(idx)
        return sorted(cand)

    def _features(self, mini, pairs, new_row=None):
        """Feature rows for pairs within `mini`, with the file-wide name counts (and the arriving
        record, not yet counted, added to them)."""
        rec = record_arrays(mini)
        for f in ("first_name", "last_name"):
            hs = _hash_strings(mini[f].to_numpy(dtype=object))
            extra = (mini[f] == new_row[f]).to_numpy() if new_row is not None else np.zeros(len(mini), bool)
            rec[f"{f}_count"] = np.array([self.counts[f].count(h) for h in hs], dtype=float) + extra
        rec["n_records"] = len(self) + (new_row is not None)
        X = features(mini, pairs, self.profiles, **(self.fopts | {"records": rec}))
        need = getattr(self.model["forest"], "n_features_in_", X.shape[1])
        if X.shape[1] != need:
            raise NotImplementedError(f"the model expects {need} features and real time builds {X.shape[1]}: "
                                      "\"match_weight_feature\" is not supported in real time yet")
        return X

    def feature_names(self):
        from custmatch.core import feature_names
        return feature_names(self.cols, self.fopts["extra_fields"], self.fopts["rich"])

    def explain(self, a, b):
        """Why two records score what they score: the features, and each one's share of the score."""
        ia, ib = self._index_of(a), self._index_of(b)
        if ia is None or ib is None:
            raise KeyError(f"not found: {a if ia is None else b}")
        mini = self._rows([ia, ib])
        X = self._features(mini, np.array([[0, 1]]))
        forest, th = self.model["forest"], self.model["threshold"]
        score = float(small_proba(forest, X)[0])
        twin = bool(self.twin and stewardship.twin_pairs(mini, np.array([[0, 1]]))[0])
        out = {"a": a, "b": b, "score": round(score, 4), "threshold": th,
               "twin_guard": twin, "a_customer": str(self._customer(ia)), "b_customer": str(self._customer(ib))}
        if hasattr(forest, "estimators_"):
            base, contrib = forest_contributions(forest, X[0])
            names = self.feature_names()
            out["base"] = round(float(base), 4)
            out["contributions"] = sorted(
                ({"feature": names[k] if k < len(names) else f"feature {k}", "value": round(float(X[0, k]), 3),
                  "contribution": round(float(c), 4)} for k, c in enumerate(contrib) if abs(c) >= 0.0005),
                key=lambda e: -abs(e["contribution"]))
        return out

    def _score(self, row, cand):
        mini = pd.concat([self._rows(cand), pd.DataFrame([row], columns=self.cols)], ignore_index=True)
        pairs = np.array([[k, len(cand)] for k in range(len(cand))])
        X = self._features(mini, pairs, new_row=row)
        p = small_proba(self.model["forest"], X)
        if self.twin:                                   # twins look like one person: kept below the threshold
            th = self.model["threshold"]
            p = np.where(stewardship.twin_pairs(mini, pairs), np.minimum(p, th * 0.999), p)
        return p

    def _decide(self, row):
        cand = self._candidates(row)
        p = self._score(row, cand) if cand else np.zeros(0)
        th = self.model["threshold"]
        best = {}                                       # customer -> its best link's score
        for c, sc in zip(cand, p):
            if sc > th:
                k = self._customer(c)
                best[k] = max(best.get(k, 0.0), float(sc))
        evidence = sorted(({"record": self._key(c), "customer_id": str(self._customer(c)), "score": round(float(s), 4)}
                           for c, s in zip(cand, p)), key=lambda e: -e["score"])[:5]
        key = f"{row['source']}:{row['record_id']}"
        rules = []
        forced = self._forced(row)
        if self.coherence:                              # fit the customer, not one sparse record of it
            for k in list(best):
                why = self._incoherent(row, k)
                if why:
                    rules.append(f"not joined to customer {k}: {why}")
                    best.pop(k)
        for c in forced:
            best[c] = 1.0
        for c in sorted(set(best), key=lambda k: -best[k]):   # customers the record may not join
            why = self._conflict([], c, new_key=key)
            if why:
                rules.append(f"kept apart from customer {c}: {why}")
                best.pop(c)
                forced.pop(c, None)
        rules += [f"joined customer {c}: {why}" for c, why in forced.items()]
        plan = self._plan(best, forced)
        # merging: each further customer must not conflict with those already joined
        out = plan[:1]
        for c in plan[1:]:
            why = self._conflict([int(x) for x in out], int(c), new_key=key)
            if why:
                rules.append(f"not merged with customer {c}: {why}")
            else:
                out.append(c)
        return out, evidence, float(p.max()) if len(p) else 0.0, rules

    COHERENCE_SAMPLE = 25

    def _incoherent(self, row, customer):
        """Why the record should not join `customer` although it links to one of its records, or
        None. A record joined the customer of its best link, so one strong link to one record of
        a customer could chain in another person; scored against the customer's records (up to
        COHERENCE_SAMPLE), it must match at least `coherence` of them. Over five replay seeds this
        lifted real-time precision 86.6 -> 90.0 for recall 75.7 -> 74.8, F1 80.8 -> 81.7, better
        on four of five seeds. A two-record customer still joins on one of its two records:
        requiring more than half cost too much recall (F1 79.4), as did a mean-score rule and a veto on
        any clear non-match (server/replay_test.py --all)."""
        members = self._members(customer)
        if len(members) < 2:
            return None
        if len(members) > self.COHERENCE_SAMPLE:
            members = list(np.random.default_rng(len(members)).choice(members, self.COHERENCE_SAMPLE, replace=False))
        p = self._score(row, members)
        if (p > self.model["threshold"]).mean() < self.coherence:
            return f"it matches {int((p > self.model['threshold']).sum())} of {len(p)} of the customer's records"
        return None

    def _plan(self, best, forced=()):
        """[customer to join, customers to merge into it...] from each linked customer's best score.
        The record joins the customer it links to most strongly. Linking to several customers
        merges them under the merge policy: "always" (any link above the threshold - one record
        then bridges two people, with none of the batch run's coherence checks), "strong" (only
        customers linked at merge_score or above), or "never" (left to the next batch run)."""
        ranked = sorted(best, key=lambda k: (-best[k], k))
        if not ranked:
            return []
        if self.merge == "always":
            others = ranked[1:]
        elif self.merge == "strong":
            others = [k for k in ranked[1:] if best[k] >= self.merge_score or k in forced]
        else:
            others = [k for k in ranked[1:] if k in forced]      # a "same" or trusted ID always merges
        return [str(k) for k in [ranked[0]] + others]

    def _add(self, row, customer_id):
        i = len(self)
        self.added.append(row)
        self.cust_added.append(int(customer_id))
        self.by_key.add(_record_hash(row["source"], row["record_id"]), i)
        for h in self._hashes(row):
            self.index.add(h, i)
        for f in ("first_name", "last_name"):
            self.counts[f].add(_hash_strings([row[f]])[0])
        for col, idx in self.trusted.items():           # a later record with this loyalty number joins it
            v = str(row.get(col, "")).strip()
            if v:
                idx.add(_hash_strings([v])[0], i)
        self.next_customer = max(self.next_customer, int(customer_id) + 1)

    def _set_customer(self, i, customer_id):
        n = len(self.rec)
        if i < n:
            self.cust[i] = customer_id
        else:
            self.cust_added[i - n] = customer_id
        self.next_customer = max(self.next_customer, int(customer_id) + 1)

    def _merge(self, into, gone):
        gone = {int(g) for g in gone}
        self.cust[np.isin(self.cust, np.fromiter(gone, dtype=np.int64))] = into
        self.cust_added = [into if c in gone else c for c in self.cust_added]

    # ---------------------------------------------------------------- the batch run's rules
    def _remember(self, a, b, decision):
        """A steward's decision on two record keys; a newer one on the same pair replaces it."""
        for d in (self.must, self.cannot):
            d[a].discard(b)
            d[b].discard(a)
        side = self.must if decision == "same" else self.cannot
        side[a].add(b)
        side[b].add(a)

    def _index_of(self, key):
        source, _, record_id = key.partition(":")
        for i in self.by_key.get(_record_hash(source, record_id)):
            if self._key(i) == key:                     # (a shared hash is checked against the key)
                return i
        return None

    def _members(self, customer):
        """Row numbers of a customer's records (a scan of a numpy array: ~1 ms at a million)."""
        n = len(self.rec)
        return np.flatnonzero(self.cust == customer).tolist() + \
            [n + k for k, c in enumerate(self.cust_added) if c == customer]

    def _conflict(self, group, other, new_key=None):
        """Why customers `group` (plus the arriving record) and customer `other` must stay apart, or
        None: a steward's "different" between them, or two records of one duplicate-free source."""
        if not self.cannot and not self.unique:
            return None
        mine = [i for c in group for i in self._members(c)]
        theirs = self._members(other)
        mine_keys = {self._key(i) for i in mine} | ({new_key} if new_key else set())
        their_keys = {self._key(i) for i in theirs}
        if any(self.cannot.get(k, set()) & their_keys for k in mine_keys):
            return "a steward's \"different\""
        if self.unique:
            src = lambda keys: {k.partition(":")[0] for k in keys} & self.unique  # noqa: E731
            if src(mine_keys) & src(their_keys):
                return "two records of a duplicate-free source"
        return None

    def _forced(self, row):
        """Customers this record must join: a steward's "same", or a shared trusted ID."""
        key = f"{row['source']}:{row['record_id']}"
        out = {}
        for other in self.must.get(key, ()):
            i = self._index_of(other)
            if i is not None:
                out[self._customer(i)] = f"a steward's \"same\" with {other}"
        for col, idx in self.trusted.items():
            v = str(row.get(col, "")).strip()
            if v:
                hits = idx.get(_hash_strings([v])[0])
                hits = [i for i in hits if str(self._rows([i])[col].iloc[0]).strip() == v]
                if 1 <= len(hits) <= self.trusted_max - 1:   # a value on more records is a placeholder
                    for i in hits:
                        out.setdefault(self._customer(i), f"trusted ID {col}")
        return out

    # ---------------------------------------------------------------- the API's operations
    def lookup(self, source, record_id):
        i = self._index_of(f"{source}:{record_id}")
        return None if i is None else str(self._customer(i))

    def match(self, source, record_id, fields, write=False):
        """The customer this record belongs to. write=True adds it (lease required): it joins the
        customer it links to, merges the customers if it links to several, or starts a new one."""
        given = dict(fields, **{self._id_column(source): record_id})
        row = self._normalise(source, given)
        if not write:
            linked, evidence, best, rules = self._decide(row)
            return {"customer_id": linked[0] if linked else None, "merges": linked[1:], "evidence": evidence,
                    "rules": rules, "provisional": True, "written": False}
        self._lease()
        with self.lock:                                 # one writer: decide and add atomically
            existing = self.lookup(source, record_id)
            if existing is not None:
                return {"customer_id": existing, "merges": [], "evidence": [], "provisional": False,
                        "written": False, "note": "record already known"}
            linked, evidence, best, rules = self._decide(row)
            cid = linked[0] if linked else str(self.next_customer)
            full = {k: str(given[k]) for k in loader.mapped_columns(self._source(source), given)}   # only what matching uses
            with self.eng.begin() as c:                 # the record and the decision, together
                loader.write_api_record(c, source, record_id, full)
                res = c.execute(insert(realtime_records).values(
                    source=source, record_id=record_id, customer_id=int(cid), score=best,
                    matched_against_run=self.run_id, created_at=db.now()))
                self._saw(records=res.inserted_primary_key[0])
                self._write_merges(c, int(cid), [int(x) for x in linked[1:]], f"record {source}:{record_id} links them")
            self._merge(int(cid), {int(x) for x in linked[1:]})
            self._add(row, cid)
        return {"customer_id": cid, "merges": linked[1:], "evidence": evidence, "rules": rules,
                "provisional": True, "written": True}

    def _lease(self):
        """Hold the write lease. Taking it back after a gap - free, expired, or another process's
        - means someone else may have written meanwhile: if so, this memory is stale."""
        if time.monotonic() > self.lease_until:
            taken, held = db.take_lease(self.eng, LEASE, self.holder, seconds=60, report=True)
            if not taken:
                raise PermissionError("another matcher holds the write lease")
            self.lease_until = time.monotonic() + 30
            if not held:
                now = self._watermark()
                if now[0] > self.seen[0] or now[1] > self.seen[1]:
                    raise StaleMatcher("another process wrote since this matcher loaded")

    def _watermark(self):
        """(highest real-time record seq, highest steward-log id) in the database."""
        from sqlalchemy import func
        with self.eng.connect() as c:
            return (c.execute(select(func.max(realtime_records.c.seq))).scalar() or 0,
                    c.execute(select(func.max(steward_log.c.id))).scalar() or 0)

    def _saw(self, records=None, log=None):
        self.seen = (max(self.seen[0], records or 0), max(self.seen[1], log or 0))

    def _write_merges(self, c, into, gone, reason):
        for g in gone:
            c.execute(insert(realtime_merges).values(into_customer=into, from_customer=g, reason=reason,
                                                     matched_against_run=self.run_id, created_at=db.now()))

    # ---------------------------------------------------------------- stewards' actions
    def _log(self, c, actor, action, a=None, b=None, customer_id=None, detail=None, note=None):
        res = c.execute(insert(steward_log).values(at=db.now(), actor=actor, action=action, a=a, b=b,
                                                   customer_id=None if customer_id is None else int(customer_id),
                                                   detail=None if detail is None else json.dumps(detail), note=note))
        self._saw(log=res.inserted_primary_key[0])

    def _model_score(self, a, b):
        try:
            return self.explain(a, b)["score"]
        except Exception:  # noqa: BLE001 - a record not here yet, a model without a breakdown
            return None

    def _effect(self, a, b, decision, keep=None):
        """What a decision would do now: (effect text, surviving customer, merged-away customer, merge?).
        keep: the customer to survive a merge (default: a's)."""
        ia, ib = self._index_of(a), self._index_of(b)
        ca = None if ia is None else self._customer(ia)
        cb = None if ib is None else self._customer(ib)
        if keep is not None and cb == keep:
            ca, cb = cb, ca
        if ca is None or cb is None:
            return "stored: a record is not here yet; it applies when it arrives", ca, cb, False
        if decision == "same" and ca != cb:
            why = self._conflict([ca], cb)
            if why:
                return f"stored, not applied now ({why}); the batch run decides", ca, cb, False
            sizes = (len(self._members(ca)), len(self._members(cb)))
            n = lambda k: f"{k} record" + ("" if k == 1 else "s")  # noqa: E731
            return f"merged customer {cb} ({n(sizes[1])}) into {ca} ({n(sizes[0])})", ca, cb, True
        if decision == "different" and ca == cb:
            return f"stored; customer {ca} is split at the next batch run (or split the record out now)", ca, cb, False
        return "stored; already so", ca, cb, False

    def _forget(self, a, b):
        for d in (self.must, self.cannot):
            d[a].discard(b)
            d[b].discard(a)

    def _apply(self, c, a, b, decision, actor, note, keep=None, why="a steward's"):
        """Write an applied decision (replacing the pair's applied one) and carry it out now."""
        c.execute(overrides.delete().where(overrides.c.a == a).where(overrides.c.b == b)
                  .where(overrides.c.status == "applied"))
        c.execute(insert(overrides).values(a=a, b=b, decision=decision, created_at=db.now(), created_by=actor,
                                           note=note, status="applied"))
        self._remember(a, b, decision)
        applied, ca, cb, merge = self._effect(a, b, decision, keep)
        if merge:
            self._write_merges(c, ca, [cb], f"{why} \"same\": {a} = {b}")
        return applied, ca, cb, merge

    def _propose(self, c, a, b, decision, actor, note):
        """Four-eyes: store a proposal (replacing the pair's earlier proposal, never its applied decision)."""
        c.execute(overrides.delete().where(overrides.c.a == a).where(overrides.c.b == b)
                  .where(overrides.c.status == "proposed"))
        c.execute(insert(overrides).values(a=a, b=b, decision=decision, created_at=db.now(), created_by=actor,
                                           note=note, status="proposed"))
        self._log(c, actor, "proposed", a, b or None, detail={"decision": decision}, note=note)
        return "proposed: applied when another steward approves it"

    def decide_override(self, a, b, decision, created_by=None, note=None, dry_run=False, propose=False, keep=None):
        """Record a steward's decision on two records (keys source:record_id) and apply it now
        where it can be: "same" merges their customers (unless a "different" or a duplicate-free
        source keeps them apart - then it waits, as in the batch run); "different" stops any later
        merge, and if the two are already in one customer the split waits for the next batch run,
        which cuts the weakest link between them. dry_run: say what it would do, change nothing.
        propose: four-eyes - stored as a proposal, applied when another steward approves it.
        keep: the customer that survives a merge."""
        decision = decision.strip().lower()
        if decision not in ("same", "different"):
            raise ValueError("decision must be same or different")
        a, b = sorted((a, b))
        if a == b:
            raise ValueError("a decision needs two different records")
        if dry_run:
            with self.lock:
                effect, ca, cb, _ = self._effect(a, b, decision, keep)
            return {"a": a, "b": b, "decision": decision, "preview": True,
                    "applied": ("would be proposed for approval; when approved: " + effect) if propose else effect,
                    "model_score": self._model_score(a, b)}
        self._lease()
        score = self._model_score(a, b)
        merge, ca, cb = False, None, None
        with self.lock:
            with self.eng.begin() as c:
                if propose:
                    applied = self._propose(c, a, b, decision, created_by, note)
                else:
                    applied, ca, cb, merge = self._apply(c, a, b, decision, created_by, note, keep)
                    self._log(c, created_by, "decided", a, b,
                              detail={"decision": decision, "model_score": score, "effect": applied}, note=note)
            if merge:
                self._merge(ca, {cb})
        ia, ib = self._index_of(a), self._index_of(b)
        return {"a": a, "b": b, "decision": decision, "applied": applied,
                "customers": {a: None if ia is None else str(self._customer(ia)),
                              b: None if ib is None else str(self._customer(ib))}}

    def _row(self, override_id, status=None):
        with self.eng.connect() as c:
            row = c.execute(select(overrides).where(overrides.c.id == override_id)).mappings().first()
        if row is None or (status and row["status"] != status):
            raise KeyError(f"no {status or ''} decision with that id".replace("  ", " "))
        return row

    def approve(self, override_id, by, note=None, dry_run=False):
        """Four-eyes: carry out a proposal - a decision, a revocation or a split - by a named
        steward other than its proposer."""
        row = self._row(override_id, "proposed")
        if by is None or by == row["created_by"]:
            raise PermissionError("a proposal is approved by a named steward other than its proposer")
        a, b, decision = row["a"], row["b"], row["decision"]
        if dry_run:
            what = {"revoke": f"revoke the decision on {a} and {b}", "split": f"split {a} out of its customer"}
            return {"id": override_id, "preview": True,
                    "applied": "would " + what.get(decision, f"apply {decision} for {a} and {b}")}
        if decision == "split":
            out = self._split(a, by, row["note"], proposed_by=row["created_by"])
            with self.eng.begin() as c:
                c.execute(overrides.delete().where(overrides.c.id == override_id))
            return {"id": override_id, "applied": out["applied"]}
        self._lease()
        merge, ca, cb = False, None, None
        with self.lock:
            with self.eng.begin() as c:
                c.execute(overrides.delete().where(overrides.c.id == override_id))
                if decision == "revoke":
                    c.execute(overrides.delete().where(overrides.c.a == a).where(overrides.c.b == b)
                              .where(overrides.c.status == "applied"))
                    self._forget(a, b)
                    applied = "revoked"
                else:
                    applied, ca, cb, merge = self._apply(c, a, b, decision, row["created_by"], row["note"],
                                                         why="a steward's (approved)")
                self._log(c, by, "approved", a, b or None,
                          detail={"decision": decision, "proposed_by": row["created_by"], "effect": applied}, note=note)
            if merge:
                self._merge(ca, {cb})
        return {"id": override_id, "applied": applied}

    def revoke(self, override_id, by, note=None, rejected=False, propose=False):
        """Withdraw a decision, or reject a proposal. A withdrawn decision stops applying at once:
        no longer kept apart or forced together, and not used by the next batch run (a merge it
        made stays until then). With four-eyes (propose), withdrawing an applied decision is
        itself a proposal; rejecting a proposal is not."""
        row = self._row(override_id)
        a, b = row["a"], row["b"]
        if rejected and row["status"] != "proposed":
            raise ValueError("only a proposal can be rejected; revoke a decision instead")
        self._lease()
        with self.lock:
            with self.eng.begin() as c:
                if propose and row["status"] == "applied":
                    return {"id": override_id, "applied": self._propose(c, a, b, "revoke", by, note)}
                c.execute(overrides.delete().where(overrides.c.id == override_id))
                self._log(c, by, "rejected" if rejected else "revoked", a, b or None,
                          detail={"decision": row["decision"], "status": row["status"], "by": row["created_by"]},
                          note=note)
            if row["status"] == "applied":
                self._forget(a, b)
        return {"id": override_id, "applied": ("rejected" if rejected else "revoked")
                + (" - a merge it made stays until the next batch run" if row["decision"] == "same"
                   and row["status"] == "applied" else "")}

    def split_out(self, key, by=None, note=None, dry_run=False, propose=False):
        """Take one record out of its customer, now: it becomes a customer of its own, and a
        "different" against each record it leaves keeps it apart in the next batch run. With
        four-eyes (propose) it waits for a second steward's approval."""
        i = self._index_of(key)
        if i is None:
            raise KeyError(f"not found: {key}")
        cid = self._customer(i)
        others = [self._key(j) for j in self._members(cid) if j != i]
        if not others:
            raise ValueError("the record is already alone in its customer")
        if dry_run:
            return {"record": key, "preview": True, "from_customer": str(cid),
                    "applied": ("would be proposed for approval; when approved: " if propose else "")
                    + f"{key} becomes a new customer; customer {cid} keeps {len(others)} record(s); "
                      f"{len(others)} \"different\" decision(s) keep them apart in later runs"}
        if propose:
            self._lease()
            with self.lock, self.eng.begin() as c:
                return {"record": key, "applied": self._propose(c, key, "", "split", by, note)}
        return self._split(key, by, note)

    def _split(self, key, by, note, proposed_by=None):
        self._lease()
        with self.lock:
            i = self._index_of(key)
            cid = self._customer(i)
            others = [self._key(j) for j in self._members(cid) if j != i]
            if not others:
                raise ValueError("the record is already alone in its customer")
            new = self.next_customer
            src, _, rid = key.partition(":")
            with self.eng.begin() as c:
                for o in others:
                    x, y = sorted((key, o))
                    c.execute(overrides.delete().where(overrides.c.a == x).where(overrides.c.b == y)
                              .where(overrides.c.status == "applied"))
                    c.execute(insert(overrides).values(a=x, b=y, decision="different", created_at=db.now(),
                                                       created_by=proposed_by or by,
                                                       note=note or f"split {key} out of customer {cid}"))
                c.execute(insert(realtime_moves).values(source=src, record_id=rid, from_customer=int(cid),
                                                        to_customer=int(new), reason=f"split out by {by or 'a steward'}",
                                                        matched_against_run=self.run_id, created_at=db.now()))
                self._log(c, by, "split", key, None, customer_id=cid,
                          detail={"from_customer": int(cid), "to_customer": int(new), "kept_apart_from": len(others)}
                          | ({"proposed_by": proposed_by} if proposed_by else {}), note=note)
            for o in others:
                self._remember(key, o, "different")
            self._set_customer(i, new)
        return {"record": key, "from_customer": str(cid), "to_customer": str(new),
                "applied": f"{key} is now customer {new}; customer {cid} keeps {len(others)} record(s)"}

    def merge_customers(self, into, other, by=None, note=None, dry_run=False, propose=False):
        """Merge customer `other` into `into`: a "same" between a record of each (so later batch
        runs keep them together), applied now unless a "different" or a duplicate-free source
        forbids it."""
        a_m, b_m = self._members(int(into)), self._members(int(other))
        if not a_m or not b_m:
            raise KeyError("customer not found")
        if int(into) == int(other):
            raise ValueError("that is the same customer")
        return self.decide_override(self._key(a_m[0]), self._key(b_m[0]), "same", by, note, dry_run=dry_run,
                                    propose=propose, keep=int(into))

    def compare(self, a, b):
        """Any two records: their score and its breakdown, and whether blocking would ever have
        compared them - "why not matched" when they were not."""
        from custmatch.core import blocking_families, _family_label
        ia, ib = self._index_of(a), self._index_of(b)
        if ia is None or ib is None:
            raise KeyError(f"not found: {a if ia is None else b}")
        out = self.explain(a, b)
        mini = self._rows([ia, ib])
        shared = []
        from custmatch.country import key_tuple
        for k, fam in enumerate(blocking_families(mini, self.profiles, True, self.extra, self.lsh)):
            label = _family_label(fam, k)
            if fam[0] == "country":                 # as key tuples: two records' keys
                fam = ("tuples", [key_tuple(*kk) for kk in zip(*fam[1])], fam[2].tolist())
            if fam[0] == "tuples":
                keys, idx = fam[1], fam[2]
                ka = {kk for kk, ii in zip(keys, idx) if ii == 0}
                kb = {kk for kk, ii in zip(keys, idx) if ii == 1}
                common = list(ka & kb)
            else:
                _, tag, arrays, idx = fam
                vals = {int(r): tuple(str(arr[n]) for arr in arrays) for n, r in enumerate(idx)}
                both = 0 in vals and 1 in vals and vals[0] == vals[1]
                common = [vals[0]] if both else []
            for kk in common:
                size = len(self.index.get(self._family_hash(fam, kk)))
                shared.append({"key": label, "records_sharing_it": size, "usable": 2 <= size <= self.max_block})
        out["blocking"] = {"shared_keys": shared, "compared": any(x["usable"] for x in shared),
                           "max_block": self.max_block}
        return out

    def _family_hash(self, fam, key):
        """The index hash of one blocking key (as core.blocking_hashes computes it)."""
        from custmatch.core import _MIX
        if fam[0] == "tuples":
            return _hash_strings(["\x1f".join(map(str, key))])[0]
        tag = fam[1]
        h = np.uint64(_hash_strings(["\x1f".join(map(str, tag))])[0])
        with np.errstate(over="ignore"):
            for v in key:
                h = (h * _MIX) ^ np.uint64(_hash_strings([v])[0])
        return h

    def _source(self, name):
        """A source's config; KeyError for one the config does not have."""
        src = next((s for s in self.cfg["sources"] if s["name"] == name), None)
        if src is None:
            raise KeyError(f"unknown source {name!r}")
        return src

    def _id_column(self, source):
        return self._source(source)["columns"]["record_id"]
