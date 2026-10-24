"""Blocking for the tool: rules, learned rule sets, split oversized blocks, SQL self-join, diagnostics.

What the three leading projects do and custmatch's first blocker did not (docs/guide.md):

  learned rules      dedupe: pick the cheapest set of blocking rules that still covers a target
                     share (default 95%) of the labelled matches - a greedy weighted set cover
                     over a library of rules - instead of fixed hand-picked keys
  split, not drop    Zingg: a block larger than max_block is refined with the next predicate
                     (first letters of the first name, birth year, street, ...) until it fits,
                     and dropped only if it still cannot be split - custmatch dropped it outright,
                     which lost the 173-record London block in Splink's fake_1000
  SQL self-join      dedupe / Splink: pairs come from a DuckDB self-join on (rule, key) with
                     DISTINCT, not Python loops over every block
  diagnostics        Splink: pairs contributed by each rule, its largest blocks, and what was split
                     or dropped - blocking_report.json in the run's output

core.block (the fixed keys the report measured) is unchanged; this module is used by the tool.
A rule is a tuple of predicate names; records agree on a rule when every predicate gives the
same non-empty key.
"""
import numpy as np
import pandas as pd

# predicate name -> (column, transform). Transforms map a cleaned string column to keys.
_TRANSFORMS = {
    "exact": lambda s: s,
    "first3": lambda s: s.str[:3].where(s.str.len() >= 3, ""),
    "first1": lambda s: s.str[:1],
    "year": lambda s: s.str.extract(r"(\d{4})", expand=False).fillna(""),
    "yearmonth": None,   # set below: needs date parsing
    "soundex": None,
}
NAME_FIELDS = ("first_name", "last_name")
FIXED_RULES = [("last_name:exact", "first_name:first3"), ("last_name:exact", "addr_street:exact"),
               ("addr_number:exact", "addr_street:exact"), ("first_name:exact", "addr_street:exact"),
               ("email:exact",), ("phone:exact",)]
# refinements tried, in order, to split an oversized block
REFINEMENTS = ["first_name:first3", "date_of_birth:year", "last_name:first3", "addr_street:exact",
               "first_name:exact", "date_of_birth:exact", "addr_number:exact"]


_SOUNDEX = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
            "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}


def soundex(word):
    """American Soundex: the first letter, then up to three digits for the consonant sounds that
    follow (Robert, Rupert -> R163). Letters with one code next to each other, or separated
    only by h or w, count once; a vowel, space or hyphen between them counts them twice - as
    jellyfish.soundex, which this replaces, does."""
    w = word.lower()
    first = next((n for n, c in enumerate(w) if "a" <= c <= "z"), None)
    if first is None:
        return ""
    out, last = [], _SOUNDEX.get(w[first], "")
    for c in w[first + 1:]:
        code = _SOUNDEX.get(c, "")              # vowels, spaces and hyphens separate: code ""
        if code and code != last:
            out.append(code)
        if c not in "hw":                       # h and w do not separate two equal codes
            last = code
    return (w[first].upper() + "".join(out) + "000")[:4]


def _soundex(s):
    return s.map(lambda v: soundex(v) if v and v[0].isalpha() else "")


def _yearmonth(s):
    from .core import parse_dates
    d = parse_dates(s.to_numpy())
    return pd.Series([f"{y:04d}{m:02d}" if y >= 0 else "" for y, m, _ in d], index=s.index)


class Keys:
    """Lazily computed predicate keys per record (cleaned strings, "" = no key)."""

    def __init__(self, df, profiles=None):
        self.df, self.cache = df, {}
        self.junk = {}
        for f in ("email", "phone"):
            if profiles and f in profiles:
                self.junk[f] = df[f].map(lambda v: profiles[f].get(v, ("personal",))[0] == "junk").to_numpy()

    def get(self, pred):
        if pred not in self.cache:
            col, how = pred.split(":")
            if col not in self.df.columns:
                self.cache[pred] = np.full(len(self.df), "", dtype=object)
            else:
                s = self.df[col].astype(str)
                if how == "soundex":
                    k = _soundex(s)
                elif how == "yearmonth":
                    k = _yearmonth(s)
                else:
                    k = _TRANSFORMS[how](s)
                k = k.to_numpy(dtype=object)
                if col in self.junk:
                    k = np.where(self.junk[col], "", k)
                self.cache[pred] = k
        return self.cache[pred]

    def codes(self, pred):
        """Integer code per record for a predicate; -1 where it has no key."""
        ck = ("codes", pred)
        if ck not in self.cache:
            k = self.get(pred)
            c, _ = pd.factorize(k)
            self.cache[ck] = np.where(k == "", -1, c).astype(np.int64)
        return self.cache[ck]

    def rule(self, rule):
        """Integer key per record for a rule (a conjunction of predicates); -1 where any
        predicate has no key. Integer codes, not joined strings: at 1M records joining strings
        made blocking the slowest stage."""
        parts = [self.codes(p) for p in rule]
        empty = np.zeros(len(self.df), bool)
        for c in parts:
            empty |= c < 0
        if len(parts) == 1:
            key = parts[0]
        else:
            key, _ = pd.MultiIndex.from_arrays(parts).factorize()
            key = key.astype(np.int64)
        return np.where(empty, -1, key)

    def label(self, rule, i):
        """Readable key of record i under a rule, for diagnostics."""
        return " | ".join(str(self.get(p)[i]) for p in rule)


def rule_library(df, cfg=None):
    """Candidate rules over the fields this data actually has."""
    from .schema import extra_fields
    # country.py's derived name columns are blocked on through country.blocking_keys, not as rules
    derived = {"name_script", "first_roots", "name_skeleton", "country", "updated_at"}
    have = [c for c in df.columns if c not in ("source", "record_id") and c not in derived
            and (df[c] != "").sum() > 1]
    single = []
    for c in have:
        single.append(f"{c}:exact")
        if c in NAME_FIELDS or c in ("addr_street",) or c in {n for n, _ in extra_fields(cfg or {})}:
            single.append(f"{c}:first3")
        if c in NAME_FIELDS:
            single.append(f"{c}:soundex")
        if c == "date_of_birth":
            single += ["date_of_birth:year", "date_of_birth:yearmonth"]
    rules = [(p,) for p in single]
    strong = [p for p in single if p.endswith(":exact") or p.endswith(":soundex")]
    for a in range(len(single)):
        for b in range(a + 1, len(single)):
            pa, pb = single[a], single[b]
            if pa.split(":")[0] == pb.split(":")[0]:
                continue
            if pa in strong or pb in strong:
                rules.append((pa, pb))
    rules += fixed_rules(cfg, df)
    return list(dict.fromkeys(rules))


def fixed_rules(cfg=None, df=None):
    """The fixed keys as rules. A predicate on a field the data never fills is dropped from its
    rule, so (last_name, addr_street) becomes (last_name) where there is no address - what the
    report's core.block did implicitly (an empty street was a shared key), and the recall it gave
    on historical_50k (87.0 vs 84.0 without) depends on it."""
    from .schema import extra_keys
    rules = [tuple(r) for r in FIXED_RULES] + [tuple(c if ":" in c else f"{c}:exact" for c in k) for k in extra_keys(cfg or {})]
    if df is None:
        return rules
    filled = {c for c in df.columns if (df[c] != "").any()}
    out = []
    for r in rules:
        kept = tuple(p for p in r if p.split(":")[0] in filled)
        if kept:
            out.append(kept)
    return list(dict.fromkeys(out))


def _n2(sizes):
    sizes = np.asarray(sizes, dtype=np.float64)
    return float(np.sum(sizes * (sizes - 1) / 2))


def learn_rules(df, keys, pos_pairs, cfg=None, target=0.95, max_block=60, max_rules=8,
                sample=50000, seed=0):
    """Greedy weighted set cover: repeatedly add the rule covering the most still-uncovered
    labelled matches per candidate pair it would generate, until `target` of the labelled
    matches that any rule can cover are covered. Returns (rules, report).

    Like dedupe, it learns on a sample: the records in labelled pairs plus up to `sample`
    random others, with block sizes scaled up to the full file for the cost - evaluating the
    whole rule library on a million records took 143 s."""
    lib = rule_library(df, cfg)
    n = len(df)
    rng = np.random.default_rng(seed)
    in_labels = np.unique(pos_pairs.ravel())
    others = np.setdiff1d(np.arange(n), in_labels)
    extra = rng.choice(others, min(sample, len(others)), replace=False) if len(others) else others
    idx = np.sort(np.concatenate([in_labels, extra]))
    sub = Keys(df.iloc[idx].reset_index(drop=True), None)
    sub.junk = {f: v[idx] for f, v in keys.junk.items()}
    pos_local = np.searchsorted(idx, pos_pairs)
    i, j = pos_local[:, 0], pos_local[:, 1]
    scale = n / len(idx)
    covers, cost = {}, {}
    for r in lib:
        k = sub.rule(r)
        covers[r] = (k[i] == k[j]) & (k[i] >= 0)
        sizes = np.bincount(k[k >= 0]) * scale if (k >= 0).any() else np.zeros(0)
        sizes = sizes[sizes > 0]
        # oversized blocks will be split; charge them roughly as pieces of max_block
        small, big = sizes[sizes <= max_block], sizes[sizes > max_block]
        cost[r] = _n2(small) + float(np.sum(np.ceil(big / max_block))) * _n2([max_block])
    coverable = np.any([covers[r] for r in lib], axis=0) if lib else np.zeros(len(i), bool)
    need = target * coverable.sum()
    chosen, covered = [], np.zeros(len(i), bool)
    while covered.sum() < need and len(chosen) < max_rules:
        best = max(lib, key=lambda r: ((covers[r] & ~covered).sum() / (cost[r] + 1.0), -cost[r]))
        if not (covers[best] & ~covered).any():
            break
        chosen.append(best)
        covered |= covers[best]
    report = {"labelled_matches": int(len(i)), "coverable_by_library": int(coverable.sum()),
              "covered": int(covered.sum()), "target": target, "library_size": len(lib),
              "rules": [{"rule": list(r), "covers": int(covers[r].sum()), "est_pairs": int(cost[r])}
                        for r in chosen]}
    return chosen, report


ADAPTIVE_WAYS = 3      # "blocking": "adaptive": refinements an oversized block is split on, together
# with no "blocking" set, the default keys give way to "adaptive" when more than this share of the
# records are only in blocks too big to compare: 0.45% at most on the five small benchmarks and
# 0.13% on BPID, 1.1% on NCVR at 200,000 people, 4.2% at a million and 8% at all five million,
# where adaptive scored 97.63 against 95.63, 94.04 against 90.29 and 88.45 against 84.33; dedupe's
# organisation files switch too, and gain (csv_example 86.95 to 89.08, patents 75.88 to 77.37)
AUTO_ADAPTIVE_SHARE = 0.01


def refinements(cfg=None):
    """REFINEMENTS, then each extra field's exact value (town, postcode): a file without dates of
    birth or streets has little else to split a large block on."""
    from .schema import extra_fields
    return REFINEMENTS + [f"{n}:exact" for n, _ in extra_fields(cfg or {})]


def _split_ways(keys, idx_block, rule, max_block, refine, ways):
    """An oversized block split on each of up to `ways` refinements, on different fields from
    the rule's and each other's, and the pieces of all of them kept - so a pair is lost only if
    it disagrees on every one. Splitting on one (_split) loses a pair whenever that field has
    the typo: on the full NCVR file, a (postcode, first initial) block split on the first
    three letters of the first name lost every pair whose first name was misspelt there."""
    used = {p.split(":")[0] for p in rule}
    out, dropped, chosen = [], 0, 0
    for p in refine:
        if chosen == ways:
            break
        col = p.split(":")[0]
        if col in used:
            continue
        sub = keys.codes(p)[idx_block]
        if (sub == sub[0]).all():
            continue
        used.add(col)
        chosen += 1
        o, d = _split(keys, idx_block, rule, max_block, [p] + [q for q in refine if q != p])
        out += o
        dropped += d
    return (out, dropped) if chosen else ([], 1)


def _split(keys, idx_block, rule, max_block, refine=REFINEMENTS):
    """Refine an oversized block with the next predicates; returns (list of record-index arrays,
    number of pieces dropped because nothing was left to split them on)."""
    if len(idx_block) <= max_block:
        return [idx_block], 0
    for p in refine:
        if p in rule:
            continue
        sub = keys.codes(p)[idx_block]
        if (sub == sub[0]).all():
            continue
        out, dropped = [], 0
        # records with no value for this refinement (a first name shorter than three letters,
        # no birth year) stay together as their own piece rather than being lost. Pieces are
        # slices of one stable sort, in order of value, as a loop over np.unique(sub) gave them
        order = np.argsort(sub, kind="stable")
        bounds = np.flatnonzero(np.diff(sub[order])) + 1
        for part in np.split(idx_block[order], bounds):
            if len(part) < 2:
                continue
            o, d = _split(keys, part, rule + (p,), max_block, refine)
            out += o
            dropped += d
        return out, dropped
    return [], 1        # nothing left to split on: drop it as a hub


def generate(df, keys, rules, max_block=60, ways=1, refine=REFINEMENTS):
    """Candidate pairs for the rules (splitting oversized blocks), plus per-rule diagnostics.
    ways > 1 splits each oversized block on that many refinements at once (_split_ways).
    Returns (pairs int32 array sorted, report)."""
    rows, report = [], []
    for n, r in enumerate(rules):
        k = keys.rule(r)
        has = np.flatnonzero(k >= 0)
        kk = k[has]
        sizes = np.bincount(kk) if len(kk) else np.zeros(0, int)
        big_codes = np.flatnonzero(sizes > max_block)
        small = sizes[kk] <= max_block
        rk, ri = kk[small], has[small]
        next_key = int(sizes.size)
        split_blocks, dropped, extra_k, extra_i = 0, 0, [], []
        # each oversized block's records as one slice of a sort, not a scan of every record per
        # block: at five million records that scan was most of blocking's time
        order = np.argsort(kk, kind="stable")
        lo, hi = np.searchsorted(kk[order], big_codes), np.searchsorted(kk[order], big_codes, side="right")
        for a, b in zip(lo.tolist(), hi.tolist()):
            idx = has[order[a:b]]
            parts, d = _split_ways(keys, idx, tuple(r), max_block, refine, ways) if ways > 1 \
                else _split(keys, idx, tuple(r), max_block)
            split_blocks += 1
            dropped += d
            for pidx in parts:
                extra_k.append(np.full(len(pidx), next_key, np.int64))
                extra_i.append(pidx)
                next_key += 1
        rows.append(pd.DataFrame({"rule": n, "key": np.concatenate([rk] + extra_k) if extra_k else rk,
                                  "rid": np.concatenate([ri] + extra_i) if extra_i else ri}))
        order = np.argsort(-sizes)[:5]
        report.append({"rule": list(r), "records_with_key": int(len(has)), "blocks": int((sizes > 0).sum()),
                       "largest_blocks": [{"key": keys.label(r, int(has[kk == c][0])), "size": int(sizes[c])}
                                          for c in order if sizes[c] > 0],
                       "oversized_blocks_split": split_blocks, "blocks_dropped_as_hubs": dropped})
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame({"rule": [], "key": [], "rid": []})
    per_rule = _block_pairs(table.rule.to_numpy(np.int64), table.key.to_numpy(np.int64), table.rid.to_numpy(np.int64))
    counts = np.bincount(per_rule.rule.to_numpy(), minlength=len(rules))
    # each pair once, credited to the first rule that produced it: one sort of int64 codes
    # (pair * rules + rule), not a pandas groupby and sort - those were half of blocking's time
    # at a million records
    nr, nrec = max(len(rules), 1), len(df)
    code = np.unique((per_rule.i.to_numpy() * nrec + per_rule.j.to_numpy()) * nr + per_rule.rule.to_numpy())
    del per_rule
    pair = code // nr
    first = np.r_[True, pair[1:] != pair[:-1]] if len(pair) else np.zeros(0, bool)
    pair, new = pair[first], np.bincount(code[first] % nr, minlength=len(rules))
    for n, entry in enumerate(report):
        entry["pairs"] = int(counts[n])
        entry["new_pairs"] = int(new[n])      # not already produced by an earlier rule
    pairs = np.column_stack([pair // nrec, pair % nrec]).astype(np.int32)
    return pairs, {"rules": report, "candidate_pairs": int(len(pairs)), "max_block": max_block}


def _block_pairs(rule, key, rid):
    """Every pair of records sharing a (rule, key) block: a DataFrame of rule, i < j. Sorted by
    block, a record pairs with the k-th record after it while that one is still in its block;
    blocks are capped at max_block, so k stays small and each step is one vectorised pass."""
    if not len(rid):
        return pd.DataFrame({"rule": np.zeros(0, np.int64), "i": np.zeros(0, np.int64), "j": np.zeros(0, np.int64)})
    order = np.lexsort((rid, key, rule))
    rule, key, rid = rule[order], key[order], rid[order]
    new = np.r_[True, (rule[1:] != rule[:-1]) | (key[1:] != key[:-1])]
    starts = np.flatnonzero(new)
    end = np.repeat(np.r_[starts[1:], len(rid)], np.diff(np.r_[starts, len(rid)]))   # block end per row
    pos = np.arange(len(rid))
    out_r, out_i, out_j = [], [], []
    for k in range(1, int((end - pos).max())):
        a = pos[pos + k < end]
        b = a + k
        out_r.append(rule[a]); out_i.append(np.minimum(rid[a], rid[b])); out_j.append(np.maximum(rid[a], rid[b]))
    if not out_r:
        return pd.DataFrame({"rule": np.zeros(0, np.int64), "i": np.zeros(0, np.int64), "j": np.zeros(0, np.int64)})
    return pd.DataFrame({"rule": np.concatenate(out_r), "i": np.concatenate(out_i), "j": np.concatenate(out_j)})


def country_pairs(df, profiles, max_block):
    """Pairs from country.blocking_keys (first-name root + DOB or surname, email user name, name
    skeleton), blocks larger than max_block skipped - the same keys core.block uses."""
    from collections import defaultdict
    from . import country
    groups = defaultdict(list)
    for k, i in country.blocking_keys(df, profiles):
        groups[k].append(i)
    out = set()
    for idx in groups.values():
        if 2 <= len(idx) <= max_block:
            idx = sorted(set(idx))
            out.update((a, b) for x, a in enumerate(idx) for b in idx[x + 1:])
    return np.array(sorted(out), dtype=np.int32).reshape(-1, 2)


def candidates_for(df, profiles, cfg, labelled_pos=None, rules=None):
    """The tool's candidate pairs. Returns (pairs, hubs_dropped, report, rules_used).

    With alias fields ("aliases", core.alias_views) the views are blocked - a record's previous
    surname or second email is a key like its own - and pairs of views become pairs of records."""
    from .core import alias_views
    views, owner = alias_views(df)
    if owner is None:
        return _candidates_for(df, profiles, cfg, labelled_pos, rules)
    if cfg.get("blocking") not in (None, "", "core") or rules is not None:
        from .io import ConfigError
        raise ConfigError('"aliases" works with the default blocking only; remove "blocking": '
                          f'"{cfg.get("blocking") or "saved rules"}" or "aliases"')
    pairs, hubs, report, used = _candidates_for(views, profiles, cfg, labelled_pos, rules, owner=owner)
    report["alias_rows"] = int(len(views) - len(df))
    return pairs, hubs, report, used


def _core_candidates(df, profiles, cfg, owner=None):
    """core.block's pairs and report: the default keys, blocks over max_block skipped."""
    from .core import block
    from .schema import extra_keys, max_block as cfg_max_block
    skipped, by_key = [], []
    pairs, hubs = block(df, profiles, extra_keys=extra_keys(cfg), max_block=cfg_max_block(cfg),
                        lsh=cfg.get("lsh_blocking"), skipped=skipped, by_key=by_key, owner=owner)
    # records never compared with anyone because their only blocks were skipped - a record
    # whose broad block (postcode alone) was skipped but whose narrow one was used is fine
    compared = np.zeros(len(df), bool)
    compared[pairs.ravel()] = True
    records = len({i for _, idx in skipped for i in idx if not compared[i]})
    top = sorted(skipped, key=lambda s: -len(s[1]))[:5]
    return pairs, hubs, {"mode": "core", "candidate_pairs": int(len(pairs)), "hubs_dropped": int(hubs),
                         "skipped_blocks_with_values": len(skipped), "records_only_in_skipped_blocks": records,
                         "largest_skipped_blocks": [{"key": " / ".join(str(v) for v in k), "records": len(idx)}
                                                    for k, idx in top],
                         "pairs_by_key": by_key, "_skipped": skipped}


def _candidates_for(df, profiles, cfg, labelled_pos=None, rules=None, owner=None):
    """candidates_for over records (or alias views).

    cfg "blocking": absent -> the fixed keys the report measured (core.block, blocks over
    max_block dropped), switching to "adaptive" when that would leave more than
    AUTO_ADAPTIVE_SHARE of the records compared with no one; "core" -> core.block, never
    switching; "fixed" -> the same kind of keys as rules, oversized blocks split;
    "adaptive" -> the fixed rules, each oversized block split on several fields at once;
    "learned" -> rules learned from labelled_pos (falls back to fixed without labels).
    rules: reuse a saved rule set (incremental matching)."""
    from .schema import max_block as cfg_max_block
    mode = cfg.get("blocking")
    mb = cfg_max_block(cfg) or 60
    if rules is None and mode in (None, "", "core"):
        pairs, hubs, report = _core_candidates(df, profiles, cfg, owner)
        records = report["records_only_in_skipped_blocks"]
        if not mode and owner is None and not cfg.get("lsh_blocking") and records > AUTO_ADAPTIVE_SHARE * len(df):
            pairs, hubs, switched, used = _candidates_for(df, profiles, cfg | {"blocking": "adaptive"}, labelled_pos)
            switched["switched_from_core"] = {"records_only_in_skipped_blocks": records, "records": len(df),
                                              "share": round(records / len(df), 4)}
            switched["_skipped"] = report["_skipped"]     # for the large-block check, as with the default keys
            return pairs, hubs, switched, used
        return pairs, hubs, report, None
    if cfg.get("lsh_blocking"):
        from .io import ConfigError
        raise ConfigError('"lsh_blocking" works with the default blocking only; remove "blocking": '
                          f'"{mode}" or "lsh_blocking"')
    keys = Keys(df, profiles)
    learn = None
    if rules is None:
        if mode == "learned" and labelled_pos is not None and len(labelled_pos):
            rules, learn = learn_rules(df, keys, labelled_pos, cfg, target=cfg.get("blocking_recall", 0.95),
                                       max_block=mb)
        else:
            rules = fixed_rules(cfg, df)
    if mode == "adaptive":
        pairs, report = generate(df, keys, rules, max_block=mb, ways=ADAPTIVE_WAYS, refine=refinements(cfg))
    else:
        pairs, report = generate(df, keys, rules, max_block=mb)
    extra = country_pairs(df, profiles, mb)        # country.py keys: nickname + DOB, email user...
    if len(extra):
        allp = np.vstack([pairs, extra]) if len(pairs) else extra
        pairs = np.unique(allp, axis=0)
    report.update(mode=mode or "saved", learned=learn, country_key_pairs=int(len(extra)))
    if cfg.get("_blocking_switched"):
        # a run that switched (a saved state's config): the default keys' skipped blocks, so the
        # large-block check sees what it saw in the run
        report["_skipped"] = _core_candidates(df, profiles, cfg, owner)[2]["_skipped"]
    hubs = sum(r["blocks_dropped_as_hubs"] for r in report["rules"])
    return pairs, hubs, report, [list(r) for r in rules]


# ------------------------------------------------------------------ similarity (LSH) blocking
def lsh_keys(df, spec, seed=0):
    """Blocking keys from MinHash locality-sensitive hashing of letter pairs - the idea behind
    anonlink's blocklib, on plain text. Records whose `fields` (joined) share most of their
    letter pairs get the same signature in at least one band with high probability, so names
    with typos in both still meet; exact keys miss them. `with` adds a field to every key, to
    keep the blocks small (a band of common names state-wide is a hub otherwise).

        "lsh_blocking": {"fields": ["first_name", "last_name"], "with": "town",
                         "bands": 16, "rows": 3}

    With b bands of r rows, two records whose letter-pair sets have Jaccard similarity s share
    a band with probability 1 - (1 - s^r)^b: for 16 x 3, 0.97 at s = 0.6 and 0.32 at s = 0.2.
    Returns a list of (record index, key) pairs."""
    import zlib
    fields, bands, rows = spec["fields"], int(spec.get("bands", 16)), int(spec.get("rows", 3))
    text = df[fields[0]].astype(str)
    for f in fields[1:]:
        text = text + " " + df[f].astype(str)
    text = text.str.strip()
    rng = np.random.default_rng(seed)
    k = bands * rows
    p = np.uint64((1 << 61) - 1)
    a = rng.integers(1, int(p), k, dtype=np.uint64)       # k hash functions (a*x + b) mod p
    b = rng.integers(0, int(p), k, dtype=np.uint64)
    other = df[spec["with"]].astype(str).to_numpy() if spec.get("with") else None
    out = []
    cache = {}
    for i, t in enumerate(text):
        if len(t) < 3 or (other is not None and not other[i]):
            continue
        if t not in cache:
            g = {f"{t[x:x + 2]}" for x in range(len(t) - 1)}
            ids = np.array([zlib.crc32(s.encode()) for s in g], dtype=np.uint64)
            # min over the record's letter pairs of k random hash functions
            sig = ((ids[:, None] * a[None, :] + b[None, :]) % p).min(axis=0)
            cache[t] = [hash(tuple(sig[x * rows:(x + 1) * rows].tolist())) for x in range(bands)]
        extra = other[i] if other is not None else ""
        out += [(i, ("lsh", band, extra, h)) for band, h in enumerate(cache[t])]
    return out


LARGE_BLOCK_SAMPLE = 50         # pairs scored per oversized block
LARGE_BLOCK_MATCH_SHARE = 0.5   # kept when at least this share of them are matches
LARGE_BLOCK_MAX = 5000          # records: larger blocks are not checked (12.5 million pairs each)


def large_block_check_on(cfg):
    """On unless "check_large_blocks": false - it found no large entity, and changed nothing, on
    six person benchmarks, and lifted dedupe's patent-applicant file from 6.93 to 75.88."""
    return cfg.get("check_large_blocks", True) and (cfg.get("blocking") in (None, "", "core")
                                                    or bool(cfg.get("_blocking_switched")))


def saved_run_large_blocks(cfg, report, predict, th, pairs):
    """For `add` and `explain`: the same check over a saved run's model, so a large entity kept by
    the run is kept again. Returns `pairs` with the kept blocks' pairs added (sorted, unique)."""
    skipped = report.pop("_skipped", [])
    if not (large_block_check_on(cfg) and skipped):
        return pairs
    extra, _ = check_large_blocks(skipped, predict, th, budget=max(1_000_000, len(pairs)))
    return np.unique(np.vstack([pairs, extra]), axis=0).astype(np.int32) if len(extra) else pairs


def check_large_blocks(skipped, predict, th, budget, seed=0, share=LARGE_BLOCK_MATCH_SHARE):
    """Blocks skipped as hubs that are one real entity - an organisation with hundreds of records -
    rather than junk (a@a.com: unrelated people agreeing on nothing else) or many people (an
    apartment building). The trained model scores a sample of each block's pairs; a block whose
    sampled pairs are mostly matches is kept, and all its pairs become candidates, largest-first
    until `budget` pairs. Returns (new pairs int32 (m, 2), i < j; report rows)."""
    rng = np.random.default_rng(seed)
    blocks = [(key, np.unique(np.asarray(idx))) for key, idx in skipped if len(idx) <= LARGE_BLOCK_MAX]
    if not blocks:
        return np.zeros((0, 2), np.int32), []
    samples, owner = [], []
    for b, (_, idx) in enumerate(blocks):
        a, c = rng.choice(idx, LARGE_BLOCK_SAMPLE), rng.choice(idx, LARGE_BLOCK_SAMPLE)
        ok = a != c
        samples.append(np.c_[np.minimum(a[ok], c[ok]), np.maximum(a[ok], c[ok])])
        owner.append(np.full(int(ok.sum()), b))
    p = predict(np.vstack(samples).astype(np.int32))      # one scoring pass for every block
    owner = np.concatenate(owner)
    match_share = np.bincount(owner, weights=p > th, minlength=len(blocks)) / np.maximum(
        np.bincount(owner, minlength=len(blocks)), 1)
    rows, new, used = [], [], 0
    for b in np.argsort(-np.array([len(idx) for _, idx in blocks]), kind="stable"):
        key, idx = blocks[b]
        n_pairs = len(idx) * (len(idx) - 1) // 2
        keep = match_share[b] >= share and used + n_pairs <= budget
        rows.append({"key": " / ".join(str(v) for v in key), "records": int(len(idx)),
                     "sampled_match_share": round(float(match_share[b]), 3), "kept": bool(keep)})
        if keep:
            i, j = np.triu_indices(len(idx), 1)
            new.append(np.c_[idx[i], idx[j]])
            used += n_pairs
    if not new:
        return np.zeros((0, 2), np.int32), rows
    return np.unique(np.vstack(new), axis=0).astype(np.int32), rows
