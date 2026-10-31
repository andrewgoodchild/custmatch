"""The customer-matching pipeline's stages, as measured in docs/report/05-customer-data.md.

profile contacts -> block (hub suppression, junk never a key) -> features (contact agreement
weighted by rarity, junk treated as missing) -> [a supervised matcher] -> connected components
-> households. `research/scripts/customer_pipeline.py` (the experiment) and `custmatch.pipeline` (the
tool) both use these functions, so the tool runs exactly the code the report measured.

Records use the canonical columns in `custmatch.io.CANONICAL`; missing values are "".
"""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

MAX_BLOCK = 60          # blocks larger than this are hubs; skip them
JUNK_MIN_RECORDS = 8    # a value this common, carried by unrelated people, is junk
JUNK_DIVERSITY = 0.5    # distinct surnames / records
FAMILY_MAX = 20         # above this an address is group quarters, not a family
HOUSEHOLD_DIVERSITY = 0.5
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

# addresses a service sends from, not a person: no-reply, notification and bounce mailboxes, as a
# word of the user name (drive-shares-noreply@, calendar-notification@, mailer-daemon@), joined
# by - or _ as services write them: BPID's people have addresses like gertie.noreply@. Mail
# clients file them under the name of the person the service wrote for, so an address book
# holds them with a real name: on one, a file-sharing relay named for one person was linked to
# ten of that person's addresses, every false link of the run
RELAY_EMAIL = re.compile(r"^(?:[^@]*[-_])?(?:no-?_?reply|do-?_?not-?_?reply|notifications?|mailer-daemon|bounces?)"
                         r"(?:[-_.+][^@]*)?@")
JUNK_PATTERNS = [                                       # values no one can own: junk on sight
    RELAY_EMAIL,
    re.compile(r"^(.)\1{6,}$"),                        # 0000000, aaaaaaa
    re.compile(r"^(?:0123456789|1234567890|9876543210)$"),
    re.compile(r"^5{3}\d{7}$"),                        # 555-xxx-xxxx
    re.compile(r"^61491570\d{3}$"),                     # ACMA fictional mobiles (E.164)
    re.compile(r"^61[2378](?:5550|7010)\d{4}$"),         # ACMA fictional landlines (E.164)
    re.compile(r"^(?:test|asdf|none|no|xxx+|abc|a)@"),
    re.compile(r"@(?:test|example|none|xxx)\.(?:com|org|net)$"),
    re.compile(r"^(?:mailinator|yopmail)\."),
]
# Numbers that look made up but can be real: people buy "gold" numbers (0411 111111, 0412 121212).
# A pattern alone is not enough - such a number is junk only when PATTERN_MIN_RECORDS or more
# records carry it and they are different people (profile_contacts).
PATTERNED_NUMBERS = [
    re.compile(r"^\d*(\d)\1{6,}$"),                     # 0400000000, 0411111111
    re.compile(r"^\d*12345678\d?$"),                  # 0412345678
    re.compile(r"^\d*(?:0123456|1234567|2345678|3456789|9876543|8765432|7654321|6543210)$"),   # ends in a run of 7
    re.compile(r"^\d*?(\d\d)\1{3,}$"),                 # a pair 4+ times at the end: 0412121212
    re.compile(r"^\d*?(\d{3})\1{2,}$"),               # a triple 3+ times: 0123123123
    re.compile(r"^\d*8675309$"),                       # 867-5309, the song's number, in any area code
]
PATTERN_MIN_RECORDS = 3


def syntactic_junk(v):
    return any(p.search(v) for p in JUNK_PATTERNS)


def looks_fake(v):
    """Junk on sight, or a patterned number: what the benchmark scripts' generators treat as fake."""
    return syntactic_junk(v) or any(p.search(v) for p in PATTERNED_NUMBERS)


def syntactic_junk_mask(values, patterns=None):
    """syntactic_junk for many values (or `patterns`, e.g. PATTERNED_NUMBERS): each pattern in one
    vectorised pass (Python's regex engine, since some patterns use backreferences). Identical
    results, without a Python call per value per pattern (6 s of a million-record run)."""
    import warnings
    s = pd.Series(list(values), dtype=object)
    mask = np.zeros(len(s), dtype=bool)
    with warnings.catch_warnings():               # the patterns' groups are for backreferences, not extraction
        warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
        for p in JUNK_PATTERNS if patterns is None else patterns:
            mask |= s.str.contains(p, regex=True, na=False).to_numpy(dtype=bool)
    return mask


def patterned_junk(values, n, names, addr_diversity, other_contacts_differ):
    """A patterned number (PATTERNED_NUMBERS) carried by PATTERN_MIN_RECORDS or more records
    that are different people - mostly different names, or, with no names, mostly different
    other contacts - and not one address: the pattern and the strangers confirm each other. One
    person's bought gold number shares their name."""
    pattern = syntactic_junk_mask(values, PATTERNED_NUMBERS)
    strangers = (np.asarray(names) > JUNK_DIVERSITY) | np.asarray(other_contacts_differ)
    return (pattern & (np.asarray(n) >= PATTERN_MIN_RECORDS) & strangers
            & ~(np.asarray(addr_diversity) <= JUNK_DIVERSITY))


# ---------------------------------------------------------------- stage 1: profile contacts
def profile_contacts(df, field, aware=True):
    """Classify each distinct value. Returns {value: (class, weight)}.

    weight is the evidential weight of agreeing on this value: rare -> 1.0, common -> ~0.
    Vectorised: one group-by over all values (a per-value loop took 7 s of an 8 s run).
    Alias values ("aliases": a second email) are profiled with the rest, as views of their record.
    """
    if any(c.startswith(field + "@") for c in alias_columns(df)):
        views, owner = alias_views(df)            # each of a record's values once, not once per view
        df = views.assign(_owner=owner).drop_duplicates(["_owner", field])
    sub = df[df[field] != ""]
    if not len(sub):
        return {}
    # blank surnames and addresses are unknown, not shared: diversity is measured over the records
    # that have one. Counting blanks as one value made na@na.com, typed by 223 unrelated web
    # customers with no address, look like one household's email (sparse Australian file).
    addr = (sub.addr_number + " " + sub.addr_street).str.strip()
    g = pd.DataFrame({"v": sub[field], "fn": sub.first_name, "ln": sub.last_name.where(sub.last_name != ""),
                      "ad": addr.where(addr != "")}).groupby("v")
    stats = pd.DataFrame({"n": g.size(), "first_names": g.fn.nunique(), "surnames": g.ln.nunique(),
                          "with_surname": g.ln.count(), "addrs": g.ad.nunique(), "with_addr": g.ad.count()})
    if not aware:
        return {v: ("personal", 1.0) for v in stats.index}
    n = stats.n.to_numpy()
    diversity, addr_diversity = name_and_address_diversity(
        n, stats.first_names.to_numpy(), stats.surnames.to_numpy(), stats.with_surname.to_numpy(),
        stats.addrs.to_numpy(), stats.with_addr.to_numpy())
    syntax = syntactic_junk_mask(stats.index)
    # junk: carried by many records that agree on nothing - not the same family, and not
    # the same address. A big household is diverse in surname but NOT in address.
    junk = syntax | ((n >= JUNK_MIN_RECORDS) & (diversity > JUNK_DIVERSITY)
                     & ~(addr_diversity <= JUNK_DIVERSITY))
    other = np.zeros(len(n), bool)
    if field in OTHER_CONTACT and OTHER_CONTACT[field] in sub:
        args = (field, sub[OTHER_CONTACT[field]].to_numpy(), sub[field].to_numpy(), n, diversity,
                stats.with_surname.to_numpy(), addr_diversity)
        junk |= contact_junk(*args)
        other = contact_junk(*args, min_records=PATTERN_MIN_RECORDS)
    junk |= patterned_junk(stats.index, n, diversity, addr_diversity, other)
    # DIFFERENT people (distinct first names) sharing a surname or an address: a household.
    # One person's records duplicated across sources share a first name, so stay "personal".
    household = ~junk & (stats.first_names.to_numpy() >= 2) & (
        (diversity <= HOUSEHOLD_DIVERSITY) | (addr_diversity <= 0.5))
    weight = np.where(junk, 0.0, np.where(household, 0.35, 1.0 / (1.0 + np.log1p(n - 1))))
    cls = np.where(junk, "junk", np.where(household, "household", "personal"))
    return {v: (str(c), float(w)) for v, c, w in zip(stats.index, cls, weight)}


OTHER_CONTACT = {"email": "phone", "phone": "email"}


def contact_junk(field, values, groups, n, names, with_surname, addr_diversity, min_records=JUNK_MIN_RECORDS):
    """A contact shared by many records that carry many different other contacts - one email with
    a dozen mobiles, one mobile with a dozen emails - and agree on no surname or address. People
    reuse the same fake email or mobile, and it collects whatever real contact they also gave;
    this holds when the records have no names, where name diversity cannot tell. values: the
    other contact per record (blank = unknown); groups: the record's value of `field`."""
    if field not in OTHER_CONTACT:
        return np.zeros(len(n), bool)
    o = pd.Series(np.asarray(values, dtype=object)).where(lambda s: s != "")
    g = o.groupby(np.asarray(groups, dtype=object), sort=True)
    distinct, count = g.nunique().to_numpy(), g.count().to_numpy()
    diverse = (count >= min_records) & (distinct / np.maximum(count, 1) > JUNK_DIVERSITY)
    family = (np.asarray(with_surname) >= 2) & (np.asarray(names) <= JUNK_DIVERSITY)
    return (np.asarray(n) >= min_records) & diverse & ~family & ~(np.asarray(addr_diversity) <= JUNK_DIVERSITY)


def name_and_address_diversity(n, first_names, surnames, with_surname, addrs, with_addr):
    """(surname diversity, address diversity) per value, over the records that have each. With
    fewer than two surnames, first-name diversity stands in; with fewer than two addresses,
    address diversity is NaN - unknown, so it neither proves a household nor clears junk."""
    n, with_surname, with_addr = (np.asarray(x, dtype=float) for x in (n, with_surname, with_addr))
    names = np.where(with_surname >= 2, np.asarray(surnames) / np.maximum(with_surname, 1),
                     np.asarray(first_names) / n)
    addr = np.where(with_addr >= 2, np.asarray(addrs) / np.maximum(with_addr, 1), np.nan)
    return names, addr


# ------------------------------------------------------------------------- stage 2: blocking
def unique_ints(a, overwrite=False):
    """np.unique of a 1-d integer array, as a sort and a mask: numpy 2.4's np.unique took 17 s on
    48M int64 pair codes, which this sorts in 1.4. The same sorted values and dtype. overwrite:
    a may be sorted in place (a temporary of the caller's)."""
    s = sort_ints(a, overwrite)
    return s[np.r_[True, s[1:] != s[:-1]]] if len(s) else s


PARALLEL_SORT = 4_000_000


def sort_ints(a, overwrite=False):
    """np.sort(a) for a 1-d integer array, on every core when it is long: slices sorted in threads
    (numpy's sort releases the GIL), then one stable sort, which merges the sorted runs. Half the
    time of np.sort on 100M int64, with no more memory. overwrite: sort a itself, not a copy."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = os.cpu_count() or 1
    s = a if overwrite and isinstance(a, np.ndarray) else np.array(a, copy=True)
    if len(s) < PARALLEL_SORT or workers < 2:
        s.sort()
        return s
    edges = np.linspace(0, len(s), workers + 1).astype(np.int64).tolist()
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda k: s[edges[k]:edges[k + 1]].sort(), range(workers)))
    s.sort(kind="stable")
    return s


def block(df, profiles, aware=True, extra_keys=(), max_block=None, lsh=None, skipped=None, by_key=None, owner=None,
          stop_above=None):
    """Candidate pairs: records sharing any blocking key. Returns (pairs int32 (m, 2), i < j,
    sorted; number of blocks skipped as hubs).

    Keys: surname + first three letters of the first name, surname + street, house number +
    street, first name + street, email, phone (not junk), the country keys (nickname root + date
    of birth, ...), `extra_blocking` and similarity (LSH) keys. extra_keys: column tuples from the
    config's `extra_blocking` - records sharing all of a tuple's (non-empty) values become
    candidates, under the same hub rule. A column may carry a transform, as blocking rules do:
    "first_name:first1", "last_name:soundex", "postcode:first3".
    skipped: a list to fill with (key, record index array) for each skipped block whose values are all
    set - a very common name, say - as opposed to blank ones (no address), which mean nothing.
    by_key: a list to fill with each key family's candidate pairs and how many of them no earlier
    family proposed (Splink's cumulative comparisons chart): exact up to BY_KEY_EXACT_PAIRS
    candidate pairs; above that, each family's pairs without the overlap ("new" is None). A key of
    two fields also counts the pairs from blocks where one was blank ("pairs_with_a_blank_part"):
    first name + street is first name alone where there is no street.

    Everything is integer codes and numpy: each key family is factorised to codes, a block is a
    (family, code), and pairs are generated block by block in vectorised steps and de-duplicated
    as int64 codes. The first version built a Python dict of key lists and a Python set of pair
    tuples - about 160 bytes per candidate pair, and most of a full-size run's memory.

    owner: df holds alias views (alias_views) and owner[k] is the record of row k - keys are
    computed per view and entered once per record, so a block's size counts records and the pairs
    are pairs of records.

    stop_above (with skipped): when more records than this are only in skipped blocks - compared
    with no one - return (None, hubs, that number) before making any pair: the caller blocks
    another way (blocking.AUTO_ADAPTIVE_SHARE). A record is compared exactly when a block it is in
    is used."""
    n = len(df) if owner is None else int(owner.max()) + 1
    limit = (max_block or MAX_BLOCK) if aware else 400   # even the naive run needs some ceiling
    from . import country
    fams = []                                           # (codes per entry, record per entry, key of a code)
    labels = []
    blank = []                                          # per family: a code's key has a blank part (or None)
    for fam in blocking_families(df, profiles, aware, extra_keys, lsh):
        labels.append(_family_label(fam, len(labels)))
        if fam[0] == "tuples":
            keys, idx = fam[1], fam[2]
            codes, uniq = pd.factorize(pd.Series(list(keys), dtype=object))
            idx = np.asarray(idx, dtype=np.int64)
            fams.append((codes.astype(np.int64), idx if owner is None else owner[idx], lambda c, u=uniq: u[c]))
            blank.append(None)
        elif fam[0] == "country":
            (tag, a, b), idx = fam[1], fam[2]
            codes, uniq = _factorize_columns([tag, a, b])     # the codes a factorize of the key tuples gives
            fams.append((codes.astype(np.int64), idx if owner is None else owner[idx],
                         lambda c, u=uniq: country.key_tuple(*u[c])))
            blank.append(None)
        else:
            _, tag, arrays, idx = fam
            codes, uniq = _factorize_columns(arrays)
            idx = idx.astype(np.int64)
            part = None
            if len(arrays) > 1 and len(codes):
                part = np.zeros(int(codes.max()) + 1, bool)
                part[codes] = np.logical_or.reduce([np.asarray(a, dtype=object) == "" for a in arrays])
            blank.append(part)
            fams.append((codes.astype(np.int64), idx if owner is None else owner[idx],
                         lambda c, u=uniq, t=tag: tuple(t) + tuple(u[c])))
    if not fams:
        return np.zeros((0, 2), np.int32), 0
    # one block id per (family, code); entries de-duplicated in case a key lists a record twice
    sizes_by_fam = [int(c.max()) + 1 if len(c) else 0 for c, _, _ in fams]
    offsets = np.cumsum([0] + sizes_by_fam[:-1])
    block_id = np.concatenate([c + o for (c, _, _), o in zip(fams, offsets)])
    rec = np.concatenate([i for _, i, _ in fams])
    entry = unique_ints(block_id * n + rec, overwrite=True)
    block_id, rec = entry // n, entry % n
    sizes = np.bincount(block_id, minlength=int(offsets[-1] + sizes_by_fam[-1]))
    hubs = int(((sizes > limit) & (sizes >= 2)).sum()) if aware else 0
    if skipped is not None and aware:
        # entries are sorted by block, so a block's records are one slice: binary search, not a scan
        # of every entry per block (that was thousands of full scans at five million records)
        big = np.flatnonzero(sizes > limit)
        starts, ends = np.searchsorted(block_id, big, side="left"), np.searchsorted(block_id, big, side="right")
        for b, lo, hi in zip(big.tolist(), starts.tolist(), ends.tolist()):
            f = int(np.searchsorted(offsets, b, side="right") - 1)
            key = fams[f][2](int(b - offsets[f]))
            if all(str(v) for v in key[1:]):
                skipped.append((key, rec[lo:hi].copy()))      # an array: as a list, 695 MB at 5M records
    keep = (sizes[block_id] >= 2) & (sizes[block_id] <= limit)
    if stop_above is not None and skipped is not None:
        lonely = np.zeros(n, bool)
        for _, idx in skipped:
            lonely[idx] = True
        lonely[rec[keep]] = False
        if lonely.sum() > stop_above:
            return None, hubs, int(lonely.sum())
    block_id, rec = block_id[keep], rec[keep]           # sorted by block, then record
    codes = _block_codes(block_id, rec, n)
    if by_key is not None:
        exact = len(codes) <= BY_KEY_EXACT_PAIRS
        seen = np.zeros(0, np.int64)
        for f, label in enumerate(labels):
            lo, hi = offsets[f], offsets[f] + sizes_by_fam[f]
            m = (block_id >= lo) & (block_id < hi)
            s = sizes[lo:hi]
            used = (s >= 2) & (s <= limit)               # the family's blocks that made pairs
            row = {"key": label, "blocks_used": int(used.sum()),
                   "blocks_skipped": int(((s > limit) & (s >= 2)).sum()) if aware else 0}
            if exact:
                fc = _block_codes(block_id[m], rec[m], n)
                new = np.setdiff1d(fc, seen, assume_unique=True)
                seen = np.sort(np.concatenate([seen, new]))   # np.union1d, as new is not in seen: 52 of 67 s at 1M
                row |= {"pairs": int(len(fc)), "new": int(len(new)), "cumulative": int(len(seen))}
            else:                                   # within-block pairs, before removing overlap
                row |= {"pairs": int((s[used] * (s[used] - 1) // 2).sum()), "new": None, "cumulative": None}
            if blank[f] is not None:                # within-block pairs, as some are also another key's
                k = used & blank[f]
                row["pairs_with_a_blank_part"] = int((s[k] * (s[k] - 1) // 2).sum())
            by_key.append(row)
    return pairs_from_codes(codes, n), hubs


BY_KEY_EXACT_PAIRS = 50_000_000
_KEY_LABELS = {"nm": "surname + first 3 letters", "ns": "surname + street", "ad": "house number + street",
               "fn": "first name + street", "email": "email", "phone": "phone"}


def _family_label(fam, k):
    if fam[0] in ("tuples", "country"):
        return "nickname, email user and name-skeleton keys" if k == 0 else "similarity (LSH)"
    tag = fam[1]
    return f"extra blocking {tag[1] + 1}" if tag[0] == "x" else _KEY_LABELS.get(tag[0], tag[0])


def _block_codes(block_id, rec, n):
    """Distinct pairs (as i * n + j, i < j) of records sharing a block; entries sorted by block,
    then record."""
    if not len(block_id):
        return np.zeros(0, np.int64)
    return join_pieces(list(block_pair_codes(rec, block_ends(block_id), n)), np.zeros(0, np.int64))


def join_pieces(pieces, empty):
    """np.concatenate(pieces), each piece freed once it is copied: the output's pages are touched
    only as they are filled, so the pieces and the whole are never held at once (pieces: a list
    the caller gives up)."""
    if len(pieces) < 2:
        return pieces.pop() if pieces else empty
    out = np.empty((sum(map(len, pieces)),) + pieces[0].shape[1:], np.result_type(*pieces))
    at = 0
    pieces.reverse()
    while pieces:
        p = pieces.pop()
        out[at:at + len(p)] = p
        at += len(p)
        del p
    return out


def pairs_from_codes(codes, n):
    """(i, j) int32 rows from pair codes i * n + j, filled column by column: no int64 copy of the
    pairs."""
    out = np.empty((len(codes), 2), np.int32)
    out[:, 0] = codes // n
    out[:, 1] = codes % n
    return out


def block_ends(block_id):
    """For entries sorted by block, the end (one past the last entry) of each entry's block."""
    starts = np.flatnonzero(np.r_[True, block_id[1:] != block_id[:-1]])
    return np.repeat(np.r_[starts[1:], len(block_id)], np.diff(np.r_[starts, len(block_id)]))


PAIR_CHUNK = 40_000_000     # within-block pairs made at once, before their repeats are removed


def block_pair_codes(rec, end, n, tag=None, n_tags=1, chunk=None):
    """The pairs of entries sharing a block, as sorted distinct codes (i * n + j) * n_tags + tag,
    in pieces: each piece the pairs whose first record is in one range of records, the ranges in
    order, so the pieces joined are sorted too. Entries are sorted by block, then record (rec
    ascending within a block); end is block_ends; tag (per entry, < n_tags) is kept with each pair,
    as the rule that made it.

    Pairs are made in vectorised steps: the k-th step pairs each entry with the k-th entry after
    it while both are in one block. A range holds about `chunk` pairs before their repeats are
    removed: made all at once and then sorted, the repeats - one pair from several keys - set the
    peak memory of blocking (391M codes, 27 GB, on five million NCVR records)."""
    chunk = chunk or PAIR_CHUNK
    later = end - 1 - np.arange(len(rec))
    # each entry pairs with every later entry of its block, all with its record as the first
    per_record = np.cumsum(np.bincount(rec, weights=later, minlength=n))
    del later
    cuts = np.searchsorted(per_record, np.arange(1, int(per_record[-1] // chunk) + 1) * chunk, side="right")
    edges = np.unique(np.r_[0, cuts, n])
    del per_record
    for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist()):
        a = np.flatnonzero((rec >= lo) & (rec < hi))
        out = []
        for k in range(1, int((end[a] - a).max()) if len(a) else 1):
            a = a[a + k < end[a]]              # entries whose block still has a k-th one after them
            code = rec[a].astype(np.int64) * n      # rec may be int32
            code += rec[a + k]
            if tag is not None:
                code *= n_tags
                code += tag[a]
            out.append(code)
        if out:
            yield unique_ints(np.concatenate(out), overwrite=True)
        del out


def _factorize_columns(arrays):
    """(codes, keys) for rows of several columns: each column factorised on its own, the codes
    combined as integers and factorised again - the same codes, in order of first appearance, as
    a MultiIndex factorize of the tuples, without building the tuples (a third of blocking)."""
    if len(arrays) == 1:
        codes, uniq = pd.factorize(arrays[0])
        return codes, _KeysAt([np.asarray(uniq, dtype=object)], None)
    parts = [pd.factorize(a) for a in arrays]
    combined = np.zeros(len(arrays[0]), dtype=np.int64)
    for codes, uniq in parts:
        # +1 keeps every code distinct; compressed after each column, so millions of values in
        # three or more columns cannot overflow 64 bits
        combined = pd.factorize(combined * (len(uniq) + 1) + codes)[0].astype(np.int64)
    codes, first = pd.factorize(combined)
    # each combined code's key: the column values of its first row
    rows = pd.Series(np.arange(len(combined))).groupby(codes).first().to_numpy() if len(combined) else np.zeros(0, int)
    return codes, _KeysAt([np.asarray(a, dtype=object) for a in arrays], rows)


class _KeysAt:
    """keys[c]: code c's key, the tuple of its columns' values, made when asked - only skipped
    blocks' keys ever are, and a tuple per key was gigabytes on five million records."""

    def __init__(self, arrays, rows):
        self.arrays, self.rows = arrays, rows

    def __getitem__(self, c):
        r = c if self.rows is None else self.rows[c]
        return tuple(a[r] for a in self.arrays)


def blocking_families(df, profiles, aware=True, extra_keys=(), lsh=None):
    """The default blocker's keys (block's docstring lists them), family by family - the one
    definition block and the real-time matcher (blocking_hashes) both use. Yields
    ("tuples", [key tuple, ...], [record index, ...]), ("country", [tag, a, b arrays], record
    indices) - country.blocking_key_columns - or ("columns", tag, [value arrays], record indices):
    a column family's key is the tag plus the records' values in those columns."""
    from . import country
    from .blocking import Keys
    n = len(df)
    *cols, idx = country.blocking_key_columns(df, profiles if aware else None)
    if len(idx):
        yield "country", cols, idx                      # country.key_tuple(tag, a, b) is a key
    if lsh:                                             # similarity keys (blocking.lsh_keys)
        from .blocking import lsh_keys
        entries = lsh_keys(df, lsh)
        if entries:
            idx, keys = zip(*entries)
            yield "tuples", list(keys), list(idx)

    def columns(tag, cols, mask=None):
        idx = np.arange(n) if mask is None else np.flatnonzero(mask)
        return "columns", tag, [np.asarray(c, dtype=object)[idx] for c in cols], idx

    kk = Keys(df)
    for m, cols in enumerate(extra_keys):
        vals = [kk.get(c if ":" in c else f"{c}:exact") for c in cols]
        yield columns(("x", m), vals, np.logical_and.reduce([np.asarray(v, dtype=object) != "" for v in vals]))
    fn, ln = df.first_name.to_numpy(dtype=object), df.last_name.to_numpy(dtype=object)
    st, num = df.addr_street.to_numpy(dtype=object), df.addr_number.to_numpy(dtype=object)
    yield columns(("nm",), [ln, df.first_name.str[:3].to_numpy(dtype=object)])
    yield columns(("ns",), [ln, st])
    yield columns(("ad",), [num, st])
    yield columns(("fn",), [fn, st])
    for field in ("email", "phone"):
        v = df[field].to_numpy(dtype=object)
        keep = v != ""
        if aware:
            prof = profiles[field]
            keep &= np.array([prof.get(x, ("personal",))[0] != "junk" if x else False for x in v], dtype=bool)
        yield columns((field,), [v], keep)


_MIX = np.uint64(0x100000001B3)


def _hash_strings(values):
    return pd.util.hash_array(np.asarray(values, dtype=object), categorize=False)


def blocking_hashes(df, profiles, extra_keys=(), lsh=None):
    """Every blocking key of every record as a 64-bit hash: (hashes uint64, record indices int64).
    The real-time matcher's index: 16 bytes an entry instead of a Python tuple in a dict of lists.
    Two keys sharing a hash (about 1 in 10^19) would only add candidates, which are then scored."""
    hs, rs = [], []
    for fam in blocking_families(df, profiles, True, extra_keys, lsh):
        if fam[0] == "tuples":
            h = _hash_strings(["\x1f".join(map(str, k)) for k in fam[1]])
            idx = np.asarray(fam[2], dtype=np.int64)
        elif fam[0] == "country":
            from .country import key_tuple
            h = _hash_strings(["\x1f".join(map(str, key_tuple(*k))) for k in zip(*fam[1])])
            idx = fam[2]
        else:
            _, tag, arrays, idx = fam
            h = np.full(len(idx), _hash_strings(["\x1f".join(map(str, tag))])[0], dtype=np.uint64)
            with np.errstate(over="ignore"):
                for a in arrays:
                    h = (h * _MIX) ^ _hash_strings(a)
        hs.append(h)
        rs.append(np.asarray(idx, dtype=np.int64))
    if not hs:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64)
    return np.concatenate(hs), np.concatenate(rs)


# ------------------------------------------------------------------------- stage 3: features
_DATE_PATTERNS = [
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$"), (0, 1, 2)),     # 1971-06-24
    (re.compile(r"^(\d{4})(\d{2})(\d{2})$"), (0, 1, 2)),           # 19560409
    (re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$"), (2, 0, 1)),     # 05/06/1994 (month first)
]


def parse_dates(values):
    """(year, month, day) int arrays, -1 where a value does not parse. Handles dates pandas
    cannot (historical_50k has births before 1677). Each distinct value is parsed once."""
    codes, uniq = pd.factorize(np.asarray(values, dtype=object), use_na_sentinel=False)
    out = np.full((len(uniq), 3), -1, dtype=np.int64)
    for k, v in enumerate(uniq):
        for pat, order in _DATE_PATTERNS:
            m = pat.match(v)
            if m:
                g = m.groups()
                out[k] = [int(g[order[0]]), int(g[order[1]]), int(g[order[2]])]
                break
    return out[codes] if len(values) else np.full((0, 3), -1, dtype=np.int64)


def record_arrays(df):
    """Per-record values the rich features need, computed once per run: parsed dates and their
    normalised strings, name counts, first initials. Recomputing them on every features() call
    cost 44 s of a 155 s run at 500k records (the in-cluster check calls it once per threshold)."""
    dates = parse_dates(df.date_of_birth.to_numpy())
    codes, uniq = pd.factorize(dates[:, 0] * 10_000 + dates[:, 1] * 100 + dates[:, 2])
    first = np.zeros(len(uniq), np.int64)
    first[codes[::-1]] = np.arange(len(codes) - 1, -1, -1)
    # each distinct date's string once: "%04d%02d%02d", as before, "-001-1-1" for no date
    strings = np.array([f"{y:04d}{m:02d}{d:02d}" for y, m, d in dates[first]])
    out = {"dates": dates,
           "date_str": strings[codes] if len(codes) else np.array([]),
           "initial": df.first_name.str[:1].to_numpy()}
    for field in ("first_name", "last_name"):
        v = pd.Series(df[field].to_numpy())
        out[f"{field}_count"] = v.map(v.value_counts()).to_numpy().astype(float)
    return out


def rich_features(df, pairs, rec=None):
    """Nine extra features for data where names and dates drift (docs/report/ and docs/guide.md: what Splink's
    comparison levels carry and the 13 standard features do not):

      rarity of an agreeing first / last name (agreeing on a rare name is strong evidence),
      the best first-last swapped name similarity, matching first initials,
      graded dates: log day difference, same year, day-month swapped, month-day equal
      (a year typo), and date strings one edit apart.  -1 where a value is missing."""
    from rapidfuzz.distance import Levenshtein
    from rapidfuzz.process import cpdist
    rec = rec if rec is not None else record_arrays(df)
    i, j = pairs[:, 0], pairs[:, 1]
    cols = []
    for field in ("first_name", "last_name"):
        v = _code_column(df, field)
        counts = rec[f"{field}_count"]
        agree = (v[i] == v[j]) & (v[i] >= 0)
        n_file = rec.get("n_records", len(v))     # the whole file's size when scoring a few records
        cols.append(np.where(v[i] < 0, -1.0, np.where(agree, np.log(n_file / counts[i]), 0.0)))
    swap = np.maximum(_similarity(df, "first_name", "last_name", i, j, JaroWinkler.normalized_similarity),
                      _similarity(df, "last_name", "first_name", i, j, JaroWinkler.normalized_similarity))
    cols.append(swap)
    fc = _code_column(df, "first_name")
    has = (fc[i] >= 0) & (fc[j] >= 0)
    cols.append(np.where(has, rec["initial"][i] == rec["initial"][j], -1.0))
    a, b = rec["dates"][i], rec["dates"][j]
    ok = (a[:, 0] >= 0) & (b[:, 0] >= 0)
    days = np.abs((a[:, 0] - b[:, 0]) * 365.25 + (a[:, 1] - b[:, 1]) * 30.44 + (a[:, 2] - b[:, 2]))
    cols.append(np.where(ok, np.log1p(days), -1.0))
    cols.append(np.where(ok, a[:, 0] == b[:, 0], -1.0))
    cols.append(np.where(ok, (a[:, 1] == b[:, 2]) & (a[:, 2] == b[:, 1]) & (a[:, 0] == b[:, 0]), -1.0))
    cols.append(np.where(ok, (a[:, 1] == b[:, 1]) & (a[:, 2] == b[:, 2]), -1.0))
    edits = np.zeros(len(pairs))                  # only where both dates parsed: elsewhere -1
    if ok.any():
        sa, sb = rec["date_str"][i[ok]], rec["date_str"][j[ok]]
        edits[ok] = cpdist(list(sa), list(sb), scorer=Levenshtein.distance, workers=-1)
    cols.append(np.where(ok, edits <= 1, -1.0))
    return np.column_stack(cols).astype(np.float32) if len(pairs) else np.zeros((0, 9), np.float32)


FEATURE_CHUNK = 2_000_000
# pairs scored at once (about 0.8 GB of features): a forest predicts in threads of a fixed number
# of rows, and on one FEATURE_CHUNK at a time the last wave of threads left most cores idle
# (NCVR, 47M pairs: 198 s, against 163 s holding every pair's features)
SCORE_CHUNK = 8_000_000


def features(df, pairs, profiles, aware=True, extra_fields=(), rich=False, records=None):
    """13 standard features, then one per extra field: `exact` -> 1 / 0 / -1 (missing),
    `similarity` -> Jaro-Winkler, `edit` -> normalised Levenshtein (dates, codes); -1 when missing.
    An extra field may name a standard column (e.g. date_of_birth) to compare it fuzzily too.

    Computed FEATURE_CHUNK pairs at a time into one float32 array: each pair's row depends only
    on its two records, and the per-chunk temporaries (string lists for the similarity scorers)
    stay bounded however many pairs there are.

    With alias fields ("aliases") each pair is compared through its best-agreeing views (best_views)."""
    chunks = feature_chunks(df, pairs, profiles, aware, extra_fields, rich, records)
    if len(pairs) <= FEATURE_CHUNK:
        return next(chunks)[1]
    X = None
    for s, part in chunks:
        if X is None:
            X = np.empty((len(pairs), part.shape[1]), np.float32)
        X[s:s + len(part)] = part
    return X


def feature_chunks(df, pairs, profiles, aware=True, extra_fields=(), rich=False, records=None):
    """features() FEATURE_CHUNK pairs at a time, as (first row, rows): for a caller that uses each
    chunk and drops it, so the features of every pair are never held at once."""
    if alias_views(df)[1] is not None:
        pairs = best_views(df, pairs)
        records = _view_records(df) if rich else None
        df = alias_views(df)[0]
    if rich and records is None and len(pairs) > FEATURE_CHUNK:
        records = record_arrays(df)
    for s in range(0, max(len(pairs), 1), FEATURE_CHUNK):
        yield s, _features(df, pairs[s:s + FEATURE_CHUNK], profiles, aware, extra_fields, rich, records)


def _features(df, pairs, profiles, aware=True, extra_fields=(), rich=False, records=None):
    from . import country
    X = _standard_features(df, pairs, profiles, aware)
    if rich:
        X = np.hstack([X, rich_features(df, pairs, records)])
    X = np.hstack([X, country.features(df, pairs, profiles)])
    if not extra_fields or not len(pairs):
        return X if not extra_fields else np.zeros((0, X.shape[1] + len(extra_fields)), np.float32)
    cols = []
    for name, compare in extra_fields:
        if compare == "years_apart":
            t = pd.to_datetime(df[name].replace("", None), errors="coerce", format="mixed", utc=True)
            yr = (t.dt.year + (t.dt.day_of_year - 1) / 365.25).to_numpy(dtype=float)
            gap = np.abs(yr[pairs[:, 0]] - yr[pairs[:, 1]])
            cols.append(np.where(np.isnan(gap), -1.0, gap))
            continue
        c = _code_column(df, name)
        ca, cb = c[pairs[:, 0]], c[pairs[:, 1]]
        both = (ca >= 0) & (cb >= 0)
        if compare in ("similarity", "edit"):
            from rapidfuzz.distance import Levenshtein
            scorer = JaroWinkler.normalized_similarity if compare == "similarity" \
                else Levenshtein.normalized_similarity
            sim = np.full(len(pairs), -1.0)       # compared only where both are filled
            k = np.flatnonzero(both)
            sim[k] = _similarity(df, name, name, pairs[k, 0], pairs[k, 1], scorer)
            cols.append(sim)
        else:
            cols.append(np.where(both, (ca == cb).astype(float), -1.0))
    return np.hstack([X, np.column_stack(cols).astype(np.float32)])


STANDARD_FEATURES = ["first name similarity", "surname similarity", "full name similarity", "middle initial agrees",
                     "house number agrees", "street similarity", "unit agrees", "email agrees", "email weight",
                     "phone agrees", "phone weight", "date of birth agrees", "SSN agrees"]
RICH_FEATURES = ["first name rarity (agreeing)", "surname rarity (agreeing)", "first/last name swapped",
                 "first initial agrees", "date gap (log days)", "birth year agrees", "day and month swapped",
                 "day and month agree", "dates one edit apart"]


def feature_names(columns, extra_fields=(), rich=False, match_weight=False):
    """The features' names, in the order core.features builds them, for a record frame with these
    columns - for explaining a score (which feature moved it, and by how much)."""
    names = list(STANDARD_FEATURES) + (list(RICH_FEATURES) if rich else [])
    columns = set(columns)
    names += [n for col, n in (("first_roots", "first names share a root"), ("email_user", "email user names agree"),
                               ("name_skeleton", "name skeletons agree (across scripts)")) if col in columns]
    names += [f"{name} ({compare})" for name, compare in extra_fields]
    return names + (["match weight (Fellegi-Sunter)"] if match_weight else [])


_WEIGHTS, _COLUMNS, _VIEWS = {}, {}, {}


def clear_caches():
    """Drop the feature caches (object columns, contact weights, alias views). The pipeline calls
    it once scoring is done: at ten million records the cached columns are several gigabytes."""
    _COLUMNS.clear()
    _WEIGHTS.clear()
    _VIEWS.clear()


# ------------------------------------------------------------------- alias fields ("aliases")
# A record may carry other values of a field: a previous or maiden surname, a second email, a
# work phone (io.load_config reads them into columns named "field@alias"). Each alias value
# makes another VIEW of its record - the record with that value in place of the field - and
# views are blocked, profiled and compared like records: two records are candidates when any
# views share a key, and a pair is compared through its two views that agree best on the alias
# fields, so the best-matching value counts (zer, ZAL-Analytics/zer: AliasPhoneticKey and
# AliasTokenOverlapSimilarity). Customers, golden records and every output stay one row per record.
ALIAS_FIELDS = ("last_name", "email", "phone")


def alias_columns(df):
    return [c for c in df.columns if "@" in c and c.split("@", 1)[0] in ALIAS_FIELDS]


def service_records(df):
    """Records with a first name only that is their email's domain ("zoomly", hello@zoomly.example):
    usually a service, product or practice saved as a person. Boolean mask."""
    if "email" not in df.columns:
        return np.zeros(len(df), bool)
    name = df.first_name.str.replace(r"[^a-z0-9]", "", regex=True)
    label = df.email.str.partition("@")[2].str.partition(".")[0].str.replace(r"[^a-z0-9]", "", regex=True)
    inside = np.array([n in d for n, d in zip(name, label)], bool)
    return ((df.last_name == "") & (name.str.len() >= 4)).to_numpy() & inside


def relay_records(df):
    """(mask, addresses): records whose email or an alias email is a service relay (RELAY_EMAIL),
    and the relay addresses found. Each distinct address is tested once."""
    mask, found = np.zeros(len(df), bool), set()
    for c in ["email"] + [c for c in alias_columns(df) if c.startswith("email@")]:
        hits = [v for v in df[c].unique() if v and RELAY_EMAIL.search(v)]
        if hits:
            mask |= df[c].isin(hits).to_numpy()
            found.update(hits)
    return mask, sorted(found)


def alias_views(df):
    """(views, owner): the records, followed by one row per combination of their alias values - a
    record with a maiden name and a second email has three more: maiden name, second email, both
    - and the record each row belongs to; (df, None) when df has no alias columns. Comparing two
    records through their best pair of views is then comparing each field through its best pair
    of values. Cached per frame, like the feature columns."""
    cols = alias_columns(df)
    if not cols:
        return df, None
    hit = _VIEWS.get(id(df))
    if hit is not None and hit[0] is df:
        return hit[1]
    if len(_VIEWS) > 2:
        _VIEWS.clear()
    fields = sorted({c.split("@", 1)[0] for c in cols})
    cur = pd.DataFrame({"owner": np.arange(len(df))} | {f: df[f].fillna("").to_numpy(dtype=object) for f in fields})
    for f in fields:
        own = df[f].fillna("").to_numpy(dtype=object)
        alts = pd.concat([pd.DataFrame({"owner": np.arange(len(df)), "alt": df[c].fillna("").to_numpy(dtype=object)})
                          for c in cols if c.split("@", 1)[0] == f], ignore_index=True)
        alts = alts[(alts.alt != "") & (alts.alt.to_numpy() != own[alts.owner.to_numpy()])].drop_duplicates()
        if len(alts):
            more = cur.merge(alts, on="owner").assign(**{f: lambda t: t.alt}).drop(columns="alt")
            cur = pd.concat([cur, more], ignore_index=True)
    if len(cur) == len(df):
        views = (df, None)
    else:
        extra = cur.iloc[len(df):]
        v = df.iloc[extra.owner.to_numpy()].copy()
        for f in fields:
            v[f] = extra[f].to_numpy()
        if "email" in fields and "email_user" in v.columns:
            v["email_user"] = [e.split("@")[0] if "@" in e else "" for e in v.email]
        views = (pd.concat([df, v], ignore_index=True), cur.owner.to_numpy())
    _VIEWS[id(df)] = (df, views, {})
    return views


def _view_records(df):
    """record_arrays over df's views, once per frame (rich features read them)."""
    views, _ = alias_views(df)
    extra = _VIEWS[id(df)][2]
    if "records" not in extra:
        extra["records"] = record_arrays(views)
    return extra["records"]


def best_views(df, pairs):
    """Pairs of records as pairs of views: for each pair, the two views (a record or one of its
    alias rows) that agree best on the alias fields - surname by Jaro-Winkler, email and phone
    exactly - the records themselves on a tie. Pairs whose records have no aliases are unchanged."""
    from rapidfuzz.process import cpdist
    views, owner = alias_views(df)
    pairs = np.asarray(pairs)
    if owner is None or not len(pairs):
        return pairs
    n = len(df)
    v_own = owner[n:]
    order = np.argsort(v_own, kind="stable")
    v_idx, v_own = np.arange(n, len(views))[order], v_own[order]
    start = np.searchsorted(v_own, np.arange(n))
    count = 1 + np.searchsorted(v_own, np.arange(n), side="right") - start
    need = np.flatnonzero((count[pairs[:, 0]] > 1) | (count[pairs[:, 1]] > 1))
    if not len(need):
        return pairs
    i, j = pairs[need, 0].astype(np.int64), pairs[need, 1].astype(np.int64)
    combos = count[i] * count[j]
    pid = np.repeat(np.arange(len(need)), combos)
    k = np.arange(combos.sum()) - np.repeat(np.cumsum(combos) - combos, combos)
    a, b = k // count[j][pid], k % count[j][pid]          # 0 = the record itself

    def view(rec, m):
        return np.where(m == 0, rec, v_idx[np.clip(start[rec] + m - 1, 0, max(len(v_idx) - 1, 0))])
    vi, vj = view(i[pid], a), view(j[pid], b)
    score = np.zeros(len(pid))
    for field in sorted({c.split("@", 1)[0] for c in alias_columns(df)}):
        col = views[field].fillna("").to_numpy(dtype=object)
        x, y = col[vi], col[vj]
        both = (x != "") & (y != "")
        if field == "last_name":
            score += np.where(both, cpdist(list(x), list(y), scorer=JaroWinkler.similarity, workers=-1), 0.0)
        else:
            score += both & (x == y)
    o = np.lexsort((k, -score, pid))                       # best score first, the records themselves on a tie
    best = o[np.r_[True, pid[o][1:] != pid[o][:-1]]]
    out = pairs.astype(np.int64).copy()
    out[need, 0], out[need, 1] = vi[best], vj[best]
    return out


def _object_column(df, c):
    """df[c] (blanks for missing) as a numpy object array, converted once per frame: features
    are computed dozens of times on the same records, and converting Arrow strings each time cost
    about 8 s of a million-record run. Keyed by the frame's identity; a frame changed in place
    must not be reused (the pipeline makes new frames rather than changing them)."""
    return _cached(df, c, lambda: df[c].fillna("").to_numpy(dtype=object))


def _cached(df, key, make):
    """make() once per frame: the per-record arrays the features read (object columns, codes)."""
    hit = _COLUMNS.get(id(df))
    if hit is None or hit[0] is not df:
        if len(_COLUMNS) > 4:
            _COLUMNS.clear()
        hit = _COLUMNS[id(df)] = (df, {})
    if key not in hit[1]:
        hit[1][key] = make()
    return hit[1][key]


def _coded(values):
    """(codes, strings) of an object array of strings: integer codes, -1 for "", and each code's
    string (strings[-1] is ""). Equal codes are equal strings, so a pair's agreement is an integer
    comparison (on object arrays each comparison calls Python's string compare; this was most of the
    feature time at a million records)."""
    codes, uniq = pd.factorize(values)
    codes = codes.astype(np.int32)
    blank = np.flatnonzero(uniq == "")
    if len(blank):
        codes[codes == blank[0]] = -1
    return codes, np.append(np.asarray(uniq, dtype=object), "")


def _codes(values):
    return _coded(values)[0]


def _coded_column(df, c, values=None):
    """_coded of df[c] (blanks for missing), or of values() under the name c, once per frame."""
    return _cached(df, ("coded", c), lambda: _coded(values() if values else _object_column(df, c)))


def _code_column(df, c):
    return _coded_column(df, c)[0]


def _similarity(df, a, b, i, j, scorer, values_a=None):
    """scorer(df[a][i], df[b][j]) per pair, computed once per distinct pair of values."""
    from rapidfuzz.process import cpdist
    ca, sa = _coded_column(df, a, values_a)
    cb, sb = (ca, sa) if b == a else _coded_column(df, b)
    return per_value_pairs(ca[i], cb[j], lambda x, y: cpdist(list(sa[x]), list(sb[y]), scorer=scorer, workers=-1))


def per_value_pairs(ca, cb, fn):
    """fn(a, b) for each pair of codes, computed once per distinct pair: fn takes two code arrays and
    returns one value per pair. Pairs of names or roots repeat thousands of times in a block."""
    key = (ca.astype(np.int64) << 32) | (cb.astype(np.int64) & 0xFFFFFFFF)
    inv, u = pd.factorize(key)                    # a hash table: 2 to 20 times faster than sorting
    return np.asarray(fn((u >> 32).astype(np.int32), (u & 0xFFFFFFFF).astype(np.uint32).view(np.int32)))[inv]


def _weight_series(profile):
    """{value: (class, weight)} as a Series of weights, built once per profile: the threshold
    search computes features dozens of times, and rebuilding it each time cost 9 s at a million."""
    hit = _WEIGHTS.get(id(profile))
    if hit is None or hit[0] is not profile:
        if len(_WEIGHTS) > 8:
            _WEIGHTS.clear()
        hit = _WEIGHTS[id(profile)] = (profile, pd.Series({k: t[1] for k, t in profile.items()}, dtype=float))
    return hit[1]


def _standard_features(df, pairs, profiles, aware=True):
    """Two blank surnames score 1.0 (agreement) and a surname against a blank 0.0. (Scoring a
    blank as unknown, -1, was tried as "missing_as_unknown" and removed: with no labelled example
    of a blank the model could not use it, and it made sparse records worse.)

    Column by column with numpy and rapidfuzz's cpdist: the first version built each pair's row
    in a Python loop - identical values, 2.8x slower at a million records (research/poc/vectorised_features.py).
    Agreement compares integer codes, and each similarity is computed once per distinct pair of
    values (_coded_column, _similarity)."""
    i, j = pairs[:, 0], pairs[:, 1]
    col = lambda c: _object_column(df, c)  # noqa: E731
    em, ph = col("email"), col("phone")
    jw = lambda c, values=None: _similarity(df, c, c, i, j, JaroWinkler.similarity, values)  # noqa: E731
    code = lambda c: _code_column(df, c)  # noqa: E731
    agree = lambda c: (code(c)[i] >= 0) & (code(c)[i] == code(c)[j])  # noqa: E731
    both = lambda c: (code(c)[i] >= 0) & (code(c)[j] >= 0)  # noqa: E731
    full = lambda: (df.first_name.fillna("") + df.last_name.fillna("")).to_numpy(dtype=object)  # noqa: E731
    e_match, p_match = agree("email"), agree("phone")

    def weight(field, v, match):
        if not aware:
            return match.astype(float)
        w = _weight_series(profiles[field])
        got = w.reindex(v[i]).to_numpy() if len(w) else np.full(len(i), np.nan)
        return np.where(match, np.nan_to_num(got, nan=1.0), 0.0)        # a value not profiled weighs 1.0

    if not len(pairs):
        return np.zeros((0, 13), dtype=np.float32)
    return np.column_stack([
        jw("first_name"), jw("last_name"), jw("full_name", full),
        agree("middle_initial"), agree("addr_number"), jw("addr_street"), agree("addr_unit"),
        e_match, weight("email", em, e_match), p_match, weight("phone", ph, p_match),
        np.where(both("date_of_birth"), agree("date_of_birth"), -1.0),
        np.where(both("ssn"), agree("ssn"), -1.0)]).astype(np.float32)


# -------------------------------------------------------------- stages 5/6: cluster + score
def pair_scores(y, pred, n_true_all):
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_true_all
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0,
            "false_merges": fp}


def partition_scores(labels_pred, labels_true, n_true_all):
    def n2(c):
        c = np.asarray(c, dtype=np.float64)
        return float(np.sum(c * (c - 1) / 2))
    _, pi = np.unique(labels_pred, return_inverse=True)
    _, ti = np.unique(labels_true, return_inverse=True)
    sizes = np.bincount(pi)
    m = coo_matrix((np.ones(len(pi)), (pi, ti))).tocsr()
    tp, pred_total = n2(m.data), n2(sizes)
    p = tp / pred_total if pred_total else 0.0
    r = tp / n_true_all
    return {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
            "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0,
            "clusters": int(len(sizes)), "largest_cluster": int(sizes.max()),
            "adjusted_rand": round(adjusted_rand_score(labels_true, labels_pred), 4)}


def components(n, pairs, keep):
    r, c = pairs[keep, 0], pairs[keep, 1]
    g = coo_matrix((np.ones(keep.sum()), (r, c)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _first_seen_labels(cc):
    """cc relabelled in order of each component's lowest record, as connected_components numbers them."""
    return pd.factorize(cc)[0].astype(np.int32)           # codes in order of first appearance


def threshold_bands(score, grid):
    """Each pair's band in a threshold grid: how many thresholds lie below its score, so the pair
    is kept at sorted(grid)[k] exactly when band > k (score > t). One byte a pair: the fold
    models' scores are only ever compared with the grid, and as float64 the five of them were 7.7
    GB of 192M pairs on all 5M NCVR records."""
    grid = np.sort(np.asarray(grid, dtype=float))
    return np.searchsorted(grid, score, side="left").astype(np.uint8 if len(grid) < 255 else np.int64)


PAIR_SLICE = 10_000_000     # pairs a step over every candidate pair takes at once


def pair_slices(n, size=None):
    """(lo, hi) ranges covering n pairs, for a step whose temporaries would otherwise be the size
    of every candidate pair (a float copy of 192M scores is 1.5 GB)."""
    size = size or PAIR_SLICE
    return [(lo, min(lo + size, n)) for lo in range(0, n, size)]


def near_threshold(score, th, band):
    """Indices of the pairs with abs(score - th) < band, a slice at a time."""
    parts = [lo + np.flatnonzero(np.abs(score[lo:hi] - th) < band) for lo, hi in pair_slices(len(score))]
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)


def components_by_threshold(n, pairs, score, grid, band=None):
    """{t: components(n, pairs, score > t)} for each threshold in grid, the same labels, from one
    pass over the pairs: from the strictest threshold down, each step joins the components already
    found with only the pairs whose score lies between it and the stricter one (NCVR, 47M pairs:
    clustering every pair afresh at each threshold was most of the threshold search).
    band: threshold_bands(score, grid), given in place of score."""
    grid = np.sort(np.asarray(grid, dtype=float))
    if band is None:
        band = threshold_bands(score, grid)
    out, cc = {}, None
    for k in range(len(grid) - 1, -1, -1):
        new = np.flatnonzero(band == k + 1)        # kept at grid[k], not at the next threshold up
        if cc is None:
            keep = np.zeros(len(pairs), bool)
            keep[new] = True
            cc = components(n, pairs, keep)
        elif len(new):
            joined = components(int(cc.max()) + 1, cc[pairs[new]].astype(np.int64), np.ones(len(new), bool))
            cc = _first_seen_labels(joined[cc])
        out[grid[k].item()] = cc
    return out


def household_partition(df, profiles, identity_labels):
    """Assign every record to a household.

    A household is an ADDRESS, not a connected component: closure over address edges chains
    separate homes together through anyone who appears at two of them (which is exactly how
    the naive version produced a 2,252-record "household"). So we partition directly:

      1. each customer (identity cluster) is placed at its modal address
      2. customers with no usable address fall back to a shared household-class contact
      3. anything left over is its own household
    """
    addr = (df.addr_number.astype(str) + "|" + df.addr_street.astype(str)
            + "|" + df.addr_unit.astype(str)).to_numpy()
    has_addr = (df.addr_number.astype(str).str.strip() != "").to_numpy()
    hh_contact = np.array([
        next((f"{f}:{v}" for f, v in (("email", e), ("phone", p))
              if v and profiles[f].get(v, ("", 0))[0] == "household"), "")
        for e, p in zip(df.email.to_numpy(), df.phone.to_numpy())])

    by_cluster = defaultdict(list)
    for i, lab in enumerate(identity_labels):
        by_cluster[lab].append(i)

    household_of = {}
    for lab, idx in by_cluster.items():
        addrs = [addr[i] for i in idx if has_addr[i]]
        if addrs:
            # most common address; ties go to the first seen. (max over a set broke ties by
            # string-hash order, which Python randomises per run - households varied run to run)
            household_of[lab] = "addr:" + Counter(addrs).most_common(1)[0][0]
            continue
        contacts = [hh_contact[i] for i in idx if hh_contact[i]]
        household_of[lab] = ("hh:" + contacts[0]) if contacts else f"own:{lab}"

    labels = np.array([household_of[lab] for lab in identity_labels])
    _, out = np.unique(labels, return_inverse=True)
    n_group = int(np.sum(np.bincount(out) > FAMILY_MAX))
    return out, n_group


