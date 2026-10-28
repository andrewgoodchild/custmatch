"""Fake / placeholder names and addresses: detected like junk contacts, then treated as missing.

Customer files carry placeholder values typed to get past a mandatory field: "test test",
"john doe", "unknown", "n/a", "123 main st", "1 test st". Left in, they are evidence of the
worst kind - two unrelated people both called "john doe" look like a strong match. Two signals,
as for contacts (core.profile_contacts):

  pattern    known placeholder words and addresses, keyboard walks, a character repeated,
             house number 0 (not digits in names - those are typos in real names)
  diversity  a full name (or an address) carried by many records that agree on nothing else -
             many different dates of birth AND many different addresses (for a name) or many
             different surnames AND dates of birth (for an address). A large family shares a
             surname and an address; a placeholder shares neither.

Single letters are NOT flagged: initials are real data ("i wyatt" in historical_50k).
Phones in the reserved fictional range (xxx-555-0100 to 0199) are blanked with them.

Dates of birth (detect_dates) the same way: a known placeholder (1900-01-01, 1899-12-31 - day
zero in spreadsheets - 0001-01-01, 1111-11-11, any 9999 date); or a date
carried by far more records than dates usually are - at least DATE_MIN_RECORDS and DATE_RATIO
times the median - whose records share neither first names nor surnames (a default typed into a
form: 12/04/2018 on 45 adults in the synthetic customer file). Where 1 January is a convention
for "year only" - a tenth or more of the file's dates, 37% in historical_50k, whose 1862-01-01
is 596 different people - 1 January dates are left alone: they say the year, and the year is
evidence - and so are its 1800-01-01 and 1900-01-01, 246 people born in 1800 in historical_50k.
Elsewhere a crowded 1 January is a default like any other. Dates in the future are NOT flagged:
in fake_1000 and the synthetic file they are typos of real dates (7985 for 1985), still one edit
from the truth, and the date features use that.

Extra fields (detect_extra): a value on a tenth or more of the records with the field and
EXTRA_RATIO times the median value's count, whose records have different surnames and no
employer's domain in common (no work domain - not a big free-mail provider - on half of them):
"Multiple Companies", a social network's placeholder, on a quarter of an address book's cards. A
real employer's staff share a domain; a state or a gender is common but so is every other value
of its field. Only checked where most records have an email: without one, a town's people look
the same as a placeholder's.

Enabled with "placeholders": true (off by default); detected values are blanked before blocking and features,
and listed in placeholders.csv. Placeholder extra-field values are only reported, whatever the
setting: ignoring "Multiple Companies" on synthetic address-book cards changed nothing (300
stratified + 150 active labels, 15 draws: 89.91 compared, 89.18 ignored) - the model already learns
that agreeing on it is worth little.
"""
import re

import numpy as np
import pandas as pd

NAME_WORDS = {"test", "testing", "unknown", "unk", "none", "null", "nil", "na", "n/a",
              "noname", "customer", "user", "anonymous", "anon", "dummy", "fake", "temp", "tbd",
              "tba", "xxx", "xx", "asdf", "qwerty", "abc", "abcd", "nobody", "someone", "somebody",
              "occupant", "resident", "refused", "declined", "notgiven"}
# not listed although placeholder-like: Tester, Guest, Sample, Client, Blank, Current are real
# surnames ("william tester" is in historical_50k)
# honorifics ("sir", "mrs") and suffixes ("iii") are real data on real people, not placeholders -
# they flagged 1,300 historical_50k records
FULL_NAMES = {"john doe", "jane doe", "john smith test", "mickey mouse", "minnie mouse",
              "donald duck", "bugs bunny", "homer simpson", "test test", "first last",
              "firstname lastname", "fname lname", "joe bloggs", "john q public", "jane roe",
              "richard roe", "baby boy", "baby girl"}
ADDRESS_PATTERNS = [re.compile(p) for p in (
    r"^1?2?3 (main|fake|test|any|anywhere|somewhere) ",
    r"\b(test|fake|unknown|none|null|n/?a|no fixed|nofixed|homeless|general delivery|not given|refused)\b",
    r"^0+ ",                    # house number 0; 1234 / 9999 are real addresses too
    r"\b(asdf|qwerty|xxx+)\b",
    r"\bany ?(street|road|st|rd)\b")]
KEYBOARD = ("qwerty", "asdf", "zxcv", "qazwsx", "abcdef", "12345")
FICTIONAL_PHONE = re.compile(r"^1?\d{3}55501\d{2}$")  # xxx-555-0100..0199, also as E.164 (1 + 10 digits)
PLACEHOLDER_DATES = {(1900, 1, 1), (1899, 12, 31), (1899, 12, 30), (1800, 1, 1), (1, 1, 1), (1111, 11, 11)}
DATE_MIN_RECORDS, DATE_RATIO = 20, 10
YEAR_ONLY_SHARE = 0.1          # 1 January on this share of dates or more: a "year only" convention


def _bad_part(part):
    if part in NAME_WORDS:
        return True
    if len(part) >= 4 and len(set(part)) == 1:              # "aaaa" (not "iii", a real suffix)
        return True
    return any(k in part for k in KEYBOARD)


def name_flags(first, last):
    """(first name is a placeholder, last name is a placeholder). A known fake full name flags
    both. Digits in a name are NOT a signal: they are scanning/typing noise in real names
    ("0atrick craigie", "4rthur lee") - flagging them hit 2,230 real historical_50k records."""
    f, l = first.strip().lower(), last.strip().lower()
    if f"{f} {l}".strip() in FULL_NAMES:
        return True, True
    return bool(f) and _bad_part(f), bool(l) and _bad_part(l)


def address_pattern(number, street):
    a = f"{str(number).strip().lower()} {str(street).strip().lower()}".strip()
    return bool(a) and any(p.search(a) for p in ADDRESS_PATTERNS)


def detect(df, min_records=5, diversity=0.6, parts=False):
    """Boolean masks (fake_name, fake_address) per record; parts=True returns
    (fake_first, fake_last, fake_address) so only the placeholder part is blanked."""
    fn, ln = df.first_name.astype(str), df.last_name.astype(str)
    flags = np.array([name_flags(a, b) for a, b in zip(fn, ln)], dtype=bool).reshape(-1, 2)
    fake_first, fake_last = flags[:, 0].copy(), flags[:, 1].copy()
    num, st = df.addr_number.astype(str), df.addr_street.astype(str)
    fake_addr = np.array([address_pattern(a, b) for a, b in zip(num, st)])
    full = (fn.str.strip() + " " + ln.str.strip()).str.strip()
    addr = (num.str.strip() + " " + st.str.strip()).str.strip()
    frame = pd.DataFrame({"full": full, "addr": addr, "dob": df.date_of_birth.astype(str),
                          "ln": ln})
    # a full name shared by many records with different DOBs and different addresses
    g = frame[frame.full.str.contains(" ")].groupby("full")
    stats = pd.DataFrame({"n": g.size(), "dob": g.dob.nunique(), "addr": g.addr.nunique()})
    bad = stats[(stats.n >= min_records) & (stats.dob / stats.n > diversity) & (stats.addr / stats.n > diversity)].index
    shared = full.isin(set(bad)).to_numpy()
    fake_first |= shared
    fake_last |= shared
    # an address shared by many records with different surnames and different DOBs
    g = frame[frame.addr != ""].groupby("addr")
    stats = pd.DataFrame({"n": g.size(), "ln": g.ln.nunique(), "dob": g.dob.nunique()})
    bad = stats[(stats.n >= 3 * min_records) & (stats.ln / stats.n > diversity) & (stats.dob / stats.n > diversity)].index
    fake_addr |= addr.isin(set(bad)).to_numpy()
    if parts:
        return fake_first, fake_last, fake_addr
    return fake_first | fake_last, fake_addr


def detect_dates(df, diversity=0.6):
    """Boolean mask: the record's date of birth is a placeholder (the module docstring says which)."""
    from .core import parse_dates
    if "date_of_birth" not in df.columns:
        return np.zeros(len(df), bool)
    dob = df.date_of_birth.fillna("").astype(str).to_numpy(dtype=object)
    has = dob != ""
    if not has.any():
        return np.zeros(len(df), bool)
    inv, uniq = pd.factorize(dob)
    uniq = np.asarray(uniq, dtype=object)
    ymd = parse_dates(uniq)
    known = np.array([tuple(r) in PLACEHOLDER_DATES or r[0] == 9999 for r in ymd.tolist()])
    counts = np.bincount(inv[has], minlength=len(uniq))
    jan1 = (ymd[:, 1] == 1) & (ymd[:, 2] == 1)
    year_only = counts[jan1].sum() >= YEAR_ONLY_SHARE * counts.sum()
    # where 1 January means "year only", 1800-01-01 and 1900-01-01 are people born in 1800 and 1900
    bad = known & ~(jan1 & year_only) & (uniq != "")
    judged = (counts > 0) & (uniq != "") & ~(jan1 & year_only)
    if judged.any():
        crowded = judged & (counts >= max(DATE_MIN_RECORDS, DATE_RATIO * np.median(counts[judged])))
        first, last = df.first_name.to_numpy(dtype=object), df.last_name.to_numpy(dtype=object)
        for u in np.flatnonzero(crowded & ~bad):
            rows = inv == u
            n = rows.sum()
            if len(set(first[rows])) / n > diversity and len(set(last[rows])) / n > diversity:
                bad[u] = True
    return bad[inv] & has


EXTRA_MIN_SHARE = 0.1          # of the records with the field
EXTRA_RATIO = 20               # times the median value's count
EXTRA_DIVERSITY = 0.6          # distinct surnames / records


def detect_extra(df, cfg):
    """{extra field: [placeholder values]} (the module docstring says which)."""
    from .core import JUNK_MIN_RECORDS
    from .country import COMMON_DOMAINS
    from .schema import extra_fields
    if "email" not in df.columns or (df.email != "").mean() < 0.5:
        return {}
    dom = df.email.str.partition("@")[2]
    work = dom.where(~dom.isin(COMMON_DOMAINS + ["me.com", "mac.com"]), "")
    out = {}
    for field, _ in extra_fields(cfg):
        if field not in df.columns:
            continue
        v = df[field]
        counts = v[v != ""].value_counts()
        crowded = counts[(counts >= max(JUNK_MIN_RECORDS, EXTRA_MIN_SHARE * counts.sum()))
                         & (counts >= EXTRA_RATIO * counts.median())]
        found = []
        for value, n in crowded.items():
            rows = (v == value).to_numpy()
            last, w = df.last_name[rows], work[rows]
            last, w = last[last != ""], w[w != ""]
            names = last.nunique() / len(last) if len(last) >= 2 else df.first_name[rows].nunique() / n
            employer = len(w) and w.value_counts().iloc[0] >= n / 2
            if names > EXTRA_DIVERSITY and not employer:
                found.append(value)
        if found:
            out[field] = found
    return out


def blank(df):
    """A copy with detected placeholder names / addresses / dates of birth blanked, and the name
    and address masks (detect_dates gives the date one). Phones in the reserved fictional range are
    blanked too."""
    fake_first, fake_last, fake_addr = detect(df, parts=True)
    fake_name = fake_first | fake_last
    out = df.copy()
    out.loc[fake_first, ["first_name", "middle_initial"]] = ""
    out.loc[fake_last, "last_name"] = ""
    out.loc[fake_addr, ["addr_number", "addr_street", "addr_unit"]] = ""
    if "date_of_birth" in out.columns:
        out.loc[detect_dates(df), "date_of_birth"] = ""
    from .country import refresh_names       # first-name roots and skeletons of blanked names
    out = refresh_names(out)
    if "phone" in out.columns:
        out.loc[out.phone.astype(str).str.match(FICTIONAL_PHONE), "phone"] = ""
    return out, fake_name, fake_addr
