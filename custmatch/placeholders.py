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
Enabled with "placeholders": true (off by default); detected values are blanked before blocking and features,
and listed in placeholders.csv.
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


def blank(df):
    """A copy with detected placeholder names / addresses blanked, and the masks. Phones in the
    reserved fictional range are blanked too."""
    fake_first, fake_last, fake_addr = detect(df, parts=True)
    fake_name = fake_first | fake_last
    out = df.copy()
    out.loc[fake_first, ["first_name", "middle_initial"]] = ""
    out.loc[fake_last, "last_name"] = ""
    out.loc[fake_addr, ["addr_number", "addr_street", "addr_unit"]] = ""
    from .country import refresh_names       # first-name roots and skeletons of blanked names
    out = refresh_names(out)
    if "phone" in out.columns:
        out.loc[out.phone.astype(str).str.match(FICTIONAL_PHONE), "phone"] = ""
    return out, fake_name, fake_addr
