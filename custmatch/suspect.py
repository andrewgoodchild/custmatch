"""Records that look entered in bad faith or by a bot: flagged for review, never changed.

Matching is not affected - a flag is a reason to look, listed in suspect_records.csv:

  profanity   a first name, surname or email word on the LDNOOBW English list (CC BY 4.0,
              downloaded on first use, pinned and checksummed). A name must be the whole field
              (not "beaner-young" or "na'shota"), an email word a whole word of letters (not
              cocktail@ or scunthorpe@); and no word that is a real name is flagged - REAL_NAMES,
              the 36 listed words that are a first name or surname of any of 5 million North
              Carolina voters (Hooker, Dick, Cumming, Felch, Negro...;
              scripts/suspect_research.py). "xx" and "xxx" are placeholders, handled as junk.
  disposable  an email at a throwaway-mail domain
  random      an email whose user part looks generated: a part of it (between dots, dashes,
              underscores or +) of 8 or more characters mixing letters and digits, whose letters
              alternate with digits (k7f2z9) or are under 15% vowels - and that shares nothing
              with the record's names

Turned off with "suspect_records": false. A list that cannot be downloaded skips the profanity
check with a warning; the run goes on.
"""
import re

import numpy as np
import pandas as pd

PROFANITY_FILE = "profanity_en.txt"
# listed words that are a first name or surname of any NCVR voter (scripts/suspect_research.py)
REAL_NAMES = frozenset({
    "anal", "ass", "bastardo", "boner", "boob", "butt", "cock", "cocks", "coon", "coons", "cum", "cumming",
    "dick", "escort", "fag", "felch", "feltch", "guro", "hooker", "horny", "juggs", "lolita", "mong",
    "negro", "nipple", "paki", "poon", "porn", "rape", "santorum", "semen", "smut", "suck", "swinger",
    "tit", "wank"})
PLACEHOLDERS = frozenset({"xx", "xxx"})
DISPOSABLE = frozenset({
    "mailinator.com", "yopmail.com", "guerrillamail.com", "sharklasers.com", "10minutemail.com",
    "temp-mail.org", "tempmail.com", "trashmail.com", "getnada.com", "dispostable.com", "maildrop.cc",
    "throwawaymail.com", "fakeinbox.com", "mintemail.com", "mailnesia.com", "spamgourmet.com"})
_WORD = re.compile(r"[a-z0-9]+")              # an email word: digits stay in it ("f0xx" is not "xx")
_ALTERNATING = re.compile(r"(?:[a-z][0-9]){3,}|(?:[0-9][a-z]){3,}")


def profanity_words():
    """The listed words (lower case), or None when the list cannot be had."""
    from .country import data_file
    from .io import ConfigError
    try:
        path = data_file(PROFANITY_FILE)
        lines = path.read_text(encoding="utf-8").splitlines()
    except (ConfigError, OSError):
        return None
    return frozenset(w.strip().lower() for w in lines if w.strip()) - REAL_NAMES - PLACEHOLDERS


def _random_user(user, names):
    letters = re.sub(r"[^a-z]", "", user)
    if any(len(n) >= 3 and n[:3] in letters for n in names):
        return False                                # jsmith84: built from the name
    for part in re.split(r"[._+-]", user):
        if len(part) < 8 or not (re.search(r"[a-z]", part) and re.search(r"\d", part)):
            continue
        a = re.sub(r"[^a-z]", "", part)
        if _ALTERNATING.search(part) or (len(a) >= 5 and sum(c in "aeiouy" for c in a) / len(a) < 0.15):
            return True
    return False


def flag(df, words=None):
    """{row: [reasons]} for the records that look suspect. words: profanity_words(), or None to
    skip that check. Column-wise, each distinct email once; only the few long mixed user parts
    left are checked against the record's names one by one."""
    first, last = df.first_name.fillna("").str.lower(), df.last_name.fillna("").str.lower()
    code, emails = pd.factorize(df.email.fillna("").str.lower())
    parts = pd.Series(emails, dtype=object).str.partition("@")
    user, domain = parts[0], parts[2]
    found = {}

    def mark(rows, reason):
        for i in rows:
            found.setdefault(int(i), []).append(reason)

    def records(mask_by_email):                     # the records whose email is flagged
        return np.flatnonzero(np.asarray(mask_by_email, dtype=bool)[code]) if len(emails) else []

    if words:
        single = {w for w in words if " " not in w}
        phrases = {w for w in words if " " in w}
        name = first.isin(single) | last.isin(single) | (first + " " + last).isin(phrases)   # whole fields
        mark(np.flatnonzero(name.to_numpy()), "profanity in name")
        tok = user.str.findall(_WORD.pattern).explode()
        mark(records(tok.isin(single).groupby(level=0).any().reindex(user.index, fill_value=False)),
             "profanity in email")
    disposable = domain.isin(DISPOSABLE).to_numpy()
    mark(records(disposable), "disposable email")
    part = user[~disposable & (user.str.len() >= 8)].str.split(r"[._+-]", regex=True).explode()
    part = part[(part.str.len() >= 8) & part.str.contains(r"\d") & part.str.contains(r"[a-z]")]
    letters = part.str.replace(r"[^a-z]", "", regex=True)
    odd = part.str.contains(_ALTERNATING.pattern) | (
        (letters.str.len() >= 5) & (letters.str.count(r"[aeiouy]") < 0.15 * letters.str.len()))
    odd_email = np.zeros(len(emails), bool)
    odd_email[np.unique(odd[odd].index.to_numpy())] = True
    mark([i for i in records(odd_email) if _random_user(user.iat[code[i]], (first.iat[i], last.iat[i]))],
         "random-looking email")
    return found


def table(raw, flags):
    """suspect_records.csv: source, record_id, reasons."""
    rows = sorted(flags)
    return pd.DataFrame({"source": raw.source.to_numpy()[rows] if rows else np.array([], dtype=object),
                         "record_id": raw.record_id.to_numpy()[rows] if rows else np.array([], dtype=object),
                         "reasons": ["; ".join(flags[r]) for r in rows]})
