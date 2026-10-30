"""Country-aware normalisation, for files where names are partial and contacts carry the match.

Enabled with "country": "AU" (any ISO 3166 region code) or several, ["AU", "NZ"], with a mapped
`country` column choosing per record; custmatch/countries.py holds what each country means.
Each step can be switched off for measurement with "normalise": {"phone": false, ...}. Measured on the sparse Australian file
(research/scripts/au_customer_data.py, research/scripts/au_validate.py; docs/report/05-customer-data.md "Sparse records").

  phone      parsed with Google's libphonenumber (`phonenumbers`) in the country's numbering
             plan and stored as E.164 digits: 0412 345 678, +61 412 345 678, 61412345678 and
             412345678 all become 61412345678. Unparseable values keep their digits.
  dates      read in the country's order (day first everywhere but the US) and stored as ISO,
             so 11/09/2005 and 2005-09-11 agree.
  email      Gmail dots and +tags removed (they reach the same inbox), +tags removed on any
             domain, me.com and mac.com read as icloud.com (one Apple account), and a big provider's name mistyped by one letter corrected (gmial.com,
             hotmail.con; fix_domain) - never a real domain such as ge.com or cloud.com.
  names      what is not a name taken out of the name fields (tidy_name): qualifications after it
             ("smith, phd, cpa", "mba", "®"), bracketed text ("william (bill)"), a phone number
             written into it; a name that is an email address becomes the name it spells
             (name_from_email: nancy.lewis@... -> nancy lewis). "(nee x)" in a surname
             becomes its alias (last_name@nee) where aliases can be used (the default blocking).
  titles     a leading title dropped from the first name: "Dr John" and "Prof. Jane" are john
             and jane (TITLES). Kept when nothing else is left: a record named only "Dr".
  nicknames  `first_roots`: the name and every name it is a diminutive of, from the
             carltonnorthern/nicknames list (US given names, Apache 2.0; custmatch/data). Mick,
             Mike -> michael; Lisa -> elizabeth | melissa. Anglicised names (Wei -> Kevin)
             cannot be listed and are left to the contacts.
  email_user the part of the email before the @ (after normalisation), for blocking and
             features: the same user name on a different or mistyped domain.

  script     names written in another script are transliterated with `anyascii` (ISC licence):
             Chinese characters become pinyin (王伟 -> wang wei, a name in one field split
             family-name-first), Arabic script its consonants (محمد -> mhmd). Arabic script
             usually omits vowels, so when a file has any such names every name also gets a
             consonant skeleton - vowels, w and y (also written for long vowels) and a final h
             dropped, doubled letters collapsed, ch read as sh - used for blocking and as a
             feature: Mohammed and mhmd are both "mhmd". Without non-Latin names nothing changes.

  postcode   the postcode at the end of an address, in the country's format (US ZIP, AU and NZ
             4 digits, UK and Canadian postcodes, written without the space), into `postcode`
             where the source has none: store records can then be compared with web sign-ups
             that give only a postcode, and a customer with no address can join the one
             household with its surname in its postcode (households.py).
  national_id  the country's identifier checked (US: an SSN that is never issued, or a known
             placeholder such as 123-45-6789, is blanked - it is not evidence of anything).

Fake numbers need no step here: core.JUNK_PATTERNS recognises ACMA's fictional ranges once
numbers are in E.164 form, and core.PATTERNED_NUMBERS runs of one digit and sequences like
12345678 - those only when strangers share them, since people buy such numbers.
"""
import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

STEPS = ("script", "phone", "dates", "email", "names", "titles", "nicknames", "local_nicknames",
         "email_user", "postcode", "national_id")
COMMON_DOMAINS = ["gmail.com", "hotmail.com", "outlook.com", "icloud.com", "yahoo.com",
                  "live.com", "msn.com", "aol.com", "me.com", "googlemail.com",
                  "bigpond.com", "bigpond.net.au", "yahoo.com.au", "optusnet.com.au",
                  "live.com.au", "tpg.com.au", "iinet.net.au", "hotmail.co.uk", "yahoo.co.uk",
                  "btinternet.com", "xtra.co.nz", "mail.com", "email.com", "ymail.com",
                  "gmx.com", "rocketmail.com", "protonmail.com", "proton.me"]


def enabled(cfg):
    if not cfg.get("country"):
        return {}
    steps = {s: True for s in STEPS}
    steps.update(cfg.get("normalise", {}))
    return steps


def record_regions(df, cfg):
    """Per record: its country code - the mapped `country` column where filled, else the default."""
    from .countries import code, countries
    default = countries(cfg)[0]
    if "country" in df.columns:
        return df.country.map(lambda v: code(v) or default).to_numpy(dtype=object)
    return np.full(len(df), default, dtype=object)


def _per_region(values, regions, fn):
    """fn(value, region) over (value, region) pairs, each distinct pair computed once."""
    vc, vu = pd.factorize(np.asarray(values, dtype=object), use_na_sentinel=False)
    rc, ru = pd.factorize(np.asarray(regions, dtype=object), use_na_sentinel=False)
    kc, ku = pd.factorize(vc.astype(np.int64) * max(len(ru), 1) + rc)
    done = np.empty(len(ku), dtype=object)
    for n, k in enumerate(ku.tolist()):
        done[n] = fn(vu[k // len(ru)], ru[k % len(ru)])
    return done[kc].tolist()


def normalise(df, cfg):
    """df with contacts, dates and names normalised for cfg['country']; unchanged without it."""
    from .countries import NATIONAL_IDS, countries, profile
    steps = enabled(cfg)
    if not steps:
        return df
    df = df.copy()
    regions = record_regions(df, cfg)
    prof = {r: profile(r, cfg) for r in set(regions)}
    if steps["script"]:
        # "_name_skeleton" is set by `run` and saved with the state, so `add` builds the same
        # feature set whatever script the new batch is in; absent, it follows the file
        df = transliterate(df, cfg.get("_name_skeleton"))
    if steps["phone"]:
        df["phone"] = _per_region(df.phone, regions, phone)
    if steps["dates"]:
        df["date_of_birth"] = _per_region(df.date_of_birth, regions, lambda v, r: iso_date(v, prof[r]["date_order"]))
    if steps["national_id"] and "ssn" in df.columns:
        # a number the country never issues, or a placeholder, is blanked - not evidence
        df["ssn"] = [NATIONAL_IDS[prof[r]["national_id"]](v) if v and prof[r]["national_id"] in NATIONAL_IDS
                     else v for v, r in zip(df.ssn, regions)]
    if steps["email"]:
        df["email"] = _map_unique(df.email, email)
    if steps["names"]:
        before = df[["first_name", "last_name"]].copy()
        nee = df.last_name.str.extract(NEE, expand=False).fillna("").str.strip()
        if (nee != "").any() and cfg.get("blocking") in (None, "", "core") and "last_name@nee" not in df.columns:
            df["last_name@nee"] = nee
        at = df.first_name.str.contains("@", regex=False) | df.last_name.str.contains("@", regex=False)
        if at.any():                                  # an email address as the name: the name it spells
            parts = [name_from_email(f if "@" in f else n) for f, n in zip(df.first_name[at], df.last_name[at])]
            df.loc[at, "first_name"] = [p[0] for p in parts]
            df.loc[at, "last_name"] = [p[1] for p in parts]
        for c in ("first_name", "last_name"):
            df[c] = _map_unique(df[c], tidy_name)
        df.attrs["names_tidied"] = int((before != df[["first_name", "last_name"]]).any(axis=1).sum())
    if steps["titles"]:
        df["first_name"] = _map_unique(df.first_name, drop_title)
    if steps["nicknames"]:
        lists = tuple(sorted({n for c in countries(cfg) for n in profile(c)["nicknames"]
                              if n == "en" or steps["local_nicknames"]}))
        df["first_roots"] = _map_unique(df.first_name, lambda v: "|".join(first_roots(v, lists)))
    if steps["email_user"]:
        df["email_user"] = df.email.str.split("@").str[0].where(df.email.str.contains("@"), "")
    if steps["postcode"] and any(prof[r]["postcode"] for r in prof):
        # "12 high st, richmond vic 3121": the postcode ends an address; web forms give it alone
        found = _per_region(df.addr_street, regions, lambda v, r: _postcode(v, prof[r]["postcode"]))
        given = df.postcode if "postcode" in df.columns else None
        df["postcode"] = found if given is None else [g if g else f for g, f in zip(given, found)]
    return df


def _postcode(address, rule):
    if not rule or not address:
        return ""
    pattern, form = rule
    m = re.search(pattern, address)
    return m.expand(form).replace(" ", "") if m else ""


_LATIN = re.compile(r"^[\x00-\x7f\u00a0-\u024f\u2000-\u206f]*$")   # incl. curly quotes, dashes
_HAN = re.compile(r"[\u3400-\u9fff]")


def transliterate(df, skeleton=None):
    """Non-Latin first / last names to Latin; adds `name_skeleton` (and `name_script`) when any
    were found - or always / never when skeleton is True / False."""
    fn, ln = df.first_name.to_numpy(dtype=object).copy(), df.last_name.to_numpy(dtype=object).copy()
    foreign = np.array([not (_LATIN.match(a) and _LATIN.match(b)) for a, b in zip(fn, ln)], dtype=bool)
    if not foreign.any() and not skeleton:
        return df
    for i in np.flatnonzero(foreign):
        f, l = fn[i], ln[i]
        if not l and _HAN.search(f) and len(f.replace(" ", "")) >= 2:
            chars = f.replace(" ", "")                      # 王伟: family name first, one field
            f, l = chars[1:], chars[0]
        fn[i], ln[i] = _romanise(f), _romanise(l)
    df = df.assign(first_name=fn, last_name=ln)
    if skeleton is False:
        return df
    df["name_script"] = foreign
    df["name_skeleton"] = name_skeletons(fn, ln)
    return df


def name_skeletons(fn, ln):
    return [f"{consonants(a)}|{consonants(b)}" if consonants(a) and consonants(b) else "" for a, b in zip(fn, ln)]


def _romanise(s):
    from anyascii import anyascii
    return re.sub(r"[^a-z\- ]", "", anyascii(s).lower()).strip()


def consonants(name):
    """Consonant skeleton for comparing Arabic-script names with romanised ones."""
    s = re.sub(r"[^a-z]", "", name.lower()).replace("ch", "sh")
    s = re.sub(r"[aeiouwy]", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s[:-1] if len(s) > 1 and s.endswith("h") else s


def _map_unique(series, fn):
    uniq = {v: fn(v) for v in series.unique()}
    return series.map(uniq)


def phone(v, region):
    if not v:
        return v
    import phonenumbers
    for text in (v, "+" + v):
        try:
            n = phonenumbers.parse(text, region)
        except phonenumbers.NumberParseException:
            continue
        if phonenumbers.is_valid_number(n):
            return phonenumbers.format_number(n, phonenumbers.PhoneNumberFormat.E164)[1:]
    return v


_DMY = re.compile(r"^(\d{1,2})[/. -](\d{1,2})[/. -](\d{4})$")
_ISO = re.compile(r"^(\d{4})[/. -](\d{1,2})[/. -](\d{1,2})$")
_COMPACT = re.compile(r"^(1[89]\d\d|20\d\d)(\d\d)(\d\d)$")                     # 19910424
_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december")
_WEEKDAY = re.compile(r"(?<=\d)(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\b")   # "18sat" -> "18"


def _text_date(v):
    """(year, month, day) from a date with its month in words - "09 nov 1953", "november 9, 1953",
    "jul 18sat 1953" - or None. One month word, a four-digit year and a day, in any order."""
    tokens = _WEEKDAY.sub("", v.lower()).replace(",", " ").split()
    months = [i + 1 for t in tokens if len(t) >= 3 for i, name in enumerate(_MONTHS) if name.startswith(t)]
    numbers = [t for t in tokens if t.isdecimal()]
    if len(months) != 1 or len(numbers) != 2 or len(tokens) != 3:
        return None
    years = [t for t in numbers if len(t) == 4]
    days = [t for t in numbers if len(t) <= 2]
    if len(years) != 1 or len(days) != 1:
        return None
    return int(years[0]), months[0], int(days[0])


def iso_date(v, order):
    """A date of birth as YYYY-MM-DD: ISO (or 19910424), day-month-year or month-day-year
    (`order`) with /, ., - or a space between, or with the month in words. Anything else - a
    partial date, a typo - is kept as written."""
    m = _ISO.match(v) or _COMPACT.match(v)
    if m:
        y, mo, d = (int(x) for x in m.groups())
    elif (m := _DMY.match(v)):
        a, b, y = (int(x) for x in m.groups())
        d, mo = (a, b) if order == "dmy" else (b, a)
    elif (t := _text_date(v)):
        y, mo, d = t
    else:
        return v
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return v
    return f"{y:04d}-{mo:02d}-{d:02d}"


def email(v):
    if "@" not in v:
        return v
    local, dom = v.rsplit("@", 1)
    dom = fix_domain(dom)
    local = local.split("+", 1)[0]
    if dom in ("gmail.com", "googlemail.com"):
        local, dom = local.replace(".", ""), "gmail.com"
    elif dom in ("me.com", "mac.com"):     # Apple's three domains are one account: jo@me.com is jo@icloud.com
        dom = "icloud.com"
    return f"{local}@{dom}" if local else v


PROVIDERS = {"gmail", "googlemail", "hotmail", "outlook", "yahoo", "icloud", "bigpond", "optusnet",
             "live", "aol", "btinternet", "xtra", "tpg", "iinet"}


@lru_cache(maxsize=None)
def fix_domain(dom):
    """A mistyped big consumer mail domain, corrected: gmial.com, gmai.com, hotmial.com,
    bigpond.con -> the real one. Deliberately narrow, and the same for every file: only a
    provider name one typo away (a swap counts as one) with the same first letter and at least
    four letters, and the same ending. A frequency rule rewrote real domains (ge.com -> me.com,
    cloud.com -> icloud.com) and differed between `run` and `add`."""
    from rapidfuzz.distance import OSA
    label, _, tld = dom.partition(".")
    if tld in ("con", "cmo", "comm", "vom", "xom"):
        tld = "com"
    elif tld.endswith((".con", ".cmo")):
        tld = tld[:-4] + ".com"
    if label in PROVIDERS or len(label) < 4:
        return f"{label}.{tld}" if tld else dom
    near = [p for p in PROVIDERS if p[0] == label[0] and OSA.distance(label, p) == 1]
    if len(near) == 1 and f"{near[0]}.{tld}" in COMMON_DOMAINS:
        return f"{near[0]}.{tld}"
    return f"{label}.{tld}" if tld else dom


NICKNAME_FILES = {"en": "nicknames.csv", "au_uk": "nicknames_au_uk.csv"}
# third-party data is not shipped: fetched once, pinned and checked, into the data folder
THIRD_PARTY = {"nicknames.csv": (
    "https://raw.githubusercontent.com/carltonnorthern/nicknames/ed160b3/names.csv",      # Apache 2.0
    "e6a4e5a55ca5b47a4146dbc118f26c71c63053d4b79ca9659de8e74efb65847e"),
    "profanity_en.txt": (       # suspect.py; LDNOOBW, CC BY 4.0
    "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
    "5faf2ba42d7b1c0977169ec3611df25a3c08eb13/en",
    "af851ecef1d5f212caba17339b12ac39cc2fef7d78c74876f67237644fcee8bd")}


def data_dir():
    """Where downloaded and built name data lives: $CUSTMATCH_DATA, else ~/.cache/custmatch."""
    import os
    return Path(os.environ.get("CUSTMATCH_DATA") or Path.home() / ".cache" / "custmatch")


def data_file(name):
    """A name-data file: shipped with custmatch (our own lists), already in data_dir(), or - for
    a third-party list - downloaded there on first use from a pinned commit and checked
    against its SHA-256."""
    shipped = Path(__file__).parent / "data" / name
    if shipped.exists():
        return shipped
    local = data_dir() / name
    if local.exists() or name not in THIRD_PARTY:
        return local
    import hashlib
    import urllib.request
    url, sha = THIRD_PARTY[name]
    try:
        import ssl
        try:                                       # python.org builds on macOS lack system certificates
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, timeout=30, context=ctx) as r:
            body = r.read()
    except OSError as e:
        from .io import ConfigError
        raise ConfigError(f"could not download {name} from {url} ({e}); download it to {local} yourself, "
                          'or turn nicknames off: "normalise": {"nicknames": false}') from e
    if hashlib.sha256(body).hexdigest() != sha:
        from .io import ConfigError
        raise ConfigError(f"{url} did not match its expected checksum; not using it")
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.replace(local)
    return local


@lru_cache(maxsize=None)
def _nickname_roots(lst="en"):
    roots = defaultdict(set)
    with open(data_file(NICKNAME_FILES[lst])) as f:
        for row in csv.DictReader(f):
            roots[row["name2"].strip().lower()].add(row["name1"].strip().lower())
    return roots


# titles written into the first-name field. Not "sir", "dame" or "lord" alone: in a first name
# they are as often a given name or a nickname as a title; nor "m" or "sr", which are initials
TITLES = {"mr", "mrs", "ms", "miss", "mx", "dr", "doctor", "prof", "professor", "assoc prof", "a/prof", "rev",
          "revd", "reverend", "fr", "hon", "sra", "herr", "frau", "mme", "mlle"}


# written after a name, never part of it; not jr or sr, which tell a father from his son. The
# short ones are names too (Ma is a Chinese surname), so they are taken only after a comma
POSTNOMINALS = {"phd", "mba", "cpa", "mbbs", "frcs", "fracs", "fracp", "facs", "msc", "bsc", "llb", "esq", "oam",
                "cfa", "cpeng", "faicd", "gaicd", "mcom", "bcom"}
SHORT_POSTNOMINALS = {"ca", "md", "ba", "ma", "jd", "ao", "ac", "am", "jp", "pe", "rn"}
NEE = re.compile(r"\((?:nee|née|born|formerly)\s+([^)]+)\)")
_PHONE_IN_NAME = re.compile(r"\s*[-–]?\s*\+?\d[\d ()\-]{6,}\d")


def name_from_email(address):
    """(first, last) from an email address used as a name: "nancy.lewis@x" -> ("nancy", "lewis"),
    "jsmith@x" -> ("jsmith", ""). Blanking it made nameless records, which few labels linked to
    each other (five such cards, one customer, on a synthetic address book)."""
    parts = [p for p in re.split(r"[._\-]+", address.split("@")[0]) if p.isalpha()]
    if len(parts) == 2 and min(map(len, parts)) >= 2:
        return parts[0], parts[1]
    return re.sub(r"\d+", "", address.split("@")[0]), ""


def tidy_name(name):
    """name without what is not a name: qualifications after it, bracketed text, a phone number
    in it, trade-mark signs; "" for an email address. "smith, phd, cpa" -> "smith",
    "william (bill)" -> "william", "jones - 0412 345 678" -> "jones"."""
    if "@" in name:
        return ""
    v = _PHONE_IN_NAME.sub("", re.sub(r"\s*\([^)]*\)", "", name))
    v = re.sub(r"[®™©]", "", v)
    head, comma, tail = v.partition(",")
    if comma and all(t.strip(".") in POSTNOMINALS | SHORT_POSTNOMINALS for t in re.split(r"[\s,]+", tail) if t):
        v = head
    toks = v.split()
    while len(toks) > 1 and toks[-1].strip(".") in POSTNOMINALS:
        toks.pop()
    return " ".join(toks)


def drop_title(name):
    """name without leading titles ("dr john" -> "john", fused "assoc.prof.dr.jane" -> "jane");
    unchanged if nothing else is left."""
    fused = re.match(r"^(?:(?:assoc|a/prof|prof|dr|mr|mrs|ms|miss|rev)\.)+(?=\w)", name)
    if fused:
        name = fused.group(0).replace(".", " ") + name[fused.end():]
    toks = name.split(" ")
    k = 0
    while k < len(toks) - 1:
        two = " ".join(toks[k:k + 2]).replace(".", "")
        if two in TITLES and k + 2 < len(toks):
            k += 2
        elif toks[k].rstrip(".") in TITLES:
            k += 1
        else:
            break
    return " ".join(toks[k:]) if k else name


def first_roots(name, lists=("en",)):
    """The name's first token and every full name it is a diminutive of, sorted. Two first names
    are compatible when their roots meet: Joe and Jos both reach joseph. (A cross-language list
    from Wikidata - Giuseppe to joseph - was tried and removed: no measured gain.)"""
    tok = name.split(" ")[0] if name else ""
    if not tok:
        return []
    roots = {tok}
    from anyascii import anyascii
    plain = anyascii(tok).lower()                  # José, Zoë: look up as jose, zoe as well
    for t in {tok, plain}:
        for lst in lists:
            roots |= _nickname_roots(lst).get(t, set())
    return sorted(roots | {plain})


def _junk_email(df, profiles):
    """Per record: its email was profiled as junk (noreply@..., test@test.com)."""
    if not profiles or "email" not in profiles:
        return np.zeros(len(df), dtype=bool)
    return np.array([profiles["email"].get(v, ("personal",))[0] == "junk" for v in df.email.to_numpy()])


def features(df, pairs, profiles=None):
    """Two columns when the country steps ran: first names share a root (1/0, -1 missing), and
    email user names agree (1/0, -1 missing) - the same person on another or mistyped domain."""
    out = []
    from .core import _cached, _coded_column, _codes, _object_column, per_value_pairs
    if "first_roots" in df.columns:
        c, r = _coded_column(df, "first_roots")
        sets = _cached(df, "first_root_sets", lambda: [frozenset(v.split("|")) for v in r])

        def roots(x, y):
            return [0.0 if sets[a].isdisjoint(sets[b]) else 1.0 for a, b in zip(x.tolist(), y.tolist())]
        a, b = c[pairs[:, 0]], c[pairs[:, 1]]
        filled = (a >= 0) & (b >= 0)
        shared = np.full(len(pairs), -1.0)        # once per distinct pair of root lists
        shared[filled] = per_value_pairs(a[filled], b[filled], roots)
        out.append(shared)
    if "email_user" in df.columns:
        # junk is not evidence; kept with the profiles it was read from
        held = _cached(df, "email_user_codes", lambda: [None, None])
        if held[0] is not profiles:
            held[:] = [profiles, _codes(np.where(_junk_email(df, profiles), "", _object_column(df, "email_user")))]
        u = held[1]
        a, b = u[pairs[:, 0]], u[pairs[:, 1]]
        out.append(np.where((a >= 0) & (b >= 0), (a == b).astype(float), -1.0))
    if "name_skeleton" in df.columns:          # only when the file has non-Latin names
        k, f = df.name_skeleton.to_numpy(), df.name_script.to_numpy()
        a, b = k[pairs[:, 0]], k[pairs[:, 1]]
        across = f[pairs[:, 0]] | f[pairs[:, 1]]
        out.append(np.where(across & (a != "") & (b != ""), (a == b).astype(float), -1.0))
    if not out:
        return np.zeros((len(pairs), 0), np.float32)
    return np.column_stack(out).astype(np.float32)


def blocking_keys(df, profiles=None):
    """(key, record index) for sparse records: first-name root + date of birth, first-name root
    + surname (Bill and William Jones), and the email user name when it is long enough to be
    distinctive."""
    keys = []
    if "first_roots" in df.columns:
        split = {}                                 # each distinct list of roots split once
        for i, (roots, dob, ln) in enumerate(zip(df.first_roots.to_numpy(dtype=object),
                                                 df.date_of_birth.to_numpy(dtype=object),
                                                 df.last_name.to_numpy(dtype=object))):
            if roots not in split:
                split[roots] = roots.split("|") if roots else []
            for r in split[roots]:
                if dob:
                    keys.append((("fd", r, dob), i))
                if ln:
                    keys.append((("fl", r, ln), i))
    if "email_user" in df.columns:
        junk = _junk_email(df, profiles)
        for i, u in enumerate(df.email_user.to_numpy(dtype=object)):
            if len(u) >= 5 and not junk[i]:
                keys.append((("eu", u), i))
    if "name_skeleton" in df.columns:
        for i, k in enumerate(df.name_skeleton.to_numpy(dtype=object)):
            if len(k) >= 5:
                keys.append((("sk", k), i))
    return keys


skeleton = consonants      # the name the tests and docs use


def refresh_names(df):
    """Recompute the name-derived columns after names change (placeholders.blank)."""
    df = df.copy()
    if "first_roots" in df.columns:          # only blanked names change; the rest keep their lists
        df["first_roots"] = df.first_roots.where(df.first_name != "", "")
    if "name_skeleton" in df.columns:
        df["name_skeleton"] = name_skeletons(df.first_name.to_numpy(), df.last_name.to_numpy())
    return df
