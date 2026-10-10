"""Country profiles: what `"country"` switches on for each country.

    "country": "US"                  one country
    "country": ["AU", "NZ"]          several: the first is the default, and a mapped `country`
                                     column (codes or names) picks the rules record by record

A profile says how that country writes things, and custmatch/country.py applies it:

  phone        numbering plan for `phonenumbers` (every ISO region works, not only these)
  date_order   how a slashed date reads: "mdy" (US) or "dmy"; "date_order" in the config wins
  postcode     where the postcode sits at the end of an address, and its canonical form
  national_id  the identifier column that means something there, cleaned and checked: a US SSN
               that the Social Security Administration never issues (area 000 or 666, group 00,
               serial 0000) or a well-known placeholder (123-45-6789, 078-05-1120, one digit
               repeated) is blanked - it is not evidence. Other countries get none: Australia's
               TFN and similar numbers may not lawfully be used to match customers
  nicknames    lists joined in first-name roots: "en" (carltonnorthern/nicknames, US given
               names) everywhere; "au_uk" (custmatch/data/nicknames_au_uk.csv: Bec, Shazza,
               Davo, Con) for Australia, New Zealand and the UK
"""
import re

PROFILES = {
    "US": {"name": "United States", "date_order": "mdy",
           "postcode": (r"(?:^|\D)(\d{5})(?:-\d{4})?\s*$", r"\1"), "national_id": "ssn",
           "nicknames": ("en",)},
    "CA": {"name": "Canada", "date_order": "dmy",     # both orders are used: set "date_order"
           "postcode": (r"(?:^|[^a-z0-9])([a-z]\d[a-z])\s?(\d[a-z]\d)\s*$", r"\1\2"),
           "national_id": None, "nicknames": ("en",)},
    "GB": {"name": "United Kingdom", "date_order": "dmy",
           "postcode": (r"(?:^|[^a-z0-9])([a-z]{1,2}\d[a-z\d]?)\s*(\d[a-z]{2})\s*$", r"\1\2"),
           "national_id": None, "nicknames": ("en", "au_uk")},
    "AU": {"name": "Australia", "date_order": "dmy",
           "postcode": (r"(?:^|\D)(\d{4})\s*$", r"\1"), "national_id": None,
           "nicknames": ("en", "au_uk")},
    "NZ": {"name": "New Zealand", "date_order": "dmy",
           "postcode": (r"(?:^|\D)(\d{4})\s*$", r"\1"), "national_id": None,
           "nicknames": ("en", "au_uk")},
}
ALIASES = {"UK": "GB", "USA": "US", "UNITED STATES": "US", "AMERICA": "US", "AUSTRALIA": "AU",
           "NEW ZEALAND": "NZ", "UNITED KINGDOM": "GB", "GREAT BRITAIN": "GB", "BRITAIN": "GB",
           "ENGLAND": "GB", "SCOTLAND": "GB", "WALES": "GB", "CANADA": "CA"}
DEFAULT = {"name": "", "date_order": "dmy", "postcode": None, "national_id": None, "nicknames": ("en",)}


def code(value):
    """ISO region code for a code or country name; "" if blank."""
    v = str(value or "").strip().upper()
    return ALIASES.get(v, v)


def countries(cfg):
    """The configured countries, default first: "AU" -> ["AU"]; ["au", "nz"] -> ["AU", "NZ"]."""
    c = cfg.get("country")
    if not c:
        return []
    return [code(x) for x in ([c] if isinstance(c, str) else c)]


def profile(region, cfg=None):
    p = dict(PROFILES.get(code(region), DEFAULT))
    if cfg and cfg.get("date_order"):
        p["date_order"] = cfg["date_order"]
    return p


# ------------------------------------------------------------------ national identifiers
_SSN_PLACEHOLDERS = {"123456789", "078051120", "219099999", "987654320", "987654321"}


def ssn(value):
    """A US SSN as 9 digits, or "" when it cannot be a real one."""
    d = re.sub(r"\D", "", str(value))
    if len(d) != 9 or d in _SSN_PLACEHOLDERS or len(set(d)) == 1:
        return ""
    area, group, serial = d[:3], d[3:5], d[5:]
    if area in ("000", "666") or group == "00" or serial == "0000":
        return ""
    return d                                   # 9xx areas are ITINs - real taxpayer numbers


NATIONAL_IDS = {"ssn": ssn}
