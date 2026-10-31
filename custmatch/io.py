"""Load customer records from CSV/Parquet sources into the pipeline's canonical columns.

A config file names each source and maps its columns onto the canonical ones:

    {
      "sources": [
        {"name": "crm", "path": "crm.csv",
         "columns": {"record_id": "id", "first_name": "fname", "last_name": "lname",
                     "date_of_birth": "dob", "address": "street_address",
                     "email": "email", "phone": "phone"}},
        {"name": "billing", "path": "billing.parquet", "columns": {...}}
      ],
      "source_priority": ["crm", "billing"]
    }

`record_id` and `first_name` are required; every other column, `last_name` included, is optional
and treated as missing when absent. An address can be given as parts (`addr_number`,
`addr_street`, `addr_unit`) or as one `address` string, which is split into a leading house
number and the rest. Paths are relative to the config file.
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

CANONICAL = ["record_id", "first_name", "middle_initial", "last_name", "date_of_birth",
             "addr_number", "addr_street", "addr_unit", "email", "phone", "ssn", "country", "updated_at"]
REQUIRED = ["record_id", "first_name"]    # a source with no surname column (web sign-ups) is fine
_UNIT = re.compile(r"(?:\b(?:apt|unit|ste|suite|flt|flat)\b|#)\s*\S+$")          # "... apt 3", "... #4"
_LEADING_UNIT = re.compile(r"^(?:(?:unit|apt|flat|suite|shop|u)\s*|#)(\w+),?\s+(\d\S*)\s+(.+)$")  # "unit 4, 12 main st"
_SLASH_UNIT = re.compile(r"^(\w+)/(\d\S*)\s+(.+)$")                                # "3/12 main st" (unit 3)


class ConfigError(ValueError):
    pass


def split_address(s):
    """'5112 145th st apt 3' -> ('5112', '145th st', 'apt 3'); a unit may also lead, as Australian
    addresses write it: 'unit 4, 12 main st', 'u4 12 main st' and '4/12 main st' -> ('12', 'main st', '4')."""
    s = " ".join(str(s).lower().split())
    for pat in (_LEADING_UNIT, _SLASH_UNIT):
        m = pat.match(s)
        if m:
            return m.group(2), m.group(3), m.group(1)
    unit = ""
    m = _UNIT.search(s)
    if m:
        unit, s = m.group(0), s[: m.start()].strip()
    tokens = s.split(" ", 1)
    if tokens and tokens[0][:1].isdigit():
        return tokens[0], (tokens[1] if len(tokens) > 1 else ""), unit
    return "", s, unit


def _read(path, encoding=None):
    """A source file as strings. CSVs are UTF-8 (a byte-order mark is fine) unless the source sets
    "encoding" - "latin-1" or "cp1252" for files exported from older systems and Excel."""
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding or "utf-8")
    except UnicodeDecodeError as e:
        raise ConfigError(f"{path.name} is not {encoding or 'UTF-8'} text ({e.reason} at byte {e.start}); "
                          'set "encoding" on its source, e.g. "latin-1" or "cp1252"') from None
    except LookupError:
        raise ConfigError(f'unknown "encoding" {encoding!r} for {path.name}') from None


_TYPOGRAPHIC = str.maketrans({c: "'" for c in "\u2018\u2019\u201b\u02bc\u2032\u00b4"} |
                             {c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})


def unicode_form(s):
    """One spelling per text: Unicode NFKC, with invisible format characters (category Cf: zero-width
    space and joiners, soft hyphen, byte-order mark, direction marks) removed. A name typed on a Mac
    or exported by some systems arrives decomposed - "Müller" as u + combining diaeresis - and
    differed from the same name typed elsewhere in every exact key and agreement feature; so did a
    name with a zero-width space pasted in from a web page, and full-width letters (ＪＯＳＥ).
    Typographic apostrophes and hyphens become ASCII ones (O’Brien, Smith‐Jones). Lower-cased.
    Accents are kept: José and Jose stay different strings, compared by similarity."""
    if s.isascii():
        return s.lower()
    # format characters out first: one between a letter and its accent would otherwise keep
    # them from composing, and a second cleaning would compose them. Lower-casing and NFKC each
    # can undo the other (İ lower-cases to i + a combining dot; NFKC turns ℌ into H), so both
    # are repeated until the text stops changing - cleaning twice gives what cleaning once gave
    s = "".join(c for c in s if unicodedata.category(c) != "Cf")
    for _ in range(4):
        t = unicodedata.normalize("NFKC", s.lower()).translate(_TYPOGRAPHIC)
        if t == s:
            break
        s = t
    return s


def _clean(series):
    """Strings, lower-cased and trimmed, "" for missing, in one Unicode form (unicode_form). Typed
    Parquet columns are turned into the text a CSV would have held: a date column as YYYY-MM-DD, a
    whole-number float column (a numeric phone column with gaps is float64) without its ".0" -
    which otherwise became a trailing digit."""
    if pd.api.types.is_datetime64_any_dtype(series):
        series = series.dt.strftime("%Y-%m-%d")
    elif pd.api.types.is_float_dtype(series):
        whole = series.dropna()
        if len(whole) and (whole == whole.round()).all():
            series = series.astype("Int64")
    out = pd.Series(series.astype(object).where(series.notna(), ""), index=series.index).astype(str)
    wide = out.str.contains(r"[^\x00-\x7f]", regex=True)
    if wide.any():                                 # only non-ASCII values can change: each distinct one once
        out = out.copy()
        out[wide] = out[wide].map({v: unicode_form(v) for v in out[wide].unique()})
    return out.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)   # ASCII values lower-cased here


def load(config_path):
    """Canonical DataFrame: one row per record, `source` + canonical columns, all strings."""
    config_path = Path(config_path)
    try:
        cfg = json.loads(config_path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"{config_path.name} is not valid JSON: {e.msg} (line {e.lineno}, column {e.colno})") from None
    return load_config(cfg, config_path.parent)


# every top-level config key custmatch reads; anything else is most likely a typo
KNOWN_KEYS = {
    "sources", "country", "date_order", "normalise", "extra_fields", "extra_blocking", "max_block", "blocking", "cache_candidates", "fold_trees",
    "blocking_recall", "lsh_blocking", "check_large_blocks", "rich_features", "record_dates", "matcher_model",
    "match_weight_feature", "max_cluster", "min_precision", "review_band", "overrides", "trusted_ids", "trusted_max", "twin_guard",
    "unique_sources", "stability_margin", "identity_events",
    "cookie_max_identities", "cookie_max_records", "cookie_name_check", "household_postcode_join", "placeholders", "suspect_records",
    "survivorship", "golden_fields", "source_priority", "contact_confidence", "hash_fields", "hash_salt_env",
    "minimal_outputs", "_name_skeleton", "_blocking_switched", "threshold_rule", "aliases"}


# options removed in 0.1.0 because measurement showed they did not earn their place
REMOVED_KEYS = {"clustering": "correlation clustering", "cc_max_size": "correlation clustering",
                "cc_missing": "correlation clustering", "ambiguous_margin": "the ambiguity rule",
                "ambiguous_different_below": "the ambiguity rule", "missing_as_unknown": "missing_as_unknown",
                "ditto_model": "the Ditto stage",
                "splink_feature": 'the Splink feature (replaced by "match_weight_feature")',
                "auto_models": '"matcher_model": "auto"', "tabpfn_estimators": "TabPFN"}


def unknown_keys(cfg):
    """Config keys custmatch does not read, each with the nearest known one: "twin_gaurd" ->
    twin_guard. Keys starting with "_" or "#" (comments) are ignored."""
    import difflib
    out = []
    for k in cfg:
        if k in KNOWN_KEYS or k.startswith(("_", "#")):
            continue
        if k in REMOVED_KEYS:
            out.append(f'"{k}" ({REMOVED_KEYS[k]} was removed in 0.1.0; see CHANGELOG.md)')
            continue
        near = difflib.get_close_matches(k, KNOWN_KEYS, n=1)
        out.append(f'"{k}"' + (f' (did you mean "{near[0]}"?)' if near else ""))
    # a mapped column custmatch neither knows nor was told to compare is read and then ignored
    used = set(CANONICAL) | {"address"} | {f.get("name") for f in cfg.get("extra_fields", [])} | \
        set(cfg.get("golden_fields", [])) | set(cfg.get("trusted_ids", [])) | set(cfg.get("hash_fields", [])) | \
        {a for names in (cfg.get("aliases") or {}).values() if isinstance(names, list) for a in names}
    for src in cfg.get("sources", []):
        for c in src.get("columns", {}):
            if c not in used:
                near = difflib.get_close_matches(c, sorted(used), n=1)
                out.append(f'"{c}" in source \'{src.get("name")}\' columns' +
                           (f' (did you mean "{near[0]}"?' if near else " (") +
                           ' or add it to "extra_fields" to compare it)')
    return out


def load_config(cfg, base_dir, data=None, whole_config=True):
    """load() for a config already in memory; source paths are relative to base_dir.
    data: {source name: DataFrame} read in place of that source's file - records that arrive as
    data (the service's real-time API) are cleaned exactly as a file's would be.
    whole_config=False: cfg holds only some of the sources (one record of one source, say), so
    checks that span all sources - every extra field mapped somewhere - are skipped."""
    config_path = Path(base_dir) / "config.json"      # only its parent is used, for relative paths
    if not cfg.get("sources"):
        raise ConfigError("config has no 'sources'")
    if "survivorship" in cfg:
        from .golden import check
        fields = set(CANONICAL) | set(cfg.get("golden_fields", [])) | {e.get("name") for e in cfg.get("extra_fields", [])}
        problems = check(cfg["survivorship"], fields)
        if problems:
            raise ConfigError("; ".join(problems))
    aliases = check_aliases(cfg, whole_config)
    frames = []
    for src in cfg["sources"]:
        name, cols = src.get("name"), src.get("columns", {})
        if not name or ("path" not in src and not (data and name in data)):
            raise ConfigError(f"each source needs a 'name' and a 'path': {src}")
        missing = [c for c in REQUIRED if c not in cols]
        if missing:
            raise ConfigError(f"source '{name}' does not map required column(s) {missing}")
        raw = data[name].astype(str) if data and name in data else \
            _read((config_path.parent / src["path"]).resolve(), src.get("encoding"))
        absent = [v for v in cols.values() if v not in raw.columns]
        if absent:
            raise ConfigError(f"source '{name}': columns {absent} not in {src['path']}")
        out = pd.DataFrame({"source": name, "record_id": raw[cols["record_id"]].astype(str)})
        for c in CANONICAL[1:]:
            out[c] = _clean(raw[cols[c]]) if c in cols else ""
        if "address" in cols:
            parts = _clean(raw[cols["address"]]).map(split_address)
            for i, c in enumerate(("addr_number", "addr_street", "addr_unit")):
                if c not in cols:
                    out[c] = parts.str[i]
        for f in cfg.get("extra_fields", []):
            if f["name"] not in CANONICAL:   # a standard column named as an extra is compared twice
                out[f["name"]] = _clean(raw[cols[f["name"]]]) if f["name"] in cols else ""
        # columns used by name but not compared: trusted identifiers and extra golden-record fields
        # (they were dropped here, so "trusted_ids" silently did nothing unless also an extra field)
        for name in [*cfg.get("trusted_ids", []), *cfg.get("golden_fields", [])]:
            if name not in out.columns:
                out[name] = _clean(raw[cols[name]]) if name in cols else ""
        for field, name in aliases:            # other values of a field: "last_name@maiden_name"
            out[f"{field}@{name}"] = _clean(raw[cols[name]]) if name in cols else ""
            if field == "phone":
                out[f"{field}@{name}"] = out[f"{field}@{name}"].str.replace(r"\D", "", regex=True)
        out["phone"] = out.phone.str.replace(r"\D", "", regex=True)
        out["middle_initial"] = out.middle_initial.str[:1]
        if src.get("group_rows"):
            out, more = group_rows(out)
            aliases = aliases + [a for a in more if a not in aliases]
        frames.append(out)
    df = pd.concat(frames, ignore_index=True)
    alias_cols = [c for c in df.columns if "@" in c]
    df[alias_cols] = df[alias_cols].fillna("")       # a source without an alias column has no alias
    if not len(df):
        raise ConfigError("no records: every source file is empty")
    mapped = {c for src in cfg["sources"] for c in src.get("columns", {})}
    unmapped = [f["name"] for f in cfg.get("extra_fields", []) if f["name"] not in mapped | set(CANONICAL)]
    if unmapped and whole_config:
        raise ConfigError(f"extra field(s) {unmapped} are not mapped in any source's \"columns\"")
    from .countries import code as country_code
    import phonenumbers
    bad = [c for c in ([cfg["country"]] if isinstance(cfg.get("country"), str) else cfg.get("country") or [])
           if country_code(c) not in phonenumbers.SUPPORTED_REGIONS]
    if bad:
        raise ConfigError(f"unknown country {bad}: use an ISO code such as AU, GB, NZ, US or CA")
    if (df.updated_at != "").any():
        # the record's date, for "most_recent" survivorship: one sortable form (ISO)
        t = pd.to_datetime(df.updated_at.where(df.updated_at != ""), errors="coerce", format="mixed",
                           dayfirst=cfg.get("date_order", "dmy") == "dmy" and bool(cfg.get("country"))
                           and "US" not in ([cfg["country"]] if isinstance(cfg["country"], str) else cfg["country"]))
        df["updated_at"] = t.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    from . import country
    df = _normalise_aliases(df, country.normalise(df, cfg), aliases, cfg)
    if cfg.get("hash_fields") and cfg.get("suspect_records", True) and whole_config:
        from . import suspect                     # in plain text: once hashed a word is not a word
        df.attrs["suspect"] = suspect.flag(df, suspect.profanity_words())
    if cfg.get("hash_fields"):
        if cfg.get("placeholders"):               # found in plain text: once hashed, "test test" is not
            from .placeholders import blank
            df = blank(df)[0]
        df = hash_fields(df, cfg)
    known = set(df.columns)
    for key in cfg.get("extra_blocking", []):
        from .blocking import _TRANSFORMS
        unknown = [c for c in key if c.split(":")[0] not in known]
        if unknown:
            raise ConfigError(f"extra_blocking {key} uses unknown field(s) {unknown}")
        bad = [c for c in key if ":" in c and c.split(":", 1)[1] not in _TRANSFORMS]
        if bad:
            raise ConfigError(f"extra_blocking {key}: unknown transform in {bad}; use one of {sorted(_TRANSFORMS)}")
    blank = df.record_id.str.strip() == ""
    if blank.any():
        raise ConfigError(f"{int(blank.sum())} record(s) without a record_id, e.g. in source "
                          f"'{df[blank].iloc[0].source}'")
    dup = df.duplicated(["source", "record_id"])
    if dup.any():
        raise ConfigError(f"{int(dup.sum())} duplicate record_id(s) within a source, e.g. "
                          f"{df[dup].iloc[0][['source', 'record_id']].tolist()}")
    return df, cfg


def group_rows(out):
    """A long-format source - one row per contact point, rows sharing a record_id one record -
    as one row per record: (records, aliases), from the source's cleaned rows. Each field takes the
    record's first filled value; its other surnames, emails and phones (core.ALIAS_FIELDS) become
    aliases, columns "email@email_2", "email@email_3", ... - compared and blocked as "aliases"
    are. Other fields keep the first value only: a second first name or address is dropped."""
    from .core import ALIAS_FIELDS
    codes, ids = pd.factorize(out.record_id)
    grouped = pd.DataFrame({"source": out.source.iloc[:len(ids)].to_numpy(), "record_id": ids})
    aliases = []
    for col in out.columns.drop(["source", "record_id"]):
        v = out[col].to_numpy(dtype=object)
        keep = v != ""
        vals = pd.DataFrame({"g": codes[keep], "v": v[keep]}).drop_duplicates()
        vals["k"] = vals.groupby("g").cumcount()
        wide = vals.pivot(index="g", columns="k", values="v").reindex(range(len(ids))).fillna("")
        grouped[col] = wide[0].to_numpy() if 0 in wide.columns else ""
        if col in ALIAS_FIELDS:
            for k in wide.columns[1:]:
                name = f"{col}_{k + 1}"
                while f"{col}@{name}" in out.columns:
                    name = "_" + name
                grouped[f"{col}@{name}"] = wide[k].to_numpy()
                aliases.append((col, name))
    return grouped, aliases


def check_aliases(cfg, whole_config=True):
    """[(field, column name)] from "aliases": {"last_name": ["maiden_name"], "email": ["email_2"]}.
    Each name is a column mapped in a source's "columns" (a source without it has no alias)."""
    from .core import ALIAS_FIELDS
    spec = cfg.get("aliases") or {}
    if not isinstance(spec, dict):
        raise ConfigError('"aliases" maps a field to a list of column names, e.g. {"last_name": ["maiden_name"]}')
    out = []
    for field, names in spec.items():
        if field not in ALIAS_FIELDS:
            raise ConfigError(f'"aliases": {field!r} cannot have aliases; use one of {list(ALIAS_FIELDS)}')
        if not isinstance(names, list) or not all(isinstance(n, str) and n for n in names):
            raise ConfigError(f'"aliases": {field!r} needs a list of column names')
        clash = [n for n in names if n in CANONICAL or "@" in n]
        if clash:
            raise ConfigError(f'"aliases": {clash} name standard columns; give each alias its own name')
        out += [(field, n) for n in names]
    if out and whole_config:
        mapped = {c for src in cfg.get("sources", []) for c in src.get("columns", {})}
        if not any(n in mapped for _, n in out):
            raise ConfigError(f'"aliases": none of {[n for _, n in out]} is mapped in a source\'s "columns"')
    return out


def _normalise_aliases(raw, df, aliases, cfg):
    """Alias columns normalised as their field is (country.normalise): a second phone to E.164,
    a second email's dots and +tags removed, a maiden name transliterated. raw: the records before
    normalisation; df: after it."""
    from . import country
    if not aliases or not country.enabled(cfg):
        return df
    df = df.copy()
    for field, name in aliases:
        col = f"{field}@{name}"
        rows = (raw[col] != "").to_numpy()
        if rows.any():
            tmp = raw.loc[rows].assign(**{field: raw.loc[rows, col]})
            df.loc[rows, col] = country.normalise(tmp, cfg)[field].to_numpy()
    return df


def key(df):
    """'source:record_id' per row - how labelled pairs and outputs name records."""
    return (df.source + ":" + df.record_id).to_numpy()


def hash_fields(df, cfg):
    """Keyed hashes (HMAC-SHA256) of the listed fields, after normalisation, so they are only
    compared for equality and never stored in plain text. The key comes from the environment
    variable named by "hash_salt_env" (default CUSTMATCH_SALT), never from the config file: the
    same key must be used for every run whose results are compared. Values recognisably fake
    (test@test.com, 0400000000) are blanked first - once hashed they could no longer be told apart."""
    import hashlib
    import hmac
    import os
    from .core import syntactic_junk
    env = cfg.get("hash_salt_env", "CUSTMATCH_SALT")
    salt = os.environ.get(env, "")
    if not salt:
        raise ConfigError(f'"hash_fields" needs a secret key in the environment variable {env}')
    key = salt.encode()
    fields = list(cfg["hash_fields"])
    if "email" in fields and "email_user" in df.columns:
        fields.append("email_user")
    df = df.copy()
    for f in fields:
        if f not in df.columns:
            raise ConfigError(f'"hash_fields": {f} is not a column of the records')
    # alias columns are hashed as their field is, with its key: a maiden name still meets a surname
    columns = [(f, f) for f in fields] + [(c, c.split("@", 1)[0]) for c in df.columns
                                          if "@" in c and c.split("@", 1)[0] in fields]
    for c, f in columns:
        uniq = {v: ("" if not v or (f in ("email", "phone") and syntactic_junk(v))
                    else hmac.new(key, f"{f}:{v}".encode(), hashlib.sha256).hexdigest()[:32])
                for v in df[c].unique()}
        df[c] = df[c].map(uniq)
    if ({"first_name", "last_name"} & set(fields)) and "name_skeleton" in df.columns:
        # the consonant skeleton of a transliterated name is the name: hashed whole, compared equal
        df["name_skeleton"] = df.name_skeleton.map(
            lambda v: hmac.new(key, f"name_skeleton:{v}".encode(), hashlib.sha256).hexdigest()[:32] if v else "")
    if "first_name" in fields and "first_roots" in df.columns:
        # the nickname roots would otherwise keep the name in plain text: each root is hashed
        # on its own, so two names still meet when a root hash is shared
        h = {}

        def one(tok):
            if tok not in h:
                h[tok] = hmac.new(key, f"first_name:{tok}".encode(), hashlib.sha256).hexdigest()[:32]
            return h[tok]
        df["first_roots"] = df.first_roots.map(lambda v: "|".join(one(t) for t in v.split("|")) if v else "")
    return df
