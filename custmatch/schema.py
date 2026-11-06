"""Optional extra fields from the config, for data that is not shaped like a customer file.

    "extra_fields":   [{"name": "birth_place", "compare": "exact"},
                       {"name": "postcode", "compare": "similarity"}],
    "extra_blocking": [["last_name", "date_of_birth"], ["postcode"]],
    "max_block": 200

`compare` is exact, similarity (Jaro-Winkler), edit (normalised Levenshtein; good for dates) or
years_apart (how far apart two dates are, in years: lets evidence age).
"record_dates": true adds {"name": "updated_at", "compare": "years_apart"} - the record's own
date, so the matcher can learn that a shared mobile years apart may be a recycled number, and
that a different address or surname years later may be a move or a marriage.
An extra field may name a standard column to add a fuzzy comparison of it.

Each extra field must also be mapped in a source's `columns`. Extra blocking keys may use
standard or extra field names. With neither present the pipeline is exactly the measured one.
"""
import pandas as pd

COMPARES = ("exact", "similarity", "edit", "years_apart")


def extra_fields(cfg):
    out = []
    for f in cfg.get("extra_fields", []):
        compare = f.get("compare", "exact")
        if compare not in COMPARES:
            from .io import ConfigError
            raise ConfigError(f"extra field {f.get('name')}: compare must be one of {COMPARES}")
        out.append((f["name"], compare))
    if cfg.get("record_dates") and ("updated_at", "years_apart") not in out:
        out.append(("updated_at", "years_apart"))
    return tuple(out)


def max_block(cfg):
    """Blocks larger than this are skipped as hubs (default 60). Raise it when a large block is
    a legitimately common value - a city - rather than a junk contact."""
    return cfg.get("max_block")


DOB_KEY_MAX_RECORDS = 500_000


def with_dob_key(df, cfg):
    """cfg with "dob_key" decided, so the run saves it and `add` blocks the same way: unset, it is
    on for a file of up to DOB_KEY_MAX_RECORDS records with dates of birth. Date of birth alone as
    a blocking key finds the records whose two names are both misspelt: on fake_1000 the default
    keys compared 66% of the true pairs and 81% with it, and clustered F1 rose from 87.11 to 93.72
    (research/scripts/home_data_benchmark.py). On a million records its blocks hold about 30
    people each: 4.6 times the candidate pairs (research/scripts/custmatch_scale.py's file).
    A state saved without the key keeps it off."""
    if "dob_key" in cfg:
        return cfg
    on = len(df) <= DOB_KEY_MAX_RECORDS and bool((df.date_of_birth != "").any())
    return cfg | {"dob_key": on}


DATE_PARTS_MIN_SHARE = 0.01


def with_date_parts(df, cfg):
    """cfg with "date_parts" decided, so the run saves it and `add` builds the same features:
    unset, it is on when at least DATE_PARTS_MIN_SHARE of the dates of birth are partial ("1985 02",
    "14 march"; core.date_parts_feature). Where every date is whole it only repeats "date of birth
    agrees": forced on, it cost the sparse Australian file 0.40 clustered F1, and no other dataset
    here has a partial date. A state saved without the key keeps it off, as its model was trained
    without the feature."""
    if "date_parts" in cfg:
        return cfg
    from .core import date_parts
    dated = df.date_of_birth.to_numpy()[df.date_of_birth.to_numpy() != ""]
    known = date_parts(dated) >= 0
    partial = known.any(1) & ~known.all(1)
    return cfg | {"date_parts": bool(len(dated)) and float(partial.mean()) >= DATE_PARTS_MIN_SHARE}


def either_order_share(df):
    """The share of the records with both names whose surname is commoner in the file as a first
    name than as a surname, or whose first name is commoner as a surname: how far the names come in
    either order. A file written surname-first throughout scores low, as a file written first-name
    first does: only the order varying within the file counts, which is what the name-words
    feature (core.name_words_feature) is for. BPID 72.0%; every other dataset here with names in
    one order at most 8.8% (NCVR's real names 8.0%), but fake_1000's 1,000 noisy names 15.6%;
    with a tenth of a file's names swapped, 15-16% (research/scripts/name_order.py)."""
    f, l = df.first_name.to_numpy(), df.last_name.to_numpy()
    both = (f != "") & (l != "")
    if not both.any():
        return 0.0
    as_first = pd.Series(f[f != ""]).value_counts()
    as_last = pd.Series(l[l != ""]).value_counts()
    count = lambda values, by: pd.Series(values).map(by).fillna(0).to_numpy()  # noqa: E731
    f, l = f[both], l[both]
    odd = (count(l, as_first) > count(l, as_last)) | (count(f, as_last) > count(f, as_first))
    return float(odd.mean())


NAME_WORDS_MIN_SHARE = 0.12


def with_name_words(df, cfg):
    """cfg with "name_words" decided, so the run saves it and `add` builds the same features:
    unset, it is on when either_order_share(df) is at least NAME_WORDS_MIN_SHARE. The name-words
    feature compares the worse-met part of a name with either part of the other, in any order: on
    BPID (72% either order) it added 0.8 points of pairwise F1, all precision, and with a tenth of
    a file's names swapped (15-16%) 1.02 on the synthetic customers and 2.18 on historical_50k of
    clustered F1; where names are in one order (under 9%) it costs up to 1.22, as the swapped-name
    feature already covers an occasional swap. 12% sits between (research/scripts/name_order.py).
    Off without the rich features, and where names are hashed (a hash is compared for equality
    only). A state saved without the key keeps it off, as its model was trained without the feature."""
    if "name_words" in cfg:
        return cfg
    hashed = "first_name" in cfg.get("hash_fields", []) or "last_name" in cfg.get("hash_fields", [])
    on = bool(cfg.get("rich_features", True)) and not hashed and either_order_share(df) >= NAME_WORDS_MIN_SHARE
    return cfg | {"name_words": on}


def extra_keys(cfg):
    """The config's extra_blocking keys, and date of birth alone when "dob_key" is on."""
    keys = tuple(tuple(k) for k in cfg.get("extra_blocking", []))
    if cfg.get("dob_key") and ("date_of_birth",) not in keys:
        keys += (("date_of_birth",),)
    return keys


def feature_options(cfg):
    """Keyword arguments for core.features from the config: extra fields and the nine rarity /
    name-swap / graded-date features - on by default ("rich_features": false turns them off).
    They lifted historical_50k from 82.1 to 87.0 and fake_1000 from 88.2 to 92.5 clustered F1,
    and cost at most 0.3 on the febrl sets (research/scripts/beat_splink.py)."""
    return {"extra_fields": extra_fields(cfg), "rich": bool(cfg.get("rich_features", True)),
            "name_words": bool(cfg.get("name_words", False)), "date_parts": bool(cfg.get("date_parts", False))}


def feature_options_for(df, cfg):
    """feature_options plus the per-record arrays, computed once for a run over df."""
    from .core import record_arrays
    opts = feature_options(cfg)
    if opts["rich"]:
        opts["records"] = record_arrays(df)
    return opts
