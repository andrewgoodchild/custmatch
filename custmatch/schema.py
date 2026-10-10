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


def extra_keys(cfg):
    return tuple(tuple(k) for k in cfg.get("extra_blocking", []))


def feature_options(cfg):
    """Keyword arguments for core.features from the config: extra fields and the nine rarity /
    name-swap / graded-date features - on by default ("rich_features": false turns them off).
    They lifted historical_50k from 82.1 to 87.0 and fake_1000 from 88.2 to 92.5 clustered F1,
    and cost at most 0.3 on the febrl sets (research/scripts/beat_splink.py)."""
    return {"extra_fields": extra_fields(cfg), "rich": bool(cfg.get("rich_features", True))}


def feature_options_for(df, cfg):
    """feature_options plus the per-record arrays, computed once for a run over df."""
    from .core import record_arrays
    opts = feature_options(cfg)
    if opts["rich"]:
        opts["records"] = record_arrays(df)
    return opts
