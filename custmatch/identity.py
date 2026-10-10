"""Cookie and device-ID stitching, as customer data platforms (CDPs) do it.

Marketing event data - page views, app opens, email clicks - carries an anonymous cookie or
device ID on every event, and an identifier only when the visitor logs in or fills a form.
A cookie seen with an email (or a phone, or a loyalty number) belongs to that person, so the
records holding that identifier are one customer, and the cookie's anonymous history is theirs.

    "identity_events": [{"path": "web_events.csv", "cookie": "anonymous_id",
                         "identifiers": {"email": "email", "loyalty_id": "user_id"}}],
    "cookie_max_identities": 1

`identifiers` maps a record field (email, phone, or a column such as a trusted ID mapped as an
extra field) to the event file's column. Event values are cleaned exactly as record values are
(Gmail dots, phone formats, hashing under "hash_fields"), so they meet.

Merge protection - the guard every CDP has - against shared devices: a cookie seen with more
than `cookie_max_identities` (default 1) different values of any one identifier is a shared
device (a family laptop, a store kiosk) and stitches nothing. Internal traffic is the worst
case: a call-centre agent acting for customers types their emails and phones into one browser.
Drop it at the source with `"exclude": {"column": "channel", "values": ["call_centre"]}` (or a
list of internal cookies, `"exclude_cookies": "staff_cookies.csv"`). And because an agent who
enters one customer's email and another's phone passes the per-identifier limit, a cookie
stitches only when every record it reaches has a compatible first name - the same, a
nickname, a typo or an initial (`"cookie_name_check": false` turns this off); otherwise it is
reported as `conflicting_names` and stitches nothing. Identifier values profiled as a
household's shared contact or junk are not used either, and a cookie that would join more than
`cookie_max_records` (default 10) records is ignored. Stitched links are deterministic like
trusted IDs: they join records whatever the model says, but a steward's `different` still wins.

Every run then writes cookies.csv: each cookie, the customer it was stitched to (or why not).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import io


def resolve_paths(cfg, base_dir):
    """The config with event paths made absolute (saved with the state, so `add` finds them)."""
    if cfg.get("identity_events"):
        cfg["identity_events"] = [
            dict(e, path=str((Path(base_dir) / e["path"]).resolve()),
                 **({"exclude_cookies": str((Path(base_dir) / e["exclude_cookies"]).resolve())}
                    if e.get("exclude_cookies") else {}))
            for e in cfg["identity_events"]]
    return cfg


def load_events(cfg, base_dir="."):
    """One row per (cookie, identifier field, value) seen together, values cleaned as records are."""
    rows = []
    for spec in cfg.get("identity_events", []):
        if "cookie" not in spec or not spec.get("identifiers"):
            raise io.ConfigError(f"identity_events needs a 'cookie' column and 'identifiers': {spec}")
        raw = io._read((Path(base_dir) / spec["path"]).resolve())
        absent = [c for c in [spec["cookie"], *spec["identifiers"].values()] if c not in raw.columns]
        if absent:
            raise io.ConfigError(f"identity_events: columns {absent} not in {spec['path']}")
        keep = np.ones(len(raw), dtype=bool)
        ex = spec.get("exclude")
        if ex:
            if ex.get("column") not in raw.columns:
                raise io.ConfigError(f"identity_events exclude: column {ex.get('column')!r} not in {spec['path']}")
            keep &= ~io._clean(raw[ex["column"]]).str.lower().isin([str(v).lower() for v in ex.get("values", [])]).to_numpy()
        if spec.get("exclude_cookies"):
            staff = io._read((Path(base_dir) / spec["exclude_cookies"]).resolve())
            keep &= ~io._clean(raw[spec["cookie"]]).isin(set(io._clean(staff.iloc[:, 0]))).to_numpy()
        raw = raw[keep].reset_index(drop=True)
        cookie = io._clean(raw[spec["cookie"]])
        from .countries import countries
        regions = countries(cfg)[1:] if spec.get("country") is None else []
        for field, col in spec["identifiers"].items():
            primary = _normalise(raw[col], field, cfg, spec.get("country"))
            rows.append(pd.DataFrame({"cookie": cookie, "field": field, "orig": primary, "value": primary}))
            # an event does not say its country: with several configured, a phone is read as each
            # country's too, so it meets its owner's record whichever country that is
            if field == "phone":
                for r in regions:
                    # `orig` keeps the first reading: one identity, however many readings
                    rows.append(pd.DataFrame({"cookie": cookie, "field": field, "orig": primary,
                                              "value": _normalise(raw[col], field, cfg, r)}))
    if not rows:
        return pd.DataFrame(columns=["cookie", "field", "value", "orig"])
    ev = pd.concat(rows, ignore_index=True)
    return ev[(ev.cookie != "") & (ev.value != "")].drop_duplicates().reset_index(drop=True)


def _normalise(values, field, cfg, region=None):
    """An event column cleaned the way the same field of a record is (as `region`'s, if given)."""
    from . import country
    v = io._clean(values).reset_index(drop=True)
    frame = pd.DataFrame({c: "" for c in io.CANONICAL[1:]}, index=v.index)
    frame.insert(0, "record_id", "")
    frame.insert(0, "source", "events")
    frame[field] = v
    if region:
        frame["country"] = region
    if field == "phone":
        frame["phone"] = frame.phone.str.replace(r"\D", "", regex=True)
    if field in ("email", "phone"):
        frame = country.normalise(frame, cfg)
    if field in cfg.get("hash_fields", []):
        frame = io.hash_fields(frame, cfg | {"hash_fields": [field]})
    return frame[field].to_numpy()


def stitch(df, cfg, base_dir=".", profiles=None):
    """(pairs of record indices to join, a per-cookie table, counts). The table has a row per
    cookie: status stitched / shared_device / too_many_records / no_known_identity, and the
    record indices it reaches."""
    ev = load_events(cfg, base_dir)
    empty = np.zeros((0, 2), dtype=np.int64)
    if not len(ev):
        return empty, pd.DataFrame(columns=["cookie", "status", "records", "identifiers"]), {}
    limit = int(cfg.get("cookie_max_identities", 1))
    max_records = int(cfg.get("cookie_max_records", 10))
    profiles = profiles or {}
    index = {}                                     # (field, value) -> record indices
    for field in ev.field.unique():
        if field not in df.columns:
            raise io.ConfigError(f"identity_events: records have no field {field!r} (map it in the sources, "
                                 "as an extra field if it is not a standard one)")
        col = df[field].to_numpy()
        for i in np.flatnonzero(col != ""):
            index.setdefault((field, col[i]), []).append(int(i))
    name_check = cfg.get("cookie_name_check", True)
    fn = df.first_name.astype(str).str.lower().to_numpy()
    roots = df.first_roots.astype(str).to_numpy() if "first_roots" in df.columns else None
    rows, out = [], set()
    for cookie, g in ev.groupby("cookie", sort=True):
        per_field = g.groupby("field").orig.nunique()      # distinct cleaned values (first reading)
        ids = [f"{f}={v}" for f, v in zip(g.field, g.value)]
        if (per_field > limit).any():
            rows.append((cookie, "shared_device", [], ids))
            continue
        recs = set()
        for f, v in zip(g.field, g.value):
            if profiles.get(f, {}).get(v, ("personal",))[0] != "personal":
                continue                           # a family email or junk value says nothing
            recs.update(index.get((f, v), ()))
        recs = sorted(recs)
        if not recs:
            rows.append((cookie, "no_known_identity", [], ids))
        elif len(recs) > max_records:
            rows.append((cookie, "too_many_records", recs, ids))
        elif name_check and not _names_agree(recs, fn, roots, hashed="first_name" in cfg.get("hash_fields", [])):
            rows.append((cookie, "conflicting_names", recs, ids))
        else:
            rows.append((cookie, "stitched", recs, ids))
            out.update((recs[0], r) for r in recs[1:])
    table = pd.DataFrame(rows, columns=["cookie", "status", "records", "identifiers"])
    counts = {"cookies": int(len(table)), **{k: int(v) for k, v in table.status.value_counts().items()},
              "links": int(len(out))}
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 2), table, counts


def _names_agree(recs, fn, roots=None, hashed=False):
    """True when the records' first names could all be one person: equal, an initial of the
    other, close in spelling (Jaro-Winkler 0.8+), or nicknames of one name. Blank names agree.
    hashed: the names are hashes, so only equality and shared (hashed) nickname roots count."""
    from rapidfuzz.distance import JaroWinkler
    named = [r for r in recs if fn[r]]
    for x in range(len(named)):
        for y in range(x + 1, len(named)):
            a, b = fn[named[x]], fn[named[y]]
            if a == b or (not hashed and min(len(a), len(b)) == 1 and a[0] == b[0]):
                continue
            if not hashed and JaroWinkler.normalized_similarity(a, b) >= 0.8:
                continue
            if roots is not None and set(roots[named[x]].split("|")) & set(roots[named[y]].split("|")):
                continue
            return False
    return True


def write_cookies(out_dir, table, customer, minimal=False):
    """cookies.csv: each cookie and the customer it belongs to - what marketing joins its
    event history on. Identifier values are left out under "minimal_outputs"."""
    if not len(table):
        return
    t = table.copy()
    t["customer_id"] = [";".join(sorted({str(int(customer[r])) for r in recs})) if s == "stitched" else ""
                        for recs, s in zip(t.records, t.status)]
    t["identifiers"] = "" if minimal else t.identifiers.map(lambda x: ";".join(sorted(set(x))))
    t[["cookie", "customer_id", "status", "identifiers"]].to_csv(Path(out_dir) / "cookies.csv", index=False)
