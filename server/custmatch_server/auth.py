"""API keys and roles.

A key is a random token shown once when it is created; the database keeps only its SHA-256 hash
(a token of 32 random bytes needs no slow hash: there is nothing to guess). A request sends it as
`Authorization: Bearer <key>`. Roles, each including the ones before it:

  read      lookups: records, customers, events, loads, runs, overrides
  match     POST /match (real-time matching; a write also needs the matcher's lease)
  steward   POST /overrides
  admin     everything (keys are managed from the command line, not the API)

Authentication is on unless CUSTMATCH_AUTH=off (a local trial); /health is always open.
"""
import hashlib
import secrets
import time

from sqlalchemy import insert, select, update

from . import db
from .schema import api_keys

ROLES = ("read", "match", "steward", "admin")
CACHE_SECONDS = 30          # a revoked key stops working within this long
CACHE_SIZE = 10_000


def _hash(key):
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(eng, name, role):
    """A new key for `name` with `role`; returns the key - shown once, never stored."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    key = "cm_" + secrets.token_urlsafe(32)
    with eng.begin() as c:
        c.execute(insert(api_keys).values(name=name, role=role, key_hash=_hash(key), prefix=key[:10],
                                          created_at=db.now()))
    return key


def revoke(eng, name):
    """Revoke every live key of `name`; returns how many."""
    with eng.begin() as c:
        return c.execute(update(api_keys).where(api_keys.c.name == name).where(api_keys.c.revoked_at.is_(None))
                         .values(revoked_at=db.now())).rowcount


def list_keys(eng):
    with eng.connect() as c:
        return [dict(r) for r in c.execute(select(api_keys.c.name, api_keys.c.role, api_keys.c.prefix,
                                                  api_keys.c.created_at, api_keys.c.revoked_at)
                                           .order_by(api_keys.c.id)).mappings()]


class Keys:
    """Checks a request's key, with a short, bounded cache so a lookup is not a query every time
    (an LRU of CACHE_SIZE: random tokens from unauthenticated callers cannot grow it without end)."""

    def __init__(self, eng):
        from collections import OrderedDict
        self.eng, self.cache = eng, OrderedDict()

    def identify(self, key):
        """(name, role) for a live key, or None."""
        if not key or len(key) > 200:
            return None
        h = _hash(key)
        hit = self.cache.get(h)
        if hit and time.monotonic() - hit[1] < CACHE_SECONDS:
            self.cache.move_to_end(h)
            return hit[0]
        with self.eng.connect() as c:
            row = c.execute(select(api_keys.c.name, api_keys.c.role).where(api_keys.c.key_hash == h)
                            .where(api_keys.c.revoked_at.is_(None))).first()
        who = (row[0], row[1]) if row else None
        self.cache[h] = (who, time.monotonic())
        self.cache.move_to_end(h)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)
        return who


def allows(role, needed):
    return ROLES.index(role) >= ROLES.index(needed)
