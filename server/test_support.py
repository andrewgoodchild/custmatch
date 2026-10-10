"""Shared by the service's test scripts and tests (not part of the installed package: it drops a
Postgres schema)."""
from custmatch_server import db


def fresh_engine(url):
    """An engine on `url` with the schema created; on Postgres the public schema is dropped first,
    so a reused scratch database starts clean."""
    if url.startswith("postgresql"):
        from sqlalchemy import text
        e0 = db.engine_from_env(url, migrate=False)
        with e0.begin() as c:
            c.execute(text("DROP SCHEMA public CASCADE"))
            c.execute(text("CREATE SCHEMA public"))
        e0.dispose()
    return db.engine_from_env(url)


def customer_of(client, key):
    """The customer a record (source:record_id) is in, through the API."""
    source, _, record_id = key.partition(":")
    return client.get(f"/records/{source}/{record_id}").json()["customer_id"]
