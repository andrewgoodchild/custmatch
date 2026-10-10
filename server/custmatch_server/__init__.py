"""custmatch as a self-hosted service - one image, one database (server/README.md).

  db       the schema and the publish transaction (SQLite or Postgres through SQLAlchemy)
  worker   a batch run: custmatch run, then publish to the database
  matcher  real-time matching against the current run, one writer per database (a lease)
  api      lookups and POST /match (FastAPI)
"""
