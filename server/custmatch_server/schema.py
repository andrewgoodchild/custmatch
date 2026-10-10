"""The database schema, as SQLAlchemy tables for queries. The migrations (migrations/versions) create
it; change it there first, then here.

  loads             every load of source records: which source, when, from where, full or incremental,
                    and what it inserted, updated, deleted and left unchanged
  source_records    the current version of every source record (raw fields, a content hash, the load
                    that first brought it and the one that last changed it; deleted_at, not removed)
  source_record_versions   every version, one row per load that changed the record
  runs              the catalog: one row per batch run, the newest load it saw (input_load_id), and
                    where its files are (artifact_uri)
  current_run       exactly one row: the run readers see
  customers         per run: one row per customer, with its golden record     (partitioned by run)
  record_customer   per run: which customer each record is in                  (partitioned by run)
  links             per run: the accepted links between records, with scores   (partitioned by run)
  customer_events   merges, splits and new customers, run by run
  realtime_records  records that arrived through the API, in arrival order (seq)
  leases            the real-time matcher's write lease
  steward_log       every steward action: decided, revoked, approved, rejected, pinned, unpinned, split
  golden_pins       golden-record values a steward pinned (they beat survivorship, in every run)
  review_claims     review pairs shown to a steward, held from the others for a while
  realtime_moves    records split out of their customer since the batch run, until a run reconciles them
  api_keys          API keys: a name, a role (read < match < steward < admin), the key's sha256 hash
  overrides         stewards' decisions: two records are the same person, or different people
                    (record keys source:record_id, a sorted before b); at most one applied and one
                    proposed (four-eyes) per pair; proposals may also revoke a decision or split a
                    record out (a = the record, b = "")
  realtime_merges   customers the real-time matcher merged since the batch run (a record linking
                    two customers strongly, or a steward's "same"), in order, until a run reconciles them

On Postgres the per-run tables are LIST-partitioned by run_id: a publish fills new partitions and
retiring a run drops them, instead of inserting and deleting tens of millions of rows. SQLite has
no partitions; there the same tables hold every run and retiring deletes rows.
"""
from sqlalchemy import (JSON, BigInteger, Column, DateTime, Float, ForeignKey, ForeignKeyConstraint,
                        Integer, MetaData, String, Table, Text)
from sqlalchemy.dialects import postgresql

meta = MetaData()
JSONB = JSON().with_variant(postgresql.JSONB(), "postgresql")
PARTITIONED = ("customers", "record_customer", "links")   # child tables are created per run

loads = Table("loads", meta,
              Column("id", Integer, primary_key=True, autoincrement=True),
              Column("source", String(64), nullable=False),
              Column("origin", Text, nullable=False),
              Column("mode", String(16), nullable=False),
              Column("status", String(16), nullable=False),
              Column("started_at", DateTime(timezone=True), nullable=False),
              Column("finished_at", DateTime(timezone=True)),
              Column("rows_in", Integer), Column("inserted", Integer), Column("updated", Integer),
              Column("deleted", Integer), Column("unchanged", Integer),
              Column("error", Text))

source_records = Table("source_records", meta,
                       Column("source", String(64), primary_key=True),
                       Column("record_id", String(256), primary_key=True),
                       Column("fields", JSONB, nullable=False),
                       Column("content_hash", String(64), nullable=False),
                       Column("first_load_id", Integer, ForeignKey("loads.id"), nullable=False),
                       Column("last_load_id", Integer, ForeignKey("loads.id"), nullable=False),
                       Column("updated_at", DateTime(timezone=True), nullable=False),
                       Column("deleted_at", DateTime(timezone=True)))

source_record_versions = Table("source_record_versions", meta,
                               Column("source", String(64), primary_key=True),
                               Column("record_id", String(256), primary_key=True),
                               Column("load_id", Integer, ForeignKey("loads.id"), primary_key=True),
                               Column("change", String(8), nullable=False),
                               Column("fields", JSONB),
                               Column("content_hash", String(64)))

runs = Table("runs", meta,
             Column("id", Integer, primary_key=True, autoincrement=True),
             Column("status", String(16), nullable=False),     # running | published | failed | retired
             Column("input_load_id", Integer, ForeignKey("loads.id")),
             Column("artifact_uri", Text),                      # the run's files: state for the matcher
             Column("summary", JSONB),
             Column("error", Text),
             Column("created_at", DateTime(timezone=True), nullable=False),
             Column("published_at", DateTime(timezone=True)))

current_run = Table("current_run", meta,
                    Column("id", Integer, primary_key=True),    # always 1: one row, one current run
                    Column("run_id", Integer, ForeignKey("runs.id"), nullable=False))

customers = Table("customers", meta,
                  Column("run_id", Integer, ForeignKey("runs.id"), primary_key=True),
                  Column("customer_id", BigInteger, primary_key=True),
                  Column("household_id", BigInteger),
                  Column("record_count", Integer, nullable=False),
                  Column("created_run", Integer, nullable=False),
                  Column("first_name", Text), Column("middle_initial", Text), Column("last_name", Text),
                  Column("date_of_birth", Text), Column("email", Text), Column("phone", Text),
                  Column("addr_number", Text), Column("addr_street", Text), Column("addr_unit", Text),
                  Column("golden_extra", JSONB),
                  postgresql_partition_by="LIST (run_id)")

record_customer = Table("record_customer", meta,
                        Column("run_id", Integer, primary_key=True),
                        Column("source", String(64), primary_key=True),
                        Column("record_id", String(256), primary_key=True),
                        Column("customer_id", BigInteger, nullable=False),
                        Column("household_id", BigInteger),
                        Column("confidence", Float),
                        ForeignKeyConstraint(["run_id", "customer_id"], ["customers.run_id", "customers.customer_id"]),
                        postgresql_partition_by="LIST (run_id)")

links = Table("links", meta,
              Column("run_id", Integer, primary_key=True),
              Column("source_a", String(64), primary_key=True), Column("record_a", String(256), primary_key=True),
              Column("source_b", String(64), primary_key=True), Column("record_b", String(256), primary_key=True),
              Column("customer_id", BigInteger),
              Column("score", Float),
              Column("is_bridge", Integer),
              Column("evidence", Text),
              postgresql_partition_by="LIST (run_id)")

customer_events = Table("customer_events", meta,
                        Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
                        Column("run_id", Integer, ForeignKey("runs.id"), nullable=False),
                        Column("at", DateTime(timezone=True)),
                        Column("event", String(32), nullable=False),
                        Column("customer_id", BigInteger),
                        Column("other_customer_ids", Text),
                        Column("record_count", Integer),
                        Column("detail", Text))

realtime_records = Table("realtime_records", meta,
                         Column("seq", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
                         Column("source", String(64), nullable=False),
                         Column("record_id", String(256), nullable=False),
                         Column("customer_id", BigInteger, nullable=False),
                         Column("score", Float),
                         Column("matched_against_run", Integer, ForeignKey("runs.id"), nullable=False),
                         Column("reconciled_run", Integer, ForeignKey("runs.id")),
                         Column("created_at", DateTime(timezone=True), nullable=False))

overrides = Table("overrides", meta,
                  Column("id", Integer, primary_key=True, autoincrement=True),
                  Column("a", String(330), nullable=False),
                  Column("b", String(330), nullable=False),
                  Column("decision", String(16), nullable=False),
                  Column("created_at", DateTime(timezone=True), nullable=False),
                  Column("created_by", String(128)),
                  Column("note", Text),
                  Column("status", String(16), nullable=False, server_default="applied"),   # or proposed (four-eyes)
                  Column("approved_by", String(128)),
                  Column("approved_at", DateTime(timezone=True)))

steward_log = Table("steward_log", meta,
                    Column("id", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
                    Column("at", DateTime(timezone=True), nullable=False),
                    Column("actor", String(128)),
                    Column("action", String(32), nullable=False),
                    Column("a", String(330)), Column("b", String(330)),
                    Column("customer_id", BigInteger),
                    Column("detail", Text), Column("note", Text))

golden_pins = Table("golden_pins", meta,
                    Column("id", Integer, primary_key=True, autoincrement=True),
                    Column("record_key", String(330), nullable=False),
                    Column("field", String(64), nullable=False),
                    Column("value", Text, nullable=False),
                    Column("created_at", DateTime(timezone=True), nullable=False),
                    Column("created_by", String(128)),
                    Column("note", Text))

review_claims = Table("review_claims", meta,
                      Column("a", String(330), primary_key=True),
                      Column("b", String(330), primary_key=True),
                      Column("holder", String(128), nullable=False),
                      Column("expires_at", DateTime(timezone=True), nullable=False))

realtime_moves = Table("realtime_moves", meta,
                       Column("seq", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
                       Column("source", String(64), nullable=False),
                       Column("record_id", String(256), nullable=False),
                       Column("from_customer", BigInteger, nullable=False),
                       Column("to_customer", BigInteger, nullable=False),
                       Column("reason", Text),
                       Column("matched_against_run", Integer, ForeignKey("runs.id"), nullable=False),
                       Column("reconciled_run", Integer, ForeignKey("runs.id")),
                       Column("created_at", DateTime(timezone=True), nullable=False))

realtime_merges = Table("realtime_merges", meta,
                        Column("seq", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True),
                        Column("into_customer", BigInteger, nullable=False),
                        Column("from_customer", BigInteger, nullable=False),
                        Column("reason", Text),
                        Column("matched_against_run", Integer, ForeignKey("runs.id"), nullable=False),
                        Column("reconciled_run", Integer, ForeignKey("runs.id")),
                        Column("created_at", DateTime(timezone=True), nullable=False))

api_keys = Table("api_keys", meta,
                 Column("id", Integer, primary_key=True, autoincrement=True),
                 Column("name", String(128), nullable=False),
                 Column("role", String(16), nullable=False),
                 Column("key_hash", String(64), nullable=False),
                 Column("prefix", String(12), nullable=False),
                 Column("created_at", DateTime(timezone=True), nullable=False),
                 Column("revoked_at", DateTime(timezone=True)))

leases = Table("leases", meta,
               Column("name", String(64), primary_key=True),
               Column("holder", String(128), nullable=False),
               Column("expires_at", DateTime(timezone=True), nullable=False))
