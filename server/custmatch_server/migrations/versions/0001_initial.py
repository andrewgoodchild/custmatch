"""The initial schema (custmatch_server/schema.py describes it).

Revision ID: 0001
Revises:
Create Date: 2026-10-05
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

JSONB = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
SEQ = sa.BigInteger().with_variant(sa.Integer, "sqlite")


def upgrade():
    pg = op.get_bind().dialect.name == "postgresql"
    part = {"postgresql_partition_by": "LIST (run_id)"} if pg else {}
    # ---- the source records, load by load
    op.create_table("loads",
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("source", sa.String(64), nullable=False),
                    sa.Column("origin", sa.Text, nullable=False),            # a file, "sync", or "api"
                    sa.Column("mode", sa.String(16), nullable=False),        # full | incremental
                    sa.Column("status", sa.String(16), nullable=False),      # running | done | failed
                    sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("finished_at", sa.DateTime(timezone=True)),
                    sa.Column("rows_in", sa.Integer), sa.Column("inserted", sa.Integer),
                    sa.Column("updated", sa.Integer), sa.Column("deleted", sa.Integer),
                    sa.Column("unchanged", sa.Integer),
                    sa.Column("error", sa.Text),
                    sa.CheckConstraint("mode IN ('full', 'incremental')", name="loads_mode"),
                    sa.CheckConstraint("status IN ('running', 'done', 'failed')", name="loads_status"))
    op.create_table("source_records",
                    sa.Column("source", sa.String(64), primary_key=True),
                    sa.Column("record_id", sa.String(256), primary_key=True),
                    sa.Column("fields", JSONB, nullable=False),
                    sa.Column("content_hash", sa.String(64), nullable=False),
                    sa.Column("first_load_id", sa.Integer, sa.ForeignKey("loads.id"), nullable=False),
                    sa.Column("last_load_id", sa.Integer, sa.ForeignKey("loads.id"), nullable=False),
                    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("deleted_at", sa.DateTime(timezone=True)))
    op.create_index("ix_source_records_live", "source_records", ["source", "deleted_at"])
    op.create_table("source_record_versions",
                    sa.Column("source", sa.String(64), primary_key=True),
                    sa.Column("record_id", sa.String(256), primary_key=True),
                    sa.Column("load_id", sa.Integer, sa.ForeignKey("loads.id"), primary_key=True),
                    sa.Column("change", sa.String(8), nullable=False),        # insert | update | delete
                    sa.Column("fields", JSONB),
                    sa.Column("content_hash", sa.String(64)),
                    sa.CheckConstraint("change IN ('insert', 'update', 'delete')", name="versions_change"))
    op.create_index("ix_versions_load", "source_record_versions", ["load_id"])

    op.create_table("runs",
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("status", sa.String(16), nullable=False),
                    sa.Column("input_load_id", sa.Integer, sa.ForeignKey("loads.id")),   # the newest load the run saw
                    sa.Column("artifact_uri", sa.Text),
                    sa.Column("summary", JSONB),
                    sa.Column("error", sa.Text),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("published_at", sa.DateTime(timezone=True)),
                    sa.CheckConstraint("status IN ('running', 'published', 'failed', 'retired')", name="runs_status"))
    op.create_table("current_run",
                    sa.Column("id", sa.Integer, primary_key=True),
                    sa.Column("run_id", sa.Integer, sa.ForeignKey("runs.id"), nullable=False),
                    sa.CheckConstraint("id = 1", name="current_run_single_row"))
    op.create_table("customers",
                    sa.Column("run_id", sa.Integer, sa.ForeignKey("runs.id"), primary_key=True),
                    sa.Column("customer_id", sa.BigInteger, primary_key=True),
                    sa.Column("household_id", sa.BigInteger),
                    sa.Column("record_count", sa.Integer, nullable=False),
                    sa.Column("created_run", sa.Integer, nullable=False),
                    *[sa.Column(c, sa.Text) for c in ("first_name", "middle_initial", "last_name", "date_of_birth",
                                                       "email", "phone", "addr_number", "addr_street", "addr_unit")],
                    sa.Column("golden_extra", JSONB),
                    **part)
    op.create_table("record_customer",
                    sa.Column("run_id", sa.Integer, primary_key=True),
                    sa.Column("source", sa.String(64), primary_key=True),
                    sa.Column("record_id", sa.String(256), primary_key=True),
                    sa.Column("customer_id", sa.BigInteger, nullable=False),
                    sa.Column("household_id", sa.BigInteger),
                    sa.Column("confidence", sa.Float),
                    sa.ForeignKeyConstraint(["run_id", "customer_id"], ["customers.run_id", "customers.customer_id"]),
                    **part)
    op.create_index("ix_record_customer_customer", "record_customer", ["run_id", "customer_id"])
    op.create_table("links",
                    sa.Column("run_id", sa.Integer, primary_key=True),
                    sa.Column("source_a", sa.String(64), primary_key=True),
                    sa.Column("record_a", sa.String(256), primary_key=True),
                    sa.Column("source_b", sa.String(64), primary_key=True),
                    sa.Column("record_b", sa.String(256), primary_key=True),
                    sa.Column("customer_id", sa.BigInteger),
                    sa.Column("score", sa.Float),
                    sa.Column("is_bridge", sa.Integer),
                    sa.Column("evidence", sa.Text),
                    **part)
    op.create_index("ix_links_customer", "links", ["run_id", "customer_id"])
    op.create_table("customer_events",
                    sa.Column("id", SEQ, primary_key=True, autoincrement=True),
                    sa.Column("run_id", sa.Integer, sa.ForeignKey("runs.id"), nullable=False),
                    sa.Column("at", sa.DateTime(timezone=True)),
                    sa.Column("event", sa.String(32), nullable=False),
                    sa.Column("customer_id", sa.BigInteger),
                    sa.Column("other_customer_ids", sa.Text),
                    sa.Column("record_count", sa.Integer),
                    sa.Column("detail", sa.Text))
    op.create_index("ix_customer_events_customer", "customer_events", ["customer_id", "id"])
    op.create_table("realtime_records",
                    sa.Column("seq", SEQ, primary_key=True, autoincrement=True),
                    sa.Column("source", sa.String(64), nullable=False),
                    sa.Column("record_id", sa.String(256), nullable=False),
                    sa.Column("customer_id", sa.BigInteger, nullable=False),
                    sa.Column("score", sa.Float),
                    sa.Column("matched_against_run", sa.Integer, sa.ForeignKey("runs.id"), nullable=False),
                    sa.Column("reconciled_run", sa.Integer, sa.ForeignKey("runs.id")),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.UniqueConstraint("source", "record_id", name="realtime_records_record"),
                    sa.ForeignKeyConstraint(["source", "record_id"], ["source_records.source", "source_records.record_id"]))
    op.create_index("ix_realtime_pending", "realtime_records", ["reconciled_run", "seq"])
    op.create_table("leases",
                    sa.Column("name", sa.String(64), primary_key=True),
                    sa.Column("holder", sa.String(128), nullable=False),
                    sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False))


def downgrade():
    for t in ("leases", "realtime_records", "customer_events", "links", "record_customer", "customers",
              "current_run", "runs", "source_record_versions", "source_records", "loads"):
        op.drop_table(t)
