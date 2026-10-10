"""Stewardship: decision status (four-eyes), an audit log, pinned golden values, review claims,
real-time record moves (custmatch_server/schema.py describes them).

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-06
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None
SEQ = sa.BigInteger().with_variant(sa.Integer, "sqlite")


def upgrade():
    with op.batch_alter_table("overrides") as t:
        t.add_column(sa.Column("status", sa.String(16), nullable=False, server_default="applied"))
        t.add_column(sa.Column("approved_by", sa.String(128)))
        t.add_column(sa.Column("approved_at", sa.DateTime(timezone=True)))
        t.create_check_constraint("overrides_status", "status in ('applied', 'proposed')")
    op.create_table("steward_log",
                    sa.Column("id", SEQ, primary_key=True, autoincrement=True),
                    sa.Column("at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("actor", sa.String(128)),
                    sa.Column("action", sa.String(32), nullable=False),
                    sa.Column("a", sa.String(330)),
                    sa.Column("b", sa.String(330)),
                    sa.Column("customer_id", sa.BigInteger),
                    sa.Column("detail", sa.Text),
                    sa.Column("note", sa.Text))
    op.create_index("ix_steward_log_at", "steward_log", ["at"])
    op.create_table("golden_pins",
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("record_key", sa.String(330), nullable=False),   # the pin follows this record's customer
                    sa.Column("field", sa.String(64), nullable=False),
                    sa.Column("value", sa.Text, nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("created_by", sa.String(128)),
                    sa.Column("note", sa.Text),
                    sa.UniqueConstraint("record_key", "field", name="golden_pins_record_field"))
    op.create_table("review_claims",
                    sa.Column("a", sa.String(330), nullable=False),
                    sa.Column("b", sa.String(330), nullable=False),
                    sa.Column("holder", sa.String(128), nullable=False),
                    sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
                    sa.PrimaryKeyConstraint("a", "b"))
    op.create_table("realtime_moves",
                    sa.Column("seq", SEQ, primary_key=True, autoincrement=True),
                    sa.Column("source", sa.String(64), nullable=False),
                    sa.Column("record_id", sa.String(256), nullable=False),
                    sa.Column("from_customer", sa.BigInteger, nullable=False),
                    sa.Column("to_customer", sa.BigInteger, nullable=False),
                    sa.Column("reason", sa.Text),
                    sa.Column("matched_against_run", sa.Integer, sa.ForeignKey("runs.id"), nullable=False),
                    sa.Column("reconciled_run", sa.Integer, sa.ForeignKey("runs.id")),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_realtime_moves_pending", "realtime_moves", ["reconciled_run", "seq"])


def downgrade():
    for t in ("realtime_moves", "review_claims", "golden_pins", "steward_log"):
        op.drop_table(t)
    with op.batch_alter_table("overrides") as t:
        t.drop_constraint("overrides_status", type_="check")
        t.drop_column("approved_at")
        t.drop_column("approved_by")
        t.drop_column("status")
