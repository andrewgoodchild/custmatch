"""Stewards' decisions (overrides) and the real-time matcher's merges (custmatch_server/schema.py
describes them).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-05
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("overrides",
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("a", sa.String(330), nullable=False),     # source:record_id, a < b
                    sa.Column("b", sa.String(330), nullable=False),
                    sa.Column("decision", sa.String(16), nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("created_by", sa.String(128)),
                    sa.Column("note", sa.Text),
                    sa.CheckConstraint("decision in ('same', 'different')", name="overrides_decision"),
                    sa.CheckConstraint("a < b", name="overrides_ordered"),
                    sa.UniqueConstraint("a", "b", name="overrides_pair"))
    op.create_table("realtime_merges",
                    sa.Column("seq", sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True,
                              autoincrement=True),
                    sa.Column("into_customer", sa.BigInteger, nullable=False),
                    sa.Column("from_customer", sa.BigInteger, nullable=False),
                    sa.Column("reason", sa.Text),
                    sa.Column("matched_against_run", sa.Integer, sa.ForeignKey("runs.id"), nullable=False),
                    sa.Column("reconciled_run", sa.Integer, sa.ForeignKey("runs.id")),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_realtime_merges_pending", "realtime_merges", ["reconciled_run", "seq"])


def downgrade():
    op.drop_table("realtime_merges")
    op.drop_table("overrides")
