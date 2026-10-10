"""Decisions: one applied and one proposed decision per pair (a proposal no longer replaces the
applied decision it would change); proposals to revoke a decision or split a record out (four-eyes);
no a < b check (Postgres collations order keys differently from Python, which orders them).

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-07
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None
COLS = "id, a, b, decision, created_at, created_by, note, status, approved_by, approved_at"


def _table(name, decisions, unique):
    op.create_table(name,
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("a", sa.String(330), nullable=False),
                    sa.Column("b", sa.String(330), nullable=False),
                    sa.Column("decision", sa.String(16), nullable=False),
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("created_by", sa.String(128)),
                    sa.Column("note", sa.Text),
                    sa.Column("status", sa.String(16), nullable=False, server_default="applied"),
                    sa.Column("approved_by", sa.String(128)),
                    sa.Column("approved_at", sa.DateTime(timezone=True)),
                    sa.CheckConstraint(f"decision in ({decisions})", name=f"{name}_decision"),
                    sa.CheckConstraint("status in ('applied', 'proposed')", name=f"{name}_status"),
                    sa.UniqueConstraint(*unique, name=f"{name}_pair"))


def _swap(decisions, unique):
    """Recreate overrides with new constraints (SQLite cannot alter CHECK constraints in place)."""
    _table("overrides_next", decisions, unique)
    op.execute(f"INSERT INTO overrides_next ({COLS}) SELECT {COLS} FROM overrides")
    op.drop_table("overrides")
    op.rename_table("overrides_next", "overrides")
    if op.get_bind().dialect.name == "postgresql":
        op.execute("SELECT setval(pg_get_serial_sequence('overrides', 'id'), COALESCE(MAX(id), 1)) FROM overrides")


def upgrade():
    _swap("'same', 'different', 'revoke', 'split'", ("a", "b", "status"))


def downgrade():
    op.execute("DELETE FROM overrides WHERE status = 'proposed'")
    _swap("'same', 'different'", ("a", "b"))
