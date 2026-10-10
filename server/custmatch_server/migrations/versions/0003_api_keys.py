"""API keys (custmatch_server/schema.py describes the table).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-05
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("api_keys",
                    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
                    sa.Column("name", sa.String(128), nullable=False),
                    sa.Column("role", sa.String(16), nullable=False),
                    sa.Column("key_hash", sa.String(64), nullable=False),     # sha256 of the key, hex
                    sa.Column("prefix", sa.String(12), nullable=False),       # to tell keys apart in a list
                    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
                    sa.Column("revoked_at", sa.DateTime(timezone=True)),
                    sa.CheckConstraint("role in ('read', 'match', 'steward', 'admin')", name="api_keys_role"),
                    sa.UniqueConstraint("key_hash", name="api_keys_hash"))
    op.create_index("ix_api_keys_name", "api_keys", ["name"])


def downgrade():
    op.drop_table("api_keys")
