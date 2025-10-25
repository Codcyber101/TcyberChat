"""initial migration - create all tables

Revision ID: 0001_initial
Revises: 
Create Date: 2025-10-23
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create all tables from SQLAlchemy metadata."""
    # Import here to ensure app path is available
    from src.database import Base

    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    """Drop all tables created by this migration."""
    from src.database import Base

    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
