"""
Database configuration and session management for SQLAlchemy and ChromaDB
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
import os
import asyncio
from typing import Optional

# Model imports moved to function to avoid circular imports
# They will be imported when initialize_database() is called

# Database URLs
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/chatbot.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

# Ensure data directories exist
os.makedirs(os.path.dirname(DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# SQLAlchemy setup
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """SQLAlchemy 2.0-style base class for declarative models."""
    pass

# ChromaDB setup
chroma_client = None
vector_store_initialized = False

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    try:
        # Import models here to avoid circular imports
        from .models.conversation import Conversation
        from .models.message import Message
        from .models.document import Document
        from .models.summary import Summary
        from .models.media_content import MediaContent

        Base.metadata.create_all(bind=engine)
        print("SQLAlchemy tables created successfully")
        return True
    except Exception as e:
        print(f"Failed to create tables: {e}")
        return False

def drop_tables():
    """Drop all database tables (for testing/reset)"""
    try:
        Base.metadata.drop_all(bind=engine)
        print("✅ SQLAlchemy tables dropped successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to drop tables: {e}")
        return False

def initialize_chroma():
    """Initialize ChromaDB vector store"""
    global chroma_client, vector_store_initialized

    try:
        import chromadb
        from chromadb.config import Settings

        chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        # Test connection by creating/getting a collection
        test_collection = chroma_client.get_or_create_collection("test_init")
        chroma_client.delete_collection("test_init")

        vector_store_initialized = True
        print("ChromaDB initialized successfully")
        return True

    except ImportError:
        print("ChromaDB not available - install chromadb package")
        return False
    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")
        return False

def initialize_database():
    """Initialize both SQLAlchemy and ChromaDB"""
    print("Initializing databases...")

    # Initialize SQLAlchemy
    sql_success = create_tables()

    # Run lightweight migrations for SQLite to add new columns when models evolved
    def run_sqlite_migrations():
        if not DATABASE_URL.startswith("sqlite"):
            return
        print("Running lightweight SQLite migrations...")
        try:
            with engine.connect() as conn:
                # Get existing columns for conversations
                result = conn.execute(text("PRAGMA table_info('conversations')"))
                rows = result.fetchall()
                existing = {r[1] for r in rows}

                # Define migrations: column name -> SQL fragment to add
                migrations = {
                    'is_pinned': "ALTER TABLE conversations ADD COLUMN is_pinned BOOLEAN DEFAULT 0",
                    'is_archived': "ALTER TABLE conversations ADD COLUMN is_archived BOOLEAN DEFAULT 0",
                    'metrics': "ALTER TABLE conversations ADD COLUMN metrics JSON",
                    'participants': "ALTER TABLE conversations ADD COLUMN participants JSON",
                    'retention_policy': "ALTER TABLE conversations ADD COLUMN retention_policy VARCHAR(100)",
                }

                for col, stmt in migrations.items():
                    if col not in existing:
                        try:
                            conn.execute(text(stmt))
                            print(f"Added missing column '{col}' to conversations table")
                        except Exception as e:
                            # Non-fatal: log and continue
                            print(f"Failed to add column {col}: {e}")
        except Exception as e:
            print(f"SQLite migration step failed: {e}")

    # Run migrations after creating tables
    run_sqlite_migrations()

    # Initialize ChromaDB
    chroma_success = initialize_chroma()

    if sql_success and chroma_success:
        print("All databases initialized successfully!")
        return True
    else:
        print("Some databases failed to initialize")
        return False

def get_database_status():
    """Get status of all database systems"""
    # Check SQLAlchemy
    sql_status = "unknown"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            sql_status = "connected"
    except Exception as e:
        sql_status = f"error: {str(e)}"

    # Check ChromaDB
    chroma_status = "not_initialized" if not vector_store_initialized else "ready"
    if chroma_client:
        try:
            collections = chroma_client.list_collections()
            chroma_status = f"ready ({len(collections)} collections)"
        except Exception as e:
            chroma_status = f"error: {str(e)}"

    return {
        "sqlalchemy": {
            "status": sql_status,
            "url": DATABASE_URL.replace(os.path.dirname(DATABASE_URL), "...")  # Hide path for security
        },
        "chromadb": {
            "status": chroma_status,
            "persist_dir": CHROMA_PERSIST_DIR
        },
        "overall_status": "healthy" if sql_status == "connected" and "ready" in chroma_status else "degraded"
    }

# Auto-initialize on import (in development) unless explicitly skipped. This
# environment variable is set by Alembic env.py to avoid side-effects during
# autogenerate and migration runs.
if __name__ != "__main__":
    if os.getenv("SKIP_DB_AUTO_INIT", "0") != "1":
        # Only auto-initialize if not running as script and not skipped
        initialize_database()