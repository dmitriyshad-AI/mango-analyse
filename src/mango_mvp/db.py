from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from mango_mvp.config import Settings


class Base(DeclarativeBase):
    pass


def build_engine(settings: Settings):
    return create_engine(settings.database_url, future=True)


def build_session_factory(settings: Settings):
    engine = build_engine(settings)
    import mango_mvp.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_columns(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(settings: Settings) -> None:
    engine = build_engine(settings)
    import mango_mvp.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_columns(engine)


def _ensure_columns(engine) -> None:
    inspector = inspect(engine)
    if "call_records" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("call_records")}
    additions = {
        "transcript_variants_json": "TEXT",
        "resolve_json": "TEXT",
        "resolve_quality_score": "FLOAT",
        "transcribe_attempts": "INTEGER DEFAULT 0 NOT NULL",
        "resolve_attempts": "INTEGER DEFAULT 0 NOT NULL",
        "analyze_attempts": "INTEGER DEFAULT 0 NOT NULL",
        "sync_attempts": "INTEGER DEFAULT 0 NOT NULL",
        "resolve_status": "VARCHAR(16) DEFAULT 'pending'",
        "next_retry_at": "DATETIME",
        "dead_letter_stage": "VARCHAR(16)",
    }
    with engine.begin() as conn:
        for column_name, sql_type in additions.items():
            if column_name in existing:
                continue
            conn.execute(
                text(f"ALTER TABLE call_records ADD COLUMN {column_name} {sql_type}")
            )
        # Keep worker/requeue queries fast on legacy DBs where these indexes did not exist.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_call_records_next_retry_at "
                "ON call_records (next_retry_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_call_records_dead_letter_stage "
                "ON call_records (dead_letter_stage)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_call_records_resolve_status "
                "ON call_records (resolve_status)"
            )
        )
