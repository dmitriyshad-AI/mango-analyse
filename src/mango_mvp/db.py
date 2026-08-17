from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from mango_mvp.config import Settings


class Base(DeclarativeBase):
    pass


def build_engine(settings: Settings):
    _ensure_sqlite_parent_dir(settings.database_url)
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["timeout"] = max(1.0, settings.sqlite_busy_timeout_ms / 1000.0)
    engine = create_engine(settings.database_url, future=True, connect_args=connect_args)

    if settings.database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record):  # type: ignore[no-redef]
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute(f"PRAGMA busy_timeout={max(0, int(settings.sqlite_busy_timeout_ms))}")
                if settings.sqlite_wal_enabled and str(settings.database_url).startswith("sqlite:///"):
                    cursor.execute("PRAGMA journal_mode=WAL")
            finally:
                cursor.close()

    return engine


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    if not database_url.startswith("sqlite"):
        return
    url = make_url(database_url)
    database = url.database
    if not database or database == ":memory:":
        return
    db_path = Path(database).expanduser()
    if db_path.name:
        db_path.parent.mkdir(parents=True, exist_ok=True)


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
        "source_recording_id": "VARCHAR(256)",
        "resolve_json": "TEXT",
        "resolve_quality_score": "FLOAT",
        "transcribe_attempts": "INTEGER DEFAULT 0 NOT NULL",
        "resolve_attempts": "INTEGER DEFAULT 0 NOT NULL",
        "analyze_attempts": "INTEGER DEFAULT 0 NOT NULL",
        "sync_attempts": "INTEGER DEFAULT 0 NOT NULL",
        "pipeline_stage": "VARCHAR(32)",
        "pipeline_worker_id": "VARCHAR(64)",
        "pipeline_claimed_at": "DATETIME",
        "analysis_worker_id": "VARCHAR(64)",
        "analysis_claimed_at": "DATETIME",
        "analysis_attempts_json": "TEXT",
        "resolve_status": "VARCHAR(16) DEFAULT 'pending'",
        "next_retry_at": "DATETIME",
        "dead_letter_stage": "VARCHAR(16)",
    }
    indexes = (
        {}
        if engine.dialect.name == "sqlite"
        else {item["name"]: item for item in inspector.get_indexes("call_records")}
    )
    with engine.begin() as conn:
        for column_name, sql_type in additions.items():
            if column_name in existing:
                continue
            conn.execute(
                text(f"ALTER TABLE call_records ADD COLUMN {column_name} {sql_type}")
            )
        duplicate_recording_id = conn.execute(
            text(
                "SELECT source_recording_id FROM call_records "
                "WHERE source_recording_id IS NOT NULL AND TRIM(source_recording_id) <> '' "
                "GROUP BY TRIM(source_recording_id) HAVING COUNT(*) > 1 LIMIT 1"
            )
        ).first()
        if duplicate_recording_id is not None:
            raise RuntimeError("duplicate source_recording_id blocks database migration")
        source_index = indexes.get("ix_call_records_source_recording_id")
        if engine.dialect.name == "sqlite":
            source_index_sql = conn.execute(
                text(
                    "SELECT sql FROM sqlite_master "
                    "WHERE type='index' AND name='ix_call_records_source_recording_id'"
                )
            ).scalar_one_or_none()
            source_index_is_raw = bool(
                source_index_sql and "TRIM(" not in str(source_index_sql).upper()
            )
        else:
            source_index_is_raw = source_index is not None
        if source_index_is_raw:
            conn.execute(text("DROP INDEX ix_call_records_source_recording_id"))
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_call_records_source_recording_id "
                "ON call_records (TRIM(source_recording_id)) "
                "WHERE source_recording_id IS NOT NULL AND TRIM(source_recording_id) <> ''"
            )
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
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_call_records_pipeline_stage "
                "ON call_records (pipeline_stage)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_call_records_pipeline_worker_id "
                "ON call_records (pipeline_worker_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_call_records_pipeline_claimed_at "
                "ON call_records (pipeline_claimed_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_call_records_analysis_worker_id "
                "ON call_records (analysis_worker_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_call_records_analysis_claimed_at "
                "ON call_records (analysis_claimed_at)"
            )
        )
