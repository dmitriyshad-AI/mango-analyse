from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.contracts import BotContextChunk
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.ingestion import compact_text
from mango_mvp.customer_timeline.mail_stage2_ingest import MAIL_STAGE2_INGEST_SOURCE_SYSTEM
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    json_dumps,
    json_loads,
    parse_datetime,
    scrub_timeline_persisted_json,
    timeline_email_content_key,
)
from mango_mvp.productization.mail_archive import read_safe_stage2_event_text


MAIL_STAGE2_ENRICH_SCHEMA_VERSION = "mail_stage2_existing_enrich_v1"


@dataclass(frozen=True)
class MailStage2ExistingEnrichConfig:
    timeline_db_path: Path
    allowed_root: Path
    archive_db_paths: Sequence[Path]
    out_dir: Path
    tenant_id: str = "foton"
    text_max_chars: int = 6000
    apply: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_db_path", Path(self.timeline_db_path).expanduser())
        object.__setattr__(self, "allowed_root", Path(self.allowed_root).expanduser())
        object.__setattr__(self, "archive_db_paths", tuple(Path(path).expanduser() for path in self.archive_db_paths))
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())
        if self.text_max_chars < 500:
            raise ValueError("text_max_chars must be >= 500")


def enrich_existing_mail_stage2_from_archives(config: MailStage2ExistingEnrichConfig) -> Mapping[str, Any]:
    started = time.monotonic()
    db_path = guard_customer_timeline_output_path(config.timeline_db_path, config.allowed_root)
    _reject_prod_path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"timeline DB does not exist: {db_path}")
    if not config.archive_db_paths:
        raise ValueError("archive_db_paths must not be empty")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    with CustomerTimelineSQLiteStore(db_path, allowed_root=config.allowed_root) as store:
        con = store._con  # noqa: SLF001 - staging maintenance uses store-owned connection/FTS helpers.
        event_rows = _load_stage2_event_rows(con, tenant_id=config.tenant_id)
        target_shas = {row["message_sha256"] for row in event_rows if row["message_sha256"]}
        archive_index = _load_archive_text_index(
            config.archive_db_paths,
            target_shas=target_shas,
            text_max_chars=config.text_max_chars,
        )
        chunk_by_event = _load_stage2_chunks_by_event(con)
        counters: Counter[str] = Counter(
            events_scanned=len(event_rows),
            archive_texts=len(archive_index["texts"]),
            archive_missing=len(target_shas - set(archive_index["texts"])),
        )
        examples: list[Mapping[str, Any]] = []
        if config.apply:
            for row in event_rows:
                text_item = archive_index["texts"].get(row["message_sha256"])
                if not text_item:
                    counters["events_without_archive_text"] += 1
                    continue
                text = str(text_item["text"])
                if not text:
                    counters["events_empty_text"] += 1
                    continue
                event_changed = _update_event_row(con, row=row, text_item=text_item, text_max_chars=config.text_max_chars)
                if event_changed:
                    counters["events_updated"] += 1
                else:
                    counters["events_duplicate"] += 1
                chunk_row = chunk_by_event.get(row["event_id"])
                if chunk_row is not None:
                    chunk_changed = _update_chunk_row(con, row=chunk_row, text_item=text_item)
                    counters["chunks_updated" if chunk_changed else "chunks_duplicate"] += 1
                elif row["customer_id"]:
                    chunk_result = _create_missing_chunk(store, row=row, text_item=text_item)
                    counters[f"chunks_{chunk_result}"] += 1
                else:
                    counters["chunks_not_created_unlinked"] += 1
                if len(examples) < 100 and event_changed:
                    examples.append(
                        {
                            "event_id": row["event_id"],
                            "message_sha256_prefix": row["message_sha256"][:12],
                            "text_chars": len(text),
                            "archive_db": text_item["archive_db"],
                            "text_status": text_item["text_status"],
                            "had_chunk": chunk_row is not None,
                            "linked": bool(row["customer_id"]),
                        }
                    )
            if counters["events_updated"] or counters["chunks_updated"] or counters["chunks_created"]:
                store._rebuild_fts_indexes()  # noqa: SLF001 - required after bulk direct maintenance updates.
            con.commit()
        report = {
            "schema_version": MAIL_STAGE2_ENRICH_SCHEMA_VERSION,
            "mode": "apply" if config.apply else "dry_run",
            "timeline_db_path": str(db_path),
            "archive_db_paths": [str(path) for path in config.archive_db_paths],
            "tenant_id": config.tenant_id,
            "text_max_chars": config.text_max_chars,
            "counts": dict(counters),
            "archive_report": archive_index["report"],
            "examples": examples,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "safety": {
                "prod_write": False,
                "crm_write": False,
                "llm_calls_total": 0,
                "mail_chunks_allowed_for_bot": 0,
                "mail_chunks_require_manager_review": True,
            },
        }
    (config.out_dir / "mail_stage2_enrich_report.json").write_text(json_dumps(report), encoding="utf-8")
    return report


def _load_stage2_event_rows(con: sqlite3.Connection, *, tenant_id: str) -> list[Mapping[str, Any]]:
    superseded_filter = "AND (superseded_by IS NULL OR superseded_by = '')" if _has_column(con, "timeline_events", "superseded_by") else ""
    rows = []
    for row in con.execute(
        f"""
        SELECT event_id, tenant_id, customer_id, opportunity_id, event_type, event_at,
               source_system, source_id, source_ref, direction, match_status, confidence,
               importance, subject, text_preview, summary, created_at, record_json
        FROM timeline_events
        WHERE tenant_id = ?
          AND source_system = ?
          {superseded_filter}
        ORDER BY event_at ASC, event_id ASC
        """,
        (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
    ):
        payload = json_loads(row["record_json"])
        record = payload.get("record") if isinstance(payload.get("record"), Mapping) else {}
        rows.append({**dict(row), "payload": payload, "message_sha256": str(record.get("message_sha256") or row["source_id"])})
    return rows


def _load_stage2_chunks_by_event(con: sqlite3.Connection) -> Mapping[str, Mapping[str, Any]]:
    rows = con.execute(
        """
        SELECT chunk_id, event_id, record_json
        FROM bot_context_chunks
        WHERE source_system = ?
          AND event_id IS NOT NULL
        """,
        (MAIL_STAGE2_INGEST_SOURCE_SYSTEM,),
    ).fetchall()
    return {str(row["event_id"]): dict(row) for row in rows}


def _load_archive_text_index(
    archive_db_paths: Sequence[Path],
    *,
    target_shas: set[str],
    text_max_chars: int,
) -> Mapping[str, Any]:
    texts: dict[str, Mapping[str, Any]] = {}
    archive_reports: list[Mapping[str, Any]] = []
    for archive_db in archive_db_paths:
        archive = Path(archive_db).expanduser().resolve(strict=True)
        matched = 0
        matched_with_text = 0
        with sqlite3.connect(f"{archive.as_uri()}?mode=ro&immutable=1", uri=True) as con:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA query_only=ON")
            for row in con.execute(
                """
                SELECT sha256, extracted_text_path, extracted_text_chars
                FROM messages
                WHERE sha256 IS NOT NULL AND sha256 != ''
                """
            ):
                sha = str(row["sha256"])
                if sha not in target_shas:
                    continue
                matched += 1
                if sha in texts and texts[sha].get("text"):
                    continue
                text, status = read_safe_stage2_event_text(row["extracted_text_path"], max_chars=text_max_chars)
                if text:
                    matched_with_text += 1
                    texts[sha] = {
                        "message_sha256": sha,
                        "text": text,
                        "text_chars": len(text),
                        "text_status": status,
                        "archive_db": str(archive),
                        "extracted_text_chars": int(row["extracted_text_chars"] or 0),
                    }
        if matched:
            archive_reports.append(
                {
                    "archive_db": str(archive),
                    "matched": matched,
                    "matched_with_text": matched_with_text,
                }
            )
    return {
        "texts": texts,
        "report": {
            "archives_scanned": len(archive_db_paths),
            "target_shas": len(target_shas),
            "matched_with_text": len(texts),
            "missing_text": len(target_shas - set(texts)),
            "archives_with_matches": archive_reports,
        },
    }


def _update_event_row(
    con: sqlite3.Connection,
    *,
    row: Mapping[str, Any],
    text_item: Mapping[str, Any],
    text_max_chars: int,
) -> bool:
    payload = dict(row["payload"])
    record = dict(payload.get("record") or {})
    text = str(text_item["text"])
    summary = compact_text(text, limit=500) or row["summary"]
    text_preview = compact_text(text, limit=240) or row["text_preview"]
    record.update(
        {
            "full_clean_text": text,
            "full_clean_text_chars": len(text),
            "stage2_enrich_schema_version": MAIL_STAGE2_ENRICH_SCHEMA_VERSION,
            "stage2_enrich_text_status": text_item["text_status"],
            "stage2_enrich_archive_db": text_item["archive_db"],
            "stage2_enrich_text_max_chars": text_max_chars,
        }
    )
    payload.update({"record": record, "text_preview": text_preview, "summary": summary})
    record_hash = stable_digest(payload)
    content_key = timeline_email_content_key(
        tenant_id=str(row["tenant_id"]),
        customer_id=row["customer_id"],
        event_type=str(row["event_type"]),
        event_at=str(row["event_at"]),
        subject=row["subject"],
        summary=summary,
    )
    existing_hash = con.execute(
        "SELECT record_hash FROM timeline_events WHERE event_id = ?",
        (row["event_id"],),
    ).fetchone()[0]
    if existing_hash == record_hash:
        return False
    con.execute(
        """
        UPDATE timeline_events
        SET text_preview = ?, summary = ?, content_key = ?, record_json = ?, record_hash = ?
        WHERE event_id = ?
        """,
        (text_preview, summary, content_key, json_dumps(payload), record_hash, row["event_id"]),
    )
    return True


def _update_chunk_row(con: sqlite3.Connection, *, row: Mapping[str, Any], text_item: Mapping[str, Any]) -> bool:
    payload = json_loads(row["record_json"])
    text = str(text_item["text"])
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "stage2_enrich_schema_version": MAIL_STAGE2_ENRICH_SCHEMA_VERSION,
            "stage2_enrich_text_status": text_item["text_status"],
        }
    )
    payload.update(
        {
            "text": text,
            "summary": compact_text(text, limit=500),
            "allowed_for_bot": False,
            "requires_manager_review": True,
            "metadata": metadata,
        }
    )
    record_hash = stable_digest(scrub_timeline_persisted_json(payload))
    existing_hash = con.execute(
        "SELECT record_hash FROM bot_context_chunks WHERE chunk_id = ?",
        (row["chunk_id"],),
    ).fetchone()[0]
    if existing_hash == record_hash:
        return False
    con.execute(
        """
        UPDATE bot_context_chunks
        SET allowed_for_bot = 0, requires_manager_review = 1, record_json = ?, record_hash = ?
        WHERE chunk_id = ?
        """,
        (json_dumps(payload), record_hash, row["chunk_id"]),
    )
    return True


def _create_missing_chunk(
    store: CustomerTimelineSQLiteStore,
    *,
    row: Mapping[str, Any],
    text_item: Mapping[str, Any],
) -> str:
    text = str(text_item["text"])
    event_at = parse_datetime(str(row["event_at"]), "event_at")
    chunk = BotContextChunk(
        tenant_id=str(row["tenant_id"]),
        customer_id=str(row["customer_id"]),
        opportunity_id=row.get("opportunity_id"),
        event_id=str(row["event_id"]),
        source_ref=str(row.get("source_ref") or row["source_id"]),
        source_system=MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
        chunk_type="email_message",
        text=text,
        summary=compact_text(text, limit=500),
        event_at=event_at,
        freshness_score=0.7,
        relevance_tags=("email", "manager_only"),
        allowed_for_bot=False,
        requires_manager_review=True,
        metadata={
            "stage2_enrich_schema_version": MAIL_STAGE2_ENRICH_SCHEMA_VERSION,
            "stage2_enrich_text_status": text_item["text_status"],
            "message_sha256": row["message_sha256"],
        },
        created_at=parse_datetime(str(row["created_at"]), "created_at"),
    )
    return store.upsert_bot_context_chunk(chunk, actor="mail_stage2_existing_enrich").status


def _reject_prod_path(path: Path) -> None:
    resolved = str(path.expanduser().resolve(strict=False))
    if "customer_timeline_prod_" in resolved:
        raise ValueError(f"refusing to enrich prod timeline path: {resolved}")


def _has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in con.execute(f"PRAGMA table_info({table})").fetchall())


__all__ = [
    "MAIL_STAGE2_ENRICH_SCHEMA_VERSION",
    "MailStage2ExistingEnrichConfig",
    "enrich_existing_mail_stage2_from_archives",
]
