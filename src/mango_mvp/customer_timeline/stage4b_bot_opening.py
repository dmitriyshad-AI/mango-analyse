from __future__ import annotations

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.canonical_readonly_import import infer_offline_brand
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.mail_stage2_ingest import MAIL_STAGE2_INGEST_SOURCE_SYSTEM
from mango_mvp.customer_timeline.safety import (
    guard_customer_timeline_output_path,
    guard_customer_timeline_writable_path,
)
from mango_mvp.customer_timeline.source_policy import (
    CHANNEL_HISTORY_SOURCE_SYSTEMS,
    MANGO_PROCESSED_SOURCE_SYSTEM,
    TELEGRAM_HISTORY_SOURCE_SYSTEM,
    WAPPI_MAX_SOURCE_SYSTEM,
    WAPPI_TELEGRAM_SOURCE_SYSTEM,
    is_non_contentful_call_record,
)
from mango_mvp.customer_timeline.store import (
    authoritative_exact_identity_rows,
    json_dumps,
    json_loads,
    open_family_identity_conflict_customer_ids,
    scrub_timeline_persisted_json,
)


STAGE4B_BOT_OPENING_SCHEMA_VERSION = "stage4b_rich_bot_opening_v3"
STAGE4B_OPENING_POLICY_VERSION = "e4b_owner_policy_linked_rich_context_v3"
MANGO_CALL_CHUNK_TYPE = "mango_call_summary"
OPENABLE_SOURCE_SYSTEMS = (
    MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
    MANGO_PROCESSED_SOURCE_SYSTEM,
    TELEGRAM_HISTORY_SOURCE_SYSTEM,
    WAPPI_TELEGRAM_SOURCE_SYSTEM,
    WAPPI_MAX_SOURCE_SYSTEM,
)
OPENABLE_CHANNEL_SOURCE_SYSTEMS = tuple(sorted(CHANNEL_HISTORY_SOURCE_SYSTEMS))
_OPEN_CONFLICT_CUSTOMERS_TEMP_TABLE = "temp_stage4b_open_conflict_customers"
_CLIENT_UNSAFE_MAIL_CHUNKS_TEMP_TABLE = "temp_stage4b_client_unsafe_mail_chunks"
_CLIENT_SAFE_MAIL_CHUNKS_TEMP_TABLE = "temp_stage4b_client_safe_mail_chunks"
_MAIL_OUTPUT_ALLOWED_CLIENT_UNSAFE_REASONS = frozenset(
    {
        "sensitive_money",
        "sensitive_tax",
        "sensitive_contract",
    }
)
_MAIL_OUTPUT_FORBIDDEN_CLIENT_UNSAFE_REASONS = frozenset(
    {
        "manager_action_required",
        "has_manager_note",
        "sensitive_credentials",
        "sensitive_bank_requisites",
        "sensitive_payment_details",
        "sensitive_personal_data",
        "sensitive_document_data",
        "sensitive_medical",
    }
)
_MAIL_OUTPUT_SECRET_TAGS = frozenset(
    {
        "sensitive_credentials",
        "sensitive_bank_requisites",
        "sensitive_payment_details",
        "sensitive_personal_data",
        "sensitive_document_data",
        "sensitive_medical",
    }
)


@dataclass(frozen=True)
class Stage4BBotOpeningConfig:
    timeline_db_path: Path
    allowed_root: Path
    out_dir: Path
    tenant_id: str = "foton"
    apply: bool = True
    allow_test_paths: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_db_path", Path(self.timeline_db_path).expanduser())
        object.__setattr__(self, "allowed_root", Path(self.allowed_root).expanduser())
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())


def run_stage4b_bot_opening(config: Stage4BBotOpeningConfig) -> Mapping[str, Any]:
    started = time.monotonic()
    db_path = guard_customer_timeline_writable_path(
        guard_customer_timeline_output_path(config.timeline_db_path, config.allowed_root)
    )
    _assert_stage4b_staging_path(db_path, config.allowed_root, allow_test_paths=config.allow_test_paths)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        before = _metrics(con, tenant_id=config.tenant_id)
        open_conflict_customers = _prepare_open_conflict_customer_ids(con, tenant_id=config.tenant_id)
        client_safe_mail_chunks = _prepare_client_safe_mail_chunk_ids(con, tenant_id=config.tenant_id)
        client_unsafe_mail_chunks = _prepare_client_unsafe_mail_chunk_ids(con, tenant_id=config.tenant_id)
        plan = _load_opening_plan(con, tenant_id=config.tenant_id)
        openable_chunk_ids = frozenset(str(row["chunk_id"]) for row in plan["rows"])
        report: dict[str, Any] = {
            "schema_version": STAGE4B_BOT_OPENING_SCHEMA_VERSION,
            "mode": "apply" if config.apply else "dry_run",
            "timeline_db_path": str(db_path),
            "tenant_id": config.tenant_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "safety": {
                "prod_write": False,
                "crm_write": False,
                "llm_calls_total": 0,
                "none_customer_groups_actioned": 0,
            },
            "before": before,
            "plan": plan["summary"],
            "open_conflict_customers_indexed": open_conflict_customers,
            "client_safe_mail_chunks_indexed": client_safe_mail_chunks,
            "client_unsafe_mail_chunks_indexed": client_unsafe_mail_chunks,
        }
        if config.apply:
            report["apply"] = _apply_opening_plan(
                con,
                plan["rows"],
                openable_chunk_ids=openable_chunk_ids,
                tenant_id=config.tenant_id,
            )
        else:
            report["apply"] = {"chunks_updated": 0, "dry_run": True}
        opened_non_contentful = _opened_mango_non_contentful_count(con, tenant_id=config.tenant_id)
        if config.apply and opened_non_contentful != 0:
            con.rollback()
            raise RuntimeError("stage4b refused to leave non-contentful Mango calls open for bot memory")
        if config.apply:
            con.commit()
        after = _metrics(con, tenant_id=config.tenant_id)
        report["after"] = after
        opened_chunk_ids = _opened_chunk_ids(con, tenant_id=config.tenant_id)
        report["final_checks"] = {
            "quick_check": con.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_check_rows": len(con.execute("PRAGMA foreign_key_check").fetchall()),
            "candidate_review_violations_after": len(openable_chunk_ids - opened_chunk_ids),
            "opened_disallowed_identity_after": len(opened_chunk_ids - openable_chunk_ids),
            "opened_mango_processed_non_strong_after": _opened_mango_non_strong_count(con, tenant_id=config.tenant_id),
            "opened_mango_processed_non_contentful_after": opened_non_contentful,
            "opened_unknown_brand_non_call_after": _opened_unknown_brand_count(
                con,
                tenant_id=config.tenant_id,
                include_mango_processed=False,
            ),
            "opened_mango_processed_unknown_brand_after": _opened_unknown_brand_count(
                con,
                tenant_id=config.tenant_id,
                only_mango_processed=True,
            ),
        }
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (config.out_dir / "stage4b_bot_opening_report.json").write_text(
            json_dumps(report),
            encoding="utf-8",
        )
        return report
    finally:
        con.close()


def _load_opening_plan(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, Any]:
    source_placeholders = ",".join("?" for _ in OPENABLE_SOURCE_SYSTEMS)
    raw_rows = con.execute(
        f"""
        SELECT
          c.chunk_id,
          c.customer_id,
          c.source_system,
          c.record_json,
          c.record_hash,
          c.allowed_for_bot,
          c.requires_manager_review,
          e.record_json AS event_record_json,
          e.match_status,
          ci.identity_status
        FROM bot_context_chunks c
        JOIN timeline_events e
          ON e.tenant_id = c.tenant_id AND e.event_id = c.event_id AND e.source_system = c.source_system
        JOIN customer_identities ci ON ci.tenant_id = c.tenant_id AND ci.customer_id = c.customer_id
        WHERE c.tenant_id = ?
          AND c.source_system IN ({source_placeholders})
          AND (
            c.source_system = ?
            OR (
              json_type(e.record_json, '$.metadata.brand_context_authorized') = 'true'
              AND json_type(c.record_json, '$.metadata.brand_context_authorized') = 'true'
            )
          )
          AND c.superseded_by IS NULL
          AND e.superseded_by IS NULL
          AND (
            ci.identity_status = 'strong'
            OR (
              c.source_system = ?
              AND ci.identity_status = 'partial'
              AND e.match_status = 'strong_unique'
            )
            OR (
              c.source_system IN (?, ?)
              AND ci.identity_status = 'partial'
              AND e.match_status IN ('manual', 'strong_unique')
            )
          )
          AND c.customer_id IS NOT NULL
          AND c.customer_id != ''
          AND c.customer_id = e.customer_id
              AND trim(coalesce(json_extract(c.record_json, '$.text'), '')) != ''
              AND NOT EXISTS (
                SELECT 1
                FROM temp_stage4b_open_conflict_customers occ
                WHERE occ.customer_id = c.customer_id
              )
              AND NOT EXISTS (
                SELECT 1
                FROM temp_stage4b_client_unsafe_mail_chunks unsafe
                WHERE unsafe.chunk_id = c.chunk_id
              )
              AND (
                c.source_system != ?
                OR EXISTS (
                  SELECT 1
                  FROM temp_stage4b_client_safe_mail_chunks safe_mail
                  WHERE safe_mail.chunk_id = c.chunk_id
                )
              )
          AND (
            (c.source_system = ? AND e.match_status = 'strong_unique')
            OR (c.source_system = ? AND e.match_status = 'strong_unique')
            OR (c.source_system = ? AND c.chunk_type = ? AND e.match_status = 'strong_unique')
            OR (c.source_system IN (?, ?) AND e.match_status IN ('manual', 'strong_unique'))
          )
          AND (
            c.source_system NOT IN (?, ?)
            OR (
              trim(coalesce(json_extract(e.record_json, '$.metadata.contact_id'), '')) != ''
              AND c.chunk_type='channel_message'
            )
          )
        ORDER BY c.event_at, c.chunk_id
        """,
        (
            tenant_id,
            *OPENABLE_SOURCE_SYSTEMS,
            MANGO_PROCESSED_SOURCE_SYSTEM,
            MANGO_PROCESSED_SOURCE_SYSTEM,
            WAPPI_TELEGRAM_SOURCE_SYSTEM,
            WAPPI_MAX_SOURCE_SYSTEM,
            MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
            MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
            TELEGRAM_HISTORY_SOURCE_SYSTEM,
            MANGO_PROCESSED_SOURCE_SYSTEM,
            MANGO_CALL_CHUNK_TYPE,
            WAPPI_TELEGRAM_SOURCE_SYSTEM,
            WAPPI_MAX_SOURCE_SYSTEM,
            WAPPI_TELEGRAM_SOURCE_SYSTEM,
            WAPPI_MAX_SOURCE_SYSTEM,
        ),
    ).fetchall()
    rows: list[sqlite3.Row] = []
    amo_contact_owners = _authoritative_amo_contact_owners(con, tenant_id=tenant_id)
    sensitivity_counts: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()
    unknown_brand_chunks = 0
    non_contentful_mango_call_chunks = 0
    already_open = 0
    to_update = 0
    for row in raw_rows:
        source_system = str(row["source_system"])
        payload = json_loads(row["record_json"])
        event_payload = json_loads(row["event_record_json"]) if row["event_record_json"] else {}
        if source_system in {WAPPI_TELEGRAM_SOURCE_SYSTEM, WAPPI_MAX_SOURCE_SYSTEM}:
            metadata = event_payload.get("metadata") if isinstance(event_payload.get("metadata"), Mapping) else {}
            contact_id = str(metadata.get("contact_id") or "").strip()
            if amo_contact_owners.get(contact_id) != str(row["customer_id"]):
                continue
        if source_system == MANGO_PROCESSED_SOURCE_SYSTEM and is_non_contentful_call_record(event_payload):
            non_contentful_mango_call_chunks += 1
            continue
        brand = _content_brand(payload, event_payload)
        brand_counts[brand] += 1
        if source_system != MANGO_PROCESSED_SOURCE_SYSTEM and brand not in {"foton", "unpk"}:
            unknown_brand_chunks += 1
            continue
        rows.append(row)
        tags = _sensitivity_tags(payload)
        sensitivity_counts.update(tags)
        if int(row["allowed_for_bot"]) == 1 and int(row["requires_manager_review"]) == 0:
            already_open += 1
        else:
            to_update += 1
    skipped = {
        **_skipped_counts(con, tenant_id=tenant_id),
        "unknown_brand_chunks": unknown_brand_chunks,
        "non_contentful_mango_call_chunks": non_contentful_mango_call_chunks,
        "client_unsafe_mail_chunks": _client_unsafe_mail_chunk_count(con),
        "mail_chunks_not_allowed_by_output_gate": _mail_without_client_safe_fact_count(con),
    }
    return {
        "rows": rows,
        "summary": {
            "candidate_chunks": len(rows),
            "already_open": already_open,
            "chunks_to_open": to_update,
            "policy_version": STAGE4B_OPENING_POLICY_VERSION,
            "sensitivity_tag_counts": dict(sensitivity_counts),
            "content_brand_counts": dict(brand_counts),
            "source_system_counts": dict(Counter(str(row["source_system"]) for row in rows)),
            "skipped": skipped,
        },
    }


def _apply_opening_plan(
    con: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
    *,
    openable_chunk_ids: frozenset[str],
    tenant_id: str,
) -> Mapping[str, Any]:
    counters: Counter[str] = Counter({"chunks_updated": 0, "chunks_already_open": 0})
    sensitivity_counts: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()
    for row in rows:
        source_system = str(row["source_system"])
        payload = json_loads(row["record_json"])
        event_payload = json_loads(row["event_record_json"]) if row["event_record_json"] else {}
        payload["allowed_for_bot"] = True
        payload["requires_manager_review"] = False
        metadata = dict(payload.get("metadata") or {})
        brand = _content_brand(payload, event_payload)
        brand_counts[brand] += 1
        tags = _opening_tags(payload, metadata=metadata, source_system=source_system, brand=brand)
        sensitivity_counts.update(tags)
        payload["relevance_tags"] = list(tags)
        metadata.update(
            {
                "client_safe": False,
                "client_safe_reason": "not_client_safe_rich_memory_opened_only_for_internal_bot_context",
                "client_safe_policy_version": "cs_v1",
                "bot_memory_allowed": True,
                "bot_memory_allowed_reason": _opening_reason(source_system),
                "bot_memory_policy_version": STAGE4B_OPENING_POLICY_VERSION,
                "content_brand": brand,
                "memory_status": "usable_memory",
                "sensitivity_tags": list(tags),
                "e4b_bot_opening": {
                    "policy_version": STAGE4B_OPENING_POLICY_VERSION,
                    "opened": True,
                    "reason": _opening_reason(source_system),
                    "source_system": source_system,
                    "match_status": str(row["match_status"]),
                },
            }
        )
        payload["metadata"] = metadata
        record_hash = stable_digest(scrub_timeline_persisted_json(payload))
        if (
            int(row["allowed_for_bot"]) == 1
            and int(row["requires_manager_review"]) == 0
            and row["record_hash"] == record_hash
        ):
            counters["chunks_already_open"] += 1
            continue
        con.execute(
            """
            UPDATE bot_context_chunks
            SET allowed_for_bot = 1,
                requires_manager_review = 0,
                record_json = ?,
                record_hash = ?
            WHERE tenant_id = ?
              AND chunk_id = ?
            """,
            (json_dumps(payload), record_hash, tenant_id, row["chunk_id"]),
        )
        counters["chunks_updated"] += 1
    retracted = _retract_non_openable_chunks(
        con,
        candidate_chunk_ids=openable_chunk_ids,
        tenant_id=tenant_id,
    )
    return {
        **dict(counters),
        **retracted,
        "sensitivity_tag_counts": dict(sensitivity_counts),
        "content_brand_counts": dict(brand_counts),
        "source_system_counts": dict(Counter(str(row["source_system"]) for row in rows)),
        "policy_version": STAGE4B_OPENING_POLICY_VERSION,
    }


def _metrics(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, int]:
    metrics = {
        "mail_stage2_chunks_total": _scalar(
            con,
            "SELECT count(*) FROM bot_context_chunks WHERE tenant_id = ? AND source_system = ?",
            (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
        ),
        "mail_stage2_chunks_active": _scalar(
            con,
            """
            SELECT count(*)
            FROM bot_context_chunks
            WHERE tenant_id = ? AND source_system = ? AND superseded_by IS NULL
            """,
            (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
        ),
        "mail_stage2_chunks_bot_visible": _scalar(
            con,
            """
            SELECT count(*)
            FROM bot_context_chunks
            WHERE tenant_id = ?
              AND source_system = ?
              AND superseded_by IS NULL
              AND allowed_for_bot = 1
              AND requires_manager_review = 0
            """,
            (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
        ),
        "mail_stage2_unmatched_events": _scalar(
            con,
            """
            SELECT count(*)
            FROM timeline_events
            WHERE tenant_id = ?
              AND source_system = ?
              AND superseded_by IS NULL
              AND (customer_id IS NULL OR customer_id = '' OR match_status != 'strong_unique')
            """,
            (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
        ),
    }
    for source_system in OPENABLE_CHANNEL_SOURCE_SYSTEMS:
        prefix = source_system.replace("_", "-")
        metrics[f"{prefix}_chunks_total"] = _scalar(
            con,
            "SELECT count(*) FROM bot_context_chunks WHERE tenant_id = ? AND source_system = ?",
            (tenant_id, source_system),
        )
        metrics[f"{prefix}_chunks_bot_visible"] = _scalar(
            con,
            """
            SELECT count(*)
            FROM bot_context_chunks
            WHERE tenant_id = ?
              AND source_system = ?
              AND superseded_by IS NULL
              AND allowed_for_bot = 1
              AND requires_manager_review = 0
            """,
            (tenant_id, source_system),
        )
    metrics["mango_processed_summary_chunks_total"] = _scalar(
        con,
        "SELECT count(*) FROM bot_context_chunks WHERE tenant_id = ? AND source_system = ?",
        (tenant_id, MANGO_PROCESSED_SOURCE_SYSTEM),
    )
    metrics["mango_processed_summary_chunks_bot_visible"] = _scalar(
        con,
        """
        SELECT count(*)
        FROM bot_context_chunks
        WHERE tenant_id = ?
          AND source_system = ?
          AND superseded_by IS NULL
          AND allowed_for_bot = 1
          AND requires_manager_review = 0
        """,
        (tenant_id, MANGO_PROCESSED_SOURCE_SYSTEM),
    )
    metrics["mango_processed_summary_unmatched_or_ambiguous_events"] = _scalar(
        con,
        """
        SELECT count(*)
        FROM timeline_events
        WHERE tenant_id = ?
          AND source_system = ?
          AND superseded_by IS NULL
          AND match_status != 'strong_unique'
        """,
        (tenant_id, MANGO_PROCESSED_SOURCE_SYSTEM),
    )
    return metrics


def _skipped_counts(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, int]:
    skipped = {
        "unmatched_or_not_strong_events": _scalar(
            con,
            """
            SELECT count(*)
            FROM timeline_events
            WHERE tenant_id = ?
              AND source_system = ?
              AND superseded_by IS NULL
              AND (customer_id IS NULL OR customer_id = '' OR match_status != 'strong_unique')
            """,
            (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
        ),
        "empty_active_chunks": _scalar(
            con,
            """
            SELECT count(*)
            FROM bot_context_chunks
            WHERE tenant_id = ?
              AND source_system = ?
              AND superseded_by IS NULL
              AND trim(coalesce(json_extract(record_json, '$.text'), '')) = ''
            """,
            (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
        ),
        "superseded_chunks": _scalar(
            con,
            """
            SELECT count(*)
            FROM bot_context_chunks
            WHERE tenant_id = ?
              AND source_system = ?
              AND superseded_by IS NOT NULL
              AND superseded_by != ''
            """,
            (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
        ),
    }
    skipped["channel_events_not_openable_identity"] = _scalar(
        con,
        """
        SELECT count(*)
        FROM timeline_events
        WHERE tenant_id = ?
          AND source_system IN (?, ?, ?, ?)
          AND superseded_by IS NULL
          AND (
            customer_id IS NULL
            OR customer_id = ''
            OR EXISTS (
              SELECT 1
              FROM temp_stage4b_open_conflict_customers occ
              WHERE occ.customer_id = timeline_events.customer_id
            )
            OR (
              source_system = ?
              AND match_status != 'strong_unique'
            )
            OR (
              source_system = ?
              AND match_status != 'strong_unique'
            )
            OR (
              source_system IN (?, ?)
              AND match_status NOT IN ('manual', 'strong_unique')
            )
          )
        """,
        (
            tenant_id,
            TELEGRAM_HISTORY_SOURCE_SYSTEM,
            MANGO_PROCESSED_SOURCE_SYSTEM,
            WAPPI_TELEGRAM_SOURCE_SYSTEM,
            WAPPI_MAX_SOURCE_SYSTEM,
            TELEGRAM_HISTORY_SOURCE_SYSTEM,
            MANGO_PROCESSED_SOURCE_SYSTEM,
            WAPPI_TELEGRAM_SOURCE_SYSTEM,
            WAPPI_MAX_SOURCE_SYSTEM,
        ),
    )
    return skipped


def _opening_reason(source_system: str) -> str:
    if source_system == MAIL_STAGE2_INGEST_SOURCE_SYSTEM:
        return "e4b_owner_policy_linked_strong_unique_non_empty_mail_stage2"
    if source_system == TELEGRAM_HISTORY_SOURCE_SYSTEM:
        return "e4b_owner_policy_linked_strong_unique_non_empty_telegram_history"
    if source_system == MANGO_PROCESSED_SOURCE_SYSTEM:
        return "e4b_owner_policy_linked_strong_unique_non_empty_mango_processed_summary"
    if source_system in {WAPPI_TELEGRAM_SOURCE_SYSTEM, WAPPI_MAX_SOURCE_SYSTEM}:
        return "e4b_owner_policy_linked_manual_non_empty_wappi_history"
    return "e4b_owner_policy_linked_non_empty_rich_context"


def _opening_tags(
    payload: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    source_system: str,
    brand: str,
) -> tuple[str, ...]:
    brand_tags = (brand,) if brand in {"foton", "unpk"} else ("brand_unknown",)
    source_tags = (source_system,)
    if source_system == MAIL_STAGE2_INGEST_SOURCE_SYSTEM:
        source_tags = ("email", MAIL_STAGE2_INGEST_SOURCE_SYSTEM)
    elif source_system == MANGO_PROCESSED_SOURCE_SYSTEM:
        source_tags = ("call", MANGO_PROCESSED_SOURCE_SYSTEM)
    elif source_system in CHANNEL_HISTORY_SOURCE_SYSTEMS:
        source_tags = ("channel", source_system)
    generated_tags = {
        "manager_review", "brand_unknown", "foton", "unpk", "bot_visible", *source_tags,
    }
    return tuple(
        dict.fromkeys(
            (
                *brand_tags,
                *(tag for tag in _metadata_tags(metadata) if tag not in generated_tags),
                *_sensitivity_tags(payload),
                "bot_visible",
                *source_tags,
            )
        )
    )


def _opened_chunk_ids(con: sqlite3.Connection, *, tenant_id: str) -> frozenset[str]:
    source_placeholders = ",".join("?" for _ in OPENABLE_SOURCE_SYSTEMS)
    return frozenset(
        str(row[0])
        for row in con.execute(
        f"""
        SELECT c.chunk_id
        FROM bot_context_chunks c
        WHERE c.tenant_id = ?
          AND c.source_system IN ({source_placeholders})
          AND c.superseded_by IS NULL
          AND c.allowed_for_bot = 1
          AND c.requires_manager_review = 0
        """,
        (tenant_id, *OPENABLE_SOURCE_SYSTEMS),
        )
    )


def _opened_mango_non_strong_count(con: sqlite3.Connection, *, tenant_id: str) -> int:
    return _scalar(
        con,
        """
        SELECT count(*)
        FROM bot_context_chunks c
        JOIN timeline_events e ON e.event_id = c.event_id
        WHERE c.tenant_id = ?
          AND c.source_system = ?
          AND c.superseded_by IS NULL
          AND e.superseded_by IS NULL
          AND c.allowed_for_bot = 1
          AND c.requires_manager_review = 0
          AND e.match_status != 'strong_unique'
        """,
        (tenant_id, MANGO_PROCESSED_SOURCE_SYSTEM),
    )


def _opened_mango_non_contentful_count(con: sqlite3.Connection, *, tenant_id: str) -> int:
    rows = con.execute(
        """
        SELECT e.record_json
        FROM bot_context_chunks c
        JOIN timeline_events e
          ON e.tenant_id=c.tenant_id AND e.event_id=c.event_id
         AND e.customer_id=c.customer_id AND e.source_system=c.source_system
        WHERE c.tenant_id=? AND c.source_system=?
          AND c.superseded_by IS NULL AND e.superseded_by IS NULL
          AND c.allowed_for_bot=1 AND c.requires_manager_review=0
        """,
        (tenant_id, MANGO_PROCESSED_SOURCE_SYSTEM),
    ).fetchall()
    return sum(1 for row in rows if is_non_contentful_call_record(json_loads(row["record_json"])))


def _opened_unknown_brand_count(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    include_mango_processed: bool = True,
    only_mango_processed: bool = False,
) -> int:
    source_placeholders = ",".join("?" for _ in OPENABLE_SOURCE_SYSTEMS)
    source_filter = ""
    extra_params: tuple[str, ...] = ()
    if only_mango_processed:
        source_filter = "AND c.source_system = ?"
        extra_params = (MANGO_PROCESSED_SOURCE_SYSTEM,)
    elif not include_mango_processed:
        source_filter = "AND c.source_system != ?"
        extra_params = (MANGO_PROCESSED_SOURCE_SYSTEM,)
    rows = con.execute(
        f"""
        SELECT c.record_json, e.record_json AS event_record_json
        FROM bot_context_chunks c
        JOIN timeline_events e ON e.event_id = c.event_id
        WHERE c.tenant_id = ?
          AND c.source_system IN ({source_placeholders})
          AND c.superseded_by IS NULL
          AND e.superseded_by IS NULL
          AND c.allowed_for_bot = 1
          AND c.requires_manager_review = 0
          {source_filter}
        """,
        (tenant_id, *OPENABLE_SOURCE_SYSTEMS, *extra_params),
    ).fetchall()
    return sum(
        1
        for row in rows
        if _content_brand(
            json_loads(row["record_json"]),
            json_loads(row["event_record_json"]) if row["event_record_json"] else {},
        )
        not in {"foton", "unpk"}
    )


def _retract_non_openable_chunks(
    con: sqlite3.Connection,
    *,
    candidate_chunk_ids: frozenset[str],
    tenant_id: str,
) -> Mapping[str, int]:
    rows = con.execute(
        f"""
        SELECT chunk_id, record_json, record_hash
        FROM bot_context_chunks
        WHERE tenant_id = ?
          AND source_system IN ({','.join('?' for _ in OPENABLE_SOURCE_SYSTEMS)})
          AND allowed_for_bot = 1
          AND requires_manager_review = 0
          AND (superseded_by IS NULL OR superseded_by = '')
        """,
        (tenant_id, *OPENABLE_SOURCE_SYSTEMS),
    ).fetchall()
    retracted = 0
    already_closed = 0
    for row in rows:
        chunk_id = str(row["chunk_id"])
        if chunk_id in candidate_chunk_ids:
            already_closed += 1
            continue
        payload = json_loads(row["record_json"])
        payload["allowed_for_bot"] = False
        payload["requires_manager_review"] = True
        tags = [tag for tag in payload.get("relevance_tags") or () if str(tag) != "bot_visible"]
        payload["relevance_tags"] = tags
        metadata = dict(payload.get("metadata") or {})
        metadata.update(
            {
                "bot_memory_allowed": False,
                "bot_memory_allowed_reason": "e4b_owner_policy_retracted_not_openable",
                "bot_memory_policy_version": STAGE4B_OPENING_POLICY_VERSION,
                "memory_status": "requires_manager_review",
                "e4b_bot_opening": {
                    "policy_version": STAGE4B_OPENING_POLICY_VERSION,
                    "opened": False,
                    "reason": "retracted_not_openable_after_identity_conflict_gate",
                },
            }
        )
        payload["metadata"] = metadata
        record_hash = stable_digest(scrub_timeline_persisted_json(payload))
        if row["record_hash"] == record_hash:
            already_closed += 1
            continue
        con.execute(
            """
            UPDATE bot_context_chunks
            SET allowed_for_bot = 0,
                requires_manager_review = 1,
                record_json = ?,
                record_hash = ?
            WHERE tenant_id = ?
              AND chunk_id = ?
            """,
            (json_dumps(payload), record_hash, tenant_id, chunk_id),
        )
        retracted += 1
    return {
        "chunks_retracted_not_openable": retracted,
        "chunks_kept_open": already_closed,
    }


def _sensitivity_tags(payload: Mapping[str, Any]) -> tuple[str, ...]:
    text = f"{payload.get('text') or ''} {payload.get('summary') or ''}".casefold()
    tags: list[str] = []
    if any(marker in text for marker in ("₽", "руб", "оплат", "стоим", "цен", "счет", "счёт", "квитанц", "возврат")):
        tags.append("money")
    if any(marker in text for marker in ("распис", "суббот", "воскрес", "заняти", "10.", "12.", "14.", "16.")):
        tags.append("schedule")
    if any(marker in text for marker in ("договор", "оферт", "персональн", "паспорт")):
        tags.append("document")
    if "@" in text:
        tags.append("email_address")
    return tuple(dict.fromkeys(tags or ("general_email",)))


def _prepare_open_conflict_customer_ids(con: sqlite3.Connection, *, tenant_id: str) -> int:
    con.execute(f"DROP TABLE IF EXISTS {_OPEN_CONFLICT_CUSTOMERS_TEMP_TABLE}")
    con.execute(
        f"""
        CREATE TEMP TABLE {_OPEN_CONFLICT_CUSTOMERS_TEMP_TABLE} (
          customer_id TEXT PRIMARY KEY
        )
        """
    )
    ids = open_family_identity_conflict_customer_ids(con, tenant_id)
    con.executemany(
        f"INSERT OR IGNORE INTO {_OPEN_CONFLICT_CUSTOMERS_TEMP_TABLE}(customer_id) VALUES (?)",
        ((customer_id,) for customer_id in sorted(ids)),
    )
    return _scalar(con, f"SELECT count(*) FROM {_OPEN_CONFLICT_CUSTOMERS_TEMP_TABLE}")


def _authoritative_amo_contact_owners(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, str]:
    owners: dict[str, str] = {}
    for row in authoritative_exact_identity_rows(con, tenant_id, link_types=("amo_contact_id",)):
        payload = json_loads(row["record_json"])
        if (
            payload.get("source_system") == "amocrm_snapshot"
            and int(row["owner_count"]) == 1
            and int(row["has_open_conflict"]) == 0
        ):
            owners[str(row["link_value"])] = str(row["customer_id"])
    return owners


def _prepare_client_unsafe_mail_chunk_ids(con: sqlite3.Connection, *, tenant_id: str) -> int:
    con.execute(f"DROP TABLE IF EXISTS {_CLIENT_UNSAFE_MAIL_CHUNKS_TEMP_TABLE}")
    con.execute(
        f"""
        CREATE TEMP TABLE {_CLIENT_UNSAFE_MAIL_CHUNKS_TEMP_TABLE} (
          chunk_id TEXT PRIMARY KEY
        )
        """
    )
    facts_exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_mail_event_facts'"
    ).fetchone()
    if facts_exists is None:
        return 0
    reason_placeholders = ",".join("?" for _ in _MAIL_OUTPUT_FORBIDDEN_CLIENT_UNSAFE_REASONS)
    tag_placeholders = ",".join("?" for _ in _MAIL_OUTPUT_SECRET_TAGS)
    con.execute(
        f"""
        INSERT OR IGNORE INTO {_CLIENT_UNSAFE_MAIL_CHUNKS_TEMP_TABLE}(chunk_id)
        SELECT c.chunk_id
        FROM bot_context_chunks c
        JOIN a2v3_mail_event_facts f ON f.event_id = c.event_id
        WHERE c.tenant_id = ?
          AND c.source_system = ?
          AND COALESCE(c.superseded_by, '') = ''
          AND (
            f.client_safe_reason IN ({reason_placeholders})
            OR EXISTS (
              SELECT 1
              FROM json_each(COALESCE(f.sensitivity_tags_json, '[]')) AS tag
              WHERE tag.value IN ({tag_placeholders})
            )
          )
        """,
        (
            tenant_id,
            MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
            *_MAIL_OUTPUT_FORBIDDEN_CLIENT_UNSAFE_REASONS,
            *_MAIL_OUTPUT_SECRET_TAGS,
        ),
    )
    return _client_unsafe_mail_chunk_count(con)


def _prepare_client_safe_mail_chunk_ids(con: sqlite3.Connection, *, tenant_id: str) -> int:
    con.execute(f"DROP TABLE IF EXISTS {_CLIENT_SAFE_MAIL_CHUNKS_TEMP_TABLE}")
    con.execute(
        f"""
        CREATE TEMP TABLE {_CLIENT_SAFE_MAIL_CHUNKS_TEMP_TABLE} (
          chunk_id TEXT PRIMARY KEY
        )
        """
    )
    facts_exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='a2v3_mail_event_facts'"
    ).fetchone()
    if facts_exists is None:
        return 0
    con.execute(
        f"""
        INSERT OR IGNORE INTO {_CLIENT_SAFE_MAIL_CHUNKS_TEMP_TABLE}(chunk_id)
        SELECT c.chunk_id
        FROM bot_context_chunks c
        JOIN a2v3_mail_event_facts f ON f.event_id = c.event_id
        WHERE c.tenant_id = ?
          AND c.source_system = ?
          AND COALESCE(c.superseded_by, '') = ''
          AND f.bot_visible = 1
          AND (
            f.client_safe = 1
            OR f.client_safe_reason IN (?, ?, ?)
          )
        """,
        (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM, *_MAIL_OUTPUT_ALLOWED_CLIENT_UNSAFE_REASONS),
    )
    return _client_safe_mail_chunk_count(con)


def _client_safe_mail_chunk_count(con: sqlite3.Connection) -> int:
    exists = con.execute(
        "SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name=?",
        (_CLIENT_SAFE_MAIL_CHUNKS_TEMP_TABLE,),
    ).fetchone()
    if exists is None:
        return 0
    return int(con.execute(f"SELECT count(*) FROM {_CLIENT_SAFE_MAIL_CHUNKS_TEMP_TABLE}").fetchone()[0])


def _mail_without_client_safe_fact_count(con: sqlite3.Connection) -> int:
    safe_exists = con.execute(
        "SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name=?",
        (_CLIENT_SAFE_MAIL_CHUNKS_TEMP_TABLE,),
    ).fetchone()
    if safe_exists is None:
        return 0
    return int(
        con.execute(
            f"""
            SELECT count(*)
            FROM bot_context_chunks c
            WHERE c.source_system = ?
              AND COALESCE(c.superseded_by, '') = ''
              AND NOT EXISTS (
                SELECT 1
                FROM {_CLIENT_SAFE_MAIL_CHUNKS_TEMP_TABLE} safe_mail
                WHERE safe_mail.chunk_id = c.chunk_id
              )
            """,
            (MAIL_STAGE2_INGEST_SOURCE_SYSTEM,),
        ).fetchone()[0]
    )


def _client_unsafe_mail_chunk_count(con: sqlite3.Connection) -> int:
    exists = con.execute(
        "SELECT 1 FROM sqlite_temp_master WHERE type='table' AND name=?",
        (_CLIENT_UNSAFE_MAIL_CHUNKS_TEMP_TABLE,),
    ).fetchone()
    if exists is None:
        return 0
    return int(con.execute(f"SELECT count(*) FROM {_CLIENT_UNSAFE_MAIL_CHUNKS_TEMP_TABLE}").fetchone()[0])


def _content_brand(payload: Mapping[str, Any], *extra_payloads: Mapping[str, Any]) -> str:
    payloads = (payload, *extra_payloads)
    for item in payloads:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        record = item.get("record") if isinstance(item.get("record"), Mapping) else {}
        for value in (
            item.get("customer_brand"),
            item.get("email_brand"),
            item.get("brand"),
            metadata.get("customer_brand") if isinstance(metadata, Mapping) else None,
            metadata.get("email_brand") if isinstance(metadata, Mapping) else None,
            metadata.get("brand") if isinstance(metadata, Mapping) else None,
            record.get("customer_brand") if isinstance(record, Mapping) else None,
            record.get("email_brand") if isinstance(record, Mapping) else None,
            record.get("brand") if isinstance(record, Mapping) else None,
        ):
            brand = str(value or "").strip().casefold()
            if brand in {"foton", "unpk"}:
                return brand
    inferred = str(
        infer_offline_brand(
            {
                "text": "\n".join(str(item.get("text") or item.get("text_preview") or "") for item in payloads),
                "summary": "\n".join(str(item.get("summary") or "") for item in payloads),
                "subject": "\n".join(
                    str(item.get("subject_full") or item.get("subject") or "") for item in payloads
                ),
            }
        )
        or ""
    ).strip().casefold()
    return inferred if inferred in {"foton", "unpk"} else "unknown"


def _metadata_tags(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    raw = metadata.get("sensitivity_tags")
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw if str(item))
    return ()


def _scalar(con: sqlite3.Connection, query: str, params: Sequence[Any] = ()) -> int:
    return int(con.execute(query, tuple(params)).fetchone()[0])


def _assert_stage4b_staging_path(path: Path, allowed_root: Path, *, allow_test_paths: bool) -> None:
    resolved_path = path.expanduser().resolve(strict=False)
    resolved_root = allowed_root.expanduser().resolve(strict=False)
    resolved = str(resolved_path)
    if allow_test_paths:
        return
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"stage4b timeline db must be under allowed_root: {resolved}") from exc
    parts = tuple(part.casefold() for part in resolved_path.parts)
    if not any(part == ".codex_local" and parts[index + 1] == "staging" for index, part in enumerate(parts[:-1])):
        raise ValueError(f"stage4b bot opening is restricted to .codex_local/staging paths: {resolved}")


__all__ = [
    "STAGE4B_BOT_OPENING_SCHEMA_VERSION",
    "STAGE4B_OPENING_POLICY_VERSION",
    "Stage4BBotOpeningConfig",
    "run_stage4b_bot_opening",
]
