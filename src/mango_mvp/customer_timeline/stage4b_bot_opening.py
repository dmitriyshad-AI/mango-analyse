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
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import json_dumps, json_loads, scrub_timeline_persisted_json


STAGE4B_BOT_OPENING_SCHEMA_VERSION = "stage4b_mail_bot_opening_v1"
STAGE4B_OPENING_POLICY_VERSION = "e4b_owner_policy_bot_knows_all_v1"


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
    db_path = guard_customer_timeline_output_path(config.timeline_db_path, config.allowed_root)
    _assert_stage4b_staging_path(db_path, config.allowed_root, allow_test_paths=config.allow_test_paths)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        before = _metrics(con, tenant_id=config.tenant_id)
        plan = _load_opening_plan(con, tenant_id=config.tenant_id)
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
        }
        if config.apply:
            report["apply"] = _apply_opening_plan(con, plan["rows"])
            con.commit()
        else:
            report["apply"] = {"chunks_updated": 0, "dry_run": True}
        after = _metrics(con, tenant_id=config.tenant_id)
        report["after"] = after
        report["final_checks"] = {
            "quick_check": con.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_check_rows": len(con.execute("PRAGMA foreign_key_check").fetchall()),
            "mail_stage2_review_violations_after": _scalar(
                con,
                """
                SELECT count(*)
                FROM bot_context_chunks c
                JOIN timeline_events e ON e.event_id = c.event_id
                WHERE c.tenant_id = ?
                  AND c.source_system = ?
                  AND c.superseded_by IS NULL
                  AND e.superseded_by IS NULL
                  AND e.match_status = 'strong_unique'
                  AND (c.customer_id IS NOT NULL AND c.customer_id != '')
                  AND trim(coalesce(json_extract(c.record_json, '$.text'), '')) != ''
                  AND (c.allowed_for_bot != 1 OR c.requires_manager_review != 0)
                """,
                (config.tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
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
    rows = con.execute(
        """
        SELECT
          c.chunk_id,
          c.record_json,
          c.record_hash,
          c.allowed_for_bot,
          c.requires_manager_review
        FROM bot_context_chunks c
        JOIN timeline_events e ON e.event_id = c.event_id
        WHERE c.tenant_id = ?
          AND c.source_system = ?
          AND c.superseded_by IS NULL
          AND e.superseded_by IS NULL
          AND e.match_status = 'strong_unique'
          AND c.customer_id IS NOT NULL
          AND c.customer_id != ''
          AND trim(coalesce(json_extract(c.record_json, '$.text'), '')) != ''
        ORDER BY c.event_at, c.chunk_id
        """,
        (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
    ).fetchall()
    sensitivity_counts: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()
    already_open = 0
    to_update = 0
    for row in rows:
        payload = json_loads(row["record_json"])
        tags = _sensitivity_tags(payload)
        sensitivity_counts.update(tags)
        brand = _content_brand(payload)
        brand_counts[brand] += 1
        if int(row["allowed_for_bot"]) == 1 and int(row["requires_manager_review"]) == 0:
            already_open += 1
        else:
            to_update += 1
    return {
        "rows": rows,
        "summary": {
            "candidate_chunks": len(rows),
            "already_open": already_open,
            "chunks_to_open": to_update,
            "policy_version": STAGE4B_OPENING_POLICY_VERSION,
            "sensitivity_tag_counts": dict(sensitivity_counts),
            "content_brand_counts": dict(brand_counts),
            "skipped": _skipped_counts(con, tenant_id=tenant_id),
        },
    }


def _apply_opening_plan(con: sqlite3.Connection, rows: Sequence[sqlite3.Row]) -> Mapping[str, Any]:
    counters: Counter[str] = Counter({"chunks_updated": 0, "chunks_already_open": 0})
    sensitivity_counts: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()
    for row in rows:
        payload = json_loads(row["record_json"])
        payload["allowed_for_bot"] = True
        payload["requires_manager_review"] = False
        metadata = dict(payload.get("metadata") or {})
        brand = _content_brand(payload)
        brand_counts[brand] += 1
        brand_tags = (brand,) if brand in {"foton", "unpk"} else ("brand_unknown",)
        tags = tuple(
            dict.fromkeys(
                (
                    *(tag for tag in _metadata_tags(metadata) if tag != "manager_review"),
                    *brand_tags,
                    *_sensitivity_tags(payload),
                    "bot_visible",
                    "email",
                    MAIL_STAGE2_INGEST_SOURCE_SYSTEM,
                )
            )
        )
        sensitivity_counts.update(tags)
        payload["relevance_tags"] = list(tags)
        metadata.update(
            {
                "client_safe": False,
                "client_safe_reason": "not_client_safe_raw_mail_memory_opened_only_for_internal_bot_context",
                "client_safe_policy_version": "cs_v1",
                "bot_memory_allowed": True,
                "bot_memory_allowed_reason": "e4b_owner_policy_linked_strong_unique_non_empty_mail_stage2",
                "bot_memory_policy_version": STAGE4B_OPENING_POLICY_VERSION,
                "content_brand": brand,
                "memory_status": "usable_memory",
                "sensitivity_tags": list(tags),
                "e4b_bot_opening": {
                    "policy_version": STAGE4B_OPENING_POLICY_VERSION,
                    "opened": True,
                    "reason": "linked_strong_unique_non_empty_mail_stage2",
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
            WHERE chunk_id = ?
            """,
            (json_dumps(payload), record_hash, row["chunk_id"]),
        )
        counters["chunks_updated"] += 1
    return {
        **dict(counters),
        "sensitivity_tag_counts": dict(sensitivity_counts),
        "content_brand_counts": dict(brand_counts),
        "policy_version": STAGE4B_OPENING_POLICY_VERSION,
    }


def _metrics(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, int]:
    return {
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


def _skipped_counts(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, int]:
    return {
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


def _content_brand(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    for value in (
        payload.get("customer_brand"),
        payload.get("email_brand"),
        payload.get("brand"),
        metadata.get("customer_brand") if isinstance(metadata, Mapping) else None,
        metadata.get("email_brand") if isinstance(metadata, Mapping) else None,
        metadata.get("brand") if isinstance(metadata, Mapping) else None,
    ):
        brand = str(value or "").strip().casefold()
        if brand in {"foton", "unpk"}:
            return brand
    inferred = str(
        infer_offline_brand(
            {
                "text": payload.get("text"),
                "summary": payload.get("summary"),
                "subject": payload.get("subject_full") or payload.get("subject"),
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
    if "customer_timeline_prod_" in resolved:
        raise ValueError(f"refusing to run stage4b bot opening on prod timeline path: {resolved}")
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
