from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.derived_signals import (
    backfill_sg_v1_signals_on_store,
)
from mango_mvp.customer_timeline.ids import stable_digest
from mango_mvp.customer_timeline.mail_stage2_ingest import MAIL_STAGE2_INGEST_SOURCE_SYSTEM
from mango_mvp.customer_timeline.mail_stage2_visibility import (
    harden_mail_stage2_bot_visibility,
    harden_mail_stage2_bot_visibility_on_store,
)
from mango_mvp.customer_timeline.objections import backfill_customer_objections_v1_on_connection
from mango_mvp.customer_timeline.safety import (
    guard_customer_timeline_output_path,
    guard_customer_timeline_writable_path,
)
from mango_mvp.customer_timeline.store import (
    CustomerTimelineSQLiteStore,
    MAIL_IDENTITY_SENTINEL_MAX,
    customer_timeline_integrity_report,
    customer_timeline_integrity_report_ok,
    customer_timeline_run_lock,
    is_mail_identity_sentinel_date,
    json_dumps,
    json_loads,
    normalize_email_content_text,
    scrub_timeline_persisted_json,
)
from mango_mvp.customer_timeline.temporal import parse_aware_utc


STAGE3_MAINTENANCE_SCHEMA_VERSION = "stage3_mail_cleanup_v1"
CHUNK_LABEL_POLICY_VERSION = "cs_v1"
EMAIL_EVENT_TYPE = "email_message"


@dataclass(frozen=True)
class Stage3MaintenanceConfig:
    timeline_db_path: Path
    allowed_root: Path
    out_dir: Path
    canonical_calls_db_path: Path | None = None
    tenant_id: str = "foton"
    apply: bool = True
    batch_size: int = 1000
    signal_as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    lock_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeline_db_path", Path(self.timeline_db_path).expanduser())
        object.__setattr__(self, "allowed_root", Path(self.allowed_root).expanduser())
        object.__setattr__(self, "out_dir", Path(self.out_dir).expanduser())
        if self.canonical_calls_db_path is not None:
            object.__setattr__(self, "canonical_calls_db_path", Path(self.canonical_calls_db_path).expanduser())
        if self.batch_size < 1 or self.batch_size > 1000:
            raise ValueError("batch_size must be between 1 and 1000")
        if self.signal_as_of.tzinfo is None:
            raise ValueError("signal_as_of must be timezone-aware")
        if self.lock_timeout_seconds < 0:
            raise ValueError("lock_timeout_seconds must not be negative")


def run_stage3_maintenance(config: Stage3MaintenanceConfig) -> Mapping[str, Any]:
    db_path = guard_customer_timeline_writable_path(
        guard_customer_timeline_output_path(config.timeline_db_path, config.allowed_root)
    )
    if not config.apply:
        return _run_stage3_maintenance_unlocked(config, db_path=db_path)
    with customer_timeline_run_lock(db_path, timeout_seconds=config.lock_timeout_seconds):
        return _run_stage3_maintenance_unlocked(config, db_path=db_path)


def _run_stage3_maintenance_unlocked(
    config: Stage3MaintenanceConfig,
    *,
    db_path: Path,
) -> Mapping[str, Any]:
    started = time.monotonic()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": STAGE3_MAINTENANCE_SCHEMA_VERSION,
        "mode": "apply" if config.apply else "dry_run",
        "timeline_db_path": str(db_path),
        "tenant_id": config.tenant_id,
        "signal_as_of": config.signal_as_of.isoformat(),
        "safety": {
            "prod_write": False,
            "crm_write": False,
            "llm_calls_total": 0,
            "none_customer_groups_actioned": 0,
        },
    }

    store_context = (
        CustomerTimelineSQLiteStore(db_path, allowed_root=config.allowed_root)
        if config.apply
        else CustomerTimelineSQLiteStore.open_read_only(db_path, allowed_root=config.allowed_root)
    )
    with store_context as store:
        con = store._con  # noqa: SLF001 - staging maintenance uses store-owned connection and FTS helpers.
        report["before"] = _metrics(con)
        identity_date_plan = _load_mail_identity_date_repair_plan(
            con,
            tenant_id=config.tenant_id,
            as_of=config.signal_as_of,
        )
        report["mail_identity_date_repair_plan"] = identity_date_plan["summary"]
        report["mail_identity_date_repair"] = (
            _apply_mail_identity_date_repair_plan(
                store,
                identity_date_plan["rows"],
                as_of=config.signal_as_of,
            )
            if config.apply
            else {"links_repaired": 0, "dry_run": True}
        )
        content_started = time.monotonic()
        if config.apply:
            content_result = store.backfill_timeline_event_content_keys(batch_size=config.batch_size)
        else:
            content_result = {
                "batches": 0,
                "rows_seen": store.count_missing_timeline_email_content_keys(),
                "rows_updated": 0,
            }
        report["content_key_backfill"] = {
            **content_result,
            "elapsed_seconds": round(time.monotonic() - content_started, 3),
        }

        duplicate_plan = _load_duplicate_plan(con, tenant_id=config.tenant_id)
        none_customer_groups = _count_none_customer_duplicate_groups(con, tenant_id=config.tenant_id)
        if duplicate_plan["none_customer_actionable"]:
            raise RuntimeError("refusing to action customer_id NULL duplicate groups")
        report["duplicate_plan"] = {
            "groups": duplicate_plan["groups"],
            "duplicate_events": duplicate_plan["duplicate_events"],
            "rows_in_groups": duplicate_plan["rows_in_groups"],
            "source_system_rows": duplicate_plan["source_system_rows"],
            "none_customer_groups_report_only": none_customer_groups,
            "mixed_preview_groups_report_only": duplicate_plan["mixed_preview_groups_report_only"],
            "mixed_preview_examples": duplicate_plan["mixed_preview_examples"],
            "examples": duplicate_plan["examples"],
        }

        if config.apply:
            report["soft_delete"] = _apply_duplicate_plan(store, duplicate_plan["groups_detail"])
        else:
            report["soft_delete"] = {"superseded_events": 0, "superseded_chunks": 0, "groups_actioned": 0}

        if config.apply:
            objections_started = time.monotonic()
            objections = backfill_customer_objections_v1_on_connection(
                con,
                canonical_calls_db_path=config.canonical_calls_db_path,
                tenant_id=config.tenant_id,
                apply=True,
                as_of=config.signal_as_of,
            )
            report["objections"] = {
                **objections,
                "elapsed_seconds": round(time.monotonic() - objections_started, 3),
            }
            signals_started = time.monotonic()
            signals = backfill_sg_v1_signals_on_store(
                store,
                tenant_id=config.tenant_id,
                as_of=config.signal_as_of,
            )
            report["derived_signals"] = {
                **signals,
                "elapsed_seconds": round(time.monotonic() - signals_started, 3),
            }
        else:
            report["objections"] = {"apply": False, "skipped": "dry_run_avoids_canonical_calls_scan"}
            report["derived_signals"] = {"apply": False, "skipped": "dry_run_avoids_full_signal_scan"}

        report["mail_stage2_visibility_hardening"] = (
            harden_mail_stage2_bot_visibility_on_store(
                store,
                defer_fts_rebuild=True,
                commit=False,
            )
            if config.apply
            else harden_mail_stage2_bot_visibility(
                db_path,
                allowed_root=config.allowed_root,
                apply=False,
                allow_test_paths=True,
            )
        )
        labels_started = time.monotonic()
        report["chunk_label_backfill"] = _backfill_chunk_labels(
            con,
            apply=config.apply,
            commit=False,
        )
        report["chunk_label_backfill"]["elapsed_seconds"] = round(time.monotonic() - labels_started, 3)

        fts_reasons = []
        if int(report["mail_stage2_visibility_hardening"].get("updated_chunks") or 0) > 0:
            fts_reasons.append("mail_stage2_visibility")
        if int(report["chunk_label_backfill"]["counts"].get("chunks_updated") or 0) > 0:
            fts_reasons.append("chunk_labels")
        if config.apply and fts_reasons:
            store._rebuild_fts_indexes()  # noqa: SLF001 - direct chunk SQL requires one atomic rebuild.
        if config.apply:
            con.commit()
        report["fts_rebuild"] = {
            "performed": bool(config.apply and fts_reasons),
            "reasons": fts_reasons,
        }
        fts_started = time.monotonic()
        fts_counts = _fts_superseded_counts(con)
        report["fts_after_rebuild"] = {
            **fts_counts,
            "elapsed_seconds": round(time.monotonic() - fts_started, 3),
        }
        report["after"] = _metrics(con)
        integrity_report = customer_timeline_integrity_report(con)
        final_checks = {
            "quick_check": con.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_check_rows": len(con.execute("PRAGMA foreign_key_check").fetchall()),
            "fts_superseded_counts": fts_counts,
            "integrity_report": integrity_report,
        }
        report["final_checks"] = final_checks
        report["validation_ok"] = bool(
            final_checks["quick_check"] == "ok"
            and final_checks["foreign_key_check_rows"] == 0
            and all(int(value) == 0 for value in fts_counts.values())
            and customer_timeline_integrity_report_ok(integrity_report)
        )

    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (config.out_dir / "stage3_maintenance_report.json").write_text(json_dumps(report), encoding="utf-8")
    return report


def _load_mail_identity_date_repair_plan(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    as_of: datetime,
) -> Mapping[str, Any]:
    repair_cutoff = as_of.astimezone(timezone.utc)
    rows = con.execute(
        """
        SELECT
          l.link_id,
          l.tenant_id,
          l.customer_id,
          l.match_class,
          l.first_seen_at,
          l.last_seen_at,
          e.event_id,
          e.event_at
        FROM identity_links l
        LEFT JOIN timeline_events e
          ON e.tenant_id = l.tenant_id
         AND e.customer_id = l.customer_id
         AND e.source_system = l.source_system
         AND e.source_ref = l.source_ref
         AND e.superseded_by IS NULL
        WHERE l.tenant_id = ?
          AND l.source_system = ?
          AND l.link_type = 'tallanto_student_id'
          AND (
            (COALESCE(l.first_seen_at, '') != '' AND substr(l.first_seen_at, 1, 10) <= '1970-01-02')
            OR (COALESCE(l.last_seen_at, '') != '' AND substr(l.last_seen_at, 1, 10) <= '1970-01-02')
          )
        ORDER BY l.link_id, e.event_at, e.event_id
        """,
        (tenant_id, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
    ).fetchall()
    by_link: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_link.setdefault(str(row["link_id"]), []).append(row)
    counters: Counter[str] = Counter({"links_scanned": len(by_link)})
    plan: list[Mapping[str, Any]] = []
    match_classes: Counter[str] = Counter()
    for link_id, candidates in by_link.items():
        base = candidates[0]
        first_seen_at = parse_aware_utc(base["first_seen_at"])
        last_seen_at = parse_aware_utc(base["last_seen_at"])
        first_corrupt = is_mail_identity_sentinel_date(base["first_seen_at"])
        last_corrupt = is_mail_identity_sentinel_date(base["last_seen_at"])
        if (
            (str(base["first_seen_at"] or "").strip() and first_seen_at is None and not first_corrupt)
            or (str(base["last_seen_at"] or "").strip() and last_seen_at is None and not last_corrupt)
        ):
            counters["invalid_or_naive_existing_dates"] += 1
            continue
        if not first_corrupt and not last_corrupt:
            counters["invalid_or_naive_existing_dates"] += 1
            continue
        event_rows = [row for row in candidates if str(row["event_id"] or "").strip()]
        evidence_event_id: str | None = None
        unknown_reason: str | None = None
        if not event_rows:
            counters["missing_exact_evidence"] += 1
            unknown_reason = "missing_exact_evidence"
        elif len(event_rows) != 1:
            counters["ambiguous_exact_evidence"] += 1
            unknown_reason = "ambiguous_exact_evidence"
        else:
            evidence_row = event_rows[0]
            evidence_at = parse_aware_utc(evidence_row["event_at"])
            if evidence_at is None:
                counters["invalid_evidence_dates"] += 1
                unknown_reason = "invalid_evidence_date"
            elif evidence_at <= MAIL_IDENTITY_SENTINEL_MAX:
                counters["legacy_evidence_dates"] += 1
                unknown_reason = "legacy_evidence_date"
            elif evidence_at > repair_cutoff:
                counters["future_evidence_dates"] += 1
                unknown_reason = "future_evidence_date"
            else:
                new_first = evidence_at if first_corrupt else first_seen_at
                new_last = evidence_at if last_corrupt else last_seen_at
                if new_first is not None and new_last is not None and new_first > new_last:
                    counters["invalid_repaired_range"] += 1
                    unknown_reason = "invalid_exact_repaired_range"
                else:
                    evidence_event_id = str(evidence_row["event_id"])
                    counters["links_exact_date"] += 1
        if unknown_reason:
            counters["links_date_cleared_unknown"] += 1
        plan.append(
            {
                "tenant_id": str(base["tenant_id"]),
                "link_id": link_id,
                "evidence_event_id": evidence_event_id,
                "unknown_reason": unknown_reason,
            }
        )
        match_classes[str(base["match_class"])] += 1
    counters["links_actionable"] = len(plan)
    return {
        "rows": tuple(plan),
        "summary": {
            **dict(counters),
            "match_class_counts_preserved": dict(sorted(match_classes.items())),
            "selection": (
                "one exact active tenant+customer+source ref event -> event_at; "
                "missing/ambiguous/invalid/future evidence -> NULL"
            ),
        },
    }


def _apply_mail_identity_date_repair_plan(
    store: CustomerTimelineSQLiteStore,
    rows: Sequence[Mapping[str, Any]],
    *,
    as_of: datetime,
) -> Mapping[str, Any]:
    statuses: Counter[str] = Counter()
    with store.bulk_write():
        for row in rows:
            result = store.repair_identity_link_seen_at(
                str(row["tenant_id"]),
                link_id=str(row["link_id"]),
                as_of=as_of,
            )
            statuses[result.status] += 1
    return {
        "links_repaired": int(statuses["updated"]),
        "write_status_counts": dict(statuses),
    }


def _load_duplicate_plan(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, Any]:
    content_key_group_rows = con.execute(
        """
        SELECT tenant_id, customer_id, content_key, count(*) AS c
        FROM timeline_events
        WHERE tenant_id = ?
          AND event_type = ?
          AND content_key IS NOT NULL
          AND superseded_by IS NULL
          AND customer_id IS NOT NULL
          AND customer_id != ''
        GROUP BY tenant_id, customer_id, content_key
        HAVING count(*) > 1
        ORDER BY c DESC, customer_id ASC, content_key ASC
        """,
        (tenant_id, EMAIL_EVENT_TYPE),
    ).fetchall()
    groups_detail: list[Mapping[str, Any]] = []
    source_system_rows: Counter[str] = Counter()
    examples: list[Mapping[str, Any]] = []
    mixed_preview_examples: list[Mapping[str, Any]] = []
    rows_in_groups = 0
    duplicate_events = 0
    mixed_preview_groups = 0
    for group in content_key_group_rows:
        rows = con.execute(
            """
            SELECT event_id, source_system, created_at, event_at, source_id, text_preview
            FROM timeline_events
            WHERE tenant_id = ?
              AND customer_id = ?
              AND content_key = ?
              AND event_type = ?
              AND superseded_by IS NULL
            ORDER BY created_at ASC, event_id ASC
            """,
            (group["tenant_id"], group["customer_id"], group["content_key"], EMAIL_EVENT_TYPE),
        ).fetchall()
        by_preview: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            by_preview.setdefault(normalize_email_content_text(row["text_preview"]), []).append(row)
        if len(by_preview) > 1:
            mixed_preview_groups += 1
            if len(mixed_preview_examples) < 20:
                mixed_preview_examples.append(
                    {
                        "customer_id_hash": stable_digest({"customer_id": group["customer_id"]})[:12],
                        "content_key_hash": stable_digest({"content_key": group["content_key"]})[:12],
                        "row_count": len(rows),
                        "preview_variants": len(by_preview),
                    }
                )
        for preview_key, preview_rows in by_preview.items():
            if len(preview_rows) < 2:
                continue
            event_ids = [str(row["event_id"]) for row in preview_rows]
            canonical_id = event_ids[0]
            duplicates = tuple(event_ids[1:])
            rows_in_groups += len(preview_rows)
            duplicate_events += len(duplicates)
            for row in preview_rows:
                source_system_rows[str(row["source_system"])] += 1
            item = {
                "tenant_id": str(group["tenant_id"]),
                "customer_id": str(group["customer_id"]),
                "content_key": str(group["content_key"]),
                "preview_hash": stable_digest({"text_preview": preview_key}),
                "canonical_event_id": canonical_id,
                "duplicate_event_ids": duplicates,
                "row_count": len(preview_rows),
            }
            groups_detail.append(item)
            if len(examples) < 20:
                examples.append(
                    {
                        "customer_id_hash": stable_digest({"customer_id": group["customer_id"]})[:12],
                        "content_key_hash": stable_digest({"content_key": group["content_key"]})[:12],
                        "preview_hash": item["preview_hash"][:12],
                        "row_count": len(preview_rows),
                        "duplicate_count": len(duplicates),
                        "source_systems": sorted({str(row["source_system"]) for row in preview_rows}),
                    }
                )
    return {
        "groups": len(groups_detail),
        "duplicate_events": duplicate_events,
        "rows_in_groups": rows_in_groups,
        "source_system_rows": dict(source_system_rows),
        "mixed_preview_groups_report_only": mixed_preview_groups,
        "mixed_preview_examples": mixed_preview_examples,
        "none_customer_actionable": False,
        "examples": examples,
        "groups_detail": groups_detail,
    }


def _count_none_customer_duplicate_groups(con: sqlite3.Connection, *, tenant_id: str) -> Mapping[str, int]:
    row = con.execute(
        """
        WITH groups AS (
          SELECT tenant_id, content_key, count(*) AS c
          FROM timeline_events
          WHERE tenant_id = ?
            AND event_type = ?
            AND content_key IS NOT NULL
            AND superseded_by IS NULL
            AND (customer_id IS NULL OR customer_id = '')
          GROUP BY tenant_id, content_key
          HAVING count(*) > 1
        )
        SELECT count(*) AS groups, coalesce(sum(c - 1), 0) AS duplicate_events, coalesce(sum(c), 0) AS rows_in_groups
        FROM groups
        """,
        (tenant_id, EMAIL_EVENT_TYPE),
    ).fetchone()
    return {key: int(row[key]) for key in ("groups", "duplicate_events", "rows_in_groups")}


def _apply_duplicate_plan(
    store: CustomerTimelineSQLiteStore,
    groups: Sequence[Mapping[str, Any]],
) -> Mapping[str, int]:
    totals = Counter()
    with store.bulk_write():
        for group in groups:
            result = store.mark_timeline_events_superseded(
                str(group["tenant_id"]),
                canonical_event_id=str(group["canonical_event_id"]),
                duplicate_event_ids=tuple(str(item) for item in group["duplicate_event_ids"]),
                actor="stage3_mail_cleanup",
                reason="stage3_content_duplicate",
            )
            totals["groups_actioned"] += 1
            totals["superseded_events"] += int(result["superseded_events"])
            totals["superseded_chunks"] += int(result["superseded_chunks"])
    return dict(totals)


def _backfill_chunk_labels(
    con: sqlite3.Connection,
    *,
    apply: bool,
    commit: bool = True,
) -> Mapping[str, Any]:
    counters: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    client_safe_reasons: Counter[str] = Counter()
    rows = con.execute(
        """
        SELECT chunk_id, source_system, allowed_for_bot, requires_manager_review, record_json, record_hash
        FROM bot_context_chunks
        WHERE superseded_by IS NULL
        ORDER BY event_at, chunk_id
        """
    ).fetchall()
    for row in rows:
        counters["chunks_seen"] += 1
        source_system = str(row["source_system"] or "")
        source_rows[source_system] += 1
        payload = json_loads(row["record_json"])
        metadata = dict(payload.get("metadata") or {})
        allowed = bool(row["allowed_for_bot"])
        review = bool(row["requires_manager_review"])
        label = _chunk_label_payload(
            source_system=source_system,
            allowed_for_bot=allowed,
            requires_manager_review=review,
            metadata=metadata,
        )
        client_safe_reasons[str(label["client_safe_reason"])] += 1
        if _label_already_present(metadata, label):
            counters["chunks_already_labeled"] += 1
            continue
        metadata.update(label)
        payload["metadata"] = metadata
        record_hash = stable_digest(scrub_timeline_persisted_json(payload))
        if record_hash == row["record_hash"]:
            counters["chunks_hash_unchanged"] += 1
            continue
        counters["chunks_to_update"] += 1
        if apply:
            con.execute(
                """
                UPDATE bot_context_chunks
                SET record_json = ?, record_hash = ?
                WHERE chunk_id = ?
                """,
                (json_dumps(payload), record_hash, row["chunk_id"]),
            )
            counters["chunks_updated"] += 1
    if apply and commit:
        con.commit()
    return {
        "counts": dict(counters),
        "source_rows": dict(source_rows),
        "client_safe_reason_counts": dict(client_safe_reasons),
        "policy_note": (
            "For chunks without A2v3 semantic facts, cs_v1 is conservative: existing bot-visible chunks stay "
            "client_safe=True by prior approval; manager-review/raw mail chunks are client_safe=False until Э4б."
        ),
    }


def _chunk_label_payload(
    *,
    source_system: str,
    allowed_for_bot: bool,
    requires_manager_review: bool,
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    existing_memory = str(metadata.get("memory_status") or "").strip()
    existing_reason = str(metadata.get("client_safe_reason") or "").strip()
    existing_tags = metadata.get("sensitivity_tags")
    if isinstance(existing_tags, list):
        tags = tuple(str(item) for item in existing_tags if str(item))
    elif isinstance(existing_tags, tuple):
        tags = tuple(str(item) for item in existing_tags if str(item))
    else:
        tags = ()
    if source_system == MAIL_STAGE2_INGEST_SOURCE_SYSTEM:
        client_safe = bool(metadata.get("client_safe")) if "client_safe" in metadata else False
        reason = existing_reason or ("a2v3_fact_safe" if client_safe else "stage2_mail_manager_review_pending")
        memory_status = existing_memory or "manager_review_required"
        tags = tags or ("email", "manager_review")
    elif allowed_for_bot and not requires_manager_review:
        client_safe = True
        reason = existing_reason or "preexisting_bot_visible"
        memory_status = existing_memory or "usable_memory"
        tags = tags or ("preexisting_bot_visible",)
    else:
        client_safe = bool(metadata.get("client_safe")) if "client_safe" in metadata else False
        reason = existing_reason or "manager_review_required"
        memory_status = existing_memory or "manager_review_required"
        tags = tags or ("manager_review",)
    return {
        "client_safe": client_safe,
        "client_safe_reason": reason,
        "client_safe_policy_version": CHUNK_LABEL_POLICY_VERSION,
        "sensitivity_tags": tuple(dict.fromkeys(tags)),
        "memory_status": memory_status,
    }


def _label_already_present(metadata: Mapping[str, Any], label: Mapping[str, Any]) -> bool:
    return (
        metadata.get("client_safe") == label["client_safe"]
        and metadata.get("client_safe_reason") == label["client_safe_reason"]
        and metadata.get("client_safe_policy_version") == label["client_safe_policy_version"]
        and tuple(metadata.get("sensitivity_tags") or ()) == tuple(label["sensitivity_tags"])
        and metadata.get("memory_status") == label["memory_status"]
    )


def _metrics(con: sqlite3.Connection) -> Mapping[str, Any]:
    events = con.execute(
        """
        SELECT
          COUNT(*) AS timeline_events,
          COALESCE(SUM(CASE WHEN superseded_by IS NULL THEN 1 ELSE 0 END), 0) AS active_timeline_events,
          COALESCE(SUM(CASE WHEN event_type = ? THEN 1 ELSE 0 END), 0) AS email_events,
          COALESCE(SUM(CASE WHEN source_system = ? THEN 1 ELSE 0 END), 0) AS mail_stage2_events,
          COALESCE(SUM(CASE WHEN content_key IS NULL
            AND customer_id IS NOT NULL AND customer_id != ''
            AND event_type = ? AND summary IS NOT NULL AND summary != '' THEN 1 ELSE 0 END), 0)
            AS content_key_missing,
          COALESCE(SUM(CASE WHEN superseded_by IS NOT NULL AND superseded_by != '' THEN 1 ELSE 0 END), 0)
            AS superseded_events
        FROM timeline_events
        """,
        (EMAIL_EVENT_TYPE, MAIL_STAGE2_INGEST_SOURCE_SYSTEM, EMAIL_EVENT_TYPE),
    ).fetchone()
    chunks = con.execute(
        """
        SELECT
          COUNT(*) AS chunks,
          COALESCE(SUM(CASE WHEN superseded_by IS NULL THEN 1 ELSE 0 END), 0) AS active_chunks,
          COALESCE(SUM(CASE WHEN superseded_by IS NOT NULL AND superseded_by != '' THEN 1 ELSE 0 END), 0)
            AS superseded_chunks,
          COALESCE(SUM(CASE WHEN superseded_by IS NULL
            AND json_extract(record_json, '$.metadata.client_safe_policy_version') IS NULL THEN 1 ELSE 0 END), 0)
            AS chunks_missing_cs_v1,
          COALESCE(SUM(CASE WHEN source_system = ? AND allowed_for_bot != 0 THEN 1 ELSE 0 END), 0)
            AS mail_stage2_chunks_allowed,
          COALESCE(SUM(CASE WHEN source_system = ? AND requires_manager_review != 1 THEN 1 ELSE 0 END), 0)
            AS mail_stage2_chunks_without_review
        FROM bot_context_chunks
        """,
        (MAIL_STAGE2_INGEST_SOURCE_SYSTEM, MAIL_STAGE2_INGEST_SOURCE_SYSTEM),
    ).fetchone()
    return {
        key: int(row[key] or 0)
        for row in (events, chunks)
        for key in row.keys()
    }


def _fts_superseded_counts(con: sqlite3.Connection) -> Mapping[str, int]:
    result = {
        "timeline_event_fts_superseded": 0,
        "timeline_event_fts_keys_superseded": 0,
        "bot_context_chunk_fts_superseded": 0,
    }
    if _table_exists(con, "timeline_event_fts"):
        result["timeline_event_fts_superseded"] = _scalar(
            con,
            """
            SELECT count(*)
            FROM timeline_event_fts
            WHERE event_id IN (
              SELECT event_id FROM timeline_events WHERE superseded_by IS NOT NULL AND superseded_by != ''
            )
            """,
        )
    if _table_exists(con, "timeline_event_fts_keys"):
        result["timeline_event_fts_keys_superseded"] = _scalar(
            con,
            """
            SELECT count(*)
            FROM timeline_event_fts_keys
            WHERE event_id IN (
              SELECT event_id FROM timeline_events WHERE superseded_by IS NOT NULL AND superseded_by != ''
            )
            """,
        )
    if _table_exists(con, "bot_context_chunk_fts"):
        result["bot_context_chunk_fts_superseded"] = _scalar(
            con,
            """
            SELECT count(*)
            FROM bot_context_chunk_fts
            WHERE chunk_id IN (
              SELECT chunk_id FROM bot_context_chunks WHERE superseded_by IS NOT NULL AND superseded_by != ''
            )
            """,
        )
    return result


def _scalar(con: sqlite3.Connection, query: str, params: Sequence[Any] = ()) -> int:
    return int(con.execute(query, tuple(params)).fetchone()[0])


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


__all__ = [
    "CHUNK_LABEL_POLICY_VERSION",
    "STAGE3_MAINTENANCE_SCHEMA_VERSION",
    "Stage3MaintenanceConfig",
    "run_stage3_maintenance",
]
