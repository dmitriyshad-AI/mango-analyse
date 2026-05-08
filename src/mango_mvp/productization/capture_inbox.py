from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.payload_archive import source_provider_call_id, source_recording_id
from mango_mvp.productization.product_db import (
    audit_product_db,
    apply_product_db_migrations,
    guard_product_db_path,
    now_utc,
)
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


CAPTURE_INBOX_SCHEMA_VERSION = "capture_inbox_v1"
CAPTURE_INBOX_READY_STATUS = "ready_for_capture"
ENQUEUE_ACTION_CODES = {"ENQUEUE_SHADOW_CAPTURE", "enqueue_shadow_capture"}


@dataclass(frozen=True)
class CaptureInboxApplySummary:
    schema_version: str
    product_db_path: str
    report_path: str
    source_reports: int
    decisions_seen: int
    enqueue_decisions: int
    inserted: int
    updated_existing: int
    already_present: int
    skipped_non_enqueue: int
    inbox_items: int
    ready_for_capture: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def apply_shadow_poll_report_to_capture_inbox(
    product_db_path: Path,
    product_root: Path,
    report_path: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root, report_path, out_path = resolve_capture_inbox_paths(
        product_db_path=product_db_path,
        product_root=product_root,
        report_path=report_path,
        out_path=out_path,
    )
    report_doc = json.loads(report_path.read_text(encoding="utf-8"))
    source_reports = extract_shadow_reports(report_doc, report_path)

    rows = []
    skipped_non_enqueue = 0
    decision_count = 0
    for source in source_reports:
        raw_ref_index = raw_payload_ref_index(source.raw_payload_path, product_root)
        for decision in source.decisions:
            decision_count += 1
            if not is_enqueue_decision(decision):
                skipped_non_enqueue += 1
                continue
            rows.append(build_inbox_row(decision, source=source, raw_ref_index=raw_ref_index))

    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        apply_product_db_migrations(con)
        inserted = 0
        updated = 0
        already_present = 0
        now = now_utc()
        for row in rows:
            result = upsert_capture_inbox_row(con, row, now=now)
            inserted += 1 if result == "inserted" else 0
            updated += 1 if result == "updated" else 0
            already_present += 1 if result == "already_present" else 0
        con.commit()
        inbox_counts = capture_inbox_counts(con)

    blocked = sum(1 for row in rows if not clean(row.get("event_key")) or not clean(row.get("audio_ref")))
    summary = CaptureInboxApplySummary(
        schema_version=CAPTURE_INBOX_SCHEMA_VERSION,
        product_db_path=str(product_db_path),
        report_path=str(report_path),
        source_reports=len(source_reports),
        decisions_seen=decision_count,
        enqueue_decisions=len(rows),
        inserted=inserted,
        updated_existing=updated,
        already_present=already_present,
        skipped_non_enqueue=skipped_non_enqueue,
        inbox_items=int(inbox_counts["items"]),
        ready_for_capture=int(inbox_counts["ready_for_capture"]),
        validation_ok=blocked == 0,
        blocked=blocked,
        warnings=updated,
    )
    report = {
        "summary": summary.to_json_dict(),
        "source_reports": [source.to_json_dict() for source in source_reports],
        "status_counts": dict(inbox_counts["status_counts"]),
        "samples": {"upserted": rows[:20]},
        "safety": safety_contract(product_db_writes=True),
    }
    if out_path:
        write_json(out_path, report)
    return report


def audit_capture_inbox(
    product_db_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    if out_path:
        guard_capture_inbox_path(out_path, product_root, "capture inbox audit output")

    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        apply_product_db_migrations(con)
        counts = capture_inbox_counts(con)
        source_counts = source_job_counts(con)
        manager_counts = manager_ref_counts(con)
        recent = con.execute(
            """
            SELECT id, tenant_id, provider, event_key, provider_call_id, status,
                   source_job_run_id, started_at, manager_ref, recording_ref,
                   raw_payload_ref, enqueue_count, first_seen_at, last_seen_at
              FROM capture_inbox_items
             ORDER BY last_seen_at DESC, id DESC
             LIMIT 50
            """
        ).fetchall()
    product_integrity = audit_product_db(product_db_path, product_root)
    blocked = int(counts["missing_audio_ref"]) + int(counts["duplicate_event_keys"])
    report = {
        "summary": {
            "schema_version": CAPTURE_INBOX_SCHEMA_VERSION,
            "product_db_path": str(product_db_path),
            "items": int(counts["items"]),
            "ready_for_capture": int(counts["ready_for_capture"]),
            "missing_audio_ref": int(counts["missing_audio_ref"]),
            "duplicate_event_keys": int(counts["duplicate_event_keys"]),
            "validation_ok": bool(product_integrity["summary"]["validation_ok"]) and blocked == 0,
            "blocked": blocked,
            "warnings": int(product_integrity["summary"]["warnings"]),
        },
        "status_counts": dict(counts["status_counts"]),
        "source_job_counts": source_counts,
        "manager_ref_counts": manager_counts,
        "recent_items": [dict(row) for row in recent],
        "product_integrity": product_integrity,
        "safety": safety_contract(product_db_writes=False),
    }
    if out_path:
        write_json(out_path, report)
    return report


@dataclass(frozen=True)
class ShadowReportSource:
    report_path: Path
    source_report_ref: str
    source_job_run_id: Optional[int]
    raw_payload_path: Optional[Path]
    decisions: Sequence[Mapping[str, Any]]

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "source_report_ref": self.source_report_ref,
            "source_job_run_id": self.source_job_run_id,
            "raw_payload_path": str(self.raw_payload_path) if self.raw_payload_path else None,
            "decisions": len(self.decisions),
        }


def extract_shadow_reports(data: Any, report_path: Path) -> Sequence[ShadowReportSource]:
    if not isinstance(data, Mapping):
        raise ValueError("shadow poll report must be a JSON object")
    if isinstance(data.get("results"), list):
        sources = []
        for index, result in enumerate(data["results"]):
            if not isinstance(result, Mapping):
                continue
            result_body = result.get("result")
            if not isinstance(result_body, Mapping):
                continue
            sources.append(
                source_from_report_body(
                    result_body,
                    report_path=report_path,
                    source_report_ref=f"{report_path}#results[{index}]",
                    source_job_run_id=optional_int(result.get("job_id")),
                )
            )
        return tuple(sources)
    return (
        source_from_report_body(
            data,
            report_path=report_path,
            source_report_ref=str(report_path),
            source_job_run_id=infer_job_id_from_path(report_path),
        ),
    )


def source_from_report_body(
    body: Mapping[str, Any],
    report_path: Path,
    source_report_ref: str,
    source_job_run_id: Optional[int],
) -> ShadowReportSource:
    decisions = body.get("decisions") or ()
    if not isinstance(decisions, list):
        decisions = []
    raw_payload = body.get("raw_payload_archive") if isinstance(body.get("raw_payload_archive"), Mapping) else {}
    raw_payload_path = None
    if isinstance(raw_payload, Mapping) and clean(raw_payload.get("path")):
        raw_payload_path = Path(clean(raw_payload.get("path"))).resolve(strict=False)
    return ShadowReportSource(
        report_path=report_path,
        source_report_ref=source_report_ref,
        source_job_run_id=source_job_run_id,
        raw_payload_path=raw_payload_path,
        decisions=tuple(decision for decision in decisions if isinstance(decision, Mapping)),
    )


def build_inbox_row(
    decision: Mapping[str, Any],
    source: ShadowReportSource,
    raw_ref_index: Mapping[str, str],
) -> Mapping[str, Any]:
    event = decision.get("event") if isinstance(decision.get("event"), Mapping) else {}
    candidate = decision.get("candidate") if isinstance(decision.get("candidate"), Mapping) else {}
    event_key = clean(event.get("event_key")) or clean(candidate.get("event_key"))
    provider_call_id = clean(event.get("provider_call_id")) or clean(candidate.get("provider_call_id"))
    provider = clean(candidate.get("provider")) or provider_from_event_key(event_key) or "mango"
    tenant_id = clean(candidate.get("tenant_id")) or tenant_from_event_key(event_key)
    recording_ref = clean(event.get("recording_ref")) or None
    recording_url = clean(event.get("recording_url")) or None
    audio_ref = clean(candidate.get("audio_ref")) or recording_url or recording_ref
    raw_payload_ref = lookup_raw_payload_ref(
        raw_ref_index,
        keys=(event_key, provider_call_id, recording_ref or "", audio_ref or ""),
    )
    return {
        "tenant_id": tenant_id,
        "provider": provider,
        "event_key": event_key,
        "provider_call_id": provider_call_id,
        "status": CAPTURE_INBOX_READY_STATUS,
        "source_job_run_id": source.source_job_run_id,
        "source_report_ref": source.source_report_ref,
        "raw_payload_ref": raw_payload_ref,
        "started_at": clean(event.get("started_at")) or clean(candidate.get("started_at")) or None,
        "ended_at": clean(event.get("ended_at")) or None,
        "direction": clean(event.get("direction")) or clean(candidate.get("direction")) or None,
        "client_phone": clean(event.get("client_phone")) or clean(candidate.get("client_phone")) or None,
        "manager_ref": clean(event.get("manager_ref")) or clean(candidate.get("manager_ref")) or None,
        "recording_ref": recording_ref,
        "recording_url": recording_url,
        "audio_ref": audio_ref,
        "decision_reason": clean(decision.get("reason")) or None,
        "candidate_json": json.dumps(candidate, ensure_ascii=False, sort_keys=True) if candidate else None,
        "event_json": json.dumps(event, ensure_ascii=False, sort_keys=True) if event else None,
    }


def upsert_capture_inbox_row(con: sqlite3.Connection, row: Mapping[str, Any], now: str) -> str:
    source_job_run_id = existing_job_run_id(con, optional_int(row.get("source_job_run_id")))
    existing = con.execute(
        """
        SELECT id, source_report_ref
          FROM capture_inbox_items
         WHERE tenant_id = ?
           AND provider = ?
           AND event_key = ?
        """,
        (clean(row.get("tenant_id")), clean(row.get("provider")), clean(row.get("event_key"))),
    ).fetchone()
    params = (
        clean(row.get("tenant_id")),
        clean(row.get("provider")),
        clean(row.get("event_key")),
        clean(row.get("provider_call_id")),
        clean(row.get("status")),
        source_job_run_id,
        clean(row.get("source_report_ref")) or None,
        clean(row.get("raw_payload_ref")) or None,
        clean(row.get("started_at")) or None,
        clean(row.get("ended_at")) or None,
        clean(row.get("direction")) or None,
        clean(row.get("client_phone")) or None,
        clean(row.get("manager_ref")) or None,
        clean(row.get("recording_ref")) or None,
        clean(row.get("recording_url")) or None,
        clean(row.get("audio_ref")) or None,
        clean(row.get("decision_reason")) or None,
        row.get("candidate_json"),
        row.get("event_json"),
    )
    if existing is None:
        con.execute(
            """
            INSERT INTO capture_inbox_items (
              tenant_id, provider, event_key, provider_call_id, status,
              source_job_run_id, source_report_ref, raw_payload_ref,
              started_at, ended_at, direction, client_phone, manager_ref,
              recording_ref, recording_url, audio_ref, decision_reason,
              candidate_json, event_json, first_seen_at, last_seen_at, enqueue_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            params + (now, now),
        )
        return "inserted"
    if clean(existing["source_report_ref"]) == clean(row.get("source_report_ref")):
        return "already_present"
    con.execute(
        """
        UPDATE capture_inbox_items
           SET provider_call_id = ?,
               status = CASE
                 WHEN status IN ('ready_for_capture', 'reserved') THEN status
                 ELSE ?
               END,
               source_job_run_id = ?,
               source_report_ref = ?,
               raw_payload_ref = coalesce(?, raw_payload_ref),
               started_at = coalesce(started_at, ?),
               ended_at = coalesce(?, ended_at),
               direction = coalesce(?, direction),
               client_phone = coalesce(?, client_phone),
               manager_ref = coalesce(?, manager_ref),
               recording_ref = coalesce(?, recording_ref),
               recording_url = coalesce(?, recording_url),
               audio_ref = coalesce(?, audio_ref),
               decision_reason = coalesce(?, decision_reason),
               candidate_json = coalesce(?, candidate_json),
               event_json = coalesce(?, event_json),
               last_seen_at = ?,
               enqueue_count = enqueue_count + 1
         WHERE id = ?
        """,
        (
            clean(row.get("provider_call_id")),
            clean(row.get("status")),
            source_job_run_id,
            clean(row.get("source_report_ref")) or None,
            clean(row.get("raw_payload_ref")) or None,
            clean(row.get("started_at")) or None,
            clean(row.get("ended_at")) or None,
            clean(row.get("direction")) or None,
            clean(row.get("client_phone")) or None,
            clean(row.get("manager_ref")) or None,
            clean(row.get("recording_ref")) or None,
            clean(row.get("recording_url")) or None,
            clean(row.get("audio_ref")) or None,
            clean(row.get("decision_reason")) or None,
            row.get("candidate_json"),
            row.get("event_json"),
            now,
            int(existing["id"]),
        ),
    )
    return "updated"


def raw_payload_ref_index(raw_payload_path: Optional[Path], product_root: Path) -> Mapping[str, str]:
    if raw_payload_path is None:
        return {}
    guard_capture_inbox_path(raw_payload_path, product_root, "raw payload archive")
    if not raw_payload_path.exists():
        return {}
    index: dict[str, str] = {}
    with raw_payload_path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                entry = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, Mapping):
                continue
            raw_payload = entry.get("raw_payload") if isinstance(entry.get("raw_payload"), Mapping) else entry
            ref = f"{raw_payload_path}#line={line_number}"
            for key in (
                clean(entry.get("event_key")),
                clean(entry.get("provider_call_id")),
                clean(entry.get("recording_id")),
                source_provider_call_id(raw_payload, entry),
                source_recording_id(raw_payload, entry),
            ):
                if key and key not in index:
                    index[key] = ref
    return index


def existing_job_run_id(con: sqlite3.Connection, job_id: Optional[int]) -> Optional[int]:
    if job_id is None:
        return None
    row = con.execute("SELECT id FROM job_runs WHERE id = ?", (job_id,)).fetchone()
    return int(row["id"]) if row else None


def capture_inbox_counts(con: sqlite3.Connection) -> Mapping[str, Any]:
    rows = con.execute(
        """
        SELECT status, count(*) AS n
          FROM capture_inbox_items
         GROUP BY status
         ORDER BY status
        """
    ).fetchall()
    status_counts = {clean(row["status"]): int(row["n"] or 0) for row in rows}
    return {
        "items": scalar_int(con, "select count(*) from capture_inbox_items"),
        "ready_for_capture": status_counts.get(CAPTURE_INBOX_READY_STATUS, 0),
        "missing_audio_ref": scalar_int(con, "select count(*) from capture_inbox_items where audio_ref is null or audio_ref = ''"),
        "duplicate_event_keys": scalar_int(
            con,
            """
            SELECT count(*)
              FROM (
                SELECT tenant_id, provider, event_key
                  FROM capture_inbox_items
                 GROUP BY tenant_id, provider, event_key
                HAVING count(*) > 1
              )
            """,
        ),
        "status_counts": status_counts,
    }


def source_job_counts(con: sqlite3.Connection) -> Mapping[str, int]:
    rows = con.execute(
        """
        SELECT coalesce(cast(source_job_run_id AS TEXT), 'none') AS source_job_run_id,
               count(*) AS n
          FROM capture_inbox_items
         GROUP BY source_job_run_id
         ORDER BY source_job_run_id
        """
    ).fetchall()
    return {clean(row["source_job_run_id"]): int(row["n"] or 0) for row in rows}


def manager_ref_counts(con: sqlite3.Connection) -> Mapping[str, int]:
    rows = con.execute(
        """
        SELECT coalesce(manager_ref, 'none') AS manager_ref, count(*) AS n
          FROM capture_inbox_items
         GROUP BY manager_ref
         ORDER BY count(*) DESC, manager_ref
         LIMIT 30
        """
    ).fetchall()
    return {clean(row["manager_ref"]): int(row["n"] or 0) for row in rows}


def resolve_capture_inbox_paths(
    product_db_path: Path,
    product_root: Path,
    report_path: Path,
    out_path: Optional[Path],
) -> tuple[Path, Path, Path, Optional[Path]]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    report_path = report_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    guard_capture_inbox_path(report_path, product_root, "shadow poll report")
    if out_path:
        guard_capture_inbox_path(out_path, product_root, "capture inbox audit output")
    return product_db_path, product_root, report_path, out_path


def guard_capture_inbox_path(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def is_enqueue_decision(decision: Mapping[str, Any]) -> bool:
    action_code = clean(decision.get("action_code")).upper()
    action = clean(decision.get("action"))
    return action_code in ENQUEUE_ACTION_CODES or action in ENQUEUE_ACTION_CODES


def lookup_raw_payload_ref(raw_ref_index: Mapping[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if clean(key) and clean(key) in raw_ref_index:
            return raw_ref_index[clean(key)]
    return None


def tenant_from_event_key(event_key: str) -> str:
    parts = event_key.split(":", 2)
    return clean(parts[0]) if len(parts) == 3 else ""


def provider_from_event_key(event_key: str) -> str:
    parts = event_key.split(":", 2)
    return clean(parts[1]) if len(parts) == 3 else ""


def infer_job_id_from_path(path: Path) -> Optional[int]:
    match = re.search(r"shadow_poll_job_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else None


def optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def scalar_int(con: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> int:
    row = con.execute(sql, tuple(params)).fetchone()
    return int(row[0] or 0) if row else 0


def safety_contract(product_db_writes: bool) -> Mapping[str, bool]:
    return {
        "product_db_writes": product_db_writes,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "download_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
