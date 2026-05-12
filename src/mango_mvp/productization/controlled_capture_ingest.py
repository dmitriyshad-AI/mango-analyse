from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.capture_inbox import (
    build_inbox_row,
    extract_shadow_reports,
    is_enqueue_decision,
    upsert_capture_inbox_row,
)
from mango_mvp.productization.product_db import apply_product_db_migrations, guard_product_db_path, now_utc
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


CONTROLLED_CAPTURE_INGEST_SCHEMA_VERSION = "controlled_capture_ingest_v1"
INGEST_ENQUEUE_CAPTURE = "INGEST_ENQUEUE_CAPTURE"
SKIP_DUPLICATE_CAPTURE_INBOX = "SKIP_DUPLICATE_CAPTURE_INBOX"
SKIP_DUPLICATE_PRODUCT_CALL = "SKIP_DUPLICATE_PRODUCT_CALL"
WAIT_DELAYED_RECORDING = "WAIT_DELAYED_RECORDING"
SKIP_NO_RECORDING = "SKIP_NO_RECORDING"
BLOCK_POLICY = "BLOCK_POLICY"
BLOCK_MISSING_EVENT_KEY = "BLOCK_MISSING_EVENT_KEY"


@dataclass(frozen=True)
class ControlledCaptureIngestSummary:
    schema_version: str
    product_db_path: str
    report_path: str
    apply: bool
    source_reports: int
    decisions_seen: int
    ingest_enqueue_capture: int
    skip_duplicate_capture_inbox: int
    skip_duplicate_product_call: int
    wait_delayed_recording: int
    skip_no_recording: int
    blocked_policy: int
    blocked_missing_event_key: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_controlled_capture_ingest_report(
    product_db_path: Path,
    product_root: Path,
    report_path: Path,
    out_path: Optional[Path] = None,
    *,
    apply: bool = False,
    delayed_recording_grace_hours: int = 24,
    now: Optional[datetime] = None,
) -> Mapping[str, Any]:
    """Plan or apply controlled Mango capture ingest from a shadow poll report.

    The default mode is read-only. When ``apply=True`` this function delegates to
    the existing capture inbox apply path, which writes only product DB inbox rows
    for enqueue decisions. It never downloads audio, runs ASR/R+A, or writes CRM.
    """

    product_db_path, product_root, report_path, out_path = resolve_controlled_ingest_paths(
        product_db_path=product_db_path,
        product_root=product_root,
        report_path=report_path,
        out_path=out_path,
    )
    if delayed_recording_grace_hours < 0:
        raise ValueError("delayed_recording_grace_hours must not be negative")

    generated_at = now or datetime.now(timezone.utc)
    report_doc = json.loads(report_path.read_text(encoding="utf-8"))
    sources = extract_shadow_reports(report_doc, report_path)
    existing_refs = read_existing_product_refs(product_db_path)

    items: list[Mapping[str, Any]] = []
    for source_index, source in enumerate(sources):
        for decision_index, decision in enumerate(source.decisions):
            items.append(
                classify_shadow_decision(
                    decision=decision,
                    source_report_ref=source.source_report_ref,
                    source_job_run_id=source.source_job_run_id,
                    decision_ref=build_decision_ref(source_index=source_index, decision_index=decision_index),
                    existing_refs=existing_refs,
                    now=generated_at,
                    delayed_recording_grace_hours=delayed_recording_grace_hours,
                )
            )

    action_counts = Counter(clean(item.get("action")) for item in items)
    blocked = int(action_counts[BLOCK_POLICY] + action_counts[BLOCK_MISSING_EVENT_KEY])
    warnings = int(
        action_counts[SKIP_DUPLICATE_CAPTURE_INBOX]
        + action_counts[SKIP_DUPLICATE_PRODUCT_CALL]
        + action_counts[WAIT_DELAYED_RECORDING]
        + action_counts[SKIP_NO_RECORDING]
    )
    apply_result = None
    if apply:
        apply_result = apply_controlled_enqueue_decisions(
            product_db_path=product_db_path,
            sources=sources,
            allowed_decision_refs={
                clean(item.get("decision_ref"))
                for item in items
                if clean(item.get("action")) == INGEST_ENQUEUE_CAPTURE and clean(item.get("decision_ref"))
            },
        )

    summary = ControlledCaptureIngestSummary(
        schema_version=CONTROLLED_CAPTURE_INGEST_SCHEMA_VERSION,
        product_db_path=str(product_db_path),
        report_path=str(report_path),
        apply=apply,
        source_reports=len(sources),
        decisions_seen=len(items),
        ingest_enqueue_capture=int(action_counts[INGEST_ENQUEUE_CAPTURE]),
        skip_duplicate_capture_inbox=int(action_counts[SKIP_DUPLICATE_CAPTURE_INBOX]),
        skip_duplicate_product_call=int(action_counts[SKIP_DUPLICATE_PRODUCT_CALL]),
        wait_delayed_recording=int(action_counts[WAIT_DELAYED_RECORDING]),
        skip_no_recording=int(action_counts[SKIP_NO_RECORDING]),
        blocked_policy=int(action_counts[BLOCK_POLICY]),
        blocked_missing_event_key=int(action_counts[BLOCK_MISSING_EVENT_KEY]),
        validation_ok=blocked == 0,
        blocked=blocked,
        warnings=warnings,
    )
    result = {
        "summary": summary.to_json_dict(),
        "action_counts": dict(sorted(action_counts.items())),
        "items": items,
        "apply_result": apply_result,
        "policy": {
            "delayed_recording_grace_hours": delayed_recording_grace_hours,
            "no_recording_before_grace_expires": WAIT_DELAYED_RECORDING,
            "no_recording_after_grace_expires": SKIP_NO_RECORDING,
            "duplicates_are_never_reenqueued": True,
        },
        "safety": safety_contract(product_db_writes=apply),
    }
    if out_path:
        write_json(out_path, result)
    return result


def apply_controlled_enqueue_decisions(
    *,
    product_db_path: Path,
    sources: Sequence[Any],
    allowed_decision_refs: set[str],
) -> Mapping[str, Any]:
    inserted = 0
    updated = 0
    already_present = 0
    skipped = 0
    now = now_utc()
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        apply_product_db_migrations(con)
        for source_index, source in enumerate(sources):
            for decision_index, decision in enumerate(source.decisions):
                row = build_inbox_row(decision, source=source, raw_ref_index={})
                decision_ref = build_decision_ref(source_index=source_index, decision_index=decision_index)
                if decision_ref not in allowed_decision_refs:
                    skipped += 1
                    continue
                result = upsert_capture_inbox_row(con, row, now=now)
                inserted += 1 if result == "inserted" else 0
                updated += 1 if result == "updated" else 0
                already_present += 1 if result == "already_present" else 0
        con.commit()
    return {
        "summary": {
            "schema_version": CONTROLLED_CAPTURE_INGEST_SCHEMA_VERSION,
            "inserted": inserted,
            "updated_existing": updated,
            "already_present": already_present,
            "skipped_not_controlled_enqueue": skipped,
            "product_db_writes": True,
        }
    }


def classify_shadow_decision(
    *,
    decision: Mapping[str, Any],
    source_report_ref: str,
    source_job_run_id: Optional[int],
    decision_ref: str,
    existing_refs: Mapping[str, set[str]],
    now: datetime,
    delayed_recording_grace_hours: int,
) -> Mapping[str, Any]:
    event = decision.get("event") if isinstance(decision.get("event"), Mapping) else {}
    candidate = decision.get("candidate") if isinstance(decision.get("candidate"), Mapping) else {}
    row = build_inbox_row(
        decision,
        source=FakeShadowSource(source_report_ref=source_report_ref, source_job_run_id=source_job_run_id),
        raw_ref_index={},
    )
    event_key = clean(row.get("event_key"))
    provider_call_id = clean(row.get("provider_call_id"))
    recording_ref = clean(row.get("recording_ref")) or clean(row.get("audio_ref"))
    source_action = clean(decision.get("action_code")).upper() or clean(decision.get("action")).upper()

    base = {
        "schema_version": CONTROLLED_CAPTURE_INGEST_SCHEMA_VERSION,
        "decision_ref": decision_ref,
        "source_report_ref": source_report_ref,
        "source_job_run_id": source_job_run_id,
        "source_action": source_action or None,
        "tenant_id": clean(row.get("tenant_id")),
        "provider": clean(row.get("provider")),
        "event_key": event_key or None,
        "provider_call_id": provider_call_id or None,
        "started_at": clean(row.get("started_at")) or None,
        "direction": clean(row.get("direction")) or None,
        "client_phone": clean(row.get("client_phone")) or None,
        "manager_ref": clean(row.get("manager_ref")) or None,
        "recording_ref": clean(row.get("recording_ref")) or None,
        "recording_url": clean(row.get("recording_url")) or None,
        "audio_ref": clean(row.get("audio_ref")) or None,
        "download_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_runtime_db": False,
        "write_crm": False,
    }

    if not event_key:
        return base | {"action": BLOCK_MISSING_EVENT_KEY, "reason": "event_key_missing"}
    if event_key in existing_refs["capture_event_keys"]:
        return base | {"action": SKIP_DUPLICATE_CAPTURE_INBOX, "reason": "event_key_already_in_capture_inbox"}
    if provider_call_id and provider_call_id in existing_refs["product_provider_call_ids"]:
        return base | {"action": SKIP_DUPLICATE_PRODUCT_CALL, "reason": "provider_call_id_already_in_product_calls"}
    if provider_call_id and provider_call_id in existing_refs["capture_provider_call_ids"]:
        return base | {"action": SKIP_DUPLICATE_CAPTURE_INBOX, "reason": "provider_call_id_already_in_capture_inbox"}

    if is_enqueue_decision(decision):
        if recording_ref:
            return base | {"action": INGEST_ENQUEUE_CAPTURE, "reason": "recording_ref_ready_for_controlled_capture"}
        return base | classify_missing_recording(
            started_at=clean(row.get("started_at")),
            now=now,
            delayed_recording_grace_hours=delayed_recording_grace_hours,
            source_action=source_action,
        )

    if is_no_recording_decision(source_action, event=event, candidate=candidate, recording_ref=recording_ref):
        return base | classify_missing_recording(
            started_at=clean(row.get("started_at")),
            now=now,
            delayed_recording_grace_hours=delayed_recording_grace_hours,
            source_action=source_action,
        )

    return base | {
        "action": BLOCK_POLICY,
        "reason": clean(decision.get("reason")) or f"source_action_blocked_by_policy:{source_action or 'unknown'}",
    }


def classify_missing_recording(
    *,
    started_at: str,
    now: datetime,
    delayed_recording_grace_hours: int,
    source_action: str,
) -> Mapping[str, Any]:
    started = parse_datetime(started_at)
    if started is not None and now - started <= timedelta(hours=delayed_recording_grace_hours):
        return {
            "action": WAIT_DELAYED_RECORDING,
            "reason": "recording_may_arrive_later",
            "source_action": source_action or None,
        }
    return {
        "action": SKIP_NO_RECORDING,
        "reason": "recording_ref_missing_after_grace_window",
        "source_action": source_action or None,
    }


def is_no_recording_decision(
    source_action: str,
    *,
    event: Mapping[str, Any],
    candidate: Mapping[str, Any],
    recording_ref: str,
) -> bool:
    if recording_ref:
        return False
    if "NO_RECORDING" in source_action or "MISSING_RECORDING" in source_action:
        return True
    for payload in (event, candidate):
        if clean(payload.get("recording_ref")) or clean(payload.get("recording_url")) or clean(payload.get("audio_ref")):
            return False
    return True


def build_decision_ref(*, source_index: int, decision_index: int) -> str:
    return f"source[{source_index}].decisions[{decision_index}]"


@dataclass(frozen=True)
class FakeShadowSource:
    source_report_ref: str
    source_job_run_id: Optional[int]


def read_existing_product_refs(product_db_path: Path) -> Mapping[str, set[str]]:
    refs = {
        "capture_event_keys": set(),
        "capture_provider_call_ids": set(),
        "product_event_keys": set(),
        "product_provider_call_ids": set(),
    }
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        if relation_exists(con, "capture_inbox_items"):
            refs["capture_event_keys"] = {
                clean(row["event_key"])
                for row in con.execute("SELECT event_key FROM capture_inbox_items WHERE event_key IS NOT NULL")
            }
            refs["capture_provider_call_ids"] = {
                clean(row["provider_call_id"])
                for row in con.execute(
                    "SELECT provider_call_id FROM capture_inbox_items WHERE provider_call_id IS NOT NULL"
                )
            }
        if relation_exists(con, "product_calls"):
            refs["product_event_keys"] = {
                clean(row["event_key"])
                for row in con.execute("SELECT event_key FROM product_calls WHERE event_key IS NOT NULL")
            }
            refs["product_provider_call_ids"] = {
                clean(row["provider_call_id"])
                for row in con.execute("SELECT provider_call_id FROM product_calls WHERE provider_call_id IS NOT NULL")
            }
    return refs


def resolve_controlled_ingest_paths(
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
    guard_under_product_root(report_path, product_root, "shadow poll report")
    if out_path:
        guard_under_product_root(out_path, product_root, "controlled ingest output")
    return product_db_path, product_root, report_path, out_path


def guard_under_product_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def relation_exists(con: sqlite3.Connection, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (clean(name),),
    ).fetchone()
    return row is not None


def safety_contract(product_db_writes: bool) -> Mapping[str, bool]:
    return {
        "product_db_writes": product_db_writes,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
