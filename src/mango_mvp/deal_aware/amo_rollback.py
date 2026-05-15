from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.deal_aware.stage1_snapshot import safe_text, stringify, write_csv


SNAPSHOT_SCHEMA_VERSION = "deal_aware_amo_pre_write_snapshot_v1"
ROLLBACK_CONFIRMATION = "ROLLBACK_DEAL_AWARE_AMO_FIELDS"
WRITER_VERSION = "deal_aware_stage6_live_writeback_snapshot_v1"

SNAPSHOT_FIELDNAMES = [
    "schema_version",
    "batch_id",
    "input_csv",
    "input_sha256",
    "row_index",
    "review_id",
    "lead_id",
    "field_name",
    "field_id",
    "field_type",
    "old_value",
    "new_value",
    "old_value_sha256",
    "new_value_sha256",
    "snapshot_taken_at",
    "writer_version",
    "operator_approval_path",
]

ROLLBACK_REPORT_FIELDNAMES = [
    "snapshot_key",
    "row_index",
    "review_id",
    "lead_id",
    "field_name",
    "field_id",
    "field_type",
    "old_value",
    "new_value",
    "current_value",
    "rollback_status",
    "reason",
    "apply",
    "attempted_at",
]


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    delay_ms: int = 750
    sleep_func: Callable[[float], None] = time.sleep


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(value: Any) -> str:
    return hashlib.sha256(stringify(value).encode("utf-8")).hexdigest()


def snapshot_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            safe_text(row.get("lead_id")),
            safe_text(row.get("field_name")),
            safe_text(row.get("new_value_sha256")),
        ]
    )


def field_catalog_by_name(field_catalog: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {safe_text(item.get("name")): item for item in field_catalog if safe_text(item.get("name"))}


def extract_custom_field_values(entity: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in entity.get("custom_fields_values") or []:
        if not isinstance(item, dict):
            continue
        values = []
        for value_item in item.get("values") or []:
            if isinstance(value_item, dict) and safe_text(value_item.get("value")):
                values.append(safe_text(value_item.get("value")))
        value = " | ".join(values)
        field_name = safe_text(item.get("field_name"))
        field_code = safe_text(item.get("field_code"))
        field_id = safe_text(item.get("field_id"))
        for key in (field_name, field_code, field_id):
            if key:
                result[key] = value
    return result


def build_pre_write_snapshot_rows(
    *,
    batch_id: str,
    input_csv: Path,
    input_sha256: str,
    row_index: int,
    review_id: str,
    lead_id: str,
    payload: dict[str, Any],
    current_lead: dict[str, Any],
    field_catalog: list[dict[str, Any]],
    operator_approval_path: Path | None,
    snapshot_taken_at: str | None = None,
) -> list[dict[str, Any]]:
    by_name = field_catalog_by_name(field_catalog)
    current_values = extract_custom_field_values(current_lead)
    taken_at = snapshot_taken_at or utc_now_iso()
    rows = []
    for field_name, raw_new_value in payload.items():
        meta = by_name.get(field_name) or {}
        field_id = safe_text(meta.get("id"))
        old_value = current_values.get(field_name) or current_values.get(field_id) or ""
        new_value = stringify(raw_new_value)
        rows.append(
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "batch_id": batch_id,
                "input_csv": str(input_csv),
                "input_sha256": input_sha256,
                "row_index": row_index,
                "review_id": review_id,
                "lead_id": lead_id,
                "field_name": field_name,
                "field_id": field_id,
                "field_type": safe_text(meta.get("type")),
                "old_value": old_value,
                "new_value": new_value,
                "old_value_sha256": sha256_text(old_value),
                "new_value_sha256": sha256_text(new_value),
                "snapshot_taken_at": taken_at,
                "writer_version": WRITER_VERSION,
                "operator_approval_path": str(operator_approval_path) if operator_approval_path else "",
            }
        )
    return rows


def append_snapshot_rows(out_root: Path, rows: list[dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "pre_write_snapshot.jsonl"
    csv_path = out_root / "pre_write_snapshot.csv"
    with jsonl_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SNAPSHOT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerows([{key: stringify(row.get(key)) for key in SNAPSHOT_FIELDNAMES} for row in rows])


def write_rollback_manifest(
    out_root: Path,
    *,
    batch_id: str,
    input_csv: Path,
    input_sha256: str,
    field_catalog_cache: Path,
    operator_approval_path: Path | None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "deal_aware_amo_rollback_manifest_v1",
        "generated_at": utc_now_iso(),
        "batch_id": batch_id,
        "input_csv": str(input_csv),
        "input_sha256": input_sha256,
        "field_catalog_cache": str(field_catalog_cache),
        "operator_approval_path": str(operator_approval_path) if operator_approval_path else "",
        "snapshot_jsonl": str(out_root / "pre_write_snapshot.jsonl"),
        "snapshot_csv": str(out_root / "pre_write_snapshot.csv"),
        "rollback_confirmation": ROLLBACK_CONFIRMATION,
        "writer_version": WRITER_VERSION,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rollback_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_snapshot_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def load_successful_rollback_keys(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else []
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = [dict(row) for row in csv.DictReader(fh)]
    return {
        safe_text(row.get("snapshot_key"))
        for row in rows
        if safe_text(row.get("snapshot_key")) and safe_text(row.get("rollback_status")) in {"restored", "dry_run_ready"}
    }


def exception_status_code(exc: BaseException) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(exc))
    if match:
        return int(match.group(1))
    return None


def is_retryable_exception(exc: BaseException) -> bool:
    status_code = exception_status_code(exc)
    return status_code == 429 or (status_code is not None and 500 <= status_code <= 599)


def call_with_retries(
    func: Callable[[], Any],
    *,
    retry_policy: RetryPolicy,
) -> Any:
    attempts = max(1, retry_policy.max_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt >= attempts or not is_retryable_exception(exc):
                raise
            delay = max(0, retry_policy.delay_ms) / 1000 * (2 ** (attempt - 1))
            if delay:
                retry_policy.sleep_func(delay)
    raise RuntimeError("unreachable retry state")


def rollback_decision(snapshot_row: dict[str, Any], current_lead: dict[str, Any]) -> dict[str, Any]:
    current_values = extract_custom_field_values(current_lead)
    field_name = safe_text(snapshot_row.get("field_name"))
    field_id = safe_text(snapshot_row.get("field_id"))
    current_value = current_values.get(field_name) or current_values.get(field_id) or ""
    old_value = safe_text(snapshot_row.get("old_value"))
    new_value = safe_text(snapshot_row.get("new_value"))
    base = {
        "snapshot_key": snapshot_key(snapshot_row),
        "row_index": safe_text(snapshot_row.get("row_index")),
        "review_id": safe_text(snapshot_row.get("review_id")),
        "lead_id": safe_text(snapshot_row.get("lead_id")),
        "field_name": field_name,
        "field_id": field_id,
        "field_type": safe_text(snapshot_row.get("field_type")),
        "old_value": old_value,
        "new_value": new_value,
        "current_value": current_value,
        "attempted_at": utc_now_iso(),
    }
    if current_value != new_value:
        return {
            **base,
            "rollback_status": "skipped",
            "reason": "current_value_changed_after_write",
        }
    if not old_value:
        return {
            **base,
            "rollback_status": "manual_restore_required",
            "reason": "old_value_empty_existing_helper_cannot_clear_field",
        }
    return {
        **base,
        "rollback_status": "dry_run_ready",
        "reason": "current_value_matches_new_value",
    }


def run_rollback(
    *,
    snapshot_rows: Iterable[dict[str, Any]],
    fetch_lead: Callable[[int], dict[str, Any]],
    send_update: Callable[..., dict[str, Any]] | None = None,
    apply: bool = False,
    confirmation: str = "",
    max_rows: int | None = None,
    batch_size: int = 10,
    retry_policy: RetryPolicy | None = None,
    resume_success_keys: set[str] | None = None,
    progress_writer: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    if apply and confirmation != ROLLBACK_CONFIRMATION:
        raise ValueError(f"--rollback-confirmation must be {ROLLBACK_CONFIRMATION!r} for --apply.")
    if apply and send_update is None:
        raise ValueError("send_update is required for rollback apply.")
    policy = retry_policy or RetryPolicy()
    resume_keys = resume_success_keys or set()
    result = []
    processed = 0
    for snapshot_row in snapshot_rows:
        if max_rows is not None and processed >= max_rows:
            break
        key = snapshot_key(snapshot_row)
        if key in resume_keys:
            result.append(
                {
                    "snapshot_key": key,
                    "row_index": safe_text(snapshot_row.get("row_index")),
                    "review_id": safe_text(snapshot_row.get("review_id")),
                    "lead_id": safe_text(snapshot_row.get("lead_id")),
                    "field_name": safe_text(snapshot_row.get("field_name")),
                    "field_id": safe_text(snapshot_row.get("field_id")),
                    "field_type": safe_text(snapshot_row.get("field_type")),
                    "old_value": safe_text(snapshot_row.get("old_value")),
                    "new_value": safe_text(snapshot_row.get("new_value")),
                    "current_value": "",
                    "rollback_status": "skipped",
                    "reason": "resume_success_already_processed",
                    "apply": str(bool(apply)),
                    "attempted_at": utc_now_iso(),
                }
            )
            processed += 1
            if progress_writer is not None:
                progress_writer(result)
            continue
        lead_id = int(safe_text(snapshot_row.get("lead_id")))
        try:
            current_lead = call_with_retries(lambda: fetch_lead(lead_id), retry_policy=policy)
            decision = rollback_decision(snapshot_row, current_lead)
            if apply and decision["rollback_status"] == "dry_run_ready":
                payload = {decision["field_name"]: decision["old_value"]}
                call_with_retries(lambda: send_update(lead_id=lead_id, field_payload=payload), retry_policy=policy)  # type: ignore[misc]
                decision["rollback_status"] = "restored"
                decision["reason"] = "restored_old_value"
                if policy.delay_ms > 0:
                    policy.sleep_func(policy.delay_ms / 1000)
            decision["apply"] = str(bool(apply))
            result.append(decision)
            if progress_writer is not None:
                progress_writer(result)
        except Exception as exc:  # noqa: BLE001
            result.append(
                {
                    "snapshot_key": key,
                    "row_index": safe_text(snapshot_row.get("row_index")),
                    "review_id": safe_text(snapshot_row.get("review_id")),
                    "lead_id": safe_text(snapshot_row.get("lead_id")),
                    "field_name": safe_text(snapshot_row.get("field_name")),
                    "field_id": safe_text(snapshot_row.get("field_id")),
                    "field_type": safe_text(snapshot_row.get("field_type")),
                    "old_value": safe_text(snapshot_row.get("old_value")),
                    "new_value": safe_text(snapshot_row.get("new_value")),
                    "current_value": "",
                    "rollback_status": "failed",
                    "reason": str(exc),
                    "apply": str(bool(apply)),
                    "attempted_at": utc_now_iso(),
                }
            )
            if progress_writer is not None:
                progress_writer(result)
        processed += 1
        if apply and batch_size > 0 and processed % batch_size == 0 and policy.delay_ms > 0:
            policy.sleep_func(policy.delay_ms / 1000)
    return result


def rollback_summary(
    *,
    rows: list[dict[str, Any]],
    snapshot_path: Path,
    apply: bool,
    max_rollback_rows: int | None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        status = safe_text(row.get("rollback_status"))
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": "deal_aware_amo_rollback_report_v1",
        "generated_at": utc_now_iso(),
        "apply": apply,
        "snapshot_path": str(snapshot_path),
        "max_rollback_rows": max_rollback_rows,
        "evaluated_rows": len(rows),
        "status_counts": counts,
        "restored": counts.get("restored", 0),
        "rolled_back": counts.get("restored", 0),
        "dry_run_ready": counts.get("dry_run_ready", 0),
        "manual_restore_required": counts.get("manual_restore_required", 0),
        "skipped": counts.get("skipped", 0),
        "skipped_changed_by_manager": sum(1 for row in rows if safe_text(row.get("reason")) == "current_value_changed_after_write"),
        "failed": counts.get("failed", 0),
        "failed_retry_exhausted": sum(1 for row in rows if safe_text(row.get("rollback_status")) == "failed" and re.search(r"\b(?:429|5\d\d)\b", safe_text(row.get("reason")))),
        "failed_permanent_error": sum(1 for row in rows if safe_text(row.get("rollback_status")) == "failed" and not re.search(r"\b(?:429|5\d\d)\b", safe_text(row.get("reason")))),
        "pending": counts.get("dry_run_ready", 0),
    }


def write_rollback_outputs(
    out_root: Path,
    *,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    apply: bool,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    prefix = "rollback_apply" if apply else "rollback_dry_run"
    write_csv(out_root / f"{prefix}_report.csv", rows)
    (out_root / f"{prefix}_report.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_root / "rollback_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "rollback_resume_state.json").write_text(
        json.dumps(
            {
                "successful_keys": sorted(
                    {
                        safe_text(row.get("snapshot_key"))
                        for row in rows
                        if safe_text(row.get("snapshot_key")) and safe_text(row.get("rollback_status")) in {"restored", "dry_run_ready"}
                    }
                )
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
