#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.utils.phone import normalize_phone

try:
    import xlsxwriter
except ImportError:  # pragma: no cover - optional runtime dependency
    xlsxwriter = None


BUCKET_READY_SINGLE = "ready_single_contact_not_written"
BUCKET_MULTI_CONTACT = "needs_manager_review_multi_contact"
BUCKET_CONTACT_ID_MISMATCH = "blocked_contact_id_mismatch"
BUCKET_TEXT_QUALITY = "needs_text_quality_review"
BUCKET_DEFERRED_NON_SALES = "deferred_non_sales_or_service"
BUCKET_ALREADY_WRITTEN = "already_written"

BUCKETS = (
    BUCKET_READY_SINGLE,
    BUCKET_MULTI_CONTACT,
    BUCKET_CONTACT_ID_MISMATCH,
    BUCKET_TEXT_QUALITY,
    BUCKET_DEFERRED_NON_SALES,
    BUCKET_ALREADY_WRITTEN,
)

SERVICE_OR_NON_SALES_CALL_TYPES = {
    "service_call",
    "existing_client_progress",
    "technical_call",
    "non_conversation",
}

QUEUE_COLUMNS = [
    "queue_bucket",
    "queue_reason",
    "source_row_index",
    "normalized_phone",
    "source_amo_contact_ids",
    "effective_contact_id",
    "written_status",
    "written_contact_id",
    "written_report",
    "dry_run_status",
    "dry_run_reason",
    "dry_run_contact_id",
    "dry_run_report",
    "crm_quality_decision",
    "crm_quality_risk_types",
    "crm_quality_matches",
    "crm_quality_report",
    "manual_review_report",
]


@dataclass(frozen=True)
class AmoWritebackQueueConfig:
    input_csv: Path
    out_root: Path
    writeback_reports: tuple[Path, ...] = ()
    dry_run_reports: tuple[Path, ...] = ()
    quality_reports: tuple[Path, ...] = ()
    manual_review_inputs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class QueueContext:
    written_by_phone: dict[str, dict[str, Any]]
    written_by_contact_id: dict[str, dict[str, Any]]
    dry_run_by_phone: dict[str, dict[str, Any]]
    dry_run_by_contact_id: dict[str, dict[str, Any]]
    quality_by_phone: dict[str, dict[str, Any]]
    quality_by_row_index: dict[str, dict[str, Any]]
    manual_review_by_phone: dict[str, dict[str, Any]]


def build_amo_writeback_queue(config: AmoWritebackQueueConfig) -> dict[str, Any]:
    out_root = config.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    input_csv = config.input_csv.expanduser().resolve()
    source_rows, source_headers = _read_csv_with_headers(input_csv)

    writeback_report_paths = _expand_report_paths(config.writeback_reports)
    dry_run_report_paths = _expand_report_paths(config.dry_run_reports)
    quality_report_paths = _expand_quality_report_paths(config.quality_reports)
    manual_review_input_paths = _expand_quality_report_paths(config.manual_review_inputs)

    context = _build_context(
        writeback_report_paths=writeback_report_paths,
        dry_run_report_paths=dry_run_report_paths,
        quality_report_paths=quality_report_paths,
        manual_review_input_paths=manual_review_input_paths,
    )

    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKETS}
    for index, row in enumerate(source_rows, start=1):
        bucket, queue_row = classify_queue_row(row, index=index, context=context)
        buckets[bucket].append(queue_row)

    headers = _queue_headers(source_headers)
    bucket_outputs: dict[str, dict[str, str]] = {}
    for bucket in BUCKETS:
        csv_path = out_root / f"{bucket}.csv"
        xlsx_path = out_root / f"{bucket}.xlsx"
        _write_csv(csv_path, headers, buckets[bucket])
        xlsx_written = _write_xlsx(xlsx_path, headers, buckets[bucket])
        bucket_outputs[bucket] = {
            "csv": str(csv_path),
            "xlsx": str(xlsx_path) if xlsx_written else "",
        }

    summary = {
        "schema_version": "amo_writeback_bucket_queue_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_csv": str(input_csv),
        "out_root": str(out_root),
        "rows": len(source_rows),
        "bucket_counts": {bucket: len(buckets[bucket]) for bucket in BUCKETS},
        "writeback_reports": [str(path) for path in writeback_report_paths],
        "dry_run_reports": [str(path) for path in dry_run_report_paths],
        "quality_reports": [str(path) for path in quality_report_paths],
        "manual_review_inputs": [str(path) for path in manual_review_input_paths],
        "written_phone_count": len(context.written_by_phone),
        "written_contact_id_count": len(context.written_by_contact_id),
        "dry_run_phone_count": len(context.dry_run_by_phone),
        "quality_blocker_phone_count": len(context.quality_by_phone),
        "manual_review_phone_count": len(context.manual_review_by_phone),
        "outputs": {
            "summary_json": str(out_root / "summary.json"),
            "buckets": bucket_outputs,
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def classify_queue_row(row: dict[str, Any], *, index: int, context: QueueContext) -> tuple[str, dict[str, Any]]:
    phone = _normalize(row.get("Телефон клиента") or row.get("phone"))
    source_contact_ids = _split_ids(row.get("AMO contact IDs") or row.get("amo_contact_ids"))
    dry_run = _lookup_report_row(
        phone=phone,
        contact_ids=source_contact_ids,
        by_phone=context.dry_run_by_phone,
        by_contact_id=context.dry_run_by_contact_id,
    )
    written = _lookup_report_row(
        phone=phone,
        contact_ids=source_contact_ids + _split_ids((dry_run or {}).get("contact_id")),
        by_phone=context.written_by_phone,
        by_contact_id=context.written_by_contact_id,
    )
    quality = context.quality_by_phone.get(phone or "") or context.quality_by_row_index.get(str(index))
    manual_review = context.manual_review_by_phone.get(phone or "")
    effective_contact_ids = _effective_contact_ids(source_contact_ids, dry_run)
    effective_contact_id = effective_contact_ids[0] if len(effective_contact_ids) == 1 else ""

    if manual_review:
        bucket = BUCKET_TEXT_QUALITY
        reason = "manual_review_input"
        if written:
            reason += ":already_written"
    elif quality:
        bucket = BUCKET_TEXT_QUALITY
        reason = "crm_quality_blocker:" + _safe_text(
            quality.get("risk_types") or quality.get("crm_text_warning_types") or quality.get("decision")
        )
        if written:
            reason += ":already_written"
    elif written:
        bucket = BUCKET_ALREADY_WRITTEN
        reason = "already_written_by_phone_or_contact_id"
    elif _is_contact_id_mismatch(dry_run):
        bucket = BUCKET_CONTACT_ID_MISMATCH
        reason = "dry_run_contact_id_mismatch"
    elif _is_deferred_non_sales_or_service(row):
        bucket = BUCKET_DEFERRED_NON_SALES
        reason = _deferred_reason(row)
    elif _is_multi_contact_or_ambiguous(row, dry_run, effective_contact_ids):
        bucket = BUCKET_MULTI_CONTACT
        reason = _manager_review_reason(row, dry_run, effective_contact_ids)
    elif _safe_text((dry_run or {}).get("status")).casefold() != "dry_run":
        bucket = BUCKET_MULTI_CONTACT
        reason = "not_verified_by_real_tunnel_dry_run"
    elif effective_contact_id and source_contact_ids and effective_contact_id not in source_contact_ids:
        bucket = BUCKET_CONTACT_ID_MISMATCH
        reason = "dry_run_contact_id_not_in_source_amo_contact_ids"
    else:
        bucket = BUCKET_READY_SINGLE
        reason = "single_contact_not_written"

    queue_row = {
        "queue_bucket": bucket,
        "queue_reason": reason,
        "source_row_index": index,
        "normalized_phone": phone or "",
        "source_amo_contact_ids": " | ".join(source_contact_ids),
        "effective_contact_id": effective_contact_id,
        "written_status": _safe_text((written or {}).get("status")),
        "written_contact_id": _safe_text((written or {}).get("contact_id")),
        "written_report": _safe_text((written or {}).get("__report_path")),
        "dry_run_status": _safe_text((dry_run or {}).get("status")),
        "dry_run_reason": _safe_text((dry_run or {}).get("reason")),
        "dry_run_contact_id": _safe_text((dry_run or {}).get("contact_id")),
        "dry_run_report": _safe_text((dry_run or {}).get("__report_path")),
        "crm_quality_decision": _safe_text((quality or {}).get("decision")),
        "crm_quality_risk_types": _safe_text(
            (quality or {}).get("risk_types") or (quality or {}).get("crm_text_warning_types")
        ),
        "crm_quality_matches": _safe_text(
            (quality or {}).get("detector_matches") or (quality or {}).get("crm_text_warning_matches")
        ),
        "crm_quality_report": _safe_text((quality or {}).get("__report_path")),
        "manual_review_report": _safe_text((manual_review or {}).get("__report_path")),
    }
    for key, value in row.items():
        if key not in queue_row:
            queue_row[key] = value
    return bucket, queue_row


def _build_context(
    *,
    writeback_report_paths: list[Path],
    dry_run_report_paths: list[Path],
    quality_report_paths: list[Path],
    manual_review_input_paths: list[Path],
) -> QueueContext:
    written_by_phone: dict[str, dict[str, Any]] = {}
    written_by_contact_id: dict[str, dict[str, Any]] = {}
    for report_path in writeback_report_paths:
        for report_row in _read_report_rows(report_path):
            if _safe_text(report_row.get("status")).casefold() != "written":
                continue
            phone = _normalize(report_row.get("phone"))
            if phone:
                written_by_phone[phone] = report_row
            for contact_id in _split_ids(report_row.get("contact_id")):
                written_by_contact_id[contact_id] = report_row

    dry_run_by_phone: dict[str, dict[str, Any]] = {}
    dry_run_by_contact_id: dict[str, dict[str, Any]] = {}
    for report_path in dry_run_report_paths:
        for report_row in _read_report_rows(report_path):
            phone = _normalize(report_row.get("phone"))
            if phone:
                dry_run_by_phone[phone] = report_row
            for contact_id in _split_ids(report_row.get("contact_id")):
                dry_run_by_contact_id[contact_id] = report_row

    quality_by_phone: dict[str, dict[str, Any]] = {}
    quality_by_row_index: dict[str, dict[str, Any]] = {}
    for report_path in quality_report_paths:
        for quality_row in _read_quality_rows(report_path):
            if not _is_quality_blocker(quality_row):
                continue
            phone = _normalize(quality_row.get("phone") or quality_row.get("Телефон клиента"))
            if phone:
                quality_by_phone[phone] = quality_row
            row_index = _safe_text(quality_row.get("row_index"))
            if row_index:
                quality_by_row_index[row_index] = quality_row

    manual_review_by_phone: dict[str, dict[str, Any]] = {}
    for input_path in manual_review_input_paths:
        for review_row in _read_quality_rows(input_path):
            phone = _normalize(
                review_row.get("phone")
                or review_row.get("Телефон клиента")
                or review_row.get("Телефон")
            )
            if phone:
                manual_review_by_phone[phone] = review_row

    return QueueContext(
        written_by_phone=written_by_phone,
        written_by_contact_id=written_by_contact_id,
        dry_run_by_phone=dry_run_by_phone,
        dry_run_by_contact_id=dry_run_by_contact_id,
        quality_by_phone=quality_by_phone,
        quality_by_row_index=quality_by_row_index,
        manual_review_by_phone=manual_review_by_phone,
    )


def _lookup_report_row(
    *,
    phone: str | None,
    contact_ids: list[str],
    by_phone: dict[str, dict[str, Any]],
    by_contact_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if phone and phone in by_phone:
        return by_phone[phone]
    for contact_id in contact_ids:
        if contact_id in by_contact_id:
            return by_contact_id[contact_id]
    return None


def _effective_contact_ids(source_contact_ids: list[str], dry_run: dict[str, Any] | None) -> list[str]:
    dry_status = _safe_text((dry_run or {}).get("status")).casefold()
    dry_ids = _split_ids((dry_run or {}).get("contact_id"))
    if dry_status == "dry_run" and len(dry_ids) == 1:
        return dry_ids
    return source_contact_ids


def _is_contact_id_mismatch(dry_run: dict[str, Any] | None) -> bool:
    reason = _safe_text((dry_run or {}).get("reason")).casefold()
    return "contact_id_mismatch" in reason


def _is_multi_contact_or_ambiguous(
    row: dict[str, Any],
    dry_run: dict[str, Any] | None,
    effective_contact_ids: list[str],
) -> bool:
    reason = _safe_text((dry_run or {}).get("reason")).casefold()
    if "multiple_exact_contacts" in reason or "multiple_amo_contact" in reason:
        return True
    if _safe_text((dry_run or {}).get("status")).casefold() == "skipped" and "multiple" in reason:
        return True
    source_ids = _split_ids(row.get("AMO contact IDs") or row.get("amo_contact_ids"))
    if len(source_ids) != 1:
        return True
    return len(effective_contact_ids) != 1


def _manager_review_reason(
    row: dict[str, Any],
    dry_run: dict[str, Any] | None,
    effective_contact_ids: list[str],
) -> str:
    dry_reason = _safe_text((dry_run or {}).get("reason"))
    if dry_reason:
        return "dry_run:" + dry_reason
    source_ids = _split_ids(row.get("AMO contact IDs") or row.get("amo_contact_ids"))
    if len(source_ids) > 1 or len(effective_contact_ids) > 1:
        return "ambiguous_multiple_contact_ids"
    return "missing_single_contact_reference"


def _is_deferred_non_sales_or_service(row: dict[str, Any]) -> bool:
    ready = _safe_text(row.get("Готово к записи в AMO")).casefold()
    call_type = _safe_text(row.get("Тип последнего свежего звонка")).casefold()
    blockers = _safe_text(row.get("CRM writeback blockers")).casefold()
    reason = _safe_text(row.get("Причина статуса AMO")).casefold()
    policy = _safe_text(row.get("CRM writeback policy")).casefold()

    if ready and ready not in {"да", "yes", "true", "1"}:
        return True
    if call_type in SERVICE_OR_NON_SALES_CALL_TYPES:
        return True
    service_markers = ("service", "existing-client", "existing_client", "non-sales", "non_conversation")
    if any(marker in blockers or marker in reason for marker in service_markers):
        return True
    return policy in {"deferred_non_sales_or_service", "service_context_manual_review"}


def _deferred_reason(row: dict[str, Any]) -> str:
    call_type = _safe_text(row.get("Тип последнего свежего звонка"))
    if call_type:
        return f"non_sales_or_service_call_type:{call_type}"
    reason = _safe_text(row.get("Причина статуса AMO") or row.get("CRM writeback blockers"))
    return reason or "source_row_not_live_sales_writeback_ready"


def _is_quality_blocker(row: dict[str, Any]) -> bool:
    decision = _safe_text(row.get("decision")).casefold()
    risk_types = _safe_text(row.get("risk_types"))
    if decision == "block":
        return True
    if risk_types:
        return True
    return False


def _read_report_rows(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = [dict(row) for row in payload["rows"] if isinstance(row, dict)]
            return [_with_report_path(row, path) for row in rows]
        report_from_summary = _report_path_from_summary(path, payload)
        if report_from_summary and report_from_summary != path:
            return _read_report_rows(report_from_summary)
        return []
    if path.suffix.lower() == ".csv":
        rows, _headers = _read_csv_with_headers(path)
        return [_with_report_path(row, path) for row in rows]
    return []


def _read_quality_rows(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
            report_csv = outputs.get("report_csv") or outputs.get("crm_text_quality_report_csv")
            if report_csv:
                report_path = Path(str(report_csv)).expanduser()
                if report_path.exists():
                    return _read_quality_rows(report_path)
            if isinstance(payload.get("rows"), list):
                return [_with_report_path(dict(row), path) for row in payload["rows"] if isinstance(row, dict)]
        return []
    rows, _headers = _read_csv_with_headers(path)
    return [_with_report_path(row, path) for row in rows]


def _report_path_from_summary(path: Path, payload: Any) -> Path | None:
    adjacent = path.with_name("contact_writeback_report.json")
    if adjacent.exists():
        return adjacent.resolve()
    if isinstance(payload, dict):
        report_dir = _safe_text(payload.get("report_dir") or (payload.get("summary") or {}).get("report_dir"))
        if report_dir:
            candidate = Path(report_dir).expanduser().resolve() / "contact_writeback_report.json"
            if candidate.exists():
                return candidate
    return None


def _expand_report_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            candidates = []
            candidates.extend(path.rglob("contact_writeback_report.json"))
            candidates.extend(path.rglob("contact_writeback_report.csv"))
            candidates.extend(path.rglob("*dry_run_report.csv"))
            candidates.extend(path.rglob("*skipped_contacts.csv"))
            result.extend(_dedupe_paths(candidates))
        elif path.exists():
            result.append(path)
    return _dedupe_paths(result)


def _expand_quality_report_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_dir():
            candidates = []
            candidates.extend(path.rglob("crm_writeback_quality_report.csv"))
            candidates.extend(path.rglob("*crm_text_quality*report*.csv"))
            result.extend(_dedupe_paths(candidates))
        elif path.exists():
            result.append(path)
    return _dedupe_paths(result)


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        result.append(resolved)
    return result


def _read_csv_with_headers(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
        return rows, list(reader.fieldnames or [])


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: _serialize_cell(row.get(header, "")) for header in headers})


def _write_xlsx(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> bool:
    if xlsxwriter is None:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = xlsxwriter.Workbook(str(path), {"constant_memory": True})
    worksheet = workbook.add_worksheet("queue")
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    header_format = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top"})
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
        worksheet.set_column(col, col, min(max(len(header) + 2, 12), 60))
    for row_idx, row in enumerate(rows, start=1):
        for col, header in enumerate(headers):
            worksheet.write(row_idx, col, _serialize_cell(row.get(header, "")), wrap)
    workbook.close()
    return True


def _queue_headers(source_headers: list[str]) -> list[str]:
    headers = list(QUEUE_COLUMNS)
    for header in source_headers:
        if header not in headers:
            headers.append(header)
    return headers


def _with_report_path(row: dict[str, Any], path: Path) -> dict[str, Any]:
    enriched = dict(row)
    enriched["__report_path"] = str(path)
    return enriched


def _normalize(value: Any) -> str | None:
    normalized = normalize_phone(_safe_text(value))
    return normalized or None


def _split_ids(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    result: list[str] = []
    for part in text.replace(",", "|").replace(";", "|").split("|"):
        item = part.strip()
        if not item:
            continue
        if item not in result:
            result.append(item)
    return result


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _serialize_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return _safe_text(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build safe amoCRM writeback bucket queue from AMO-ready CSV and reports.")
    parser.add_argument("--input", required=True, help="AMO-ready CSV.")
    parser.add_argument("--out-root", required=True, help="Output directory for bucket CSV/XLSX files and summary.json.")
    parser.add_argument(
        "--writeback-report",
        action="append",
        default=[],
        help="Live writeback report JSON/CSV or directory. May be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run-report",
        action="append",
        default=[],
        help="Dry-run report JSON/CSV or directory. May be passed multiple times.",
    )
    parser.add_argument(
        "--quality-report",
        action="append",
        default=[],
        help="CRM quality report CSV/summary JSON or directory. May be passed multiple times.",
    )
    parser.add_argument(
        "--manual-review-input",
        action="append",
        default=[],
        help="CSV/XLSX-like CSV with phones that must stay in manual/text quality review. May be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_amo_writeback_queue(
        AmoWritebackQueueConfig(
            input_csv=Path(args.input),
            out_root=Path(args.out_root),
            writeback_reports=tuple(Path(path) for path in args.writeback_report),
            dry_run_reports=tuple(Path(path) for path in args.dry_run_report),
            quality_reports=tuple(Path(path) for path in args.quality_report),
            manual_review_inputs=tuple(Path(path) for path in args.manual_review_input),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
