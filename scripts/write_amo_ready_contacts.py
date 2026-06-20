from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.crm_text_quality_detector import (
    detect_crm_text_quality_risks,
    has_blocking_crm_text_findings,
)
from mango_mvp.quality.tenant_text_normalizer import normalize_manager_text
from mango_mvp.crm_card_aggregator import apply_contact_card_payload, contact_ready_blocker
from mango_mvp.utils.phone import normalize_phone

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional runtime dependency
    pd = None

try:
    import xlsxwriter
except ImportError:  # pragma: no cover - optional runtime dependency
    xlsxwriter = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_EXPORT_POINTER = PROJECT_ROOT / "stable_runtime" / "CANONICAL_EXPORT.txt"
LEGACY_ROOT_AMO_READY_XLSX = PROJECT_ROOT / "АКТУАЛЬНО_AMO_ready.xlsx"
REPORT_ROOT = PROJECT_ROOT / "stable_runtime" / "amocrm_runtime" / "contact_writebacks"
TARGET_CONTACT_FIELDS = (
    "Статус матчинга",
    "AI-приоритет",
    "AI-рекомендованный следующий шаг",
    "Последняя AI-сводка",
    "Авто история общения",
)
REQUIRED_TEXTAREA_CONTACT_FIELDS = (
    "AI-рекомендованный следующий шаг",
    "Последняя AI-сводка",
    "Авто история общения",
)
ENV_FILES = (
    PROJECT_ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    PROJECT_ROOT / "prod_runtime_transfer" / ".env.private",
)


def _default_amo_ready_input() -> Path:
    """Use the active runtime export instead of the stale root Excel artifact."""
    if CANONICAL_EXPORT_POINTER.exists():
        export_name = CANONICAL_EXPORT_POINTER.read_text(encoding="utf-8").strip()
        if export_name:
            candidate = PROJECT_ROOT / "stable_runtime" / export_name / "amo_export_ready_ru.csv"
            if candidate.exists():
                return candidate
    return LEGACY_ROOT_AMO_READY_XLSX
LIVE_WRITE_CONFIRMATION = "WRITE_AMO_LIVE"
TEXT_COMPACTION_SUFFIX = " [сжато]"
MAX_AMO_TEXT_FIELD_CHARS = 240
AMO_TEXTAREA_FIELD_CHAR_LIMIT = 60000


def _quality_gate_summary_passed(path_value: str | None) -> bool:
    path_text = _safe_text(path_value)
    if not path_text:
        raise ValueError("Live amoCRM writeback requires --quality-gate-summary with Stage15 passed summary.json.")
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Quality gate summary does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Quality gate summary is not valid JSON: {path}") from exc
    if not bool(payload.get("passed")):
        raise ValueError("Quality gate summary is not passed.")
    readiness = payload.get("readiness") or {}
    if not bool(readiness.get("crm_quality_writeback_ready")):
        raise ValueError("Quality gate summary does not allow CRM quality writeback.")
    return True


def _crm_writeback_quality_summary_passed(
    path_value: str | None,
    *,
    expected_input: str | None = None,
) -> bool:
    path_text = _safe_text(path_value)
    if not path_text:
        raise ValueError("Live amoCRM writeback requires --crm-writeback-quality-summary from run_crm_writeback_quality_gate.py.")
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"CRM writeback quality summary does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"CRM writeback quality summary is not valid JSON: {path}") from exc
    if not bool(payload.get("passed")):
        raise ValueError("CRM writeback quality summary is not passed.")
    expected_input_text = _safe_text(expected_input)
    summary_input_text = _safe_text(payload.get("input"))
    if expected_input_text:
        if not summary_input_text:
            raise ValueError("CRM writeback quality summary is missing input path.")
        expected_input_path = Path(expected_input_text).expanduser().resolve()
        summary_input_path = Path(summary_input_text).expanduser().resolve()
        if summary_input_path != expected_input_path:
            raise ValueError(
                "CRM writeback quality summary input does not match --input: "
                f"{summary_input_path} != {expected_input_path}"
            )
    population = payload.get("population_recall") or {}
    if population and not bool(population.get("passed_for_live")):
        raise ValueError("CRM writeback population recall gate does not allow live writeback.")
    crm_text_quality = payload.get("crm_text_quality")
    if not isinstance(crm_text_quality, dict):
        raise ValueError("CRM writeback quality summary is missing crm_text_quality gate.")
    if not bool(crm_text_quality.get("passed_for_live")):
        raise ValueError("CRM text quality gate does not allow live writeback.")
    if int(crm_text_quality.get("blocking_rows") or 0) != 0:
        raise ValueError("CRM text quality gate has blocking rows.")
    return True


def _live_write_enabled(args: argparse.Namespace) -> bool:
    execute_live_write = bool(getattr(args, "execute_live_write", False))
    confirmation = str(getattr(args, "live_confirmation", "") or "").strip()
    if execute_live_write and confirmation != LIVE_WRITE_CONFIRMATION:
        raise ValueError(
            f"Live amoCRM writeback requires --live-confirmation {LIVE_WRITE_CONFIRMATION!r}."
        )
    if execute_live_write:
        _quality_gate_summary_passed(getattr(args, "quality_gate_summary", ""))
        _crm_writeback_quality_summary_passed(
            getattr(args, "crm_writeback_quality_summary", ""),
            expected_input=getattr(args, "input", ""),
        )
    if confirmation and not execute_live_write:
        raise ValueError("--live-confirmation is only valid together with --execute-live-write.")
    return execute_live_write


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or (pd is not None and pd.isna(value))):
        return ""
    return str(value).strip()


def _load_env_files() -> None:
    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    os.environ.setdefault(
        "DATABASE_URL",
        f"sqlite:///{(PROJECT_ROOT / 'stable_runtime' / 'amocrm_runtime' / 'amo_runtime.db').resolve()}",
    )


def _compose_last_summary(row: dict[str, Any]) -> str:
    summary = _safe_text(row.get("Краткое резюме последнего свежего звонка"))
    if summary:
        return _compact_without_ellipsis(summary, limit=AMO_TEXTAREA_FIELD_CHAR_LIMIT)
    history = _safe_text(row.get("Краткая история общения"))
    if not history:
        return ""
    return _compact_without_ellipsis(history, limit=AMO_TEXTAREA_FIELD_CHAR_LIMIT)


def _compose_auto_history(row: dict[str, Any]) -> str:
    blocks: list[str] = []

    history = normalize_manager_text(row.get("Краткая история общения"))
    chronology = normalize_manager_text(row.get("Хронология общения (последние 5 касаний)"))
    objections = normalize_manager_text(row.get("Возражения"))
    next_step = normalize_manager_text(row.get("Следующий шаг"))
    follow_up = _safe_text(row.get("Рекомендуемая дата следующего контакта"))
    priority = _safe_text(row.get("Приоритет лида"))
    probability = _safe_text(row.get("Вероятность продажи, %"))
    product = normalize_manager_text(row.get("Рекомендуемый продукт"))
    products = normalize_manager_text(row.get("Продукты интереса"))
    tallanto_history = normalize_manager_text(row.get("История общения Tallanto"))

    if history:
        blocks.append("Сводка клиента:\n" + history)

    facts: list[str] = []
    if product:
        facts.append(f"Рекомендуемый продукт: {product}")
    if products:
        facts.append(f"Продукты интереса: {products}")
    if objections:
        facts.append(f"Возражения: {objections}")
    if next_step:
        facts.append(f"Следующий шаг: {next_step}")
    if follow_up:
        facts.append(f"Рекомендуемая дата следующего контакта: {follow_up}")
    if priority:
        facts.append(f"Приоритет лида: {priority}")
    if probability:
        facts.append(f"Вероятность продажи, %: {probability}")
    if chronology:
        if os.getenv("CRM_AUTO_HISTORY_CHRONOLOGY_TEXT", "0") == "1" and not _is_redundant_history_block(
            history, chronology
        ):
            facts.append(
                "Хронология:\n"
                + _compact_without_ellipsis(chronology, limit=AMO_TEXTAREA_FIELD_CHAR_LIMIT)
            )
        else:
            facts.append("Хронология: есть в полной рабочей таблице")
    if facts:
        blocks.append("\n".join(facts))

    if tallanto_history:
        blocks.append("История общения Tallanto:\n" + tallanto_history)

    composed = "\n\n".join(block for block in blocks if block.strip()).strip()
    if len(composed) > AMO_TEXTAREA_FIELD_CHAR_LIMIT:
        composed = _compact_without_ellipsis(composed, limit=AMO_TEXTAREA_FIELD_CHAR_LIMIT)
    return composed


def _compact_without_ellipsis(text: Any, *, limit: int) -> str:
    value = normalize_manager_text(text)
    if len(value) <= limit:
        return value
    budget = max(20, limit - len(TEXT_COMPACTION_SUFFIX))
    candidate = value[:budget].rstrip()
    word_boundary = max(candidate.rfind(" "), candidate.rfind(","), candidate.rfind(";"), candidate.rfind("."))
    if word_boundary >= int(budget * 0.55):
        candidate = candidate[:word_boundary].rstrip(" ,;.")
    return f"{candidate}{TEXT_COMPACTION_SUFFIX}"


def _token_set(value: str) -> set[str]:
    return {token for token in re.findall(r"[а-яa-z0-9]{4,}", value.casefold()) if token}


def _is_redundant_history_block(history: str, chronology: str) -> bool:
    history_tokens = _token_set(history)
    chronology_tokens = _token_set(chronology)
    if len(chronology_tokens) < 5 or not history_tokens:
        return False
    return len(history_tokens & chronology_tokens) / max(len(chronology_tokens), 1) >= 0.8


SERVICE_CONTEXT_CALL_TYPES = {"service_call", "existing_client_progress", "technical_call"}


def _split_ids(value: Any) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []
    return [part for part in re.split(r"[|,;\s]+", text) if part.strip()]


def _expected_amo_contact_ids(row: dict[str, Any]) -> set[str]:
    return {part for part in _split_ids(row.get("AMO contact IDs")) if part.isdigit()}


def _contact_id_mismatch_reason(row: dict[str, Any], contact_id: int) -> str:
    expected_ids = _expected_amo_contact_ids(row)
    if expected_ids and str(contact_id) not in expected_ids:
        return "contact_id_mismatch_with_source_amo_contact_ids"
    return ""


def _contact_row_guard_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    ready = _safe_text(row.get("Готово к записи в AMO")).casefold()
    if ready not in {"да", "yes", "true", "1"}:
        reasons.append("row_not_marked_amo_ready")
    call_type = _safe_text(row.get("Тип последнего свежего звонка")).casefold()
    if call_type in SERVICE_CONTEXT_CALL_TYPES:
        reasons.append(f"service_or_existing_client_context:{call_type}")
    amo_ids = _split_ids(row.get("AMO contact IDs"))
    if len(amo_ids) != 1:
        reasons.append("missing_amo_contact_id" if not amo_ids else "multiple_amo_contact_ids")
    policy = _safe_text(row.get("CRM writeback policy"))
    if policy and policy != "live_update_ready":
        reasons.append(f"crm_writeback_policy:{policy}")
    blockers = _safe_text(row.get("CRM writeback blockers"))
    if blockers:
        reasons.append(f"crm_writeback_blockers:{blockers}")
    card_blocker = contact_ready_blocker(row)
    if card_blocker:
        reasons.append(card_blocker)
    return reasons


def _find_catalog_field(field_catalog: list[dict[str, Any]], field_name: str) -> dict[str, Any] | None:
    normalized = field_name.strip().casefold()
    for item in field_catalog:
        if str(item.get("name") or "").strip().casefold() == normalized:
            return item
    return None


def _contact_field_catalog_guard_reasons(field_catalog: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for field_name in TARGET_CONTACT_FIELDS:
        meta = _find_catalog_field(field_catalog, field_name)
        if meta is None or meta.get("id") is None:
            reasons.append(f"missing_contact_field:{field_name}")
            continue
        field_type = _safe_text(meta.get("type")).casefold()
        if field_name in REQUIRED_TEXTAREA_CONTACT_FIELDS:
            if field_type != "textarea":
                reasons.append(f"contact_field_not_textarea:{field_name}:{field_type or '<missing>'}")
            if bool(meta.get("is_api_only")):
                reasons.append(f"contact_field_api_only_not_supported:{field_name}")
            if not _safe_text(meta.get("group_id")):
                reasons.append(f"contact_field_missing_group:{field_name}")
    return reasons


def _build_contact_payload(row: dict[str, Any]) -> dict[str, Any]:
    card_payload = apply_contact_card_payload(row)
    if card_payload is not None:
        return card_payload
    payload = {
        "Статус матчинга": _safe_text(row.get("Статус матчинга Tallanto")),
        "AI-приоритет": _safe_text(row.get("Приоритет лида")),
        "AI-рекомендованный следующий шаг": _compact_without_ellipsis(
            row.get("Следующий шаг"),
            limit=AMO_TEXTAREA_FIELD_CHAR_LIMIT,
        ),
        "Последняя AI-сводка": _compose_last_summary(row),
        "Авто история общения": _compose_auto_history(row),
    }
    return {key: value for key, value in payload.items() if value}


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if pd is None:
        raise RuntimeError("Reading .xlsx input requires pandas/openpyxl. Use CSV input in this runtime.")
    frame = pd.read_excel(path).fillna("")
    return frame.to_dict(orient="records")


def _write_report_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            headers.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            safe_row = {}
            for key in headers:
                value = row.get(key, "")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                safe_row[key] = value
            writer.writerow(safe_row)


def _write_report_xlsx(path: Path, rows: list[dict[str, Any]]) -> None:
    if pd is not None:
        pd.DataFrame(rows).to_excel(path, index=False)
        return
    if xlsxwriter is None:
        return
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            headers.append(key)
    workbook = xlsxwriter.Workbook(str(path), {"constant_memory": True})
    worksheet = workbook.add_worksheet("report")
    wrap = workbook.add_format({"text_wrap": True, "valign": "top"})
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, wrap)
        worksheet.set_column(col, col, min(max(len(header) + 2, 12), 60))
    for row_idx, row in enumerate(rows, start=1):
        for col, header in enumerate(headers):
            value = row.get(header, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            worksheet.write(row_idx, col, _safe_text(value), wrap)
    workbook.close()


def _load_skip_phones(report_path: Path | None) -> set[str]:
    if report_path is None or not report_path.exists():
        return set()
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    rows = payload.get("rows") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return set()
    result: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _safe_text(row.get("status")) != "written":
            continue
        phone = normalize_phone(row.get("phone"))
        if phone:
            result.add(phone)
    return result


def _call_with_retry(fn, *args, **kwargs):
    delays = (1.5, 3.0, 6.0)
    for attempt in range(len(delays) + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            message = str(exc)
            if "HTTP 429" in message or "Failed to reach amoCRM" in message:
                if attempt >= len(delays):
                    raise
                time.sleep(delays[attempt])
                continue
            raise


def _preflight_runtime_db(session: Any) -> tuple[bool, str]:
    try:
        from sqlalchemy import text

        session.execute(text("SELECT 1"))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run/live writeback AMO-ready export в контакты amoCRM.")
    parser.add_argument("--input", default=str(_default_amo_ready_input()), help="Путь к .xlsx/.csv с AMO-ready. По умолчанию берется активный export из stable_runtime/CANONICAL_EXPORT.txt.")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число строк для записи.")
    parser.add_argument(
        "--skip-written-from-report",
        default=None,
        help="JSON-отчет прошлого прогона; контакты со статусом written будут пропущены.",
    )
    parser.add_argument(
        "--execute-live-write",
        action="store_true",
        help="Разрешить live-запись в amoCRM. Без этого флага скрипт делает только dry-run отчет.",
    )
    parser.add_argument(
        "--offline-preview",
        action="store_true",
        help=(
            "Не подключаться к amoCRM и не искать контакты; только собрать payload preview. "
            "Используется, когда DB tunnel/OAuth runtime недоступен."
        ),
    )
    parser.add_argument(
        "--live-confirmation",
        default="",
        help=f"Контрольная строка для live-записи: {LIVE_WRITE_CONFIRMATION}.",
    )
    parser.add_argument(
        "--quality-gate-summary",
        default="",
        help="Путь к Stage15 summary.json с passed=true и crm_quality_writeback_ready=true.",
    )
    parser.add_argument(
        "--crm-writeback-quality-summary",
        default="",
        help="Путь к summary.json от run_crm_writeback_quality_gate.py; обязателен для live-записи.",
    )
    parser.add_argument("--expected-written", type=int, default=None, help="Fail if live written count differs.")
    parser.add_argument("--expected-dry-run", type=int, default=None, help="Fail if dry-run count differs.")
    args = parser.parse_args()
    if args.offline_preview and args.execute_live_write:
        print("Refusing live amoCRM writeback: --offline-preview cannot be combined with --execute-live-write.", file=sys.stderr)
        return 2
    try:
        live_write = _live_write_enabled(args)
    except ValueError as exc:
        print(f"Refusing live amoCRM writeback: {exc}", file=sys.stderr)
        return 2

    search_contacts_by_phone = None
    send_contact_custom_field_update = None
    SessionLocal = None
    if not args.offline_preview:
        _load_env_files()

        from mango_mvp.amocrm_runtime.amo_integration import (
            fetch_contact_field_catalog,
            search_contacts_by_phone,
            send_contact_custom_field_update,
        )
        from mango_mvp.amocrm_runtime.db import SessionLocal

    input_path = Path(args.input).resolve()
    rows = _read_rows(input_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    skip_phones = _load_skip_phones(Path(args.skip_written_from_report).resolve()) if args.skip_written_from_report else set()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = REPORT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    session = SessionLocal() if SessionLocal is not None else None
    report_rows: list[dict[str, Any]] = []
    try:
        if session is not None:
            ok, error = _preflight_runtime_db(session)
            if not ok:
                summary = {
                    "run_id": run_id,
                    "mode": "live_write" if live_write else "dry_run",
                    "live_write": live_write,
                    "offline_preview": bool(args.offline_preview),
                    "input": str(input_path),
                    "total_rows": len(rows),
                    "written": 0,
                    "dry_run": 0,
                    "offline_preview_rows": 0,
                    "skipped": 0,
                    "failed": 0,
                    "preflight_failed": True,
                    "preflight_error": error,
                    "target_fields": list(TARGET_CONTACT_FIELDS),
                    "report_dir": str(run_dir),
                }
                (run_dir / "contact_writeback_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (run_dir / "runtime_preflight_error.txt").write_text(error, encoding="utf-8")
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return 2
            field_catalog = fetch_contact_field_catalog(session, force_refresh=True)
            catalog_guard_reasons = _contact_field_catalog_guard_reasons(field_catalog)
            if catalog_guard_reasons:
                summary = {
                    "run_id": run_id,
                    "mode": "live_write" if live_write else "dry_run",
                    "live_write": live_write,
                    "offline_preview": bool(args.offline_preview),
                    "input": str(input_path),
                    "total_rows": len(rows),
                    "written": 0,
                    "dry_run": 0,
                    "offline_preview_rows": 0,
                    "skipped": 0,
                    "failed": 0,
                    "preflight_failed": True,
                    "preflight_error": "AMO contact field catalog is not safe for writeback: "
                    + " | ".join(catalog_guard_reasons),
                    "field_catalog_guard_reasons": catalog_guard_reasons,
                    "target_fields": list(TARGET_CONTACT_FIELDS),
                    "report_dir": str(run_dir),
                }
                (run_dir / "contact_writeback_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (run_dir / "runtime_preflight_error.txt").write_text(summary["preflight_error"], encoding="utf-8")
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return 2
        total = len(rows)
        for index, source_row in enumerate(rows, start=1):
            phone = normalize_phone(_safe_text(source_row.get("Телефон клиента")))
            report_row = {
                "row_index": index,
                "mode": "live_write" if live_write else "dry_run",
                "phone": phone or _safe_text(source_row.get("Телефон клиента")),
                "status": "",
                "reason": "",
                "contact_id": "",
                "contact_name": "",
                "updated_fields": [],
                "preview_payload": {},
                "payload_sha256": "",
            }
            if not phone:
                report_row["status"] = "skipped"
                report_row["reason"] = "invalid_phone"
                report_rows.append(report_row)
                continue
            if phone in skip_phones:
                report_row["status"] = "skipped"
                report_row["reason"] = "written_in_previous_report"
                report_rows.append(report_row)
                continue

            guard_reasons = _contact_row_guard_reasons(source_row)
            if live_write and guard_reasons:
                report_row["status"] = "skipped"
                report_row["reason"] = "live_guard:" + " | ".join(guard_reasons)
                report_rows.append(report_row)
                continue

            payload = _build_contact_payload(source_row)
            if not payload:
                report_row["status"] = "skipped"
                report_row["reason"] = "empty_payload"
                report_rows.append(report_row)
                continue
            report_row["preview_payload"] = payload
            report_row["payload_sha256"] = _payload_sha256(payload)
            payload_text_findings = detect_crm_text_quality_risks(payload, min_severity="P2")
            if live_write and has_blocking_crm_text_findings(payload_text_findings):
                report_row["status"] = "skipped"
                report_row["reason"] = "live_guard:crm_text_quality:" + " | ".join(
                    sorted({finding.risk_type for finding in payload_text_findings})
                )
                report_row["updated_fields"] = list(payload.keys())
                report_rows.append(report_row)
                continue
            if args.offline_preview:
                report_row["status"] = "offline_preview"
                report_row["reason"] = "amo_lookup_not_executed"
                report_row["updated_fields"] = list(payload.keys())
                report_rows.append(report_row)
                continue

            try:
                assert session is not None
                assert search_contacts_by_phone is not None
                assert send_contact_custom_field_update is not None
                contacts = _call_with_retry(search_contacts_by_phone, session, phone=phone, limit=10)
                if not contacts:
                    report_row["status"] = "skipped"
                    report_row["reason"] = "contact_not_found_in_amo"
                    report_rows.append(report_row)
                    continue
                if len(contacts) > 1:
                    report_row["status"] = "skipped"
                    report_row["reason"] = "multiple_exact_contacts_in_amo"
                    report_row["contact_id"] = " | ".join(str(int(item.get("id") or 0)) for item in contacts)
                    report_rows.append(report_row)
                    continue

                contact = contacts[0]
                contact_id = int(contact.get("id") or 0)
                contact_name = _safe_text(contact.get("name"))
                mismatch_reason = _contact_id_mismatch_reason(source_row, contact_id)
                if mismatch_reason:
                    report_row["status"] = "skipped"
                    report_row["reason"] = mismatch_reason
                    report_row["contact_id"] = contact_id
                    report_row["contact_name"] = contact_name
                    report_rows.append(report_row)
                    continue
                if not live_write:
                    report_row["status"] = "dry_run"
                    report_row["reason"] = "live_write_not_confirmed"
                    report_row["contact_id"] = contact_id
                    report_row["contact_name"] = contact_name
                    report_row["updated_fields"] = list(payload.keys())
                    report_rows.append(report_row)
                    continue

                result = _call_with_retry(
                    send_contact_custom_field_update,
                    session,
                    contact_id=contact_id,
                    field_payload=payload,
                )
                session.commit()
                report_row["status"] = "written"
                report_row["contact_id"] = contact_id
                report_row["contact_name"] = contact_name
                report_row["updated_fields"] = result.get("updated_fields") or []
            except Exception as exc:
                session.rollback()
                report_row["status"] = "failed"
                report_row["reason"] = str(exc)

            report_rows.append(report_row)
            if index % 25 == 0 or index == total:
                written = sum(1 for row in report_rows if row["status"] == "written")
                dry_run = sum(1 for row in report_rows if row["status"] == "dry_run")
                failed = sum(1 for row in report_rows if row["status"] == "failed")
                print(f"[{index}/{total}] written={written} dry_run={dry_run} failed={failed}", flush=True)
    finally:
        if session is not None:
            session.close()

    summary = {
        "run_id": run_id,
        "mode": "live_write" if live_write else "dry_run",
        "live_write": live_write,
        "offline_preview": bool(args.offline_preview),
        "input": str(input_path),
        "total_rows": len(rows),
        "written": sum(1 for row in report_rows if row["status"] == "written"),
        "dry_run": sum(1 for row in report_rows if row["status"] == "dry_run"),
        "offline_preview_rows": sum(1 for row in report_rows if row["status"] == "offline_preview"),
        "skipped": sum(1 for row in report_rows if row["status"] == "skipped"),
        "failed": sum(1 for row in report_rows if row["status"] == "failed"),
        "expected_written": args.expected_written,
        "expected_dry_run": args.expected_dry_run,
        "expected_count_mismatch": False,
        "target_fields": list(TARGET_CONTACT_FIELDS),
        "report_dir": str(run_dir),
    }
    if args.expected_written is not None and summary["written"] != args.expected_written:
        summary["expected_count_mismatch"] = True
    if args.expected_dry_run is not None and summary["dry_run"] != args.expected_dry_run:
        summary["expected_count_mismatch"] = True

    (run_dir / "contact_writeback_report.json").write_text(
        json.dumps({"summary": summary, "rows": report_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report_csv(run_dir / "contact_writeback_report.csv", report_rows)
    _write_report_xlsx(run_dir / "contact_writeback_report.xlsx", report_rows)
    (run_dir / "contact_writeback_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 and not summary["expected_count_mismatch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
