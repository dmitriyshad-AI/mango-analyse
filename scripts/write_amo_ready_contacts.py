from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from mango_mvp.utils.phone import normalize_phone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_XLSX = PROJECT_ROOT / "АКТУАЛЬНО_AMO_ready.xlsx"
REPORT_ROOT = PROJECT_ROOT / "stable_runtime" / "amocrm_runtime" / "contact_writebacks"
TARGET_CONTACT_FIELDS = (
    "Статус матчинга",
    "AI-приоритет",
    "AI-рекомендованный следующий шаг",
    "Последняя AI-сводка",
    "Авто история общения",
)
ENV_FILES = (
    PROJECT_ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    PROJECT_ROOT / "prod_runtime_transfer" / ".env.private",
)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
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
        return summary
    history = _safe_text(row.get("Краткая история общения"))
    if not history:
        return ""
    return history if len(history) <= 252 else history[:252].rstrip() + "..."


def _compose_auto_history(row: dict[str, Any]) -> str:
    blocks: list[str] = []

    history = _safe_text(row.get("Краткая история общения"))
    chronology = _safe_text(row.get("Хронология общения (последние 5 касаний)"))
    objections = _safe_text(row.get("Возражения"))
    next_step = _safe_text(row.get("Следующий шаг"))
    follow_up = _safe_text(row.get("Рекомендуемая дата следующего контакта"))
    priority = _safe_text(row.get("Приоритет лида"))
    probability = _safe_text(row.get("Вероятность продажи, %"))
    product = _safe_text(row.get("Рекомендуемый продукт"))
    products = _safe_text(row.get("Продукты интереса"))
    tallanto_history = _safe_text(row.get("История общения Tallanto"))

    if history:
        blocks.append("Сводка клиента:\n" + history)
    if chronology:
        blocks.append("Хронология общения (последние 5 касаний):\n" + chronology)

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
    if facts:
        blocks.append("\n".join(facts))

    if tallanto_history:
        blocks.append("История общения Tallanto:\n" + tallanto_history)

    return "\n\n".join(block for block in blocks if block.strip()).strip()


def _build_contact_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "Статус матчинга": _safe_text(row.get("Статус матчинга Tallanto")),
        "AI-приоритет": _safe_text(row.get("Приоритет лида")),
        "AI-рекомендованный следующий шаг": _safe_text(row.get("Следующий шаг")),
        "Последняя AI-сводка": _compose_last_summary(row),
        "Авто история общения": _compose_auto_history(row),
    }
    return {key: value for key, value in payload.items() if value}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path)
    frame = frame.fillna("")
    return frame.to_dict(orient="records")


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Записать АКТУАЛЬНО_AMO_ready в контакты amoCRM.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_XLSX), help="Путь к .xlsx/.csv с AMO_ready.")
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число строк для записи.")
    parser.add_argument(
        "--skip-written-from-report",
        default=None,
        help="JSON-отчет прошлого прогона; контакты со статусом written будут пропущены.",
    )
    args = parser.parse_args()

    _load_env_files()

    from mango_mvp.amocrm_runtime.amo_integration import (
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

    session = SessionLocal()
    report_rows: list[dict[str, Any]] = []
    try:
        total = len(rows)
        for index, source_row in enumerate(rows, start=1):
            phone = normalize_phone(_safe_text(source_row.get("Телефон клиента")))
            report_row = {
                "row_index": index,
                "phone": phone or _safe_text(source_row.get("Телефон клиента")),
                "status": "",
                "reason": "",
                "contact_id": "",
                "contact_name": "",
                "updated_fields": [],
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

            payload = _build_contact_payload(source_row)
            if not payload:
                report_row["status"] = "skipped"
                report_row["reason"] = "empty_payload"
                report_rows.append(report_row)
                continue

            try:
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
                failed = sum(1 for row in report_rows if row["status"] == "failed")
                print(f"[{index}/{total}] written={written} failed={failed}", flush=True)
    finally:
        session.close()

    summary = {
        "run_id": run_id,
        "input": str(input_path),
        "total_rows": len(rows),
        "written": sum(1 for row in report_rows if row["status"] == "written"),
        "skipped": sum(1 for row in report_rows if row["status"] == "skipped"),
        "failed": sum(1 for row in report_rows if row["status"] == "failed"),
        "target_fields": list(TARGET_CONTACT_FIELDS),
        "report_dir": str(run_dir),
    }

    (run_dir / "contact_writeback_report.json").write_text(
        json.dumps({"summary": summary, "rows": report_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(report_rows).to_csv(run_dir / "contact_writeback_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(report_rows).to_excel(run_dir / "contact_writeback_report.xlsx", index=False)
    (run_dir / "contact_writeback_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
