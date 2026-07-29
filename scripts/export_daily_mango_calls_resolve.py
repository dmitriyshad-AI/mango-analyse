#!/usr/bin/env python3
"""Read-only daily export of Mango calls for the head of sales."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from zoneinfo import ZoneInfo

from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill

from mango_mvp.productization.capture_staging import sanitize_filename_part
from mango_mvp.services.export_excel import call_to_row
from mango_mvp.utils.filename_repair import repair_manager_name


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "product_data/mango_calls_two_processes"
DEFAULT_READY_DB = PIPELINE_ROOT / "drop/mango_calls_ready.sqlite"
DEFAULT_WORKING_DB = PIPELINE_ROOT / "working/mango_calls_pipeline.sqlite"
DEFAULT_OUT = Path("/Users/dmitrijfabarisov/Yandex.Disk.localized/Mango Calls Resolve")
DEFAULT_MANAGER_USERS = ROOT / (
    "_local_archive_mango_api_downloads_20260507/quarantine_import/"
    "raw_payload_archive/mango_users_config_20260507.json"
)
MOSCOW = ZoneInfo("Europe/Moscow")
TRANSCRIPT_CHUNK = 30_000

CALL_TYPE_RU = {
    "sales_call": "Продажа / подбор обучения",
    "service_call": "Сервисный вопрос",
    "existing_client_progress": "Текущий клиент / продолжение",
    "technical_call": "Технический вопрос",
    "non_conversation": "Разговор не состоялся",
}
STATUS_RU = {"done": "Готово", "skipped": "Пропущено по правилу", "manual": "Нужна ручная проверка", "pending": "Ожидает", "failed": "Ошибка", "in_progress": "В работе"}
PRICE_RU = {"high": "Высокая", "medium": "Средняя", "low": "Низкая"}
CHANNEL_RU = {"phone": "Телефон", "email": "Электронная почта", "telegram": "Telegram", "whatsapp": "WhatsApp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readonly_uri(path: Path, *, immutable: bool = False) -> str:
    option = "&immutable=1" if immutable else ""
    return f"file:{quote(str(path.resolve()), safe='/:')}?mode=ro{option}"


def verify_ready_drop(db: Path) -> dict[str, Any]:
    manifest_path = db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {"sha256": sha256_file(db), "size_bytes": db.stat().st_size}
    with sqlite3.connect(readonly_uri(db, immutable=True), uri=True) as con:
        actual["quick_check"] = str(con.execute("PRAGMA quick_check").fetchone()[0])
    expected = (manifest.get("sha256"), int(manifest.get("size_bytes") or 0), manifest.get("quick_check"), manifest.get("status"))
    observed = (actual["sha256"], actual["size_bytes"], actual["quick_check"], "ready")
    if expected != observed:
        raise RuntimeError("готовая база и её контрольный файл не совпадают")
    return {**actual, "published_at": manifest.get("published_at")}


def load_manager_map(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for item in payload.get("users", []):
        extension = str((item.get("telephony") or {}).get("extension") or "").strip()
        name = repair_manager_name((item.get("general") or {}).get("name"))
        if extension and name and name.casefold() != "admin":
            result[extension] = name
    return result


def parse_json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def day_bounds_utc(day: date) -> tuple[str, str]:
    start = datetime.combine(day, time.min, MOSCOW)
    end = start + timedelta(days=1)
    return tuple(value.astimezone(timezone.utc).replace(tzinfo=None).isoformat(sep=" ") for value in (start, end))  # type: ignore[return-value]


def read_day(db: Path, day: date, *, immutable: bool) -> list[sqlite3.Row]:
    with sqlite3.connect(readonly_uri(db, immutable=immutable), uri=True, timeout=30) as con:
        con.row_factory = sqlite3.Row
        start, end = day_bounds_utc(day)
        return con.execute(
            "SELECT * FROM call_records WHERE started_at>=? AND started_at<? ORDER BY started_at,id",
            (start, end),
        ).fetchall()


def translate_transcript(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"(?m)^\s*MANAGER\s*:", "Менеджер:", text, flags=re.IGNORECASE)
    return re.sub(r"(?m)^\s*CLIENT\s*:", "Клиент:", text, flags=re.IGNORECASE)


def evidenced_fio(value: Any, transcript: str) -> str:
    fio = " ".join(str(value or "").split())
    if len(fio.split()) < 2:
        return ""
    return fio if fio.casefold() in transcript.casefold() else ""


def clean_summary(value: Any) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(
        r"^\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}\s+менеджер\s+\S+\s+(?:общался с клиентом|пытался связаться с клиентом)\.\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?:^|\s)Приоритет лида:\s*[^.]+\.\s*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:^|\s)Итог:\s*Оценка на основе содержания звонка\.\s*", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def processing_issues(row: sqlite3.Row, analysis: Mapping[str, Any], resolve: Mapping[str, Any]) -> list[str]:
    flags = analysis.get("quality_flags") if isinstance(analysis.get("quality_flags"), dict) else {}
    issues: list[str] = []
    if row["resolve_status"] not in {"done", "skipped"}:
        issues.append("Разделение ролей не завершено автоматически")
    if row["analysis_status"] != "done":
        issues.append("Смысловой анализ не готов")
    if analysis.get("needs_review"):
        issues.append("Смысловой анализ запросил ручную проверку")
    if flags.get("transcript_quality_requires_manual_review"):
        issues.append("Качество расшифровки требует проверки")
    if resolve.get("decision") == "manual_review_required" or row["resolve_status"] == "manual":
        issues.append("Нужно вручную проверить разделение реплик")
    return list(dict.fromkeys(issues))


def normalize_row(row: sqlite3.Row, names: Mapping[str, str]) -> dict[str, Any]:
    analysis, resolve = parse_json(row["analysis_json"]), parse_json(row["resolve_json"])
    raw = dict(row)
    started_utc = datetime.fromisoformat(str(row["started_at"])).replace(tzinfo=timezone.utc)
    raw["started_at"] = started_utc
    base = call_to_row(SimpleNamespace(**raw), analysis) if analysis else {}
    started = started_utc.astimezone(MOSCOW)
    transcript = translate_transcript(row["transcript_text"])
    extension = str(row["manager_name"] or "").strip()
    resolve_ok = row["resolve_status"] == "done" or (row["resolve_status"] == "skipped" and resolve.get("decision") == "skip_short_call")
    complete = bool(row["transcription_status"] == "done" and resolve_ok and row["analysis_status"] == "done" and analysis)
    chunks = [transcript[i : i + TRANSCRIPT_CHUNK] for i in range(0, len(transcript), TRANSCRIPT_CHUNK)] or [""]
    issues = processing_issues(row, analysis, resolve)
    return {
        "id": int(row["id"]), "call_id": str(row["source_call_id"] or row["id"]), "started": started,
        "extension": extension, "manager": names.get(extension, ""), "direction": "Входящий" if row["direction"] == "inbound" else "Исходящий",
        "phone": str(row["phone"] or ""), "duration": float(row["duration_sec"] or 0), "source": Path(str(row["source_file"])),
        "complete": complete, "issues": issues, "chunks": chunks, "base": base,
        "parent_fio": evidenced_fio(base.get("parent_fio"), transcript), "child_fio": evidenced_fio(base.get("child_fio"), transcript),
        "resolve_status": str(row["resolve_status"] or ""), "analysis_status": str(row["analysis_status"] or ""),
    }


def merged_day_rows(ready_db: Path, working_db: Path, day: date, names: Mapping[str, str]) -> tuple[list[dict[str, Any]], int]:
    ready = read_day(ready_db, day, immutable=True)
    working = read_day(working_db, day, immutable=False)
    ready_ids = {str(row["source_call_id"] or row["id"]) for row in ready}
    pending = [row for row in working if str(row["source_call_id"] or row["id"]) not in ready_ids]
    merged = [normalize_row(row, names) for row in [*ready, *pending]]
    merged.sort(key=lambda item: (item["started"], item["id"]))
    return merged, len(pending)


def publish_audio(rows: Sequence[dict[str, Any]], audio_root: Path, target: Path) -> tuple[int, int]:
    audio_root = audio_root.resolve()
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    copied = reused = 0
    for row in rows:
        source = row["source"].resolve()
        if not source.is_file() or source.stat().st_size <= 0 or not source.is_relative_to(audio_root):
            raise RuntimeError(f"небезопасный или отсутствующий аудиофайл звонка {row['call_id']}")
        source_sha = sha256_file(source)
        filename = f"{row['started']:%Y-%m-%d__%H-%M-%S}__call_{sanitize_filename_part(row['call_id'])}__{source_sha[:12]}{source.suffix.lower()}"
        destination = target / filename
        if destination.exists():
            if sha256_file(destination) != source_sha:
                raise RuntimeError(f"конфликт аудиофайла {destination.name}")
            reused += 1
        else:
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != source_sha:
                raise RuntimeError(f"ошибка проверки копии {destination.name}")
            os.replace(temporary, destination)
            copied += 1
        destination.chmod(0o600)
        row["audio"], row["audio_sha256"] = destination, source_sha
    return copied, reused


def workbook_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    parts = max((len(row["chunks"]) for row in rows), default=1)
    headers = [
        "Дата и время", "ФИО менеджера", "Добавочный номер", "Направление", "Телефон клиента",
        "ФИО собеседника из разговора (не подтверждено CRM)", "ФИО ученика из разговора (не подтверждено CRM)",
        "Длительность, сек", "Статус обработки", "Тип звонка по смысловому анализу", "Краткое содержание разговора",
        "Продукт", "Предметы", "Формат", "Целевые экзамены", "Класс", "Школа", "Возражения и ограничения",
        "Следующий шаг", "Срок следующего шага", "Предпочтительный канал", "Озвученный бюджет",
        "Чувствительность к цене", "Интерес к скидке", "Нужна проверка", "Причина проверки", "Аудиозапись",
    ] + [f"Расшифровка разговора, часть {i}" for i in range(1, parts + 1)]
    values: list[list[Any]] = []
    for row in rows:
        base = row["base"]
        issues = "; ".join(row["issues"])
        values.append([
            row["started"].replace(tzinfo=None), row["manager"], row["extension"], row["direction"], row["phone"], row["parent_fio"], row["child_fio"],
            round(row["duration"], 1), "Готово" if row["complete"] else "Обработка не завершена",
            CALL_TYPE_RU.get(base.get("call_type", ""), base.get("call_type", "")), clean_summary(base.get("history_summary", "")),
            base.get("interests_products") or base.get("recommended_product", ""), base.get("interests_subjects", ""),
            base.get("interests_format", ""), base.get("exam_targets", ""), base.get("grade_current", ""), base.get("school", ""),
            base.get("objections", ""), base.get("next_step_action", ""), base.get("next_step_due_raw", ""),
            CHANNEL_RU.get(base.get("preferred_channel", ""), base.get("preferred_channel", "")), base.get("budget", ""),
            PRICE_RU.get(base.get("price_sensitivity", ""), base.get("price_sensitivity", "")),
            "Да" if str(base.get("discount_interest", "")).casefold() in {"true", "yes", "да", "1"} else "",
            "Да" if issues else "Нет", issues, row["audio"].name,
        ] + row["chunks"] + [""] * (parts - len(row["chunks"])))
    return headers, values


def append_safe(sheet: Any, values: Sequence[Any]) -> None:
    sheet.append([ILLEGAL_CHARACTERS_RE.sub("", value) if isinstance(value, str) else value for value in values])
    for cell in sheet[sheet.max_row]:
        if isinstance(cell.value, str) and cell.value.startswith(("=", "+", "-", "@")):
            cell.data_type = "s"


def format_sheet(sheet: Any, *, table: bool = True) -> None:
    if table:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    style_header(sheet[1])
    for column in sheet.columns:
        header = str(column[0].value or "")
        width = 65 if "Расшифровка" in header else 42 if header in {"Краткое содержание разговора", "Возражения и ограничения", "Следующий шаг", "Причина проверки"} else 18
        sheet.column_dimensions[column[0].column_letter].width = width
        for cell in column[1:]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if isinstance(cell.value, datetime):
                cell.number_format = "dd.mm.yyyy hh:mm:ss"


def style_header(cells: Sequence[Any]) -> None:
    for cell in cells:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def write_workbook(path: Path, day: date, rows: Sequence[dict[str, Any]], manager_source: str, source_meta: Mapping[str, Any]) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    ready, unfinished = [row for row in rows if row["complete"]], [row for row in rows if not row["complete"]]
    summary = wb.create_sheet("Сводка")
    for line in (["Ежедневный отчёт РОПа", day.isoformat()], ["Всего звонков", len(rows)], ["Полностью обработано", len(ready)],
                 ["Обработка не завершена", len(unfinished)], ["Требуют проверки", sum(bool(row["issues"]) for row in rows)],
                 ["ФИО менеджера не найдено", sum(not row["manager"] for row in rows)],
                 ["Источник ФИО менеджеров", manager_source or "не задан"], ["Готовый снимок опубликован", source_meta.get("published_at") or ""]):
        append_safe(summary, line)
    summary.append([])
    append_safe(summary, ["Менеджер", "Полностью обработанных звонков", "Часов"])
    counts = Counter((row["manager"] or f"Добавочный {row['extension']}") for row in ready)
    for manager, count in counts.most_common():
        hours = sum(row["duration"] for row in ready if (row["manager"] or f"Добавочный {row['extension']}") == manager) / 3600
        append_safe(summary, [manager, count, round(hours, 2)])
    for title, subset in (("Звонки", ready), ("Проблемы данных", [row for row in rows if row["issues"]])):
        sheet = wb.create_sheet(title)
        headers, values = workbook_rows(subset)
        append_safe(sheet, headers)
        for value_row, source_row in zip(values, subset):
            append_safe(sheet, value_row)
            audio_cell = sheet.cell(sheet.max_row, headers.index("Аудиозапись") + 1)
            audio_cell.hyperlink, audio_cell.style = source_row["audio"].as_uri(), "Hyperlink"
        format_sheet(sheet)
    description = wb.create_sheet("Описание полей")
    for line in (["Правило", "Описание"], ["Период", "Полные календарные сутки по Москве."],
                 ["ФИО менеджера", "Из архивного справочника Mango; пустое значение означает, что подтверждённого соответствия нет."],
                 ["ФИО клиента/ученика", "Из смыслового анализа, только если полное имя найдено в расшифровке; CRM не подтверждено."],
                 ["Расшифровка", "Метки ролей переведены на русский; текст не сокращается, длинные диалоги разбиты на соседние столбцы."],
                 ["Следующий шаг и возражения", "Подсказка смыслового анализа, а не автоматическое поручение; сверить с записью и CRM."],
                 ["Незавершённые", "На листе «Проблемы данных»; не смешиваются с полностью обработанными звонками."],
                 ["Использование", "Внутренний отчёт. Не применять для санкций или KPI без прослушивания и контекста CRM."]):
        append_safe(description, line)
    format_sheet(summary, table=False)
    style_header(summary[9])
    summary.column_dimensions["A"].width = 32
    summary.column_dimensions["B"].width = 36
    summary.column_dimensions["C"].width = 12
    format_sheet(description)
    temporary = path.with_suffix(".tmp.xlsx")
    wb.save(temporary)
    checked = load_workbook(temporary, read_only=True, data_only=False)
    if checked.sheetnames != ["Сводка", "Звонки", "Проблемы данных", "Описание полей"] or checked["Звонки"].max_row != len(ready) + 1:
        raise RuntimeError("проверка созданной таблицы не пройдена")
    checked.close()
    os.replace(temporary, path)
    path.chmod(0o600)


def export_day(ready_db: Path, working_db: Path, output_root: Path, day: date, manager_users: Path | None) -> Mapping[str, Any]:
    if day >= datetime.now(MOSCOW).date():
        raise ValueError("можно выгружать только завершённые сутки по Москве")
    source_before = verify_ready_drop(ready_db)
    rows, working_only = merged_day_rows(ready_db, working_db, day, load_manager_map(manager_users))
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_root.chmod(0o700)
    audio_dir = output_root / f"Записи разговоров {day.isoformat()}"
    copied, reused = publish_audio(rows, PIPELINE_ROOT / "working/audio", audio_dir)
    source_after = verify_ready_drop(ready_db)
    if source_before["sha256"] != source_after["sha256"]:
        raise RuntimeError("готовая база изменилась во время выгрузки; повторите запуск")
    xlsx = output_root / f"Отчёт РОП по звонкам {day.isoformat()}.xlsx"
    write_workbook(xlsx, day, rows, manager_users.name if manager_users else "", source_before)
    manifest = {
        "schema_version": "daily_mango_calls_resolve_export_v1", "day": day.isoformat(), "rows": len(rows),
        "ready_rows": sum(row["complete"] for row in rows), "unfinished_rows": sum(not row["complete"] for row in rows),
        "working_only_rows": working_only, "audio_copied": copied, "audio_reused": reused,
        "source_ready_db_sha256": source_before["sha256"], "xlsx": xlsx.name, "audio_dir": audio_dir.name,
        "audio": [{"call_id": row["call_id"], "file": row["audio"].name, "sha256": row["audio_sha256"]} for row in rows],
    }
    manifest_path = output_root / f"Отчёт РОП по звонкам {day.isoformat()}.manifest.json"
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    manifest_path.chmod(0o600)
    return {**manifest, "xlsx": str(xlsx), "audio_dir": str(audio_dir), "manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Выгрузить суточный пакет звонков для РОПа.")
    parser.add_argument("--ready-db", type=Path, default=DEFAULT_READY_DB)
    parser.add_argument("--working-db", type=Path, default=DEFAULT_WORKING_DB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--day", type=date.fromisoformat, default=datetime.now(MOSCOW).date() - timedelta(days=1))
    parser.add_argument("--manager-users", type=Path, default=DEFAULT_MANAGER_USERS)
    args = parser.parse_args(argv)
    result = export_day(args.ready_db, args.working_db, args.out, args.day, args.manager_users)
    print(json.dumps({key: value for key, value in result.items() if key != "audio"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
