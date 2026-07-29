from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest
from openpyxl import Workbook, load_workbook

from scripts import export_daily_mango_calls_resolve as exporter


SCHEMA = """
CREATE TABLE call_records (
    id INTEGER PRIMARY KEY,
    source_file TEXT,
    source_filename TEXT,
    source_call_id TEXT,
    duration_sec REAL,
    phone TEXT,
    manager_name TEXT,
    direction TEXT,
    started_at TEXT,
    transcription_status TEXT,
    resolve_status TEXT,
    analysis_status TEXT,
    transcript_text TEXT,
    resolve_json TEXT,
    analysis_json TEXT
)
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analysis() -> dict:
    return {
        "history_summary": "Клиент обсудил летний лагерь и попросил договор.",
        "structured_fields": {
            "people": {"parent_fio": "Анна Иванова", "child_fio": "Пётр"},
            "student": {"grade_current": "7", "school": "Лицей"},
            "interests": {"products": ["летний лагерь"], "subjects": ["математика"], "format": ["очно"], "exam_targets": []},
            "commercial": {"budget": "100000 рублей", "price_sensitivity": "medium", "discount_interest": True},
            "contacts": {"preferred_channel": "phone"},
            "objections": ["нужно обсудить договор"],
            "next_step": {"action": "Отправить договор", "due": "завтра"},
        },
        "quality_flags": {"call_type": "sales_call", "transcript_quality_requires_manual_review": False},
        "needs_review": False,
    }


def _insert(db: Path, *, pending: bool = False, call_id: str = "call-ready", started: str = "2026-07-28 08:00:00", audio: Path) -> None:
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO call_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                2 if pending else 1,
                str(audio),
                audio.name,
                call_id,
                125.0,
                "+79990001122" if not pending else "+79990001123",
                "19",
                "inbound" if not pending else "outbound",
                started,
                "done",
                "pending" if pending else "done",
                "pending" if pending else "done",
                "MANAGER:\nЗдравствуйте, Анна Иванова.\nCLIENT:\nДобрый день.",
                "{}" if pending else json.dumps({"decision": "automatic"}, ensure_ascii=False),
                "{}" if pending else json.dumps(_analysis(), ensure_ascii=False),
            ),
        )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "pipeline"
    audio_root = root / "working/audio"
    audio_root.mkdir(parents=True)
    ready_audio, pending_audio = audio_root / "ready.mp3", audio_root / "pending.mp3"
    ready_audio.write_bytes(b"ready-audio")
    pending_audio.write_bytes(b"pending-audio")
    ready_db, working_db = root / "drop/ready.sqlite", root / "working/working.sqlite"
    ready_db.parent.mkdir(parents=True)
    for db in (ready_db, working_db):
        with sqlite3.connect(db) as con:
            con.execute(SCHEMA)
    _insert(ready_db, audio=ready_audio)
    _insert(working_db, audio=ready_audio)
    _insert(working_db, pending=True, call_id="call-pending", audio=pending_audio)
    manifest = {
        "status": "ready",
        "quick_check": "ok",
        "sha256": _sha(ready_db),
        "size_bytes": ready_db.stat().st_size,
        "published_at": "2026-07-29T00:00:00Z",
    }
    ready_db.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    users = tmp_path / "users.json"
    users.write_text(json.dumps({"users": [{"general": {"name": "Коршунова Анастасия"}, "telephony": {"extension": "19"}}]}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(exporter, "PIPELINE_ROOT", root)
    return ready_db, working_db, users, tmp_path / "out"


def test_moscow_calendar_day_has_exact_utc_bounds() -> None:
    assert exporter.day_bounds_utc(date(2026, 7, 28)) == ("2026-07-27 21:00:00", "2026-07-28 21:00:00")


def test_evidenced_fio_requires_full_name_in_transcript() -> None:
    transcript = "Менеджер: Анна Иванова, добрый день."
    assert exporter.evidenced_fio("Анна Иванова", transcript) == "Анна Иванова"
    assert exporter.evidenced_fio("Пётр", transcript) == ""
    assert exporter.evidenced_fio("Мария Петрова", transcript) == ""


def test_summary_removes_only_duplicated_technical_preamble() -> None:
    raw = "28.07.2026 12:24 менеджер 202 общался с клиентом. Обсудили лагерь. Приоритет лида: теплый. Итог: Есть согласованный следующий шаг."
    assert exporter.clean_summary(raw) == "Обсудили лагерь. Итог: Есть согласованный следующий шаг."


def test_export_merges_pending_rows_and_preserves_dialogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, out = _fixture(tmp_path, monkeypatch)
    result = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users)
    assert (result["rows"], result["ready_rows"], result["unfinished_rows"], result["working_only_rows"]) == (2, 1, 1, 1)
    assert result["audio_copied"] == 2
    assert len(list(Path(result["audio_dir"]).glob("*.mp3"))) == 2
    assert all("7999" not in path.name for path in Path(result["audio_dir"]).glob("*.mp3"))

    wb = load_workbook(result["xlsx"], data_only=False)
    assert wb.sheetnames == ["Сводка", "Звонки", "Проблемы данных", "Описание полей"]
    sheet = wb["Звонки"]
    headers = {cell.value: cell.column for cell in sheet[1]}
    transcript = sheet.cell(2, headers["Расшифровка разговора, часть 1"]).value
    assert "Менеджер:" in transcript and "Клиент:" in transcript
    assert "MANAGER:" not in transcript and "CLIENT:" not in transcript
    assert sheet.cell(2, headers["ФИО собеседника из разговора (не подтверждено CRM)"]).value == "Анна Иванова"
    assert sheet.cell(2, headers["ФИО ученика из разговора (не подтверждено CRM)"]).value in (None, "")
    phone = sheet.cell(2, headers["Телефон клиента"])
    assert phone.value == "+79990001122" and phone.data_type == "s"
    assert wb["Проблемы данных"].max_row == 2
    wb.close()


def test_repeated_export_reuses_identical_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, out = _fixture(tmp_path, monkeypatch)
    exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users)
    second = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users)
    assert (second["audio_copied"], second["audio_reused"]) == (0, 2)


def test_existing_audio_with_other_content_blocks_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, out = _fixture(tmp_path, monkeypatch)
    first = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users)
    next(Path(first["audio_dir"]).glob("*.mp3")).write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="конфликт аудиофайла"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users)


def test_current_moscow_day_is_rejected_before_reading_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="завершённые сутки"):
        exporter.export_day(tmp_path / "missing-ready", tmp_path / "missing-working", tmp_path / "out", datetime.now(exporter.MOSCOW).date(), None)


def test_ready_database_manifest_mismatch_blocks_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, out = _fixture(tmp_path, monkeypatch)
    ready_db.write_bytes(ready_db.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="контрольный файл"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users)


def test_audio_outside_pipeline_root_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"audio")
    row = {"id": 1, "call_id": "unsafe", "source": outside, "started": datetime(2026, 7, 28, 10, 0, tzinfo=exporter.MOSCOW)}
    with pytest.raises(RuntimeError, match="небезопасный"):
        exporter.publish_audio([row], tmp_path / "allowed", tmp_path / "out")


def test_long_transcript_is_split_without_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, out = _fixture(tmp_path, monkeypatch)
    long_text = "MANAGER:\n" + "а" * 70_000 + "\nCLIENT:\nконец"
    for db in (ready_db, working_db):
        with sqlite3.connect(db) as con:
            con.execute("UPDATE call_records SET transcript_text=? WHERE source_call_id='call-ready'", (long_text,))
    manifest_path = ready_db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"sha256": _sha(ready_db), "size_bytes": ready_db.stat().st_size})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users)
    wb = load_workbook(result["xlsx"], read_only=True, data_only=True)
    sheet = wb["Звонки"]
    header = next(sheet.iter_rows(values_only=True))
    transcript_columns = [i for i, value in enumerate(header) if str(value).startswith("Расшифровка разговора, часть")]
    row = next(sheet.iter_rows(min_row=2, max_row=2, values_only=True))
    restored = "".join(str(row[i] or "") for i in transcript_columns)
    assert restored == exporter.translate_transcript(long_text)
    assert all(len(str(row[i] or "")) <= exporter.TRANSCRIPT_CHUNK for i in transcript_columns)
    wb.close()


def test_excel_formula_prefixes_remain_plain_text(tmp_path: Path) -> None:
    wb = Workbook()
    sheet = wb.active
    values = ["=1+1", "+79990000000", "-10", "@служебное"]
    exporter.append_safe(sheet, values)
    path = tmp_path / "safe.xlsx"
    wb.save(path)
    checked = load_workbook(path, data_only=False)
    cells = list(checked.active[1])
    assert [cell.value for cell in cells] == values
    assert all(cell.data_type == "s" for cell in cells)
    checked.close()
