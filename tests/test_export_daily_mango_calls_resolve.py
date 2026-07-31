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
    transcript_variants_json TEXT,
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
    dialogue = [
        "[00:01.0] Менеджер (Коршунова Анастасия): Здравствуйте, Анна Иванова.",
        "[00:02.0] Клиент: Добрый день. Ищу сыну Петру, он в седьмом классе, очный летний лагерь с математикой.",
        "[00:03.0] Клиент: Бюджет около ста тысяч рублей, цена важна. Есть скидка? Сначала нужно обсудить договор.",
        "[00:04.0] Клиент: Отправьте договор и свяжитесь со мной завтра по телефону.",
        "[00:05.0] Менеджер (Коршунова Анастасия): Хорошо, отправлю договор и позвоню завтра.",
    ]
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO call_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                "MANAGER:\nЗдравствуйте, Анна Иванова. Хорошо, отправлю договор и позвоню завтра.\nCLIENT:\nИщу сыну Петру очный летний лагерь с математикой. Бюджет около ста тысяч рублей, цена важна. Есть скидка? Сначала нужно обсудить договор. Отправьте договор и свяжитесь со мной завтра по телефону.",
                json.dumps({
                    "dialogue_lines": dialogue,
                    "call_topology": "simple_two_party",
                    "role_mapping": {
                        "confirmed": True,
                        "manager_quality_allowed": True,
                        "topology": "simple_two_party",
                    },
                    "manager": {"physical_channel": "left"},
                    "client": {"physical_channel": "right"},
                }, ensure_ascii=False),
                "{}" if pending else json.dumps({"decision": "automatic"}, ensure_ascii=False),
                "{}" if pending else json.dumps(_analysis(), ensure_ascii=False),
            ),
        )


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path, Path]:
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
    tallanto = tmp_path / "tallanto.csv"
    tallanto.write_text(
        'ID,Имя,Фамилия,ФИО родителя,Тел. (родителя),Тел. (доп.)\n1,Пётр,Иванов,Анна Иванова,+79990001122,\n',
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(exporter, "PIPELINE_ROOT", root)
    return ready_db, working_db, users, tallanto, tmp_path / "out"


class FakeTallantoClient:
    calls = 0

    def get_entry_list(self, **_: object) -> dict[str, object]:
        self.calls += 1
        return {
            "entry_list": [{
                "id": "2", "first_name": "Пётр", "last_name": "Петров",
                "phone_mobile": "+79990001123", "phone_work": "", "date_modified": "2099-01-01 00:00:00",
            }],
            "next_offset": None,
        }


class FailingTallantoClient:
    def get_entry_list(self, **_: object) -> dict[str, object]:
        raise exporter.TallantoApiError("temporary failure")


def test_moscow_calendar_day_has_exact_utc_bounds() -> None:
    assert exporter.day_bounds_utc(date(2026, 7, 28)) == ("2026-07-27 21:00:00", "2026-07-28 21:00:00")


def test_current_mango_users_override_archived_manager_name(tmp_path: Path) -> None:
    archived = tmp_path / "users.json"
    archived.write_text(json.dumps({"users": [{"general": {"name": "Старое имя"}, "telephony": {"extension": "405"}}]}), encoding="utf-8")
    current = [{"general": {"name": "Новое имя"}, "telephony": {"extension": "405"}}]
    assert exporter.load_manager_map(archived, current)["405"] == "Новое имя"
    assert exporter.manager_name_issue("Ольга") == "В справочнике Mango указано неполное имя менеджера"
    assert exporter.manager_name_issue("Новое имя") == ""


def test_role_blocks_are_not_presented_as_confirmed_chronology(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exporter, "PIPELINE_ROOT", tmp_path)
    text, confirmed = exporter.ordered_dialogue(tmp_path / "audio/call.mp3", {}, "MANAGER:\nПервый блок\nCLIENT:\nВторой блок")
    assert not confirmed
    assert text.startswith("Порядок реплик не сохранён")
    assert "Менеджер:\nПервый блок" in text and "Клиент:\nВторой блок" in text


def test_sealed_dialogue_never_reads_mutable_transcript_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(exporter, "PIPELINE_ROOT", tmp_path)
    source = tmp_path / "audio" / "call.mp3"
    external = tmp_path / "working" / "transcripts" / "audio" / "call_text.txt"
    external.parent.mkdir(parents=True)
    external.write_text("[00:01.00] Менеджер: ВНЕШНИЙ ИЗМЕНЯЕМЫЙ ТЕКСТ", encoding="utf-8")

    text, confirmed = exporter.ordered_dialogue(
        source, {}, "MANAGER:\nСохранённый текст\nCLIENT:\nОтвет", allow_file_fallback=False,
    )

    assert confirmed is False and "ВНЕШНИЙ" not in text and "Сохранённый текст" in text


def test_sealed_merge_does_not_open_working_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready, working = tmp_path / "ready.sqlite", tmp_path / "working.sqlite"
    calls: list[Path] = []

    def fake_read(path: Path, day: date, *, immutable: bool):
        del day, immutable
        calls.append(path)
        return []

    monkeypatch.setattr(exporter, "read_day", fake_read)
    rows, pending = exporter.merged_day_rows(ready, working, date(2026, 7, 29), {}, sealed_only=True)
    assert rows == [] and pending == 0 and calls == [ready]


def test_estimated_timecodes_are_not_treated_as_confirmed_chronology() -> None:
    lines = ["[~00:01] Менеджер (Иван): Добрый день.", "[~00:05] Клиент: Здравствуйте."]
    _, confirmed = exporter.ordered_dialogue(Path("call.mp3"), {"dialogue_lines": lines}, "")
    assert not confirmed


def test_summary_removes_only_duplicated_technical_preamble() -> None:
    raw = "28.07.2026 12:24 менеджер 202 общался с клиентом. Обсудили лагерь. Приоритет лида: теплый. Итог: Есть согласованный следующий шаг."
    assert exporter.clean_summary(raw) == "Обсудили лагерь. Итог: Есть согласованный следующий шаг."


def test_export_merges_pending_rows_and_preserves_dialogue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    result = exporter.export_day(
        ready_db, working_db, out, date(2026, 7, 28), users,
        tallanto_export=tallanto, tallanto_client=FakeTallantoClient(),
    )
    assert (result["rows"], result["ready_rows"], result["unfinished_rows"], result["working_only_rows"]) == (2, 1, 1, 1)
    assert result["transcripts_copied"] == 2
    assert result["xlsx_sha256"] == _sha(Path(result["xlsx"]))
    assert result["tallanto_match_sources"] == {"выгрузка Tallanto": 1, "Tallanto API": 1}
    files = list(Path(result["transcript_dir"]).glob("*.txt"))
    assert len(files) == 2 and not list(out.rglob("*.mp3"))
    assert all("7999" not in path.name for path in files)

    wb = load_workbook(result["xlsx"], data_only=False)
    assert wb.sheetnames == ["Сводка", "Звонки", "Проблемы данных", "Описание полей"]
    sheet = wb["Звонки"]
    headers = {cell.value: cell.column for cell in sheet[1]}
    transcript = sheet.cell(2, headers["Расшифровка разговора, часть 1"]).value
    assert transcript.index("Менеджер:") < transcript.index("Клиент:") < transcript.rindex("Менеджер:")
    assert "MANAGER:" not in transcript and "CLIENT:" not in transcript
    assert sheet.cell(2, headers["ФИО клиента из Tallanto"]).value == "Анна Иванова"
    assert headers["Расшифровка разговора, часть 1"] == headers["Краткое содержание разговора"] + 1
    assert "Статус обработки" not in headers and "Школа" not in headers and "Аудиозапись" not in headers
    phone = sheet.cell(2, headers["Телефон клиента"])
    assert phone.value == "+79990001122" and phone.data_type == "s"
    transcript_link = sheet.cell(2, headers["Файл полной расшифровки"]).hyperlink
    assert transcript_link is not None and not transcript_link.target.startswith("file:")
    assert wb["Проблемы данных"].max_row == 2
    problem_headers = {cell.value: cell.column for cell in wb["Проблемы данных"][1]}
    assert wb["Проблемы данных"].cell(2, problem_headers["ФИО клиента из Tallanto"]).value == "Петров Пётр"
    assert "audio" not in json.dumps(result, ensure_ascii=False).casefold()
    wb.close()


def test_repeated_export_reuses_identical_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    kwargs = {"tallanto_export": tallanto, "tallanto_client": FakeTallantoClient()}
    first = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)
    xlsx_mtime = Path(first["xlsx"]).stat().st_mtime_ns
    second = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)
    assert (second["transcripts_copied"], second["transcripts_reused"]) == (0, 2)
    assert second["reused"] is True
    assert Path(second["xlsx"]).stat().st_mtime_ns == xlsx_mtime


def test_timed_dialogue_without_role_evidence_is_review_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    for db in (ready_db, working_db):
        with sqlite3.connect(db) as con:
            row = con.execute("SELECT transcript_variants_json FROM call_records WHERE source_call_id='call-ready'").fetchone()
            payload = json.loads(row[0])
            payload.pop("role_mapping", None)
            con.execute("UPDATE call_records SET transcript_variants_json=? WHERE source_call_id='call-ready'", (json.dumps(payload, ensure_ascii=False),))
    manifest_path = ready_db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"sha256": _sha(ready_db), "size_bytes": ready_db.stat().st_size})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, tallanto_export=tallanto, tallanto_client=FakeTallantoClient())

    assert result["manager_ready_rows"] == 0
    wb = load_workbook(result["xlsx"], read_only=True, data_only=True)
    assert wb["Звонки"].max_row == 1
    values = list(wb["Проблемы данных"].iter_rows(values_only=True))
    assert any("Роли менеджера и клиента не подтверждены" in str(cell) for row in values for cell in row)
    assert any("Роли не подтверждены; не использовать для оценки сотрудника" in str(cell) for row in values for cell in row)
    headers = {value: index for index, value in enumerate(values[0])}
    row = next(item for item in values[1:] if item[headers["Телефон клиента"]] == "+79990001122")
    for column in ("Тип звонка по смысловому анализу", "Краткое содержание разговора", "Продукт", "Возражения и ограничения", "Следующий шаг"):
        assert row[headers[column]] is None
    wb.close()


def test_conflicting_topology_is_review_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    for db in (ready_db, working_db):
        with sqlite3.connect(db) as con:
            row = con.execute("SELECT transcript_variants_json FROM call_records WHERE source_call_id='call-ready'").fetchone()
            payload = json.loads(row[0])
            payload["call_topology"] = "conference_or_multi_party"
            con.execute("UPDATE call_records SET transcript_variants_json=? WHERE source_call_id='call-ready'", (json.dumps(payload, ensure_ascii=False),))
    manifest_path = ready_db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"sha256": _sha(ready_db), "size_bytes": ready_db.stat().st_size})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, tallanto_export=tallanto, tallanto_client=FakeTallantoClient())

    assert result["manager_ready_rows"] == 0
    transcript_name = f"call_{hashlib.sha256(b'call-ready').hexdigest()[:20]}.txt"
    transcript = (Path(result["transcript_dir"]) / transcript_name).read_text(encoding="utf-8")
    assert "Менеджер:" not in transcript


def test_call_id_is_hashed_in_transcript_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    sensitive_id = "Иванов+79990001122@example.com"
    for db in (ready_db, working_db):
        with sqlite3.connect(db) as con:
            con.execute("UPDATE call_records SET source_call_id=? WHERE source_call_id='call-ready'", (sensitive_id,))
    manifest_path = ready_db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"sha256": _sha(ready_db), "size_bytes": ready_db.stat().st_size})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, tallanto_export=tallanto, tallanto_client=FakeTallantoClient())

    names = " ".join(path.name for path in Path(result["transcript_dir"]).glob("*.txt"))
    assert "Иванов" not in names and "79990001122" not in names and "example" not in names
    assert sensitive_id not in Path(result["manifest"]).read_text(encoding="utf-8")


def test_changed_started_at_updates_one_stable_transcript_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    kwargs = {"tallanto_export": tallanto, "tallanto_client": FakeTallantoClient()}
    first = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)
    names_before = {path.name for path in Path(first["transcript_dir"]).glob("*.txt")}
    with sqlite3.connect(ready_db) as con:
        con.execute("UPDATE call_records SET started_at='2026-07-28 09:01:00' WHERE source_call_id='call-ready'")
    manifest_path = ready_db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"sha256": _sha(ready_db), "size_bytes": ready_db.stat().st_size})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    second = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)

    assert second["reused"] is False
    assert second["transcript_dir"] != first["transcript_dir"]
    assert {path.name for path in Path(second["transcript_dir"]).glob("*.txt")} == names_before


@pytest.mark.parametrize("unexpected", ["unexpected.txt", "unexpected.partial"])
def test_unexpected_transcript_file_blocks_unchanged_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unexpected: str) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    kwargs = {"tallanto_export": tallanto, "tallanto_client": FakeTallantoClient()}
    first = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)
    (Path(first["transcript_dir"]) / unexpected).write_text("unexpected", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected transcript files"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)


def test_xlsx_failure_publishes_no_transcripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(exporter, "write_workbook", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("xlsx failed")))

    with pytest.raises(OSError, match="xlsx failed"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, tallanto_export=tallanto, tallanto_client=FakeTallantoClient())

    assert not list(out.rglob("*.txt")) and not list(out.glob("*.xlsx"))


def test_xlsx_verification_failure_removes_internal_pii_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(exporter, "load_workbook", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("verify failed")))

    with pytest.raises(OSError, match="verify failed"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, tallanto_export=tallanto, tallanto_client=FakeTallantoClient())

    assert not list(out.rglob("*.xlsx")) and not list(out.rglob("*.txt"))


def test_existing_immutable_transcript_generation_fails_closed_on_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    kwargs = {"tallanto_export": tallanto, "tallanto_client": FakeTallantoClient()}
    first = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)
    next(Path(first["transcript_dir"]).glob("*.txt")).write_text("corrupted", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable transcript generation is inconsistent"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)


def test_existing_immutable_xlsx_generation_fails_closed_on_corruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    kwargs = {"tallanto_export": tallanto, "tallanto_client": FakeTallantoClient()}
    first = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)
    Path(first["xlsx"]).write_bytes(b"corrupted")

    with pytest.raises(RuntimeError, match="immutable XLSX generation is inconsistent"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)


def test_unreferenced_xlsx_generation_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    kwargs = {"tallanto_export": tallanto, "tallanto_client": FakeTallantoClient()}
    first = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)
    Path(first["manifest"]).unlink()

    with pytest.raises(RuntimeError, match="unreferenced immutable XLSX"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)


def test_removed_call_uses_new_exact_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    kwargs = {"tallanto_export": tallanto, "tallanto_client": FakeTallantoClient()}
    first = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)
    with sqlite3.connect(working_db) as con:
        con.execute("DELETE FROM call_records WHERE source_call_id='call-pending'")

    second = exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users, **kwargs)

    assert second["transcript_dir"] != first["transcript_dir"]
    assert len(list(Path(second["transcript_dir"]).glob("*.txt"))) == 1
    current = json.loads(Path(second["manifest"]).read_text(encoding="utf-8"))
    assert current["transcript_dir"] == Path(second["transcript_dir"]).name


def test_transcript_generation_rename_failure_leaves_no_partial_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "generation"
    rows = [{"transcript": "full dialogue", "transcript_file": target / "call_a.txt"}]
    rows[0]["transcript_sha256"] = hashlib.sha256(b"full dialogue\n").hexdigest()
    monkeypatch.setattr(exporter.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("rename failed")))

    with pytest.raises(OSError, match="rename failed"):
        exporter.publish_transcripts(rows, target)

    assert not target.exists() and not list(tmp_path.glob(".generation.staging-*"))


def test_tallanto_issue_revokes_manager_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    tallanto.write_text('ID,Имя,Фамилия,ФИО родителя,Тел. (родителя),Тел. (доп.)\n', encoding="utf-8-sig")

    result = exporter.export_day(
        ready_db, working_db, out, date(2026, 7, 28), users,
        tallanto_export=tallanto, tallanto_client=FakeTallantoClient(),
    )

    assert result["manager_ready_rows"] == 0
    wb = load_workbook(result["xlsx"], read_only=True, data_only=True)
    assert wb["Звонки"].max_row == 1 and wb["Проблемы данных"].max_row == 3
    wb.close()


def test_current_moscow_day_is_rejected_before_reading_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="завершённые сутки"):
        exporter.export_day(tmp_path / "missing-ready", tmp_path / "missing-working", tmp_path / "out", datetime.now(exporter.MOSCOW).date(), None)


def test_ready_database_manifest_mismatch_blocks_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, _, out = _fixture(tmp_path, monkeypatch)
    ready_db.write_bytes(ready_db.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="контрольный файл"):
        exporter.export_day(ready_db, working_db, out, date(2026, 7, 28), users)


def test_tallanto_multiple_matches_are_not_selected(tmp_path: Path) -> None:
    export = tmp_path / "tallanto.csv"
    export.write_text(
        'ID,Имя,Фамилия,ФИО родителя,Тел. (родителя),Тел. (доп.)\n1,Иван,Иванов,,+79990001122,\n2,Пётр,Петров,,+79990001122,\n',
        encoding="utf-8-sig",
    )
    rows = [{"phone": "+79990001122", "issues": [], "client_fio": "", "tallanto_source": ""}]
    exporter.apply_tallanto_names(rows, export, FakeTallantoClient())  # type: ignore[arg-type]
    assert rows[0]["client_fio"] == ""
    assert "несколькими карточками" in rows[0]["issues"][0]


def test_tallanto_api_is_loaded_once_for_all_missing_phones(tmp_path: Path) -> None:
    export = tmp_path / "tallanto.csv"
    export.write_text('ID,Имя,Фамилия,ФИО родителя,Тел. (родителя),Тел. (доп.)\n', encoding="utf-8-sig")
    rows = [
        {"phone": "+79990001123", "issues": [], "client_fio": "", "tallanto_source": ""},
        {"phone": "+79990001124", "issues": [], "client_fio": "", "tallanto_source": ""},
    ]
    client = FakeTallantoClient()
    exporter.apply_tallanto_names(rows, export, client)  # type: ignore[arg-type]
    assert client.calls == 1
    assert rows[0]["client_fio"] == "Петров Пётр"
    assert rows[1]["client_fio"] == ""


def test_tallanto_api_failure_is_not_reported_as_phone_absent(tmp_path: Path) -> None:
    export = tmp_path / "tallanto.csv"
    export.write_text('ID,Имя,Фамилия,ФИО родителя,Тел. (родителя),Тел. (доп.)\n', encoding="utf-8-sig")
    rows = [{"phone": "+79990001123", "issues": [], "client_fio": "", "tallanto_source": ""}]
    exporter.apply_tallanto_names(rows, export, FailingTallantoClient())  # type: ignore[arg-type]
    assert rows[0]["issues"] == ["Не удалось проверить телефон через Tallanto API"]


def test_missing_tallanto_export_blocks_matching(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="выгрузка Tallanto не найдена"):
        exporter.apply_tallanto_names([], tmp_path / "missing.csv", FakeTallantoClient())  # type: ignore[arg-type]


def test_long_transcript_is_split_without_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ready_db, working_db, users, tallanto, out = _fixture(tmp_path, monkeypatch)
    lines = [f"[00:01.0] Менеджер (Иван): {'а' * 70_000}", "[00:02.0] Клиент: конец"]
    for db in (ready_db, working_db):
        with sqlite3.connect(db) as con:
                con.execute(
                    "UPDATE call_records SET transcript_variants_json=? WHERE source_call_id='call-ready'",
                    (json.dumps({
                        "dialogue_lines": lines,
                        "call_topology": "simple_two_party",
                        "role_mapping": {"confirmed": True, "manager_quality_allowed": True, "topology": "simple_two_party"},
                        "manager": {"physical_channel": "left"},
                        "client": {"physical_channel": "right"},
                    }, ensure_ascii=False),),
                )
    manifest_path = ready_db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"sha256": _sha(ready_db), "size_bytes": ready_db.stat().st_size})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = exporter.export_day(
        ready_db, working_db, out, date(2026, 7, 28), users,
        tallanto_export=tallanto, tallanto_client=FakeTallantoClient(),
    )
    wb = load_workbook(result["xlsx"], read_only=True, data_only=True)
    sheet = wb["Звонки"]
    header = next(sheet.iter_rows(values_only=True))
    transcript_columns = [i for i, value in enumerate(header) if str(value).startswith("Расшифровка разговора, часть")]
    row = next(sheet.iter_rows(min_row=2, max_row=2, values_only=True))
    restored = "".join(str(row[i] or "") for i in transcript_columns)
    expected, confirmed = exporter.ordered_dialogue(Path("ignored"), {"dialogue_lines": lines}, "")
    assert confirmed and restored == expected
    assert all(len(str(row[i] or "")) <= exporter.TRANSCRIPT_CHUNK for i in transcript_columns)
    transcript_name = f"call_{hashlib.sha256(b'call-ready').hexdigest()[:20]}.txt"
    txt = Path(result["transcript_dir"]) / transcript_name
    assert txt.read_text(encoding="utf-8").rstrip() == expected
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
