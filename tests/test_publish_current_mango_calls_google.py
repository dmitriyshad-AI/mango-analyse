from __future__ import annotations

import json
import re
import shutil
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.conftest import dual_strict_source, ready_capture_proof

from mango_mvp.productization import owner_only_io
from mango_mvp.productization.mango_calls_service_contract import (
    STAGE10_SCHEMA,
    approved_runtime_fingerprint,
    sha256_file,
)
from mango_mvp.productization.ready_publication import (
    commit_ready_generation,
    inspect_ready_publication,
)
from scripts import publish_current_mango_calls_google as publisher


OWNER = "owner@example.test"
SERVICE = "calls@example.test"
ROP = "rop@example.test"


def _interrupt_ready_publication(ready_db: Path) -> None:
    staged = ready_db.parent / "synthetic-next.sqlite"
    shutil.copy2(ready_db, staged)
    staged.chmod(0o600)
    with sqlite3.connect(staged) as con:
        con.execute("CREATE TABLE synthetic_publication_marker(value INTEGER)")
    manifest_path = ready_db.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        ready_db=str(ready_db),
        sha256=sha256_file(staged),
        size_bytes=staged.stat().st_size,
        ready_mtime_ns=staged.stat().st_mtime_ns,
    )

    def crash(stage: str) -> None:
        if stage == "db_replaced":
            raise RuntimeError("synthetic ready publication crash")

    with pytest.raises(RuntimeError, match="synthetic ready publication crash"):
        commit_ready_generation(ready_db, staged, manifest, checkpoint=crash)


def permissions(*, extra: str | None = None, public: bool = False) -> list[dict[str, str]]:
    result = [
        {"type": "user", "role": "owner", "emailAddress": OWNER},
        {"type": "user", "role": "writer", "emailAddress": SERVICE},
        {"type": "user", "role": "writer", "emailAddress": ROP},
    ]
    if extra:
        result.append({"type": "user", "role": "reader", "emailAddress": extra})
    if public:
        result.append({"type": "anyone", "role": "reader"})
    return result


def _column_number(name: str) -> int:
    value = 0
    for character in name:
        value = value * 26 + ord(character) - 64
    return value


class Gateway:
    def __init__(self, *, acl: list[dict[str, str]] | None = None) -> None:
        self.acl = acl or permissions()
        self._sheets: dict[str, int] = {}
        self.values: dict[str, list[list[object]]] = {}
        self.next_id = 1
        self.writes: list[object] = []
        self.banners: dict[str, str] = {}
        self._protections: dict[str, list[dict[str, object]]] = {}
        self._hidden_columns: dict[str, set[int]] = {}

    def permissions(self) -> list[dict[str, str]]:
        return list(self.acl)

    def file_permissions(self, _file_id: str) -> list[dict[str, str]]:
        return list(self.acl)

    def sheets(self) -> list[dict[str, object]]:
        return [
            {"title": title, "sheetId": sheet_id}
            for title, sheet_id in self._sheets.items()
        ]

    def batch_sheet_requests(self, requests: list[dict[str, object]]) -> None:
        self.writes.append(("sheet", requests))
        for request in requests:
            if "addSheet" in request:
                title = request["addSheet"]["properties"]["title"]
                self._sheets[str(title)] = self.next_id
                self.next_id += 1
                self.values[str(title)] = []
                self._protections[str(title)] = []
                self._hidden_columns[str(title)] = set()
            elif "updateSheetProperties" in request:
                props = request["updateSheetProperties"]["properties"]
                old = next(
                    name for name, sheet_id in self._sheets.items()
                    if sheet_id == props["sheetId"]
                )
                new = str(props["title"])
                self._sheets[new] = self._sheets.pop(old)
                self.values[new] = self.values.pop(old)
                if old in self.banners:
                    self.banners[new] = self.banners.pop(old)
                self._protections[new] = self._protections.pop(old, [])
                self._hidden_columns[new] = self._hidden_columns.pop(old, set())
            elif "updateDimensionProperties" in request:
                dimension = request["updateDimensionProperties"]["range"]
                sheet_id = int(dimension["sheetId"])
                title = next(
                    name for name, current_id in self._sheets.items()
                    if current_id == sheet_id
                )
                if request["updateDimensionProperties"]["properties"].get(
                    "hiddenByUser"
                ) is True:
                    self._hidden_columns.setdefault(title, set()).update(
                        range(int(dimension["startIndex"]), int(dimension["endIndex"]))
                    )
            elif "addProtectedRange" in request:
                payload = dict(request["addProtectedRange"]["protectedRange"])
                sheet_id = int(payload["range"]["sheetId"])
                title = next(
                    name for name, current_id in self._sheets.items()
                    if current_id == sheet_id
                )
                self._protections.setdefault(title, []).append(payload)

    def protections(self, title: str) -> list[dict[str, object]]:
        return [dict(value) for value in self._protections.get(title, [])]

    def column_hidden(self, title: str, column_index: int) -> bool:
        return column_index in self._hidden_columns.get(title, set())

    def read_values(self, title: str) -> list[list[object]]:
        return [list(row) for row in self.values.get(title, [])]

    def read_banner(self, title: str) -> str:
        return self.banners.get(title, "")

    def clear_values(self, title: str) -> None:
        self.writes.append(("clear", title))
        if self.values.get(title):
            self.values[title] = self.values[title][:1]

    def write_values(self, data: list[dict[str, object]]) -> None:
        self.writes.append(("values", data))
        for item in data:
            raw_range = str(item["range"])
            title = raw_range.split("!", 1)[0].strip("'").replace("''", "'")
            values = item["values"]
            if raw_range.endswith("!A1"):
                self.banners[title] = str(values[0][0])
                continue
            if re.search(r"!A1:[A-Z]+2$", raw_range):
                self.banners[title] = str(values[0][0])
                self.values[title] = [list(values[1])]
                continue
            block = re.search(r"!A(\d+):([A-Z]+)(\d+)$", raw_range)
            if block:
                start = int(block.group(1))
                for row_offset, values_row in enumerate(values):
                    sheet_row = start + row_offset
                    index = sheet_row - 2
                    while len(self.values[title]) <= index:
                        self.values[title].append([])
                    self.values[title][index] = list(values_row)
                continue
            match = re.search(r"!([A-Z]+)(\d+)$", raw_range)
            assert match
            column = _column_number(match.group(1))
            sheet_row = int(match.group(2))
            index = sheet_row - 2
            while len(self.values[title]) <= index:
                self.values[title].append([])
            row = self.values[title][index]
            while len(row) < column:
                row.append("")
            row[column - 1] = values[0][0]


def manager_row(key: str, *, summary: str = "Обсудили программу обучения.") -> dict[str, object]:
    return {
        "call_key": key,
        "Дата и время": "11.08.2026 12:00:00",
        "Менеджер": "Менеджер",
        "Направление": "Входящий",
        "Клиент": "Клиент",
        "Телефон": "+70000000000",
        "Длительность": 60.0,
        "Тип разговора": "Продажа",
        "Краткое содержание": summary,
        "Результат": "Назначен следующий контакт",
        "Интерес клиента": "Курс",
        "Главное возражение": "Стоимость",
        "Следующий шаг": "Перезвонить",
        "Срок": "12.08.2026",
        publisher.TRANSCRIPT_LINK_HEADER: "",
        "Нужна проверка": "Нет",
        "Причина проверки": "",
    }


def publish(gateway: Gateway, rows: list[dict[str, object]], **kwargs: object) -> dict[str, object]:
    return dict(
        publisher.publish_current(
            gateway,
            day=date(2026, 8, 11),
            desired_rows=rows,
            owner_email=OWNER,
            allowed_emails=(SERVICE, ROP),
            pilot_started_day=date(2026, 8, 1),
            retention_approved=False,
            prior_day=None,
            **kwargs,
        )
    )


def test_credentials_policy_uses_path_of_exact_opened_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cloud = tmp_path / "Yandex.Disk" / "credentials.json"
    cloud.parent.mkdir()
    cloud.write_text(json.dumps({"client_email": "cloud@example.test"}), encoding="utf-8")
    cloud.chmod(0o600)
    local = tmp_path / "local.json"
    local.write_text(json.dumps({"client_email": "local@example.test"}), encoding="utf-8")
    local.chmod(0o600)
    original = publisher.read_stable_regular_bytes_with_path

    def swap_after_read(path: Path, **kwargs: object) -> tuple[bytes, Path]:
        raw, opened_path = original(path, **kwargs)
        path.unlink()
        path.symlink_to(local)
        return raw, opened_path

    monkeypatch.setattr(
        publisher,
        "read_stable_regular_bytes_with_path",
        swap_after_read,
    )

    with pytest.raises(RuntimeError, match="outside repository and cloud"):
        publisher.validate_credentials(cloud)


def test_credentials_reject_hardlink_with_cloud_alias(tmp_path: Path) -> None:
    local = tmp_path / "local" / "credentials.json"
    local.parent.mkdir()
    local.write_text(json.dumps({"client_email": "linked@example.test"}), encoding="utf-8")
    local.chmod(0o600)
    cloud = tmp_path / "Yandex.Disk" / "credentials.json"
    cloud.parent.mkdir()
    cloud.hardlink_to(local)

    with pytest.raises(RuntimeError, match="owner-only 0600"):
        publisher.validate_credentials(local)


def test_credentials_reject_parent_move_during_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_root = tmp_path / "local"
    local_root.mkdir()
    credentials = local_root / "credentials.json"
    credentials.write_text(
        json.dumps({"client_email": "moved@example.test"}),
        encoding="utf-8",
    )
    credentials.chmod(0o600)
    cloud_root = tmp_path / "Yandex.Disk"
    cloud_root.mkdir()
    moved_root = cloud_root / "local"
    original_read = owner_only_io.os.read
    swapped = False

    def move_parent_then_read(descriptor: int, size: int) -> bytes:
        nonlocal swapped
        if not swapped:
            local_root.rename(moved_root)
            local_root.symlink_to(moved_root, target_is_directory=True)
            swapped = True
        return original_read(descriptor, size)

    monkeypatch.setattr(owner_only_io.os, "read", move_parent_then_read)

    with pytest.raises(RuntimeError, match="owner-only 0600"):
        publisher.validate_credentials(credentials)

    assert swapped is True


def test_credentials_reject_macos_cloudstorage_path(tmp_path: Path) -> None:
    credentials = (
        tmp_path
        / "Library"
        / "CloudStorage"
        / "GoogleDrive-test"
        / "credentials.json"
    )
    credentials.parent.mkdir(parents=True)
    credentials.write_text(
        json.dumps({"client_email": "cloud@example.test"}),
        encoding="utf-8",
    )
    credentials.chmod(0o600)

    with pytest.raises(RuntimeError, match="outside repository and cloud"):
        publisher.validate_credentials(credentials)


def test_three_header_based_upserts_preserve_rop_sort_and_unknown_column() -> None:
    gateway = Gateway()
    first = publish(gateway, [manager_row("call-1")])
    assert first["rows"] == 1
    header = gateway.values[publisher.CURRENT_TITLE][0]
    row = gateway.values[publisher.CURRENT_TITLE][1]
    row.extend([""] * (len(header) - len(row)))
    row[header.index("Комментарий РОПа")] = "Проверено"
    row[header.index("Решение РОПа")] = "Принято"
    header.insert(3, "Новая колонка РОПа")
    row.insert(3, "Не менять")
    gateway.values[publisher.CURRENT_TITLE].append(["", "служебная строка"])
    gateway.values[publisher.CURRENT_TITLE][1:] = list(
        reversed(gateway.values[publisher.CURRENT_TITLE][1:])
    )

    second = publish(gateway, [manager_row("call-1", summary="Обновлённая сводка")])
    third = publish(gateway, [manager_row("call-1", summary="Обновлённая сводка")])

    rows = gateway.values[publisher.CURRENT_TITLE]
    header = rows[0]
    matching = [row for row in rows[1:] if row and row[header.index("call_key")] == "call-1"]
    assert len(matching) == 1
    assert matching[0][header.index("Комментарий РОПа")] == "Проверено"
    assert matching[0][header.index("Решение РОПа")] == "Принято"
    assert matching[0][header.index("Новая колонка РОПа")] == "Не менять"
    assert second["managed_cell_updates"] == 1
    assert third["status"] == "unchanged" and third["managed_cell_updates"] == 0


def test_quarantine_is_only_in_review_is_idempotent_and_clears() -> None:
    gateway = Gateway()
    quarantine = publisher.quarantine_review_rows(
        [
            {
                "call_key": "quarantine-1",
                "started_at": "2026-08-11T09:30:00+00:00",
                "code": "recording_retry_expired",
                "reason": "Аудиозапись не появилась в Mango в течение 72 часов.",
                "action": (
                    "Проверить запись в Mango и повторить загрузку вручную, "
                    "если файл появился."
                ),
            }
        ],
        day=date(2026, 8, 11),
    )
    summary = {
        "mango_unique": 2,
        "ready_unique": 1,
        "quarantine_unique": 1,
        "pending_unique": 0,
        "unexplained_missing": 0,
        "consistency_ok": True,
        "closure_ok": True,
    }

    first = publish(
        gateway,
        [manager_row("call-1")],
        quarantine_rows=quarantine,
        stage10_summary=summary,
    )
    second = publish(
        gateway,
        [manager_row("call-1")],
        quarantine_rows=quarantine,
        stage10_summary=summary,
    )

    assert first["quarantine_rows"] == 1
    assert second["status"] == "unchanged"
    current = gateway.values[publisher.CURRENT_TITLE]
    review = gateway.values[publisher.REVIEW_TITLE]
    assert [row[0] for row in current[1:] if row] == ["call-1"]
    assert [row[0] for row in review[1:] if row] == ["quarantine-1"]
    assert review[1][review[0].index("Телефон")] == ""
    assert review[1][review[0].index(publisher.TRANSCRIPT_LINK_HEADER)] == ""
    assert review[1][review[0].index("Следующий шаг")].startswith(
        "Техническое действие с данными:"
    )

    cleared = publish(
        gateway,
        [manager_row("call-1")],
        quarantine_rows=[],
        stage10_summary={**summary, "mango_unique": 1, "quarantine_unique": 0},
    )
    assert cleared["status"] == "updated"
    assert not [row for row in gateway.values[publisher.REVIEW_TITLE][1:] if any(row)]


@pytest.mark.parametrize(
    "acl",
    [permissions(extra="outsider@example.test"), permissions(public=True)],
)
def test_acl_failure_precedes_every_write(acl: list[dict[str, str]]) -> None:
    gateway = Gateway(acl=acl)
    with pytest.raises(RuntimeError, match="ACL"):
        publish(gateway, [manager_row("call-1")])
    assert gateway.writes == []


def test_duplicate_or_missing_required_header_fails_before_cell_write() -> None:
    for bad_header in (
        [*publisher.HEADERS[:-1], publisher.HEADERS[-2]],
        list(publisher.HEADERS[:-1]),
    ):
        gateway = Gateway()
        gateway._sheets[publisher.CURRENT_TITLE] = 1
        gateway.values[publisher.CURRENT_TITLE] = [bad_header]
        with pytest.raises(RuntimeError, match="headers"):
            publish(gateway, [manager_row("call-1")])
        assert not [item for item in gateway.writes if item[0] == "values"]


def test_existing_unhidden_call_key_column_fails_before_cell_write() -> None:
    gateway = Gateway()
    publish(gateway, [manager_row("call-1")])
    key_index = list(gateway.values[publisher.CURRENT_TITLE][0]).index("call_key")
    gateway._hidden_columns[publisher.CURRENT_TITLE].discard(key_index)
    writes_before = list(gateway.writes)

    with pytest.raises(RuntimeError, match="hidden-column readback"):
        publish(gateway, [manager_row("call-1")])

    assert gateway.writes == writes_before


def test_day_15_without_retention_policy_stops_before_google_write() -> None:
    gateway = Gateway()
    with pytest.raises(RuntimeError, match="retention"):
        publisher.publish_current(
            gateway,
            day=date(2026, 8, 15),
            desired_rows=[manager_row("call-1")],
            owner_email=OWNER,
            allowed_emails=(SERVICE, ROP),
            pilot_started_day=date(2026, 8, 1),
            retention_approved=False,
            prior_day=None,
        )
    assert gateway.writes == []


def test_moscow_rotation_renames_old_sheet_and_preserves_rop() -> None:
    gateway = Gateway()
    publish(gateway, [manager_row("call-1")])
    header = gateway.values[publisher.CURRENT_TITLE][0]
    row = gateway.values[publisher.CURRENT_TITLE][1]
    row.extend([""] * (len(header) - len(row)))
    row[header.index("Комментарий РОПа")] = "Сохранить"

    report = publisher.publish_current(
        gateway,
        day=date(2026, 8, 12),
        desired_rows=[manager_row("call-2")],
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
        pilot_started_day=date(2026, 8, 1),
        retention_approved=False,
        prior_day=date(2026, 8, 11),
    )

    assert report["rotated_from"] == "2026-08-11"
    archived = gateway.values["Звонки 2026-08-11 — предварительно"]
    assert archived[1][archived[0].index("Комментарий РОПа")] == "Сохранить"
    assert gateway.values[publisher.CURRENT_TITLE][1][0] == "call-2"


def _ready_fixture(tmp_path: Path, *, analyzed: bool) -> tuple[Path, Path]:
    db = tmp_path / "ready.sqlite"
    with sqlite3.connect(db) as con:
        con.execute(
            """
            CREATE TABLE call_records (
                id INTEGER, source_call_id TEXT, started_at TEXT, manager_name TEXT,
                phone TEXT, direction TEXT, duration_sec REAL,
                transcription_status TEXT, resolve_status TEXT, analysis_status TEXT,
                analysis_json TEXT, transcript_variants_json TEXT,
                transcript_text TEXT, source_file TEXT
            )
            """
        )
        analysis = (
            {
                "analysis_schema_version": "v3",
                "history_summary": "Безопасная краткая сводка",
                "structured_fields": {
                    "interests": {"products": ["Летняя школа", "Онлайн-курс"]},
                    "objections": ["Стоимость", "Расписание"],
                    "next_step": {
                        "action": "Перезвонить после обсуждения",
                        "due": "2026-08-12",
                    },
                },
                "quality_flags": {
                    "call_type": "sales_call",
                    "needs_review": True,
                    "review_reasons": ["synthetic_review"],
                },
                "needs_review": True,
                "review_reasons": ["synthetic_review"],
            }
            if analyzed
            else {}
        )
        con.execute(
            "INSERT INTO call_records VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "call-1",
                "2026-08-11T09:00:00+00:00",
                "Менеджер",
                "+70000000000",
                "inbound",
                45,
                "done",
                "done",
                "done" if analyzed else "pending",
                json.dumps(analysis, ensure_ascii=False),
                json.dumps(
                    {
                        "mode": "stereo",
                        "primary_provider": "mlx",
                        "secondary_provider": "gigaam",
                        "manager": {
                            "variant_a": "Здравствуйте",
                            "variant_b": "Здравствуйте",
                        },
                        "client": {
                            "variant_a": "Нужен курс",
                            "variant_b": "Нужен курс",
                        },
                    },
                    ensure_ascii=False,
                ),
                "СЕКРЕТНЫЙ ПОЛНЫЙ ДИАЛОГ",
                "/Users/private/call.mp3",
            ),
        )
    verdict = {
        "schema_version": STAGE10_SCHEMA,
        "quarantine_items": [],
        "day": "2026-08-11",
        "generated_at": "2026-08-11T10:00:00+00:00",
        "mango_enumeration_complete": True,
        "mango_unique": 1,
        "ready_unique": 1,
        "quarantine_unique": 0,
        "pending_unique": 0,
        "unexplained_missing": 0,
        "state_overlap_count": 0,
        "pending_awaiting_recording": 0,
        "pending_over_sla": 0,
        "quarantine_without_reason": 0,
        "ready_without_dual_asr_or_explicit_exception": 0,
        "ready_without_resolve": 0,
        "ready_without_analyze": 0 if analyzed else 1,
        "duplicate_call_keys": 0,
        "oldest_pending_age_minutes": 0,
        "state_not_in_mango_enumeration": 0,
        "independent_zero_enumerations": 0,
        "consistency_ok": True,
        "closure_ok": analyzed,
    }
    enumeration_source = {
        "mode": "strict_service",
        "since": "2026-08-10T21:00:00+00:00",
        "until": "2026-08-11T21:00:00+00:00",
        "cursor": "not_applicable_stats_request_result",
        "pages": None,
        "pagination": "not_applicable_stats_request_result",
        "requests": 1,
        "covered_intervals": [
            {
                "since": "2026-08-10T21:00:00+00:00",
                "until": "2026-08-11T21:00:00+00:00",
                "result_complete": True,
            }
        ],
    }
    enumeration_source = dual_strict_source(
        enumeration_source,
        call_keys=["call-1"],
        calls_by_day={"2026-08-11": ["call-1"]},
    )
    verdict["mango_enumeration_source"] = enumeration_source
    capture_proof, capture_proof_sha256 = ready_capture_proof(
        enumeration_source,
        zero_by_day={"2026-08-11": 0},
    )
    manifest = {
        "schema_version": "mango_calls_ready_v3",
        "created_at_utc": "2026-08-11T10:00:01+00:00",
        "published_at": "2026-08-11T10:00:02+00:00",
        "status": "ready",
        "consistency_ok": True,
        "closure_ok": analyzed,
        "moscow_dates": ["2026-08-11"],
        "daily_verdicts": {"2026-08-11": verdict},
        "producer_git_sha": "a" * 40,
        "host_id": "m1-host",
        "run_id": "run-1",
        "mango_window": {
            "since": "2026-08-10T21:00:00+00:00",
            "until": "2026-08-11T21:00:00+00:00",
        },
        "mango_enumeration_complete": True,
        "mango_enumeration_source": enumeration_source,
        "capture_proof": capture_proof,
        "capture_proof_sha256": capture_proof_sha256,
        "capture_proof_run_id": enumeration_source["dual_enumeration"][
            "proof_run_id"
        ],
        "manifest_snapshot": {"end_offset": 1, "sha256": "b" * 64},
        "provenance_mode": "strict_service",
        "quick_check": "ok",
        "integrity_check": "ok",
        "runtime_fingerprint": approved_runtime_fingerprint(),
        "sha256": sha256_file(db),
        "size_bytes": db.stat().st_size,
    }
    manifest_path = db.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return db, manifest_path


def _rewrite_analysis(
    db: Path,
    manifest_path: Path,
    mutate: object,
) -> None:
    with sqlite3.connect(db) as con:
        payload = json.loads(
            con.execute("SELECT analysis_json FROM call_records").fetchone()[0]
        )
        assert callable(mutate)
        mutate(payload)
        con.execute(
            "UPDATE call_records SET analysis_json=?",
            (json.dumps(payload, ensure_ascii=False),),
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(sha256=sha256_file(db), size_bytes=db.stat().st_size)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_projection_never_contains_full_dialogue_path_or_diagnostic_json(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=False)
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    serialized = json.dumps(rows, ensure_ascii=False)
    assert "СЕКРЕТНЫЙ ПОЛНЫЙ ДИАЛОГ" not in serialized
    assert "/Users/" not in serialized and "analysis_json" not in serialized
    assert rows[0]["Краткое содержание"] == publisher.NEUTRAL_SUMMARY
    assert set(rows[0]) == set(publisher.MANAGED_HEADERS)


def test_capture_quarantine_is_visible_as_safe_manager_review_row(
    tmp_path: Path,
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verdict = manifest["daily_verdicts"]["2026-08-11"]
    verdict.update(
        mango_unique=2,
        quarantine_unique=1,
        quarantine_items=[
            {
                "call_key": "capture-quarantine-1",
                "started_at": "2026-08-11T09:30:00+00:00",
                "code": "recording_retry_expired",
                "reason": "Аудиозапись не появилась в Mango в течение 72 часов.",
                "action": (
                    "Проверить запись в Mango и повторить загрузку вручную, "
                    "если файл появился."
                ),
            }
        ],
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )

    assert len(rows) == 1
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=rows,
        ready_manifest=manifest,
        now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
    )
    quarantine_rows = publisher.quarantine_review_rows(
        plan["quarantine_items"], day=date(2026, 8, 11)
    )
    assert len(quarantine_rows) == 1
    quarantine = quarantine_rows[0]
    assert quarantine["Тип разговора"] == "Карантин данных"
    assert quarantine["Нужна проверка"] == "Да"
    assert quarantine["Телефон"] == ""
    assert quarantine[publisher.TRANSCRIPT_LINK_HEADER] == ""
    assert "72 часов" in quarantine["Причина проверки"]
    assert "повторить загрузку вручную" in quarantine["Следующий шаг"]
    serialized = json.dumps(quarantine, ensure_ascii=False).casefold()
    assert "error" not in serialized and "/users/" not in serialized

    assert len(publisher.validate_safe_plan_payload(
        plan,
        expected_day=date(2026, 8, 11),
        now=datetime(2026, 8, 11, 10, 1, tzinfo=timezone.utc),
    )) == 1


def test_private_link_requires_exact_acl_evidence(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)
    evidence = {
        "call-1": {
            "url": "https://drive.google.com/file/d/private/view",
            "acl_readback_ok": True,
            "allowed_emails": [OWNER, SERVICE, ROP],
        }
    }
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
        link_evidence=evidence,
    )
    assert rows[0][publisher.TRANSCRIPT_LINK_HEADER].startswith("https://")
    evidence["call-1"]["allowed_emails"] = [OWNER, SERVICE, "outsider@example.test"]
    with pytest.raises(RuntimeError, match="ACL"):
        publisher.load_manager_rows(
            ready_db=db,
            ready_manifest=manifest,
            day=date(2026, 8, 11),
            owner_email=OWNER,
            allowed_emails=(SERVICE, ROP),
            link_evidence=evidence,
        )


def test_projection_uses_real_analyze_schema_without_list_repr(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)
    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Тип разговора"] == "Продажа / подбор обучения"
    assert row["Интерес клиента"] == "Летняя школа | Онлайн-курс"
    assert row["Главное возражение"] == "Стоимость | Расписание"
    assert row["Следующий шаг"] == "Перезвонить после обсуждения"
    assert row["Срок"] == "2026-08-12"
    assert row["Результат"] == "Следующий шаг выделен анализом"
    assert row["Нужна проверка"] == "Да"
    assert "текущей схемой Analyze" not in str(row["Причина проверки"])


def test_missing_second_asr_is_explicitly_marked_for_review(tmp_path: Path) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    with sqlite3.connect(db) as con:
        raw = con.execute(
            "SELECT transcript_variants_json FROM call_records"
        ).fetchone()[0]
        variants = json.loads(raw)
        variants.pop("secondary_provider")
        for role in ("manager", "client"):
            variants[role].pop("variant_b")
        con.execute(
            "UPDATE call_records SET transcript_variants_json=?",
            (json.dumps(variants, ensure_ascii=False),),
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"sha256": sha256_file(db), "size_bytes": db.stat().st_size})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Нужна проверка"] == "Да"
    assert "Вторая расшифровка GigaAM не готова" in str(row["Причина проверки"])
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=[row],
        ready_manifest=manifest_payload,
        now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
    )
    summary = publisher._summary_rows(
        [row],
        {
            **plan["stage10_counts"],
            "consistency_ok": plan["consistency_ok"],
            "closure_ok": plan["closure_ok"],
        },
    )

    assert plan["stage10_counts"]["processing_ready_unique"] == 0
    assert plan["row_completion_ok"] is False
    assert plan["closure_ok"] is False
    assert ["Полностью готово", 0] in summary
    assert ["День закрыт", "Нет"] in summary


def test_v2_non_conversation_has_deterministic_result(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)

    def mutate(payload: dict[str, object]) -> None:
        payload["quality_flags"] = {"call_type": "non_conversation"}
        payload["needs_review"] = False
        structured = payload["structured_fields"]
        assert isinstance(structured, dict)
        structured["next_step"] = {"action": "", "due": ""}

    _rewrite_analysis(db, manifest, mutate)
    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Результат"] == "Разговор не состоялся"
    assert "Итог разговора" not in str(row["Причина проверки"])


def test_v2_without_supported_outcome_gets_neutral_result_and_review(
    tmp_path: Path,
) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)

    def mutate(payload: dict[str, object]) -> None:
        payload["quality_flags"] = {"call_type": "sales_call"}
        payload["needs_review"] = False
        structured = payload["structured_fields"]
        assert isinstance(structured, dict)
        structured["next_step"] = {"action": "", "due": ""}

    _rewrite_analysis(db, manifest, mutate)
    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Результат"] == "Итог не зафиксирован"
    assert row["Нужна проверка"] == "Да"
    assert "Итог разговора требует ручной проверки" in str(row["Причина проверки"])


def test_legacy_explicit_result_has_priority(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)
    _rewrite_analysis(
        db,
        manifest,
        lambda payload: payload.update(result="Договор отправлен"),
    )

    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Результат"] == "Договор отправлен"


@pytest.mark.parametrize("length", [2_001, 32_000])
def test_analyze_summary_is_preserved_without_truncation(
    tmp_path: Path, length: int
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    expected = "я" * length
    _rewrite_analysis(
        db,
        manifest_path,
        lambda payload: payload.update(history_summary=expected),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        ready_manifest_payload=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=rows,
        ready_manifest=manifest,
        now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
    )

    assert plan["rows"][0]["Краткое содержание"] == expected


def test_summary_over_analyze_limit_is_replaced_per_row(tmp_path: Path) -> None:
    db, manifest = _ready_fixture(tmp_path, analyzed=True)
    _rewrite_analysis(
        db,
        manifest,
        lambda payload: payload.update(history_summary="я" * 32_001),
    )

    row = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )[0]

    assert row["Краткое содержание"] == publisher.OVERSIZED_SUMMARY
    assert row["Нужна проверка"] == "Да"
    assert "предел Analyze" in str(row["Причина проверки"])


def test_safe_plan_expires_after_sixty_minutes(tmp_path: Path) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        ready_manifest_payload=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    generated = datetime(2026, 8, 11, 10, tzinfo=timezone.utc)
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=rows,
        ready_manifest=manifest,
        now=generated,
    )

    assert publisher.validate_safe_plan_payload(
        plan,
        expected_day=date(2026, 8, 11),
        now=generated + timedelta(minutes=60),
    )
    with pytest.raises(RuntimeError, match="expired"):
        publisher.validate_safe_plan_payload(
            plan,
            expected_day=date(2026, 8, 11),
            now=generated + timedelta(minutes=60, seconds=1),
        )


def test_safe_plan_validator_rejects_rows_removed_after_build(
    tmp_path: Path,
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        ready_manifest_payload=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    generated = datetime(2026, 8, 11, 10, tzinfo=timezone.utc)
    plan = dict(
        publisher.build_safe_plan(
            day=date(2026, 8, 11),
            rows=rows,
            ready_manifest=manifest,
            now=generated,
        )
    )
    plan["rows"] = []

    with pytest.raises(
        RuntimeError, match="row completion state|rows do not match Stage10"
    ):
        publisher.validate_safe_plan_payload(
            plan,
            expected_day=date(2026, 8, 11),
            now=generated,
        )


def test_current_plan_uses_green_target_day_even_when_older_day_is_red(
    tmp_path: Path,
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    green = manifest["daily_verdicts"]["2026-08-11"]
    red = json.loads(json.dumps(green))
    red.update(
        day="2026-08-10",
        mango_unique=2,
        unexplained_missing=1,
        consistency_ok=False,
        closure_ok=False,
    )
    red_source = red["mango_enumeration_source"]
    red_source.update(
        since="2026-08-09T21:00:00+00:00",
        rolling_since="2026-08-09T21:00:00+00:00",
        until="2026-08-10T21:00:00+00:00",
    )
    red_source["covered_intervals"] = [
        {
            "since": "2026-08-09T21:00:00+00:00",
            "until": "2026-08-10T21:00:00+00:00",
            "result_complete": True,
        }
    ]
    red["mango_enumeration_source"] = dual_strict_source(
        red_source,
        call_keys=["call-1", "missing-call"],
        calls_by_day={"2026-08-10": ["call-1", "missing-call"]},
    )
    manifest.update(consistency_ok=False, closure_ok=False)
    manifest["moscow_dates"] = ["2026-08-10", "2026-08-11"]
    manifest["daily_verdicts"] = {
        "2026-08-10": red,
        "2026-08-11": green,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows = publisher.load_manager_rows(
        ready_db=db,
        ready_manifest=manifest_path,
        ready_manifest_payload=manifest,
        day=date(2026, 8, 11),
        owner_email=OWNER,
        allowed_emails=(SERVICE, ROP),
    )
    plan = publisher.build_safe_plan(
        day=date(2026, 8, 11),
        rows=rows,
        ready_manifest=manifest,
        now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
    )

    assert plan["consistency_ok"] is True
    assert plan["moscow_day"] == "2026-08-11"


def test_zero_stage10_rejects_nonempty_manager_rows(tmp_path: Path) -> None:
    _db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verdict = manifest["daily_verdicts"]["2026-08-11"]
    verdict.update(
        mango_unique=0,
        ready_unique=0,
        quarantine_unique=0,
        pending_unique=0,
        unexplained_missing=0,
        independent_zero_enumerations=2,
        closure_ok=True,
    )
    original_source = manifest["mango_enumeration_source"]
    zero_source = dual_strict_source(
        {
            "mode": "strict_service",
            "since": original_source["since"],
            "rolling_since": original_source["rolling_since"],
            "until": original_source["until"],
            "cursor": original_source["cursor"],
            "pages": original_source["pages"],
            "pagination": original_source["pagination"],
            "requests": 1,
            "covered_intervals": [
                {
                    "since": original_source["rolling_since"],
                    "until": original_source["until"],
                    "result_complete": True,
                }
            ],
            "catch_up": False,
        },
        call_keys=[],
        calls_by_day={},
    )
    verdict["mango_enumeration_source"] = zero_source
    manifest["mango_enumeration_source"] = zero_source
    capture_proof, capture_proof_sha256 = ready_capture_proof(
        zero_source,
        zero_by_day={"2026-08-11": 2},
    )
    manifest["capture_proof"] = capture_proof
    manifest["capture_proof_sha256"] = capture_proof_sha256
    manifest["capture_proof_run_id"] = zero_source["dual_enumeration"][
        "proof_run_id"
    ]

    with pytest.raises(RuntimeError, match="Stage10 balance"):
        publisher.build_safe_plan(
            day=date(2026, 8, 11),
            rows=[manager_row("impossible-row")],
            ready_manifest=manifest,
            now=datetime(2026, 8, 11, 10, tzinfo=timezone.utc),
        )


def test_manifest_and_publication_lock_symlinks_are_rejected(
    tmp_path: Path,
) -> None:
    real_manifest = tmp_path / "real-manifest.json"
    real_manifest.write_text("{}\n", encoding="utf-8")
    manifest_link = tmp_path / "manifest.json"
    manifest_link.symlink_to(real_manifest)
    with pytest.raises(RuntimeError, match="manifest"):
        publisher.stable_json_object(manifest_link, label="ready manifest")

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir(mode=0o700)
    victim = tmp_path / "victim.txt"
    victim.write_text("must remain unchanged", encoding="utf-8")
    lock = lock_dir / "publication.lock"
    lock.symlink_to(victim)
    with pytest.raises(RuntimeError, match="lock is unsafe"):
        with publisher.publication_lock(lock):
            pass
    assert victim.read_text(encoding="utf-8") == "must remain unchanged"


def test_dry_run_uses_one_ready_manifest_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    original = publisher.stable_json_object
    reads: list[object] = []
    seen_payloads: list[object] = []

    def stable(path: Path, *, label: str) -> dict[str, object]:
        reads.append((path, label))
        return dict(original(path, label=label))

    def rows(**kwargs: object) -> list[dict[str, object]]:
        seen_payloads.append(kwargs.get("ready_manifest_payload"))
        return [manager_row("call-1")]

    monkeypatch.setattr(publisher, "stable_json_object", stable)
    monkeypatch.setattr(publisher, "load_manager_rows", rows)

    assert publisher.main(
        [
            "--ready-db",
            str(db),
            "--ready-manifest",
            str(manifest_path),
            "--owner-email",
            OWNER,
            "--day",
            "2026-08-11",
        ]
    ) == 0
    assert len(reads) == 1
    assert len(seen_payloads) == 1
    assert isinstance(seen_payloads[0], dict)


def test_dry_run_recovers_interrupted_ready_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, manifest_path = _ready_fixture(tmp_path, analyzed=True)
    _interrupt_ready_publication(db)
    monkeypatch.setattr(
        publisher,
        "load_manager_rows",
        lambda **_kwargs: [manager_row("call-1")],
    )

    result = publisher.main(
        [
            "--ready-db",
            str(db),
            "--ready-manifest",
            str(manifest_path),
            "--owner-email",
            OWNER,
            "--day",
            "2026-08-11",
        ]
    )

    assert result == 0
    assert inspect_ready_publication(db)["recovery_required"] is False


def test_execute_requires_owner_approval_of_exact_safe_plan_sha() -> None:
    with pytest.raises(RuntimeError, match="approved SHA-256"):
        publisher.main(
            [
                "--execute",
                "--confirmation",
                publisher.CONFIRMATION,
                "--config",
                "/not/read/config.json",
                "--safe-plan",
                "/not/read/plan.json",
                "--state",
                "/not/read/state.json",
                "--day",
                "2026-08-11",
            ]
        )
