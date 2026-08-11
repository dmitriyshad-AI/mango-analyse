from __future__ import annotations

import json
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.productization.mango_calls_service_contract import (
    approved_runtime_fingerprint,
    sha256_file,
)
from scripts import publish_current_mango_calls_google as publisher


OWNER = "owner@example.test"
SERVICE = "calls@example.test"
ROP = "rop@example.test"


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
                analysis_json TEXT, transcript_text TEXT, source_file TEXT
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
            "INSERT INTO call_records VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                "СЕКРЕТНЫЙ ПОЛНЫЙ ДИАЛОГ",
                "/Users/private/call.mp3",
            ),
        )
    verdict = {
        "schema_version": "mango_calls_stage10_verdict_v1",
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
    verdict["mango_enumeration_source"] = enumeration_source
    manifest = {
        "schema_version": "mango_calls_ready_v2",
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
    assert row["Результат"] == ""
    assert row["Нужна проверка"] == "Да"
    assert "текущей схемой Analyze" in str(row["Причина проверки"])


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
        return []

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
