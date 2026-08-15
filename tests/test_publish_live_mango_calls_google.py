from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import publish_live_mango_calls_google as publisher


def record(**overrides):
    payload = {
        "id": 7,
        "source_call_id": "call-7",
        "started_at": "2026-08-14 09:03:04",
        "phone": "+70000000000",
        "manager_name": "mango_manager_1",
        "direction": "outbound",
        "duration_sec": 142.5,
        "analysis_json": json.dumps(
            {
                "quality_flags": {"call_type": "sales_call"},
                "target_product": "Математика",
                "history_summary": "Обсудили обучение.",
                "follow_up_reason": "Информация предоставлена",
                "structured_fields": {
                    "objections": ["Цена"],
                    "next_step": {"action": "Перезвонить", "due": "завтра"},
                },
                "review_reasons": ["Проверить роли"],
            },
            ensure_ascii=False,
        ),
        "transcript_variants_json": json.dumps(
            {
                "role_mapping": {"left": "manager", "right": "client"},
                "dialogue_lines": [
                    "[00:01.0] Дорожка левая: Добрый день",
                    "[00:02.0] Дорожка левая: представлюсь",
                    "[00:03.0] Дорожка правая: Здравствуйте",
                ],
            },
            ensure_ascii=False,
        ),
        "transcript_text": "",
        "analysis_status": "done",
        "sync_status": "pending",
    }
    payload.update(overrides)
    return payload


def projected(**overrides):
    return publisher.call_projection(record(**overrides), {"mango_manager_1": "Иван Иванов"})


def with_number(call, number=10):
    return publisher.desired_row(call, number)


def state_for(call, row, *, status="verified"):
    return {
        "schema_version": publisher.STATE_SCHEMA,
        "destination_id": "dest",
        "data_errors": {},
        "entries": {
            call["call_key"]: {
                "display_number": int(row[0]),
                "status": status,
                "projection_version": publisher.PROJECTION_VERSION,
                "source_fingerprint": call["source_fingerprint"],
                "started_epoch": call["started_epoch"],
                "planned_row_sha256": (
                    publisher.physical_row_hash(row) if status == "reserved" else None
                ),
                "last_verified_row_sha256": publisher.physical_row_hash(row),
                "attempts": 1,
            }
        },
    }


def _owner_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.chmod(path, 0o600)


def _destination(sheet_id=0):
    return f"google_sheets:v1:spreadsheet:{sheet_id}:mango:production"


def _sqlite(tmp_path: Path, records):
    db_path = tmp_path / "calls.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE call_records ("
        "id INTEGER PRIMARY KEY, source_call_id TEXT, started_at TEXT, phone TEXT, "
        "manager_name TEXT, direction TEXT, duration_sec REAL, analysis_json TEXT, "
        "transcript_variants_json TEXT, transcript_text TEXT, analysis_status TEXT, "
        "sync_status TEXT, sync_attempts INTEGER NOT NULL DEFAULT 0)"
    )
    columns = (
        "id", "source_call_id", "started_at", "phone", "manager_name", "direction",
        "duration_sec", "analysis_json", "transcript_variants_json", "transcript_text",
        "analysis_status", "sync_status",
    )
    connection.executemany(
        f"INSERT INTO call_records ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        [tuple(item.get(column) for column in columns) for item in records],
    )
    connection.commit()
    connection.close()
    return db_path


def _decode_cell(cell):
    value = cell.get("userEnteredValue") or {}
    for field in ("stringValue", "numberValue", "boolValue"):
        if field in value:
            return value[field]
    return ""


class FakeLiveGoogleGateway:
    def __init__(self, rows=(), *, title="Звонки", sheet_id=0):
        self.rows = [list(row) for row in rows]
        self.row_heights = [publisher.row_height(row[9]) for row in self.rows]
        self.title = title
        self.sheet_id = sheet_id
        self.batch_calls = 0
        self.values_calls = 0
        self.raise_after_apply = False
        self.on_batch_applied = None
        self.before_values = None

    def sheets(self):
        return [{"title": self.title, "sheetId": self.sheet_id}]

    def values(self, _title):
        self.values_calls += 1
        if self.before_values is not None:
            callback, self.before_values = self.before_values, None
            callback()
        return [list(publisher.LIVE_HEADERS), *[list(row) for row in self.rows]]

    def batch_sheet_requests(self, requests):
        self.batch_calls += 1
        q_values = None
        for request in requests:
            if "updateCells" in request:
                update = request["updateCells"]
                target = update.get("range")
                if target and int(target.get("startColumnIndex", 0)) == 0:
                    start = int(target["startRowIndex"]) - 1
                    decoded = [
                        [_decode_cell(cell) for cell in row.get("values") or ()]
                        for row in update.get("rows") or ()
                    ]
                    self.rows[start:start + len(decoded)] = decoded
                elif int((update.get("start") or {}).get("columnIndex", -1)) == 16:
                    q_values = [
                        _decode_cell((row.get("values") or [{}])[0])
                        for row in update.get("rows") or ()
                    ]
            elif "appendCells" in request:
                appended = [
                    [_decode_cell(cell) for cell in row.get("values") or ()]
                    for row in request["appendCells"].get("rows") or ()
                ]
                self.rows.extend(appended)
                self.row_heights.extend([21] * len(appended))
            elif "sortRange" in request:
                assert q_values is not None and len(q_values) == len(self.rows)
                ordered = sorted(zip(q_values, self.rows), key=lambda item: item[0], reverse=True)
                self.rows = [row for _key, row in ordered]
            elif "updateDimensionProperties" in request:
                update = request["updateDimensionProperties"]
                target = update.get("range") or {}
                if target.get("dimension") != "ROWS":
                    continue
                start = int(target["startIndex"]) - 1
                end = int(target["endIndex"]) - 1
                size = int((update.get("properties") or {})["pixelSize"])
                while len(self.row_heights) < end:
                    self.row_heights.append(21)
                self.row_heights[start:end] = [size] * (end - start)
        if self.on_batch_applied is not None:
            self.on_batch_applied(self)
        if self.raise_after_apply:
            self.raise_after_apply = False
            raise TimeoutError("simulated lost Google response")

    def layout(self, _title, _last_row):
        column_metadata = [{} for _ in range(10)]
        column_metadata[9] = {"pixelSize": 320}
        return {
            "sheets": [{
                "data": [{
                    "columnMetadata": column_metadata,
                    "rowMetadata": [
                        {"pixelSize": size} for size in self.row_heights
                    ],
                    "rowData": [
                        {
                            "values": [
                                {
                                    "userEnteredFormat": (
                                        {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}
                                        if index == 9
                                        else {"wrapStrategy": "CLIP", "verticalAlignment": "TOP"}
                                        if index == 15
                                        else {}
                                    )
                                }
                                for index in range(16)
                            ]
                        }
                        for _row in self.rows
                    ],
                }]
            }]
        }


def _harness(tmp_path, monkeypatch, *, records, rows=(), state=None, fake=None, sheet_id=0):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    db_path = _sqlite(private, records)
    credentials = private / "credentials.json"
    manager = private / "manager.json"
    state_path = private / "state.json"
    lock = private / "publisher.lock"
    config = private / "config.json"
    _owner_json(credentials, {"client_email": "publisher@example.invalid"})
    _owner_json(manager, {"mapping": {"mango_manager_1": "Иван Иванов"}})
    if state is not None:
        _owner_json(state_path, state)
    _owner_json(
        config,
        {
            "schema_version": publisher.CONFIG_SCHEMA,
            "spreadsheet_id": "spreadsheet",
            "sheet_id": sheet_id,
            "sheet_title": "Звонки",
            "working_db": str(db_path),
            "manager_identity": str(manager),
            "credentials": str(credentials),
            "state": str(state_path),
            "lock": str(lock),
            "summary_width_px": 320,
            "batch_limit": 25,
            "expected_code_sha": "a" * 40,
        },
    )
    fake = fake or FakeLiveGoogleGateway(rows, sheet_id=sheet_id)
    monkeypatch.setattr(publisher, "authorized_session", lambda _info: object())
    monkeypatch.setattr(
        publisher, "LiveGoogleGateway", lambda _session, _spreadsheet_id: fake
    )
    monkeypatch.setattr(
        publisher.subprocess,
        "check_output",
        lambda command, text=True: "a" * 40 + "\n" if command[-1] == "HEAD" else "",
    )
    return {
        "config": config,
        "db": db_path,
        "state": state_path,
        "fake": fake,
        "private": private,
    }


def _run_execute(config, *extra):
    return publisher.run(
        [
            "--config", str(config), "--execute", "--confirmation", publisher.CONFIRMATION,
            *extra,
        ]
    )


def _db_status(db_path, call_id=7):
    connection = sqlite3.connect(db_path)
    result = connection.execute(
        "SELECT sync_status,sync_attempts FROM call_records WHERE id=?", (call_id,)
    ).fetchone()
    connection.close()
    return result


def test_live_headers_are_exact_production_contract():
    assert publisher.LIVE_HEADERS == (
        "№", "Дата и время (МСК)", "Менеджер", "Направление", "Длительность",
        "Категория", "Телефон клиента", "Нужна проверка", "Тема",
        "Конспект разговора", "Результат", "Возражение / причина",
        "Следующий шаг", "Срок", "Что проверить РОПу", "Полная расшифровка",
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "0 мин 0 с"), (6.5, "0 мин 7 с"), (142.5, "2 мин 23 с"), (179.49, "2 мин 59 с")],
)
def test_duration_is_half_up_and_human_readable(seconds, expected):
    assert publisher.format_duration(seconds) == expected


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_duration_fails_closed(value):
    with pytest.raises(ValueError):
        publisher.format_duration(value)


def test_projection_converts_naive_utc_to_moscow_once_and_keeps_full_transcript():
    call = projected()
    row = with_number(call)
    assert row[1] == "2026-08-14 12:03:04"
    assert row[4] == "2 мин 23 с"
    assert row[15] == (
        "[00:01.0] Менеджер: Добрый день представлюсь\n"
        "[00:03.0] Клиент: Здравствуйте"
    )


def test_projection_converts_only_generated_summary_prefix_to_moscow():
    analysis = json.loads(record()["analysis_json"])
    analysis["history_summary"] = "14.08.2026 09:03 — Менеджер: обсудили занятие в 15:00"
    row = with_number(
        projected(analysis_json=json.dumps(analysis, ensure_ascii=False))
    )
    assert row[9] == "14.08.2026 12:03 — Менеджер: обсудили занятие в 15:00"


def test_projection_does_not_shift_non_generated_time_inside_summary():
    analysis = json.loads(record()["analysis_json"])
    analysis["history_summary"] = "Договорились созвониться в 15:00"
    row = with_number(
        projected(analysis_json=json.dumps(analysis, ensure_ascii=False))
    )
    assert row[9] == "Договорились созвониться в 15:00"


def test_projection_uses_one_canonical_summary_field_for_google():
    analysis = json.loads(record()["analysis_json"])
    analysis["summary"] = "Краткий конспект для рабочей таблицы"
    analysis["history_summary"] = "14.08.2026 09:03 — служебная память"
    row = with_number(
        projected(analysis_json=json.dumps(analysis, ensure_ascii=False))
    )
    assert row[9] == "Краткий конспект для рабочей таблицы"


def test_transcript_fallback_removes_technical_labels():
    call = projected(
        transcript_variants_json=json.dumps({"full": {"final": "CHANNEL_LEFT: Текст\nsha256: secret"}})
    )
    assert call["tail"][-1] == "[00:00.0] Не определено: Текст"


def test_projection_rejects_empty_or_oversized_transcript():
    with pytest.raises(ValueError, match="empty"):
        projected(transcript_variants_json="{}", transcript_text="")
    with pytest.raises(ValueError, match="50000"):
        projected(
            transcript_variants_json=json.dumps({"full": {"final": "я" * 50_001}}, ensure_ascii=False)
        )


@pytest.mark.parametrize(
    "variants",
    [
        {"dialogue_lines": ["[00:01.0] Спикер 1: Текст"]},
        {
            "role_mapping": {"left": "manager"},
            "dialogue_lines": ["[00:01.0] Дорожка правая: Текст"],
        },
    ],
)
def test_projection_rejects_unknown_or_unmapped_dialogue_role(variants):
    with pytest.raises(ValueError, match="unknown or unmapped role"):
        projected(transcript_variants_json=json.dumps(variants, ensure_ascii=False))


def test_legacy_unknown_role_can_only_identify_an_existing_row():
    raw = record(
        transcript_variants_json=json.dumps(
            {"dialogue_lines": ["[00:01.0] Спикер 1: Старый текст"]},
            ensure_ascii=False,
        )
    )
    identity = publisher.call_identity(raw)
    expected = "[00:01.0] Не определено: Старый текст"
    assert identity["transcript_sha"] == publisher.hashlib.sha256(
        expected.encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError):
        publisher.call_projection(raw, {})


def test_source_change_changes_fingerprint():
    assert projected()["source_fingerprint"] != projected(phone="+71111111111")["source_fingerprint"]


def test_unrelated_manager_mapping_does_not_change_fingerprint():
    raw = record()
    first = publisher.call_projection(raw, {"mango_manager_1": "Иван Иванов"})
    second = publisher.call_projection(
        raw,
        {"mango_manager_1": "Иван Иванов", "unrelated_manager": "Другой менеджер"},
    )
    assert first["source_fingerprint"] == second["source_fingerprint"]


def test_sheet_identity_accepts_current_and_minute_legacy_time():
    call = projected()
    current = with_number(call)
    assert publisher.identity_matches(publisher.sheet_identity(current), call)
    legacy = list(current)
    legacy[1] = "2026-08-14 12:03"
    legacy[4] = "142,5"
    assert publisher.identity_matches(publisher.sheet_identity(legacy), call)


def test_sheet_identity_accepts_proven_live_legacy_phone_and_zero_seconds():
    call = projected(started_at="2026-08-14 09:03:37")
    legacy = with_number(call)
    legacy[1] = "14.08.2026 12:03:00"
    legacy[6] = "'+70000000000"
    assert publisher.identity_matches(publisher.sheet_identity(legacy), call)


def test_normalize_values_rejects_headers_gaps_and_q_data():
    header = [*publisher.LIVE_HEADERS]
    row = with_number(projected())
    assert publisher.normalize_values([header, row])[1] == [row]
    with pytest.raises(ValueError, match="header"):
        publisher.normalize_values([["wrong"], row])
    with pytest.raises(ValueError, match="Q:Y"):
        publisher.normalize_values([header, [*row, "leaked"]])
    with pytest.raises(ValueError, match="contiguous"):
        publisher.normalize_values([header, row, [], row])


@pytest.mark.parametrize("nonblank", [0, False])
def test_normalize_values_treats_zero_and_false_as_nonblank(nonblank):
    header = [*publisher.LIVE_HEADERS]
    row = with_number(projected())

    with pytest.raises(ValueError, match="header"):
        publisher.normalize_values([[*header, nonblank], row])
    with pytest.raises(ValueError, match="Q:Y"):
        publisher.normalize_values([header, [*row, nonblank]])

    marker_row = [""] * 16
    marker_row[0] = nonblank
    parsed_rows = publisher.normalize_values([header, marker_row])[1]
    assert parsed_rows == [marker_row]


def test_reconcile_matches_one_call_and_blocks_unknown_or_duplicate():
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    assert publisher.reconcile([row], {call["call_key"]: call}, state)[0] == {call["call_key"]: 0}
    unknown = list(row)
    unknown[6] = "+79999999999"
    with pytest.raises(RuntimeError, match=r"ambiguous.*Google row 2 \(№ 10\)"):
        publisher.reconcile([unknown], {call["call_key"]: call}, state)
    with pytest.raises(RuntimeError, match="Google rows 2 and 3"):
        publisher.reconcile([row, row], {call["call_key"]: call}, state)


def test_reconcile_physical_validation_error_names_google_row():
    call = projected()
    bad = with_number(call)
    bad[1] = "not-a-date"

    with pytest.raises(RuntimeError, match=r"date.*Google row 2 \(№ 10\)"):
        publisher.reconcile(
            [bad],
            {call["call_key"]: call},
            publisher.default_state("dest"),
        )


def test_reconcile_blocks_two_calls_with_same_business_identity():
    first = projected()
    second_record = record(id=8, source_call_id="call-8")
    second = publisher.call_projection(second_record, {"mango_manager_1": "Иван Иванов"})
    row = with_number(first)
    with pytest.raises(RuntimeError, match="ambiguous"):
        publisher.reconcile([row], {first["call_key"]: first, second["call_key"]: second}, publisher.default_state("dest"))


def test_reservation_is_stable_and_zero_change_is_zero_write():
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    mapping, _ = publisher.reconcile([row], {call["call_key"]: call}, state)
    _state, selected = publisher.reserve(state, {call["call_key"]: call}, [row], mapping, limit=25)
    assert selected == []

    changed = projected(analysis_json=record()["analysis_json"].replace("Обсудили обучение.", "Новый конспект."))
    state, selected = publisher.reserve(state, {changed["call_key"]: changed}, [row], mapping, limit=25)
    assert selected == [changed["call_key"]]
    assert state["entries"][changed["call_key"]]["display_number"] == 10
    assert state["entries"][changed["call_key"]]["status"] == "reserved"


def test_missing_call_gets_next_display_number():
    first = projected()
    first_row = with_number(first, 17)
    second = publisher.call_projection(record(id=8, source_call_id="call-8", started_at="2026-08-14 10:00:00"), {})
    state = state_for(first, first_row)
    mapping, _ = publisher.reconcile([first_row], {first["call_key"]: first, second["call_key"]: second}, state)
    state, selected = publisher.reserve(
        state, {first["call_key"]: first, second["call_key"]: second}, [first_row], mapping, limit=25
    )
    assert selected == [second["call_key"]]
    assert state["entries"][second["call_key"]]["display_number"] == 18


def test_deleted_verified_row_is_restored_before_new_and_stale_rows():
    deleted = projected()
    new = publisher.call_projection(
        record(id=8, source_call_id="call-8", started_at="2026-08-14 10:00:00"), {}
    )
    stale = publisher.call_projection(
        record(id=9, source_call_id="call-9", started_at="2026-08-14 11:00:00"), {}
    )
    deleted_row = with_number(deleted, 10)
    stale_row = with_number(stale, 11)
    state = state_for(deleted, deleted_row)
    state["entries"].update(state_for(stale, stale_row)["entries"])
    physical_stale = list(stale_row)
    physical_stale[9] = "Устаревший конспект"
    state, selected = publisher.reserve(
        state,
        {deleted["call_key"]: deleted, new["call_key"]: new, stale["call_key"]: stale},
        [physical_stale],
        {stale["call_key"]: 0},
        limit=1,
    )
    assert selected == [deleted["call_key"]]


def test_missing_verified_row_with_nonpublishable_source_fails_closed():
    call = projected()
    state = state_for(call, with_number(call))
    with pytest.raises(RuntimeError, match="source is not publishable"):
        publisher.verify_sheet_snapshot(
            rows=[], call_to_row={}, identities={}, state=state, recoverable_keys=[]
        )


def test_unknown_result_is_recovered_from_exact_planned_readback():
    call = projected()
    row = with_number(call)
    state = state_for(call, row, status="reserved")
    mapping, _ = publisher.reconcile([row], {call["call_key"]: call}, state)
    assert publisher.applied_reservations(
        state, {call["call_key"]: call}, [row], mapping
    ) == [call["call_key"]]
    assert state["entries"][call["call_key"]]["status"] == "reserved"


def test_google_cells_never_emit_formula_values():
    for prefix in ("=", "+", "-", "@"):
        assert publisher.google_cell(prefix + "danger") == {
            "userEnteredValue": {"stringValue": prefix + "danger"}
        }


def test_height_depends_only_on_summary():
    summary = "Короткий конспект"
    assert publisher.row_height(summary) == publisher.row_height(summary)
    assert publisher.row_height(summary) != publisher.row_height("д" * 900)


def test_height_requests_skip_rows_that_already_have_the_target_height():
    requests = publisher.height_requests(
        0,
        [42, 78, 78, 96],
        current_heights=[42, 42, 78, 96],
    )

    assert len(requests) == 1
    target = requests[0]["updateDimensionProperties"]["range"]
    assert (target["startIndex"], target["endIndex"]) == (2, 3)

    swapped = publisher.height_requests(
        0,
        [78, 42],
        current_heights=[42, 78],
    )
    assert [
        (
            item["updateDimensionProperties"]["range"]["startIndex"],
            item["updateDimensionProperties"]["range"]["endIndex"],
        )
        for item in swapped
    ] == [(1, 2), (2, 3)]


def test_live_gateway_uses_extended_timeout_for_atomic_batch():
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {}

    class Session:
        timeout = None

        def post(self, _url, **kwargs):
            self.timeout = kwargs["timeout"]
            return Response()

    session = Session()
    publisher.LiveGoogleGateway(session, "sheet").batch_sheet_requests([])
    assert session.timeout == 180


def test_layout_formula_error_names_cell_and_display_number():
    row = with_number(projected())
    fake = FakeLiveGoogleGateway([row])
    payload = fake.layout("Звонки", 2)
    values = payload["sheets"][0]["data"][0]["rowData"][0]["values"]
    values[15]["userEnteredValue"] = {"formulaValue": '="hidden"'}

    with pytest.raises(RuntimeError, match=r"P2 \(№ 10\)"):
        publisher.verify_layout(payload, [row], 320)


def test_build_batch_is_one_atomic_request_list_with_sort_clear_and_layout():
    first = projected()
    second = publisher.call_projection(record(id=8, source_call_id="call-8", started_at="2026-08-14 10:00:00"), {})
    first_row = with_number(first, 10)
    state = state_for(first, first_row)
    state["entries"][second["call_key"]] = {
        "display_number": 11,
        "status": "reserved",
        "projection_version": publisher.PROJECTION_VERSION,
        "source_fingerprint": second["source_fingerprint"],
        "started_epoch": second["started_epoch"],
        "planned_row_sha256": publisher.physical_row_hash(with_number(second, 11)),
        "last_verified_row_sha256": None,
        "attempts": 1,
    }
    requests, final_rows = publisher.build_batch(
        sheet_id=0,
        rows=[first_row],
        row_to_call={0: first["call_key"]},
        calls={first["call_key"]: first, second["call_key"]: second},
        identities={first["call_key"]: first, second["call_key"]: second},
        state=state,
        selected=[second["call_key"]],
        summary_width_px=320,
    )
    kinds = [next(iter(request)) for request in requests]
    assert kinds.count("appendCells") == 1
    assert kinds.count("sortRange") == 1
    assert kinds.count("repeatCell") >= 3  # Q clear plus J/P formats
    assert final_rows[0][0] == 11
    assert all(len(row) == 16 for row in final_rows)


def test_q_tie_breaker_is_display_number_and_stays_exact():
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    requests, _ = publisher.build_batch(
        sheet_id=0,
        rows=[row],
        row_to_call={0: call["call_key"]},
        calls={call["call_key"]: call},
        identities={call["call_key"]: call},
        state=state,
        selected=[call["call_key"]],
        summary_width_px=320,
    )
    q_request = next(item["updateCells"] for item in requests if "updateCells" in item and item["updateCells"].get("start", {}).get("columnIndex") == 16)
    actual = q_request["rows"][0]["values"][0]["userEnteredValue"]["numberValue"]
    assert actual == call["started_epoch"] * 1_000_000 + 10
    assert actual < 2**53


def test_run_rejects_wrong_sheet_title_id_pair_before_any_read_or_write(tmp_path, monkeypatch):
    fake = FakeLiveGoogleGateway(title="Звонки", sheet_id=99)
    env = _harness(tmp_path, monkeypatch, records=[record()], fake=fake, sheet_id=0)

    with pytest.raises(RuntimeError, match="title/id"):
        publisher.run(["--config", str(env["config"])])

    assert fake.values_calls == 0
    assert fake.batch_calls == 0


def test_shadow_without_state_checks_real_layout_before_bootstrap(tmp_path, monkeypatch):
    call = projected()
    row = with_number(call)
    fake = FakeLiveGoogleGateway([row])
    env = _harness(tmp_path, monkeypatch, records=[record()], rows=[row], fake=fake)
    valid_layout = fake.layout

    def wrong_height(title, last_row):
        payload = valid_layout(title, last_row)
        payload["sheets"][0]["data"][0]["rowMetadata"][0]["pixelSize"] = 21
        return payload

    fake.layout = wrong_height
    with pytest.raises(RuntimeError, match=r"Google row 2 \(№ 10\)"):
        publisher.run(["--config", str(env["config"])])

    assert fake.batch_calls == 0
    assert not env["state"].exists()


def test_shadow_without_state_checks_newest_first_before_bootstrap(tmp_path, monkeypatch):
    older_record = record()
    newer_record = record(
        id=8,
        source_call_id="call-8",
        started_at="2026-08-14 10:03:04",
        phone="+71111111111",
    )
    older = publisher.call_projection(older_record, {"mango_manager_1": "Иван Иванов"})
    newer = publisher.call_projection(newer_record, {"mango_manager_1": "Иван Иванов"})
    wrong_order = [with_number(older, 10), with_number(newer, 11)]
    fake = FakeLiveGoogleGateway(wrong_order)
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[older_record, newer_record],
        rows=wrong_order,
        fake=fake,
    )

    with pytest.raises(RuntimeError, match="newest-first"):
        publisher.run(["--config", str(env["config"])])

    assert fake.batch_calls == 0
    assert not env["state"].exists()


def test_execute_rejects_explicit_zero_limit_without_state_or_google_change(
    tmp_path, monkeypatch
):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)
    before = env["state"].read_bytes()

    with pytest.raises(RuntimeError, match="between 1 and 25"):
        _run_execute(env["config"], "--limit", "0")

    assert fake.batch_calls == 0
    assert env["state"].read_bytes() == before


def test_run_refuses_repeated_bootstrap_over_nonempty_state(tmp_path, monkeypatch):
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    state["destination_id"] = _destination()
    fake = FakeLiveGoogleGateway([row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], rows=[row], state=state, fake=fake
    )

    with pytest.raises(RuntimeError, match="non-empty"):
        publisher.run(
            [
                "--config", str(env["config"]), "--bootstrap", "--confirmation",
                publisher.BOOTSTRAP_CONFIRMATION,
            ]
        )

    assert fake.batch_calls == 0
    assert json.loads(env["state"].read_text(encoding="utf-8"))["entries"] == state["entries"]


def test_run_recovers_applied_timeout_without_duplicate_row(tmp_path, monkeypatch):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    fake.raise_after_apply = True
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)

    with pytest.raises(TimeoutError, match="lost Google response"):
        _run_execute(env["config"])

    reserved = json.loads(env["state"].read_text(encoding="utf-8"))
    assert next(iter(reserved["entries"].values()))["status"] == "reserved"
    assert len(fake.rows) == 1
    assert fake.batch_calls == 1
    assert _db_status(env["db"]) == ("pending", 0)

    report = _run_execute(env["config"])

    verified = json.loads(env["state"].read_text(encoding="utf-8"))
    assert report["status"] == "no_change"
    assert next(iter(verified["entries"].values()))["status"] == "verified"
    assert len(fake.rows) == 1
    assert fake.batch_calls == 1
    assert _db_status(env["db"]) == ("done", 1)


def test_no_change_pending_sync_requires_layout_before_done(tmp_path, monkeypatch):
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    state["destination_id"] = _destination()
    fake = FakeLiveGoogleGateway([row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], rows=[row], state=state, fake=fake
    )

    def broken_layout(_title, _last_row):
        raise RuntimeError("layout is broken")

    fake.layout = broken_layout
    with pytest.raises(RuntimeError, match="layout is broken"):
        _run_execute(env["config"])

    assert _db_status(env["db"]) == ("pending", 0)
    assert fake.batch_calls == 0


def test_run_source_change_between_verified_state_and_sync_never_marks_done(
    tmp_path, monkeypatch
):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)
    original_finalize = publisher.finalize_verified

    def finalize_then_change(*args, **kwargs):
        proofs = original_finalize(*args, **kwargs)
        changed = record()["analysis_json"].replace(
            "Обсудили обучение.", "Источник изменился до sync."
        )
        connection = sqlite3.connect(env["db"])
        connection.execute("UPDATE call_records SET analysis_json=? WHERE id=7", (changed,))
        connection.commit()
        connection.close()
        return proofs

    monkeypatch.setattr(publisher, "finalize_verified", finalize_then_change)

    with pytest.raises(RuntimeError, match="source changed before sync"):
        _run_execute(env["config"])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert next(iter(persisted["entries"].values()))["status"] == "verified"
    assert fake.batch_calls == 1
    assert _db_status(env["db"]) == ("pending", 0)


def test_run_fails_if_unselected_google_row_changes_concurrently(tmp_path, monkeypatch):
    first_record = record()
    second_record = record(
        id=8,
        source_call_id="call-8",
        started_at="2026-08-14 10:03:04",
        phone="+71111111111",
    )
    first = publisher.call_projection(first_record, {"mango_manager_1": "Иван Иванов"})
    second = publisher.call_projection(second_record, {"mango_manager_1": "Иван Иванов"})
    stale_first = with_number(first, 10)
    stale_first[9] = "Старый конспект"
    second_row = with_number(second, 11)
    state = state_for(first, stale_first)
    state["destination_id"] = _destination()
    state["entries"].update(state_for(second, second_row)["entries"])
    fake = FakeLiveGoogleGateway([second_row, stale_first])

    def change_unselected(gateway):
        for row in gateway.rows:
            if int(row[0]) == 11:
                row[9] = "Параллельная чужая правка"

    fake.on_batch_applied = change_unselected
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[first_record, second_record],
        state=state,
        fake=fake,
    )

    with pytest.raises(RuntimeError, match="unselected Google row changed concurrently"):
        _run_execute(env["config"], "--limit", "1")

    assert fake.batch_calls == 1
    assert _db_status(env["db"], 7) == ("pending", 0)
    assert _db_status(env["db"], 8) == ("pending", 0)


def test_invalid_source_after_prewrite_crash_does_not_block_other_call(
    tmp_path, monkeypatch
):
    original = projected()
    reserved_row = with_number(original, 10)
    state = state_for(original, reserved_row, status="reserved")
    state["destination_id"] = _destination()
    malformed = record(
        transcript_variants_json=json.dumps(
            {"dialogue_lines": ["[00:01.0] Спикер 1: Неизвестная роль"]},
            ensure_ascii=False,
        )
    )
    valid = record(
        id=8,
        source_call_id="call-8",
        started_at="2026-08-14 10:03:04",
        phone="+71111111111",
    )
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[malformed, valid],
        state=state,
        fake=fake,
    )

    report = _run_execute(env["config"], "--limit", "1")

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert report["status"] == "published"
    assert report["published"] == 1
    assert len(fake.rows) == 1
    assert persisted["entries"][original["call_key"]]["status"] == "reserved"
    assert _db_status(env["db"], 7) == ("pending", 0)
    assert _db_status(env["db"], 8) == ("done", 1)


@pytest.mark.parametrize("corruption", ["duplicate_number", "invalid_hash", "invalid_status"])
def test_run_rejects_corrupt_owner_state_without_google_write(
    tmp_path, monkeypatch, corruption
):
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    state["destination_id"] = _destination()
    entry = state["entries"][call["call_key"]]
    if corruption == "duplicate_number":
        duplicate = dict(entry)
        state["entries"]["mango:mango_office:call-8"] = duplicate
    elif corruption == "invalid_hash":
        entry["source_fingerprint"] = "not-a-sha256"
    else:
        entry["status"] = "published"
    fake = FakeLiveGoogleGateway([row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], state=state, fake=fake
    )

    with pytest.raises(RuntimeError, match="publisher state"):
        _run_execute(env["config"])

    assert fake.batch_calls == 0


def test_run_reports_malformed_dialogue_as_data_error_without_partial_transcript(
    tmp_path, monkeypatch
):
    malformed = record(
        transcript_variants_json=json.dumps(
            {
                "role_mapping": {"left": "manager"},
                "dialogue_lines": [
                    "[00:01.0] Дорожка левая: Валидно",
                    "сломанная строка",
                ],
            },
            ensure_ascii=False,
        )
    )
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[malformed], fake=fake)

    report = publisher.run(["--config", str(env["config"])])

    assert report["status"] == "shadow_ok"
    assert report["analysis_done"] == 0
    assert report["data_errors"] == 1
    assert report["data_error_codes"] == {"projection_dialogue_line_malformed": 1}
    assert fake.rows == []
    assert fake.batch_calls == 0


def test_run_does_not_report_pending_empty_transcript_as_data_error(
    tmp_path, monkeypatch
):
    pending = record(
        analysis_status="pending",
        analysis_json=None,
        transcript_variants_json=None,
        transcript_text=None,
        duration_sec=None,
    )
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[pending], fake=fake)

    report = publisher.run(["--config", str(env["config"])])

    assert report["status"] == "shadow_ok"
    assert report["analysis_done"] == 0
    assert report["data_errors"] == 0
    assert fake.batch_calls == 0


@pytest.mark.parametrize("source_call_id", [None, ""])
def test_run_reports_done_row_without_source_call_id(tmp_path, monkeypatch, source_call_id):
    invalid = record(source_call_id=source_call_id)
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[invalid], fake=fake)

    report = publisher.run(["--config", str(env["config"])])

    assert report["status"] == "shadow_ok"
    assert report["analysis_done"] == 0
    assert report["data_errors"] == 1
    assert report["data_error_codes"] == {"identity_source_call_id_empty": 1}
    assert fake.batch_calls == 0


@pytest.mark.parametrize("source_call_id", ["x" * 600, "bad\ncall\nid"])
def test_run_rejects_unsafe_source_call_id_before_google_write(
    tmp_path, monkeypatch, source_call_id
):
    invalid = record(source_call_id=source_call_id)
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[invalid], fake=fake)

    first = publisher.run(["--config", str(env["config"])])
    second = publisher.run(["--config", str(env["config"])])

    assert first["data_error_codes"] == {"identity_source_call_id_invalid": 1}
    assert second["data_error_codes"] == first["data_error_codes"]
    assert fake.batch_calls == 0


@pytest.mark.parametrize("analysis_json", [None, "", "not-json", "[]", "null", "{}"])
def test_run_rejects_done_row_without_valid_analysis_object(
    tmp_path, monkeypatch, analysis_json
):
    invalid = record(analysis_json=analysis_json)
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[invalid], fake=fake)

    report = publisher.run(["--config", str(env["config"])])

    assert report["status"] == "shadow_ok"
    assert report["analysis_done"] == 0
    assert report["data_error_codes"] == {"projection_analysis_json_invalid": 1}
    assert fake.batch_calls == 0
    assert _db_status(env["db"]) == ("pending", 0)


def test_run_allows_only_one_writer_for_same_lock(tmp_path, monkeypatch):
    fake = FakeLiveGoogleGateway()
    entered = threading.Event()
    release = threading.Event()

    def block_first_values():
        entered.set()
        assert release.wait(timeout=5)

    fake.before_values = block_first_values
    env = _harness(tmp_path, monkeypatch, records=[record()], fake=fake)
    first_result = []
    first_error = []

    def first_run():
        try:
            first_result.append(publisher.run(["--config", str(env["config"])]))
        except Exception as exc:  # pragma: no cover - surfaced by assertions below
            first_error.append(exc)

    thread = threading.Thread(target=first_run)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="publisher is active"):
            publisher.run(["--config", str(env["config"])])
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_error == []
    assert first_result[0]["status"] == "shadow_ok"
    assert fake.batch_calls == 0


def test_run_db_busy_fails_before_sync_done(tmp_path, monkeypatch):
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    state["destination_id"] = _destination()
    fake = FakeLiveGoogleGateway([row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], rows=[row], state=state, fake=fake
    )
    real_connect = sqlite3.connect

    class BusyConnection:
        def __init__(self, inner):
            object.__setattr__(self, "inner", inner)

        def __setattr__(self, name, value):
            if name == "row_factory":
                self.inner.row_factory = value
            else:
                object.__setattr__(self, name, value)

        def execute(self, sql, parameters=()):
            if sql == "BEGIN IMMEDIATE":
                raise sqlite3.OperationalError("database is locked")
            return self.inner.execute(sql, parameters)

        def rollback(self):
            return self.inner.rollback()

        def close(self):
            return self.inner.close()

    def busy_connect(database, *args, **kwargs):
        connection = real_connect(database, *args, **kwargs)
        if isinstance(database, str) and database.startswith("file:"):
            return connection
        return BusyConnection(connection)

    monkeypatch.setattr(publisher.sqlite3, "connect", busy_connect)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _run_execute(env["config"])

    assert fake.batch_calls == 0
    assert _db_status(env["db"]) == ("pending", 0)
