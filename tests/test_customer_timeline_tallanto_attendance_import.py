from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from openpyxl import Workbook

import mango_mvp.customer_timeline.tallanto_attendance_import as attendance_module

from mango_mvp.customer_timeline.contracts import CustomerIdentity, IdentityLink, IdentityLinkType, IdentityStatus, TimelineDirection, TimelineEvent, TimelineEventType
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.tallanto_attendance_import import (
    TallantoAttendanceApiIncrementConfig,
    TallantoAttendanceImportConfig,
    run_tallanto_attendance_api_increment,
    run_tallanto_attendance_import,
)


def test_attendance_import_maps_barcode_via_tallanto_snapshot_and_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    contacts = _contacts(tmp_path, (("101", "barcode-1"),))
    attendance = _attendance(tmp_path, "barcode-1")

    config = TallantoAttendanceImportConfig(db, tmp_path, contacts, attendance, apply=True)
    first = run_tallanto_attendance_import(config)
    second = run_tallanto_attendance_import(config)

    assert first["counts"]["created"] == 1
    assert first["counts"]["would_create"] == 1
    assert second["counts"]["duplicate"] == 1
    assert second["counts"]["would_duplicate"] == 1
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT customer_id,event_type,summary,record_json FROM timeline_events WHERE source_system='tallanto_attendance'").fetchone()
    assert row[:3] == (
        "customer:student",
        "tallanto_attendance",
        "Списание за занятие подтверждено в Tallanto; физическое присутствие отдельно не доказано.",
    )
    assert '"attendance_evidence":"class_writeoff"' in row[3]


def test_attendance_import_leaves_unknown_barcode_unmatched(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(db, tmp_path, _contacts(tmp_path, (("101", "other"),)), _attendance(tmp_path, "missing"), apply=False)
    )
    assert report["counts"]["unmatched"] == 1
    assert report["counts"]["unmatched_barcodes"] == 1
    assert report["validation_ok"] is False
    assert report["validation_errors"] == ["all_rows_unmatched"]
    with pytest.raises(ValueError, match="all_rows_unmatched"):
        run_tallanto_attendance_import(
            TallantoAttendanceImportConfig(
                db,
                tmp_path,
                _contacts(tmp_path, (("101", "other"),)),
                _attendance(tmp_path, "missing"),
                apply=True,
            )
        )
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT count(*) FROM timeline_events WHERE source_system='tallanto_attendance'").fetchone()[0] == 0


def test_attendance_import_matches_numeric_excel_barcode_to_text_report(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)

    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", 1234567890123),)),
            _attendance(tmp_path, "1234567890123"),
            apply=True,
        )
    )

    assert report["counts"]["created"] == 1


def test_attendance_import_honours_manual_exact_identity_link(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with sqlite3.connect(db) as con:
        con.execute(
            "UPDATE identity_links SET match_class='manual' "
            "WHERE tenant_id='foton' AND link_type='tallanto_student_id' AND link_value='101'"
        )

    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", "barcode-1"),)),
            _attendance(tmp_path, "barcode-1"),
            apply=False,
        )
    )

    assert report["counts"]["resolved"] == 1
    assert report["counts"].get("unmatched", 0) == 0


def test_attendance_import_reuses_unique_historical_tallanto_customer_link(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM identity_links WHERE link_type='tallanto_student_id'")
    now = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:student",
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=now,
                source_system="tallanto_attendance",
                source_id="old-attendance",
                direction=TimelineDirection.SYSTEM,
                match_status="strong_unique",
                confidence=1.0,
                record={"logical_key": {"tallanto_id": "101"}},
                created_at=now,
            )
        )

    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", "barcode-1"),)),
            _attendance(tmp_path, "barcode-1"),
            apply=False,
        )
    )

    assert report["counts"]["resolved"] == 1
    assert report["counts"].get("unmatched", 0) == 0


def test_attendance_import_reuses_history_when_current_link_is_only_inferred(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE identity_links SET match_class='inferred' WHERE link_type='tallanto_student_id'")
    now = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:student",
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=now,
                source_system="tallanto_attendance",
                source_id="trusted-history",
                direction=TimelineDirection.SYSTEM,
                match_status="strong_unique",
                confidence=1.0,
                record={"logical_key": {"tallanto_id": "101"}},
                created_at=now,
            )
        )

    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", "barcode-1"),)),
            _attendance(tmp_path, "barcode-1"),
            apply=False,
        )
    )

    assert report["counts"]["resolved"] == 1
    assert report["counts"].get("unmatched", 0) == 0


def test_attendance_import_does_not_override_open_tallanto_identity_conflict_with_history(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    now = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:student",
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=now,
                source_system="tallanto_attendance",
                source_id="trusted-history",
                direction=TimelineDirection.SYSTEM,
                match_status="strong_unique",
                confidence=1.0,
                record={"logical_key": {"tallanto_id": "101"}},
                created_at=now,
            )
        )
    with sqlite3.connect(db) as con:
        con.execute("UPDATE identity_links SET match_class='inferred' WHERE link_type='tallanto_student_id'")
        con.execute(
            "INSERT INTO timeline_conflicts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "conflict:tallanto:101",
                "foton",
                "tallanto_identity_ambiguous",
                "high",
                "open",
                now.isoformat(),
                None,
                "hash:tallanto:101",
                json.dumps({"entity_refs": ["tallanto_student_id:101"]}),
            ),
        )

    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", "barcode-1"),)),
            _attendance(tmp_path, "barcode-1"),
            apply=False,
        )
    )

    assert report["counts"].get("resolved", 0) == 0
    assert report["counts"]["unmatched"] == 1


@pytest.mark.parametrize(
    ("match_status", "confidence", "superseded"),
    (
        ("inferred", 1.0, False),
        ("strong_unique", 0.98, False),
        ("strong_unique", 1.0, True),
    ),
)
def test_attendance_import_does_not_promote_untrusted_historical_identity(
    tmp_path: Path,
    match_status: str,
    confidence: float,
    superseded: bool,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM identity_links WHERE link_type='tallanto_student_id'")
    now = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        result = store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:student",
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=now,
                source_system="tallanto_attendance",
                source_id="old-attendance",
                direction=TimelineDirection.SYSTEM,
                match_status=match_status,
                confidence=confidence,
                record={"logical_key": {"tallanto_id": "101"}},
                created_at=now,
            )
        )
    if superseded:
        with sqlite3.connect(db) as con:
            con.execute(
                "UPDATE timeline_events SET superseded_by='replacement' WHERE event_id=?",
                (result.record_id,),
            )

    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", "barcode-1"),)),
            _attendance(tmp_path, "barcode-1"),
            apply=False,
        )
    )

    assert report["validation_ok"] is False
    assert report["counts"]["unmatched"] == 1


def test_attendance_import_resolves_new_barcode_via_exact_tallanto_and_amo_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:student",
                link_type=IdentityLinkType.AMO_CONTACT_ID,
                link_value="7001",
                source_system="amo",
                source_ref="amo:contact:7001",
            )
        )

    class FakeTallantoClient:
        def get_entry_by_fields(self, **_kwargs):
            return {"id": "202", "barcode": "new-barcode", "amo_id": "7001"}

    monkeypatch.setattr(attendance_module, "_build_tallanto_client", lambda _path: FakeTallantoClient())
    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", "old-barcode"),)),
            _attendance(tmp_path, "new-barcode"),
            apply=True,
            tallanto_env_file=tmp_path / "tallanto.env",
        )
    )

    assert report["counts"]["api_resolved_barcodes"] == 1
    assert report["counts"]["created"] == 1
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT customer_id FROM identity_links WHERE link_type='tallanto_student_id' AND link_value='202'"
        ).fetchone()[0] == "customer:student"


def test_attendance_import_blocks_conflicting_tallanto_and_amo_customers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_customer(db, tmp_path, "customer:tallanto")
    _seed_customer(db, tmp_path, "customer:amo")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:tallanto",
                link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
                link_value="202",
                source_system="tallanto_snapshot",
                source_ref="tallanto:202",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:amo",
                link_type=IdentityLinkType.AMO_CONTACT_ID,
                link_value="7001",
                source_system="amo",
                source_ref="amo:contact:7001",
            )
        )

    class FakeTallantoClient:
        def get_entry_by_fields(self, **_kwargs):
            return {"id": "202", "barcode": "new-barcode", "amo_id": "7001"}

    monkeypatch.setattr(attendance_module, "_build_tallanto_client", lambda _path: FakeTallantoClient())
    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", "old-barcode"),)),
            _attendance(tmp_path, "new-barcode"),
            apply=False,
            tallanto_env_file=tmp_path / "tallanto.env",
        )
    )

    assert report["counts"]["api_identity_conflicts"] == 1
    assert report["counts"]["api_resolved_barcodes"] == 0
    assert report["validation_ok"] is False


def test_attendance_import_refuses_apply_outside_staging(tmp_path: Path) -> None:
    db = tmp_path / "timeline.sqlite"
    _seed_tallanto_customer(db, tmp_path)
    with pytest.raises(ValueError, match="staging"):
        run_tallanto_attendance_import(
            TallantoAttendanceImportConfig(db, tmp_path, _contacts(tmp_path, (("101", "barcode-1"),)), _attendance(tmp_path, "barcode-1"), apply=True)
        )


def test_attendance_import_refuses_staging_name_prefix(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging_evil" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with pytest.raises(ValueError, match="staging"):
        run_tallanto_attendance_import(
            TallantoAttendanceImportConfig(
                db,
                tmp_path,
                _contacts(tmp_path, (("101", "barcode-1"),)),
                _attendance(tmp_path, "barcode-1"),
                apply=True,
            )
        )


def test_attendance_import_refuses_duplicate_contact_barcode(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    contacts = _contacts(tmp_path, (("101", "barcode-1"), ("202", "barcode-1")))

    with pytest.raises(ValueError, match="duplicate contact barcodes"):
        run_tallanto_attendance_import(
            TallantoAttendanceImportConfig(db, tmp_path, contacts, _attendance(tmp_path, "barcode-1"), apply=True)
        )


@pytest.mark.parametrize(
    ("attendance", "expected_error"),
    (
        ("empty", "empty_report"),
        ("invalid", "all_rows_invalid_class_at"),
    ),
)
def test_attendance_import_rejects_empty_or_all_invalid_report(
    tmp_path: Path,
    attendance: str,
    expected_error: str,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    report_path = _attendance(
        tmp_path,
        "barcode-1",
        include_row=attendance != "empty",
        class_at="not-a-date" if attendance == "invalid" else "01.07.2026 10:00",
    )
    report = run_tallanto_attendance_import(
        TallantoAttendanceImportConfig(
            db,
            tmp_path,
            _contacts(tmp_path, (("101", "barcode-1"),)),
            report_path,
            apply=False,
        )
    )
    assert report["validation_ok"] is False
    assert report["validation_errors"] == [expected_error]


def test_failed_ingestion_status_survives_event_transaction_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)

    def fail_event(*_args, **_kwargs):
        raise RuntimeError("synthetic write failure")

    monkeypatch.setattr(CustomerTimelineSQLiteStore, "upsert_event", fail_event)
    with pytest.raises(RuntimeError, match="synthetic write failure"):
        run_tallanto_attendance_import(
            TallantoAttendanceImportConfig(
                db,
                tmp_path,
                _contacts(tmp_path, (("101", "barcode-1"),)),
                _attendance(tmp_path, "barcode-1"),
                apply=True,
            )
        )

    with sqlite3.connect(db) as con:
        assert con.execute("SELECT status,error FROM ingestion_runs").fetchone() == (
            "failed",
            "attendance_write_failed",
        )
        assert con.execute(
            "SELECT count(*) FROM timeline_events WHERE source_system='tallanto_attendance'"
        ).fetchone()[0] == 0


@pytest.mark.parametrize(
    ("status", "abonement", "attendance", "absence", "writeoff", "active"),
    (
        ("visit", "abon-1", True, False, True, True),
        ("no-show", "abon-1", False, True, True, True),
        ("planned", "abon-1", False, False, False, False),
    ),
)
def test_attendance_api_preserves_visit_no_show_and_unfinished_semantics(
    tmp_path: Path,
    status: str,
    abonement: str,
    attendance: bool,
    absence: bool,
    writeoff: bool,
    active: bool,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    client = FakeAttendanceApi(status=status, abonement=abonement)

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=client,
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["created"] == 1
    assert report["status"] == "completed"
    assert report["cursor_after"] != report["cursor_before"]
    assert client.http_methods == {"GET"}
    assert sum("most_class_contacts_c.most_class_id IN" in query for query in client.queries) == 1
    with sqlite3.connect(db) as con:
        raw, summary = con.execute(
            "SELECT record_json,summary FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone()
    payload = __import__("json").loads(raw)["record"]
    assert payload["attendance_confirmed"] is attendance
    assert payload["physical_absence_confirmed"] is absence
    assert payload["writeoff_confirmed"] is writeoff
    assert payload["writeoff_amount"] is None
    assert payload["fact_active"] is active
    assert ("не завершён" in summary) is (not active)


def test_attendance_api_honours_manual_exact_identity_link(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with sqlite3.connect(db) as con:
        con.execute(
            "UPDATE identity_links SET match_class='manual' "
            "WHERE tenant_id='foton' AND link_type='tallanto_student_id' AND link_value='101'"
        )

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["created"] == 1
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT customer_id FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone() == ("customer:student",)


def test_attendance_api_unfinished_update_removes_previous_active_fact(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit", modified="2026-07-23 10:00:00"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="planned", modified="2026-07-24 10:00:00"),
        now=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
    )

    with sqlite3.connect(db) as con:
        rows = con.execute(
            "SELECT record_json FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchall()
    assert len(rows) == 1
    assert __import__("json").loads(rows[0][0])["record"]["fact_active"] is False


def test_attendance_api_preserves_tallanto_amo_identity_conflict_with_durable_retry(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_customer(db, tmp_path, "customer:tallanto")
    _seed_customer(db, tmp_path, "customer:amo")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:tallanto",
                link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
                link_value="101",
                source_system="tallanto_snapshot",
                source_ref="tallanto:101",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:amo",
                link_type=IdentityLinkType.AMO_CONTACT_ID,
                link_value="amo-101",
                source_system="amo",
                source_ref="amo:amo-101",
            )
        )

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["identity_conflict"] == 1
    assert report["counts"]["events_resolved"] == 0
    assert report["status"] == "completed"
    assert report["validation_ok"] is True
    assert report["validation_errors"] == []
    with sqlite3.connect(db) as con:
        cursor = con.execute(
            "SELECT last_cursor_ts, metadata_json FROM ingestion_cursors "
            "WHERE source_system='tallanto_attendance_api'"
        ).fetchone()
        assert cursor is not None
        assert cursor[0] == report["cursor_after"]
        assert json.loads(cursor[1])["metadata"]["class_overlap_until"] == report["class_overlap_after"]
        assert con.execute(
            "SELECT status FROM ingestion_runs WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == "completed"
        assert con.execute(
            "SELECT conflict_type,status FROM timeline_conflicts"
        ).fetchone() == ("tallanto_attendance_api_identity_conflict", "open")


def test_attendance_api_quarantines_event_when_exact_owner_becomes_conflicting(tmp_path: Path) -> None:
    from mango_mvp.customer_timeline.manager_dossier import _chronology_rows

    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    first = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    assert first["counts"]["created"] == 1
    _seed_customer(db, tmp_path, "customer:unseen")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:unseen",
                link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
                link_value="202",
                source_system="tallanto_snapshot",
                source_ref="tallanto:202",
            )
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:unseen",
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
                source_system="tallanto_attendance_api",
                source_id="unseen-relationship",
                direction=TimelineDirection.SYSTEM,
                match_status="strong_unique",
                confidence=1.0,
                record={"tallanto_student_id": "202", "writeoff_confirmed": True},
            )
        )
        store.record_conflict(
            "foton",
            conflict_type="tallanto_identity_conflict",
            entity_refs=("customer:student", "tallanto_student_id:101"),
            severity="high",
        )
        store.record_conflict(
            "foton",
            conflict_type="tallanto_identity_conflict",
            entity_refs=("customer:unseen", "tallanto_student_id:202"),
            severity="high",
        )

    second = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
    )
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        event = con.execute(
            "SELECT customer_id,match_status,confidence,record_json,record_hash FROM timeline_events "
            "WHERE source_system='tallanto_attendance_api' "
            "AND json_extract(record_json,'$.record.tallanto_student_id')='101'"
        ).fetchone()
        unseen_owner = con.execute(
            "SELECT customer_id FROM timeline_events WHERE source_id='unseen-relationship'"
        ).fetchone()[0]
        second_hash = str(event["record_hash"])
        chronology = _chronology_rows(con, tenant_id="foton", customer_id="customer:student", limit=20)

    third = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc),
    )
    with sqlite3.connect(db) as con:
        repeated = con.execute(
            "SELECT COUNT(*),MAX(CASE WHEN json_extract(record_json,'$.record.tallanto_student_id')='101' "
            "THEN record_hash END) FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone()

    metadata = json.loads(event["record_json"])["metadata"]
    assert second["status"] == "completed"
    assert second["counts"]["existing_events_identity_conflict"] == 1
    assert second["counts"]["existing_event_quarantined"] == 1
    assert event["customer_id"] is None
    assert event["match_status"] == "ambiguous"
    assert float(event["confidence"]) == 0.0
    assert metadata["pending_attribution"] is True
    assert not any(row.section == "Хронология" for row in chronology)
    assert unseen_owner == "customer:unseen"
    assert third["counts"]["existing_events_identity_conflict"] == 0
    assert third["counts"].get("existing_event_quarantined", 0) == 0
    assert repeated == (2, second_hash)


@pytest.mark.parametrize(
    ("conflict_type", "candidate_count", "expected_resolved"),
    (
        ("shared_family_phone", None, True),
        ("tallanto_identity_ambiguous", 1, True),
        ("tallanto_identity_ambiguous", True, True),
        ("tallanto_identity_ambiguous", 4, True),
        ("tallanto_identity_conflict", None, False),
    ),
)
def test_attendance_api_uses_exact_student_owner_without_ignoring_real_ambiguity(
    tmp_path: Path,
    conflict_type: str,
    candidate_count: int | None,
    expected_resolved: bool,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    metadata = {} if candidate_count is None else {"candidate_customer_count": candidate_count}
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.record_conflict(
            "foton",
            conflict_type=conflict_type,
            entity_refs=("customer:student", "tallanto_student_id:101"),
            severity="high",
            metadata=metadata,
        )

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    with sqlite3.connect(db) as con:
        event_count = con.execute(
            "SELECT count(*) FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0]
        assert con.execute(
            "SELECT status FROM timeline_conflicts WHERE conflict_type=?",
            (conflict_type,),
        ).fetchone()[0] == "open"
        api_conflict = con.execute(
            "SELECT conflict_type, severity FROM timeline_conflicts "
            "WHERE conflict_type='tallanto_attendance_api_identity_conflict'"
        ).fetchone()
    if expected_resolved:
        assert report["status"] == "completed"
        assert report["counts"]["events_resolved"] == 1
        assert event_count == 1
    else:
        assert report["status"] == "completed"
        assert report["counts"]["identity_conflict"] == 1
        assert report["cursor_after"] != report["cursor_before"]
        assert report["unresolved_breakdown"]["blocking_count"] == 0
        assert event_count == 0
        assert api_conflict == ("tallanto_attendance_api_identity_conflict", "high")


def test_attendance_api_closes_durable_retry_when_source_record_disappears(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_customer(db, tmp_path, "customer:tallanto")
    _seed_customer(db, tmp_path, "customer:amo")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(IdentityLink(
            tenant_id="foton", customer_id="customer:tallanto",
            link_type=IdentityLinkType.TALLANTO_STUDENT_ID, link_value="101",
            source_system="tallanto_snapshot", source_ref="tallanto:101",
        ))
        store.upsert_identity_link(IdentityLink(
            tenant_id="foton", customer_id="customer:amo",
            link_type=IdentityLinkType.AMO_CONTACT_ID, link_value="amo-101",
            source_system="amo", source_ref="amo:amo-101",
        ))
    first = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True), client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    class MissingRetryApi(FakeAttendanceApi):
        def request(self, *, module, http_method, query_items, **kwargs):
            order_by = dict(query_items).get("order_by", "")
            if module == "ClassContactsRelationship" and not order_by.startswith("date_modified DESC"):
                return {"entry_list": [], "result_count": 0, "total_count": 0}
            return super().request(
                module=module, http_method=http_method, query_items=query_items, **kwargs
            )

        def get_entry_by_id(self, *, module, entry_id, **kwargs):
            if module == "ClassContactsRelationship":
                error = RuntimeError("synthetic not found")
                error.category = "not_found"
                raise error
            return super().get_entry_by_id(module=module, entry_id=entry_id, **kwargs)

    second = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True), client=MissingRetryApi(status="visit"),
        now=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
    )

    assert first["cursor_after"] != first["cursor_before"]
    assert second["validation_errors"] == []
    assert second["counts"]["identity_retry_source_missing"] == 1
    assert second["status"] == "completed"
    assert second["class_overlap_after"] != second["class_overlap_before"]
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT status FROM timeline_conflicts "
            "WHERE conflict_type='tallanto_attendance_api_identity_conflict'"
        ).fetchone() == ("resolved",)


def test_attendance_api_does_not_close_retry_when_direct_lookup_finds_record(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_customer(db, tmp_path, "customer:tallanto")
    _seed_customer(db, tmp_path, "customer:amo")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(IdentityLink(
            tenant_id="foton", customer_id="customer:tallanto",
            link_type=IdentityLinkType.TALLANTO_STUDENT_ID, link_value="101",
            source_system="tallanto_snapshot", source_ref="tallanto:101",
        ))
        store.upsert_identity_link(IdentityLink(
            tenant_id="foton", customer_id="customer:amo",
            link_type=IdentityLinkType.AMO_CONTACT_ID, link_value="amo-101",
            source_system="amo", source_ref="amo:amo-101",
        ))
        store.record_conflict(
            "foton",
            conflict_type="tallanto_attendance_api_identity_conflict",
            entity_refs=("tallanto:class-contact:rel-1", "tallanto:contact:101"),
            severity="high",
            metadata={"relationship_id": "rel-1", "contact_id": "101", "most_class_id": "class-1"},
        )

    class DirectLookupApi(FakeAttendanceApi):
        def request(self, *, module, http_method, query_items, **kwargs):
            order_by = dict(query_items).get("order_by", "")
            if module == "ClassContactsRelationship" and not order_by.startswith("date_modified DESC"):
                return {"entry_list": [], "result_count": 0, "total_count": 0}
            return super().request(
                module=module, http_method=http_method, query_items=query_items, **kwargs
            )

        def get_entry_by_id(self, *, module, entry_id, **kwargs):
            if module == "ClassContactsRelationship":
                return {
                    "id": entry_id, "most_class_id": "class-1", "contact_id": "101",
                    "most_class_contacts_status": "visit", "most_class_abonements": "abon-1",
                    "date_modified": "2026-07-23 10:00:00", "date_entry": "2026-07-22 09:00:00",
                }
            return super().get_entry_by_id(module=module, entry_id=entry_id, **kwargs)

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True), client=DirectLookupApi(status="visit"),
        now=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["identity_retry_source_missing"] == 0
    assert report["counts"]["identity_conflict"] == 1
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT status FROM timeline_conflicts "
            "WHERE conflict_type='tallanto_attendance_api_identity_conflict'"
        ).fetchone() == ("open",)


def test_attendance_api_holds_cursor_for_malformed_durable_retry_metadata(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.record_conflict(
            "foton",
            conflict_type="tallanto_attendance_api_identity_conflict",
            entity_refs=("tallanto:class-contact:rel-old", "tallanto:contact:101"),
            severity="high",
            metadata={"relationship_id": "rel-old", "contact_id": "101"},
        )

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True), client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["validation_errors"] == ["identity_retry_metadata_invalid"]
    assert report["counts"]["identity_retry_metadata_invalid"] == 1
    assert report["cursor_after"] == report["cursor_before"]


def test_attendance_api_rejects_two_exact_tallanto_owners(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    _seed_customer(db, tmp_path, "customer:other")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:other",
                link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
                link_value="101",
                source_system="legacy_tallanto",
                source_ref="legacy:tallanto:101",
            )
        )

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["status"] == "completed"
    assert report["counts"]["identity_conflict"] == 1
    assert report["unresolved_breakdown"]["blocking_count"] == 0
    assert report["counts"]["events_resolved"] == 0
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT count(*) FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == 0


def test_attendance_api_blocks_ambiguous_student_id_without_exact_owner(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.record_conflict(
            "foton",
            conflict_type="tallanto_identity_ambiguous",
            entity_refs=("tallanto_student_id:101",),
            severity="high",
            metadata={"candidate_customer_count": 3},
        )

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["identity_conflict"] == 1
    assert report["counts"]["events_resolved"] == 0
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == 0


def test_attendance_api_partially_imports_reliable_events_and_retries_idempotently(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    _seed_customer(db, tmp_path, "customer:tallanto-conflict")
    _seed_customer(db, tmp_path, "customer:amo-conflict")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:tallanto-conflict",
                link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
                link_value="202",
                source_system="tallanto_snapshot",
                source_ref="tallanto:202",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:amo-conflict",
                link_type=IdentityLinkType.AMO_CONTACT_ID,
                link_value="amo-202",
                source_system="amo",
                source_ref="amo:amo-202",
            )
        )

    first = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=MixedAttendanceApi(),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    second = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=MixedAttendanceApi(),
        now=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
    )

    assert first["status"] == second["status"] == "partial"
    assert first["unresolved_count"] == second["unresolved_count"] == 2
    assert first["counts"]["created"] == 1
    assert second["counts"]["duplicate"] == 1
    assert first["counts"]["unresolved_created"] == 2
    assert second["counts"]["unresolved_duplicate"] == 2
    assert first["cursor_after"] == first["cursor_before"]
    assert second["cursor_after"] == second["cursor_before"]
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT count(*) FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM timeline_conflicts WHERE status='open'").fetchone()[0] == 2
        cursor = con.execute(
            "SELECT last_cursor_ts, metadata_json FROM ingestion_cursors "
            "WHERE source_system='tallanto_attendance_api'"
        ).fetchone()
        assert cursor is not None
        assert cursor[0] == second["cursor_before"]
        assert json.loads(cursor[1])["metadata"]["class_overlap_until"] == second["class_overlap_after"]
        assert con.execute(
            "SELECT count(*) FROM ingestion_runs WHERE source_system='tallanto_attendance_api' AND status='partial'"
        ).fetchone()[0] == 2


def test_attendance_api_accepts_exact_student_and_parent_in_same_confident_family(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_customer(db, tmp_path, "customer:student")
    _seed_customer(db, tmp_path, "customer:parent")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:student",
                link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
                link_value="101",
                source_system="tallanto_snapshot",
                source_ref="tallanto:101",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:parent",
                link_type=IdentityLinkType.AMO_CONTACT_ID,
                link_value="amo-101",
                source_system="amo",
                source_ref="amo:amo-101",
            )
        )
    with sqlite3.connect(db) as con:
        con.executemany(
            "INSERT INTO family_members_v1 "
            "(tenant_id,family_id,customer_id,membership_status,confidence,reason,created_at,updated_at,record_hash,record_json) "
            "VALUES ('foton','family:1',?,'confident','high','test','2026-07-24T00:00:00+00:00',"
            "'2026-07-24T00:00:00+00:00','test','{}')",
            (("customer:student",), ("customer:parent",)),
        )

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=False),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["identity_same_family"] == 1
    assert report["counts"]["events_resolved"] == 1
    assert report["counts"].get("identity_conflict", 0) == 0


def test_attendance_api_resolves_prior_conflict_from_direct_family_amo_link(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_customer(db, tmp_path, "customer:student")
    _seed_customer(db, tmp_path, "customer:parent")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(IdentityLink(
            tenant_id="foton", customer_id="customer:student",
            link_type=IdentityLinkType.TALLANTO_STUDENT_ID, link_value="101",
            source_system="tallanto_snapshot", source_ref="tallanto:101",
        ))
        store.upsert_identity_link(IdentityLink(
            tenant_id="foton", customer_id="customer:parent",
            link_type=IdentityLinkType.AMO_CONTACT_ID, link_value="amo-101",
            source_system="amo", source_ref="amo:amo-101",
        ))

    first = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True), client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )
    assert first["counts"]["identity_conflict"] == 1

    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(IdentityLink(
            tenant_id="foton", customer_id="customer:student",
            link_type=IdentityLinkType.AMO_CONTACT_ID, link_value="amo-101",
            source_system="tallanto_snapshot", source_ref="tallanto:101",
            match_class="ambiguous", confidence=1.0,
            evidence={"relationship": "family_amo_contact"},
        ))

    second = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True), client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc),
    )

    assert second["status"] == "completed"
    assert second["counts"]["identity_direct_family"] == 1
    assert second["counts"]["events_resolved"] == 1
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT status FROM timeline_conflicts WHERE conflict_type='tallanto_attendance_api_identity_conflict'"
        ).fetchone()[0] == "resolved"


def test_attendance_api_retries_and_closes_legacy_unmatched_conflict(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.record_conflict(
            "foton",
            conflict_type="tallanto_attendance_api_identity_unmatched",
            entity_refs=("tallanto:class-contact:rel-1", "tallanto:contact:101"),
            severity="low",
            metadata={
                "reason": "identity_unmatched",
                "relationship_id": "rel-1",
                "contact_id": "101",
                "most_class_id": "class-1",
            },
        )

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True), client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["events_resolved"] == 1
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT status FROM timeline_conflicts "
            "WHERE conflict_type='tallanto_attendance_api_identity_unmatched'"
        ).fetchone() == ("resolved",)


def test_attendance_api_fails_loud_on_incomplete_pagination(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    client = FakeAttendanceApi(status="visit", incomplete=True)

    with pytest.raises(ValueError, match="ended before total_count"):
        run_tallanto_attendance_api_increment(
            _api_config(db, tmp_path, apply=False),
            client=client,
            now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
        )

    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT count(*) FROM ingestion_cursors WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == 0


def test_attendance_api_cursor_rolls_back_when_event_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    monkeypatch.setattr(
        CustomerTimelineSQLiteStore,
        "upsert_event",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic")),
    )

    with pytest.raises(RuntimeError, match="synthetic"):
        run_tallanto_attendance_api_increment(
            _api_config(db, tmp_path, apply=True),
            client=FakeAttendanceApi(status="visit"),
            now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
        )

    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT count(*) FROM ingestion_cursors WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT status FROM ingestion_runs WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == "failed"


def test_attendance_api_rolls_back_reliable_event_when_unresolved_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    monkeypatch.setattr(
        CustomerTimelineSQLiteStore,
        "record_conflict",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic unresolved write failure")),
    )

    with pytest.raises(RuntimeError, match="synthetic unresolved write failure"):
        run_tallanto_attendance_api_increment(
            _api_config(db, tmp_path, apply=True),
            client=MixedAttendanceApi(),
            now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
        )

    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT count(*) FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM timeline_conflicts").fetchone()[0] == 0
        assert con.execute(
            "SELECT count(*) FROM ingestion_cursors WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT status FROM ingestion_runs WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == "failed"


def test_attendance_api_marks_technical_time_when_class_date_is_missing(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit", class_date_missing=True),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["class_date_missing"] == 1
    with sqlite3.connect(db) as con:
        raw, summary = con.execute(
            "SELECT record_json,summary FROM timeline_events WHERE source_system='tallanto_attendance_api'"
        ).fetchone()
    assert __import__("json").loads(raw)["record"]["event_at_source"] == "relationship.date_entry"
    assert "Точная дата занятия" in summary


# --- D4: expected unmatched / identity conflict / infrastructure error are
# distinct, never-silenced buckets. Only an explicit conflict has a durable
# class-based retry queue, so the other two still hold freshness. ---


def test_attendance_api_expected_unmatched_blocks_cursor_until_identity_exists(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    # Deliberately do not seed any tallanto/amo identity link for contact
    # "101": the Contact fetch still returns it (so resolution was actually
    # attempted with full data), it just has nothing to resolve to yet.
    CustomerTimelineSQLiteStore(db, allowed_root=tmp_path).close()
    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["identity_unmatched_expected"] == 1
    assert report["counts"].get("identity_conflict", 0) == 0
    assert report["counts"].get("identity_infrastructure_gap", 0) == 0
    assert report["unresolved_count"] == 1
    assert report["unresolved_breakdown"] == {
        "identity_conflict": 0,
        "identity_infrastructure_gap": 0,
        "identity_unmatched_expected": 1,
        "blocking_count": 1,
        "identity_conflict_has_durable_retry": True,
        "expected_unmatched_blocks_freshness": True,
        "input_fully_processed": True,
    }
    assert report["status"] == "partial"
    assert report["validation_ok"] is False
    assert report["validation_errors"] == ["identity_unmatched_expected"]
    assert report["cursor_after"] == report["cursor_before"]
    assert report["class_overlap_after"] == report["class_overlap_before"]
    with sqlite3.connect(db) as con:
        cursor = con.execute(
            "SELECT last_cursor_ts, metadata_json FROM ingestion_cursors "
            "WHERE source_system='tallanto_attendance_api'"
        ).fetchone()
        assert cursor[0] == report["cursor_before"]
        assert json.loads(cursor[1])["metadata"]["class_overlap_until"] == report["class_overlap_after"]
        # Still visible, never silenced: recorded as a (low severity) conflict.
        assert con.execute(
            "SELECT conflict_type, severity FROM timeline_conflicts"
        ).fetchone() == ("tallanto_attendance_api_identity_unmatched_expected", "low")
        # No strong link and no event were fabricated for the unresolved contact.
        assert con.execute("SELECT count(*) FROM timeline_events WHERE source_system='tallanto_attendance_api'").fetchone()[0] == 0
        assert con.execute(
            "SELECT count(*) FROM identity_links WHERE link_value='101' AND link_type='tallanto_student_id'"
        ).fetchone()[0] == 0

    _seed_customer(db, tmp_path, "customer:later-linked")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        for link_type, value in (
            (IdentityLinkType.TALLANTO_STUDENT_ID, "101"),
            (IdentityLinkType.AMO_CONTACT_ID, "amo-101"),
        ):
            store.upsert_identity_link(
                IdentityLink(
                    tenant_id="foton",
                    customer_id="customer:later-linked",
                    link_type=link_type,
                    link_value=value,
                    source_system="test",
                    source_ref=f"test:{value}",
                )
            )

    retried = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 5, tzinfo=timezone.utc),
    )
    assert retried["counts"]["created"] == 1
    assert retried["cursor_after"] != retried["cursor_before"]
    assert retried["class_overlap_after"] != retried["class_overlap_before"]
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT status FROM timeline_conflicts "
            "WHERE conflict_type='tallanto_attendance_api_identity_unmatched_expected'"
        ).fetchone() == ("resolved",)


def test_attendance_api_infrastructure_gap_blocks_freshness_and_cursor(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)

    class InfrastructureGapApi(FakeAttendanceApi):
        def request(self, *, module, http_method, query_items, **kwargs):
            if module == "Contact":
                # The Contact fetch itself came back incomplete for this
                # run -- contact "101" (referenced by the relationship
                # below) is never returned, unlike a genuine "nothing links
                # here yet" case where the contact row is present.
                return {"entry_list": [], "result_count": 0, "total_count": 0}
            return super().request(module=module, http_method=http_method, query_items=query_items, **kwargs)

    CustomerTimelineSQLiteStore(db, allowed_root=tmp_path).close()
    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=InfrastructureGapApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["identity_infrastructure_gap"] == 1
    assert report["counts"].get("identity_conflict", 0) == 0
    assert report["counts"].get("identity_unmatched_expected", 0) == 0
    assert report["unresolved_breakdown"]["blocking_count"] == 1
    assert report["status"] == "partial"
    assert report["validation_ok"] is False
    assert report["validation_errors"] == ["identity_infrastructure_gap"]
    assert report["cursor_after"] == report["cursor_before"]
    with sqlite3.connect(db) as con:
        cursor = con.execute(
            "SELECT last_cursor_ts, metadata_json FROM ingestion_cursors "
            "WHERE source_system='tallanto_attendance_api'"
        ).fetchone()
        assert cursor[0] == report["cursor_before"]
        assert json.loads(cursor[1])["metadata"]["class_overlap_until"] == report["class_overlap_after"]
        assert con.execute(
            "SELECT conflict_type, severity FROM timeline_conflicts"
        ).fetchone() == ("tallanto_attendance_api_identity_infrastructure_gap", "medium")


def test_attendance_api_conflict_never_creates_a_strong_link(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_customer(db, tmp_path, "customer:tallanto")
    _seed_customer(db, tmp_path, "customer:amo")
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:tallanto",
                link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
                link_value="101",
                source_system="tallanto_snapshot",
                source_ref="tallanto:101",
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:amo",
                link_type=IdentityLinkType.AMO_CONTACT_ID,
                link_value="amo-101",
                source_system="amo",
                source_ref="amo:amo-101",
            )
        )
    links_before = _identity_link_count(db)

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=FakeAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["counts"]["identity_conflict"] == 1
    # No new identity link (strong or otherwise) was written for the
    # conflicting contact -- conflict never merges/links, only the
    # pre-existing two customers' own links remain.
    assert _identity_link_count(db) == links_before
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT count(*) FROM timeline_events WHERE source_system='tallanto_attendance_api'").fetchone()[0] == 0


def test_attendance_api_empty_first_run_does_not_advance_cursor(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    CustomerTimelineSQLiteStore(db, allowed_root=tmp_path).close()

    class EmptyAttendanceApi(FakeAttendanceApi):
        def request(self, *, module, http_method, query_items, **_kwargs):
            self.http_methods.add(http_method)
            return {"entry_list": [], "result_count": 0, "total_count": 0}

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=EmptyAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["status"] == "partial"
    assert report["validation_errors"] == ["no_attendance_events"]
    assert report["cursor_after"] == report["cursor_before"]
    with sqlite3.connect(db) as con:
        cursor = con.execute(
            "SELECT last_cursor_ts, metadata_json FROM ingestion_cursors "
            "WHERE source_system='tallanto_attendance_api'"
        ).fetchone()
        assert cursor[0] == report["cursor_before"]
        assert json.loads(cursor[1])["metadata"]["class_overlap_until"] == report["class_overlap_after"]


def test_attendance_api_empty_repeat_with_existing_history_is_valid_no_op(tmp_path: Path) -> None:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    _seed_tallanto_customer(db, tmp_path)
    now = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(db, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id="customer:student",
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=now,
                source_system="tallanto_attendance_api",
                source_id="existing-api-attendance",
                direction=TimelineDirection.SYSTEM,
                created_at=now,
            )
        )

    class EmptyAttendanceApi(FakeAttendanceApi):
        def request(self, *, module, http_method, query_items, **_kwargs):
            self.http_methods.add(http_method)
            return {"entry_list": [], "result_count": 0, "total_count": 0}

    report = run_tallanto_attendance_api_increment(
        _api_config(db, tmp_path, apply=True),
        client=EmptyAttendanceApi(status="visit"),
        now=datetime(2026, 7, 24, 12, 0, tzinfo=timezone.utc),
    )

    assert report["status"] == "completed"
    assert report["validation_ok"] is True
    assert report["validation_errors"] == []
    assert report["counts"]["existing_events_before"] == 1
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT count(*) FROM ingestion_cursors WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == 1


def _identity_link_count(db: Path) -> int:
    with sqlite3.connect(db) as con:
        return int(con.execute("SELECT count(*) FROM identity_links").fetchone()[0])


def _seed_tallanto_customer(db: Path, root: Path) -> None:
    _seed_customer(db, root, "customer:student")
    with CustomerTimelineSQLiteStore(db, allowed_root=root) as store:
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:student",
                link_type=IdentityLinkType.TALLANTO_STUDENT_ID,
                link_value="101",
                source_system="tallanto_snapshot",
                source_ref="tallanto:101",
            )
        )


def _api_config(db: Path, root: Path, *, apply: bool) -> TallantoAttendanceApiIncrementConfig:
    return TallantoAttendanceApiIncrementConfig(
        timeline_db=db,
        allowed_root=root,
        tallanto_env_file=root / "unused.env",
        initial_since=datetime(2026, 7, 13, tzinfo=timezone.utc),
        apply=apply,
    )


class FakeAttendanceApi:
    def __init__(
        self,
        *,
        status: str,
        abonement: str = "abon-1",
        modified: str = "2026-07-23 10:00:00",
        incomplete: bool = False,
        class_date_missing: bool = False,
    ) -> None:
        self.status = status
        self.abonement = abonement
        self.modified = modified
        self.incomplete = incomplete
        self.class_date_missing = class_date_missing
        self.http_methods: set[str] = set()
        self.queries: list[str] = []

    def request(self, *, module, http_method, query_items, **_kwargs):
        self.http_methods.add(http_method)
        query = dict(query_items).get("query", "")
        self.queries.append(query)
        order_by = dict(query_items).get("order_by", "")
        relationship = {
            "id": "rel-1",
            "most_class_id": "class-1",
            "contact_id": "101",
            "most_class_contacts_status": self.status,
            "most_class_abonements": self.abonement,
            "date_modified": self.modified,
            "date_entry": "2026-07-22 09:00:00",
        }
        class_row = {
            "id": "class-1",
            "name": "Физика",
            "date_start": "2026-07-23 12:00:00",
            "status": "held",
            "cource_id": "course-1",
            "filial": "МФТИ",
        }
        if module == "ClassContactsRelationship" and order_by.startswith("date_modified DESC"):
            return {"entry_list": [{"id": "rel-1", "date_modified": self.modified}]}
        if module == "most_class":
            if self.class_date_missing and query.startswith("most_class.date_start"):
                return {"entry_list": [], "result_count": 0, "total_count": 0}
            if self.class_date_missing:
                class_row.pop("date_start")
            return {"entry_list": [class_row], "result_count": 1, "total_count": 1}
        if module == "Contact":
            return {
                "entry_list": [{"id": "101", "amo_id": "amo-101"}],
                "result_count": 1,
                "total_count": 1,
            }
        if module == "ClassContactsRelationship":
            total = 2 if self.incomplete and "fields_values[most_class_id]" not in dict(query_items) else 1
            return {"entry_list": [relationship], "result_count": 1, "total_count": total}
        raise AssertionError((module, query))

    def get_entry_by_id(self, *, module, entry_id, **_kwargs):
        if module == "most_class":
            return {
                "id": entry_id,
                "name": "Физика",
                "date_start": "2026-07-23 12:00:00",
            }
        raise AssertionError(module)


class MixedAttendanceApi(FakeAttendanceApi):
    def __init__(self) -> None:
        super().__init__(status="visit")

    def request(self, *, module, http_method, query_items, **kwargs):
        if module not in {"ClassContactsRelationship", "Contact"}:
            return super().request(
                module=module,
                http_method=http_method,
                query_items=query_items,
                **kwargs,
            )
        self.http_methods.add(http_method)
        order_by = dict(query_items).get("order_by", "")
        relationships = [
            {
                "id": f"rel-{index}",
                "most_class_id": "class-1",
                "contact_id": contact_id,
                "most_class_contacts_status": "visit",
                "most_class_abonements": "abon-1",
                "date_modified": "2026-07-23 10:00:00",
                "date_entry": "2026-07-22 09:00:00",
            }
            for index, contact_id in enumerate(("101", "202", "303"), start=1)
        ]
        if module == "Contact":
            rows = [
                {"id": "101", "amo_id": "amo-101"},
                {"id": "202", "amo_id": "amo-202"},
                {"id": "303", "amo_id": ""},
            ]
            return {"entry_list": rows, "result_count": len(rows), "total_count": len(rows)}
        if order_by.startswith("date_modified DESC"):
            return {"entry_list": [relationships[-1]]}
        return {
            "entry_list": relationships,
            "result_count": len(relationships),
            "total_count": len(relationships),
        }


def _seed_customer(db: Path, root: Path, customer_id: str) -> None:
    with CustomerTimelineSQLiteStore(db, allowed_root=root) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id=customer_id,
                identity_status=IdentityStatus.STRONG,
            )
        )


def _contacts(tmp_path: Path, rows: tuple[tuple[object, object], ...]) -> Path:
    path = tmp_path / "contacts.xlsx"
    workbook = Workbook(); sheet = workbook.active
    sheet.append(("ID", "Текстовое значение штрихкода"))
    for row in rows: sheet.append(row)
    workbook.save(path)
    return path


def _attendance(
    tmp_path: Path,
    barcode: str,
    *,
    include_row: bool = True,
    class_at: str = "01.07.2026 10:00",
) -> Path:
    path = tmp_path / "attendance.xlsx"
    workbook = Workbook(); sheet = workbook.active
    sheet.append(("Фамилия", "Имя", "Штрихкод", "Абонемент", "Сумма списания", "Дата списания", "Тип списания", "Занятие", "Филиал занятия", "Дата занятия", "День рождения"))
    if include_row:
        sheet.append(("Иванов", "Иван", barcode, "Абонемент", "1000", "01.07.2026 12:00", "Безналичный расчёт", "Физика 8 класс", "МФТИ", class_at, ""))
    workbook.save(path)
    return path
