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


def test_attendance_api_preserves_tallanto_amo_identity_conflict_without_advancing_cursor(tmp_path: Path) -> None:
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
    assert report["status"] == "partial"
    assert report["validation_ok"] is False
    assert report["validation_errors"] == ["identity_conflict"]
    with sqlite3.connect(db) as con:
        cursor = con.execute(
            "SELECT last_cursor_ts, metadata_json FROM ingestion_cursors "
            "WHERE source_system='tallanto_attendance_api'"
        ).fetchone()
        assert cursor is not None
        assert cursor[0] == report["cursor_before"]
        assert json.loads(cursor[1])["metadata"]["class_overlap_until"] == report["class_overlap_after"]
        assert con.execute(
            "SELECT status FROM ingestion_runs WHERE source_system='tallanto_attendance_api'"
        ).fetchone()[0] == "partial"
        assert con.execute(
            "SELECT conflict_type,status FROM timeline_conflicts"
        ).fetchone() == ("tallanto_attendance_api_identity_conflict", "open")


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
# three distinct, never-silenced buckets; all block freshness until there is
# a durable retry queue. ---


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

    def request(self, *, module, http_method, query_items, **_kwargs):
        self.http_methods.add(http_method)
        query = dict(query_items).get("query", "")
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
