from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline import CustomerIdentity, IdentityLink, IdentityMatchClass, IdentityStatus
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_mango_call_timeline_increment.py"

spec = importlib.util.spec_from_file_location("build_mango_call_timeline_increment", SCRIPT_PATH)
producer = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = producer
spec.loader.exec_module(producer)


def seed_customer_with_phone(db_path: Path, allowed_root: Path, *, customer_id: str, phone: str) -> None:
    with CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                customer_id=customer_id,
                identity_status=IdentityStatus.STRONG,
                primary_phone=phone,
                source_ref=f"seed:{customer_id}",
                first_seen_at=NOW,
                last_seen_at=NOW,
                touch_count=1,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer_id,
                link_type="phone",
                link_value=phone,
                source_system="seed",
                source_ref=f"seed:{customer_id}",
                match_class=IdentityMatchClass.STRONG_UNIQUE,
                confidence=0.95,
                first_seen_at=NOW,
                last_seen_at=NOW,
            )
        )


def create_call_records_db(path: Path, rows: list[dict]) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE call_records (
              id INTEGER PRIMARY KEY,
              source_call_id TEXT,
              source_filename TEXT,
              source_file TEXT,
              started_at TEXT,
              phone TEXT,
              manager_name TEXT,
              direction TEXT,
              duration_sec REAL,
              analysis_status TEXT,
              analysis_json TEXT,
              amocrm_contact_id TEXT,
              amocrm_lead_id TEXT
            )
            """
        )
        con.executemany(
            """
            INSERT INTO call_records (
              id, source_call_id, source_filename, source_file, started_at, phone,
              manager_name, direction, duration_sec, analysis_status, analysis_json,
              amocrm_contact_id, amocrm_lead_id
            )
            VALUES (
              :id, :source_call_id, :source_filename, :source_file, :started_at, :phone,
              :manager_name, :direction, :duration_sec, :analysis_status, :analysis_json,
              :amocrm_contact_id, :amocrm_lead_id
            )
            """,
            rows,
        )


def create_canonical_calls_db(path: Path, rows: list[dict]) -> None:
    with sqlite3.connect(path) as con:
        con.execute(
            """
            CREATE TABLE canonical_calls (
              canonical_call_id INTEGER PRIMARY KEY,
              source_call_id TEXT,
              source_filename TEXT,
              source_file TEXT,
              started_at TEXT,
              phone TEXT,
              manager_name TEXT,
              direction TEXT,
              duration_sec REAL,
              analysis_status TEXT,
              analysis_json TEXT,
              amocrm_contact_id TEXT,
              amocrm_lead_id TEXT
            )
            """
        )
        con.executemany(
            """
            INSERT INTO canonical_calls (
              canonical_call_id, source_call_id, source_filename, source_file, started_at, phone,
              manager_name, direction, duration_sec, analysis_status, analysis_json,
              amocrm_contact_id, amocrm_lead_id
            )
            VALUES (
              :canonical_call_id, :source_call_id, :source_filename, :source_file, :started_at, :phone,
              :manager_name, :direction, :duration_sec, :analysis_status, :analysis_json,
              :amocrm_contact_id, :amocrm_lead_id
            )
            """,
            rows,
        )


def analysis(summary: str = "Клиент уточнил стоимость.", *, call_type: str = "sales_call") -> str:
    return json.dumps(
        {
            "history_summary": summary,
            "call_quality_current": {"call_type": call_type},
            "next_step": "Передать менеджеру.",
        },
        ensure_ascii=False,
    )


def run_producer(tmp_path: Path, *, timeline_db: Path, package_db: Path, limit: int | None = None) -> tuple[list[dict], dict]:
    out_jsonl = tmp_path / "mango_increment.jsonl"
    report_out = tmp_path / "producer_report.json"
    argv = [
        "--timeline-db",
        str(timeline_db),
        "--package-db",
        str(package_db),
        "--out-jsonl",
        str(out_jsonl),
        "--report-out",
        str(report_out),
    ]
    if limit is not None:
        argv.extend(["--limit", str(limit)])
    assert producer.main(argv) == 0
    events = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(report_out.read_text(encoding="utf-8"))
    return events, report


def run_canonical_producer(tmp_path: Path, *, timeline_db: Path, canonical_db: Path) -> tuple[list[dict], dict]:
    out_jsonl = tmp_path / "mango_canonical_increment.jsonl"
    report_out = tmp_path / "producer_canonical_report.json"
    argv = [
        "--timeline-db",
        str(timeline_db),
        "--canonical-db",
        str(canonical_db),
        "--out-jsonl",
        str(out_jsonl),
        "--report-out",
        str(report_out),
    ]
    assert producer.main(argv) == 0
    events = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(report_out.read_text(encoding="utf-8"))
    return events, report


def test_producer_uses_existing_identity_links_and_mango_processed_summary(tmp_path: Path) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_phone(timeline_db, tmp_path, customer_id="customer:one", phone="+79161112233")
    seed_customer_with_phone(timeline_db, tmp_path, customer_id="customer:family-a", phone="+79164445566")
    seed_customer_with_phone(timeline_db, tmp_path, customer_id="customer:family-b", phone="+79164445566")
    package_db = tmp_path / "calls.sqlite"
    create_call_records_db(
        package_db,
        [
            {
                "id": 1,
                "source_call_id": "27100000001",
                "source_filename": "call-one.wav",
                "source_file": "/ignored/call-one.wav",
                "started_at": "2026-06-25T09:00:00+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": "Менеджер",
                "direction": "inbound",
                "duration_sec": 120,
                "analysis_status": "done",
                "analysis_json": analysis(),
                "amocrm_contact_id": "amo-contact-1",
                "amocrm_lead_id": "amo-lead-1",
            },
            {
                "id": 2,
                "source_call_id": "27100000002",
                "source_filename": "call-family.wav",
                "source_file": "/ignored/call-family.wav",
                "started_at": "2026-06-25T09:05:00+00:00",
                "phone": "+7 916 444-55-66",
                "manager_name": "Менеджер",
                "direction": "inbound",
                "duration_sec": 140,
                "analysis_status": "done",
                "analysis_json": analysis(),
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            },
        ],
    )

    events, report = run_producer(tmp_path, timeline_db=timeline_db, package_db=package_db)

    assert [event["source_system"] for event in events] == ["mango_processed_summary", "mango_processed_summary"]
    assert [event["event_type"] for event in events] == ["mango_call", "mango_call"]
    assert events[0]["customer_id"] == "customer:one"
    assert events[0]["match_class"] == "strong_unique"
    assert "customer_id" not in events[1]
    assert events[1]["match_class"] == "ambiguous"
    assert events[1]["identity_resolution_reason"] == "multiple_existing_customers"
    assert report["identity_resolution_counts"] == {"strong_unique": 1, "ambiguous": 1}
    assert report["safety"]["writes_amo"] is False
    assert report["safety"]["runs_analyze"] is False
    assert "+79161112233" not in json.dumps(report, ensure_ascii=False)


def test_producer_reads_only_done_rows_with_valid_analysis_json(tmp_path: Path) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_phone(timeline_db, tmp_path, customer_id="customer:one", phone="+79161112233")
    package_db = tmp_path / "calls.sqlite"
    create_call_records_db(
        package_db,
        [
            {
                "id": 1,
                "source_call_id": "done-valid",
                "source_filename": "done.wav",
                "source_file": "/ignored/done.wav",
                "started_at": "2026-06-25T09:00:00+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": None,
                "direction": None,
                "duration_sec": None,
                "analysis_status": "done",
                "analysis_json": analysis(),
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            },
            {
                "id": 2,
                "source_call_id": "not-done",
                "source_filename": "pending.wav",
                "source_file": "/ignored/pending.wav",
                "started_at": "2026-06-25T09:01:00+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": None,
                "direction": None,
                "duration_sec": None,
                "analysis_status": "pending",
                "analysis_json": analysis(),
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            },
            {
                "id": 3,
                "source_call_id": "invalid-json",
                "source_filename": "invalid.wav",
                "source_file": "/ignored/invalid.wav",
                "started_at": "2026-06-25T09:02:00+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": None,
                "direction": None,
                "duration_sec": None,
                "analysis_status": "done",
                "analysis_json": "not json",
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            },
        ],
    )

    events, report = run_producer(tmp_path, timeline_db=timeline_db, package_db=package_db)

    assert [event["original_call_id"] for event in events] == ["done-valid"]
    assert report["rows_read"] == 1
    assert report["events_written"] == 1


def test_canonical_source_id_uses_canonical_call_id_for_existing_timeline_compatibility(tmp_path: Path) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_phone(timeline_db, tmp_path, customer_id="customer:one", phone="+79161112233")
    canonical_db = tmp_path / "canonical.sqlite"
    create_canonical_calls_db(
        canonical_db,
        [
            {
                "canonical_call_id": 43409,
                "source_call_id": "provider-id-if-present",
                "source_filename": "2025-11-03__10-37-33__34604932284__manager.mp3",
                "source_file": "/ignored/2025-11-03__10-37-33__34604932284__manager.mp3",
                "started_at": "2025-11-03T10:37:33+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": None,
                "direction": None,
                "duration_sec": None,
                "analysis_status": "done",
                "analysis_json": analysis(),
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            }
        ],
    )

    events, report = run_canonical_producer(tmp_path, timeline_db=timeline_db, canonical_db=canonical_db)

    assert events[0]["call_id"] == "43409"
    assert events[0]["source_ref"] == "mango:43409"
    assert report["source_counts"] == {"canonical_calls": 1}


def test_package_duplicate_source_call_id_is_stable_even_when_limit_selects_one_row(tmp_path: Path) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_phone(timeline_db, tmp_path, customer_id="customer:one", phone="+79161112233")
    package_db = tmp_path / "calls.sqlite"
    create_call_records_db(
        package_db,
        [
            {
                "id": 1,
                "source_call_id": "same-provider-id",
                "source_filename": "first.wav",
                "source_file": "/ignored/first.wav",
                "started_at": "2026-06-25T09:00:00+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": None,
                "direction": None,
                "duration_sec": None,
                "analysis_status": "done",
                "analysis_json": analysis(),
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            },
            {
                "id": 2,
                "source_call_id": "same-provider-id",
                "source_filename": "second.wav",
                "source_file": "/ignored/second.wav",
                "started_at": "2026-06-25T09:05:00+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": None,
                "direction": None,
                "duration_sec": None,
                "analysis_status": "done",
                "analysis_json": analysis(),
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            },
        ],
    )

    events, _report = run_producer(tmp_path, timeline_db=timeline_db, package_db=package_db, limit=1)

    assert len(events) == 1
    assert events[0]["call_id"].startswith("provider:same-provider-id:")
