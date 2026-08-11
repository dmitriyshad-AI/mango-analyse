from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import CustomerIdentity, IdentityLink, IdentityMatchClass, IdentityStatus
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 6, 25, 9, 0, tzinfo=timezone.utc)
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_mango_call_timeline_increment.py"

spec = importlib.util.spec_from_file_location("build_mango_call_timeline_increment", SCRIPT_PATH)
producer = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = producer
spec.loader.exec_module(producer)


@pytest.mark.parametrize("schema", ["CREATE TABLE other (id INTEGER)", "CREATE TABLE call_records (id INTEGER)"])
def test_ready_call_rows_rejects_missing_required_schema(tmp_path: Path, schema: str) -> None:
    db = tmp_path / "broken.sqlite"
    with sqlite3.connect(db) as con:
        con.execute(schema)

    with pytest.raises(ValueError, match="required call"):
        producer.read_ready_call_rows(db, table="call_records", source_kind="call_records")


def test_brand_evidence_is_deterministic_single_both_none() -> None:
    assert producer.detect_brand_evidence("Позвонили из центра Фотон") == ("single", ("foton",))
    assert producer.detect_brand_evidence("УНПК МФТИ") == ("single", ("unpk",))
    assert producer.detect_brand_evidence("Фотон и УНПК") == ("both", ("foton", "unpk"))
    assert producer.detect_brand_evidence("Обсудили занятия") == ("none", ())


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
              transcript_text TEXT,
              amocrm_contact_id TEXT,
              amocrm_lead_id TEXT
            )
            """
        )
        con.executemany(
            """
            INSERT INTO call_records (
              id, source_call_id, source_filename, source_file, started_at, phone,
              manager_name, direction, duration_sec, analysis_status, analysis_json, transcript_text,
              amocrm_contact_id, amocrm_lead_id
            )
            VALUES (
              :id, :source_call_id, :source_filename, :source_file, :started_at, :phone,
              :manager_name, :direction, :duration_sec, :analysis_status, :analysis_json, :transcript_text,
              :amocrm_contact_id, :amocrm_lead_id
            )
            """,
            [{**row, "transcript_text": row.get("transcript_text", "")} for row in rows],
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


def run_producer(
    tmp_path: Path,
    *,
    timeline_db: Path,
    package_db: Path,
    limit: int | None = None,
    strict_service_ready: bool = False,
) -> tuple[list[dict], dict]:
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
    if strict_service_ready:
        argv.append("--strict-service-ready")
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
                "transcript_text": "Обсуждали центр Фотон.",
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
                "analysis_json": analysis("Обсуждали УНПК."),
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
    assert [event["brand_evidence"] for event in events] == ["single", "single"]
    assert [event["brand_evidence_brands"] for event in events] == [["foton"], ["unpk"]]
    assert report["brand_evidence_counts"] == {"single": 2}
    assert report["brand_counts"] == {"foton": 1, "unpk": 1}
    assert report["safety"]["writes_amo"] is False
    assert report["safety"]["runs_analyze"] is False
    assert "+79161112233" not in json.dumps(report, ensure_ascii=False)


def test_producer_fails_loudly_on_done_row_with_invalid_analysis_json(tmp_path: Path) -> None:
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

    with pytest.raises(ValueError, match="invalid done analysis_json"):
        run_producer(tmp_path, timeline_db=timeline_db, package_db=package_db)


def test_service_ready_predicate_keeps_quarantine_out_of_timeline(
    tmp_path: Path,
) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_phone(
        timeline_db,
        tmp_path,
        customer_id="customer:one",
        phone="+79161112233",
    )
    package_db = tmp_path / "strict-ready.sqlite"
    variants = json.dumps(
        {
            "mode": "mono_or_fallback",
            "primary_provider": "mlx",
            "secondary_provider": "gigaam",
            "full": {
                "variant_a": "готовый Whisper",
                "variant_b": "готовый GigaAM",
            },
        },
        ensure_ascii=False,
    )
    with sqlite3.connect(package_db) as con:
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
              transcription_status TEXT,
              transcript_variants_json TEXT,
              resolve_status TEXT,
              analysis_status TEXT,
              analysis_json TEXT,
              dead_letter_stage TEXT,
              last_error TEXT,
              pipeline_stage TEXT,
              pipeline_worker_id TEXT,
              pipeline_claimed_at TEXT,
              analysis_worker_id TEXT,
              analysis_claimed_at TEXT
            )
            """
        )
        con.executemany(
            """
            INSERT INTO call_records VALUES (
              :id, :source_call_id, :source_filename, :source_file,
              :started_at, :phone, :manager_name, :direction, :duration_sec,
              :transcription_status, :transcript_variants_json,
              :resolve_status, :analysis_status, :analysis_json,
              :dead_letter_stage, :last_error, :pipeline_stage,
              :pipeline_worker_id, :pipeline_claimed_at,
              :analysis_worker_id, :analysis_claimed_at
            )
            """,
            [
                {
                    "id": 1,
                    "source_call_id": "shared-call",
                    "source_filename": "ready.wav",
                    "source_file": "/ignored/ready.wav",
                    "started_at": "2026-06-25T09:00:00+00:00",
                    "phone": "+7 916 111-22-33",
                    "manager_name": "Менеджер",
                    "direction": "inbound",
                    "duration_sec": 60,
                    "transcription_status": "done",
                    "transcript_variants_json": variants,
                    "resolve_status": "done",
                    "analysis_status": "done",
                    "analysis_json": analysis("Готовый звонок."),
                    "dead_letter_stage": None,
                    "last_error": None,
                    "pipeline_stage": None,
                    "pipeline_worker_id": None,
                    "pipeline_claimed_at": None,
                    "analysis_worker_id": None,
                    "analysis_claimed_at": None,
                },
                {
                    "id": 2,
                    "source_call_id": "shared-call",
                    "source_filename": "quarantine.wav",
                    "source_file": "/ignored/quarantine.wav",
                    "started_at": "2026-06-25T09:05:00+00:00",
                    "phone": "+7 916 111-22-33",
                    "manager_name": "Менеджер",
                    "direction": "inbound",
                    "duration_sec": 60,
                    "transcription_status": "failed",
                    "transcript_variants_json": variants,
                    "resolve_status": "failed",
                    "analysis_status": "done",
                    "analysis_json": analysis("Не публиковать."),
                    "dead_letter_stage": "resolve",
                    "last_error": "synthetic private failure",
                    "pipeline_stage": None,
                    "pipeline_worker_id": None,
                    "pipeline_claimed_at": None,
                    "analysis_worker_id": None,
                    "analysis_claimed_at": None,
                },
            ],
        )

    events, report = run_producer(
        tmp_path,
        timeline_db=timeline_db,
        package_db=package_db,
        strict_service_ready=True,
    )

    assert report["rows_read"] == 1
    assert report["rows_selected"] == 1
    assert report["events_written"] == 1
    assert len(events) == 1
    stable_ready_id = events[0]["call_id"]
    assert stable_ready_id.startswith("provider:shared-call:")
    serialized = json.dumps(events, ensure_ascii=False)
    assert "Не публиковать" not in serialized

    with sqlite3.connect(package_db) as con:
        con.execute(
            """
            UPDATE call_records
            SET transcription_status='done', resolve_status='done',
                dead_letter_stage=NULL, last_error=NULL
            WHERE id=2
            """
        )

    recovered_events, recovered_report = run_producer(
        tmp_path,
        timeline_db=timeline_db,
        package_db=package_db,
        strict_service_ready=True,
    )

    assert recovered_report["rows_read"] == 2
    assert recovered_report["events_written"] == 2
    assert len({event["call_id"] for event in recovered_events}) == 2
    recovered_ready = next(
        event for event in recovered_events if event["call_at"].startswith("2026-06-25T09:00:00")
    )
    assert recovered_ready["call_id"] == stable_ready_id


def test_migrated_legacy_columns_do_not_imply_strict_service_mode(
    tmp_path: Path,
) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_phone(
        timeline_db,
        tmp_path,
        customer_id="customer:one",
        phone="+79161112233",
    )
    package_db = tmp_path / "migrated-legacy.sqlite"
    create_call_records_db(
        package_db,
        [
            {
                "id": 1,
                "source_call_id": "legacy-call",
                "source_filename": "legacy.wav",
                "source_file": "/ignored/legacy.wav",
                "started_at": "2026-06-25T09:00:00+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": "Менеджер",
                "direction": "inbound",
                "duration_sec": 60,
                "analysis_status": "done",
                "analysis_json": analysis("Исторический готовый звонок."),
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            }
        ],
    )
    with sqlite3.connect(package_db) as con:
        con.execute("ALTER TABLE call_records ADD COLUMN transcription_status TEXT")
        con.execute("ALTER TABLE call_records ADD COLUMN transcript_variants_json TEXT")
        con.execute("ALTER TABLE call_records ADD COLUMN resolve_status TEXT")
        con.execute("ALTER TABLE call_records ADD COLUMN dead_letter_stage TEXT")
        con.execute(
            """
            UPDATE call_records
            SET transcription_status='done', transcript_variants_json='{}',
                resolve_status='done'
            """
        )

    legacy_events, legacy_report = run_producer(
        tmp_path,
        timeline_db=timeline_db,
        package_db=package_db,
    )
    assert legacy_report["events_written"] == 1
    assert [event["call_id"] for event in legacy_events] == ["provider:legacy-call"]
    with pytest.raises(ValueError, match="strict service readiness columns"):
        run_producer(
            tmp_path,
            timeline_db=timeline_db,
            package_db=package_db,
            strict_service_ready=True,
        )


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


def test_package_duplicate_source_call_id_is_stable_when_sibling_is_not_done(tmp_path: Path) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    seed_customer_with_phone(timeline_db, tmp_path, customer_id="customer:one", phone="+79161112233")
    package_db = tmp_path / "calls.sqlite"
    create_call_records_db(
        package_db,
        [
            {
                "id": 1,
                "source_call_id": "same-provider-id",
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
                "source_call_id": "same-provider-id",
                "source_filename": "pending.wav",
                "source_file": "/ignored/pending.wav",
                "started_at": "2026-06-25T09:05:00+00:00",
                "phone": "+7 916 111-22-33",
                "manager_name": None,
                "direction": None,
                "duration_sec": None,
                "analysis_status": "pending",
                "analysis_json": "",
                "amocrm_contact_id": None,
                "amocrm_lead_id": None,
            },
        ],
    )

    events, report = run_producer(tmp_path, timeline_db=timeline_db, package_db=package_db)

    assert report["events_written"] == 1
    assert events[0]["call_id"].startswith("provider:same-provider-id:")
