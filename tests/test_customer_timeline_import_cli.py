from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
from pathlib import Path

from mango_mvp.customer_timeline.import_cli import (
    TimelineImportCliConfig,
    decode_delimiter,
    main,
    run_timeline_import_cli,
    timeline_import_cli_safety_contract,
)
from mango_mvp.customer_timeline.ingestion import file_sha256
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


SHA = "c" * 64


def test_dry_run_cli_writes_report_without_creating_timeline_db(tmp_path: Path) -> None:
    source = tmp_path / "amocrm_entities.json"
    source.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "entity_id": "lead-1",
                        "entity_type": "lead",
                        "name": "ЕГЭ математика",
                        "phone": "+7 999 111-22-33",
                        "updated_at": "2026-05-12T10:00:00+00:00",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "reports" / "timeline_import_preview.json"
    timeline_db = tmp_path / "customer_timeline" / "customer_timeline.sqlite"

    rc = main(
        [
            "--tenant-id",
            "foton",
            "--source-kind",
            "amocrm_snapshot",
            "--source-path",
            str(source),
            "--allowed-root",
            str(tmp_path),
            "--timeline-db",
            str(timeline_db),
            "--source-ref",
            "amo-test",
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["mode"] == "dry_run_preview"
    assert report["dry_run"] is True
    assert report["validation_ok"] is True
    assert report["operator_summary"]["records_loaded"] == 1
    assert report["operator_summary"]["write_applied"] is False
    assert report["summary"]["writes_planned"] == 5
    assert report["source"]["inventory"]["unchanged"] is True
    assert report["normalization"]["by_source_record"][0]["counts"]["customers"] == 1
    assert report["writes"]["planned_counts_by_type"]["timeline_event"] == 1
    assert report["import_report"]["normalized_counts"]["customers"] == 1
    assert report["import_report"]["normalized_counts"]["opportunities"] == 1
    assert report["safety"]["requires_apply_for_db_write"] is True
    assert report["safety"]["write_product_timeline_db"] is False
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["run_ra"] is False
    assert not timeline_db.exists()


def test_apply_cli_imports_tallanto_csv_idempotently_and_records_conflict(tmp_path: Path) -> None:
    source = tmp_path / "students.csv"
    source.write_text(
        "entity_id\tname\temail\tphone\tcourse\tupdated_at\n"
        "s1\tИван Петров\tparent@example.com\t+7 916 111-22-33\tЕГЭ математика\t2026-05-01T10:00:00+00:00\n"
        "s2\tМария Петрова\tparent@example.com\t+7 916 111-22-33\tЕГЭ русский\t2026-05-01T10:05:00+00:00\n",
        encoding="cp1251",
    )
    before = source_snapshot(source)
    timeline_db = tmp_path / "customer_timeline.sqlite"
    report_one = tmp_path / "report_one.json"
    report_two = tmp_path / "report_two.json"
    argv = [
        "--tenant-id",
        "foton",
        "--source-kind",
        "tallanto_snapshot",
        "--source-path",
        str(source),
        "--allowed-root",
        str(tmp_path),
        "--timeline-db",
        str(timeline_db),
        "--source-ref",
        "students.csv",
        "--idempotency-key",
        "students-v1",
        "--csv-encoding",
        "cp1251",
        "--csv-delimiter",
        "\\t",
        "--apply",
    ]

    assert main([*argv, "--out", str(report_one)]) == 0
    assert main([*argv, "--out", str(report_two)]) == 0

    report = json.loads(report_two.read_text(encoding="utf-8"))
    store = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    summary = store.summary()
    runs = store.list_ingestion_runs("foton")["items"]
    conflicts = store.list_audit_log("foton", entity_type="timeline_conflict")["items"]
    store.close()
    assert before == source_snapshot(source)
    assert report["mode"] == "apply"
    assert report["safety"]["write_product_timeline_db"] is True
    assert report["source_unchanged"] is True
    assert report["summary"]["conflicts_open"] == 1
    assert report["conflicts"]["counts_by_type"]["ambiguous_identity"] == 1
    assert report["writes"]["status_counts"]
    assert report["store_summary_before"]["counts"]["ingestion_runs"] == 1
    assert report["store_summary_after"]["counts"]["ingestion_runs"] == 1
    assert summary["counts"]["customer_identities"] == 2
    assert summary["counts"]["timeline_conflicts"] == 1
    assert len(runs) == 1
    assert conflicts[0]["action"] == "timeline_conflict_created"


def test_sqlite_source_is_read_only_and_imports_mail_metadata(tmp_path: Path) -> None:
    source_db = tmp_path / "mail_source.sqlite"
    with sqlite3.connect(source_db) as con:
        con.execute(
            """
            CREATE TABLE messages (
              message_id TEXT,
              message_date_iso TEXT,
              subject TEXT,
              from_email TEXT,
              to_email TEXT,
              text_preview TEXT,
              raw_eml_path TEXT,
              sha256 TEXT,
              raw_size_bytes INTEGER
            )
            """
        )
        con.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "m-1",
                "2026-05-03T09:00:00+00:00",
                "Стоимость курса",
                "client@example.com",
                "edu@kmipt.ru",
                "Подскажите стоимость курса",
                "/mail/raw/m-1.eml",
                SHA,
                2048,
            ),
        )
    before = source_snapshot(source_db)
    timeline_db = tmp_path / "timeline.sqlite"
    report_path = tmp_path / "mail_report.json"

    rc = main(
        [
            "--tenant-id",
            "foton",
            "--source-kind",
            "mail_archive",
            "--source-path",
            str(source_db),
            "--allowed-root",
            str(tmp_path),
            "--timeline-db",
            str(timeline_db),
            "--source-ref",
            "mail-handoff",
            "--sqlite-table",
            "messages",
            "--sqlite-source-ref-column",
            "message_id",
            "--out",
            str(report_path),
            "--apply",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    store = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    summary = store.summary()
    event = store.search_timeline("foton", "стоимость")["items"][0]["record"]
    store.close()
    assert rc == 0
    assert before == source_snapshot(source_db)
    assert report["source_unchanged"] is True
    assert report["import_report"]["normalized_counts"]["events"] == 1
    assert summary["counts"]["timeline_events"] == 1
    assert summary["counts"]["event_artifacts"] == 1
    assert event["event_type"] == "email_message"


def test_cli_rejects_stable_runtime_and_runtime_looking_target_db(tmp_path: Path) -> None:
    stable_root = tmp_path / "stable_runtime"
    stable_root.mkdir()
    stable_source = stable_root / "rows.json"
    stable_source.write_text("[]", encoding="utf-8")
    safe_source = tmp_path / "rows.json"
    safe_source.write_text("[]", encoding="utf-8")

    assert (
        main(
            [
                "--tenant-id",
                "foton",
                "--source-kind",
                "amocrm_snapshot",
                "--source-path",
                str(stable_source),
                "--allowed-root",
                str(tmp_path),
                "--timeline-db",
                str(tmp_path / "timeline.sqlite"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "--tenant-id",
                "foton",
                "--source-kind",
                "amocrm_snapshot",
                "--source-path",
                str(safe_source),
                "--allowed-root",
                str(tmp_path),
                "--timeline-db",
                str(tmp_path / "mango_product_appliance.sqlite"),
            ]
        )
        == 2
    )


def test_cli_does_not_use_network_or_subprocess(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network/subprocess must not be used")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    monkeypatch.setattr(os, "system", fail)
    monkeypatch.setattr(socket, "socket", fail)
    source = tmp_path / "messages.jsonl"
    source.write_text(
        json.dumps(
            {
                "channel": "telegram",
                "channel_thread_id": "thread-1",
                "channel_message_id": "msg-1",
                "channel_user_id": "tg-1",
                "text": "Здравствуйте",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    rc = main(
        [
            "--tenant-id",
            "foton",
            "--source-kind",
            "channel_snapshot",
            "--source-path",
            str(source),
            "--allowed-root",
            str(tmp_path),
            "--out",
            str(report_path),
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert report["safety"]["network_calls"] is False
    assert report["safety"]["subprocess_calls"] is False
    assert report["safety"]["send_messenger"] is False


def test_operation_plan_is_parent_first_and_preview_is_directly_callable(tmp_path: Path) -> None:
    source = tmp_path / "calls.jsonl"
    source.write_text(
        json.dumps(
            {
                "call_id": "call-1",
                "client_phone": "+79991112233",
                "call_at": "2026-05-04T10:00:00+00:00",
                "summary": "Клиент интересуется оплатой.",
                "recommended_action": "Перезвонить завтра",
                "audio_path": "/audio/call-1.mp3",
                "audio_path_sha256": SHA,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_timeline_import_cli(
        TimelineImportCliConfig(
            tenant_id="foton",
            source_kind="mango_processed_summary",
            source_path=source,
            allowed_root=tmp_path,
            timeline_db=tmp_path / "customer_timeline.sqlite",
            source_ref="calls",
        )
    )
    operations = report["preview"]["operation_plan"]["items"]

    assert report["mode"] == "dry_run_preview"
    assert operations[0]["record_type"] == "customer_identity"
    assert [item["record_type"] for item in operations[:4]] == [
        "customer_identity",
        "identity_link",
        "identity_link",
        "timeline_event",
    ]
    assert '"call_id":' not in json.dumps(report, ensure_ascii=False)
    assert report["source"]["records"][0]["payload_hash"]
    assert decode_delimiter("\\t") == "\t"
    assert timeline_import_cli_safety_contract(write_product_timeline_db=False)["write_product_timeline_db"] is False


def source_snapshot(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
    }
