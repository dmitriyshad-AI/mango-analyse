from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityStatus,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.safety import assert_customer_timeline_safety_contract
from mango_mvp.customer_timeline.ingestion import timeline_ingestion_safety_contract
from scripts.import_tallanto_payments_to_timeline import (
    TallantoPaymentsImportConfig,
    main,
    run_tallanto_payments_import,
)


NOW = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)


def test_dry_run_stdin_mcp_snapshot_imports_payments_and_abonements_without_creating_db(tmp_path: Path) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(
            source=None,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=False,
        ),
        stdin_text=json.dumps(mcp_snapshot(), ensure_ascii=False),
    )

    assert report["validation_ok"] is True
    assert timeline_db.exists() is False
    assert report["mode"] == "dry_run_preview"
    assert report["summary"]["records_loaded"] == 2
    assert report["summary"]["payment_events"] == 1
    assert report["summary"]["abonement_events"] == 1
    assert report["import_report"]["normalized_counts"]["events"] == 2
    assert report["import_report"]["normalized_counts"]["opportunities"] == 2
    assert report["import_report"]["normalized_counts"]["bot_context_chunks"] == 0
    assert report["source"]["path"] == "stdin"
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["write_tallanto"] is False
    assert report["safety"]["send_messenger"] is False
    assert report["safety"]["run_asr"] is False
    assert report["safety"]["run_ra"] is False
    assert report["safety"]["write_product_timeline_db"] is False
    assert report["import_report"]["safety"]["write_product_timeline_db"] is False
    assert report["safety"]["bot_safe_payment_amounts"] is False


def test_apply_links_existing_tallanto_customer_is_idempotent_and_keeps_amounts_out_of_bot_safe_chunks(
    tmp_path: Path,
) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    existing_customer_id = seed_customer_with_tallanto_link(timeline_db, tmp_path, customer_id="existing-1", tallanto_id="contact-1")
    config = TallantoPaymentsImportConfig(
        source=None,
        timeline_db=timeline_db,
        allowed_root=tmp_path,
        tenant_id="foton",
        apply=True,
    )

    first = run_tallanto_payments_import(config, stdin_text=json.dumps(mcp_snapshot(), ensure_ascii=False))
    second = run_tallanto_payments_import(config, stdin_text=json.dumps(mcp_snapshot(), ensure_ascii=False))

    events = fetch_all_json(timeline_db, "timeline_events")
    opportunities = fetch_all_json(timeline_db, "customer_opportunities")
    chunks = fetch_all_json(timeline_db, "bot_context_chunks")
    payment = next(item for item in events if item["event_type"] == "tallanto_payment")
    abonement = next(item for item in events if item["event_type"] == "tallanto_abonement")
    payment_opp = next(item for item in opportunities if item["source_id"] == "payment:payment-1")
    abonement_opp = next(item for item in opportunities if item["source_id"] == "abonement:abonement-1")

    assert first["validation_ok"] is True
    assert second["validation_ok"] is True
    assert first["links"]["unique_existing_tallanto_matches"] == 1
    assert count_rows(timeline_db, "customer_identities") == 1
    assert count_rows(timeline_db, "timeline_events") == 2
    assert count_rows(timeline_db, "ingestion_runs") == 1
    assert payment["customer_id"] == existing_customer_id
    assert abonement["customer_id"] == existing_customer_id
    assert payment["record"]["amount"] == 12163
    assert payment_opp["product_context"]["amount"] == 12163
    assert abonement["record"]["visits_left"] == 3
    assert abonement_opp["product_context"]["visits_left"] == 3
    assert chunks == []
    assert bot_safe_amount_leaks(timeline_db) == 0
    assert second["import_report"]["write_status_counts"]["duplicate"] >= 4
    assert "safe short note" not in db_dump(timeline_db)
    assert "contact_notice" not in db_dump(timeline_db)
    assert "internal_notice" not in db_dump(timeline_db)


def test_ambiguous_tallanto_contact_id_creates_conflict_without_first_match_merge(tmp_path: Path) -> None:
    timeline_db = tmp_path / "customer_timeline.sqlite"
    first_id = seed_customer_with_tallanto_link(timeline_db, tmp_path, customer_id="existing-1", tallanto_id="contact-1")
    second_id = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="existing-2",
        tallanto_id="contact-1",
        source_ref="seed-2",
    )

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(
            source=None,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        ),
        stdin_text=json.dumps({"most_finances": mcp_response("most_finances", [payment_row()])}, ensure_ascii=False),
    )

    event = fetch_one_json(timeline_db, "timeline_events")
    links = fetch_all_json(timeline_db, "identity_links")
    conflicts = fetch_all_json(timeline_db, "timeline_conflicts")
    ambiguous_link = next(
        item
        for item in links
        if item["source_system"] == "tallanto_crm_call" and item["link_type"] == "tallanto_student_id"
    )

    assert report["validation_ok"] is True
    assert report["links"]["ambiguous_tallanto_matches"] == 1
    assert event["match_status"] == "ambiguous"
    assert event["customer_id"] not in {first_id, second_id}
    assert ambiguous_link["match_class"] == "ambiguous"
    assert conflicts
    assert any(item["conflict_type"] == "tallanto_identity_ambiguous" for item in conflicts)


def test_cli_stdin_defaults_to_dry_run_and_does_not_create_db(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    timeline_db = tmp_path / "customer_timeline.sqlite"
    monkeypatch.setattr("sys.stdin", _Stdin(json.dumps(mcp_snapshot(), ensure_ascii=False)))

    rc = main(
        [
            "--source",
            "-",
            "--timeline-db",
            str(timeline_db),
            "--allowed-root",
            str(tmp_path),
            "--tenant-id",
            "foton",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert timeline_db.exists() is False
    assert report["mode"] == "dry_run_preview"
    assert report["summary"]["payment_events"] == 1
    assert report["summary"]["write_applied"] is False


def test_importer_safety_contract_and_no_network_or_subprocess(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Tallanto B2 importer must not use subprocess or network APIs")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)
    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(
            source=None,
            timeline_db=tmp_path / "customer_timeline.sqlite",
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=False,
        ),
        stdin_text=json.dumps(mcp_snapshot(), ensure_ascii=False),
    )
    safety = report["safety"]

    assert_customer_timeline_safety_contract(timeline_ingestion_safety_contract())
    assert report["validation_ok"] is True
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["send_messenger"] is False
    assert safety["live_send"] is False
    assert safety["run_asr"] is False
    assert safety["run_ra"] is False
    assert safety["network_calls"] is False
    assert safety["subprocess_calls"] is False
    assert safety["write_product_timeline_db"] is False


def mcp_snapshot() -> dict[str, object]:
    return {
        "most_finances": mcp_response("most_finances", [payment_row()]),
        "most_abonements": mcp_response("most_abonements", [abonement_row()]),
        "most_class": mcp_response("most_class", [class_row()]),
    }


def mcp_response(module: str, records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "module": module,
                            "count": len(records),
                            "limit": len(records),
                            "records": records,
                        },
                        ensure_ascii=False,
                    ),
                }
            ]
        },
    }


def payment_row() -> dict[str, object]:
    return {
        "id": "payment-1",
        "contact_id": "contact-1",
        "cost": 12163,
        "date_payment": "2026-06-01",
        "direction": "in",
        "direction_translated": "Поступление на баланс",
        "type": "sbp",
        "type_translated": "СБП",
        "most_abonements_id": "abonement-1",
        "most_class_id": "class-1",
        "name": "Оплата за абонемент",
        "description": "safe short note",
        "provider_raw_payload": {"secret": "must_not_be_stored"},
    }


def abonement_row() -> dict[str, object]:
    return {
        "id": "abonement-1",
        "contact_id": "contact-1",
        "name": "Физика",
        "cost": 12163,
        "discount": 1000,
        "num_visit": 12,
        "num_visit_left": "3",
        "start_date": "2026-06-01",
        "finish_date": "2026-09-01",
        "type_translated": "Стандартный",
        "filial": {"mfti": "МФТИ"},
        "contact_notice": "must_not_be_stored",
        "internal_notice": "must_not_be_stored",
    }


def class_row() -> dict[str, object]:
    return {
        "id": "class-1",
        "name": "Физика ЕГЭ",
        "cource_name": "Физика 2026",
        "subject_name": "Физика",
        "cost": 1500,
        "date_start": "2026-06-01 10:00:00",
        "date_finish": "2026-06-01 12:00:00",
    }


def seed_customer_with_tallanto_link(
    db_path: Path,
    allowed_root: Path,
    *,
    customer_id: str,
    tallanto_id: str,
    source_ref: str = "seed",
) -> str:
    customer = CustomerIdentity(
        tenant_id="foton",
        customer_id=customer_id,
        identity_status=IdentityStatus.STRONG,
        display_name=customer_id,
        source_ref=source_ref,
        first_seen_at=NOW,
        last_seen_at=NOW,
        touch_count=1,
        created_at=NOW,
        updated_at=NOW,
    )
    link = IdentityLink(
        tenant_id="foton",
        customer_id=customer.customer_id,
        link_type="tallanto_student_id",
        link_value=tallanto_id,
        source_system="tallanto_snapshot",
        source_ref=source_ref,
        match_class="strong_unique",
        confidence=1.0,
        first_seen_at=NOW,
        last_seen_at=NOW,
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=allowed_root)
    try:
        store.upsert_customer(customer)
        store.upsert_identity_link(link)
    finally:
        store.close()
    return customer.customer_id


def fetch_all_json(db_path: Path, table: str) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as con:
        return [json.loads(row[0]) for row in con.execute(f"SELECT record_json FROM {table} ORDER BY record_json")]


def fetch_one_json(db_path: Path, table: str) -> dict[str, object]:
    rows = fetch_all_json(db_path, table)
    assert len(rows) == 1
    return rows[0]


def count_rows(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def db_dump(db_path: Path) -> str:
    with sqlite3.connect(db_path) as con:
        return "\n".join(
            row[0]
            for table in ("timeline_events", "customer_opportunities", "bot_context_chunks")
            for row in con.execute(f"SELECT record_json FROM {table}")
        )


def bot_safe_amount_leaks(db_path: Path) -> int:
    with sqlite3.connect(db_path) as con:
        return int(
            con.execute(
                """
                SELECT COUNT(*)
                FROM bot_context_chunks
                WHERE allowed_for_bot = 1
                  AND (
                    record_json LIKE '%"amount"%'
                    OR record_json LIKE '%"cost"%'
                    OR record_json LIKE '%"visits_left"%'
                  )
                """
            ).fetchone()[0]
        )


class _Stdin:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text
