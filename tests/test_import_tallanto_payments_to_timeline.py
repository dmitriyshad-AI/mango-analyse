from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    IdentityLink,
    IdentityStatus,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.safety import assert_customer_timeline_safety_contract
from mango_mvp.customer_timeline.ingestion import timeline_ingestion_safety_contract
from mango_mvp.customer_timeline.stage5_money_ingest import refresh_customer_purchases_v1
from scripts.import_tallanto_payments_to_timeline import (
    TallantoPaymentsImportConfig,
    fetch_tallanto_module_strict,
    main,
    run_tallanto_money_api_increment,
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
    timeline_db = staging_timeline_db(tmp_path)
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


def test_payment_without_contact_uses_unambiguous_abonement_contact(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    existing_customer_id = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="existing-1",
        tallanto_id="contact-1",
    )
    payment = {**payment_row()}
    payment.pop("contact_id")
    snapshot = {
        "most_finances": [payment],
        "most_abonements": [abonement_row()],
    }

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(
            source=None,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        ),
        stdin_text=json.dumps(snapshot, ensure_ascii=False),
    )

    payment_event = next(
        item for item in fetch_all_json(timeline_db, "timeline_events") if item["event_type"] == "tallanto_payment"
    )
    assert report["validation_ok"] is True
    assert payment_event["customer_id"] == existing_customer_id
    assert payment_event["record"]["contact_id"] == "contact-1"
    assert payment_event["record"]["contact_id_source"] == "abonement"


def test_conflicting_direct_and_abonement_contacts_do_not_pick_first_customer(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    direct_customer = seed_customer_with_tallanto_link(
        timeline_db, tmp_path, customer_id="direct", tallanto_id="contact-direct"
    )
    abonement_customer = seed_customer_with_tallanto_link(
        timeline_db, tmp_path, customer_id="abonement", tallanto_id="contact-abonement", source_ref="seed-2"
    )
    payment = {**payment_row(), "contact_id": "contact-direct"}
    abonement = {**abonement_row(), "contact_id": "contact-abonement"}

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(
            source=None,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        ),
        stdin_text=json.dumps({"most_finances": [payment], "most_abonements": [abonement]}),
    )

    payment_event = next(
        item for item in fetch_all_json(timeline_db, "timeline_events") if item["event_type"] == "tallanto_payment"
    )
    assert report["validation_ok"] is True
    assert payment_event["match_status"] == "ambiguous"
    assert payment_event["customer_id"] not in {direct_customer, abonement_customer}
    assert payment_event["record"]["contact_id_conflict"] is True
    conflict = fetch_one_json(timeline_db, "timeline_conflicts")
    assert conflict["metadata"]["alternate_contact_id"] == "contact-abonement"


def test_conflicting_duplicate_abonement_rows_fail_instead_of_picking_input_order(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    first = {**abonement_row(), "contact_id": "contact-first"}
    second = {**abonement_row(), "contact_id": "contact-second"}

    with pytest.raises(ValueError, match="conflicting duplicate id"):
        run_tallanto_payments_import(
            TallantoPaymentsImportConfig(
                source=None,
                timeline_db=timeline_db,
                allowed_root=tmp_path,
                tenant_id="foton",
                apply=True,
            ),
            stdin_text=json.dumps({"most_abonements": [first, second]}),
        )


def test_payment_with_missing_abonement_owner_is_reported_as_unresolved(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    payment = {**payment_row(), "most_abonements_id": "missing-abonement"}
    payment.pop("contact_id")

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(
            source=None,
            timeline_db=timeline_db,
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        ),
        stdin_text=json.dumps({"most_finances": [payment]}),
    )

    event = fetch_one_json(timeline_db, "timeline_events")
    conflict = fetch_one_json(timeline_db, "timeline_conflicts")
    refresh_customer_purchases_v1(timeline_db, allowed_root=tmp_path, tenant_id="foton")
    assert report["stats"]["unresolved_payment_owners"] == 1
    assert event["match_status"] == "ambiguous"
    assert event["record"]["contact_id_source"] == "unresolved_abonement"
    assert conflict["conflict_type"] == "tallanto_payment_owner_unresolved"
    assert "tallanto_abonement_id:missing-abonement" in conflict["entity_refs"]
    assert count_rows(timeline_db, "customer_purchases_v1") == 0


def test_payment_without_any_identity_relation_is_reported_as_unresolved(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    payment = payment_row()
    payment.pop("contact_id")
    payment.pop("most_abonements_id")

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(None, timeline_db, tmp_path, "foton", apply=True),
        stdin_text=json.dumps({"most_finances": [payment]}),
    )

    event = fetch_one_json(timeline_db, "timeline_events")
    conflict = fetch_one_json(timeline_db, "timeline_conflicts")
    assert report["stats"]["unresolved_payment_owners"] == 1
    assert event["match_status"] == "ambiguous"
    assert event["record"]["contact_id_source"] == "unresolved_owner"
    assert conflict["conflict_type"] == "tallanto_payment_owner_unresolved"


def test_payment_arriving_before_card_relinks_after_identity_appears(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    payment = {**payment_row(), "most_abonements_id": "late-abonement"}
    payment.pop("contact_id")
    config = TallantoPaymentsImportConfig(
        source=None,
        timeline_db=timeline_db,
        allowed_root=tmp_path,
        tenant_id="foton",
        apply=True,
    )
    run_tallanto_payments_import(config, stdin_text=json.dumps({"most_finances": [payment]}))
    actual_customer = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="late-card",
        tallanto_id="contact-late",
        source_ref="late-card",
    )
    resolved_payload = {
        "most_finances": [payment],
        "most_abonements": [{**abonement_row(), "id": "late-abonement", "contact_id": "contact-late"}],
    }

    report = run_tallanto_payments_import(config, stdin_text=json.dumps(resolved_payload))
    refresh_customer_purchases_v1(timeline_db, allowed_root=tmp_path, tenant_id="foton")

    payment_event = next(
        item for item in fetch_all_json(timeline_db, "timeline_events") if item["event_type"] == "tallanto_payment"
    )
    open_unresolved = [
        item
        for item in fetch_all_json(timeline_db, "timeline_conflicts")
        if item["conflict_type"] == "tallanto_payment_owner_unresolved" and item["status"] == "open"
    ]
    assert report["stats"]["unresolved_payment_owners"] == 0
    assert report["links"]["resolved_payment_owner_conflicts"] == 1
    assert payment_event["customer_id"] == actual_customer
    assert payment_event["match_status"] == "strong_unique"
    assert open_unresolved == []
    assert count_rows(timeline_db, "customer_purchases_v1") == 1


def test_ambiguous_tallanto_contact_id_creates_conflict_without_first_match_merge(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    first_id = seed_customer_with_tallanto_link(timeline_db, tmp_path, customer_id="existing-1", tallanto_id="contact-1")
    second_id = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="existing-2",
        tallanto_id="contact-1",
        source_ref="seed-2",
    )

    config = TallantoPaymentsImportConfig(
        source=None,
        timeline_db=timeline_db,
        allowed_root=tmp_path,
        tenant_id="foton",
        apply=True,
    )
    payload = json.dumps({"most_finances": mcp_response("most_finances", [payment_row()])}, ensure_ascii=False)
    report = run_tallanto_payments_import(config, stdin_text=payload)
    repeat = run_tallanto_payments_import(config, stdin_text=payload)

    event = fetch_one_json(timeline_db, "timeline_events")
    links = fetch_all_json(timeline_db, "identity_links")
    conflicts = fetch_all_json(timeline_db, "timeline_conflicts")
    ambiguous_link = next(
        item
        for item in links
        if item["source_system"] == "tallanto_crm_call" and item["link_type"] == "tallanto_student_id"
    )

    assert report["validation_ok"] is True
    assert repeat["validation_ok"] is True
    assert report["links"]["ambiguous_tallanto_matches"] == 1
    assert repeat["links"]["ambiguous_tallanto_matches"] == 1
    assert repeat["import_report"]["write_status_counts"].get("created", 0) == 0
    assert event["match_status"] == "ambiguous"
    assert event["customer_id"] not in {first_id, second_id}
    assert ambiguous_link["match_class"] == "ambiguous"
    assert conflicts
    assert any(item["conflict_type"] == "tallanto_identity_ambiguous" for item in conflicts)
    assert len(conflicts) == 1


def test_apply_refuses_non_staging_and_prod_paths(tmp_path: Path) -> None:
    non_staging = tmp_path / "customer_timeline.sqlite"
    prod_path = tmp_path / "customer_timeline_prod_20260621" / "customer_timeline.sqlite"
    prod_path.parent.mkdir(parents=True)

    with pytest.raises(ValueError, match=".codex_local/staging"):
        TallantoPaymentsImportConfig(
            source=None,
            timeline_db=non_staging,
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )
    with pytest.raises(ValueError, match="prod timeline"):
        TallantoPaymentsImportConfig(
            source=None,
            timeline_db=prod_path,
            allowed_root=tmp_path,
            tenant_id="foton",
            apply=True,
        )


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


def test_tallanto_money_api_full_rescan_imports_sanitized_rows_idempotently(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="existing-1",
        tallanto_id="contact-1",
    )
    client = _TallantoMoneyClient()
    config = TallantoPaymentsImportConfig(
        source=None,
        timeline_db=timeline_db,
        allowed_root=tmp_path,
        tenant_id="foton",
        apply=True,
        source_label="tallanto_api:get_entry_list",
    )

    first = run_tallanto_money_api_increment(config, env_file=tmp_path / "unused.env", client=client)
    second = run_tallanto_money_api_increment(config, env_file=tmp_path / "unused.env", client=client)

    assert first["validation_ok"] is True
    assert first["api"]["modules"]["most_finances"] == {"pages": 2, "records": 1}
    assert first["api"]["modules"]["most_abonements"] == {"pages": 2, "records": 1}
    assert first["api"]["raw_payload_persisted"] is False
    assert first["safety"]["network_calls"] is True
    assert first["safety"]["write_tallanto"] is False
    assert second["import_report"]["write_status_counts"]["duplicate"] >= 4
    assert "must_not_be_stored" not in db_dump(timeline_db)


def test_tallanto_money_api_rejects_duplicate_ids_across_pages() -> None:
    class DuplicateClient:
        def get_entry_list(self, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs["offset"] == 0:
                return {"entry_list": [{"id": "same"}], "next_offset": 1, "total_count": 2}
            return {"entry_list": [{"id": "same"}], "next_offset": None, "total_count": 2}

    with pytest.raises(ValueError, match="duplicate id"):
        fetch_tallanto_module_strict(DuplicateClient(), module="most_finances")


def test_tallanto_money_api_requires_total_count() -> None:
    class Client:
        def get_entry_list(self, **_kwargs):  # type: ignore[no-untyped-def]
            return {"entry_list": [{"id": "payment-1"}], "next_offset": None}

    with pytest.raises(ValueError, match="misses total_count"):
        fetch_tallanto_module_strict(Client(), module="most_finances")


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


class _TallantoMoneyClient:
    def get_entry_list(self, *, module: str, offset: int, **_kwargs):  # type: ignore[no-untyped-def]
        if offset:
            return {"entry_list": [], "next_offset": None, "total_count": 1}
        if module == "most_finances":
            row = {**payment_row(), "private_note": "must_not_be_stored"}
        elif module == "most_abonements":
            row = {**abonement_row(), "private_note": "must_not_be_stored"}
        else:
            raise AssertionError(module)
        return {"entry_list": [row], "next_offset": 1, "total_count": 1}


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


def staging_timeline_db(tmp_path: Path) -> Path:
    path = tmp_path / ".codex_local" / "staging" / "customer_timeline.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
