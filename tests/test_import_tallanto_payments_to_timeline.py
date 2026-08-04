from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    DerivedSignal,
    IdentityLink,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
)
from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiError
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.safety import assert_customer_timeline_safety_contract
from mango_mvp.customer_timeline.ingestion import TALLANTO_TIMEZONE, timeline_ingestion_safety_contract
from mango_mvp.customer_timeline.stage5_money_ingest import refresh_customer_purchases_v1
from scripts.import_tallanto_payments_to_timeline import (
    TallantoPaymentsImportConfig,
    build_tallanto_records,
    fetch_tallanto_module_ids_strict,
    fetch_tallanto_module_strict,
    _latest_money_sync,
    load_existing_money_class_context,
    load_tallanto_customer_lookup,
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
    assert report["import_report"]["normalized_counts"]["opportunities"] == 0
    assert report["import_report"]["normalized_counts"]["customers"] == 0
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
    assert payment["subject"] == "Физика ЕГЭ"
    assert payment_opp["product_context"]["amount"] == 12163
    assert abonement["record"]["visits_left"] == 3
    assert abonement["subject"] == "Физика ЕГЭ"
    assert abonement_opp["product_context"]["visits_left"] == 3
    assert chunks == []
    assert bot_safe_amount_leaks(timeline_db) == 0
    assert second["import_report"]["write_status_counts"]["duplicate"] >= 4
    assert "safe short note" not in db_dump(timeline_db)
    assert "contact_notice" not in db_dump(timeline_db)
    assert "internal_notice" not in db_dump(timeline_db)


def test_first_payment_pass_sees_manual_identity_still_in_wal(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    writer = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    try:
        customer = CustomerIdentity(
            tenant_id="foton",
            customer_id="manual-owner",
            identity_status=IdentityStatus.STRONG,
            display_name="manual-owner",
            source_ref="manual-card",
            first_seen_at=NOW,
            last_seen_at=NOW,
            touch_count=1,
            created_at=NOW,
            updated_at=NOW,
        )
        writer.upsert_customer(customer)
        writer.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=customer.customer_id,
                link_type="tallanto_student_id",
                link_value="contact-1",
                source_system="tallanto_snapshot",
                source_ref="manual-card",
                match_class="manual",
                confidence=1.0,
                first_seen_at=NOW,
                last_seen_at=NOW,
            )
        )
        assert timeline_db.with_name(timeline_db.name + "-wal").exists()

        lookup = load_tallanto_customer_lookup(
            timeline_db,
            tenant_id="foton",
            contact_ids={"contact-1"},
        )
    finally:
        writer.close()

    assert lookup.unique_customer_ids == {"contact-1": "manual-owner"}
    assert lookup.unique_match_classes["contact-1"].value == "manual"


def test_open_exact_identity_conflict_blocks_strong_payment_owner(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    customer_id = seed_customer_with_tallanto_link(
        timeline_db, tmp_path, customer_id="conflicted-owner", tallanto_id="contact-conflict"
    )
    writer = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    try:
        writer.record_conflict(
            "foton",
            conflict_type="tallanto_identity_conflict",
            entity_refs=(f"customer:{customer_id}",),
            actor="test",
        )
    finally:
        writer.close()

    lookup = load_tallanto_customer_lookup(
        timeline_db, tenant_id="foton", contact_ids={"contact-conflict"}
    )

    assert lookup.unique_customer_ids == {}
    assert lookup.ambiguous_customer_ids == {"contact-conflict": (customer_id,)}


def test_open_family_conflict_blocks_strong_payment_owner(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    customer_id = seed_customer_with_tallanto_link(
        timeline_db, tmp_path, customer_id="family-conflicted-owner", tallanto_id="contact-family-conflict"
    )
    writer = CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path)
    try:
        writer._con.execute(
            "INSERT INTO family_members_v1 "
            "(tenant_id,family_id,customer_id,membership_status,confidence,reason,created_at,updated_at,record_hash,record_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                "foton", "family:conflicted", customer_id, "confident", "high", "test", NOW.isoformat(),
                NOW.isoformat(), "test-hash", '{}',
            ),
        )
        writer.record_conflict(
            "foton",
            conflict_type="family_identity_conflict",
            entity_refs=("family:conflicted",),
            actor="test",
        )
    finally:
        writer.close()

    lookup = load_tallanto_customer_lookup(
        timeline_db, tenant_id="foton", contact_ids={"contact-family-conflict"}
    )

    assert lookup.unique_customer_ids == {}
    assert lookup.ambiguous_customer_ids == {"contact-family-conflict": (customer_id,)}


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
    assert payment_event["customer_id"] is None
    assert count_rows(timeline_db, "customer_identities") == 2
    assert count_rows(timeline_db, "customer_opportunities") == 1
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
    assert event["customer_id"] is None
    assert event["record"]["contact_id_source"] == "unresolved_abonement"
    assert conflict["conflict_type"] == "tallanto_payment_owner_unresolved"
    assert "tallanto_abonement_id:missing-abonement" in conflict["entity_refs"]
    assert count_rows(timeline_db, "customer_purchases_v1") == 0
    assert count_rows(timeline_db, "customer_identities") == 0
    assert count_rows(timeline_db, "customer_opportunities") == 0


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
    assert event["customer_id"] is None
    assert event["record"]["contact_id_source"] == "unresolved_owner"
    assert conflict["conflict_type"] == "tallanto_payment_owner_unresolved"
    assert count_rows(timeline_db, "customer_identities") == 0
    assert count_rows(timeline_db, "customer_opportunities") == 0


@pytest.mark.parametrize("match_class", ["strong_unique", "manual"])
def test_payment_arriving_before_card_relinks_after_identity_appears(
    tmp_path: Path,
    match_class: str,
) -> None:
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
    resolved_payload = {
        "most_abonements": [{**abonement_row(), "id": "late-abonement", "contact_id": "contact-late"}],
    }

    incomplete = run_tallanto_payments_import(config, stdin_text=json.dumps(resolved_payload))
    refresh_customer_purchases_v1(timeline_db, allowed_root=tmp_path, tenant_id="foton")
    assert incomplete["links"]["resolved_payment_owner_conflicts"] == 0
    assert incomplete["stats"]["local_unowned_payment_retries"] == 0
    assert count_rows(timeline_db, "customer_purchases_v1") == 0
    assert count_rows(timeline_db, "customer_identities") == 0

    actual_customer = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="late-card",
        tallanto_id="contact-late",
        source_ref="late-card",
        match_class=match_class,
    )
    report = run_tallanto_payments_import(config, stdin_text=json.dumps(resolved_payload))
    refresh_customer_purchases_v1(timeline_db, allowed_root=tmp_path, tenant_id="foton")
    repeat = run_tallanto_payments_import(config, stdin_text=json.dumps(resolved_payload))
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
    assert report["stats"]["local_unowned_payment_retries"] == 1
    assert report["links"]["resolved_payment_owner_conflicts"] == 1
    assert repeat["links"]["resolved_payment_owner_conflicts"] == 0
    assert payment_event["customer_id"] == actual_customer
    assert payment_event["match_status"] == match_class
    assert open_unresolved == []
    assert count_rows(timeline_db, "customer_purchases_v1") == 1
    assert count_rows(timeline_db, "timeline_events") == 2


def test_unknown_direct_tallanto_contact_stays_unowned_with_explicit_conflict(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(None, timeline_db, tmp_path, "foton", apply=True),
        stdin_text=json.dumps({"most_finances": [payment_row()]}),
    )

    event = fetch_one_json(timeline_db, "timeline_events")
    conflict = fetch_one_json(timeline_db, "timeline_conflicts")
    assert report["stats"]["unmatched_contact_ids"] == 1
    assert event["customer_id"] is None
    assert conflict["conflict_type"] == "tallanto_payment_owner_unresolved"
    assert count_rows(timeline_db, "customer_identities") == 0
    assert count_rows(timeline_db, "customer_opportunities") == 0


def test_new_identity_conflict_detaches_event_and_deletes_stale_opportunity(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    config = TallantoPaymentsImportConfig(None, timeline_db, tmp_path, "foton", apply=True)
    seed_customer_with_tallanto_link(
        timeline_db, tmp_path, customer_id="owner-a", tallanto_id="contact-1", source_ref="owner-a"
    )
    run_tallanto_payments_import(
        config,
        stdin_text=json.dumps({"most_finances": [payment_row()], "most_class": [class_row()]}),
    )
    assert count_rows(timeline_db, "customer_opportunities") == 1
    with sqlite3.connect(timeline_db) as con:
        opportunity_id = str(con.execute("SELECT opportunity_id FROM customer_opportunities").fetchone()[0])
    with CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path) as store:
        store.upsert_signal(DerivedSignal(
            tenant_id="foton", customer_id="owner-a", opportunity_id=opportunity_id,
            signal_type="sales", severity="medium", evidence_text="Тестовый сигнал",
        ))
        store.upsert_bot_context_chunk(BotContextChunk(
            tenant_id="foton", customer_id="owner-a", opportunity_id=opportunity_id,
            chunk_type="manager_only", text="Тестовый контекст", source_system="tallanto_crm_call",
            source_ref="test:stale-opportunity", allowed_for_bot=False, requires_manager_review=True,
        ))
    seed_customer_with_tallanto_link(
        timeline_db, tmp_path, customer_id="owner-b", tallanto_id="contact-1", source_ref="owner-b"
    )

    report = run_tallanto_payments_import(
        config,
        stdin_text=json.dumps({"most_finances": [payment_row()], "most_class": [class_row()]}),
    )

    event = fetch_one_json(timeline_db, "timeline_events")
    assert report["import_report"]["write_status_counts"]["deleted"] == 1
    assert event["customer_id"] is None and event["opportunity_id"] is None
    assert count_rows(timeline_db, "customer_opportunities") == 0
    with sqlite3.connect(timeline_db) as con:
        signal = con.execute("SELECT status,opportunity_id FROM derived_signals").fetchone()
        chunk = con.execute("SELECT superseded_by,opportunity_id FROM bot_context_chunks").fetchone()
    assert tuple(signal) == ("stale", None)
    assert chunk[0] and chunk[1] is None


def test_local_unowned_direct_payment_relinks_after_card_without_network_replay(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    config = TallantoPaymentsImportConfig(None, timeline_db, tmp_path, "foton", apply=True)
    run_tallanto_payments_import(
        config,
        stdin_text=json.dumps({
            "most_finances": [{**payment_row(), "contact_id": "late-direct"}],
            "most_class": [class_row()],
        }),
    )
    actual_customer = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="late-direct-card",
        tallanto_id="late-direct",
        source_ref="late-direct-card",
    )

    report = run_tallanto_payments_import(config, stdin_text="{}")

    event = fetch_one_json(timeline_db, "timeline_events")
    assert report["stats"]["local_unowned_payment_retries"] == 1
    assert report["links"]["resolved_payment_owner_conflicts"] == 1
    assert event["customer_id"] == actual_customer
    assert event["subject"] == class_row()["name"]
    assert event["record"]["class_name"] == class_row()["name"]
    assert count_rows(timeline_db, "customer_opportunities") == 1


def test_local_assigned_weak_payment_is_rechecked_after_identity_improves(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    customer_id = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="real-card",
        tallanto_id="contact-weak",
    )
    with CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=customer_id,
                event_type=TimelineEventType.TALLANTO_PAYMENT,
                event_at=NOW,
                source_system="tallanto_crm_call",
                source_id="most_finances:weak-payment",
                source_ref="tallanto:most_finances:weak-payment",
                direction=TimelineDirection.SYSTEM,
                subject="Физика ЕГЭ",
                match_status="unmatched",
                record={
                    "payment_id": "weak-payment",
                    "contact_id": "contact-weak",
                    "class_id": "class-1",
                    "class_name": "Физика ЕГЭ",
                },
            )
        )
    with sqlite3.connect(timeline_db) as con:
        con.execute(
            "UPDATE timeline_events SET superseded_by='' WHERE source_id='most_finances:weak-payment'"
        )

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(None, timeline_db, tmp_path, "foton", apply=True),
        stdin_text="{}",
    )

    with sqlite3.connect(timeline_db) as con:
        owner, match_status = con.execute(
            "SELECT customer_id,match_status FROM timeline_events WHERE source_id='most_finances:weak-payment'"
        ).fetchone()
    assert report["stats"]["local_weak_payment_retries"] == 1
    assert (owner, match_status) == (customer_id, "strong_unique")


def test_local_legacy_shell_money_rows_relink_without_network_replay(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    phantom = CustomerIdentity(
        tenant_id="foton",
        customer_id="legacy-shell",
        identity_status=IdentityStatus.PARTIAL,
        source_ref="tallanto:contact:contact-1",
        summary={"source_system": "tallanto_crm_call"},
        first_seen_at=NOW,
        last_seen_at=NOW,
        touch_count=2,
        created_at=NOW,
        updated_at=NOW,
    )
    unmatched_link = IdentityLink(
        tenant_id="foton",
        customer_id=phantom.customer_id,
        link_type="tallanto_student_id",
        link_value="contact-1",
        source_system="tallanto_crm_call",
        source_ref="legacy-import",
        match_class="unmatched",
        confidence=0.0,
        first_seen_at=NOW,
        last_seen_at=NOW,
    )
    legacy_events = (
        TimelineEvent(
            tenant_id="foton",
            customer_id=phantom.customer_id,
            event_type=TimelineEventType.TALLANTO_PAYMENT,
            event_at=NOW,
            source_system="tallanto_crm_call",
            source_id="most_finances:legacy-payment",
            source_ref="tallanto:most_finances:legacy-payment",
            direction=TimelineDirection.SYSTEM,
            subject="Физика",
            record={"payment_id": "legacy-payment", "contact_id": "contact-1", "amount": 1000},
        ),
        TimelineEvent(
            tenant_id="foton",
            customer_id=phantom.customer_id,
            event_type=TimelineEventType.TALLANTO_ABONEMENT,
            event_at=NOW,
            source_system="tallanto_crm_call",
            source_id="most_abonements:legacy-abonement",
            source_ref="tallanto:most_abonements:legacy-abonement",
            direction=TimelineDirection.SYSTEM,
            subject="Физика",
            record={"abonement_id": "legacy-abonement", "contact_id": "contact-1", "amount": 1000},
        ),
    )
    with CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path) as store:
        store.upsert_customer(phantom)
        store.upsert_identity_link(unmatched_link)
        for event in legacy_events:
            store.upsert_event(event)
    actual_customer = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="actual-card",
        tallanto_id="contact-1",
        source_ref="actual-card",
    )

    report = run_tallanto_payments_import(
        TallantoPaymentsImportConfig(None, timeline_db, tmp_path, "foton", apply=True),
        stdin_text="{}",
    )

    events = fetch_all_json(timeline_db, "timeline_events")
    assert report["stats"]["local_unowned_payment_retries"] == 1
    assert report["stats"]["local_unowned_abonement_retries"] == 1
    assert {item["customer_id"] for item in events} == {actual_customer}
    assert {item["subject"] for item in events} == {"Физика"}
    assert count_rows(timeline_db, "customer_opportunities") == 2


@pytest.mark.parametrize("match_class", ["strong_unique", "manual"])
def test_ambiguous_tallanto_contact_id_creates_conflict_without_first_match_merge(
    tmp_path: Path,
    match_class: str,
) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    first_id = seed_customer_with_tallanto_link(
        timeline_db, tmp_path, customer_id="existing-1", tallanto_id="contact-1", match_class=match_class
    )
    second_id = seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="existing-2",
        tallanto_id="contact-1",
        source_ref="seed-2",
        match_class=match_class,
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
    assert event["customer_id"] is None
    assert ambiguous_link["match_class"] == "ambiguous"
    assert ambiguous_link["customer_id"] is None
    assert conflicts
    assert any(item["conflict_type"] == "tallanto_identity_ambiguous" for item in conflicts)
    assert len(conflicts) == 1
    assert count_rows(timeline_db, "customer_identities") == 2
    assert count_rows(timeline_db, "customer_opportunities") == 0


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
    assert first["api"]["modules"]["most_finances"] == {"pages": 1, "records": 1, "mode": "full"}
    assert first["api"]["modules"]["most_abonements"] == {"pages": 1, "records": 1, "mode": "full"}
    assert first["api"]["modules"]["most_class"] == {
        "pages": 1, "records": 1, "batches": 1, "mode": "id_batches",
    }
    assert first["api"]["raw_payload_persisted"] is False
    assert first["safety"]["network_calls"] is True
    assert first["safety"]["write_tallanto"] is False
    assert second["import_report"]["write_status_counts"]["duplicate"] >= 4
    money_queries = [item for item in client.queries if item[0] in {"most_finances", "most_abonements"}]
    assert all(query and query.startswith(f"{module}.date_modified <=") for module, query in money_queries[:2])
    assert all(
        query and query.startswith(f"{module}.date_modified >=") and "date_modified <=" in query
        for module, query in money_queries[2:]
    )
    incremental_cutoff = datetime.fromisoformat(money_queries[2][1].split("'", 2)[1]).replace(
        tzinfo=TALLANTO_TIMEZONE
    )
    assert datetime.now(TALLANTO_TIMEZONE) - incremental_cutoff < timedelta(minutes=10)
    assert "must_not_be_stored" not in db_dump(timeline_db)
    with sqlite3.connect(timeline_db) as con:
        assert con.execute(
            "SELECT event_at FROM timeline_events WHERE source_id='most_finances:payment-1'"
        ).fetchone()[0].endswith("+03:00")
        assert con.execute(
            "SELECT subject FROM timeline_events WHERE source_id='most_abonements:abonement-1'"
        ).fetchone()[0] == "Физика ЕГЭ"


def test_incremental_payment_reuses_existing_abonement_owner() -> None:
    row = {**payment_row(), "contact_id": ""}
    records, stats, _ = build_tallanto_records(
        {"most_finances": [row]},
        source_path=None,
        existing_abonement_contacts={"abonement-1": "contact-1"},
    )

    assert stats.unresolved_payment_owners == 0
    assert records[0].payload["contact_id"] == "contact-1"
    assert records[0].payload["_contact_id_source"] == "abonement"


def test_canonical_module_names_are_not_counted_twice_when_repeated_as_aliases() -> None:
    records, stats, class_lookup = build_tallanto_records(
        {
            "most_finances": [{"name": "no-id"}],
            "most_abonements": [{"name": "no-id"}],
            "most_class": [{"name": "no-id"}],
        },
        source_path=None,
    )

    assert records == ()
    assert (stats.payment_rows, stats.abonement_rows, stats.class_rows, stats.skipped) == (1, 1, 1, 2)
    assert class_lookup == {}


def test_existing_attendance_supplies_exact_abonement_class_relation(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    with CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path) as store:
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=None,
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=NOW,
                source_system="tallanto_attendance_api",
                source_id="relation-1",
                source_ref="tallanto:class-contact:relation-1",
                direction=TimelineDirection.SYSTEM,
                subject="Физика ЕГЭ",
                record={"abonement_id": "abonement-1", "most_class_id": "class-1"},
            )
        )
        store.upsert_event(
            TimelineEvent(
                tenant_id="foton",
                customer_id=None,
                event_type=TimelineEventType.TALLANTO_ATTENDANCE,
                event_at=NOW,
                source_system="tallanto_attendance_api",
                source_id="relation-without-subject",
                source_ref="tallanto:class-contact:relation-without-subject",
                direction=TimelineDirection.SYSTEM,
                record={"abonement_id": "abonement-2", "most_class_id": "class-2"},
            )
        )

    classes, abonement_classes = load_existing_money_class_context(timeline_db, "foton")
    records, stats, _ = build_tallanto_records(
        {
            "most_finances": [{"id": "payment-1", "most_abonements_id": "abonement-1"}],
            "most_abonements": [{"id": "abonement-1", "name": "Абонемент по физике"}],
        },
        source_path=None,
        existing_class_lookup=classes,
        existing_abonement_classes=abonement_classes,
    )

    assert classes == {"class-1": {"id": "class-1", "name": "Физика ЕГЭ"}}
    assert abonement_classes == {"abonement-1": "class-1", "abonement-2": "class-2"}
    assert records[0].payload["most_class_id"] == "class-1"
    assert records[1].payload["most_class_id"] == "class-1"
    assert stats.payment_class_relations_resolved == 1
    assert stats.abonement_class_relations_resolved == 1


def test_strict_fetch_retries_only_current_page_after_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class Client:
        calls = 0

        def get_entry_list(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TallantoApiError("rate limited", status_code=429, category="rate_limited")
            return {"entry_list": [{"id": "row-1"}], "total_count": 1, "next_offset": None}

    sleeps: list[float] = []
    monkeypatch.setattr("scripts.import_tallanto_payments_to_timeline.time.sleep", sleeps.append)
    client = Client()

    rows, stats = fetch_tallanto_module_strict(
        client, module="most_abonements", rate_limit_wait_seconds=30.0
    )

    assert [row["id"] for row in rows] == ["row-1"]
    assert stats == {"pages": 1, "records": 1}
    assert client.calls == 2
    assert sleeps == [30.0]


def test_money_api_repairs_legacy_payment_product_relations_once(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    seed_customer_with_tallanto_link(
        timeline_db,
        tmp_path,
        customer_id="existing-1",
        tallanto_id="contact-1",
    )
    config = TallantoPaymentsImportConfig(None, timeline_db, tmp_path, "foton", apply=True)
    run_tallanto_payments_import(
        config,
        stdin_text=json.dumps({
            "most_finances": [{
                "id": "payment-1", "contact_id": "contact-1", "most_abonements_id": "abonement-1",
            }],
            "most_abonements": [{"id": "abonement-1", "contact_id": "contact-1"}],
        }),
    )
    with CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path) as store:
        store.upsert_ingestion_cursor(
            "foton",
            "tallanto_money_api",
            last_cursor_ts=NOW - timedelta(days=1),
            actor="test",
        )
    client = _TallantoMoneyClient()

    report = run_tallanto_money_api_increment(config, env_file=tmp_path / "unused.env", client=client)

    money_queries = [item for item in client.queries if item[0] in {"most_finances", "most_abonements"}]
    assert report["api"]["performance"]["product_backfill"] is True
    assert report["api"]["performance"]["mode"] == "product_backfill"
    assert report["api"]["performance"]["module_modes"] == {
        "most_finances": "full", "most_abonements": "full",
    }
    assert money_queries[0][1].startswith("most_finances.date_modified <=")
    assert money_queries[1][1].startswith("most_abonements.date_modified <=")
    with sqlite3.connect(timeline_db) as con:
        subjects = dict(con.execute("SELECT event_type,subject FROM timeline_events"))
    assert subjects == {"tallanto_payment": "Физика ЕГЭ", "tallanto_abonement": "Физика ЕГЭ"}

    repeat_client = _TallantoMoneyClient()
    repeat = run_tallanto_money_api_increment(config, env_file=tmp_path / "unused.env", client=repeat_client)
    assert repeat["api"]["performance"]["product_backfill"] is False
    assert repeat["api"]["performance"]["module_modes"] == {
        "most_finances": "incremental", "most_abonements": "incremental",
    }


def test_money_api_never_uses_legacy_import_run_as_its_cursor(tmp_path: Path) -> None:
    timeline_db = staging_timeline_db(tmp_path)
    with CustomerTimelineSQLiteStore(timeline_db, allowed_root=tmp_path) as store:
        run = store.start_ingestion_run(
            tenant_id="foton", source_system="tallanto_crm_call", source_ref="legacy",
            run_kind="timeline_import", idempotency_key="legacy", actor="test",
        )
        store.finish_ingestion_run(run.run_id, status="completed", accepted_count=1, actor="test")

    assert _latest_money_sync(timeline_db, "foton") is None


def test_tallanto_class_lookup_batches_ids_and_rejects_foreign_rows() -> None:
    class Client:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def get_entry_list(self, *, query: str, **_kwargs):  # type: ignore[no-untyped-def]
            self.queries.append(query)
            requested = query.split("(", 1)[1].rsplit(")", 1)[0].replace("'", "").split(",")
            return {"entry_list": [{"id": value} for value in requested], "total_count": len(requested)}

    client = Client()
    rows, stats = fetch_tallanto_module_ids_strict(
        client,
        module="most_class",
        ids={f"class-{index}" for index in range(205)},
        select_fields=("id", "name"),
    )

    assert len(rows) == 205
    assert stats == {"pages": 3, "records": 205, "batches": 3}
    assert len(client.queries) == 3

    class ForeignClient:
        def get_entry_list(self, **_kwargs):  # type: ignore[no-untyped-def]
            return {"entry_list": [{"id": "foreign"}], "total_count": 1}

    with pytest.raises(ValueError, match="unrequested id"):
        fetch_tallanto_module_ids_strict(
            ForeignClient(), module="most_class", ids={"wanted"}, select_fields=("id", "name")
        )


def test_tallanto_money_api_rejects_duplicate_ids_across_pages() -> None:
    class DuplicateClient:
        def get_entry_list(self, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs["offset"] == 0:
                return {"entry_list": [{"id": "same"}], "next_offset": 1, "total_count": 2}
            return {"entry_list": [{"id": "same"}], "next_offset": None, "total_count": 2}

    with pytest.raises(ValueError, match="duplicate id"):
        fetch_tallanto_module_strict(DuplicateClient(), module="most_finances")


def test_tallanto_full_pagination_reads_every_page_in_order() -> None:
    class Client:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        def get_entry_list(self, *, offset: int, **_kwargs):  # type: ignore[no-untyped-def]
            self.offsets.append(offset)
            rows = [{"id": str(index)} for index in range(offset, min(offset + 50, 200))]
            return {
                "entry_list": rows,
                "next_offset": offset + 50 if offset + 50 < 200 else None,
                "total_count": 200,
            }

    client = Client()
    rows, stats = fetch_tallanto_module_strict(client, module="most_finances")

    assert len(rows) == 200
    assert stats == {"pages": 4, "records": 200}
    assert client.offsets == [0, 50, 100, 150]


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
    def __init__(self) -> None:
        self.queries: list[tuple[str, object]] = []

    def get_entry_list(self, *, module: str, offset: int, **_kwargs):  # type: ignore[no-untyped-def]
        if offset == 0:
            self.queries.append((module, _kwargs.get("query")))
        if offset:
            return {"entry_list": [], "next_offset": None, "total_count": 1}
        if module == "most_finances":
            row = {**payment_row(), "private_note": "must_not_be_stored"}
        elif module == "most_abonements":
            row = {**abonement_row(), "private_note": "must_not_be_stored"}
        elif module == "most_class":
            row = class_row()
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
    match_class: str = "strong_unique",
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
        match_class=match_class,
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
