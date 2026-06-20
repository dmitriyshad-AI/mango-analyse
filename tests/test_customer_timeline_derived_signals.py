from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.customer_timeline import (
    DUPLICATE_CONTACT_SIGNAL,
    HOT_LEAD_SILENT_SIGNAL,
    PAID_NO_ACCESS_SIGNAL,
    CustomerIdentity,
    CustomerTimelineReadApi,
    CustomerTimelineReadApiConfig,
    CustomerTimelineSQLiteStore,
    DerivedSignalInputs,
    IdentityStatus,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    derive_active_signals,
    recompute_customer_signals,
)
from scripts.derive_customer_timeline_signals import (
    DeriveCustomerTimelineSignalsConfig,
    _list_customer_ids,
    run_derive_customer_timeline_signals,
)


NOW = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
TENANT = "foton"
CUSTOMER = "customer:1"


def test_paid_no_access_signal_requires_payment_without_tallanto_access() -> None:
    payment = tallanto_payment("payment-1", abonement_id="abonement-1", event_at=NOW)

    signals = derive_active_signals(DerivedSignalInputs(TENANT, CUSTOMER, events=(payment,)))

    assert [signal.signal_type for signal in signals] == [PAID_NO_ACCESS_SIGNAL]
    signal = signals[0]
    assert signal.severity.value == "high"
    assert signal.requires_manager_review is True
    assert signal.status.value == "active"
    assert signal.expires_at == NOW + timedelta(days=90)
    assert "Проверить оплату" in (signal.recommended_action or "")


def test_paid_no_access_is_not_created_when_abonement_or_class_access_exists() -> None:
    payment = tallanto_payment("payment-1", abonement_id="abonement-1", class_id="class-1", event_at=NOW)
    active_abonement = tallanto_abonement("abonement-1", visits_left=3, event_at=NOW + timedelta(minutes=1))
    class_access = tallanto_class_access("class-1", event_at=NOW + timedelta(minutes=2))

    with_abonement = derive_active_signals(DerivedSignalInputs(TENANT, CUSTOMER, events=(payment, active_abonement)))
    with_class = derive_active_signals(DerivedSignalInputs(TENANT, CUSTOMER, events=(payment, class_access)))

    assert with_abonement == ()
    assert with_class == ()


def test_hot_lead_silent_7d_is_created_only_without_new_touch() -> None:
    interest = touch_event("call-1", NOW - timedelta(days=8), summary="Клиент интересуется курсом ЕГЭ и ценой")
    later_system_payment = tallanto_payment("payment-1", event_at=NOW - timedelta(days=3))

    signals = derive_active_signals(
        DerivedSignalInputs(TENANT, CUSTOMER, events=(interest, later_system_payment), as_of=NOW)
    )

    assert [signal.signal_type for signal in signals] == [PAID_NO_ACCESS_SIGNAL, HOT_LEAD_SILENT_SIGNAL]
    hot = next(signal for signal in signals if signal.signal_type == HOT_LEAD_SILENT_SIGNAL)
    assert hot.severity.value == "medium"
    assert hot.requires_manager_review is False
    assert hot.expires_at == (NOW - timedelta(days=8)) + timedelta(days=30)
    assert hot.metadata["silence_days"] == 7


def test_hot_lead_silent_7d_is_not_created_before_threshold_or_after_any_new_touch() -> None:
    interest = touch_event("call-1", NOW - timedelta(days=8), summary="Хочу узнать стоимость обучения")
    manager_touch = touch_event("call-2", NOW - timedelta(days=2), direction="outbound", summary="Менеджер перезвонил")
    young_interest = touch_event("call-3", NOW - timedelta(days=3), summary="Интерес к летней программе")

    assert derive_active_signals(DerivedSignalInputs(TENANT, CUSTOMER, events=(interest, manager_touch), as_of=NOW)) == ()
    assert derive_active_signals(DerivedSignalInputs(TENANT, CUSTOMER, events=(young_interest,), as_of=NOW)) == ()


def test_duplicate_contact_signal_is_created_from_open_amo_or_ambiguous_identity_conflict() -> None:
    conflict = {
        "conflict_id": "timeline_conflict:1",
        "tenant_id": TENANT,
        "conflict_type": "shared_amo_contact_across_customers",
        "severity": "medium",
        "status": "open",
        "entity_refs": [CUSTOMER, "customer:2", "amo_contact:42"],
        "created_at": NOW.isoformat(),
    }

    signals = derive_active_signals(DerivedSignalInputs(TENANT, CUSTOMER, events=(), conflicts=(conflict,)))

    assert [signal.signal_type for signal in signals] == [DUPLICATE_CONTACT_SIGNAL]
    signal = signals[0]
    assert signal.severity.value == "medium"
    assert signal.requires_manager_review is True
    assert signal.expires_at == NOW + timedelta(days=180)
    assert signal.metadata["conflict_id"] == "timeline_conflict:1"


def test_duplicate_contact_signal_ignores_resolved_or_unrelated_conflicts() -> None:
    resolved = {
        "conflict_id": "timeline_conflict:1",
        "tenant_id": TENANT,
        "conflict_type": "ambiguous_identity",
        "severity": "medium",
        "status": "resolved",
        "entity_refs": [CUSTOMER, "customer:2"],
        "created_at": NOW.isoformat(),
    }
    unrelated = {
        "conflict_id": "timeline_conflict:2",
        "tenant_id": TENANT,
        "conflict_type": "source_data_gap",
        "severity": "low",
        "status": "open",
        "entity_refs": [CUSTOMER],
        "created_at": NOW.isoformat(),
    }

    assert derive_active_signals(DerivedSignalInputs(TENANT, CUSTOMER, events=(), conflicts=(resolved, unrelated))) == ()


def test_recompute_paid_no_access_is_idempotent_and_resolves_when_access_appears(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    store.upsert_event(event_object(tallanto_payment("payment-1", abonement_id="abonement-1", event_at=NOW)))

    first = recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW + timedelta(days=1), apply=True)
    second = recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW + timedelta(days=1), apply=True)
    active_row = fetch_signal_rows(store.db_path)[0]
    signal_id = active_row["signal_id"]

    assert first.signal_type_counts == {PAID_NO_ACCESS_SIGNAL: 1}
    assert first.write_status_counts == {"created": 1}
    assert second.write_status_counts == {"duplicate": 1}
    assert count_rows(store.db_path, "derived_signals") == 1
    assert active_row["status"] == "active"

    store.upsert_event(event_object(tallanto_abonement("abonement-1", visits_left=2, event_at=NOW + timedelta(hours=1))))
    resolved = recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW + timedelta(days=2), apply=True)
    resolved_row = fetch_signal_rows(store.db_path)[0]

    assert resolved.status_counts == {"resolved": 1}
    assert resolved.write_status_counts == {"updated": 1}
    assert resolved_row["signal_id"] == signal_id
    assert resolved_row["status"] == "resolved"
    store.close()


def test_recompute_hot_lead_resolves_on_new_touch_and_keeps_deterministic_expiry(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    interest_at = NOW - timedelta(days=8)
    store.upsert_event(event_object(touch_event("call-1", interest_at, summary="Клиент интересуется курсом и ценой")))

    active = recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW, apply=True)
    active_row = fetch_signal_rows(store.db_path)[0]

    assert active.status_counts == {"active": 1}
    assert active_row["signal_type"] == HOT_LEAD_SILENT_SIGNAL
    assert active_row["expires_at"] == (interest_at + timedelta(days=30)).isoformat()

    store.upsert_event(
        event_object(touch_event("call-2", NOW - timedelta(days=1), direction="outbound", summary="Менеджер связался"))
    )
    resolved = recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW + timedelta(days=1), apply=True)
    resolved_row = fetch_signal_rows(store.db_path)[0]

    assert resolved.status_counts == {"resolved": 1}
    assert resolved_row["status"] == "resolved"
    assert resolved_row["expires_at"] == (interest_at + timedelta(days=30)).isoformat()
    store.close()


def test_recompute_duplicate_contact_resolves_when_conflict_is_closed(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    refs = (CUSTOMER, "customer:2", "amo_contact:42")
    store.record_conflict(TENANT, conflict_type="ambiguous_identity", entity_refs=refs, status="open")

    active = recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW, apply=True)
    signal_id = fetch_signal_rows(store.db_path)[0]["signal_id"]

    assert active.signal_type_counts == {DUPLICATE_CONTACT_SIGNAL: 1}
    assert active.status_counts == {"active": 1}

    store.record_conflict(TENANT, conflict_type="ambiguous_identity", entity_refs=refs, status="resolved")
    resolved = recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW + timedelta(days=1), apply=True)
    resolved_row = fetch_signal_rows(store.db_path)[0]

    assert resolved.status_counts == {"resolved": 1}
    assert resolved_row["signal_id"] == signal_id
    assert resolved_row["status"] == "resolved"
    store.close()


def test_read_api_hides_resolved_and_stale_signals(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    interest_at = NOW - timedelta(days=40)
    store.upsert_event(event_object(touch_event("call-1", interest_at, summary="Хочу узнать цену курса")))
    recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW, apply=True)
    store.record_conflict(TENANT, conflict_type="shared_amo_lead_across_customers", entity_refs=(CUSTOMER, "customer:2"), status="open")
    recompute_customer_signals(store, TENANT, CUSTOMER, as_of=NOW, apply=True)
    store.close()

    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path)) as api:
        profile = api.customer_profile(TENANT, CUSTOMER, event_limit=10)
        search = api.search(TENANT, "Горячий", customer_id=CUSTOMER, scopes=("signals",))

    signal_types = {item["signal_type"] for item in profile["signals"]}
    rows = fetch_signal_rows(tmp_path / "customer_timeline.sqlite")
    assert {row["status"] for row in rows} == {"active", "stale"}
    assert signal_types == {DUPLICATE_CONTACT_SIGNAL}
    assert search["result"]["items"] == []


def test_derive_signals_cli_dry_run_is_read_only_and_apply_writes_only_timeline_db(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    store.upsert_event(event_object(tallanto_payment("payment-1", event_at=NOW)))
    store.close()
    db_path = tmp_path / "customer_timeline.sqlite"

    dry_run = run_derive_customer_timeline_signals(
        DeriveCustomerTimelineSignalsConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            tenant_id=TENANT,
            customer_id=CUSTOMER,
            as_of=NOW + timedelta(days=1),
        )
    )
    assert dry_run["mode"] == "dry_run"
    assert dry_run["summary"]["signal_type_counts"] == {PAID_NO_ACCESS_SIGNAL: 1}
    assert count_rows(db_path, "derived_signals") == 0

    applied = run_derive_customer_timeline_signals(
        DeriveCustomerTimelineSignalsConfig(
            timeline_db=db_path,
            allowed_root=tmp_path,
            tenant_id=TENANT,
            customer_id=CUSTOMER,
            as_of=NOW + timedelta(days=1),
            apply=True,
        )
    )
    assert applied["mode"] == "apply"
    assert applied["safety"]["write_customer_timeline_sqlite"] is True
    assert applied["safety"]["llm_calls"] is False
    assert applied["safety"]["network_calls"] is False
    assert count_rows(db_path, "derived_signals") == 1


def test_derive_signals_customer_listing_paginates_until_limit(tmp_path: Path) -> None:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path)
    try:
        for index in range(3):
            store.upsert_customer(
                CustomerIdentity(
                    tenant_id=TENANT,
                    customer_id=f"customer:{index}",
                    identity_status=IdentityStatus.STRONG,
                    display_name=f"Тестовый клиент {index}",
                    source_ref=f"test:{index}",
                    first_seen_at=NOW,
                    last_seen_at=NOW + timedelta(minutes=index),
                    touch_count=1,
                    created_at=NOW,
                    updated_at=NOW + timedelta(minutes=index),
                )
            )

        customer_ids = _list_customer_ids(store, TENANT, 2)
        all_customer_ids = _list_customer_ids(store, TENANT, 10)
    finally:
        store.close()

    assert len(customer_ids) == 2
    assert len(all_customer_ids) == 3
    assert len(set(all_customer_ids)) == 3


def tallanto_payment(
    payment_id: str,
    *,
    abonement_id: str | None = None,
    class_id: str | None = None,
    event_at: datetime,
) -> dict[str, object]:
    return {
        "event_id": f"timeline_event:{payment_id}",
        "tenant_id": TENANT,
        "customer_id": CUSTOMER,
        "event_type": "tallanto_payment",
        "event_at": event_at.isoformat(),
        "direction": "system",
        "source_id": f"most_finances:{payment_id}",
        "summary": "Поступление на баланс",
        "record": {
            "payment_id": payment_id,
            "amount": 12163,
            "payment_direction": "Поступление на баланс",
            "payment_status": "paid",
            "abonement_id": abonement_id,
            "class_id": class_id,
        },
    }


def tallanto_abonement(abonement_id: str, *, visits_left: int, event_at: datetime) -> dict[str, object]:
    return {
        "event_id": f"timeline_event:{abonement_id}",
        "tenant_id": TENANT,
        "customer_id": CUSTOMER,
        "event_type": "tallanto_abonement",
        "event_at": event_at.isoformat(),
        "direction": "system",
        "record": {
            "abonement_id": abonement_id,
            "visits_left": visits_left,
        },
    }


def tallanto_class_access(class_id: str, *, event_at: datetime) -> dict[str, object]:
    return {
        "event_id": f"timeline_event:{class_id}",
        "tenant_id": TENANT,
        "customer_id": CUSTOMER,
        "event_type": "tallanto_group",
        "event_at": event_at.isoformat(),
        "direction": "system",
        "record": {
            "module": "most_class",
            "class_id": class_id,
        },
    }


def touch_event(event_id: str, event_at: datetime, *, summary: str, direction: str = "inbound") -> dict[str, object]:
    return {
        "event_id": f"timeline_event:{event_id}",
        "tenant_id": TENANT,
        "customer_id": CUSTOMER,
        "event_type": "mango_call",
        "event_at": event_at.isoformat(),
        "direction": direction,
        "subject": "Диалог с клиентом",
        "summary": summary,
        "text_preview": summary,
        "record": {},
    }


def seeded_store(tmp_path: Path) -> CustomerTimelineSQLiteStore:
    store = CustomerTimelineSQLiteStore(tmp_path / "customer_timeline.sqlite", allowed_root=tmp_path)
    store.upsert_customer(
        CustomerIdentity(
            tenant_id=TENANT,
            customer_id=CUSTOMER,
            identity_status=IdentityStatus.STRONG,
            display_name="Тестовый клиент",
            source_ref="test",
            first_seen_at=NOW,
            last_seen_at=NOW,
            touch_count=1,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    return store


def event_object(payload: dict[str, object]) -> TimelineEvent:
    return TimelineEvent(
        tenant_id=str(payload["tenant_id"]),
        customer_id=str(payload["customer_id"]),
        event_type=TimelineEventType(str(payload["event_type"])),
        event_at=datetime.fromisoformat(str(payload["event_at"])),
        source_system="test_source",
        source_id=str(payload.get("source_id") or payload["event_id"]),
        direction=TimelineDirection(str(payload["direction"])),
        subject=str(payload.get("subject") or ""),
        text_preview=str(payload.get("text_preview") or ""),
        summary=str(payload.get("summary") or ""),
        match_status="strong_unique",
        record=payload.get("record") if isinstance(payload.get("record"), dict) else {},
        created_at=datetime.fromisoformat(str(payload["event_at"])),
    )


def fetch_signal_rows(db_path: Path) -> list[dict[str, object]]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        return [
            {**dict(row), **json.loads(row["record_json"])}
            for row in con.execute("SELECT signal_id, status, expires_at, record_json FROM derived_signals ORDER BY signal_id")
        ]


def count_rows(db_path: Path, table: str) -> int:
    with sqlite3.connect(db_path) as con:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
