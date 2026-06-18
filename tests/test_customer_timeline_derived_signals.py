from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mango_mvp.customer_timeline import (
    DUPLICATE_CONTACT_SIGNAL,
    HOT_LEAD_SILENT_SIGNAL,
    PAID_NO_ACCESS_SIGNAL,
    DerivedSignalInputs,
    derive_active_signals,
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
