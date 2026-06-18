from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.customer_timeline.contracts import DerivedSignal, SignalSeverity, SignalStatus
from mango_mvp.customer_timeline.ids import normalize_key, optional_text, require_text, require_timezone


DERIVED_SIGNAL_RECOMPUTE_SCHEMA_VERSION = "customer_timeline_derived_signals_v1"
PAID_NO_ACCESS_SIGNAL = "paid_no_access"
HOT_LEAD_SILENT_SIGNAL = "hot_lead_silent_7d"
DUPLICATE_CONTACT_SIGNAL = "duplicate_contact"
DEFAULT_HOT_LEAD_SILENCE_DAYS = 7
SIGNAL_TTL_DAYS: Mapping[str, int] = {
    PAID_NO_ACCESS_SIGNAL: 90,
    HOT_LEAD_SILENT_SIGNAL: 30,
    DUPLICATE_CONTACT_SIGNAL: 180,
}

INTEREST_MARKERS = (
    "заявк",
    "интерес",
    "интересует",
    "хочу",
    "подбер",
    "стоим",
    "цен",
    "курс",
    "обуч",
    "занят",
    "запис",
    "пробн",
    "егэ",
    "огэ",
    "летн",
)
PAYMENT_IN_MARKERS = ("in", "поступ", "оплат", "приход", "зачисл")
PAYMENT_OUT_MARKERS = ("out", "refund", "возврат", "отмен", "cancel")
ACTIVE_ABONEMENT_MARKERS = ("active", "актив", "действ", "открыт")
INACTIVE_ABONEMENT_MARKERS = ("closed", "закры", "отмен", "cancel", "expired", "истек")
DUPLICATE_CONFLICT_TYPES = {
    "ambiguous_identity",
    "shared_amo_contact",
    "shared_amo_contact_across_customers",
    "shared_amo_lead",
    "shared_amo_lead_across_customers",
}


@dataclass(frozen=True)
class DerivedSignalInputs:
    tenant_id: str
    customer_id: str
    events: Sequence[Mapping[str, Any]]
    conflicts: Sequence[Mapping[str, Any]] = ()
    as_of: Optional[datetime] = None
    hot_lead_silence_days: int = DEFAULT_HOT_LEAD_SILENCE_DAYS

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", normalize_key(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "customer_id", require_text(self.customer_id, "customer_id"))
        if self.as_of is not None:
            require_timezone(self.as_of, "as_of")
        if self.hot_lead_silence_days <= 0:
            raise ValueError("hot_lead_silence_days must be positive")
        object.__setattr__(self, "events", tuple(dict(item) for item in self.events))
        object.__setattr__(self, "conflicts", tuple(dict(item) for item in self.conflicts))


def derive_active_signals(inputs: DerivedSignalInputs) -> tuple[DerivedSignal, ...]:
    events = tuple(sorted(inputs.events, key=lambda item: (_event_at(item).isoformat(), str(item.get("event_id") or ""))))
    signals: list[DerivedSignal] = []
    signals.extend(_derive_paid_no_access(inputs.tenant_id, inputs.customer_id, events))
    hot_lead = _derive_hot_lead_silent(inputs.tenant_id, inputs.customer_id, events, inputs.as_of, inputs.hot_lead_silence_days)
    if hot_lead is not None:
        signals.append(hot_lead)
    signals.extend(_derive_duplicate_contact(inputs.tenant_id, inputs.customer_id, inputs.conflicts))
    return tuple(signals)


def signal_expires_at(signal_type: str, event_at: datetime) -> datetime:
    require_timezone(event_at, "event_at")
    ttl_days = SIGNAL_TTL_DAYS[normalize_key(signal_type, "signal_type")]
    return event_at + timedelta(days=ttl_days)


def _derive_paid_no_access(
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
) -> tuple[DerivedSignal, ...]:
    access_events = tuple(event for event in events if _is_access_event(event))
    signals: list[DerivedSignal] = []
    for payment in events:
        if not _is_paid_payment(payment):
            continue
        if any(_access_matches_payment(access, payment) for access in access_events):
            continue
        event_id = require_text(payment.get("event_id"), "event_id")
        event_at = _event_at(payment)
        payment_ref = _payment_ref(payment)
        signals.append(
            DerivedSignal(
                tenant_id=tenant_id,
                customer_id=customer_id,
                event_id=event_id,
                source_event_ids=(event_id,),
                signal_type=PAID_NO_ACCESS_SIGNAL,
                severity=SignalSeverity.HIGH,
                confidence=0.93,
                evidence_text=f"Оплата Tallanto {payment_ref} есть, активный доступ не найден.",
                recommended_action="Проверить оплату в Tallanto и выдать клиенту доступ или объяснить задержку.",
                requires_manager_review=True,
                status=SignalStatus.ACTIVE,
                expires_at=signal_expires_at(PAID_NO_ACCESS_SIGNAL, event_at),
                metadata={
                    "source": "deterministic_tallanto_events",
                    "payment_ref": payment_ref,
                    "access_predicate": "active_abonement_visits_left_or_matching_most_class",
                },
                created_at=event_at,
            )
        )
    return tuple(signals)


def _derive_hot_lead_silent(
    tenant_id: str,
    customer_id: str,
    events: Sequence[Mapping[str, Any]],
    as_of: Optional[datetime],
    silence_days: int,
) -> Optional[DerivedSignal]:
    if as_of is None:
        return None
    interest_events = tuple(event for event in events if _is_interest_event(event))
    if not interest_events:
        return None
    latest_interest = max(interest_events, key=lambda item: (_event_at(item), str(item.get("event_id") or "")))
    interest_at = _event_at(latest_interest)
    if as_of < interest_at + timedelta(days=silence_days):
        return None
    if any(_is_touch_event(event) and _event_at(event) > interest_at for event in events):
        return None
    event_id = require_text(latest_interest.get("event_id"), "event_id")
    return DerivedSignal(
        tenant_id=tenant_id,
        customer_id=customer_id,
        event_id=event_id,
        source_event_ids=(event_id,),
        signal_type=HOT_LEAD_SILENT_SIGNAL,
        severity=SignalSeverity.MEDIUM,
        confidence=0.82,
        evidence_text=f"Горячий интерес был {interest_at.date().isoformat()}, касаний нет {silence_days}+ дней.",
        recommended_action="Связаться с клиентом и уточнить, актуален ли интерес.",
        requires_manager_review=False,
        status=SignalStatus.ACTIVE,
        expires_at=signal_expires_at(HOT_LEAD_SILENT_SIGNAL, interest_at),
        metadata={
            "source": "deterministic_timeline_events",
            "silence_days": silence_days,
            "as_of": as_of.isoformat(),
        },
        created_at=interest_at,
    )


def _derive_duplicate_contact(
    tenant_id: str,
    customer_id: str,
    conflicts: Sequence[Mapping[str, Any]],
) -> tuple[DerivedSignal, ...]:
    signals: list[DerivedSignal] = []
    for conflict in sorted(conflicts, key=lambda item: str(item.get("conflict_id") or "")):
        conflict_type = normalize_key(conflict.get("conflict_type"), "conflict_type")
        status = normalize_key(conflict.get("status") or "open", "conflict_status")
        if status not in {"open", "active"}:
            continue
        if not _is_duplicate_contact_conflict(conflict_type):
            continue
        if not _conflict_mentions_customer(conflict, customer_id):
            continue
        conflict_id = require_text(conflict.get("conflict_id"), "conflict_id")
        created_at = _parse_datetime(conflict.get("created_at"), "created_at")
        signals.append(
            DerivedSignal(
                tenant_id=tenant_id,
                customer_id=customer_id,
                signal_type=DUPLICATE_CONTACT_SIGNAL,
                severity=SignalSeverity.MEDIUM,
                confidence=0.9,
                evidence_text=f"Открытый конфликт дубля контакта: {conflict_id}.",
                recommended_action="Проверить дубль контакта/сделки и выбрать корректную карточку перед ответом клиенту.",
                requires_manager_review=True,
                status=SignalStatus.ACTIVE,
                expires_at=signal_expires_at(DUPLICATE_CONTACT_SIGNAL, created_at),
                metadata={
                    "source": "timeline_conflicts",
                    "conflict_id": conflict_id,
                    "conflict_type": conflict_type,
                },
                created_at=created_at,
            )
        )
    return tuple(signals)


def _is_paid_payment(event: Mapping[str, Any]) -> bool:
    if event.get("event_type") != "tallanto_payment":
        return False
    record = _record(event)
    amount = _number(record.get("amount") or record.get("cost") or record.get("payment_summa"))
    if amount is None or amount <= 0:
        return False
    status_text = _joined_lower(record.get("payment_direction"), record.get("payment_status"), record.get("payment_type"), event.get("summary"))
    return any(marker in status_text for marker in PAYMENT_IN_MARKERS) and not any(
        marker in status_text for marker in PAYMENT_OUT_MARKERS
    )


def _is_access_event(event: Mapping[str, Any]) -> bool:
    record = _record(event)
    event_type = str(event.get("event_type") or "")
    if event_type == "tallanto_abonement":
        visits_left = _number(record.get("visits_left") or record.get("num_visit_left"))
        if visits_left is not None:
            return visits_left > 0
        status_text = _joined_lower(record.get("status"), event.get("summary"))
        return any(marker in status_text for marker in ACTIVE_ABONEMENT_MARKERS) and not any(
            marker in status_text for marker in INACTIVE_ABONEMENT_MARKERS
        )
    module = str(record.get("module") or "").strip().lower()
    return event_type == "tallanto_group" or module == "most_class"


def _access_matches_payment(access: Mapping[str, Any], payment: Mapping[str, Any]) -> bool:
    access_record = _record(access)
    payment_record = _record(payment)
    payment_abonement_id = optional_text(payment_record.get("abonement_id") or payment_record.get("most_abonements_id"))
    access_abonement_id = optional_text(access_record.get("abonement_id") or access_record.get("most_abonements_id"))
    if payment_abonement_id and access_abonement_id:
        return payment_abonement_id == access_abonement_id
    payment_class_id = optional_text(payment_record.get("class_id") or payment_record.get("most_class_id"))
    access_class_id = optional_text(access_record.get("class_id") or access_record.get("most_class_id"))
    if payment_class_id and access_class_id:
        return payment_class_id == access_class_id
    return not payment_abonement_id and not payment_class_id


def _is_interest_event(event: Mapping[str, Any]) -> bool:
    if not _is_touch_event(event):
        return False
    text = _joined_lower(event.get("subject"), event.get("text_preview"), event.get("summary"))
    return any(marker in text for marker in INTEREST_MARKERS)


def _is_touch_event(event: Mapping[str, Any]) -> bool:
    return str(event.get("direction") or "").strip().lower() in {"inbound", "outbound"}


def _is_duplicate_contact_conflict(conflict_type: str) -> bool:
    return (
        conflict_type in DUPLICATE_CONFLICT_TYPES
        or "shared_amo_contact" in conflict_type
        or "shared_amo_lead" in conflict_type
        or "ambiguous_identity" in conflict_type
    )


def _conflict_mentions_customer(conflict: Mapping[str, Any], customer_id: str) -> bool:
    refs = tuple(str(item) for item in conflict.get("entity_refs") or ())
    return any(customer_id == ref or customer_id in ref for ref in refs)


def _record(event: Mapping[str, Any]) -> Mapping[str, Any]:
    record = event.get("record")
    return record if isinstance(record, Mapping) else {}


def _event_at(event: Mapping[str, Any]) -> datetime:
    return _parse_datetime(event.get("event_at") or event.get("created_at"), "event_at")


def _parse_datetime(value: Any, field_name: str) -> datetime:
    text = require_text(value, field_name)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    require_timezone(parsed, field_name)
    return parsed


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return None


def _payment_ref(payment: Mapping[str, Any]) -> str:
    record = _record(payment)
    return (
        optional_text(record.get("payment_id"))
        or optional_text(record.get("source_id"))
        or optional_text(payment.get("source_id"))
        or require_text(payment.get("event_id"), "event_id")
    )


def _joined_lower(*values: Any) -> str:
    return " ".join(str(value or "").casefold() for value in values if value is not None)
