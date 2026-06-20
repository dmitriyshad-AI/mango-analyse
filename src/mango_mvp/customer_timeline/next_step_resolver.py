from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


CUSTOMER_TIMELINE_NEXT_STEP_SCHEMA_VERSION = "customer_timeline_next_step_resolution_v1"

NEXT_STEP_STATUS_ACTIVE = "active"
NEXT_STEP_STATUS_CLOSED = "closed"
NEXT_STEP_STATUS_EMPTY = "empty"
NEXT_STEP_STATUS_NEEDS_MANAGER_REVIEW = "needs_manager_review"

MANAGER_REVIEW_ACTION = "Уточнить у менеджера"

DOCUMENT_STEP_MARKERS = (
    "документ",
    "материал",
    "презентац",
    "договор",
    "файл",
    "форму",
    "програм",
    "почт",
    "отправ",
    "высл",
    "направ",
)
PAYMENT_STEP_MARKERS = ("оплат", "счет", "счёт", "чек", "квитанц", "платеж", "платёж")
CALLBACK_STEP_MARKERS = ("перезвон", "созвон", "связ", "набрать", "позвон")

SENT_MARKERS = ("отправлен", "отправили", "отправил", "выслан", "выслали", "направлен", "направили", "прикреп", "во влож", "приклады")
DONE_MARKERS = ("сделан", "закрыт", "выполн", "прош", "поступ", "оплачен", "получил", "получили")
NEGATION_MARKERS = ("не приш", "не получил", "не получили", "не дош", "ошиб", "отказ")
QUESTION_MARKERS = ("?", "уточн", "непонят", "проверь", "проверить", "сомнен")

NON_CLOSING_EVENT_TYPES = {"system_note"}
NON_CLOSING_MARKERS = (
    "outbound_campaign",
    "campaign",
    "массов",
    "рассыл",
    "service_notification",
    "служеб",
    "автоуведом",
    "system notification",
    "bounce",
    "delivery status",
    "undeliver",
    "недостав",
)


@dataclass(frozen=True)
class NextStepResolution:
    status: str
    action: str
    display_text: str
    confidence: str
    reason_code: str
    source_event_id: str = ""
    source_event_at: str = ""
    source_channel: str = ""
    source_event_type: str = ""
    previous_step: str = ""
    closing_event_id: str = ""
    closing_event_at: str = ""
    closing_channel: str = ""
    ignored_event_ids: tuple[str, ...] = ()

    def to_json_dict(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = CUSTOMER_TIMELINE_NEXT_STEP_SCHEMA_VERSION
        payload["ignored_event_ids"] = list(self.ignored_event_ids)
        return payload


def resolve_customer_next_step(
    events: Sequence[Mapping[str, Any]],
    *,
    readiness: Mapping[str, Any] | None = None,
    conflicts: Sequence[Mapping[str, Any]] = (),
    customer_id: str | None = None,
) -> NextStepResolution:
    scoped_events, skipped_ids = _scope_events(events, customer_id=customer_id)
    customer_ids = {str(event.get("customer_id") or "") for event in scoped_events if str(event.get("customer_id") or "")}
    if customer_id is None and len(customer_ids) > 1:
        return _manager_review(
            "mixed_customer_events",
            "в ленте переданы события разных customer_id",
            ignored_event_ids=tuple(skipped_ids),
        )
    if _has_open_ambiguous_identity(readiness or {}, conflicts):
        return _manager_review(
            "ambiguous_identity_open",
            "открыт конфликт идентичности",
            ignored_event_ids=tuple(skipped_ids),
        )

    relevant: list[Mapping[str, Any]] = []
    ignored = list(skipped_ids)
    for event in _sort_events(scoped_events):
        if _is_non_closing_service_event(event):
            ignored.append(_event_id(event))
            continue
        relevant.append(event)

    if not relevant:
        return NextStepResolution(
            status=NEXT_STEP_STATUS_EMPTY,
            action="",
            display_text="Активный следующий шаг не найден",
            confidence="low",
            reason_code="no_relevant_events",
            ignored_event_ids=tuple(_dedupe(ignored)),
        )

    step_candidates: list[tuple[int, Mapping[str, Any], str, str]] = []
    for index, event in enumerate(relevant):
        action = _extract_next_step(event)
        if action:
            step_candidates.append((index, event, action, _step_kind(action)))

    if not step_candidates:
        return NextStepResolution(
            status=NEXT_STEP_STATUS_EMPTY,
            action="",
            display_text="Активный следующий шаг не найден",
            confidence="low",
            reason_code="no_explicit_next_step",
            source_event_id=_event_id(relevant[-1]),
            source_event_at=str(relevant[-1].get("event_at") or ""),
            source_channel=_source_channel(relevant[-1]),
            source_event_type=str(relevant[-1].get("event_type") or ""),
            ignored_event_ids=tuple(_dedupe(ignored)),
        )

    step_index, step_event, action, kind = step_candidates[-1]
    later_events = relevant[step_index + 1 :]
    contradiction = _first_contradiction(later_events, kind)
    if contradiction is not None:
        return _manager_review(
            "contradictory_later_event",
            "более позднее событие противоречит закрытию шага",
            source_event=contradiction,
            previous_step=action,
            ignored_event_ids=tuple(_dedupe(ignored)),
        )

    closing_event = _latest_closing_event(later_events, kind)
    if closing_event is not None:
        return NextStepResolution(
            status=NEXT_STEP_STATUS_CLOSED,
            action="",
            display_text=f"Шаг закрыт: {_closing_label(kind)} ({_source_suffix(closing_event)})",
            confidence="high",
            reason_code=f"{kind}_closed_by_later_event",
            source_event_id=_event_id(step_event),
            source_event_at=str(step_event.get("event_at") or ""),
            source_channel=_source_channel(step_event),
            source_event_type=str(step_event.get("event_type") or ""),
            previous_step=action,
            closing_event_id=_event_id(closing_event),
            closing_event_at=str(closing_event.get("event_at") or ""),
            closing_channel=_source_channel(closing_event),
            ignored_event_ids=tuple(_dedupe(ignored)),
        )

    return NextStepResolution(
        status=NEXT_STEP_STATUS_ACTIVE,
        action=action,
        display_text=f"{action} ({_source_suffix(step_event)})" if _source_suffix(step_event) else action,
        confidence="high",
        reason_code="latest_relevant_event_has_active_next_step",
        source_event_id=_event_id(step_event),
        source_event_at=str(step_event.get("event_at") or ""),
        source_channel=_source_channel(step_event),
        source_event_type=str(step_event.get("event_type") or ""),
        ignored_event_ids=tuple(_dedupe(ignored)),
    )


def _scope_events(events: Sequence[Mapping[str, Any]], *, customer_id: str | None) -> tuple[list[Mapping[str, Any]], list[str]]:
    if not customer_id:
        return [dict(event) for event in events], []
    scoped: list[Mapping[str, Any]] = []
    skipped: list[str] = []
    for event in events:
        event_customer_id = str(event.get("customer_id") or "")
        if event_customer_id and event_customer_id != customer_id:
            skipped.append(_event_id(event))
            continue
        scoped.append(dict(event))
    return scoped, skipped


def _sort_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(events, key=lambda event: (str(event.get("event_at") or ""), _event_id(event)))


def _has_open_ambiguous_identity(readiness: Mapping[str, Any], conflicts: Sequence[Mapping[str, Any]]) -> bool:
    if conflicts:
        for conflict in conflicts:
            if str(conflict.get("status") or "open").casefold() != "open":
                continue
            conflict_type = str(conflict.get("conflict_type") or "").casefold()
            summary = str(conflict.get("summary") or "").casefold()
            if "ambiguous_identity" in conflict_type or ("ambiguous" in conflict_type and "identity" in conflict_type):
                return True
            if "ambiguous_identity" in summary:
                return True
        return False
    return int(readiness.get("open_conflicts") or 0) > 0


def _manager_review(
    reason_code: str,
    detail: str,
    *,
    source_event: Mapping[str, Any] | None = None,
    previous_step: str = "",
    ignored_event_ids: tuple[str, ...] = (),
) -> NextStepResolution:
    suffix = f": {detail}" if detail else ""
    return NextStepResolution(
        status=NEXT_STEP_STATUS_NEEDS_MANAGER_REVIEW,
        action=MANAGER_REVIEW_ACTION,
        display_text=f"{MANAGER_REVIEW_ACTION}{suffix}",
        confidence="low",
        reason_code=reason_code,
        source_event_id=_event_id(source_event or {}),
        source_event_at=str((source_event or {}).get("event_at") or ""),
        source_channel=_source_channel(source_event or {}),
        source_event_type=str((source_event or {}).get("event_type") or ""),
        previous_step=previous_step,
        ignored_event_ids=ignored_event_ids,
    )


def _extract_next_step(event: Mapping[str, Any]) -> str:
    record = _mapping(event.get("record"))
    call_analysis = _mapping(record.get("call_analysis") or event.get("call_analysis"))
    for value in (
        call_analysis.get("next_step"),
        record.get("next_step"),
        record.get("recommended_action"),
        event.get("next_step"),
        event.get("recommended_action"),
    ):
        text = _compact(value)
        if text:
            return text
    return ""


def _step_kind(action: str) -> str:
    text = action.casefold()
    if _has_any(text, DOCUMENT_STEP_MARKERS):
        return "documents"
    if _has_any(text, PAYMENT_STEP_MARKERS):
        return "payment"
    if _has_any(text, CALLBACK_STEP_MARKERS):
        return "callback"
    return "generic"


def _latest_closing_event(events: Sequence[Mapping[str, Any]], step_kind: str) -> Mapping[str, Any] | None:
    matches = [event for event in events if _event_closes_step(event, step_kind)]
    return matches[-1] if matches else None


def _event_closes_step(event: Mapping[str, Any], step_kind: str) -> bool:
    event_type = str(event.get("event_type") or "").casefold()
    text = _event_text(event)
    if step_kind == "callback":
        return event_type == "mango_call" and not _extract_next_step(event)
    if step_kind == "payment":
        return event_type == "tallanto_payment" or (_has_any(text, PAYMENT_STEP_MARKERS) and _has_any(text, DONE_MARKERS))
    if step_kind == "documents":
        return _has_any(text, DOCUMENT_STEP_MARKERS) and (_has_any(text, SENT_MARKERS) or _has_any(text, DONE_MARKERS))
    return False


def _first_contradiction(events: Sequence[Mapping[str, Any]], step_kind: str) -> Mapping[str, Any] | None:
    for event in events:
        text = _event_text(event)
        if step_kind in {"documents", "payment"} and _has_any(text, NEGATION_MARKERS):
            if step_kind == "documents" and _has_any(text, DOCUMENT_STEP_MARKERS):
                return event
            if step_kind == "payment" and _has_any(text, PAYMENT_STEP_MARKERS):
                return event
        if _has_any(text, QUESTION_MARKERS) and (step_kind == "generic" or _has_step_context(text, step_kind)):
            return event
    return None


def _has_step_context(text: str, step_kind: str) -> bool:
    if step_kind == "documents":
        return _has_any(text, DOCUMENT_STEP_MARKERS)
    if step_kind == "payment":
        return _has_any(text, PAYMENT_STEP_MARKERS)
    if step_kind == "callback":
        return _has_any(text, CALLBACK_STEP_MARKERS)
    return True


def _is_non_closing_service_event(event: Mapping[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "").casefold()
    if event_type in NON_CLOSING_EVENT_TYPES:
        return True
    text = _event_text(event)
    source = " ".join(
        str(event.get(key) or "").casefold()
        for key in ("source_system", "source_id", "source_ref", "subject", "direction")
    )
    record = _mapping(event.get("record"))
    metadata = _mapping(event.get("metadata"))
    flags = " ".join(
        str(value).casefold()
        for value in (
            record.get("event_kind"),
            record.get("message_type"),
            record.get("category"),
            record.get("campaign_type"),
            record.get("outbound_campaign"),
            record.get("is_bounce"),
            record.get("service_notification"),
            metadata.get("event_kind"),
            metadata.get("campaign_type"),
        )
        if value not in (None, "")
    )
    joined = f"{event_type} {source} {flags} {text}"
    return _has_any(joined, NON_CLOSING_MARKERS)


def _event_text(event: Mapping[str, Any]) -> str:
    record = _mapping(event.get("record"))
    metadata = _mapping(event.get("metadata"))
    call_analysis = _mapping(record.get("call_analysis") or event.get("call_analysis"))
    values = [
        event.get("subject"),
        event.get("text_preview"),
        event.get("summary"),
        event.get("stage_before"),
        event.get("stage_after"),
        call_analysis.get("history_summary"),
        call_analysis.get("summary"),
        call_analysis.get("next_step"),
        record.get("text"),
        record.get("body"),
        record.get("summary"),
        record.get("payment_status"),
        record.get("payment_direction"),
        record.get("payment_type"),
        record.get("status"),
        metadata.get("label"),
    ]
    return " ".join(_compact(value) for value in values if _compact(value)).casefold()


def _source_suffix(event: Mapping[str, Any]) -> str:
    date = _format_date_ru(str(event.get("event_at") or ""))
    channel = _source_channel(event)
    if date and channel:
        return f"от {date}, {channel}"
    if date:
        return f"от {date}"
    return channel


def _source_channel(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("event_type") or "").casefold()
    source = str(event.get("source_system") or "").casefold()
    if event_type == "mango_call":
        return "звонок"
    if event_type == "email_message":
        return "почта"
    if event_type in {"telegram_message", "whatsapp_message", "max_message", "web_chat_message"}:
        return "мессенджер"
    if event_type == "tallanto_payment" or source.startswith("tallanto"):
        return "Tallanto"
    if event_type.startswith("amo_") or source.startswith("amo"):
        return "AMO"
    return source or event_type


def _closing_label(step_kind: str) -> str:
    if step_kind == "documents":
        return "документы/материалы отправлены"
    if step_kind == "payment":
        return "оплата/чек подтверждены"
    if step_kind == "callback":
        return "контакт состоялся"
    return "более позднее событие выполнило шаг"


def _format_date_ru(value: str) -> str:
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", value)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{day}.{month}.{year}"


def _event_id(event: Mapping[str, Any]) -> str:
    return str(event.get("event_id") or event.get("source_id") or "")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _compact(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _has_any(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


__all__ = [
    "CUSTOMER_TIMELINE_NEXT_STEP_SCHEMA_VERSION",
    "MANAGER_REVIEW_ACTION",
    "NEXT_STEP_STATUS_ACTIVE",
    "NEXT_STEP_STATUS_CLOSED",
    "NEXT_STEP_STATUS_EMPTY",
    "NEXT_STEP_STATUS_NEEDS_MANAGER_REVIEW",
    "NextStepResolution",
    "resolve_customer_next_step",
]
