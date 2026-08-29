from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.derived_signals import _is_active_deal_at
from mango_mvp.customer_timeline.store import (
    open_family_identity_conflict_customer_ids,
    trusted_family_customer_ids_by_customer,
)
from mango_mvp.insights.sanitizers import has_personal_data_risk
from mango_mvp.customer_timeline.source_policy import is_non_contentful_call_record


CUSTOMER_TIMELINE_NEXT_STEP_SCHEMA_VERSION = "customer_timeline_next_step_resolution_v1"

NEXT_STEP_STATUS_ACTIVE = "active"
NEXT_STEP_STATUS_CLOSED = "closed"
NEXT_STEP_STATUS_EMPTY = "empty"
NEXT_STEP_STATUS_NEEDS_MANAGER_REVIEW = "needs_manager_review"

_DIRECT_CONTACT_OPTOUT_RE = re.compile(
    r"(?:"
    r"\b(?:больше\s+)?(?:мне\s+|нам\s+|со\s+мной\s+)?не\s+"
    r"(?:пишите|звоните|беспокойте|связывайтесь|связываться)\b|"
    r"\b(?:просьба|прошу|пожалуйста)\s+не\s+"
    r"(?:писать|звонить|беспокоить|связываться)\b|"
    r"\bне\s+(?:надо|нужно)\s+(?:(?:больше|мне|нам)\s+){0,3}"
    r"(?:писать|звонить|беспокоить|связываться)"
    r"(?:\s+(?:мне|нам))?\b|"
    r"\bперестаньте\s+(?:мне\s+|нам\s+)?(?:писать|звонить|беспокоить)\b|"
    r"\b(?:удалите|уберите|исключите)\s+(?:меня|мой\s+номер)\s+"
    r"(?:из\s+)?(?:рассылки|базы)\b|"
    r"\bудалите\s+мои\s+данные\b|"
    r"\b(?:я\s+)?не\s+хочу\s+(?:больше\s+)?(?:получать\s+)?"
    r"(?:рассылку|сообщения|звонки|письма)\b|"
    r"\bотпишите\s+меня\b|\bхочу\s+отписаться\b"
    r")",
    re.IGNORECASE,
)

_TEMPORARY_CONTACT_PAUSE_RE = re.compile(
    r"\b(?:пока|сейчас|сегодня|до\s+[а-яё0-9][а-яё0-9./-]*|в\s+течение)\b",
    re.IGNORECASE,
)

_PERMANENT_CONTACT_OPTOUT_RE = re.compile(
    r"\b(?:больше|никогда|навсегда|перестаньте|удалите|уберите|исключите|"
    r"отпишите|отписаться)\b|\bне\s+хочу\s+(?:больше\s+)?(?:получать\s+)?"
    r"(?:рассылку|сообщения|звонки|письма)\b",
    re.IGNORECASE,
)

_REPORTED_CONTACT_OPTOUT_RE = re.compile(
    r"\b(?:менеджер|оператор|сотрудник)\b[^.!?;,]{0,30}"
    r"\b(?:сказал(?:а|и)?|написал(?:а|и)?|просил(?:а|и)?)\b",
    re.IGNORECASE,
)

_DIRECT_MESSAGE_EVENT_TYPES = frozenset({
    "email_message",
    "telegram_message",
    "whatsapp_message",
    "max_message",
    "web_chat_message",
    "wappi_message",
})

_MANAGER_AMO_TERMINAL_STATUS_IDS = frozenset({"142", "143"})
_MANAGER_AMO_ACTIVE_STATUS_TEXTS = frozenset({
    "active", "new", "observed", "open", "в работе", "первичный контакт", "переговоры",
})
_MANAGER_AMO_CLOSED_STATUS_TEXTS = frozenset({
    "closed", "lost", "won", "закрыта", "закрыто", "не реализовано", "успешно реализовано",
})
_MANAGER_ACTION_UNSAFE_CANDIDATE_REASONS = frozenset({
    "identity_conflict_open",
    "task_source_invalid",
    "task_match_not_strong",
    "task_pending_attribution",
    "task_id_missing",
    "task_source_ref_mismatch",
    "duplicate_task_id",
    "task_id_provenance_mismatch",
    "task_entity_type_invalid",
    "task_lead_missing",
    "task_lead_owner_missing",
    "task_lead_owner_foreign",
    "task_lead_owner_ambiguous",
    "task_owner_actor_mismatch",
    "task_action_mismatch",
    "task_due_mismatch",
    "task_event_time_invalid",
    "task_opportunity_missing",
    "task_opportunity_foreign_customer",
    "task_opportunity_not_amo_deal",
    "task_opportunity_source_invalid",
    "task_opportunity_lead_mismatch",
    "task_opportunity_provenance_mismatch",
    "task_opportunity_provenance_source_invalid",
    "durable_opt_out",
})

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

SUMMARY_ACTION_MARKERS = (
    *DOCUMENT_STEP_MARKERS,
    *PAYMENT_STEP_MARKERS,
    *CALLBACK_STEP_MARKERS,
    "whatsapp",
    "ватсап",
    "мессендж",
    "сообщени",
    "письм",
    "email",
    "уточн",
    "провер",
    "исправ",
    "обнов",
    "подготов",
    "переда",
    "продублир",
    "заполн",
    "оформ",
)
SUMMARY_ACTION_VERBS = (
    "отправ",
    "высл",
    "направ",
    "перезвон",
    "позвон",
    "связ",
    "уточн",
    "провер",
    "подготов",
    "продублир",
    "переда",
    "оформ",
    "пообещ",
)
SUMMARY_NO_STEP_MARKERS = (
    "следующий шаг не",
    "шаг не соглас",
    "шаг не определ",
    "дальнейшие действия не",
    "договоренностей нет",
    "договорённостей нет",
    "без договорен",
    "без договорён",
    "ничего не согласовали",
    "не договорились",
)
SUMMARY_NON_CONVERSATION_MARKERS = (
    "значимого диалога",
    "живого разговора",
    "содержательного обсуждения",
    "не содержит запроса",
    "запрос носит сервисный характер",
    "ошибочн",
    "техническ",
    "автоинформ",
    "номер не используется",
    "контакт с потенциальным клиентом не состоялся",
    "неактуален",
    "не подтвердил релевантный контакт",
    "не связано с учебным центром",
    "не выразил интерес",
    "продолжение диалога невозможно",
)

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

SUMMARY_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
SUMMARY_TAIL_RE = re.compile(
    r"\s+(?:итог|обсудили|обсуждали|возражения|ограничения|контекст|важно|примечание)\s*[:—-].*$",
    re.IGNORECASE,
)
INCOMPLETE_ACTION_END_RE = re.compile(r"(?:\b(?:и|в|во|на|по|с|со|для|к|ко|о|об|от|до|или|а|но|чтобы)|[,—-])$", re.IGNORECASE)
NEW_YEAR_PHRASE_RE = re.compile(r"\bпосле\s+нового\s+года\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-я]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")
BOOKING_CODE_RE = re.compile(r"\b\d{2,}(?:[-\s]\d{2,})+\b|\b\d{6,}\b")
ROLE_PERSON_RE = re.compile(
    r"\b(?P<role>менеджер|куратор|администратор|оператор|клиент(?:ка)?|родител[ьи]|мама|папа|"
    r"ученик|ученица|реб[её]нок|студент(?:ка)?)\s+"
    r"[А-ЯЁ][а-яё]+(?:[-\s]+[А-ЯЁ][а-яё]+){0,2}\b"
)
SINGLE_PERSON_TARGET_RE = re.compile(
    r"\b(?P<verb>передать|перезвонить|позвонить|отправить|направить|выслать)\s+"
    r"[А-ЯЁ][а-яё]{2,}\b"
)
PERSON_NAME_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}(?:[-\s]+[А-ЯЁ][а-яё]{2,}){1,2}\b")
SUMMARY_CUE_PATTERNS = (
    re.compile(
        r"(?:следующ(?:ий|его)\s+шаг|дальнейш(?:ий|ие)\s+(?:шаг|действия)|"
        r"договор[её]нност[ьи]|итог(?:овый)?\s+шаг)\s*(?:[:—-]|\s+это\s+)\s*(?P<action>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:договорились|согласовали|согласовано|решили)[,\s]*(?:о\s+том,?\s*)?(?:что\s+)?(?P<action>.+)",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:нужно|надо|требуется|необходимо)\b\s+(?P<action>.+)", re.IGNORECASE),
    re.compile(
        r"(?i:(?:менеджер|куратор|администратор|оператор))"
        r"(?!\s+(?i:не)\b)"
        r"(?:\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})?\s+"
        r"(?P<action>(?i:(?:отправит|пришл[её]т|вышлет|направит|перезвонит|свяжется|пообещал[аи]?\s+"
        r"уточнит|проверит|подготовит|продублирует|передаст|оформит|согласует)).+)",
    ),
    re.compile(
        r"(?:менеджер|куратор|администратор|оператор)[^.?!;]{0,160}\b"
        r"(?P<action>пообещал[аи]?\s+(?:отправить|выслать|направить|перезвонить|связаться|"
        r"уточнить|проверить|подготовить|продублировать|передать|оформить|согласовать).+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?i:(?:клиент(?:ка)?|родител[ьи]|мама|папа))"
        r"(?:\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})?\s+"
        r"(?i:(?:жд[её]т|попросил[аи]?|просил[аи]?|запросил[аи]?|ожидает))\s+(?P<action>.+)",
    ),
)


@dataclass(frozen=True)
class NextStepResolution:
    status: str
    action: str
    display_text: str
    confidence: str
    reason_code: str
    resolution_kind: str = "historical_hint"
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

    def to_informational_json_dict(self) -> Mapping[str, Any]:
        """Expose history without allowing an old promise to masquerade as a current task."""
        payload = dict(self.to_json_dict())
        payload["historical_status"] = self.status
        payload["historical_reason_code"] = self.reason_code
        if self.resolution_kind == "historical_hint" and self.status in {
            NEXT_STEP_STATUS_ACTIVE,
            NEXT_STEP_STATUS_NEEDS_MANAGER_REVIEW,
        }:
            payload["historical_action"] = payload.get("action") or ""
            payload["historical_display_text"] = payload.get("display_text") or ""
            payload["action"] = ""
            payload["display_text"] = ""
            if self.status == NEXT_STEP_STATUS_ACTIVE:
                payload["status"] = NEXT_STEP_STATUS_EMPTY
                payload["reason_code"] = "historical_hint_only"
        return payload


@dataclass(frozen=True)
class ManagerActionResolution:
    """The only actionable next-step contract: a fully proven open AMO task."""

    status: str = NEXT_STEP_STATUS_NEEDS_MANAGER_REVIEW
    action: str = ""
    reason: str = "amo_task_missing"
    responsible_ref: str = ""
    responsible_name: str = ""
    due_at: str = ""
    action_provenance: Mapping[str, str] = field(default_factory=dict)
    owner_provenance: Mapping[str, str] = field(default_factory=dict)
    due_provenance: Mapping[str, str] = field(default_factory=dict)
    readiness_state: str = "review"
    readiness_reason_codes: tuple[str, ...] = field(default_factory=lambda: ("amo_task_missing",))
    resolution_kind: str = "proven_manager_action"


@dataclass(frozen=True)
class ManagerActionReadSnapshot:
    task_rows_by_customer: Mapping[str, tuple[Mapping[str, Any], ...]]
    opportunities_by_id: Mapping[str, sqlite3.Row]
    lead_owner_customer_ids_by_lead_id: Mapping[str, tuple[str, ...]]
    conflict_customer_ids: frozenset[str]
    freshness_failures: tuple[str, ...]
    family_customer_ids_by_customer: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    contact_restrictions_by_customer: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def event_has_explicit_contact_opt_out(event: Mapping[str, Any]) -> bool:
    """Recognize only a direct inbound stop-contact request, never a call-quality hint."""
    if _compact(event.get("direction")).casefold() != "inbound":
        return False
    event_type = _compact(event.get("event_type")).casefold()
    source_system = _compact(event.get("source_system")).casefold()
    record = _mapping(event.get("record"))
    if (
        event_type not in _DIRECT_MESSAGE_EVENT_TYPES
        and "wappi" not in source_system
    ):
        return False
    message = _mapping(record.get("message"))
    for value in (
        message.get("text"),
        record.get("full_clean_text"),
        event.get("text_preview"),
        event.get("summary"),
        record.get("text"),
        record.get("body"),
    ):
        text = re.sub(r"\s+", " ", _compact(value)).strip()
        for match in _DIRECT_CONTACT_OPTOUT_RE.finditer(text):
            sentence_start = max(text.rfind(mark, 0, match.start()) for mark in ".!?;\n") + 1
            sentence_ends = [text.find(mark, match.end()) for mark in ".!?;\n"]
            sentence_end = min((pos for pos in sentence_ends if pos >= 0), default=len(text))
            sentence = text[sentence_start:sentence_end]
            clause_start = max(text.rfind(mark, 0, match.start()) for mark in ".!?;,\n") + 1
            prefix = text[clause_start:match.start()]
            if (
                not _REPORTED_CONTACT_OPTOUT_RE.search(prefix)
                and (
                    _PERMANENT_CONTACT_OPTOUT_RE.search(sentence)
                    or not _TEMPORARY_CONTACT_PAUSE_RE.search(sentence)
                )
            ):
                return True
    return False


def load_durable_contact_restrictions_batch(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: Sequence[str],
    as_of: datetime,
) -> tuple[Mapping[str, tuple[str, ...]], Mapping[str, tuple[str, ...]]]:
    """Load trusted family scopes and high-confidence contact restrictions in one batch."""
    scopes = trusted_family_customer_ids_by_customer(
        con,
        tenant_id=tenant_id,
        customer_ids=customer_ids,
        as_of=as_of,
    )
    restrictions: dict[str, tuple[str, ...]] = {customer_id: () for customer_id in scopes}
    roots_by_member: dict[str, set[str]] = defaultdict(set)
    for root, members in scopes.items():
        for member in members:
            roots_by_member[member].add(root)
    selected_members = tuple(sorted(roots_by_member))
    if not selected_members:
        return scopes, restrictions
    selected = json.dumps(selected_members, ensure_ascii=False)
    for row in con.execute(
        "SELECT customer_id,record_json FROM customer_identities WHERE tenant_id=? "
        "AND customer_id IN (SELECT value FROM json_each(?))",
        (tenant_id, selected),
    ):
        record = _safe_json_object(row["record_json"])
        metadata = _mapping(record.get("metadata"))
        positive = (
            record.get("no_contact"),
            record.get("opt_out"),
            record.get("do_not_contact"),
            metadata.get("no_contact"),
            metadata.get("opt_out"),
            metadata.get("do_not_contact"),
        )
        blocked = any(
            value is True or str(value).strip().casefold() in {"1", "true", "yes", "да"}
            for value in positive
        )
        allowed_values = (metadata.get("contact_allowed"), record.get("contact_allowed"))
        blocked = blocked or any(
            value is False
            or (
                value is not None
                and str(value).strip().casefold() in {"0", "false", "no", "нет"}
            )
            for value in allowed_values
        )
        if blocked:
            for root in roots_by_member.get(str(row["customer_id"]), ()):
                restrictions[root] = ("durable_opt_out",)
    rows = con.execute(
        "SELECT customer_id,event_type,source_system,direction,text_preview,summary,record_json "
        "FROM timeline_events WHERE tenant_id=? "
        "AND customer_id IN (SELECT value FROM json_each(?)) "
        "AND julianday(event_at)<=julianday(?) "
        "AND direction='inbound' "
        "AND (event_type IN ('email_message','telegram_message','whatsapp_message',"
        "                    'max_message','web_chat_message','wappi_message') "
        "     OR source_system LIKE '%wappi%') "
        "AND match_status IN ('strong_unique','manual') "
        "AND (superseded_by IS NULL OR superseded_by='') "
        "AND COALESCE(json_extract(record_json,'$.metadata.pending_attribution'),0) NOT IN (1,'true')",
        (tenant_id, selected, as_of.isoformat()),
    ).fetchall()
    for row in rows:
        stored = _safe_json_object(row["record_json"])
        event = dict(row)
        event["record"] = _mapping(stored.get("record"))
        if event_has_explicit_contact_opt_out(event):
            for root in roots_by_member.get(str(row["customer_id"]), ()):
                restrictions[root] = ("durable_opt_out",)
    return scopes, restrictions


def load_durable_contact_restrictions(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
    as_of: datetime,
) -> tuple[str, ...]:
    """Return high-confidence, cutoff-safe contact restrictions for one trusted family."""
    _, restrictions = load_durable_contact_restrictions_batch(
        con,
        tenant_id=tenant_id,
        customer_ids=(customer_id,),
        as_of=as_of,
    )
    return restrictions[customer_id]


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


def resolve_customer_manager_action(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
    as_of: datetime | None = None,
    read_snapshot: ManagerActionReadSnapshot | None = None,
) -> ManagerActionResolution:
    """Resolve one current action only from an exact, fresh, open AMO task."""
    checked_at = as_of or datetime.now(timezone.utc)
    if checked_at.tzinfo is None or checked_at.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    checked_at = checked_at.astimezone(timezone.utc)
    snapshot = read_snapshot or load_manager_action_read_snapshot(
        con,
        tenant_id=tenant_id,
        customer_ids=(customer_id,),
        as_of=checked_at,
    )
    rows = snapshot.task_rows_by_customer.get(customer_id, ())
    global_failures: list[str] = []
    if customer_id in snapshot.conflict_customer_ids:
        global_failures.append("identity_conflict_open")
    restrictions = snapshot.contact_restrictions_by_customer.get(customer_id)
    if restrictions is None:
        restrictions = load_durable_contact_restrictions(
            con,
            tenant_id=tenant_id,
            customer_id=customer_id,
            as_of=checked_at,
        )
    global_failures.extend(restrictions)
    global_failures.extend(snapshot.freshness_failures)
    candidates = [
        _manager_action_from_amo_task(
            tenant_id=tenant_id,
            customer_id=customer_id,
            row=row,
            as_of=checked_at,
            global_failures=global_failures,
            duplicate_task_id=int(row["global_task_id_count"] or 0) > 1,
            opportunities_by_id=snapshot.opportunities_by_id,
            lead_owner_customer_ids_by_lead_id=snapshot.lead_owner_customer_ids_by_lead_id,
        )
        for row in rows
    ]
    ready = [item for item in candidates if item[0].readiness_state == "ready"]
    if ready:
        return min(ready, key=lambda item: (item[1], item[2]))[0]
    if candidates:
        return min(candidates, key=lambda item: (item[3], item[1], item[2]))[0]
    reasons = tuple(dict.fromkeys((*global_failures, "amo_task_missing")))
    return ManagerActionResolution(reason=reasons[0], readiness_reason_codes=reasons)


def load_manager_action_read_snapshot(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: Sequence[str],
    as_of: datetime,
    family_customer_ids_by_customer: Mapping[str, tuple[str, ...]] | None = None,
    contact_restrictions_by_customer: Mapping[str, tuple[str, ...]] | None = None,
) -> ManagerActionReadSnapshot:
    """Read selected customers' tasks and only their referenced opportunities once."""
    selected_customer_ids = tuple(sorted({str(value) for value in customer_ids if str(value)}))
    if family_customer_ids_by_customer is None or contact_restrictions_by_customer is None:
        family_scopes, contact_restrictions = load_durable_contact_restrictions_batch(
            con,
            tenant_id=tenant_id,
            customer_ids=selected_customer_ids,
            as_of=as_of,
        )
    else:
        family_scopes = {
            customer_id: tuple(family_customer_ids_by_customer.get(customer_id) or (customer_id,))
            for customer_id in selected_customer_ids
        }
        contact_restrictions = {
            customer_id: tuple(contact_restrictions_by_customer.get(customer_id) or ())
            for customer_id in selected_customer_ids
        }
    task_rows_by_customer: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    if selected_customer_ids:
        task_rows = con.execute(
            """
            SELECT event_id, customer_id, opportunity_id, event_at, event_type, source_system,
                   source_id, source_ref, match_status, superseded_by, record_json
            FROM (
              SELECT task.event_id, task.customer_id, task.opportunity_id, task.event_at,
                     task.event_type, task.source_system, task.source_id, task.source_ref,
                     task.match_status, task.superseded_by, task.record_json,
                     ROW_NUMBER() OVER (
                       PARTITION BY task.customer_id ORDER BY task.event_at DESC, task.event_id DESC
                     ) AS customer_row_number
              FROM timeline_events AS task
              WHERE task.tenant_id = ? AND task.event_type = 'amo_task'
                AND task.customer_id IN (SELECT value FROM json_each(?))
                AND julianday(task.event_at)<=julianday(?)
            )
            WHERE customer_row_number <= 500
            ORDER BY customer_id, event_at DESC, event_id DESC
            """,
            (tenant_id, json.dumps(selected_customer_ids, ensure_ascii=False), as_of.isoformat()),
        ).fetchall()
        selected_task_ids = tuple(sorted({
            str(row["source_id"])
            for row in task_rows
            if str(row["source_id"] or "")
        }))
        global_task_id_counts: dict[str, int] = {}
        if selected_task_ids:
            global_task_id_counts = {
                str(row["source_id"]): int(row["global_task_id_count"] or 0)
                for row in con.execute(
                    """
                    SELECT task.source_id, COUNT(*) AS global_task_id_count
                    FROM timeline_events AS task
                    JOIN json_each(?) AS selected_task ON selected_task.value = task.source_id
                    WHERE task.tenant_id = ? AND task.event_type = 'amo_task'
                      AND julianday(task.event_at)<=julianday(?)
                    GROUP BY task.source_id
                    """,
                    (json.dumps(selected_task_ids, ensure_ascii=False), tenant_id, as_of.isoformat()),
                ).fetchall()
            }
        for row in task_rows:
            task = dict(row)
            task["global_task_id_count"] = global_task_id_counts.get(str(row["source_id"] or ""), 0)
            task_rows_by_customer[str(row["customer_id"])].append(task)
    referenced_opportunity_ids = tuple(sorted({
        str(row["opportunity_id"])
        for rows in task_rows_by_customer.values()
        for row in rows
        if str(row["opportunity_id"] or "")
    }))
    opportunities_by_id: dict[str, sqlite3.Row] = {}
    if referenced_opportunity_ids:
        opportunities_by_id = {
            str(row["opportunity_id"]): row
            for row in con.execute(
                """
                SELECT opportunity_id, customer_id, opportunity_type, source_system, source_id,
                       status, opened_at, closed_at
                FROM customer_opportunities
                WHERE tenant_id=? AND opportunity_id IN (SELECT value FROM json_each(?))
                """,
                (tenant_id, json.dumps(referenced_opportunity_ids, ensure_ascii=False)),
            ).fetchall()
        }
    referenced_lead_ids = tuple(sorted({
        str(row["source_id"])
        for row in opportunities_by_id.values()
        if str(row["source_id"] or "")
    }))
    lead_owners: dict[str, set[str]] = defaultdict(set)
    if referenced_lead_ids and _table_exists(con, "identity_links"):
        for row in con.execute(
            """
            SELECT link_value,customer_id
            FROM identity_links
            WHERE tenant_id=? AND link_type='amo_lead_id'
              AND match_class IN ('strong_unique','manual')
              AND customer_id IS NOT NULL AND customer_id!=''
              AND (first_seen_at IS NULL OR julianday(first_seen_at)<=julianday(?))
              AND link_value IN (SELECT value FROM json_each(?))
            """,
            (
                tenant_id,
                as_of.isoformat(),
                json.dumps(referenced_lead_ids, ensure_ascii=False),
            ),
        ):
            lead_owners[str(row["link_value"])].add(str(row["customer_id"]))
    return ManagerActionReadSnapshot(
        task_rows_by_customer={key: tuple(value) for key, value in task_rows_by_customer.items()},
        opportunities_by_id=opportunities_by_id,
        lead_owner_customer_ids_by_lead_id={
            lead_id: tuple(sorted(lead_owners.get(lead_id, ())))
            for lead_id in referenced_lead_ids
        },
        conflict_customer_ids=open_family_identity_conflict_customer_ids(
            con,
            tenant_id,
            as_of=as_of.isoformat(),
        ),
        freshness_failures=_amo_tasks_freshness_failures(con, tenant_id=tenant_id, as_of=as_of),
        family_customer_ids_by_customer=family_scopes,
        contact_restrictions_by_customer=contact_restrictions,
    )


def _manager_action_from_amo_task(
    *,
    tenant_id: str,
    customer_id: str,
    row: Mapping[str, Any],
    as_of: datetime,
    global_failures: Sequence[str],
    duplicate_task_id: bool,
    opportunities_by_id: Mapping[str, sqlite3.Row],
    lead_owner_customer_ids_by_lead_id: Mapping[str, tuple[str, ...]],
) -> tuple[ManagerActionResolution, datetime, str, int]:
    stored = _safe_json_object(row["record_json"])
    record = _mapping(stored.get("record"))
    metadata = _mapping(stored.get("metadata"))
    next_step = _mapping(record.get("next_step"))
    provenance = _mapping(record.get("provenance"))
    task_id = _compact(row["source_id"])
    lead_id = _compact(provenance.get("entity_id"))
    action = _compact(record.get("action_text"))
    next_action = _compact(next_step.get("action"))
    responsible_id = _compact(record.get("responsible_user_id"))
    actor_ref = _compact(stored.get("actor_ref"))
    actor_name = _compact(stored.get("actor_name") or record.get("responsible_user_name"))
    complete_till = _parse_absolute_iso_datetime(record.get("complete_till"))
    next_due = _parse_absolute_iso_datetime(next_step.get("due"))
    event_at = _parse_absolute_iso_datetime(row["event_at"])
    due_sort = complete_till or datetime.max.replace(tzinfo=timezone.utc)
    failures = list(global_failures)

    if row["superseded_by"] not in (None, ""):
        failures.append("task_superseded")
    if _compact(row["source_system"]) != "amocrm_snapshot":
        failures.append("task_source_invalid")
    if _compact(row["match_status"]) != "strong_unique":
        failures.append("task_match_not_strong")
    if metadata.get("pending_attribution") in (True, 1, "true"):
        failures.append("task_pending_attribution")
    if not task_id:
        failures.append("task_id_missing")
    if _compact(row["source_ref"]) != f"amo:task:{task_id}":
        failures.append("task_source_ref_mismatch")
    if duplicate_task_id:
        failures.append("duplicate_task_id")
    if task_id != _compact(provenance.get("task_id")):
        failures.append("task_id_provenance_mismatch")
    if _compact(provenance.get("entity_type")).casefold() not in {"lead", "leads"}:
        failures.append("task_entity_type_invalid")
    if not lead_id:
        failures.append("task_lead_missing")
    else:
        lead_owners = lead_owner_customer_ids_by_lead_id.get(lead_id, ())
        if not lead_owners:
            failures.append("task_lead_owner_missing")
        elif len(lead_owners) > 1:
            failures.append("task_lead_owner_ambiguous")
        elif lead_owners[0] != customer_id:
            failures.append("task_lead_owner_foreign")
    completed = record.get("completed")
    if completed is True:
        failures.append("task_completed")
    elif completed is not False:
        failures.append("task_completion_unknown")
    if not is_meaningful_manager_action(action):
        failures.append("task_action_not_concrete")
    if completed is False and action != next_action:
        failures.append("task_action_mismatch")
    if not responsible_id:
        failures.append("task_owner_missing")
    elif not responsible_id.isdecimal() or int(responsible_id) <= 0:
        failures.append("task_owner_invalid")
    elif actor_ref != f"amo:user:{responsible_id}":
        failures.append("task_owner_actor_mismatch")
    complete_till_text = _compact(record.get("complete_till"))
    if not complete_till_text:
        failures.append("task_due_missing")
    elif complete_till is None:
        failures.append("task_due_not_absolute")
    elif complete_till <= as_of:
        failures.append("task_due_not_future")
    if completed is False and next_due != complete_till:
        failures.append("task_due_mismatch")
    if event_at is None or event_at > as_of:
        failures.append("task_event_time_invalid")

    opportunity_id = _compact(row["opportunity_id"])
    opportunity = opportunities_by_id.get(opportunity_id)
    if opportunity is None:
        failures.append("task_opportunity_missing")
    else:
        if _compact(opportunity["customer_id"]) != customer_id:
            failures.append("task_opportunity_foreign_customer")
        if _compact(opportunity["opportunity_type"]) != "amo_deal":
            failures.append("task_opportunity_not_amo_deal")
        if _compact(opportunity["source_system"]) != "amocrm_snapshot":
            failures.append("task_opportunity_source_invalid")
        if _compact(opportunity["source_id"]) != lead_id:
            failures.append("task_opportunity_lead_mismatch")
        if _compact(provenance.get("opportunity_source_id")) != lead_id:
            failures.append("task_opportunity_provenance_mismatch")
        if _compact(provenance.get("opportunity_source_system")) != "amocrm_snapshot":
            failures.append("task_opportunity_provenance_source_invalid")
        opportunity_state = _manager_amo_opportunity_state(dict(opportunity))
        if not _is_active_deal_at(dict(opportunity), as_of=as_of):
            failures.append("task_opportunity_not_active")
        elif opportunity_state != "active":
            # A later close is compatible with an active historical cutoff.
            closed_at = _parse_iso_datetime(opportunity["closed_at"])
            if closed_at is None or closed_at <= as_of:
                failures.append("task_opportunity_status_missing")

    reasons = tuple(dict.fromkeys(failures))
    action_provenance = {
        "customer_id": customer_id,
        "event_id": _compact(row["event_id"]),
        "event_at": _compact(row["event_at"]),
        "source_system": _compact(row["source_system"]),
        "task_id": task_id,
        "lead_id": lead_id,
        "opportunity_id": opportunity_id,
    }
    owner_provenance = {**action_provenance, "field": "responsible_user_id"} if responsible_id else {}
    due_provenance = {
        **action_provenance,
        "field": "due_at",
        "source_field": "complete_till",
    } if record.get("complete_till") else {}
    candidate_safe_to_display = not any(
        reason in _MANAGER_ACTION_UNSAFE_CANDIDATE_REASONS for reason in reasons
    )
    primary_reason = next(
        (reason for reason in reasons if reason in _MANAGER_ACTION_UNSAFE_CANDIDATE_REASONS),
        reasons[0] if reasons else "",
    )
    return (
        ManagerActionResolution(
            status=NEXT_STEP_STATUS_ACTIVE if not reasons else NEXT_STEP_STATUS_NEEDS_MANAGER_REVIEW,
            action=action if candidate_safe_to_display else "",
            reason=primary_reason,
            responsible_ref=(
                actor_ref
                if candidate_safe_to_display and responsible_id and actor_ref == f"amo:user:{responsible_id}"
                else ""
            ),
            responsible_name=actor_name if candidate_safe_to_display else "",
            due_at=complete_till.isoformat() if candidate_safe_to_display and complete_till is not None else "",
            action_provenance=action_provenance if candidate_safe_to_display else {},
            owner_provenance=owner_provenance if candidate_safe_to_display else {},
            due_provenance=due_provenance if candidate_safe_to_display else {},
            readiness_state="ready" if not reasons else "review",
            readiness_reason_codes=reasons,
        ),
        due_sort,
        task_id,
        0 if completed is False else 1,
    )


def _manager_amo_opportunity_state(opportunity: Mapping[str, Any]) -> str:
    if _compact(opportunity.get("closed_at")):
        return "closed"
    status = _compact(opportunity.get("status")).casefold()
    if not status:
        return "unknown"
    if status.isdecimal():
        if status in _MANAGER_AMO_TERMINAL_STATUS_IDS:
            return "closed"
        return "active" if int(status) > 0 else "unknown"
    if status in _MANAGER_AMO_ACTIVE_STATUS_TEXTS:
        return "active"
    if status in _MANAGER_AMO_CLOSED_STATUS_TEXTS:
        return "closed"
    return "unknown"


def _amo_tasks_freshness_failures(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    as_of: datetime,
    max_age_hours: float = 36.0,
) -> tuple[str, ...]:
    failures: list[str] = []
    if not _table_exists(con, "ingestion_cursors"):
        failures.append("amo_tasks_cursor_missing")
    else:
        cursor = con.execute(
            "SELECT updated_at,metadata_json FROM ingestion_cursors "
            "WHERE tenant_id=? AND source_system='amo_tasks_updated_at'",
            (tenant_id,),
        ).fetchone()
        checked = _parse_absolute_iso_datetime(cursor["updated_at"] if cursor else None)
        if checked is None:
            failures.append("amo_tasks_cursor_missing")
        elif checked > as_of + timedelta(minutes=5):
            failures.append("amo_tasks_cursor_in_future")
        elif as_of - checked > timedelta(hours=max_age_hours):
            failures.append("amo_tasks_cursor_stale")
        if cursor is not None:
            cursor_payload = _safe_json_object(cursor["metadata_json"])
            cursor_metadata = _mapping(cursor_payload.get("metadata"))
            if cursor_metadata.get("bootstrap_complete") is not True:
                failures.append("amo_tasks_cursor_incomplete")
            if _compact(cursor_metadata.get("last_status")).casefold() != "ok":
                failures.append("amo_tasks_cursor_not_ok")
    if not _table_exists(con, "ingestion_runs"):
        failures.append("amo_tasks_import_missing")
    else:
        run = con.execute(
            """
            SELECT status, finished_at
            FROM ingestion_runs
            WHERE tenant_id=? AND source_system='amocrm_snapshot'
              AND (run_kind='amo_tasks_incremental' OR source_ref='amocrm:tasks:updated_at')
            ORDER BY finished_at DESC, started_at DESC, run_id DESC
            LIMIT 1
            """,
            (tenant_id,),
        ).fetchone()
        finished = _parse_absolute_iso_datetime(run["finished_at"] if run else None)
        if run is None or finished is None:
            failures.append("amo_tasks_import_missing")
        elif _compact(run["status"]) != "completed":
            failures.append("amo_tasks_import_not_completed")
        elif finished > as_of + timedelta(minutes=5):
            failures.append("amo_tasks_import_in_future")
        elif as_of - finished > timedelta(hours=max_age_hours):
            failures.append("amo_tasks_import_stale")
    return tuple(failures)


def is_meaningful_manager_action(value: str) -> bool:
    text = _compact(value).casefold().strip(" .;:—-")
    if not text or text in {"уточнить у менеджера", "связаться с клиентом", "позвонить клиенту"}:
        return False
    if "посмотреть историю" in text:
        return False
    if _looks_incomplete_action(text):
        return False
    if re.search(r"\b(?:что[-\s]?(?:то|нибудь)|как[-\s]?нибудь)\b", text):
        return False
    return len(text.split()) >= 3


def _safe_json_object(value: str | None) -> Mapping[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _parse_iso_datetime(value: Any) -> datetime | None:
    text = _compact(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _parse_absolute_iso_datetime(value: Any) -> datetime | None:
    text = _compact(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


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
        text = _next_step_value(value)
        if text:
            return text
    return _extract_next_step_from_summary(event)


def _next_step_value(value: Any) -> str:
    if isinstance(value, Mapping):
        return _compact(value.get("action"))
    return _compact(value)


def extract_next_step_action(event: Mapping[str, Any]) -> str:
    return _extract_next_step(event)


def _extract_next_step_from_summary(event: Mapping[str, Any]) -> str:
    if str(event.get("event_type") or "").casefold() != "mango_call":
        return ""
    if is_non_contentful_call_record(event):
        return ""
    summary = _call_summary_text(event)
    if not summary or _summary_has_no_next_step(summary) or _summary_is_non_conversation(summary):
        return ""

    candidates: list[str] = []
    for sentence in _summary_sentences(summary):
        if _summary_has_no_next_step(sentence):
            continue
        if not (_has_any(sentence.casefold(), SUMMARY_ACTION_MARKERS) and _has_any(sentence.casefold(), SUMMARY_ACTION_VERBS)):
            continue
        action = _candidate_action_from_sentence(sentence)
        if not action:
            continue
        action = _sanitize_extracted_next_step(action, event)
        if action and _candidate_has_step_marker(action):
            candidates.append(action)
    return candidates[-1] if candidates else ""


def _call_summary_text(event: Mapping[str, Any]) -> str:
    record = _mapping(event.get("record"))
    call_analysis = _mapping(record.get("call_analysis") or event.get("call_analysis"))
    for value in (
        event.get("summary"),
        record.get("summary"),
        call_analysis.get("summary"),
        call_analysis.get("history_summary"),
        event.get("text_preview"),
    ):
        text = _compact(value)
        if text:
            return text
    return ""


def _summary_sentences(summary: str) -> tuple[str, ...]:
    parts = SUMMARY_SENTENCE_RE.split(summary)
    result: list[str] = []
    for part in parts:
        for item in part.split(";"):
            text = _compact(item).strip(" .;")
            if text:
                result.append(text)
    return tuple(result)


def _summary_has_no_next_step(value: str) -> bool:
    text = value.casefold().replace("ё", "е")
    return any(marker.replace("ё", "е") in text for marker in SUMMARY_NO_STEP_MARKERS)


def _summary_is_non_conversation(value: str) -> bool:
    text = value.casefold().replace("ё", "е")
    return any(marker.replace("ё", "е") in text for marker in SUMMARY_NON_CONVERSATION_MARKERS)


def _candidate_action_from_sentence(sentence: str) -> str:
    for pattern in SUMMARY_CUE_PATTERNS:
        match = pattern.search(sentence)
        if match:
            return _compact(match.group("action"))
    return ""


def _sanitize_extracted_next_step(action: str, event: Mapping[str, Any]) -> str:
    text = SUMMARY_TAIL_RE.sub("", _compact(action)).strip(" .;:—-")
    for name in _actor_names(event):
        text = re.sub(rf"\b{re.escape(name)}\b", "менеджер", text, flags=re.IGNORECASE)
    text = EMAIL_RE.sub("<email_masked>", text)
    text = PHONE_RE.sub("<phone_masked>", text)
    text = BOOKING_CODE_RE.sub("<number_masked>", text)
    text = NEW_YEAR_PHRASE_RE.sub("после праздников", text)
    text = ROLE_PERSON_RE.sub(lambda match: match.group("role"), text)
    text = SINGLE_PERSON_TARGET_RE.sub(lambda match: f"{match.group('verb')} клиенту", text)
    text = PERSON_NAME_RE.sub("<name_masked>", text)
    text = _compact(text).strip(" .;:—-")
    if not text:
        return ""
    if _looks_incomplete_action(text):
        return ""
    if has_personal_data_risk(text):
        text = _pii_safe_fallback_step(text)
    if not text or has_personal_data_risk(text):
        return ""
    return text[:1].upper() + text[1:]


def _looks_incomplete_action(action: str) -> bool:
    return bool(INCOMPLETE_ACTION_END_RE.search(action.strip()))


def _actor_names(event: Mapping[str, Any]) -> tuple[str, ...]:
    record = _mapping(event.get("record"))
    metadata = _mapping(event.get("metadata"))
    values = (
        record.get("actor_name"),
        record.get("manager_name"),
        record.get("operator_name"),
        metadata.get("actor_name"),
        metadata.get("manager_name"),
        metadata.get("operator_name"),
    )
    return tuple(text for value in values if (text := _compact(value)))


def _candidate_has_step_marker(action: str) -> bool:
    text = action.casefold()
    return _has_any(text, SUMMARY_ACTION_MARKERS)


def _pii_safe_fallback_step(action: str) -> str:
    text = action.casefold()
    if _has_any(text, DOCUMENT_STEP_MARKERS):
        return "Отправить документы/материалы"
    if _has_any(text, PAYMENT_STEP_MARKERS):
        return "Уточнить оплату/чек"
    if _has_any(text, CALLBACK_STEP_MARKERS):
        return "Перезвонить клиенту"
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
    "ManagerActionReadSnapshot",
    "ManagerActionResolution",
    "NextStepResolution",
    "extract_next_step_action",
    "is_meaningful_manager_action",
    "load_manager_action_read_snapshot",
    "resolve_customer_manager_action",
    "resolve_customer_next_step",
]
