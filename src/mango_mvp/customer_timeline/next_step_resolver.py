from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from mango_mvp.insights.sanitizers import has_personal_data_risk


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
NEGATION_MARKERS = (
    "не приш",
    "не получил",
    "не получили",
    "не дош",
    "не прош",
    "отказ",
    "ошибка оплаты",
    "ошибка платеж",
    "ошибка платёж",
    "ошибка в счете",
    "ошибка в счёте",
    "ошибочно спис",
)
EXPLICIT_DATA_CONFLICT_MARKERS = (
    "конфликт",
    "противореч",
    "не совпада",
    "расхожд",
    "несовместим",
    "разные данные",
    "другой бренд",
    "другая школа",
    "другой класс",
    "другой предмет",
    "не тот клиент",
    "не тот ребенок",
    "не тот ребёнок",
)
GRADE_KEYS = ("grade", "class", "student_class", "child_class", "school_class", "klass")
BRAND_KEYS = ("brand", "tenant", "school_brand")

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
        r"(?:менеджер|куратор|администратор|оператор)"
        r"(?:\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})?\s+"
        r"(?P<action>(?:отправит|пришл[её]т|вышлет|направит|перезвонит|свяжется|пообещал[аи]?\s+"
        r"уточнит|проверит|подготовит|продублирует|передаст|оформит|согласует).+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:менеджер|куратор|администратор|оператор)[^.?!;]{0,160}\b"
        r"(?P<action>пообещал[аи]?\s+(?:отправить|выслать|направить|перезвонить|связаться|"
        r"уточнить|проверить|подготовить|продублировать|передать|оформить|согласовать).+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:клиент(?:ка)?|родител[ьи]|мама|папа)"
        r"(?:\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){0,2})?\s+"
        r"(?:жд[её]т|попросил[аи]?|просил[аи]?|запросил[аи]?|ожидает)\s+(?P<action>.+)",
        re.IGNORECASE,
    ),
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
    contradiction = _first_contradiction(relevant[: step_index + 1], later_events, kind)
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
    return _extract_next_step_from_summary(event)


def extract_next_step_action(event: Mapping[str, Any]) -> str:
    return _extract_next_step(event)


def _extract_next_step_from_summary(event: Mapping[str, Any]) -> str:
    if str(event.get("event_type") or "").casefold() != "mango_call":
        return ""
    if _call_record_is_not_contentful(event):
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


def _call_record_is_not_contentful(event: Mapping[str, Any]) -> bool:
    record = _mapping(event.get("record"))
    value = str(record.get("contentful") or "").strip().casefold()
    return value in {"0", "false", "нет", "no", "non_conversation"}


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


def _first_contradiction(
    previous_events: Sequence[Mapping[str, Any]],
    later_events: Sequence[Mapping[str, Any]],
    step_kind: str,
) -> Mapping[str, Any] | None:
    for event in later_events:
        text = _event_text(event)
        if _has_explicit_data_conflict(event):
            return event
        if step_kind in {"documents", "payment"} and _has_any(text, NEGATION_MARKERS):
            if step_kind == "documents" and _has_any(text, DOCUMENT_STEP_MARKERS):
                return event
            if step_kind == "payment" and _has_any(text, PAYMENT_STEP_MARKERS):
                return event
    return _first_structured_slot_conflict(previous_events, later_events)


def _has_explicit_data_conflict(event: Mapping[str, Any]) -> bool:
    text = " ".join(
        item
        for item in (
            _event_text(event),
            _nested_text(_mapping(event.get("record"))),
            _nested_text(_mapping(event.get("metadata"))),
        )
        if item
    ).casefold()
    return _has_any(text, EXPLICIT_DATA_CONFLICT_MARKERS)


def _first_structured_slot_conflict(
    previous_events: Sequence[Mapping[str, Any]],
    later_events: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    previous = _structured_slot_values(previous_events)
    if not any(previous.values()):
        return None
    for event in later_events:
        current = _structured_slot_values((event,))
        for slot in ("brand", "grade"):
            if _slot_values_conflict(previous.get(slot, set()), current.get(slot, set()), slot=slot):
                return event
    return None


def _structured_slot_values(events: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {"brand": set(), "grade": set()}
    for event in events:
        record = _mapping(event.get("record"))
        metadata = _mapping(event.get("metadata"))
        call_analysis = _mapping(record.get("call_analysis") or event.get("call_analysis"))
        containers = (record, metadata, call_analysis)
        values["brand"].update(_normalized_values_for_keys(containers, BRAND_KEYS, _normalize_brand))
        values["grade"].update(_normalized_values_for_keys(containers, GRADE_KEYS, _normalize_grade))
    return values


def _normalized_values_for_keys(
    containers: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    normalizer: Any,
) -> set[str]:
    result: set[str] = set()
    for container in containers:
        for key in keys:
            for value in _iter_text_values(container.get(key)):
                normalized = normalizer(value)
                if normalized:
                    result.add(normalized)
    return result


def _slot_values_conflict(previous: set[str], current: set[str], *, slot: str) -> bool:
    if not previous or not current or previous & current:
        return False
    if slot == "subject" and (len(previous) > 1 or len(current) > 1):
        return False
    return True


def _normalize_brand(value: str) -> str:
    text = value.casefold()
    if "фотон" in text or "foton" in text:
        return "foton"
    if "унпк" in text or "unpk" in text or "мфти" in text:
        return "unpk"
    return ""


def _normalize_grade(value: str) -> str:
    match = re.search(r"\b([1-9]|1[0-1])\s*(?:класс|кл\.?|grade)?\b", value.casefold())
    return match.group(1) if match else ""


def _iter_text_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = _compact(value)
        return (text,) if text else ()
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(_iter_text_values(item))
        return tuple(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[str] = []
        for item in value:
            result.extend(_iter_text_values(item))
        return tuple(result)
    text = _compact(value)
    return (text,) if text else ()


def _nested_text(value: Mapping[str, Any]) -> str:
    return " ".join(item.casefold() for item in _iter_text_values(value))


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
    "extract_next_step_action",
    "resolve_customer_next_step",
]
