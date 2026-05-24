from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from mango_mvp.channels.new_lead_funnel import (
    extract_format,
    extract_goal,
    extract_grade,
    extract_product,
    extract_subjects,
    normalize_brand,
    normalize_text,
)


DIALOGUE_MEMORY_SCHEMA_VERSION = "dialogue_memory_v2_2026_05_23"
MAX_TURNS = 10
MAX_PROMPT_TURNS = 4

QUESTION_KIND_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("live_availability", ("мест", "налич", "брон", "заброни")),
    (
        "price_fix",
        (
            "зафикс",
            "закреп",
            "подраст",
            "выраст",
            "оформить по текущ",
            "оформить по текущей",
            "по текущей цене",
            "по текущим условиям",
            "забронировать по цене",
            "забронировать цену",
            "что нужно для записи",
        ),
    ),
    ("price", ("сколько", "стоим", "цен", "прайс")),
    ("installment", ("рассроч", "долями", "частями", "помесяч", "банк", "одобрен")),
    ("schedule", ("распис", "когда", "во сколько", "дни", "время")),
    ("trial", ("пробн", "фрагмент занят")),
    ("address", ("адрес", "где", "куда ехать", "площадк", "метро")),
    ("camp", ("лвш", "лагер", "смен", "менделеево", "прожив", "питан")),
    ("platform", ("платформ", "запис", "мтс линк", "webinar", "вебинар")),
    ("identity", ("ты бот", "вы бот", "кто вы", "с кем я общаюсь", "ты gpt", "вы gpt")),
    ("off_topic", ("айфон", "iphone", "погода", "сочинение", "биткоин", "крипт")),
)

P0_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("refund", ("возврат", "вернуть деньги", "верните деньги", "деньги назад", "расторг")),
    ("complaint", ("жалоб", "претензи", "недовол", "возмущ", "обман", "ужас")),
    ("legal_threat", ("суд", "прокурат", "роспотреб", "иск", "адвокат", "по закону")),
    ("payment_dispute", ("оплатил", "оплатила", "оплата не", "платеж не", "деньги списали")),
)

COMMITMENT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("manager_handoff", ("передам менеджеру", "передам ответственному", "менеджер проверит", "менеджер свяжется")),
    ("check_availability", ("проверит наличие", "проверим наличие", "проверит места", "проверим места")),
    ("send_material", ("пришлем", "пришлю", "отправим", "отправлю")),
)

CURRENT_TERMS_SAFE_REQUEST_TEXT = "Передам менеджеру запрос на оформление по текущим условиям."
CURRENT_TERMS_REQUIRED_SLOTS = ("grade", "subject", "format")
CURRENT_TERMS_SLOT_QUESTIONS: Mapping[str, str] = {
    "grade": "Подскажите, пожалуйста, в каком классе ребёнок?",
    "subject": "Подскажите, пожалуйста, какой предмет или направление интересует?",
    "format": "Подскажите, пожалуйста, какой формат рассматриваете: онлайн или очно?",
}
CURRENT_TERMS_FORBIDDEN_PROMISES = (
    "seat",
    "contract",
    "payment",
    "reservation",
    "discount",
    "enrollment",
)
CURRENT_TERMS_FORBIDDEN_PROMISES_RU = (
    "место",
    "договор",
    "оплата",
    "бронь",
    "скидка",
    "запись без проверки",
)


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    text: str

    def to_json_dict(self) -> Mapping[str, str]:
        return {"role": self.role, "text": self.text[:700]}


@dataclass(frozen=True)
class DialogueSlot:
    value: str
    source: str
    confidence: float = 0.0

    def to_json_dict(self) -> Mapping[str, Any]:
        return {"value": self.value, "source": self.source, "confidence": round(float(self.confidence), 3)}


@dataclass(frozen=True)
class DialogueQuestion:
    text: str = ""
    kind: str = ""
    answered: bool = False

    def to_json_dict(self) -> Mapping[str, Any]:
        return {"text": self.text[:260], "kind": self.kind, "answered": self.answered}


@dataclass(frozen=True)
class DialogueMemory:
    session_id: str
    active_brand: str
    turns: tuple[DialogueTurn, ...] = ()
    known_slots: Mapping[str, DialogueSlot] = field(default_factory=dict)
    open_question: DialogueQuestion = field(default_factory=DialogueQuestion)
    answered_questions: tuple[str, ...] = ()
    last_bot_commitments: tuple[str, ...] = ()
    sales_stage: str = "cold"
    risk_flags: tuple[str, ...] = ()
    handoff_state: str = "none"
    fact_refs: tuple[str, ...] = ()
    route_history: tuple[str, ...] = ()
    topic_focus: Mapping[str, str] = field(default_factory=dict)
    unanswered_questions: tuple[str, ...] = ()
    safe_answered_parts: tuple[str, ...] = ()
    pending_manager_actions: tuple[str, ...] = ()
    client_confirmed_slots: Mapping[str, str] = field(default_factory=dict)
    crm_known_slots: Mapping[str, str] = field(default_factory=dict)
    bot_inferred_slots: Mapping[str, str] = field(default_factory=dict)
    do_not_reask_slots: tuple[str, ...] = ()
    conversation_summary_short: str = ""
    open_loop_summary: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    schema_version: str = DIALOGUE_MEMORY_SCHEMA_VERSION

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "active_brand": self.active_brand,
            "turns": [turn.to_json_dict() for turn in self.turns],
            "known_slots": {key: slot.to_json_dict() for key, slot in self.known_slots.items()},
            "open_question": self.open_question.to_json_dict(),
            "answered_questions": list(self.answered_questions),
            "last_bot_commitments": list(self.last_bot_commitments),
            "sales_stage": self.sales_stage,
            "risk_flags": list(self.risk_flags),
            "handoff_state": self.handoff_state,
            "fact_refs": list(self.fact_refs),
            "route_history": list(self.route_history),
            "topic_focus": dict(self.topic_focus),
            "unanswered_questions": list(self.unanswered_questions),
            "safe_answered_parts": list(self.safe_answered_parts),
            "pending_manager_actions": list(self.pending_manager_actions),
            "client_confirmed_slots": dict(self.client_confirmed_slots),
            "crm_known_slots": dict(self.crm_known_slots),
            "bot_inferred_slots": dict(self.bot_inferred_slots),
            "do_not_reask_slots": list(self.do_not_reask_slots),
            "conversation_summary_short": self.conversation_summary_short,
            "open_loop_summary": self.open_loop_summary,
            "updated_at": self.updated_at,
        }

    def to_prompt_view(self) -> Mapping[str, Any]:
        known_values = {key: slot.value for key, slot in self.known_slots.items() if slot.value}
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "active_brand": self.active_brand,
            "recent_turns": [turn.to_json_dict() for turn in self.turns[-MAX_PROMPT_TURNS:]],
            "known_slots": known_values,
            "slot_sources": {key: slot.source for key, slot in self.known_slots.items() if slot.value},
            "open_question": self.open_question.to_json_dict(),
            "answered_questions": list(self.answered_questions[-5:]),
            "last_bot_commitments": list(self.last_bot_commitments[-5:]),
            "sales_stage": self.sales_stage,
            "risk_flags": list(self.risk_flags),
            "handoff_state": self.handoff_state,
            "fact_refs": list(self.fact_refs[-8:]),
            "route_history": list(self.route_history[-5:]),
            "topic_focus": dict(self.topic_focus),
            "unanswered_questions": list(self.unanswered_questions[-5:]),
            "safe_answered_parts": list(self.safe_answered_parts[-8:]),
            "pending_manager_actions": list(self.pending_manager_actions[-5:]),
            "client_confirmed_slots": dict(self.client_confirmed_slots),
            "crm_known_slots": dict(self.crm_known_slots),
            "bot_inferred_slots": dict(self.bot_inferred_slots),
            "do_not_ask_again": list(self.do_not_reask_slots) or sorted(known_values),
            "conversation_summary_short": self.conversation_summary_short,
            "open_loop_summary": self.open_loop_summary,
            "safe_next_action": safe_next_action(self),
            "next_best_action_hint": next_best_action_hint(self),
        }


def build_dialogue_memory(
    *,
    current_message: str,
    active_brand: str,
    recent_messages: Sequence[str] = (),
    known_slots: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    previous_memory: Mapping[str, Any] | DialogueMemory | None = None,
    session_id: str = "",
) -> DialogueMemory:
    brand = normalize_brand(active_brand)
    turns = _turns_from_previous(previous_memory)
    if not turns:
        turns = tuple(_parse_recent_messages(recent_messages))
    current_text = _clean(current_message)
    if current_text:
        turns = (*turns, DialogueTurn("client", current_text))[-MAX_TURNS:]

    slot_map = _slots_from_previous(previous_memory)
    _merge_slots(slot_map, known_slots or {}, source_name="provided_context", confidence=0.82)
    _merge_slots(slot_map, _extract_slots_from_turns(turns), source_name="dialogue_memory", confidence=0.9)
    _merge_slots(slot_map, _extract_slots_from_text(current_text), source_name="dialogue_memory", confidence=0.95, override=True)

    open_question = _detect_open_question(current_text)
    if not open_question.text:
        previous = dialogue_memory_from_mapping(previous_memory) if isinstance(previous_memory, Mapping) else previous_memory
        if isinstance(previous, DialogueMemory) and previous.open_question.text and not previous.open_question.answered:
            open_question = previous.open_question
    risks = _detect_risk_flags("\n".join(turn.text for turn in turns if turn.role == "client"))
    commitments = _detect_commitments(turns)
    fact_refs = _fact_refs(context or {})
    route_history = _route_history(previous_memory)
    handoff = "required" if risks else ("suggested" if any("manager" in item for item in commitments) else "none")
    sales_stage = _sales_stage(slot_map, open_question=open_question, risk_flags=risks, handoff_state=handoff)
    unanswered = _unanswered_questions(previous_memory, open_question=open_question)
    topic_focus = _topic_focus(slot_map, open_question=open_question, active_brand=brand)
    client_confirmed = _slots_by_source(slot_map, {"dialogue_memory"})
    crm_known = _slots_by_source(slot_map, {"provided_context"})
    return DialogueMemory(
        session_id=session_id or _stable_session_id(brand, turns),
        active_brand=brand,
        turns=turns,
        known_slots=dict(slot_map),
        open_question=open_question,
        answered_questions=_answered_questions(previous_memory),
        last_bot_commitments=commitments,
        sales_stage=sales_stage,
        risk_flags=risks,
        handoff_state=handoff,
        fact_refs=fact_refs,
        route_history=route_history,
        topic_focus=topic_focus,
        unanswered_questions=unanswered,
        safe_answered_parts=_safe_answered_parts_from_previous(previous_memory),
        pending_manager_actions=_pending_manager_actions(commitments),
        client_confirmed_slots=client_confirmed,
        crm_known_slots=crm_known,
        bot_inferred_slots=_bot_inferred_slots(previous_memory),
        do_not_reask_slots=_do_not_reask_slots(slot_map),
        conversation_summary_short=_conversation_summary_short(slot_map, topic_focus=topic_focus, open_question=open_question),
        open_loop_summary=_open_loop_summary(open_question=open_question, risk_flags=risks, pending_actions=_pending_manager_actions(commitments)),
    )


def update_dialogue_memory_after_answer(
    memory: DialogueMemory | Mapping[str, Any],
    *,
    answer_text: str,
    route: str = "",
    fact_refs: Sequence[str] = (),
    safety_flags: Sequence[str] = (),
) -> DialogueMemory:
    current = memory if isinstance(memory, DialogueMemory) else dialogue_memory_from_mapping(memory)
    answer = _clean(answer_text)
    turns = current.turns
    if answer:
        turns = (*turns, DialogueTurn("bot", answer))[-MAX_TURNS:]
    route_history = tuple(dict.fromkeys([*current.route_history, str(route or "").strip()]))[-8:]
    risks = tuple(dict.fromkeys([*current.risk_flags, *_risk_flags_from_safety(safety_flags)]))
    commitments = tuple(dict.fromkeys([*current.last_bot_commitments, *_detect_commitments(turns)]))[-8:]
    answered = current.answered_questions
    open_question = current.open_question
    unanswered = tuple(current.unanswered_questions)
    if open_question.text and _answer_closes_question(answer, open_question.kind):
        answered = (*answered, open_question.text)[-8:]
        open_question = DialogueQuestion(open_question.text, open_question.kind, True)
        unanswered = tuple(item for item in unanswered if item != current.open_question.text)
    handoff = "required" if risks or route == "manager_only" else current.handoff_state
    safe_parts = tuple(dict.fromkeys([*current.safe_answered_parts, *_safe_answered_parts(answer, current.open_question.kind)]))[-12:]
    pending_actions = _pending_manager_actions(commitments)
    return DialogueMemory(
        session_id=current.session_id,
        active_brand=current.active_brand,
        turns=turns,
        known_slots=dict(current.known_slots),
        open_question=open_question,
        answered_questions=answered,
        last_bot_commitments=commitments,
        sales_stage=_sales_stage(current.known_slots, open_question=open_question, risk_flags=risks, handoff_state=handoff),
        risk_flags=risks,
        handoff_state=handoff,
        fact_refs=tuple(dict.fromkeys([*current.fact_refs, *[str(item) for item in fact_refs if str(item).strip()]]))[-12:],
        route_history=route_history,
        topic_focus=dict(current.topic_focus),
        unanswered_questions=unanswered,
        safe_answered_parts=safe_parts,
        pending_manager_actions=pending_actions,
        client_confirmed_slots=dict(current.client_confirmed_slots),
        crm_known_slots=dict(current.crm_known_slots),
        bot_inferred_slots=dict(current.bot_inferred_slots),
        do_not_reask_slots=tuple(current.do_not_reask_slots),
        conversation_summary_short=current.conversation_summary_short,
        open_loop_summary=_open_loop_summary(open_question=open_question, risk_flags=risks, pending_actions=pending_actions),
    )


def dialogue_memory_from_mapping(payload: Mapping[str, Any] | None) -> DialogueMemory:
    data = dict(payload or {})
    slots = {}
    for key, raw in (data.get("known_slots") or {}).items() if isinstance(data.get("known_slots"), Mapping) else ():
        if isinstance(raw, Mapping):
            slots[str(key)] = DialogueSlot(str(raw.get("value") or ""), str(raw.get("source") or "unknown"), float(raw.get("confidence") or 0.0))
        else:
            slots[str(key)] = DialogueSlot(str(raw or ""), "unknown", 0.0)
    open_raw = data.get("open_question") if isinstance(data.get("open_question"), Mapping) else {}
    return DialogueMemory(
        session_id=str(data.get("session_id") or ""),
        active_brand=normalize_brand(data.get("active_brand")),
        turns=tuple(
            DialogueTurn(str(item.get("role") or ""), _clean(item.get("text")))
            for item in (data.get("turns") or [])
            if isinstance(item, Mapping)
        )[-MAX_TURNS:],
        known_slots=slots,
        open_question=DialogueQuestion(
            str(open_raw.get("text") or ""),
            str(open_raw.get("kind") or ""),
            bool(open_raw.get("answered")),
        ),
        answered_questions=tuple(str(item) for item in (data.get("answered_questions") or ()) if str(item).strip()),
        last_bot_commitments=tuple(str(item) for item in (data.get("last_bot_commitments") or ()) if str(item).strip()),
        sales_stage=str(data.get("sales_stage") or "cold"),
        risk_flags=tuple(str(item) for item in (data.get("risk_flags") or ()) if str(item).strip()),
        handoff_state=str(data.get("handoff_state") or "none"),
        fact_refs=tuple(str(item) for item in (data.get("fact_refs") or ()) if str(item).strip()),
        route_history=tuple(str(item) for item in (data.get("route_history") or ()) if str(item).strip()),
        topic_focus=_plain_str_mapping(data.get("topic_focus")),
        unanswered_questions=tuple(str(item) for item in (data.get("unanswered_questions") or ()) if str(item).strip()),
        safe_answered_parts=tuple(str(item) for item in (data.get("safe_answered_parts") or ()) if str(item).strip()),
        pending_manager_actions=tuple(str(item) for item in (data.get("pending_manager_actions") or ()) if str(item).strip()),
        client_confirmed_slots=_plain_str_mapping(data.get("client_confirmed_slots")),
        crm_known_slots=_plain_str_mapping(data.get("crm_known_slots")),
        bot_inferred_slots=_plain_str_mapping(data.get("bot_inferred_slots")),
        do_not_reask_slots=tuple(str(item) for item in (data.get("do_not_reask_slots") or ()) if str(item).strip()),
        conversation_summary_short=str(data.get("conversation_summary_short") or "")[:500],
        open_loop_summary=str(data.get("open_loop_summary") or "")[:500],
    )


def next_best_action_hint(memory: DialogueMemory) -> str:
    if memory.risk_flags:
        return "handoff_required"
    if memory.open_question.text and not memory.open_question.answered:
        return f"answer_open_question:{memory.open_question.kind or 'other'}"
    slots = memory.known_slots
    if not slots.get("grade"):
        return "ask_one_question:grade"
    if not slots.get("subject") and memory.sales_stage in {"interest", "qualification"}:
        return "ask_one_question:subject"
    if not slots.get("format") and memory.sales_stage in {"qualification", "offer"}:
        return "ask_one_question:format"
    return "offer_next_step"


def safe_next_action(memory: DialogueMemory) -> Mapping[str, Any]:
    """Prompt metadata for current-terms requests; never promises booking or payment."""

    if not _needs_current_terms_action(memory):
        return {}
    base: dict[str, Any] = {
        "type": "manager_current_terms_request",
        "client_safe_text": CURRENT_TERMS_SAFE_REQUEST_TEXT,
        "requires_manager": True,
        "forbidden_promises": list(CURRENT_TERMS_FORBIDDEN_PROMISES),
        "do_not_promise": list(CURRENT_TERMS_FORBIDDEN_PROMISES_RU),
    }
    missing_slot = _first_missing_current_terms_slot(memory.known_slots)
    if not missing_slot:
        return base
    question = CURRENT_TERMS_SLOT_QUESTIONS[missing_slot]
    return {
        **base,
        "type": "ask_one_slot_for_current_terms_request",
        "missing_slot": missing_slot,
        "slot_question": question,
        "client_safe_text": f"{question} После этого передам менеджеру запрос на оформление по текущим условиям.",
    }


def _parse_recent_messages(messages: Sequence[str]) -> tuple[DialogueTurn, ...]:
    turns: list[DialogueTurn] = []
    for raw in messages:
        text = _clean(raw)
        if not text:
            continue
        lowered = text.casefold()
        if lowered.startswith(("клиент:", "user:", "client:")):
            role = "client"
            text = text.split(":", 1)[1].strip() if ":" in text else text
        elif lowered.startswith(("ответ:", "бот:", "bot:", "assistant:")):
            role = "bot"
            text = text.split(":", 1)[1].strip() if ":" in text else text
        else:
            role = "client"
        turns.append(DialogueTurn(role, text))
    return tuple(turns[-MAX_TURNS:])


def _extract_slots_from_turns(turns: Sequence[DialogueTurn]) -> Mapping[str, Any]:
    client_text = "\n".join(turn.text for turn in turns if turn.role == "client")
    return _extract_slots_from_text(client_text)


def _extract_slots_from_text(client_text: str) -> Mapping[str, Any]:
    normalized = normalize_text(client_text)
    return {
        "grade": extract_grade(normalized),
        "subject": extract_subjects(normalized),
        "format": _normalize_format(extract_format(normalized)),
        "goal": extract_goal(normalized),
        "product": extract_product(normalized),
    }


def _merge_slots(
    target: dict[str, DialogueSlot],
    source_map: Mapping[str, Any],
    *,
    source_name: str,
    confidence: float,
    override: bool = False,
) -> None:
    aliases = {
        "grade": ("grade", "class", "student_grade", "klass"),
        "subject": ("subject", "course_subject", "interest_subject"),
        "format": ("format", "course_format", "preferred_format"),
        "goal": ("goal", "learning_goal"),
        "product": ("product", "course_type"),
        "city": ("city", "city_or_location"),
        "location": ("location", "address", "site"),
        "student_name": ("student_name", "child_name"),
        "parent_name": ("parent_name", "parent_full_name"),
        "phone_known": ("phone_known",),
    }
    for normalized, keys in aliases.items():
        if not override and normalized in target and target[normalized].value:
            continue
        value = ""
        for key in keys:
            raw = source_map.get(key)
            if raw in (None, "", False):
                continue
            value = "yes" if raw is True else str(raw).strip()
            break
        if normalized == "format":
            value = _normalize_format(value)
        if value:
            target[normalized] = DialogueSlot(value=value[:160], source=source_name, confidence=confidence)


def _detect_open_question(text: str) -> DialogueQuestion:
    clean = _clean(text)
    if not clean:
        return DialogueQuestion()
    normalized = normalize_text(clean)
    is_question = (
        "?" in clean
        or _is_current_terms_question(normalized)
        or any(marker in normalized for _, markers in QUESTION_KIND_MARKERS for marker in markers)
    )
    if not is_question:
        return DialogueQuestion()
    kind = "other"
    if _is_current_terms_question(normalized):
        kind = "price_fix"
    else:
        for candidate, markers in QUESTION_KIND_MARKERS:
            if any(marker in normalized for marker in markers):
                kind = candidate
                break
    return DialogueQuestion(text=clean[:260], kind=kind, answered=False)


def _detect_risk_flags(text: str) -> tuple[str, ...]:
    normalized = normalize_text(text)
    flags = [flag for flag, markers in P0_MARKERS if any(marker in normalized for marker in markers)]
    return tuple(dict.fromkeys(flags))


def _detect_commitments(turns: Sequence[DialogueTurn]) -> tuple[str, ...]:
    commitments: list[str] = []
    for turn in turns:
        if turn.role != "bot":
            continue
        normalized = normalize_text(turn.text)
        for code, markers in COMMITMENT_MARKERS:
            if any(marker in normalized for marker in markers):
                commitments.append(code)
    return tuple(dict.fromkeys(commitments))[-8:]


def _answer_closes_question(answer: str, kind: str) -> bool:
    normalized = normalize_text(answer)
    if not answer:
        return False
    if kind == "price_fix":
        return "текущ" in normalized or "оформ" in normalized or "услов" in normalized
    if kind == "price":
        return bool(re.search(r"\d[\d\s\u00a0]{1,9}\s*(?:₽|руб)", answer, re.I)) or "зависит" in normalized
    if kind == "installment":
        return any(marker in normalized for marker in ("рассроч", "долями", "частями", "помесяч", "банк"))
    if kind == "trial":
        return "пробн" in normalized or "фрагмент" in normalized
    if kind == "identity":
        return "цифровой помощник" in normalized or "не живой оператор" in normalized
    if kind == "address":
        return any(marker in normalized for marker in ("адрес", "сретенка", "красносельск", "пацаева", "мфти"))
    return True


def _needs_current_terms_action(memory: DialogueMemory) -> bool:
    return (
        not memory.risk_flags
        and memory.active_brand in {"foton", "unpk"}
        and memory.open_question.kind == "price_fix"
        and bool(memory.open_question.text)
        and not memory.open_question.answered
    )


def _first_missing_current_terms_slot(slots: Mapping[str, DialogueSlot]) -> str:
    for key in CURRENT_TERMS_REQUIRED_SLOTS:
        slot = slots.get(key)
        if not slot or not slot.value:
            return key
    return ""


def _is_current_terms_question(normalized: str) -> bool:
    if not normalized:
        return False
    if any(marker in normalized for marker in ("мест", "брон", "заброни")):
        return False
    if any(marker in normalized for marker in ("зафикс", "закреп", "подраст", "выраст")):
        return True
    if any(
        marker in normalized
        for marker in (
            "текущая цена",
            "цена на сейчас",
            "актуальная цена",
            "актуально на сейчас",
            "по текущей цене",
            "по текущим условиям",
            "по этой цене",
        )
    ):
        return True
    if "оформ" in normalized and any(marker in normalized for marker in ("текущ", "сейчас", "цен", "услов")):
        return True
    if ("брон" in normalized or "заброни" in normalized) and "цен" in normalized:
        return True
    if any(marker in normalized for marker in ("что нужно", "что надо", "какие данные нужны")) and any(
        marker in normalized for marker in ("запис", "оформ")
    ):
        return True
    return False


def _sales_stage(
    slots: Mapping[str, DialogueSlot],
    *,
    open_question: DialogueQuestion,
    risk_flags: Sequence[str],
    handoff_state: str,
) -> str:
    if risk_flags or handoff_state == "required":
        return "handoff_required"
    if open_question.kind in {"price", "price_fix", "installment", "trial", "camp"}:
        return "offer"
    if slots.get("grade") and slots.get("subject") and slots.get("format"):
        return "offer"
    if slots.get("grade") or slots.get("subject"):
        return "qualification"
    return "interest" if open_question.text else "cold"


def _turns_from_previous(previous: Mapping[str, Any] | DialogueMemory | None) -> tuple[DialogueTurn, ...]:
    if isinstance(previous, DialogueMemory):
        return previous.turns
    if isinstance(previous, Mapping):
        return dialogue_memory_from_mapping(previous).turns
    return ()


def _slots_from_previous(previous: Mapping[str, Any] | DialogueMemory | None) -> dict[str, DialogueSlot]:
    if isinstance(previous, DialogueMemory):
        return dict(previous.known_slots)
    if isinstance(previous, Mapping):
        return dict(dialogue_memory_from_mapping(previous).known_slots)
    return {}


def _answered_questions(previous: Mapping[str, Any] | DialogueMemory | None) -> tuple[str, ...]:
    if isinstance(previous, DialogueMemory):
        return previous.answered_questions
    if isinstance(previous, Mapping):
        return tuple(str(item) for item in previous.get("answered_questions", ()) if str(item).strip())
    return ()


def _route_history(previous: Mapping[str, Any] | DialogueMemory | None) -> tuple[str, ...]:
    if isinstance(previous, DialogueMemory):
        return previous.route_history
    if isinstance(previous, Mapping):
        return tuple(str(item) for item in previous.get("route_history", ()) if str(item).strip())
    return ()


def _fact_refs(context: Mapping[str, Any]) -> tuple[str, ...]:
    refs: list[str] = []
    confirmed = context.get("confirmed_facts")
    if isinstance(confirmed, Mapping):
        refs.extend(str(key) for key in confirmed if str(key).strip())
    facts_context = context.get("facts_context")
    if isinstance(facts_context, Mapping):
        refs.extend(str(item) for item in facts_context.get("confirmed_fact_ids", ()) if str(item).strip())
    return tuple(dict.fromkeys(refs))[-12:]


def _plain_str_mapping(value: Any) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(raw)[:160] for key, raw in value.items() if str(key).strip() and str(raw or "").strip()}


def _slots_by_source(slots: Mapping[str, DialogueSlot], source_names: set[str]) -> Mapping[str, str]:
    return {
        key: slot.value
        for key, slot in slots.items()
        if slot.value and slot.source in source_names and key not in {"phone_known"}
    }


def _do_not_reask_slots(slots: Mapping[str, DialogueSlot]) -> tuple[str, ...]:
    return tuple(sorted(key for key, slot in slots.items() if slot.value))


def _topic_focus(
    slots: Mapping[str, DialogueSlot],
    *,
    open_question: DialogueQuestion,
    active_brand: str,
) -> Mapping[str, str]:
    focus: dict[str, str] = {"brand": active_brand}
    for key in ("grade", "subject", "format", "goal", "product", "city", "location"):
        slot = slots.get(key)
        if slot and slot.value:
            focus[key] = slot.value
    if open_question.kind:
        focus["question_kind"] = open_question.kind
    if open_question.kind == "camp" or normalize_text(focus.get("product", "")).find("лвш") >= 0:
        focus.setdefault("product_family", "camp")
    elif open_question.kind in {"price", "price_fix", "installment", "trial"}:
        focus.setdefault("product_family", "regular_course")
    return focus


def _unanswered_questions(
    previous: Mapping[str, Any] | DialogueMemory | None,
    *,
    open_question: DialogueQuestion,
) -> tuple[str, ...]:
    previous_items: tuple[str, ...] = ()
    if isinstance(previous, DialogueMemory):
        previous_items = previous.unanswered_questions
    elif isinstance(previous, Mapping):
        previous_items = tuple(str(item) for item in previous.get("unanswered_questions", ()) if str(item).strip())
    items = list(previous_items)
    if open_question.text and not open_question.answered and open_question.text not in items:
        items.append(open_question.text)
    return tuple(items[-6:])


def _safe_answered_parts_from_previous(previous: Mapping[str, Any] | DialogueMemory | None) -> tuple[str, ...]:
    if isinstance(previous, DialogueMemory):
        return previous.safe_answered_parts
    if isinstance(previous, Mapping):
        return tuple(str(item) for item in previous.get("safe_answered_parts", ()) if str(item).strip())
    return ()


def _bot_inferred_slots(previous: Mapping[str, Any] | DialogueMemory | None) -> Mapping[str, str]:
    if isinstance(previous, DialogueMemory):
        return dict(previous.bot_inferred_slots)
    if isinstance(previous, Mapping):
        return _plain_str_mapping(previous.get("bot_inferred_slots"))
    return {}


def _pending_manager_actions(commitments: Sequence[str]) -> tuple[str, ...]:
    actions = []
    for item in commitments:
        if item in {"manager_handoff", "check_availability", "send_material"}:
            actions.append(item)
    return tuple(dict.fromkeys(actions))[-5:]


def _safe_answered_parts(answer: str, question_kind: str) -> tuple[str, ...]:
    normalized = normalize_text(answer)
    parts: list[str] = []
    checks = (
        ("price", ("₽", "руб", "стоим", "цена", "цен")),
        ("installment", ("рассроч", "долями", "частями", "помесяч", "банк")),
        ("trial", ("пробн", "фрагмент")),
        ("address", ("адрес", "сретенка", "красносельск", "пацаева", "мфти")),
        ("transport", ("трансфер", "место сбора", "из москв")),
        ("camp_living", ("прожив", "питан", "менделеево", "лагер", "лвш")),
        ("identity", ("цифровой помощник", "не живой оператор")),
        ("availability_handoff", ("налич", "мест", "проверит менеджер")),
    )
    for code, markers in checks:
        if any(marker in normalized for marker in markers):
            parts.append(code)
    if question_kind and not parts and _answer_closes_question(answer, question_kind):
        parts.append(question_kind)
    return tuple(dict.fromkeys(parts))


def _conversation_summary_short(
    slots: Mapping[str, DialogueSlot],
    *,
    topic_focus: Mapping[str, str],
    open_question: DialogueQuestion,
) -> str:
    parts = []
    for key, label in (("grade", "класс"), ("subject", "предмет"), ("format", "формат"), ("product", "продукт")):
        slot = slots.get(key)
        if slot and slot.value:
            parts.append(f"{label}: {slot.value}")
    if topic_focus.get("product_family"):
        parts.append(f"семья продукта: {topic_focus['product_family']}")
    if open_question.kind:
        parts.append(f"последний вопрос: {open_question.kind}")
    return "; ".join(parts)[:500]


def _open_loop_summary(
    *,
    open_question: DialogueQuestion,
    risk_flags: Sequence[str],
    pending_actions: Sequence[str],
) -> str:
    if risk_flags:
        return "Есть риск-тема: ответ только через менеджера."
    if open_question.text and not open_question.answered:
        return f"Нужно сначала закрыть прямой вопрос клиента: {open_question.kind or 'other'}."
    if pending_actions:
        return "Есть обещанное менеджерское действие: " + ", ".join(pending_actions)
    return ""


def _risk_flags_from_safety(flags: Sequence[str]) -> tuple[str, ...]:
    text = normalize_text(" ".join(str(item) for item in flags))
    result: list[str] = []
    if "refund" in text or "возврат" in text:
        result.append("refund")
    if "legal" in text or "суд" in text:
        result.append("legal_threat")
    if "complaint" in text or "жалоб" in text:
        result.append("complaint")
    if "p0" in text or "high_risk" in text:
        result.append("p0")
    return tuple(dict.fromkeys(result))


def _stable_session_id(brand: str, turns: Sequence[DialogueTurn]) -> str:
    seed = "|".join([brand, *[f"{turn.role}:{turn.text[:80]}" for turn in turns[-3:]]])
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"dialogue_memory:{digest[:24]}"


def _normalize_format(value: Any) -> str:
    text = normalize_text(value)
    if "онлайн" in text or "online" in text or "дистан" in text:
        return "онлайн"
    if "очно" in text or "offline" in text or "офлайн" in text:
        return "очно"
    return str(value or "").strip()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
