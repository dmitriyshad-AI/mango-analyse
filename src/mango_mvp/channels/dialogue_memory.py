from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from mango_mvp.channels.new_lead_funnel import (
    extract_format,
    extract_goal,
    extract_grade,
    extract_product,
    extract_subjects,
    normalize_brand,
    normalize_text,
)
from mango_mvp.channels.held_state import HeldState, held_state_from_mapping, update_held
from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.p0_recall_spec import memory_risk_flags_from_text
from mango_mvp.channels.semantic_roles import tag_message_roles
from mango_mvp.channels.text_signals import has_any_marker, has_marker


DIALOGUE_MEMORY_SCHEMA_VERSION = "dialogue_memory_v2_2026_05_23"
MAX_TURNS = 20
MAX_PROMPT_TURNS = 20

QUESTION_KIND_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("live_availability", ("мест", "налич", "брон", "заброни")),
    ("payment_method", ("банковск", "перевод", "на счет", "на счёт", "по счету", "по счёту", "ежемесячный счет", "ежемесячный счёт")),
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
    ("camp", ("лвш", "лагер", "летн", "городск", "смен", "менделеево", "прожив", "питан")),
    ("platform", ("платформ", "запис", "мтс линк", "webinar", "вебинар")),
    ("identity", ("ты бот", "вы бот", "кто вы", "с кем я общаюсь", "ты gpt", "вы gpt")),
    ("off_topic", ("айфон", "iphone", "погода", "сочинение", "биткоин", "крипт")),
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
AUTONOMOUS_P0_LATCH_RELEASE_NEUTRAL_TURNS = 5
AUTONOMOUS_P0_LATCH_RELEASE_EVENT = "autonomous_neutral_p0_latch_release_5_turns"
HARD_P0_LATCH_CODES = {"legal", "legal_threat", "payment_dispute"}
MEMORY_LLM_RECOMMENDED_REASONING = "low"
MEMORY_LLM_RECOMMENDED_MODEL_CLASS = "small_fast_memory_model"
_MEMORY_LLM_SLOT_KEYS = ("grade", "subject", "format", "goal", "product", "city", "location")
_MEMORY_LLM_TOPIC_KEYS = (
    "grade",
    "subject",
    "format",
    "goal",
    "product",
    "city",
    "location",
    "product_family",
    "question_kind",
)
_MEMORY_LLM_QUESTION_KINDS = {kind for kind, _ in QUESTION_KIND_MARKERS} | {"other", "price_fix"}
_MEMORY_LLM_UNSAFE_SUMMARY_FACT_RE = re.compile(
    r"(?:₽|руб(?:\.|лей|ля|ль)?|%|\b\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"(?:\s+\d{4})?\b)",
    re.I,
)
_MEMORY_LLM_OVERRIDABLE_SLOT_SOURCES = {"dialogue_memory", "memory_llm", "bot_inferred", "unknown"}


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
class DialogueP0Latch:
    active: bool = False
    codes: tuple[str, ...] = ()
    primary_risk: str = ""
    started_at: str = ""
    trigger_turn_id: str = ""
    release_event_id: str = ""

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "active": self.active,
            "codes": list(self.codes),
            "primary_risk": self.primary_risk,
            "started_at": self.started_at,
            "trigger_turn_id": self.trigger_turn_id,
            "release_event_id": self.release_event_id,
        }


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
    p0_latch: DialogueP0Latch = field(default_factory=DialogueP0Latch)
    route_history: tuple[str, ...] = ()
    topic_focus: Mapping[str, str] = field(default_factory=dict)
    unanswered_questions: tuple[str, ...] = ()
    safe_answered_parts: tuple[str, ...] = ()
    pending_manager_actions: tuple[str, ...] = ()
    client_confirmed_slots: Mapping[str, str] = field(default_factory=dict)
    crm_known_slots: Mapping[str, str] = field(default_factory=dict)
    bot_inferred_slots: Mapping[str, str] = field(default_factory=dict)
    do_not_reask_slots: tuple[str, ...] = ()
    held_state: HeldState = field(default_factory=HeldState)
    current_message_roles: Mapping[str, Any] = field(default_factory=dict)
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
            "p0_latch": self.p0_latch.to_json_dict(),
            "route_history": list(self.route_history),
            "topic_focus": dict(self.topic_focus),
            "unanswered_questions": list(self.unanswered_questions),
            "safe_answered_parts": list(self.safe_answered_parts),
            "pending_manager_actions": list(self.pending_manager_actions),
            "client_confirmed_slots": dict(self.client_confirmed_slots),
            "crm_known_slots": dict(self.crm_known_slots),
            "bot_inferred_slots": dict(self.bot_inferred_slots),
            "do_not_reask_slots": list(self.do_not_reask_slots),
            "held_state": self.held_state.to_json_dict(),
            "current_message_roles": dict(self.current_message_roles),
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
            "p0_latch": self.p0_latch.to_json_dict(),
            "route_history": list(self.route_history[-5:]),
            "topic_focus": dict(self.topic_focus),
            "unanswered_questions": list(self.unanswered_questions[-5:]),
            "safe_answered_parts": list(self.safe_answered_parts[-8:]),
            "pending_manager_actions": list(self.pending_manager_actions[-5:]),
            "client_confirmed_slots": dict(self.client_confirmed_slots),
            "crm_known_slots": dict(self.crm_known_slots),
            "bot_inferred_slots": dict(self.bot_inferred_slots),
            "do_not_ask_again": list(self.do_not_reask_slots) or sorted(known_values),
            "held_state": self.held_state.to_prompt_view(),
            "current_message_roles": dict(self.current_message_roles),
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
    previous = dialogue_memory_from_mapping(previous_memory) if isinstance(previous_memory, Mapping) else previous_memory
    if not open_question.text:
        if isinstance(previous, DialogueMemory) and previous.open_question.text and not previous.open_question.answered:
            open_question = previous.open_question
    previous_held = previous.held_state if isinstance(previous, DialogueMemory) else HeldState()
    current_roles = tag_message_roles(current_text, context=previous_held.tagger_context())
    current_risk_flags = _detect_risk_flags(current_text)
    held_p0_required = bool(previous_held.p0_latched or current_risk_flags or current_roles.refund_frame == "dispute")
    held_state = update_held(previous_held, current_text, current_roles, p0_required=held_p0_required)
    p0_latch = _next_p0_latch(
        previous.p0_latch if isinstance(previous, DialogueMemory) else DialogueP0Latch(),
        current_message=current_text,
        current_risk_flags=current_risk_flags,
        context=context,
        session_id=session_id,
        turns=turns,
    )
    latch_released = _p0_latch_released(previous.p0_latch if isinstance(previous, DialogueMemory) else DialogueP0Latch(), p0_latch)
    if latch_released:
        held_state = replace(held_state, p0_latched=False, p0_codes=())
    risks = current_risk_flags if latch_released else _detect_risk_flags("\n".join(turn.text for turn in turns if turn.role == "client"))
    if p0_latch.active:
        risks = tuple(dict.fromkeys([*risks, *p0_latch.codes, "p0"]))
    commitments = _detect_commitments(turns)
    fact_refs = _fact_refs(context or {})
    route_history = _route_history(previous_memory)
    handoff = "required" if p0_latch.active or risks else ("suggested" if any("manager" in item for item in commitments) else "none")
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
        p0_latch=p0_latch,
        route_history=route_history,
        topic_focus=topic_focus,
        unanswered_questions=unanswered,
        safe_answered_parts=_safe_answered_parts_from_previous(previous_memory),
        pending_manager_actions=_pending_manager_actions(commitments),
        client_confirmed_slots=client_confirmed,
        crm_known_slots=crm_known,
        bot_inferred_slots=_bot_inferred_slots(previous_memory),
        do_not_reask_slots=_do_not_reask_slots(slot_map),
        held_state=held_state,
        current_message_roles=current_roles.to_prompt_view(),
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
    memory_llm_fn: Callable[[str], object] | None = None,
) -> DialogueMemory:
    current = memory if isinstance(memory, DialogueMemory) else dialogue_memory_from_mapping(memory)
    answer = _clean(answer_text)
    turns = current.turns
    if answer:
        turns = (*turns, DialogueTurn("bot", answer))[-MAX_TURNS:]
    route_history = tuple(dict.fromkeys([*current.route_history, str(route or "").strip()]))[-8:]
    safety_risks = _risk_flags_from_safety(safety_flags)
    risks = tuple(dict.fromkeys([*current.risk_flags, *safety_risks]))
    p0_latch = _next_p0_latch(
        current.p0_latch,
        current_message="",
        current_risk_flags=safety_risks,
        context={"route": route, "safety_flags": list(safety_flags)},
        session_id=current.session_id,
        turns=turns,
    )
    if p0_latch.active:
        risks = tuple(dict.fromkeys([*risks, *p0_latch.codes, "p0"]))
    commitments = tuple(dict.fromkeys([*current.last_bot_commitments, *_detect_commitments(turns)]))[-8:]
    answered = current.answered_questions
    open_question = current.open_question
    unanswered = tuple(current.unanswered_questions)
    if open_question.text and _answer_closes_question(answer, open_question.kind):
        answered = (*answered, open_question.text)[-8:]
        open_question = DialogueQuestion(open_question.text, open_question.kind, True)
        unanswered = tuple(item for item in unanswered if item != current.open_question.text)
    handoff = "required" if p0_latch.active or risks or route == "manager_only" else current.handoff_state
    safe_parts = tuple(dict.fromkeys([*current.safe_answered_parts, *_safe_answered_parts(answer, current.open_question.kind)]))[-12:]
    pending_actions = _pending_manager_actions(commitments)
    updated = DialogueMemory(
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
        p0_latch=p0_latch,
        route_history=route_history,
        topic_focus=dict(current.topic_focus),
        unanswered_questions=unanswered,
        safe_answered_parts=safe_parts,
        pending_manager_actions=pending_actions,
        client_confirmed_slots=dict(current.client_confirmed_slots),
        crm_known_slots=dict(current.crm_known_slots),
        bot_inferred_slots=dict(current.bot_inferred_slots),
        do_not_reask_slots=tuple(current.do_not_reask_slots),
        held_state=current.held_state,
        current_message_roles=dict(current.current_message_roles),
        conversation_summary_short=current.conversation_summary_short,
        open_loop_summary=_open_loop_summary(open_question=open_question, risk_flags=risks, pending_actions=pending_actions),
    )
    if memory_llm_fn is None:
        return updated
    try:
        payload = update_memory_llm(updated.turns[-8:], updated, memory_llm_fn=memory_llm_fn)
    except Exception:
        return updated
    return _apply_memory_llm_update(updated, payload)


def update_memory_llm(
    recent_turns: Sequence[DialogueTurn | Mapping[str, Any]],
    prev_memory: DialogueMemory | Mapping[str, Any],
    *,
    memory_llm_fn: Callable[[str], object] | None,
) -> Mapping[str, Any]:
    """Optional post-answer memory enrichment. Callers should use a low-effort small model."""

    if memory_llm_fn is None:
        return {}
    memory = prev_memory if isinstance(prev_memory, DialogueMemory) else dialogue_memory_from_mapping(prev_memory)
    coerced_turns: list[DialogueTurn] = []
    for item in recent_turns:
        turn = _coerce_turn(item)
        if turn.text:
            coerced_turns.append(turn)
    turns = tuple(coerced_turns[-8:])
    prompt = build_memory_llm_prompt(turns, memory)
    raw = memory_llm_fn(prompt)
    if isinstance(raw, Mapping):
        return raw
    return _extract_json_object(raw)


def build_memory_llm_prompt(recent_turns: Sequence[DialogueTurn], prev_memory: DialogueMemory) -> str:
    turns_payload = [turn.to_json_dict() for turn in recent_turns[-8:]]
    memory_payload = prev_memory.to_prompt_view()
    return (
        "Ты обновляешь краткую рабочую память Telegram-диалога ПОСЛЕ ответа бота.\n"
        f"Режим вызова: {MEMORY_LLM_RECOMMENDED_REASONING} reasoning; использовать отдельную мелкую/быструю модель "
        f"класса {MEMORY_LLM_RECOMMENDED_MODEL_CLASS}, не основную модель ответа.\n"
        "Задача: извлеки только явно сказанное в последних репликах и в прошлой памяти. Не выдумывай.\n"
        "Бренд задаёт канал: active_brand менять нельзя, даже если клиент назвал другой бренд.\n"
        "Верни строгий JSON без markdown:\n"
        "{\n"
        '  "slots": {"grade": "", "subject": "", "format": "", "goal": "", "product": "", "city": "", "location": ""},\n'
        '  "topic": {"grade": "", "subject": "", "format": "", "goal": "", "product": "", "product_family": "", "question_kind": ""},\n'
        '  "open_question": {"text": "", "kind": "", "answered": false},\n'
        '  "commitments": [],\n'
        '  "summary": ""\n'
        "}\n"
        "Правила:\n"
        "- slots/topic заполняй предметом, классом, форматом, продуктом, городом/площадкой только если это явно следует из диалога.\n"
        "- open_question — последний незакрытый вопрос клиента, если он есть.\n"
        "- commitments — только обещания бота: передать менеджеру, проверить наличие, прислать материал.\n"
        "- summary — 1-2 смысловые фразы: кто клиент, что обсуждали, на чём остановились; "
        "включи безопасное обещание бота, если оно уже было. Без цен/дат/обещаний, если они не звучали явно.\n\n"
        "ПРЕДЫДУЩАЯ ПАМЯТЬ JSON:\n"
        f"{json.dumps(memory_payload, ensure_ascii=False, sort_keys=True)}\n\n"
        "ПОСЛЕДНИЕ РЕПЛИКИ JSON:\n"
        f"{json.dumps(turns_payload, ensure_ascii=False, sort_keys=True)}"
    )


def _apply_memory_llm_update(memory: DialogueMemory, payload: Mapping[str, Any]) -> DialogueMemory:
    if not isinstance(payload, Mapping) or not payload:
        return memory

    slots = dict(memory.known_slots)
    llm_slots = _memory_llm_slots(payload.get("slots"))
    _merge_memory_llm_slots(slots, llm_slots, memory=memory)

    open_question = _memory_llm_open_question(payload.get("open_question"), fallback=memory.open_question)
    topic_focus = _memory_llm_topic(payload.get("topic"), slots=slots, open_question=open_question, memory=memory)
    commitments = _memory_llm_commitments(payload.get("commitments"), previous=memory.last_bot_commitments)
    summary = _memory_llm_summary(payload.get("summary")) or memory.conversation_summary_short

    client_confirmed = _slots_by_source(slots, {"dialogue_memory"})
    bot_inferred = {
        **dict(memory.bot_inferred_slots),
        **_slots_by_source(slots, {"memory_llm"}),
    }
    return replace(
        memory,
        known_slots=slots,
        open_question=open_question,
        last_bot_commitments=commitments,
        sales_stage=_sales_stage(slots, open_question=open_question, risk_flags=memory.risk_flags, handoff_state=memory.handoff_state),
        topic_focus=topic_focus,
        client_confirmed_slots=client_confirmed,
        bot_inferred_slots=bot_inferred,
        do_not_reask_slots=_do_not_reask_slots(slots),
        pending_manager_actions=_pending_manager_actions(commitments),
        conversation_summary_short=summary[:500],
        open_loop_summary=_open_loop_summary(
            open_question=open_question,
            risk_flags=memory.risk_flags,
            pending_actions=_pending_manager_actions(commitments),
        ),
    )


def _memory_llm_slots(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key in _MEMORY_LLM_SLOT_KEYS:
        raw = value.get(key)
        text = _clean(raw)
        if text:
            result[key] = text
    return result


def _merge_memory_llm_slots(
    target: dict[str, DialogueSlot],
    source_map: Mapping[str, Any],
    *,
    memory: DialogueMemory,
) -> None:
    latest_client_text = _latest_client_text(memory.turns)
    for key, raw in source_map.items():
        if key not in _MEMORY_LLM_SLOT_KEYS:
            continue
        value = _clean(raw)
        if key == "format":
            value = _normalize_format(value)
        if not value:
            continue
        existing = target.get(key)
        if existing and not _memory_llm_can_override_slot(key, value, existing, latest_client_text=latest_client_text):
            continue
        target[key] = DialogueSlot(value=value[:160], source="memory_llm", confidence=0.74)


def _memory_llm_can_override_slot(
    key: str,
    value: str,
    existing: DialogueSlot,
    *,
    latest_client_text: str,
) -> bool:
    if not existing.value:
        return True
    if existing.source not in _MEMORY_LLM_OVERRIDABLE_SLOT_SOURCES:
        return False
    if existing.source == "dialogue_memory":
        return _memory_llm_slot_supported_by_latest_client(key, value, latest_client_text=latest_client_text)
    return True


def _memory_llm_slot_supported_by_latest_client(key: str, value: str, *, latest_client_text: str) -> bool:
    normalized = normalize_text(latest_client_text)
    candidate = normalize_text(value)
    if not normalized or not candidate:
        return False
    if key == "grade":
        digits = re.findall(r"\d{1,2}", candidate)
        if digits:
            grade = digits[0]
            ordinal_stems = {
                "1": ("перв",),
                "2": ("втор",),
                "3": ("трет",),
                "4": ("четвер",),
                "5": ("пят",),
                "6": ("шест",),
                "7": ("седьм", "седм"),
                "8": ("восьм",),
                "9": ("девят",),
                "10": ("десят",),
                "11": ("одиннадцат",),
            }.get(grade, ())
            return bool(
                re.search(rf"\b{re.escape(grade)}\s*(?:класс|кл\.?)?\b", normalized, re.I)
                or any(stem in normalized for stem in ordinal_stems)
            )
    if key == "subject":
        aliases = {
            "информатика": ("информат", "айти", "it", "программ"),
            "математика": ("математ", "матем"),
            "физика": ("физик",),
            "химия": ("хими",),
            "биология": ("биолог",),
            "русский": ("русск",),
            "английский": ("англ",),
        }
        return any(alias in normalized for alias in aliases.get(candidate, (candidate,)))
    if key == "format":
        if candidate == "онлайн":
            return has_any_marker(normalized, ("онлайн", "online", "дистанц", "удален", "удалён"))
        if candidate == "очно":
            return has_any_marker(normalized, ("очно", "офлайн", "offline", "площадк", "адрес"))
    if candidate in normalized:
        return True
    return False


def _latest_client_text(turns: Sequence[DialogueTurn]) -> str:
    for turn in reversed(turns):
        if turn.role == "client" and turn.text:
            return turn.text
    return ""


def _memory_llm_topic(
    value: Any,
    *,
    slots: Mapping[str, DialogueSlot],
    open_question: DialogueQuestion,
    memory: DialogueMemory,
) -> Mapping[str, str]:
    topic = dict(_topic_focus(slots, open_question=open_question, active_brand=memory.active_brand))
    if isinstance(value, Mapping):
        for key in _MEMORY_LLM_TOPIC_KEYS:
            raw = _clean(value.get(key))
            if raw and not topic.get(key):
                topic[key] = raw[:160]
    topic["brand"] = memory.active_brand
    return topic


def _memory_llm_open_question(value: Any, *, fallback: DialogueQuestion) -> DialogueQuestion:
    if not isinstance(value, Mapping):
        return fallback
    text = _clean(value.get("text"))[:260]
    if not text:
        return fallback
    kind = _clean(value.get("kind"))
    if kind not in _MEMORY_LLM_QUESTION_KINDS:
        kind = fallback.kind or "other"
    answered = bool(value.get("answered"))
    if fallback.text and fallback.answered and text == fallback.text:
        return fallback
    return DialogueQuestion(text=text, kind=kind, answered=answered)


def _memory_llm_commitments(value: Any, *, previous: Sequence[str]) -> tuple[str, ...]:
    items = list(previous)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for raw in value:
            text = _clean(raw)
            if text:
                items.append(text[:120])
    return tuple(dict.fromkeys(items))[-8:]


def _memory_llm_summary(value: Any) -> str:
    text = _clean(value)
    if not text:
        return ""
    if _MEMORY_LLM_UNSAFE_SUMMARY_FACT_RE.search(text):
        return ""
    return text[:500]


def _coerce_turn(raw: DialogueTurn | Mapping[str, Any]) -> DialogueTurn:
    if isinstance(raw, DialogueTurn):
        return raw
    if isinstance(raw, Mapping):
        return DialogueTurn(str(raw.get("role") or raw.get("speaker") or ""), _clean(raw.get("text")))
    return DialogueTurn("", _clean(raw))


def _extract_json_object(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raw = str(value or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            payload = json.loads(raw[start : end + 1])
        except Exception:
            return {}
    return payload if isinstance(payload, Mapping) else {}


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
        p0_latch=_p0_latch_from_mapping(data.get("p0_latch")),
        route_history=tuple(str(item) for item in (data.get("route_history") or ()) if str(item).strip()),
        topic_focus=_plain_str_mapping(data.get("topic_focus")),
        unanswered_questions=tuple(str(item) for item in (data.get("unanswered_questions") or ()) if str(item).strip()),
        safe_answered_parts=tuple(str(item) for item in (data.get("safe_answered_parts") or ()) if str(item).strip()),
        pending_manager_actions=tuple(str(item) for item in (data.get("pending_manager_actions") or ()) if str(item).strip()),
        client_confirmed_slots=_plain_str_mapping(data.get("client_confirmed_slots")),
        crm_known_slots=_plain_str_mapping(data.get("crm_known_slots")),
        bot_inferred_slots=_plain_str_mapping(data.get("bot_inferred_slots")),
        do_not_reask_slots=tuple(str(item) for item in (data.get("do_not_reask_slots") or ()) if str(item).strip()),
        held_state=held_state_from_mapping(data.get("held_state") if isinstance(data.get("held_state"), Mapping) else {}),
        current_message_roles=dict(data.get("current_message_roles") or {}) if isinstance(data.get("current_message_roles"), Mapping) else {},
        conversation_summary_short=str(data.get("conversation_summary_short") or "")[:500],
        open_loop_summary=str(data.get("open_loop_summary") or "")[:500],
    )


def next_best_action_hint(memory: DialogueMemory) -> str:
    if memory.p0_latch.active or memory.risk_flags:
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
        or any(has_any_marker(normalized, markers) for _, markers in QUESTION_KIND_MARKERS)
    )
    if not is_question:
        return DialogueQuestion()
    kind = "other"
    if _is_current_terms_question(normalized):
        kind = "price_fix"
    else:
        for candidate, markers in QUESTION_KIND_MARKERS:
            if has_any_marker(normalized, markers):
                kind = candidate
                break
    return DialogueQuestion(text=clean[:260], kind=kind, answered=False)


def _detect_risk_flags(text: str) -> tuple[str, ...]:
    return memory_risk_flags_from_text(text)


def _detect_commitments(turns: Sequence[DialogueTurn]) -> tuple[str, ...]:
    commitments: list[str] = []
    for turn in turns:
        if turn.role != "bot":
            continue
        normalized = normalize_text(turn.text)
        for code, markers in COMMITMENT_MARKERS:
            if has_any_marker(normalized, markers):
                commitments.append(code)
    return tuple(dict.fromkeys(commitments))[-8:]


def _answer_closes_question(answer: str, kind: str) -> bool:
    normalized = normalize_text(answer)
    if not answer:
        return False
    if kind == "price_fix":
        return has_any_marker(normalized, ("текущ", "оформ", "услов"))
    if kind == "price":
        return bool(re.search(r"\d[\d\s\u00a0]{1,9}\s*(?:₽|руб)", answer, re.I)) or has_marker(normalized, "зависит")
    if kind == "installment":
        return has_any_marker(normalized, ("рассроч", "долями", "частями", "помесяч", "банк"))
    if kind == "trial":
        return has_any_marker(normalized, ("пробн", "фрагмент"))
    if kind == "identity":
        return has_any_marker(normalized, ("цифровой помощник", "не живой оператор"))
    if kind == "address":
        return has_any_marker(normalized, ("адрес", "сретенка", "красносельск", "пацаева", "мфти"))
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
    if has_any_marker(normalized, ("мест", "брон", "заброни")):
        return False
    if has_any_marker(normalized, ("зафикс", "закреп", "подраст", "выраст")):
        return True
    if has_any_marker(
        normalized,
        (
            "текущая цена",
            "цена на сейчас",
            "актуальная цена",
            "актуально на сейчас",
            "по текущей цене",
            "по текущим условиям",
            "по этой цене",
        ),
    ):
        return True
    if has_marker(normalized, "оформ") and has_any_marker(normalized, ("текущ", "сейчас", "цен", "услов")):
        return True
    if has_any_marker(normalized, ("брон", "заброни")) and has_marker(normalized, "цен"):
        return True
    if has_any_marker(normalized, ("что нужно", "что надо", "какие данные нужны")) and has_any_marker(normalized, ("запис", "оформ")):
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


def _p0_latch_from_mapping(value: Any) -> DialogueP0Latch:
    if not isinstance(value, Mapping):
        return DialogueP0Latch()
    return DialogueP0Latch(
        active=bool(value.get("active")),
        codes=tuple(str(item) for item in (value.get("codes") or ()) if str(item).strip()),
        primary_risk=str(value.get("primary_risk") or ""),
        started_at=str(value.get("started_at") or ""),
        trigger_turn_id=str(value.get("trigger_turn_id") or ""),
        release_event_id=str(value.get("release_event_id") or ""),
    )


def _next_p0_latch(
    previous: DialogueP0Latch,
    *,
    current_message: str,
    current_risk_flags: Sequence[str],
    context: Mapping[str, Any] | None,
    session_id: str,
    turns: Sequence[DialogueTurn] = (),
) -> DialogueP0Latch:
    release_event = _p0_latch_release_event(context)
    if release_event:
        result = DialogueP0Latch(release_event_id=release_event)
        trace_event(
            context,
            "_next_p0_latch",
            {
                "previous_active": previous.active,
                "previous_codes": list(previous.codes),
                "current_risk_flags": list(current_risk_flags),
                "release_event": release_event,
                "autonomous_release": False,
                "next_active": result.active,
                "next_codes": list(result.codes),
            },
        )
        return result
    if previous.active:
        autonomous_release = _autonomous_p0_latch_release_event(
            previous,
            turns=turns,
            current_risk_flags=current_risk_flags,
        )
        if autonomous_release:
            result = DialogueP0Latch(release_event_id=autonomous_release)
            trace_event(
                context,
                "_next_p0_latch",
                {
                    "previous_active": previous.active,
                    "previous_codes": list(previous.codes),
                    "current_risk_flags": list(current_risk_flags),
                    "release_event": autonomous_release,
                    "autonomous_release": True,
                    "next_active": result.active,
                    "next_codes": list(result.codes),
                },
            )
            return result
        trace_event(
            context,
            "_next_p0_latch",
            {
                "previous_active": previous.active,
                "previous_codes": list(previous.codes),
                "current_risk_flags": list(current_risk_flags),
                "release_event": "",
                "autonomous_release": False,
                "next_active": previous.active,
                "next_codes": list(previous.codes),
            },
        )
        return previous
    codes = tuple(dict.fromkeys(_latchable_p0_codes(current_risk_flags)))
    if not codes:
        result = DialogueP0Latch()
        trace_event(
            context,
            "_next_p0_latch",
            {
                "previous_active": previous.active,
                "previous_codes": list(previous.codes),
                "current_risk_flags": list(current_risk_flags),
                "release_event": "",
                "autonomous_release": False,
                "next_active": result.active,
                "next_codes": list(result.codes),
            },
        )
        return result
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    trigger_seed = f"{session_id}|{current_message[:160]}|{','.join(codes)}"
    trigger_id = hashlib.sha256(trigger_seed.encode("utf-8", errors="ignore")).hexdigest()[:16]
    result = DialogueP0Latch(
        active=True,
        codes=codes,
        primary_risk=_primary_p0_risk(codes),
        started_at=now,
        trigger_turn_id=trigger_id,
    )
    trace_event(
        context,
        "_next_p0_latch",
        {
            "previous_active": previous.active,
            "previous_codes": list(previous.codes),
            "current_risk_flags": list(current_risk_flags),
            "release_event": "",
            "autonomous_release": False,
            "next_active": result.active,
            "next_codes": list(result.codes),
        },
    )
    return result


def _latchable_p0_codes(flags: Sequence[str]) -> tuple[str, ...]:
    mapping = {
        "refund": "refund",
        "legal": "legal_threat",
        "legal_threat": "legal_threat",
        "complaint": "complaint",
        "reputation_threat": "complaint",
        "payment_dispute": "payment_dispute",
    }
    return tuple(mapping[str(item)] for item in flags if str(item) in mapping)


def _primary_p0_risk(codes: Sequence[str]) -> str:
    for code in ("legal_threat", "refund", "complaint", "payment_dispute"):
        if code in codes:
            return code
    return str(codes[0]) if codes else ""


def _p0_latch_release_event(context: Mapping[str, Any] | None) -> str:
    if not isinstance(context, Mapping):
        return ""
    for key in ("manager_clear_p0_latch", "manager_resolved_p0", "manager_took_over", "p0_latch_release_event"):
        value = context.get(key)
        if value:
            return str(value if not isinstance(value, bool) else key)[:120]
    return ""


def _autonomous_p0_latch_release_event(
    previous: DialogueP0Latch,
    *,
    turns: Sequence[DialogueTurn],
    current_risk_flags: Sequence[str],
) -> str:
    if not previous.active or _has_hard_p0_latch_code(previous.codes):
        return ""
    if _latchable_p0_codes(current_risk_flags):
        return ""
    client_texts = [turn.text for turn in turns if turn.role == "client"]
    recent_client_texts = client_texts[-AUTONOMOUS_P0_LATCH_RELEASE_NEUTRAL_TURNS:]
    if len(recent_client_texts) < AUTONOMOUS_P0_LATCH_RELEASE_NEUTRAL_TURNS:
        return ""
    if any(_latchable_p0_codes(_detect_risk_flags(text)) for text in recent_client_texts):
        return ""
    return AUTONOMOUS_P0_LATCH_RELEASE_EVENT


def _has_hard_p0_latch_code(codes: Sequence[str]) -> bool:
    return any(str(code or "").strip() in HARD_P0_LATCH_CODES for code in codes)


def _p0_latch_released(previous: DialogueP0Latch, current: DialogueP0Latch) -> bool:
    return bool(previous.active and not current.active and current.release_event_id)


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
        if has_any_marker(normalized, markers):
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
    benign_presale_refund = "presale_refund_policy" in text and "zero_collect_refund" not in text and "final_p0_text_override" not in text
    result: list[str] = []
    if not benign_presale_refund and ("refund" in text or "возврат" in text):
        result.append("refund")
    if "legal" in text or "суд" in text:
        result.append("legal_threat")
    if "complaint" in text or "жалоб" in text:
        result.append("complaint")
    if "payment_dispute" in text or "спор по оплат" in text:
        result.append("payment_dispute")
    if "conversation_intent_plan_p0" in text or "final_p0_text_override" in text:
        result.append("p0")
    return tuple(dict.fromkeys(result))


def _stable_session_id(brand: str, turns: Sequence[DialogueTurn]) -> str:
    seed = "|".join([brand, *[f"{turn.role}:{turn.text[:80]}" for turn in turns[-3:]]])
    digest = hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()
    return f"dialogue_memory:{digest[:24]}"


def _normalize_format(value: Any) -> str:
    text = normalize_text(value)
    if has_any_marker(text, ("онлайн", "online", "дистан")):
        return "онлайн"
    if has_any_marker(text, ("очно", "offline", "офлайн")):
        return "очно"
    return str(value or "").strip()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())
