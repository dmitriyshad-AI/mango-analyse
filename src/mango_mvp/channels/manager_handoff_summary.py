from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from mango_mvp.channels.new_lead_funnel import LeadFunnelState


MANAGER_HANDOFF_SUMMARY_SCHEMA_VERSION = "manager_handoff_summary_v1_2026_05_23"

INTERNAL_TOKEN_RE = re.compile(
    r"\b(?:AMO|Tallanto|CRM|source_id|source:|lead_id|contact_id|token|api[_-]?key|debug_impersonation)\b",
    re.I,
)

P0_ZERO_COLLECT_WARNING = (
    "Не просить в клиентском чате ФИО, номер договора, телефон, email, сумму, причину возврата или платёжные данные; "
    "передать ответственному сотруднику."
)


def build_manager_handoff_summary(
    *,
    brand: str,
    client_message: str,
    answer_text: str,
    route: str,
    topic_id: str = "",
    risk_level: str = "",
    safety_flags: Sequence[str] = (),
    missing_facts: Sequence[str] = (),
    manager_checklist: Sequence[str] = (),
    funnel_state: LeadFunnelState | Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> str:
    state = funnel_state.to_json_dict() if isinstance(funnel_state, LeadFunnelState) else dict(funnel_state or {})
    filled_slots = state.get("filled_slots") if isinstance(state.get("filled_slots"), Mapping) else {}
    missing_slots = state.get("missing_slots") if isinstance(state.get("missing_slots"), Sequence) and not isinstance(state.get("missing_slots"), str) else ()
    safety = tuple(str(item) for item in safety_flags if str(item).strip())
    checks = [str(item).strip() for item in manager_checklist if str(item).strip()]
    facts = [str(item).strip() for item in missing_facts if str(item).strip()]
    p0 = is_p0_handoff(topic_id=topic_id, route=route, risk_level=risk_level, safety_flags=safety, state=state)

    lines = [
        f"Бренд: {brand_label(brand)}",
        f"Маршрут: {clean(route) or 'не указан'}",
        f"Категория риска: {clean(risk_level) or 'не указана'}",
        f"Вопрос клиента: {clean(client_message, limit=700)}",
        f"Что уже известно: {render_slots(filled_slots)}",
        f"Недостающие поля: {render_list(missing_slots) or 'нет явных'}",
        f"Что ответили клиенту: {clean(answer_text, limit=700)}",
        f"Что нужно проверить: {render_list([*facts, *checks]) or 'проверить следующий шаг по ситуации'}",
        f"Рекомендуемый следующий шаг: {clean(state.get('next_best_question') or next_step_label(state.get('next_step_type')), limit=240) or 'обработать вручную'}",
        f"Что нельзя обещать: {P0_ZERO_COLLECT_WARNING if p0 else default_forbidden_promises()}",
    ]
    summary = "\n".join(lines)
    return redact_internal_tokens(summary)


def manager_handoff_metadata(summary: str, *, funnel_state: LeadFunnelState | Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    state = funnel_state.to_json_dict() if isinstance(funnel_state, LeadFunnelState) else dict(funnel_state or {})
    return {
        "schema_version": MANAGER_HANDOFF_SUMMARY_SCHEMA_VERSION,
        "manager_summary": summary,
        "next_step_type": str(state.get("next_step_type") or ""),
        "lead_stage": str(state.get("lead_stage") or ""),
        "client_segment": str(state.get("client_segment") or ""),
        "missing_slots": list(state.get("missing_slots") or []),
        "filled_slots": dict(state.get("filled_slots") or {}),
    }


def is_p0_handoff(
    *,
    topic_id: str,
    route: str,
    risk_level: str,
    safety_flags: Sequence[str],
    state: Mapping[str, Any],
) -> bool:
    text = " ".join([topic_id, route, risk_level, " ".join(safety_flags), json.dumps(state, ensure_ascii=False)]).casefold()
    return any(
        marker in text
        for marker in (
            "refund",
            "complaint",
            "legal",
            "p0",
            "возврат",
            "жалоб",
            "суд",
            "прокурат",
            "manager_only_p0",
        )
    )


def render_slots(slots: Mapping[str, Any]) -> str:
    public_keys = {
        "grade": "класс",
        "subject": "предмет",
        "format": "формат",
        "city": "город",
        "location": "площадка",
        "goal": "цель",
        "product": "продукт",
        "camp_direction": "лагерь",
        "shift": "смена",
        "student_name": "ученик",
        "parent_name": "родитель",
        "phone_known": "контакт в Telegram/контексте есть",
    }
    parts: list[str] = []
    for key, label in public_keys.items():
        value = slots.get(key)
        if value not in ("", None, False):
            parts.append(label if value is True else f"{label}: {clean(value, limit=120)}")
    return ", ".join(parts) if parts else "нет явных данных"


def render_list(items: Any) -> str:
    if isinstance(items, str):
        values = [items]
    elif isinstance(items, Sequence):
        values = [str(item).strip() for item in items if str(item).strip()]
    else:
        values = []
    return "; ".join(clean(item, limit=180) for item in values[:8])


def next_step_label(value: Any) -> str:
    mapping = {
        "ask_grade": "уточнить класс ребёнка",
        "ask_subject": "уточнить предмет или направление",
        "ask_format": "уточнить онлайн или очный формат",
        "ask_goal": "уточнить цель обучения",
        "ask_camp_class": "уточнить класс ребёнка для лагеря",
        "offer_group_check": "проверить подходящую группу",
        "offer_manager_seat_check": "проверить наличие мест",
        "offer_online_fragment": "предложить фрагмент занятия",
        "offer_manager_check": "проверить вопрос менеджером",
        "manager_only_p0": "обработать P0 без сбора данных в чате",
    }
    return mapping.get(str(value or ""), str(value or ""))


def default_forbidden_promises() -> str:
    return "не обещать место, скидку, расписание, срок связи, возврат, оплату или документ без проверенного факта."


def brand_label(brand: str) -> str:
    normalized = str(brand or "").strip().casefold()
    if normalized == "foton":
        return "Фотон"
    if normalized == "unpk":
        return "УНПК МФТИ"
    return "неизвестен"


def redact_internal_tokens(text: str) -> str:
    cleaned = INTERNAL_TOKEN_RE.sub("[internal]", str(text or ""))
    cleaned = cleaned.replace("{", "(").replace("}", ")")
    return cleaned.strip()


def clean(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]
