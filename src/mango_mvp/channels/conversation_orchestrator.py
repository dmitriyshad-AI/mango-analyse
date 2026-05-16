from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


CONVERSATION_ORCHESTRATOR_SCHEMA_VERSION = "conversation_orchestrator_v1"

P0_REFUND_THEME = "theme:009_refund"
P0_NEGATIVE_FEEDBACK_THEME = "theme:019b_negative_feedback"
P0_LEGAL_THEME = "theme:029_legal_question"

P0_THEME_ROUTING_RULES: Mapping[str, Mapping[str, Any]] = {
    P0_REFUND_THEME: {
        "risk_code": "p0_refund_or_training_refusal",
        "priority": "P0",
        "handoff_target": "Менеджер группы / РОП",
        "reason": "Возврат денег или отказ от обучения нельзя решать автономно.",
        "notify_rop": True,
    },
    P0_NEGATIVE_FEEDBACK_THEME: {
        "risk_code": "p0_negative_feedback_or_conflict",
        "priority": "P0",
        "handoff_target": "Менеджер группы / РОП",
        "reason": "Негативная обратная связь, жалоба или конфликт требуют ручной реакции.",
        "notify_rop": True,
    },
    P0_LEGAL_THEME: {
        "risk_code": "p0_legal_question",
        "priority": "P0",
        "handoff_target": "Менеджер группы / РОП / юрист",
        "reason": "Юридический вопрос нельзя трактовать и закрывать ботом.",
        "notify_rop": True,
    },
}


@dataclass(frozen=True)
class ConversationRoutingDecision:
    theme_id: str
    priority: str
    route_type: str
    handoff_target: str
    reason: str
    notify_rop: bool
    bot_may_answer: bool
    metadata: Mapping[str, Any]

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": CONVERSATION_ORCHESTRATOR_SCHEMA_VERSION,
            **asdict(self),
        }


def route_question_catalog_theme(theme_id: str, *, context: Mapping[str, Any] | None = None) -> ConversationRoutingDecision:
    normalized_theme_id = str(theme_id or "").strip()
    context_payload = dict(context or {})
    rule = P0_THEME_ROUTING_RULES.get(normalized_theme_id)
    if rule:
        return ConversationRoutingDecision(
            theme_id=normalized_theme_id,
            priority=str(rule["priority"]),
            route_type="manager_handoff_p0",
            handoff_target=str(rule["handoff_target"]),
            reason=str(rule["reason"]),
            notify_rop=bool(rule["notify_rop"]),
            bot_may_answer=False,
            metadata={
                "risk_code": rule["risk_code"],
                "source": "question_catalog_theme",
                "context_keys": tuple(sorted(str(key) for key in context_payload.keys())),
            },
        )
    return ConversationRoutingDecision(
        theme_id=normalized_theme_id,
        priority=str(context_payload.get("priority") or "normal"),
        route_type="standard_policy",
        handoff_target=str(context_payload.get("handoff_target") or ""),
        reason="Нет P0-правила маршрутизации для темы.",
        notify_rop=False,
        bot_may_answer=True,
        metadata={
            "source": "question_catalog_theme",
            "context_keys": tuple(sorted(str(key) for key in context_payload.keys())),
        },
    )


def conversation_orchestrator_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": CONVERSATION_ORCHESTRATOR_SCHEMA_VERSION,
        "live_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "routing_only": True,
        "p0_themes": sorted(P0_THEME_ROUTING_RULES),
    }

