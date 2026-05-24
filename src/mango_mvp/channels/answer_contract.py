from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


ANSWER_CONTRACT_SCHEMA_VERSION = "answer_contract_v2_2026_05_24"


@dataclass(frozen=True)
class AnswerContract:
    active_brand: str
    direct_question: str = ""
    primary_intent: str = ""
    topic_id: str = ""
    answer_policy: str = ""
    route_bias: str = ""
    known_slots: Mapping[str, str] = field(default_factory=dict)
    do_not_reask_slots: tuple[str, ...] = ()
    required_fact_keys: tuple[str, ...] = ()
    required_fact_ids: tuple[str, ...] = ()
    facts_resolved_by_intent: tuple[str, ...] = ()
    route: str = ""
    route_reason: str = ""
    must_answer_first: bool = False
    answerable_safe_parts: tuple[str, ...] = ()
    manager_parts: tuple[str, ...] = ()
    risk_codes: tuple[str, ...] = ()
    p0_required: bool = False
    blocks_autonomy: bool = False
    blocks_rewriter: bool = False
    forbidden_assumptions: tuple[str, ...] = ()
    safe_next_step: str = ""
    source: str = "conversation_intent_plan+dialogue_memory+safety"

    def to_prompt_view(self) -> Mapping[str, Any]:
        return {
            "schema_version": ANSWER_CONTRACT_SCHEMA_VERSION,
            "active_brand": self.active_brand,
            "direct_question": self.direct_question,
            "primary_intent": self.primary_intent,
            "topic_id": self.topic_id,
            "answer_policy": self.answer_policy,
            "route_bias": self.route_bias,
            "known_slots": dict(self.known_slots),
            "do_not_reask_slots": list(self.do_not_reask_slots),
            "required_fact_keys": list(self.required_fact_keys),
            "required_fact_ids": list(self.required_fact_ids),
            "facts_resolved_by_intent": list(self.facts_resolved_by_intent),
            "route": self.route,
            "route_reason": self.route_reason,
            "must_answer_first": self.must_answer_first,
            "answerable_safe_parts": list(self.answerable_safe_parts),
            "manager_parts": list(self.manager_parts),
            "risk_codes": list(self.risk_codes),
            "p0_required": self.p0_required,
            "blocks_autonomy": self.blocks_autonomy,
            "blocks_rewriter": self.blocks_rewriter,
            "forbidden_assumptions": list(self.forbidden_assumptions),
            "safe_next_step": self.safe_next_step,
            "source": self.source,
        }


def build_answer_contract(
    *,
    active_brand: str,
    conversation_intent_plan: Mapping[str, Any] | None = None,
    dialogue_memory_view: Mapping[str, Any] | None = None,
    safety_decision: Mapping[str, Any] | None = None,
    known_slots: Mapping[str, Any] | None = None,
    confirmed_facts: Mapping[str, Any] | None = None,
) -> AnswerContract:
    plan = conversation_intent_plan if isinstance(conversation_intent_plan, Mapping) else {}
    memory = dialogue_memory_view if isinstance(dialogue_memory_view, Mapping) else {}
    safety = safety_decision if isinstance(safety_decision, Mapping) else {}
    memory_open = memory.get("open_question") if isinstance(memory.get("open_question"), Mapping) else {}
    direct_question = str(plan.get("direct_question") or memory_open.get("text") or "").strip()[:260]
    primary_intent = str(plan.get("primary_intent") or memory_open.get("kind") or "").strip()
    topic_id = str(plan.get("topic_id") or "").strip()
    answer_policy = str(plan.get("answer_policy") or "").strip()
    route_bias = str(plan.get("route_bias") or "").strip()
    required_fact_keys = _text_tuple(plan.get("required_fact_keys"))
    resolved_fact_ids = _fact_ids_from_confirmed(confirmed_facts)
    merged_slots: dict[str, str] = {}
    _merge_slots(merged_slots, memory.get("known_slots") if isinstance(memory.get("known_slots"), Mapping) else {})
    _merge_slots(merged_slots, plan.get("known_slots") if isinstance(plan.get("known_slots"), Mapping) else {})
    _merge_slots(merged_slots, known_slots or {})
    risk_codes = _text_tuple(safety.get("risk_codes"))
    p0_required = _truthy(safety.get("p0_required"))
    blocks_autonomy = _truthy(safety.get("blocks_autonomy")) or p0_required
    blocks_rewriter = _truthy(safety.get("blocks_rewriter")) or p0_required
    answerable_parts, manager_parts = _parts_for_intent(primary_intent, p0_required=p0_required)
    must_answer_first = bool(direct_question and not p0_required and primary_intent not in {"off_topic"})
    route = _contract_route(
        p0_required=p0_required,
        route_bias=route_bias,
        answer_policy=answer_policy,
        resolved_fact_ids=resolved_fact_ids,
    )
    route_reason = _contract_route_reason(
        p0_required=p0_required,
        risk_codes=risk_codes,
        answer_policy=answer_policy,
        resolved_fact_ids=resolved_fact_ids,
    )
    return AnswerContract(
        active_brand=_normalize_brand(active_brand),
        direct_question=direct_question,
        primary_intent=primary_intent,
        topic_id=topic_id,
        answer_policy=answer_policy,
        route_bias=route_bias,
        known_slots=merged_slots,
        do_not_reask_slots=tuple(sorted(merged_slots)),
        required_fact_keys=required_fact_keys,
        required_fact_ids=resolved_fact_ids,
        facts_resolved_by_intent=resolved_fact_ids,
        route=route,
        route_reason=route_reason,
        must_answer_first=must_answer_first,
        answerable_safe_parts=answerable_parts,
        manager_parts=manager_parts,
        risk_codes=risk_codes,
        p0_required=p0_required,
        blocks_autonomy=blocks_autonomy,
        blocks_rewriter=blocks_rewriter,
        forbidden_assumptions=_forbidden_assumptions(primary_intent, merged_slots),
        safe_next_step=str(plan.get("next_step_hint") or memory.get("safe_next_action") or "").strip()[:260],
    )


def _contract_route(
    *,
    p0_required: bool,
    route_bias: str,
    answer_policy: str,
    resolved_fact_ids: tuple[str, ...],
) -> str:
    if p0_required:
        return "manager_only"
    bias = route_bias.strip()
    if bias:
        return bias
    if answer_policy in {"answer_directly_if_fact_verified", "answer_safe_parts_then_manager_live_check"} and resolved_fact_ids:
        return "bot_answer_self_for_pilot"
    if answer_policy == "answer_safe_parts_then_manager_live_check":
        return "draft_for_manager"
    return "draft_for_manager"


def _contract_route_reason(
    *,
    p0_required: bool,
    risk_codes: tuple[str, ...],
    answer_policy: str,
    resolved_fact_ids: tuple[str, ...],
) -> str:
    if p0_required:
        return "p0_required:" + ",".join(risk_codes or ("high_risk",))
    if resolved_fact_ids:
        return "facts_resolved_by_intent"
    if answer_policy:
        return answer_policy
    return "default_draft_until_fact_verified"


def _parts_for_intent(intent: str, *, p0_required: bool) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if p0_required:
        return (), ("p0_manager_handoff",)
    safe_map = {
        "pricing": ("price",),
        "price_fix": ("current_terms_process",),
        "installment": ("installment",),
        "discount": ("discount",),
        "trial": ("trial",),
        "camp": ("camp",),
        "schedule": ("schedule",),
        "format": ("format",),
        "address": ("address",),
        "document": ("document",),
        "matkap": ("matkap",),
        "tax": ("tax",),
        "identity": ("identity",),
    }
    if intent == "live_availability":
        return ("safe_product_info",), ("availability_check",)
    return safe_map.get(intent, ("general_help",)), ()


def _forbidden_assumptions(intent: str, slots: Mapping[str, str]) -> tuple[str, ...]:
    result = ["unsupported_price_date_or_schedule", "unstated_goal_or_subject"]
    if not slots.get("subject"):
        result.append("do_not_invent_subject")
    if not slots.get("format"):
        result.append("do_not_invent_format")
    if intent in {"camp", "live_availability"}:
        result.append("do_not_claim_places_available")
    return tuple(dict.fromkeys(result))


def _merge_slots(target: dict[str, str], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        normalized = _slot_key(key)
        if not normalized:
            continue
        text = str(value.get("value") if isinstance(value, Mapping) else value or "").strip()
        if text:
            target.setdefault(normalized, text[:160])


def _slot_key(value: Any) -> str:
    key = str(value or "").strip()
    aliases = {
        "class": "grade",
        "student_grade": "grade",
        "course_subject": "subject",
        "interest_subject": "subject",
        "course_format": "format",
        "preferred_format": "format",
    }
    key = aliases.get(key, key)
    return key if key in {"grade", "subject", "format", "goal", "product", "active_brand"} else ""


def _text_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item or "").strip() for item in value if str(item or "").strip())
    return ()


def _fact_ids_from_confirmed(value: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    result: list[str] = []
    for key in value:
        text = str(key or "").strip()
        if text:
            result.append(text)
    return tuple(dict.fromkeys(result))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "да"}


def _normalize_brand(value: Any) -> str:
    text = str(value or "unknown").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return "unknown"
