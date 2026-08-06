from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Optional

from mango_mvp.channels.p0_recall_spec import is_benign_hypothetical_refund
from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.direct_path import (
    _direct_p0_text_hygiene_enabled,
    _payment_refund_dispute_split_enabled,
    _text_hygiene_payment_fix_enabled,
)
from mango_mvp.channels.subscription_llm_parts.policy_routing import (
    PAYMENT_LINK_SAFE_TEXT,
    TAX_DEDUCTION_PROCESS_SAFE_TEXT,
)
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    append_reading_trace_record,
    semantic_reading_trace_record,
)
from mango_mvp.channels.subscription_llm_parts.support import _p0_model_led_enabled


_P0_HYGIENE_KINDS = frozenset(
    {
        "refund",
        "payment_dispute",
        "complaint",
        "legal_threat",
        "legal",
        "cancellation_service_request",
        "contract_dispute",
        "paid_operation_context",
        "tax",
        "neutral_manager",
    }
)

_P0_HYGIENE_LEGACY_KIND = {
    "contract_dispute": "legal_threat",
    "paid_operation_context": "refund",
}

_REFUND_PROMISE_RE = re.compile(
    r"(?iu)\b(?:"
    r"да,\s*)?"
    r"(?:верн[её]м|возвраща(?:ется|ем|ют)|можно\s+вернуть|оформим\s+возврат|получите\s+возврат)"
    r"\b"
)

_SALES_OR_PAYMENT_NUDGE_RE = re.compile(
    r"(?iu)\b(?:"
    r"оплат(?:ите|ить|а|е|у|ой)|запиш(?:итесь|ем|у|итесь)|подбер[её]м\s+групп|подобрать\s+групп|"
    r"оставьте\s+(?:заявку|телефон)|пришлите\s+чек|перейд[её]м\s+к\s+записи|"
    r"переход(?:ить|ите|им)?\s+к\s+следующему\s+шагу|если\s+условия\s+подходят|"
    r"успейт|не\s+тяните|скидк\w*|мест[ао]\s+(?:есть|остал)"
    r")\b"
)

_P0_DETAIL_COLLECTION_RE = re.compile(
    r"(?iu)\b(?:"
    r"(?:напишите|уточните|пришлите|отправьте|укажите|сообщите)\b"
    r"[^.!?\n]{0,90}?"
    r"(?:что\s+именно|какое\s+(?:поле|место)|где\s+ошибка|как\s+должно\s+быть\s+правильно|"
    r"номер\s+договора|данн(?:ые|ых)|фио|паспорт|скан|фото|чек|квитанц|сумм[ауы]|причин[ауы])"
    r"|"
    r"(?:какое\s+(?:поле|место)[^.!?\n]{0,80}(?:неверно|ошибк)|как\s+должно\s+быть\s+правильно)"
    r")\b"
)

_POSTPAYMENT_CONTEXT_HANDOFF_RE = re.compile(
    r"(?iu)\b(?:точн\w*\s+сумм\w*|по\s+данным\s+запис[ииь]\s+и\s+оплат\w*|"
    r"данным\s+запис[ииь]|после\s+оплат\w*)\b"
)

_ROUTE_REFUND_RE = re.compile(
    r"(?iu)\b(?:возврат|вернуть|верн[её]м|деньг[иа]?|оплат[ауы]|плат[её]ж|списал|претензи|договор|паспортн)"
)

_ROUTE_LEGAL_CONTRACT_RE = re.compile(
    r"(?iu)\b(?:договор|документ|паспорт|фио|ф\s*и\s*о|подпис|претензи|ошибк[аи]\s+в\s+договор)"
)

_PRESALE_REFUND_CONTEXT_RE = re.compile(
    r"(?iu)(?:"
    r"(?:передума\w*|не\s+подойд[её]т|ничем\s+не\s+грозит|можно\s+ли\s+спокойно)[^.!?\n]{0,100}"
    r"(?:до\s+оплат\w*|перед\s+оплат\w*|ещ[её]\s+до\s+оплат\w*)"
    r"|(?:до\s+оплат\w*|перед\s+оплат\w*|ещ[её]\s+до\s+оплат\w*)[^.!?\n]{0,100}"
    r"(?:передума\w*|не\s+подойд[её]т|ничем\s+не\s+грозит|можно\s+ли\s+спокойно|услови\w*\s+возврат)"
    r"|(?:запис[ьи]\s+и\s+оплат\w*|оплат\w*\s+и\s+запис[ьи])[^.!?\n]{0,60}"
    r"(?:нет|ещ[её]\s+нет|не\s+было)[^.!?\n]{0,100}(?:передума\w*|спокойно)"
    r")"
)


_REFUND_MEANING_WORDS = (
    "возврат",
    "вернуть",
    "верните",
    "вернете",
    "вернёте",
    "вернем",
    "вернём",
    "деньги назад",
)

_PAYMENT_NON_REFUND_WORDS = (
    "оплат",
    "платеж",
    "платёж",
    "списал",
    "списали",
    "списание",
    "чек",
    "квитанц",
    "доступ",
    "логин",
    "пароль",
)

_SEMANTIC_FRAME_TEXT_MEANING_CONFIDENCE = 0.90
_SEMANTIC_FRAME_TEXT_MEANING_SCHEMA_VERSION = "semantic_frame_v1_2026_07_01"
_SEMANTIC_CLIENT_TEMPLATE_KINDS = frozenset({"refund", "tax", "payment_dispute"})
_SEMANTIC_FRAME_TAX_ACTIONS = frozenset({"answer_question", "send_document"})
_SEMANTIC_FRAME_REFUND_ACTIONS = frozenset({"refund_or_cancel"})
_SEMANTIC_FRAME_PAYMENT_RISK_CLASSES = frozenset({"payment_dispute"})
_SEMANTIC_FRAME_PAYMENT_READINESS = frozenset({"dispute"})
_SEMANTIC_FRAME_PAID_SERVICE_READINESS = frozenset(
    {
        "paid",
        "paid_transfer",
        "paid_context",
        "payment_done",
        "payment_completed",
    }
)
_SEMANTIC_FRAME_STRUCTURED_ACTIONS = frozenset(
    {
        "answer_question",
        "check_availability",
        "enroll",
        "send_materials",
        "send_payment_link",
        "send_document",
        "refund_or_cancel",
        "handoff_manager",
        "unknown",
    }
)


def scrub_direct_path_p0_text(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    client_message: str = "",
) -> SubscriptionDraftResult:
    if not _direct_p0_text_hygiene_enabled(context):
        return result
    if not _direct_path_p0_text_hygiene_applies(result, context=context, client_message=client_message):
        return result

    legacy_kind = _direct_path_p0_hygiene_kind_legacy(result, context=context, client_message=client_message)
    kind = _direct_path_p0_hygiene_kind(result, context=context, client_message=client_message)
    semantic_guard = _semantic_frame_text_meaning_guard(
        result,
        context=context,
        legacy_kind=legacy_kind,
        chosen_kind=kind,
    )
    text_needs_scrub = _direct_path_p0_text_needs_scrub(result.draft_text)
    kind_mismatch_needs_scrub = _direct_path_p0_text_kind_mismatch_needs_scrub(result.draft_text, kind)
    if not text_needs_scrub and not kind_mismatch_needs_scrub and not semantic_guard.get("force_scrub"):
        return result

    safe_text = _direct_path_p0_safe_text(kind, context=context)
    metadata = dict(result.metadata)
    action_decision = metadata.get("action_decision")
    p0_latched = isinstance(action_decision, Mapping) and action_decision.get("p0_latched") is True
    force_manager_route = bool(
        result.route == "bot_answer_self_for_pilot"
        and (
            p0_latched
            or (
                kind in _P0_HYGIENE_KINDS
                and (text_needs_scrub or kind_mismatch_needs_scrub or semantic_guard.get("force_scrub"))
            )
        )
    )
    hygiene_metadata = {
        "applied": True,
        "kind": kind or "p0",
        "legacy_kind": legacy_kind or "",
        "original_text_removed": True,
    }
    if semantic_guard:
        hygiene_metadata["semantic_frame_text_meaning_guard"] = dict(semantic_guard)
        metadata["semantic_frame_text_meaning_guard"] = dict(semantic_guard)
        direct = dict(metadata.get("direct_path") or {}) if isinstance(metadata.get("direct_path"), Mapping) else {}
        direct["semantic_frame_text_meaning_guard"] = dict(semantic_guard)
        metadata["direct_path"] = direct
        changed_fields = ("draft_text",)
        if force_manager_route:
            changed_fields = ("draft_text", "route")
        metadata = append_reading_trace_record(
            metadata,
            semantic_reading_trace_record(
                reading_class="p0_text_meaning",
                enabled=True,
                status="applied",
                decision=str(semantic_guard.get("chosen_kind") or kind or "p0"),
                reason=str(semantic_guard.get("reason") or "semantic_frame_owns_p0_text_meaning"),
                source="inline",
                confidence=semantic_guard.get("confidence", 0.0),
                changed_fields=changed_fields,
                conflicts=(legacy_kind,) if semantic_guard.get("conflict") and legacy_kind else (),
                metadata=semantic_guard,
            ),
        )
    metadata["direct_p0_text_hygiene"] = hygiene_metadata
    metadata["final_p0_text_override"] = True
    extra_flags = ["direct_p0_text_hygiene"]
    if semantic_guard.get("conflict") or force_manager_route:
        extra_flags.append("correct_route_wrong_p0_text")
    flags = tuple(dict.fromkeys([*result.safety_flags, *extra_flags]))
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "P0 text hygiene: убрать обещания исхода и продающий хвост перед ручной проверкой менеджером.",
            ]
        )
    )
    route = result.route
    if force_manager_route:
        route = "manager_only"
    return replace(
        result,
        route=route,
        draft_text=safe_text,
        safety_flags=flags,
        manager_checklist=checklist,
        metadata=metadata,
    )


def _direct_path_p0_text_hygiene_applies(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    client_message: str,
) -> bool:
    model_led = _p0_model_led_enabled(context)
    if not model_led and _is_benign_presale_refund_question(client_message) and not _manager_high_risk_signal(result):
        return False
    kind = _direct_path_p0_hygiene_kind(result, context=context, client_message=client_message)
    if kind == "forward_payment":
        return True
    if kind in _P0_HYGIENE_KINDS:
        return True
    flags = {str(flag or "").strip().casefold() for flag in result.safety_flags}
    if flags.intersection(
        {
            "refund",
            "payment_dispute",
            "complaint",
            "legal",
            "legal_threat",
            "zero_collect_required",
            "refund_policy_manager_only",
        }
    ):
        return True
    if not model_led and str(result.route or "") == "manager_only" and _ROUTE_REFUND_RE.search(str(client_message or "")):
        return True
    if isinstance(context, Mapping):
        memory = context.get("dialogue_memory_view")
        if isinstance(memory, Mapping):
            latch = memory.get("p0_latch")
            if isinstance(latch, Mapping) and bool(latch.get("active")):
                return True
    return False


def _is_benign_presale_refund_question(client_message: str) -> bool:
    value = str(client_message or "")
    if _PRESALE_REFUND_CONTEXT_RE.search(value):
        return True
    if is_benign_hypothetical_refund(value) and re.search(
        r"(?iu)\b(?:до\s+оплат\w*|перед\s+оплат\w*|ещ[её]\s+до\s+оплат\w*|до\s+запис[ииь]|заранее)\b",
        value,
    ):
        return True
    return False


def _manager_high_risk_signal(result: SubscriptionDraftResult) -> bool:
    if str(result.route or "") != "manager_only":
        return False
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    if re.search(r"high_risk|zero_collect|refund|payment_dispute|legal|complaint|p0", flags):
        return True
    return any(
        bool(metadata.get(key))
        for key in (
            "combined_high_risk_manager_only",
            "autonomy_blocked_high_risk",
            "forced_route_high_risk",
            "forced_route_high_risk_input",
            "final_p0_text_override",
            "zero_collect_legal_guarded",
            "zero_collect_refund_guarded",
            "payment_dispute_manager_only",
        )
    )


def _direct_path_p0_hygiene_kind(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    client_message: str = "",
) -> str:
    legacy_kind = _direct_path_p0_hygiene_kind_legacy(result, context=context, client_message=client_message)
    return _semantic_frame_adjusted_hygiene_kind(
        legacy_kind,
        result,
        context=context,
        client_message=client_message,
    )


def _direct_path_p0_hygiene_kind_legacy(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    client_message: str = "",
) -> str:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    for key in ("direct_path_model_p0",):
        value = metadata.get(key)
        if isinstance(value, Mapping) and bool(value.get("is_p0")) and bool(value.get("route_applied")):
            kind = _normalize_p0_hygiene_kind(value.get("p0_kind"))
            return kind if _p0_model_led_enabled(context) else _payment_fix_adjusted_kind(
                kind,
                raw_kind=value.get("p0_kind"),
                context=context,
                client_message=client_message,
            )
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping):
        value = direct.get("model_p0")
        if isinstance(value, Mapping) and bool(value.get("is_p0")) and bool(value.get("route_applied")):
            kind = _normalize_p0_hygiene_kind(value.get("p0_kind"))
            return kind if _p0_model_led_enabled(context) else _payment_fix_adjusted_kind(
                kind,
                raw_kind=value.get("p0_kind"),
                context=context,
                client_message=client_message,
            )
    if _p0_model_led_enabled(context):
        return _context_p0_latch_kind(context)
    flags = {str(flag or "").strip().casefold() for flag in result.safety_flags}
    for kind in ("payment_dispute", "refund", "complaint", "legal_threat", "legal"):
        if kind in flags or f"direct_path_model_p0_{kind}" in flags:
            return kind
    if _semantic_frame_payment_class(result, context=context) == "forward_payment":
        return "forward_payment"
    if _text_hygiene_payment_fix_enabled(context) and not _client_message_has_refund_meaning(client_message):
        frame = _semantic_frame_from_result_metadata(result)
        if _semantic_frame_paid_service_issue(frame):
            return "payment_dispute"
        if _context_p0_latch_kind(context) == "payment_dispute":
            return "payment_dispute"
    text = str(client_message or "").casefold().replace("ё", "е")
    if re.search(r"\b(?:списал|платеж|платежн|чек|квитанц|оплата\s+прошла|деньги\s+списал)\b", text):
        return "payment_dispute"
    if _ROUTE_LEGAL_CONTRACT_RE.search(str(client_message or "")):
        return "legal_threat"
    if _ROUTE_REFUND_RE.search(str(client_message or "")):
        if _text_hygiene_payment_fix_enabled(context) and not _client_message_has_refund_meaning(client_message):
            if _client_message_has_payment_non_refund_meaning(client_message):
                return "payment_dispute"
            return ""
        return "refund"
    return ""


def _semantic_frame_adjusted_hygiene_kind(
    legacy_kind: str,
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    client_message: str,
) -> str:
    del client_message
    guard = _semantic_frame_text_meaning_guard(
        result,
        context=context,
        legacy_kind=legacy_kind,
        chosen_kind=legacy_kind,
    )
    chosen = str(guard.get("chosen_kind") or "").strip()
    return chosen or legacy_kind


def _payment_fix_adjusted_kind(
    kind: str,
    *,
    raw_kind: object,
    context: Optional[Mapping[str, Any]],
    client_message: str,
) -> str:
    if not _text_hygiene_payment_fix_enabled(context):
        return kind
    raw = str(raw_kind or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if raw not in {
        "paid_context",
        "paid_operation",
        "paid_operation_context",
        "paid_refund_context",
        "paid_transfer_context",
    }:
        return kind
    if _client_message_has_refund_meaning(client_message):
        return "refund"
    if _client_message_has_payment_non_refund_meaning(client_message):
        return "payment_dispute"
    return "payment_dispute"


def _client_message_has_refund_meaning(client_message: str) -> bool:
    text = str(client_message or "").casefold().replace("ё", "е")
    if any(word.replace("ё", "е") in text for word in _REFUND_MEANING_WORDS):
        return True
    payment_or_money = any(word in text for word in ("оплат", "деньг", "платеж", "сумм"))
    if payment_or_money and ("обратно" in text or "отда" in text):
        return True
    return False


def _client_message_has_payment_non_refund_meaning(client_message: str) -> bool:
    text = str(client_message or "").casefold().replace("ё", "е")
    return any(word.replace("ё", "е") in text for word in _PAYMENT_NON_REFUND_WORDS)


def _normalize_p0_hygiene_kind(value: object) -> str:
    kind = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if kind == "legal":
        return "legal"
    if kind in {"contract", "contract_issue", "document_dispute", "contract_claim"}:
        kind = "contract_dispute"
    if kind in {"cancellation", "cancel_service", "service_cancellation", "enrollment_cancel"}:
        kind = "cancellation_service_request"
    if kind in {"paid_context", "paid_operation", "paid_refund_context", "paid_transfer_context"}:
        kind = "paid_operation_context"
    if kind in {"payment", "payment_issue", "payment_problem", "payment_claim"}:
        kind = "payment_dispute"
    return _P0_HYGIENE_LEGACY_KIND.get(kind, kind)


def _semantic_frame_payment_class(result: SubscriptionDraftResult, *, context: Optional[Mapping[str, Any]] = None) -> str:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if not _payment_refund_dispute_split_enabled(context):
        return ""
    if str(result.route or "") == "manager_only":
        return ""
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    frame = metadata.get("semantic_frame")
    if not isinstance(frame, Mapping):
        frame = metadata.get("semantic_frame_shadow")
    if not isinstance(frame, Mapping):
        frame = direct.get("semantic_frame") if isinstance(direct.get("semantic_frame"), Mapping) else {}
    if not isinstance(frame, Mapping):
        frame = direct.get("semantic_frame_shadow") if isinstance(direct.get("semantic_frame_shadow"), Mapping) else {}
    if not isinstance(frame, Mapping) or not frame:
        return ""
    risk = str(frame.get("risk_class") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    requested_action = str(frame.get("requested_action") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    readiness = str(frame.get("payment_readiness") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    frame_forward = requested_action == "send_payment_link" or readiness in {
        "ready",
        "ready_to_pay",
        "pay_now",
        "wants_to_pay",
    }
    if not frame_forward:
        return ""
    if risk in {
        "p0",
        "high",
        "critical",
        "high_risk",
        "refund",
        "refund_claim",
        "payment_dispute",
        "dispute",
        "complaint",
        "legal",
        "legal_threat",
    }:
        return ""
    action_decision = metadata.get("action_decision") if isinstance(metadata.get("action_decision"), Mapping) else {}
    if action_decision:
        action = str(action_decision.get("action") or "").strip().casefold()
        if action != "send_payment_link":
            return ""
        if action_decision.get("no_live_execution") is False:
            return ""
        return "forward_payment"
    return "forward_payment"


def _semantic_frame_text_meaning_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    legacy_kind: str,
    chosen_kind: str,
) -> Mapping[str, Any]:
    if not _text_hygiene_payment_fix_enabled(context):
        return {}
    frame = _semantic_frame_from_result_metadata(result)
    if not frame:
        return {}
    confidence = _semantic_frame_confidence(frame)
    if confidence < _SEMANTIC_FRAME_TEXT_MEANING_CONFIDENCE:
        return {}
    if not _semantic_frame_text_meaning_frame_is_valid(frame):
        return {}
    legacy = str(legacy_kind or "").strip()
    semantic_kind = _semantic_frame_client_text_kind(
        frame,
        context=context,
        result=result,
        legacy_kind=legacy,
    )
    if not semantic_kind:
        if legacy in _SEMANTIC_CLIENT_TEMPLATE_KINDS:
            semantic_kind = "neutral_manager"
        else:
            return {}
    neutral_p0_frame = semantic_kind == "neutral_manager" and _semantic_frame_requires_neutral_p0_text(frame)
    if legacy not in _SEMANTIC_CLIENT_TEMPLATE_KINDS and semantic_kind == "neutral_manager" and not neutral_p0_frame:
        return {}
    conflict = bool(legacy and legacy != semantic_kind and legacy in _SEMANTIC_CLIENT_TEMPLATE_KINDS)
    if not conflict and str(chosen_kind or "") == semantic_kind and not neutral_p0_frame:
        return {}
    return {
        "schema_version": "semantic_frame_text_meaning_guard_v1_2026_07_06",
        "applied": True,
        "legacy_kind": legacy,
        "frame_kind": semantic_kind,
        "chosen_kind": semantic_kind,
        "conflict": conflict,
        "force_scrub": conflict or semantic_kind in {"tax", "neutral_manager"},
        "confidence": confidence,
        "threshold": _SEMANTIC_FRAME_TEXT_MEANING_CONFIDENCE,
        "frame_source": str(frame.get("source") or ""),
        "frame_schema_version": str(frame.get("schema_version") or ""),
        "reason": "semantic_frame_owns_p0_text_meaning",
    }


def _semantic_frame_from_result_metadata(result: SubscriptionDraftResult) -> Mapping[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    for container in (metadata, direct):
        frame = container.get("semantic_frame")
        if isinstance(frame, Mapping) and frame:
            return frame
        frame = container.get("semantic_frame_shadow")
        if isinstance(frame, Mapping) and frame:
            return frame
    return {}


def _semantic_frame_confidence(frame: Mapping[str, Any]) -> float:
    try:
        value = float(frame.get("confidence"))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _semantic_frame_text_meaning_frame_is_valid(frame: Mapping[str, Any]) -> bool:
    source = str(frame.get("source") or "").strip().casefold()
    schema = str(frame.get("schema_version") or "").strip()
    action = _semantic_frame_enum_value(frame.get("requested_action"))
    if source != "inline":
        return False
    if schema != _SEMANTIC_FRAME_TEXT_MEANING_SCHEMA_VERSION:
        return False
    if action not in _SEMANTIC_FRAME_STRUCTURED_ACTIONS:
        return False
    return True


def _semantic_frame_client_text_kind(
    frame: Mapping[str, Any],
    *,
    context: Optional[Mapping[str, Any]],
    result: SubscriptionDraftResult,
    legacy_kind: str,
) -> str:
    action = _semantic_frame_enum_value(frame.get("requested_action"))
    risk = _semantic_frame_enum_value(frame.get("risk_class"))
    readiness = _semantic_frame_enum_value(frame.get("payment_readiness"))
    answerability = _semantic_frame_enum_value(frame.get("answerability"))
    must_handoff = _semantic_frame_bool(frame.get("must_handoff"))
    if action in _SEMANTIC_FRAME_REFUND_ACTIONS:
        return "refund"
    if risk in _SEMANTIC_FRAME_PAYMENT_RISK_CLASSES or readiness in _SEMANTIC_FRAME_PAYMENT_READINESS:
        return "payment_dispute"
    if _semantic_frame_paid_service_issue(frame):
        return "payment_dispute"
    plan = _conversation_plan_for_text_meaning(context, result=result)
    if _conversation_plan_is_tax(plan) and action in _SEMANTIC_FRAME_TAX_ACTIONS:
        return "tax"
    if _semantic_frame_requires_neutral_p0_text(frame):
        return "neutral_manager"
    if (
        legacy_kind in _SEMANTIC_CLIENT_TEMPLATE_KINDS
        and must_handoff is True
        and answerability in {"manager_only", "uncertain"}
    ):
        return "neutral_manager"
    return ""


def _conversation_plan_for_text_meaning(
    context: Optional[Mapping[str, Any]],
    *,
    result: SubscriptionDraftResult,
) -> Mapping[str, Any]:
    containers: list[Mapping[str, Any]] = []
    if isinstance(context, Mapping):
        containers.append(context)
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    containers.append(metadata)
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    containers.append(direct)
    for container in containers:
        value = container.get("conversation_intent_plan")
        if isinstance(value, Mapping):
            return value
    return {}


def _semantic_frame_paid_service_issue(frame: Mapping[str, Any]) -> bool:
    if not frame:
        return False
    if not _semantic_frame_text_meaning_frame_is_valid(frame):
        return False
    if _semantic_frame_confidence(frame) < _SEMANTIC_FRAME_TEXT_MEANING_CONFIDENCE:
        return False
    action = _semantic_frame_enum_value(frame.get("requested_action"))
    if action in _SEMANTIC_FRAME_REFUND_ACTIONS:
        return False
    readiness = _semantic_frame_enum_value(frame.get("payment_readiness"))
    if readiness not in _SEMANTIC_FRAME_PAID_SERVICE_READINESS:
        return False
    must_handoff = _semantic_frame_bool(frame.get("must_handoff"))
    answerability = _semantic_frame_enum_value(frame.get("answerability"))
    if action != "handoff_manager" and must_handoff is not True:
        return False
    if answerability and answerability not in {"manager_only", "uncertain"}:
        return False
    return True


def _semantic_frame_requires_neutral_p0_text(frame: Mapping[str, Any]) -> bool:
    risk = _semantic_frame_enum_value(frame.get("risk_class"))
    action = _semantic_frame_enum_value(frame.get("requested_action"))
    answerability = _semantic_frame_enum_value(frame.get("answerability"))
    must_handoff = _semantic_frame_bool(frame.get("must_handoff"))
    return risk == "p0" and action == "handoff_manager" and must_handoff is True and answerability == "manager_only"


def _context_p0_latch_kind(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return ""
    memory = context.get("dialogue_memory_view")
    if not isinstance(memory, Mapping):
        return ""
    latch = memory.get("p0_latch")
    if not isinstance(latch, Mapping) or not bool(latch.get("active")):
        return ""
    raw_codes = latch.get("codes")
    if isinstance(raw_codes, str):
        codes = {raw_codes}
    elif isinstance(raw_codes, (list, tuple, set, frozenset)):
        codes = {str(code or "") for code in raw_codes}
    else:
        codes = set()
    normalized = [str(latch.get("primary_risk") or "").strip().casefold()]
    normalized.extend(code.strip().casefold() for code in codes)
    for kind in normalized:
        mapped = _P0_HYGIENE_LEGACY_KIND.get(kind, kind)
        if mapped in _P0_HYGIENE_KINDS:
            return mapped
    return ""


def _conversation_plan_is_tax(plan: Mapping[str, Any]) -> bool:
    primary = _semantic_frame_enum_value(plan.get("primary_intent"))
    payment_source = _semantic_frame_enum_value(plan.get("payment_source"))
    refund_frame = _semantic_frame_enum_value(plan.get("refund_frame"))
    topic_id = str(plan.get("topic_id") or "").strip()
    return (
        primary == "tax"
        and payment_source == "tax_deduction"
        and refund_frame in {"", "none"}
        and topic_id == "theme:008_tax_deduction"
    )


def _semantic_frame_enum_value(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _semantic_frame_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "1", "yes", "да"}:
            return True
        if lowered in {"false", "0", "no", "нет"}:
            return False
    return None


def _direct_path_p0_text_needs_scrub(text: str) -> bool:
    value = str(text or "")
    return bool(
        _REFUND_PROMISE_RE.search(value)
        or _SALES_OR_PAYMENT_NUDGE_RE.search(value)
        or _P0_DETAIL_COLLECTION_RE.search(value)
        or _POSTPAYMENT_CONTEXT_HANDOFF_RE.search(value)
        or re.search(r"(?iu)\b(?:по\s+возврату|обращени[ея]\s+по\s+возврату|вопрос\s+по\s+возврату)\b", value)
    )


def _direct_path_p0_text_kind_mismatch_needs_scrub(text: str, kind: str) -> bool:
    value = str(text or "").casefold()
    normalized_kind = str(kind or "").strip()
    if normalized_kind in {"payment_dispute", "tax", "neutral_manager"} and "возврат" in value:
        return True
    if normalized_kind == "refund" and "по оплате нужно сверить данные" in value:
        return True
    return False


def _direct_path_p0_safe_text(kind: str, *, context: Optional[Mapping[str, Any]] = None) -> str:
    if kind == "forward_payment":
        return PAYMENT_LINK_SAFE_TEXT
    if kind == "cancellation_service_request":
        return "Передам вопрос менеджеру: он проверит вашу запись и подскажет безопасный порядок действий."
    if kind == "payment_dispute":
        return "По оплате нужно сверить данные в системе. Передам вопрос менеджеру: он проверит ситуацию и ответит по точным данным."
    if kind == "tax":
        return TAX_DEDUCTION_PROCESS_SAFE_TEXT
    if kind in {"complaint", "legal_threat", "legal"}:
        return "Такой вопрос нужно проверить вручную. Передам его менеджеру, чтобы он сверил документы и ответил по точным данным."
    if kind == "neutral_manager":
        return "Понимаю, это важно. Передам вопрос менеджеру, чтобы он сверил ситуацию и ответил по точным данным."
    if kind == "refund":
        return (
            "Возможность возврата, сумму и порядок действий должен подтвердить менеджер. "
            "Передам ему ваш вопрос, чтобы он проверил ситуацию по данным записи и оплаты."
        )
    if _payment_refund_dispute_split_enabled(context):
        return "Такой вопрос нужно проверить вручную. Передам его менеджеру, чтобы он сверил ситуацию и ответил по точным данным."
    return (
        "Возможность возврата, сумму и порядок действий должен подтвердить менеджер. "
        "Передам ему ваш вопрос, чтобы он проверил ситуацию по данным записи и оплаты."
    )
