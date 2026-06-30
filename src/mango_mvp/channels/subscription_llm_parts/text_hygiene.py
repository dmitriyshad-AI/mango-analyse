from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Optional

from mango_mvp.channels.p0_recall_spec import is_benign_hypothetical_refund
from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.direct_path import (
    _direct_p0_text_hygiene_enabled,
    _payment_refund_dispute_split_enabled,
)
from mango_mvp.channels.subscription_llm_parts.policy_routing import PAYMENT_LINK_SAFE_TEXT


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


def scrub_direct_path_p0_text(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    client_message: str = "",
) -> SubscriptionDraftResult:
    if not _direct_p0_text_hygiene_enabled(context):
        return result
    if _is_benign_presale_refund_question(client_message) and not _manager_high_risk_signal(result):
        if _direct_path_p0_text_needs_scrub(result.draft_text):
            return _direct_path_presale_refund_text(result)
        return result
    if not _direct_path_p0_text_hygiene_applies(result, context=context, client_message=client_message):
        return result
    if not _direct_path_p0_text_needs_scrub(result.draft_text):
        return result

    kind = _direct_path_p0_hygiene_kind(result, context=context, client_message=client_message)
    safe_text = _direct_path_p0_safe_text(kind, context=context)
    metadata = dict(result.metadata)
    metadata["direct_p0_text_hygiene"] = {
        "applied": True,
        "kind": kind or "p0",
        "original_text_removed": True,
    }
    metadata["final_p0_text_override"] = True
    flags = tuple(dict.fromkeys([*result.safety_flags, "direct_p0_text_hygiene"]))
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "P0 text hygiene: убрать обещания исхода и продающий хвост перед ручной проверкой менеджером.",
            ]
        )
    )
    return replace(result, draft_text=safe_text, safety_flags=flags, manager_checklist=checklist, metadata=metadata)


def _direct_path_p0_text_hygiene_applies(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    client_message: str,
) -> bool:
    if _is_benign_presale_refund_question(client_message) and not _manager_high_risk_signal(result):
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
    if str(result.route or "") == "manager_only" and _ROUTE_REFUND_RE.search(str(client_message or "")):
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
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    for key in ("direct_path_model_p0",):
        value = metadata.get(key)
        if isinstance(value, Mapping) and bool(value.get("is_p0")):
            return _normalize_p0_hygiene_kind(value.get("p0_kind"))
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping):
        value = direct.get("model_p0")
        if isinstance(value, Mapping) and bool(value.get("is_p0")):
            return _normalize_p0_hygiene_kind(value.get("p0_kind"))
    flags = {str(flag or "").strip().casefold() for flag in result.safety_flags}
    for kind in ("payment_dispute", "refund", "complaint", "legal_threat", "legal"):
        if kind in flags or f"direct_path_model_p0_{kind}" in flags:
            return kind
    if _semantic_frame_payment_class(result, context=context) == "forward_payment":
        return "forward_payment"
    text = str(client_message or "").casefold().replace("ё", "е")
    if re.search(r"\b(?:списал|платеж|платежн|чек|квитанц|оплата\s+прошла|деньги\s+списал)\b", text):
        return "payment_dispute"
    if _ROUTE_LEGAL_CONTRACT_RE.search(str(client_message or "")):
        return "legal_threat"
    if _ROUTE_REFUND_RE.search(str(client_message or "")):
        return "refund"
    return ""


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
    frame = metadata.get("semantic_frame_shadow")
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


def _direct_path_p0_text_needs_scrub(text: str) -> bool:
    value = str(text or "")
    return bool(
        _REFUND_PROMISE_RE.search(value)
        or _SALES_OR_PAYMENT_NUDGE_RE.search(value)
        or _P0_DETAIL_COLLECTION_RE.search(value)
        or _POSTPAYMENT_CONTEXT_HANDOFF_RE.search(value)
    )


def _direct_path_p0_safe_text(kind: str, *, context: Optional[Mapping[str, Any]] = None) -> str:
    if kind == "forward_payment":
        return PAYMENT_LINK_SAFE_TEXT
    if kind == "cancellation_service_request":
        return "Передам вопрос менеджеру: он проверит вашу запись и подскажет безопасный порядок действий."
    if kind == "payment_dispute":
        return "По оплате нужно сверить данные в системе. Передам вопрос менеджеру: он проверит ситуацию и ответит по точным данным."
    if kind in {"complaint", "legal_threat", "legal"}:
        return "Такой вопрос нужно проверить вручную. Передам его менеджеру, чтобы он сверил документы и ответил по точным данным."
    if _payment_refund_dispute_split_enabled(context):
        return "Такой вопрос нужно проверить вручную. Передам его менеджеру, чтобы он сверил ситуацию и ответил по точным данным."
    return (
        "По возврату точную сумму и порядок действий должен подтвердить менеджер. "
        "Передам ему ваш вопрос, чтобы он проверил ситуацию по данным записи и оплаты."
    )


def _direct_path_presale_refund_text(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    metadata["direct_presale_policy_text_hygiene"] = {
        "applied": True,
        "kind": "presale_policy",
        "original_text_removed": True,
    }
    flags = tuple(dict.fromkeys([*result.safety_flags, "direct_presale_policy_text_hygiene"]))
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "Presale text hygiene: сохранить вопрос до оплаты как не-P0 и убрать ложную привязку к записи/оплате.",
            ]
        )
    )
    return replace(
        result,
        draft_text=(
            "До оплаты можно спокойно уточнить условия заранее. Если передумаете до записи и оплаты, "
            "порядок оформления менеджер подтвердит по выбранному курсу и договору."
        ),
        safety_flags=flags,
        manager_checklist=checklist,
        metadata=metadata,
    )
