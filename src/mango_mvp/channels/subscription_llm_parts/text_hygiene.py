from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Optional

from mango_mvp.channels.p0_recall_spec import is_benign_hypothetical_refund
from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.direct_path import _direct_p0_text_hygiene_enabled


_P0_HYGIENE_KINDS = frozenset({"refund", "payment_dispute", "complaint", "legal_threat", "legal"})

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
    r"успейт|не\s+тяните|скидк\w*|мест[ао]\s+(?:есть|остал)"
    r")\b"
)

_ROUTE_REFUND_RE = re.compile(
    r"(?iu)\b(?:возврат|вернуть|верн[её]м|деньг[иа]?|оплат[ауы]|плат[её]ж|списал|претензи|договор|паспортн)"
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
    if not _direct_path_p0_text_needs_scrub(result.draft_text):
        return result

    kind = _direct_path_p0_hygiene_kind(result)
    safe_text = _direct_path_p0_safe_text(kind)
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
    if is_benign_hypothetical_refund(client_message):
        return False
    kind = _direct_path_p0_hygiene_kind(result)
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


def _direct_path_p0_hygiene_kind(result: SubscriptionDraftResult) -> str:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    for key in ("direct_path_model_p0",):
        value = metadata.get(key)
        if isinstance(value, Mapping) and bool(value.get("is_p0")):
            return str(value.get("p0_kind") or "").strip().casefold()
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping):
        value = direct.get("model_p0")
        if isinstance(value, Mapping) and bool(value.get("is_p0")):
            return str(value.get("p0_kind") or "").strip().casefold()
    flags = {str(flag or "").strip().casefold() for flag in result.safety_flags}
    for kind in ("payment_dispute", "refund", "complaint", "legal_threat", "legal"):
        if kind in flags or f"direct_path_model_p0_{kind}" in flags:
            return kind
    return ""


def _direct_path_p0_text_needs_scrub(text: str) -> bool:
    value = str(text or "")
    return bool(_REFUND_PROMISE_RE.search(value) or _SALES_OR_PAYMENT_NUDGE_RE.search(value))


def _direct_path_p0_safe_text(kind: str) -> str:
    if kind == "payment_dispute":
        return "По оплате нужно сверить данные в системе. Передам вопрос менеджеру: он проверит ситуацию и ответит по точным данным."
    if kind in {"complaint", "legal_threat", "legal"}:
        return "Такой вопрос нужно проверить вручную. Передам его менеджеру, чтобы он сверил документы и ответил по точным данным."
    return (
        "По возврату точную сумму и порядок действий должен подтвердить менеджер. "
        "Передам ему ваш вопрос, чтобы он проверил ситуацию по данным записи и оплаты."
    )
