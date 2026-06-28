from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.fact_venue_scope import VENUE_SCOPE_ANY, normalize_fact_venue, normalize_requested_scope
from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult, _normalize_output_sanitizer_text
from mango_mvp.channels.subscription_llm_parts.support import _explicit_truthy_setting, _normalize_fact_match_text

RELIABLE_ANSWERER_STEP1_ENV = "TELEGRAM_RELIABLE_ANSWERER_STEP1"
ANSWER_COVERAGE_PLAN_SCHEMA_VERSION = "answer_coverage_plan_v1_2026_06_25"
RELIABLE_ANSWERER_TRACE_SCHEMA_VERSION = "reliable_answerer_trace_v1_2026_06_25"

_FACETS = (
    "price",
    "schedule",
    "address",
    "dates",
    "format",
    "documents",
    "platform",
    "trial",
    "enrollment",
    "availability",
    "other",
)

_VENUE_SENSITIVE_FACETS = {"address", "price", "schedule", "dates"}

_FACET_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "price": (r"\b(?:стоим|сколько\s+стоит|цен[аы]|оплат|руб|₽|семестр|год)\w*",),
    "schedule": (r"\b(?:расписан|когда\s+занят|дни\s+занят|по\s+каким\s+дням|время\s+занят|график)\w*",),
    "address": (r"\b(?:адрес|где\s+(?:проход|занят|курсы)|куда\s+приход|локаци|филиал|очно)\w*",),
    "dates": (r"\b(?:даты?|старт|начал[оа]|учебн\w+\s+год|когда\s+начин|срок)\w*",),
    "format": (r"\b(?:формат|онлайн|очно|запись|дистанц)\w*",),
    "documents": (r"\b(?:договор|справк|акт|квитанц|сч[её]т|лицензи|документ)\w*",),
    "platform": (r"\b(?:платформ|мтс\s+линк|zoom|ссылка|доступ|личн\w+\s+кабинет)\w*",),
    "trial": (r"\b(?:пробн|тестов\w+\s+занят|диагностик)\w*",),
    "enrollment": (r"\b(?:запис|оформ|заявк|поступить|попасть|анкета)\w*",),
    "availability": (r"\b(?:мест[ао]|наличи[ея]|свободн\w+\s+мест|группа\s+есть|набирается|брон)\w*",),
}

_FACT_FACET_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "price": (r"\b(?:стоим|цен[аы]|руб|₽|семестр|абонемент|оплат)\w*",),
    "schedule": (r"\b(?:расписан|занятия\s+проходят|дни\s+занят|время\s+занят|график)\w*",),
    "address": (r"\b(?:адрес|филиал|москва|долгопрудн|менделеево|сретенка|красносельск)\w*",),
    "dates": (r"\b(?:дат[аы]|старт|начал[оа]|учебн\w+\s+год|срок)\w*",),
    "format": (r"\b(?:формат|онлайн|очно|запись|дистанц)\w*",),
    "documents": (r"\b(?:договор|справк|акт|квитанц|сч[её]т|лицензи|документ)\w*",),
    "platform": (r"\b(?:платформ|мтс\s+линк|zoom|ссылка|доступ|личн\w+\s+кабинет)\w*",),
    "trial": (r"\b(?:пробн|тестов\w+\s+занят|диагностик)\w*",),
    "enrollment": (r"\b(?:запис|оформ|заявк|анкета|поступ)\w*",),
    "availability": (r"\b(?:мест[ао]|наличи[ея]|свободн\w+\s+мест|группа\s+есть|набирается|брон)\w*",),
}

_AVAILABILITY_PROMISE_RE = re.compile(
    r"(?:"
    r"\bмест[ао]\s+(?:есть|доступн\w*|остал[ио]сь|за\s+вами)\b|"
    r"\bостал[ио]сь\s+\d{1,3}\s+мест\b|"
    r"\b(?:забронирую|забронируем|бронь\s+поставим|поставим\s+бронь)\b|"
    r"\b(?:запишу|запишем|оформим\s+место|место\s+оформим)\b|"
    r"\b(?:группа\s+(?:есть|доступн\w*|набирается)|попад[её]те\s+в\s+группу)\b"
    r")",
    re.I,
)

_HANDOFF_ONLY_RE = re.compile(
    r"\b(?:передам\s+менеджеру|менеджер\s+(?:уточнит|проверит|свяжется|ответит)|не\s+могу\s+подсказать|"
    r"нужно\s+уточнить\s+у\s+менеджера|лучше\s+передать\s+менеджеру)\b",
    re.I,
)

_MISSING_ACK_RE = re.compile(
    r"\b(?:уточн[юитм]|провер[юитм]|свер[юитм]|менеджер\s+(?:уточнит|проверит|сверит)|"
    r"нужно\s+уточнить|пока\s+не\s+буду\s+обещать)\b",
    re.I,
)

_P0_REFUND_RE = re.compile(r"\b(?:верн(?:уть|ите|ули|ул[аи]?)|возврат\w*|деньги\s+назад|отказ\w*\s+от\s+обучени)\b", re.I)
_P0_COMPLAINT_RE = re.compile(r"\b(?:жалоб\w*|претензи\w*|недовол\w*|безобрази\w*|обман\w*)\b", re.I)
_P0_LEGAL_RE = re.compile(r"\b(?:юрист\w*|суд\w*|досудебн\w*|роспотреб|прокуратур\w*|иск\w*)\b", re.I)
_P0_PAYMENT_RE = re.compile(r"\b(?:оплат\w*|плат[её]ж\w*|чек\w*|квитанц\w*|списан\w*|деньг\w*)\b", re.I)
_P0_ACCESS_PROBLEM_RE = re.compile(
    r"\b(?:доступ\w*|ссылк\w*|кабинет\w*|платформ\w*)\b.*\b(?:не\s+(?:работа\w*|приш[её]л\w*|выслал\w*|активир\w*)|"
    r"заблокир\w*|закрыт\w*|нет\s+доступ\w*|не\s+могу\s+зайти)\b|"
    r"\b(?:заблокир\w*|закрыт\w*|нет\s+доступ\w*|не\s+могу\s+зайти)\b.*\b(?:доступ\w*|ссылк\w*|кабинет\w*|платформ\w*)\b",
    re.I,
)
_P0_FLAG_RE = re.compile(r"\b(?:p0|payment_dispute|refund|complaint|legal|high_risk|manager_only_p0|funnel_p0)\b", re.I)
_FOTON_RE = re.compile(r"\b(?:foton|фотон\w*)\b", re.I)
_UNPK_RE = re.compile(r"\b(?:unpk|унпк|мфти)\b", re.I)


def reliable_answerer_step1_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    explicit = _explicit_truthy_setting(
        context,
        RELIABLE_ANSWERER_STEP1_ENV,
        aliases=("reliable_answerer_step1", "reliable_answerer_step1_enabled"),
    )
    return bool(explicit) if explicit is not None else False


def reliable_answerer_step1_bypass_reason(
    client_message: str = "",
    *,
    context: Optional[Mapping[str, Any]] = None,
    result: Optional[SubscriptionDraftResult] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return why Step1 must stay inert for this turn."""

    if _has_p0_signal(client_message, context=context, result=result, metadata=metadata):
        return "p0"
    if _has_cross_brand_signal(client_message, context=context, result=result, metadata=metadata):
        return "cross_brand"
    return ""


def reliable_answerer_step1_active_for_turn(
    client_message: str = "",
    *,
    context: Optional[Mapping[str, Any]] = None,
    result: Optional[SubscriptionDraftResult] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> bool:
    return reliable_answerer_step1_enabled(context) and not reliable_answerer_step1_bypass_reason(
        client_message,
        context=context,
        result=result,
        metadata=metadata,
    )


def client_facets_from_text(text: Any) -> tuple[str, ...]:
    normalized = _normalize_fact_match_text(text)
    facets = [
        facet
        for facet, patterns in _FACET_PATTERNS.items()
        if any(re.search(pattern, normalized, re.I) for pattern in patterns)
    ]
    return tuple(dict.fromkeys(facets or ["other"]))


def build_answer_coverage_plan(
    client_message: str,
    *,
    fact_pack: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    enabled = reliable_answerer_step1_active_for_turn(client_message, context=context)
    facts = fact_pack.get("facts") if isinstance(fact_pack, Mapping) and isinstance(fact_pack.get("facts"), Mapping) else {}
    fact_meta = (
        fact_pack.get("fact_metadata")
        if isinstance(fact_pack, Mapping) and isinstance(fact_pack.get("fact_metadata"), Mapping)
        else {}
    )
    exact_keys = tuple(str(key) for key in (fact_pack.get("exact_keys") or facts.keys()) if str(key).strip()) if isinstance(fact_pack, Mapping) else tuple(str(key) for key in facts)
    adjacent_keys = tuple(str(key) for key in (fact_pack.get("adjacent_keys") or ()) if str(key).strip()) if isinstance(fact_pack, Mapping) else ()
    client_facets = client_facets_from_text(client_message)
    if not enabled:
        return {
            "schema_version": ANSWER_COVERAGE_PLAN_SCHEMA_VERSION,
            "enabled": False,
            "env": RELIABLE_ANSWERER_STEP1_ENV,
            "bypass_reason": reliable_answerer_step1_bypass_reason(client_message, context=context),
            "client_facets": list(client_facets),
            "covered_facets": [],
            "missing_facets": [],
            "blocked_facets": [],
            "requested_scope": _requested_scope_from_pack(fact_pack),
            "must_not_handoff_whole_answer": False,
        }
    requested_scope = _requested_scope_from_pack(fact_pack)
    covered: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    blocked: list[dict[str, str]] = []
    for facet in client_facets:
        matching_exact: list[str] = []
        blocked_keys: list[str] = []
        for key in exact_keys:
            key_text = f"{key} {facts.get(key) or ''} {fact_meta.get(key, {}) if isinstance(fact_meta, Mapping) else ''}"
            if not _fact_covers_facet(facet, key_text):
                continue
            meta = fact_meta.get(key) if isinstance(fact_meta, Mapping) and isinstance(fact_meta.get(key), Mapping) else {}
            if _fact_blocked_for_facet(facet, meta, requested_scope=requested_scope):
                blocked_keys.append(key)
                continue
            matching_exact.append(key)
        if matching_exact:
            covered.append({"facet": facet, "fact_keys": matching_exact[:8], "coverage": "covered", "source": "confirmed_kb"})
            continue
        if blocked_keys:
            blocked.append({"facet": facet, "reason": "venue_sensitive_fact_without_matching_venue", "fact_keys": blocked_keys[:8]})
            continue
        adjacent_matches = [
            key
            for key in adjacent_keys
            if _fact_covers_facet(facet, f"{key} {facts.get(key) or ''} {fact_meta.get(key, {}) if isinstance(fact_meta, Mapping) else ''}")
        ]
        reason = "only_adjacent_facts" if adjacent_matches else "no_confirmed_fact"
        missing.append({"facet": facet, "reason": reason})
    return {
        "schema_version": ANSWER_COVERAGE_PLAN_SCHEMA_VERSION,
        "enabled": enabled,
        "env": RELIABLE_ANSWERER_STEP1_ENV,
        "client_facets": list(client_facets),
        "covered_facets": covered,
        "missing_facets": missing,
        "blocked_facets": blocked,
        "requested_scope": requested_scope,
        "must_not_handoff_whole_answer": True,
    }


def reliable_answerer_prompt_block(context: Optional[Mapping[str, Any]], plan: Mapping[str, Any]) -> str:
    if not reliable_answerer_step1_enabled(context):
        return ""
    if not plan.get("enabled"):
        return ""
    covered = ", ".join(str(item.get("facet") or "") for item in plan.get("covered_facets") or [] if isinstance(item, Mapping))
    missing = ", ".join(str(item.get("facet") or "") for item in plan.get("missing_facets") or [] if isinstance(item, Mapping))
    blocked = ", ".join(str(item.get("facet") or "") for item in plan.get("blocked_facets") or [] if isinstance(item, Mapping))
    state = []
    if covered:
        state.append(f"покрыто фактами: {covered}")
    if missing:
        state.append(f"нет проверенного факта: {missing}")
    if blocked:
        state.append(f"заблокировано по скоупу/площадке: {blocked}")
    state_text = "; ".join(state) or "проверенных фактов нет"
    return (
        "Надёжный ответчик:\n"
        "- Если в фактах есть проверенный ответ хотя бы на часть вопроса — ответь на эту часть.\n"
        "- Если другая часть требует проверки менеджера — не сдавай весь ответ: напиши, что именно уточнит менеджер.\n"
        "- Не обещай места, бронь, запись, оплату, наличие группы без отдельного проверенного факта.\n"
        "- Порядок для составного вопроса: известные пункты → что проверит менеджер → один следующий шаг.\n"
        "- Не используй CRM/Tallanto/customer memory как источник цен/расписания/адресов/дат/мест.\n"
        f"План покрытия вопроса: {state_text}."
    )


def covered_facets_in_text(draft_text: str, plan: Mapping[str, Any]) -> tuple[str, ...]:
    text = _normalize_fact_match_text(draft_text)
    facets: list[str] = []
    for item in plan.get("covered_facets") or []:
        if not isinstance(item, Mapping):
            continue
        facet = str(item.get("facet") or "").strip()
        if not facet:
            continue
        if facet == "other":
            if text and not _HANDOFF_ONLY_RE.search(text):
                facets.append(facet)
            continue
        if any(re.search(pattern, text, re.I) for pattern in _FACET_PATTERNS.get(facet, ())):
            facets.append(facet)
    return tuple(dict.fromkeys(facets))


def missing_facets_acknowledged(draft_text: str, plan: Mapping[str, Any]) -> tuple[str, ...]:
    text = _normalize_fact_match_text(draft_text)
    if not _MISSING_ACK_RE.search(text):
        return ()
    return tuple(
        str(item.get("facet") or "").strip()
        for item in [*(plan.get("missing_facets") or []), *(plan.get("blocked_facets") or [])]
        if isinstance(item, Mapping) and str(item.get("facet") or "").strip()
    )


def availability_promise_detected(draft_text: str, plan: Optional[Mapping[str, Any]] = None) -> bool:
    del plan
    return bool(_AVAILABILITY_PROMISE_RE.search(_normalize_fact_match_text(draft_text)))


def whole_handoff_detected(draft_text: str, plan: Optional[Mapping[str, Any]] = None) -> bool:
    text = _normalize_fact_match_text(draft_text)
    if not _HANDOFF_ONLY_RE.search(text):
        return False
    if plan and covered_facets_in_text(text, plan):
        return False
    return True


def reliable_answerer_trace(draft_text: str, plan: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    active_plan = plan if isinstance(plan, Mapping) else {}
    return {
        "schema_version": RELIABLE_ANSWERER_TRACE_SCHEMA_VERSION,
        "enabled": bool(active_plan.get("enabled")),
        "answer_coverage_plan": dict(active_plan),
        "covered_facets_in_text": list(covered_facets_in_text(draft_text, active_plan)),
        "missing_facets_acknowledged": list(missing_facets_acknowledged(draft_text, active_plan)),
        "whole_handoff_detected": whole_handoff_detected(draft_text, active_plan),
        "availability_promise_detected": availability_promise_detected(draft_text, active_plan),
    }


def apply_reliable_answerer_output_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not reliable_answerer_step1_enabled(context):
        return result
    bypass_reason = reliable_answerer_step1_bypass_reason(
        client_message,
        context=context,
        result=result,
        metadata=result.metadata,
    )
    if bypass_reason:
        return _strip_reliable_answerer_metadata(result, bypass_reason)
    plan = _coverage_plan_from_metadata(result.metadata)
    trace = reliable_answerer_trace(result.draft_text, plan)
    metadata = _metadata_with_reliable_trace(result.metadata, trace)
    if trace["availability_promise_detected"] and "availability" not in covered_facets_in_text(result.draft_text, plan):
        flags = tuple(dict.fromkeys([*result.safety_flags, "reliable_answerer_availability_promise_blocked"]))
        checklist = tuple(
            dict.fromkeys(
                [
                    *result.manager_checklist,
                    "Нельзя обещать наличие места/группы/смены без отдельного live-факта; менеджер должен проверить доступность.",
                ]
            )
        )
        metadata["reliable_answerer_availability_promise_blocked"] = True
        return replace(
            result,
            route="draft_for_manager",
            draft_text=(
                "Наличие места или группы нужно проверить у менеджера: он сверит актуальную доступность и подскажет следующий шаг."
            ),
            safety_flags=flags,
            manager_checklist=checklist,
            forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, "availability_promise"])),
            metadata=metadata,
        )
    return replace(result, metadata=metadata)


def preserve_partial_answer_for_live_status(
    result: SubscriptionDraftResult,
    *,
    reason: str,
    checklist_item: str,
) -> SubscriptionDraftResult | None:
    plan = _coverage_plan_from_metadata(result.metadata)
    if not isinstance(plan, Mapping) or not plan.get("enabled"):
        return None
    trace = reliable_answerer_trace(result.draft_text, plan)
    if trace["availability_promise_detected"]:
        return None
    if not trace["covered_facets_in_text"]:
        return None
    metadata = _metadata_with_reliable_trace(result.metadata, trace)
    metadata[reason] = True
    metadata["reliable_answerer_live_status_partial_preserved"] = True
    flags = tuple(dict.fromkeys([*result.safety_flags, reason, "reliable_answerer_live_status_partial_preserved"]))
    checklist = tuple(dict.fromkeys([*result.manager_checklist, checklist_item]))
    return replace(
        result,
        route="draft_for_manager",
        draft_text=_append_live_status_caveat(result.draft_text),
        safety_flags=flags,
        manager_checklist=checklist,
        metadata=metadata,
    )


def _coverage_plan_from_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(metadata.get("answer_coverage_plan"), Mapping):
        return metadata["answer_coverage_plan"]  # type: ignore[index]
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    if isinstance(direct.get("answer_coverage_plan"), Mapping):
        return direct["answer_coverage_plan"]  # type: ignore[index]
    return {}


def _metadata_with_reliable_trace(metadata: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    result["reliable_answerer"] = dict(trace)
    direct = dict(result.get("direct_path") or {}) if isinstance(result.get("direct_path"), Mapping) else {}
    if direct:
        direct["reliable_answerer"] = dict(trace)
        result["direct_path"] = direct
    return result


def _strip_reliable_answerer_metadata(result: SubscriptionDraftResult, reason: str) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    metadata.pop("answer_coverage_plan", None)
    metadata.pop("reliable_answerer", None)
    metadata["reliable_answerer_bypassed_reason"] = reason
    if isinstance(metadata.get("direct_path"), Mapping):
        direct = dict(metadata["direct_path"])  # type: ignore[index]
        direct.pop("answer_coverage_plan", None)
        direct.pop("reliable_answerer", None)
        direct["reliable_answerer_bypassed_reason"] = reason
        metadata["direct_path"] = direct
    return replace(result, metadata=metadata)


def _requested_scope_from_pack(fact_pack: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(fact_pack, Mapping):
        return "unspecified"
    llm = fact_pack.get("llm_retrieve") if isinstance(fact_pack.get("llm_retrieve"), Mapping) else {}
    venue_scope = llm.get("venue_scope") if isinstance(llm.get("venue_scope"), Mapping) else {}
    return normalize_requested_scope(venue_scope.get("requested_scope") or fact_pack.get("requested_scope") or "unspecified")


def _fact_covers_facet(facet: str, key_text: str) -> bool:
    if facet == "other":
        return bool(str(key_text or "").strip())
    normalized = _normalize_fact_match_text(key_text)
    return any(re.search(pattern, normalized, re.I) for pattern in _FACT_FACET_PATTERNS.get(facet, ()))


def _fact_blocked_for_facet(facet: str, meta: Mapping[str, Any], *, requested_scope: str) -> bool:
    venue = normalize_fact_venue(meta.get("venue") or VENUE_SCOPE_ANY)
    requested = normalize_requested_scope(requested_scope)
    if facet in _VENUE_SENSITIVE_FACETS and venue in {"", "unspecified", VENUE_SCOPE_ANY}:
        return True
    if requested not in {"unspecified", VENUE_SCOPE_ANY} and venue not in {VENUE_SCOPE_ANY, requested}:
        return True
    return False


def _append_live_status_caveat(draft_text: str) -> str:
    base = _normalize_output_sanitizer_text(draft_text)
    caveat = "Менеджер отдельно проверит актуальное наличие места или группы: до live-проверки не обещаем, что место доступно."
    if _normalize_fact_match_text(caveat) in _normalize_fact_match_text(base):
        return base
    if not base:
        return caveat
    return f"{base}\n\n{caveat}"


def _has_p0_signal(
    client_message: str,
    *,
    context: Optional[Mapping[str, Any]],
    result: Optional[SubscriptionDraftResult],
    metadata: Optional[Mapping[str, Any]],
) -> bool:
    text = _normalize_fact_match_text(client_message)
    if _P0_REFUND_RE.search(text) or _P0_COMPLAINT_RE.search(text) or _P0_LEGAL_RE.search(text):
        return True
    if _P0_PAYMENT_RE.search(text) and _P0_ACCESS_PROBLEM_RE.search(text):
        return True
    if result is not None:
        if str(result.risk_level or "").strip().casefold() in {"high", "p0", "critical", "high_risk"}:
            return True
        if any(_P0_FLAG_RE.search(str(flag or "")) for flag in result.safety_flags):
            return True
    meta = metadata if isinstance(metadata, Mapping) else result.metadata if result is not None and isinstance(result.metadata, Mapping) else {}
    if _metadata_has_model_p0(meta):
        return True
    if isinstance(context, Mapping):
        for source in (context, context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), Mapping) else {}):
            if not isinstance(source, Mapping):
                continue
            funnel = source.get("funnel_state") if isinstance(source.get("funnel_state"), Mapping) else {}
            if any(_P0_FLAG_RE.search(str(funnel.get(key) or "")) for key in ("lead_stage", "next_step_type", "risk_level")):
                return True
            latch = source.get("p0_latch") if isinstance(source.get("p0_latch"), Mapping) else {}
            if latch and (latch.get("active") or latch.get("had_hard_p0_claim")):
                return True
            risk_flags = source.get("risk_flags")
            if isinstance(risk_flags, Sequence) and not isinstance(risk_flags, (str, bytes, bytearray)):
                if any(_P0_FLAG_RE.search(str(flag or "")) for flag in risk_flags):
                    return True
    return False


def _metadata_has_model_p0(metadata: Mapping[str, Any]) -> bool:
    meta = metadata.get("direct_path_model_p0") if isinstance(metadata.get("direct_path_model_p0"), Mapping) else {}
    if not meta and isinstance(metadata.get("direct_path"), Mapping):
        direct = metadata["direct_path"]  # type: ignore[index]
        meta = direct.get("model_p0") if isinstance(direct.get("model_p0"), Mapping) else {}
    if not isinstance(meta, Mapping):
        return False
    risk_level = str(meta.get("risk_level") or "").strip().casefold()
    return bool(meta.get("is_p0")) or risk_level in {"high", "p0", "critical", "high_risk"} or bool(meta.get("p0_kind"))


def _has_cross_brand_signal(
    client_message: str,
    *,
    context: Optional[Mapping[str, Any]],
    result: Optional[SubscriptionDraftResult],
    metadata: Optional[Mapping[str, Any]],
) -> bool:
    text = _normalize_fact_match_text(client_message)
    has_foton = bool(_FOTON_RE.search(text))
    has_unpk = bool(_UNPK_RE.search(text))
    if has_foton and has_unpk:
        return True
    active_brand = ""
    if isinstance(context, Mapping):
        active_brand = str(context.get("active_brand") or context.get("brand") or "").strip().casefold()
    if active_brand == "foton" and has_unpk:
        return True
    if active_brand == "unpk" and has_foton:
        return True
    meta = metadata if isinstance(metadata, Mapping) else result.metadata if result is not None and isinstance(result.metadata, Mapping) else {}
    gate = meta.get("authoritative_output_gate") if isinstance(meta.get("authoritative_output_gate"), Mapping) else {}
    findings = gate.get("findings") if isinstance(gate.get("findings"), Sequence) and not isinstance(gate.get("findings"), (str, bytes, bytearray)) else ()
    return any("brand" in str(item.get("code") if isinstance(item, Mapping) else item).casefold() for item in findings)
