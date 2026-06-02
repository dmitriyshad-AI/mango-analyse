from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.contracts import ChannelMessage
from mango_mvp.channels.answer_contract import build_answer_contract
from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.conversation_intent_plan import build_conversation_intent_plan
from mango_mvp.channels.dialogue_memory import DialogueMemory, build_dialogue_memory
from mango_mvp.channels.fact_retrieval import key_matches, select_confirmed_facts as select_recall_confirmed_facts
from mango_mvp.channels.fact_scope_spec import (
    FACT_COMPATIBLE_NEIGHBOR_SCOPES,
    detect_fact_scopes,
    fact_scopes_allowed,
    scope_family_for,
)
from mango_mvp.channels.few_shot_reference import (
    build_few_shot_reference,
    build_gold_answer_context,
    build_gold_answers_v3_summary,
)
from mango_mvp.channels.pilot_context import PilotContext, build_pilot_context
from mango_mvp.knowledge_base.fact_registry import classify_fact_types, fact_type_from_key
from mango_mvp.knowledge_base.kc_context import limit_context_chunks


NO_KNOWLEDGE_SNAPSHOT_VERSION = "knowledge_snapshot_missing"
MAX_KNOWLEDGE_SNIPPETS = 8
MAX_KNOWLEDGE_SNIPPET_CHARS = 700
MAX_KNOWLEDGE_CONTEXT_CHARS = 4500
AUTONOMY_MATRIX_SAFE_TOPIC_IDS = {
    "theme:001_pricing",
    "theme:005_discounts",
    "theme:006_installment",
    "theme:007_matkap_payment",
    "theme:008_tax_deduction",
    "theme:011_contract",
    "theme:012_certificates",
    "theme:013_schedule",
    "theme:014_format",
    "theme:015_address",
    "theme:016_program",
    "theme:018_materials_homework",
    "theme:019a_positive_feedback",
    "theme:020_enrollment",
    "theme:021_continuation",
    "theme:022_age_level_testing",
    "theme:023_trial_class",
    "theme:024_account_access",
    "theme:025_missing_links_access",
    "theme:026_camp_general",
    "theme:027_camp_living_conditions",
    "theme:028_transport_logistics",
    "service:S5_general_consultation",
}

_FRESH_STATUSES = {"fresh", "fresh_verified", "verified", "document_verified", "allowed_after_fact_check"}
_BLOCKING_STATUSES = {
    "metadata_only",
    "unknown",
    "stale",
    "stale_or_conflicting",
    "needs_manager_confirmation",
    "internal_only",
    "do_not_use",
    "missing",
}
_FORBIDDEN_SNIPPET_STATUSES = {"internal_only", "do_not_use"}
_TOPIC_REQUIRED_FACT_KEYS = {
    "pricing": ("prices.current",),
    "price": ("prices.current",),
    "payment_method": ("payment_methods.current",),
    "payment_status": ("payment_methods.current",),
    "discount": ("discounts.current",),
    "installment": ("installment_terms.current",),
    "schedule": ("schedule.current",),
    "refund": ("refund_policy.current",),
    "matkap": ("matkap_documents.current",),
    "tax": ("tax_deduction_procedure.current",),
    "document": ("documents.current",),
    "platform": ("platform.current",),
    "account_access": ("platform.current",),
    "trial": ("trial_class.current",),
    "program": ("programs.current",),
    "transport": ("transport.current",),
    "logistics": ("transport.current",),
}


@dataclass(frozen=True)
class KnowledgeSnapshotContext:
    facts_context: Mapping[str, Any]
    confirmed_facts: Mapping[str, Any]
    knowledge_snippets: tuple[str, ...]
    missing_facts: tuple[str, ...]
    context_warnings: tuple[str, ...]
    knowledge_base_version: str


def build_telegram_pilot_context(
    message: ChannelMessage | str,
    *,
    active_brand: str = "unknown",
    brand_policy: Mapping[str, Any] | None = None,
    payment_context: Mapping[str, Any] | None = None,
    theme: Mapping[str, Any] | str | None = None,
    rop_policy: Mapping[str, Any] | None = None,
    kc_snapshot: Mapping[str, Any] | None = None,
    snapshot_path: str | Path | None = None,
    recent_messages: Sequence[str] = (),
    client_identity: Mapping[str, Any] | None = None,
    customer_summary: str = "",
    amo_context: Mapping[str, Any] | None = None,
    tallanto_context: Mapping[str, Any] | None = None,
    timeline_context: Mapping[str, Any] | None = None,
    risk_flags: Sequence[str] = (),
    known_slots: Mapping[str, Any] | None = None,
    dialogue_memory: Mapping[str, Any] | DialogueMemory | None = None,
    session_id: str = "",
) -> PilotContext:
    """Build PilotContext for Telegram manager drafts from a compact KC snapshot."""

    current_message = message.text if isinstance(message, ChannelMessage) else str(message or "")
    merged_policy = merge_theme_and_rop_policy(theme=theme, rop_policy=rop_policy)
    snapshot, snapshot_warnings = _load_snapshot(kc_snapshot=kc_snapshot, snapshot_path=snapshot_path)
    memory = build_dialogue_memory(
        current_message=current_message,
        active_brand=active_brand,
        recent_messages=recent_messages,
        known_slots=known_slots or {},
        previous_memory=dialogue_memory,
        session_id=session_id,
    )
    memory_view = memory.to_prompt_view()
    memory_known_slots = memory_view.get("known_slots") if isinstance(memory_view.get("known_slots"), Mapping) else {}
    merged_known_slots = {**dict(known_slots or {}), **dict(memory_known_slots)}
    base_topic_id = _topic_id(theme=theme, rop_policy=merged_policy)
    intent_plan = build_conversation_intent_plan(
        current_message=current_message,
        active_brand=active_brand,
        topic_id=base_topic_id,
        known_slots=merged_known_slots,
        dialogue_memory_view=memory_view,
        recent_messages=recent_messages,
    )
    retrieval_topics = tuple(dict.fromkeys([*intent_plan.answer_topics, *intent_plan.topic_roles]))
    held_state = replace(
        memory.held_state,
        active_fact_scope=intent_plan.fact_scope or memory.held_state.active_fact_scope,
        active_topics=retrieval_topics or memory.held_state.active_topics,
        required_fact_keys=tuple(intent_plan.required_fact_keys) or memory.held_state.required_fact_keys,
    )
    memory = replace(memory, held_state=held_state)
    memory_view = memory.to_prompt_view()
    intent_view = intent_plan.to_prompt_view()
    safety_decision = classify_answer_safety(
        client_message=current_message,
        context={"conversation_intent_plan": intent_view, "dialogue_memory_view": memory_view, "recent_messages": recent_messages},
        topic_id=intent_plan.topic_id,
    )
    policy_for_snapshot = dict(merged_policy)
    policy_for_snapshot.setdefault("topic_id", intent_plan.topic_id)
    existing_required = _text_list(policy_for_snapshot.get("required_fact_keys"))
    held_retrieval = memory.held_state.retrieval_context()
    held_required = _text_list(held_retrieval.get("required_fact_keys"))
    policy_for_snapshot["required_fact_keys"] = list(
        dict.fromkeys([*existing_required, *intent_plan.required_fact_keys, *held_required])
    )
    held_scope = _clean_text(held_retrieval.get("active_fact_scope"))
    if intent_plan.fact_scope:
        policy_for_snapshot["fact_scope"] = intent_plan.fact_scope
    elif held_scope:
        policy_for_snapshot["fact_scope"] = held_scope
    active_topics = tuple(
        dict.fromkeys(
            [
                *intent_plan.answer_topics,
                *intent_plan.topic_roles,
                *_text_list(held_retrieval.get("active_topics")),
            ]
        )
    )
    if active_topics:
        policy_for_snapshot["active_topics"] = list(active_topics)
    if intent_plan.blocked_neighbor_scopes:
        policy_for_snapshot["blocked_neighbor_scopes"] = list(intent_plan.blocked_neighbor_scopes)
    knowledge_query = intent_plan.fact_query_text or _contextual_message_for_knowledge_lookup(current_message, known_slots=merged_known_slots)
    knowledge_query = _contextual_message_with_recent_product(knowledge_query, current_message=current_message, recent_messages=recent_messages)
    snapshot_context = build_knowledge_snapshot_context(
        message_text=knowledge_query,
        theme={"topic_id": intent_plan.topic_id},
        rop_policy=policy_for_snapshot,
        kc_snapshot=snapshot,
        snapshot_warnings=snapshot_warnings,
        active_brand=active_brand,
    )
    policy_for_prompt = dict(policy_for_snapshot)
    if snapshot_context.facts_context.get("required_fact_keys"):
        policy_for_prompt.setdefault("required_fact_keys", snapshot_context.facts_context["required_fact_keys"])
    topic_id = _topic_id(theme=theme, rop_policy=policy_for_prompt)
    few_shot_reference = build_few_shot_reference(
        message_text=current_message,
        active_brand=active_brand,
        topic_id=topic_id,
        confirmed_facts=snapshot_context.confirmed_facts,
        missing_facts=snapshot_context.missing_facts,
        known_slots=merged_known_slots,
    )
    gold_answer_context = build_gold_answer_context(
        message_text=current_message,
        active_brand=active_brand,
        topic_id=topic_id,
        confirmed_facts=snapshot_context.confirmed_facts,
    )
    gold_answers_v3 = build_gold_answers_v3_summary(gold_answer_context)
    answer_contract = build_answer_contract(
        active_brand=active_brand,
        conversation_intent_plan=intent_view,
        dialogue_memory_view=memory_view,
        safety_decision=safety_decision.to_json_dict(),
        known_slots=merged_known_slots,
        confirmed_facts=snapshot_context.confirmed_facts,
    )
    policy_for_prompt.setdefault(
        "autonomy_policy",
        {
            "allow_autonomous": bool(
                topic_id in AUTONOMY_MATRIX_SAFE_TOPIC_IDS
                and snapshot_context.facts_context.get("client_safe_fact_verified") is True
                and not snapshot_context.missing_facts
            ),
            "allowed_topic_ids": [topic_id] if topic_id in AUTONOMY_MATRIX_SAFE_TOPIC_IDS else [],
            "default": "draft_for_manager_or_manager_only",
            "fact_requirement": "client_safe_fact_verified",
            "p0_overrides_autonomy": True,
        },
    )

    return build_pilot_context(
        message,
        active_brand=active_brand,
        brand_policy=brand_policy or {},
        payment_context=payment_context or {},
        recent_messages=recent_messages,
        client_identity=client_identity,
        customer_summary=customer_summary,
        amo_context=amo_context,
        tallanto_context=tallanto_context,
        timeline_context=timeline_context,
        rop_policy=policy_for_prompt,
        facts_context=snapshot_context.facts_context,
        confirmed_facts=snapshot_context.confirmed_facts,
        missing_facts=snapshot_context.missing_facts,
        required_fact_keys=tuple(snapshot_context.facts_context.get("required_fact_keys", ())),
        knowledge_snippets=snapshot_context.knowledge_snippets,
        context_warnings=snapshot_context.context_warnings,
        knowledge_base_version=snapshot_context.knowledge_base_version,
        risk_flags=risk_flags,
        dialogue_memory_view=memory_view,
        conversation_intent_plan=intent_view,
        answer_contract=answer_contract.to_prompt_view(),
        gold_answers_v3=gold_answers_v3,
        gold_answer_context=gold_answer_context,
        answer_quality_reference=few_shot_reference,
        few_shot_style_examples=tuple(few_shot_reference.get("style_examples") or ()),
        few_shot_correction_examples=tuple(few_shot_reference.get("correction_examples") or ()),
    )


def _contextual_message_for_knowledge_lookup(message_text: str, *, known_slots: Mapping[str, Any]) -> str:
    additions: list[str] = []
    grade = _clean_text(known_slots.get("grade") or known_slots.get("class") or known_slots.get("student_grade"))
    subject = _clean_text(known_slots.get("subject") or known_slots.get("course_subject") or known_slots.get("interest_subject"))
    course_format = _clean_text(known_slots.get("format") or known_slots.get("course_format") or known_slots.get("preferred_format"))
    if grade and "класс" not in grade.casefold():
        additions.append(f"{grade} класс")
    elif grade:
        additions.append(grade)
    if subject:
        additions.append(subject)
    if course_format:
        additions.append(course_format)
    if not additions:
        return str(message_text or "")
    return " ".join([str(message_text or ""), "Контекст диалога:", *additions]).strip()


def _contextual_message_with_recent_product(
    query: str,
    *,
    current_message: str,
    recent_messages: Sequence[str],
) -> str:
    current = _normalize_match_text(current_message)
    recent = _normalize_match_text(" ".join(str(item or "") for item in recent_messages[-8:]))
    if not current or not recent:
        return query
    followup_markers = (
        "трансфер",
        "добир",
        "дорог",
        "заезд",
        "из москв",
        "прожив",
        "питан",
        "что входит",
        "мест",
        "смен",
        "программ",
        "там",
    )
    if not any(marker in current for marker in followup_markers):
        return query
    additions: list[str] = []
    if "выездн" in current:
        additions.append("ЛВШ Менделеево выездной лагерь")
    elif any(marker in recent for marker in ("лвш", "менделеево", "выездн")):
        additions.append("ЛВШ Менделеево выездной лагерь")
    elif ("городск" in recent and "лагер" in recent) or "без прожив" in recent or ("летн" in recent and "школ" in recent):
        additions.append("городской лагерь")
    if not additions:
        return query
    return " ".join([query, "Контекст продукта:", *additions]).strip()


def build_telegram_pilot_context_from_snapshot(
    message: ChannelMessage | str,
    *,
    snapshot_path: str | Path | None = None,
    kc_snapshot: Mapping[str, Any] | None = None,
    topic_id: str = "",
    required_fact_keys: Sequence[str] = (),
    rop_policy: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> PilotContext:
    """Compatibility wrapper for dry-run scripts created during the KB build."""

    merged_policy = dict(rop_policy or {})
    if topic_id:
        merged_policy.setdefault("topic_id", topic_id)
    if required_fact_keys:
        merged_policy.setdefault("required_fact_keys", list(required_fact_keys))
    return build_telegram_pilot_context(
        message,
        theme={"topic_id": topic_id} if topic_id else None,
        rop_policy=merged_policy,
        kc_snapshot=kc_snapshot,
        snapshot_path=snapshot_path,
        **kwargs,
    )


def build_knowledge_snapshot_context(
    *,
    message_text: str,
    theme: Mapping[str, Any] | str | None = None,
    rop_policy: Mapping[str, Any] | None = None,
    kc_snapshot: Mapping[str, Any] | None = None,
    snapshot_warnings: Sequence[str] = (),
    active_brand: str = "unknown",
) -> KnowledgeSnapshotContext:
    policy = merge_theme_and_rop_policy(theme=theme, rop_policy=rop_policy)
    required_fact_keys = required_fact_keys_for_message(message_text, theme=theme, rop_policy=policy)
    active = _normalize_active_brand(active_brand)
    policy_fact_scope = _clean_text(policy.get("fact_scope"))
    if active == "unpk" and policy_fact_scope == "trial_offline":
        required_fact_keys = tuple(
            key for key in required_fact_keys if str(key or "").split(".", 1)[0] != "trial_online_fragment"
        )
    if not kc_snapshot:
        return _missing_snapshot_context(required_fact_keys, snapshot_warnings=snapshot_warnings)

    version = _snapshot_version(kc_snapshot)
    topic_id = _topic_id(theme=theme, rop_policy=policy)
    required_fact_types = _expand_required_fact_types(
        {fact_type_from_key(key) for key in required_fact_keys},
        topic_id=_topic_id(theme=theme, rop_policy=policy),
        query=message_text,
    )
    fact_scope = policy_fact_scope
    if fact_scope == "city_day_camp":
        required_fact_types.update({"camp_city", "program", "price", "deadline", "location"})
    elif fact_scope == "residential_lvsh":
        required_fact_types.update({"camp_lvsh", "program", "price", "deadline", "location"})
    active_topics = tuple(_text_list(policy.get("active_topics")))
    blocked_neighbor_scopes = _effective_recall_blocked_scopes(
        _text_list(policy.get("blocked_neighbor_scopes")),
        required_fact_keys=required_fact_keys,
        fact_scope=_clean_text(policy.get("fact_scope")),
        active_brand=active_brand,
    )
    facts = _records(kc_snapshot.get("facts"))
    sources = _records(kc_snapshot.get("sources"))
    scoped_facts = [
        fact
        for fact in facts
        if _record_matches_scope_only(fact, fact_scope=fact_scope, blocked_neighbor_scopes=blocked_neighbor_scopes)
    ]
    if fact_scope or blocked_neighbor_scopes:
        facts = scoped_facts
    chunks = [
        chunk
        for chunk in _chunk_records(kc_snapshot, active_brand=active)
        if _chunk_matches_scope_only(chunk, fact_scope=fact_scope, blocked_neighbor_scopes=blocked_neighbor_scopes)
    ]
    selected_chunks = limit_context_chunks(
        chunks,
        query=f"{topic_id} {message_text}",
        required_fact_keys=required_fact_keys,
        active_brand=active,
        max_chunks=MAX_KNOWLEDGE_SNIPPETS,
        max_chunk_chars=MAX_KNOWLEDGE_SNIPPET_CHARS,
        total_char_limit=MAX_KNOWLEDGE_CONTEXT_CHARS,
    )
    selected_chunks = [
        chunk
        for chunk in selected_chunks
        if _chunk_matches_context(
            chunk,
            required_fact_types=required_fact_types,
            topic_id=topic_id,
            query=message_text,
            fact_scope=fact_scope,
            blocked_neighbor_scopes=blocked_neighbor_scopes,
        )
    ]
    confirmed_facts = _select_confirmed_facts(
        facts,
        required_fact_types=required_fact_types,
        required_fact_keys=required_fact_keys,
        active_topics=active_topics,
        topic_id=topic_id,
        query=message_text,
        active_brand=active,
        fact_scope=fact_scope,
        blocked_neighbor_scopes=blocked_neighbor_scopes,
    )
    missing_facts, stale_or_blocked = _missing_fact_keys(
        required_fact_keys=required_fact_keys,
        required_fact_types=required_fact_types,
        facts=facts,
        sources=sources,
        selected_chunk_fact_types={fact_type for chunk in selected_chunks for fact_type in chunk.fact_types},
        confirmed_facts=confirmed_facts,
    )
    warnings = list(snapshot_warnings)
    if not selected_chunks and not confirmed_facts:
        warnings.append("knowledge_context_not_found")
    if missing_facts:
        warnings.extend(("facts_missing", "precise_answer_blocked"))
    if stale_or_blocked:
        warnings.append("facts_stale")

    knowledge_snippets = _knowledge_snippets(selected_chunks)
    if len(knowledge_snippets) < MAX_KNOWLEDGE_SNIPPETS:
        knowledge_snippets = (
            *knowledge_snippets,
            *_manager_pattern_snippets(
                kc_snapshot,
                topic_id=topic_id,
                active_brand=active,
                fact_scope=fact_scope,
                blocked_neighbor_scopes=blocked_neighbor_scopes,
            ),
        )[:MAX_KNOWLEDGE_SNIPPETS]
    precise_answers_allowed = not missing_facts and not stale_or_blocked
    facts_fresh = bool(confirmed_facts) and precise_answers_allowed
    selected_source_ids = _dedupe(
        [chunk.source_id for chunk in selected_chunks]
        + [
            _clean_text(fact.get("source_id"))
            for fact in facts
            if (
                _clean_text(fact.get("fact_id") or fact.get("id")) in confirmed_facts
                and _clean_text(fact.get("source_id"))
            )
        ]
    )
    facts_context: dict[str, Any] = {
        "knowledge_base_version": version,
        "snapshot_found": True,
        "active_brand": active,
        "fresh": facts_fresh,
        "facts_fresh": facts_fresh,
        "client_safe": facts_fresh,
        "client_safe_fact_verified": facts_fresh,
        "autonomy_fact_verified": facts_fresh,
        "missing": bool(missing_facts),
        "facts_missing": bool(missing_facts),
        "stale": bool(stale_or_blocked),
        "facts_stale": bool(stale_or_blocked),
        "precise_answers_allowed": precise_answers_allowed,
        "required_fact_keys": list(required_fact_keys),
        "confirmed_fact_ids": list(confirmed_facts.keys()),
        "selected_chunk_ids": [chunk.chunk_id for chunk in selected_chunks],
        "source_ids": selected_source_ids[:12],
    }
    if fact_scope:
        facts_context["fact_scope"] = fact_scope
    if active_topics:
        facts_context["active_topics"] = list(active_topics)
    if blocked_neighbor_scopes:
        facts_context["blocked_neighbor_scopes"] = list(blocked_neighbor_scopes)
    if confirmed_facts:
        facts_context["confirmed_facts"] = dict(confirmed_facts)

    return KnowledgeSnapshotContext(
        facts_context=facts_context,
        confirmed_facts=confirmed_facts,
        knowledge_snippets=knowledge_snippets,
        missing_facts=missing_facts,
        context_warnings=tuple(_dedupe(warnings)),
        knowledge_base_version=version,
    )


def merge_theme_and_rop_policy(
    *,
    theme: Mapping[str, Any] | str | None = None,
    rop_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(theme, Mapping):
        merged.update(theme)
        if theme.get("theme_id") and not theme.get("topic_id"):
            merged["topic_id"] = theme["theme_id"]
        if theme.get("theme_name") and not theme.get("topic_name"):
            merged["topic_name"] = theme["theme_name"]
    elif theme:
        merged["topic_id"] = str(theme)
    merged.update(dict(rop_policy or {}))
    if merged.get("theme_id") and not merged.get("topic_id"):
        merged["topic_id"] = merged["theme_id"]
    if merged.get("theme_name") and not merged.get("topic_name"):
        merged["topic_name"] = merged["theme_name"]
    return merged


def required_fact_keys_for_message(
    message_text: str,
    *,
    theme: Mapping[str, Any] | str | None = None,
    rop_policy: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    keys: list[str] = []
    for container in (theme if isinstance(theme, Mapping) else None, rop_policy):
        if not isinstance(container, Mapping):
            continue
        for field_name in ("required_fact_keys", "required_facts", "fact_keys"):
            keys.extend(_text_list(container.get(field_name)))
        for fact_type in _text_list(container.get("required_fact_types")):
            keys.append(f"{fact_type}.current")

    topic_text = " ".join(
        _clean_text(value)
        for value in (
            _topic_id(theme=theme, rop_policy=rop_policy),
            rop_policy.get("topic_name") if isinstance(rop_policy, Mapping) else "",
            rop_policy.get("theme_name") if isinstance(rop_policy, Mapping) else "",
        )
        if _clean_text(value)
    ).casefold()
    for marker, marker_keys in _TOPIC_REQUIRED_FACT_KEYS.items():
        if marker in topic_text:
            keys.extend(marker_keys)

    requirement_message = str(message_text or "").split("Контекст диалога:", 1)[0]
    requirement_message = requirement_message.split("Известные данные:", 1)[0]
    requirement_message = requirement_message.split("Нужные факты:", 1)[0]
    text = f"{topic_text} {requirement_message}".casefold()
    is_refund_topic = "theme:009_refund" in topic_text or "refund" in topic_text
    if (
        re.search(r"физтех|олимпиад", text)
        and re.search(r"онлайн|дистанц", text)
        and not re.search(r"не\s+олимпиад|обычн|регулярн", text)
    ):
        keys.append("olympiad_online.current")
    if re.search(r"стоим|цен[аы]|сколько стоит|прайс|руб", text):
        keys.append("prices.current")
    recording_question = re.search(r"запис[ьи]\s+(?:урок|занят)|урок[ауы]\s+в\s+запис|посмотреть\s+запис|запис\w*\s+сохраня", text)
    if not is_refund_topic and re.search(r"распис|когда|во сколько|суббот|воскрес|слот|занят", text) and not recording_question:
        keys.append("schedule.current")
        if re.search(r"по каким дням|выходн|суббот|воскрес|слот", text):
            keys.insert(0, "schedule_weekend.current")
    if recording_question:
        if re.search(r"онлайн|дистанц", text):
            keys.append("online_recordings.current")
        elif re.search(r"очно|офлайн", text):
            keys.append("offline_recordings.current")
        else:
            keys.append("recordings.current")
    if re.search(r"преподав|педагог|учитель|кто\s+вед|кто\s+работает", text):
        keys.append("teachers.current")
    if re.search(r"скид|льгот|промокод|акци", text):
        keys.append("discounts.current")
        if re.search(r"за\s+год|годов|год\s", text):
            keys.insert(0, "discounts_year.current")
        if re.search(r"семестр|полугод", text):
            keys.insert(0, "discounts_semester.current")
    invoice_monthly_question = bool(
        re.search(r"помесяч|каждый\s+месяц|ежемесяч|по\s+месяцам", text)
        and re.search(r"по\s+сч[её]ту|сч[её]т|банковск\w*\s+перевод|реквизит|не\s+рассроч|не\s+долями|не\s+частями", text)
    )
    if re.search(r"рассроч", text) and not invoice_monthly_question:
        keys.append("installment_terms.current")
    payment_method_question = re.search(r"сбп|реквизит|карт|ссылк[ау] на оплат|как\s+оплат|куда\s+оплат|банковск\w*\s+перевод|на\s+сч[её]т|по\s+сч[её]ту", text)
    payment_terms_question = re.search(r"скид|за\s+год|семестр|помесяч|рассроч|долями|частями", text)
    if invoice_monthly_question or (payment_method_question and not payment_terms_question):
        keys.append("payment_methods.current")
    if re.search(r"маткап|материн", text):
        keys.append("matkap_documents.current")
        if re.search(r"сфр|рассматри|срок|сколько|дней|рабоч", text):
            keys.insert(0, "matkap_timeline.current")
    if not is_refund_topic and re.search(r"договор|справ|налог|возврат|чек|квитанц", text):
        keys.append("documents.current")
    if re.search(r"личный кабинет|кабинет|платформ|логин|парол", text):
        keys.append("platform.current")
    if re.search(r"электрон|документооборот|скан-коп", text):
        keys.append("platform_documents.current")
    if re.search(r"формат|онлайн|очно|офлайн|дистанц", text):
        keys.append("formats.current")
    if re.search(r"адрес|где\s+вы|где\s+находит|куда\s+ехать|куда\s+ездить|площадк|метро", text):
        keys.append("locations.current")
    if re.search(r"пробн|фрагмент|попроб", text):
        keys.insert(0, "trial_online_fragment.current")
    if re.search(r"программ|предмет|летн|пробн|чему учат|содержание", text):
        keys.append("programs.current")
    if re.search(r"трансфер|добир|дорог|заезд|из москв|самостоятельн", text):
        keys.append("transport.current")
    return tuple(_dedupe(_stable_fact_key(key) for key in keys if _clean_text(key)))


def _missing_snapshot_context(
    required_fact_keys: Sequence[str],
    *,
    snapshot_warnings: Sequence[str],
) -> KnowledgeSnapshotContext:
    missing = tuple(required_fact_keys) or ("knowledge_snapshot",)
    warnings = _dedupe([*snapshot_warnings, "knowledge_snapshot_missing", "facts_missing", "precise_answer_blocked"])
    return KnowledgeSnapshotContext(
        facts_context={
            "knowledge_base_version": NO_KNOWLEDGE_SNAPSHOT_VERSION,
            "snapshot_found": False,
            "fresh": False,
            "facts_fresh": False,
            "missing": True,
            "facts_missing": True,
            "stale": False,
            "facts_stale": False,
            "precise_answers_allowed": False,
            "required_fact_keys": list(required_fact_keys),
        },
        confirmed_facts={},
        knowledge_snippets=(),
        missing_facts=missing,
        context_warnings=tuple(warnings),
        knowledge_base_version=NO_KNOWLEDGE_SNAPSHOT_VERSION,
    )


def _load_snapshot(
    *,
    kc_snapshot: Mapping[str, Any] | None,
    snapshot_path: str | Path | None,
) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    if kc_snapshot is not None:
        return dict(kc_snapshot), ()
    if snapshot_path is None:
        return None, ()
    try:
        with Path(snapshot_path).open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None, ("knowledge_snapshot_unreadable",)
    if not isinstance(loaded, Mapping):
        return None, ("knowledge_snapshot_invalid",)
    return dict(loaded), ()


def _snapshot_version(snapshot: Mapping[str, Any]) -> str:
    metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), Mapping) else {}
    for value in (
        snapshot.get("run_id"),
        snapshot.get("snapshot_id"),
        snapshot.get("version"),
        metadata.get("run_id"),
        metadata.get("version"),
        snapshot.get("schema_version"),
    ):
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return "kc_knowledge_snapshot_unknown"


def _chunk_records(snapshot: Mapping[str, Any], *, active_brand: str = "unknown") -> list[Mapping[str, Any]]:
    chunks = _records(snapshot.get("chunks") or snapshot.get("knowledge_chunks"))
    result: list[Mapping[str, Any]] = []
    for chunk in chunks:
        if not _record_allowed_for_active_brand(chunk, active_brand=active_brand):
            continue
        if _truthy(chunk.get("forbidden_for_client")):
            continue
        status = _stable_status(chunk.get("freshness_status"))
        if status in _FORBIDDEN_SNIPPET_STATUSES:
            continue
        text = _clean_text(
            chunk.get("text")
            or chunk.get("client_safe_text")
            or chunk.get("manager_text")
            or chunk.get("short_fact"),
            max_chars=1600,
        )
        if not text:
            continue
        if status in _BLOCKING_STATUSES and _has_precise_claim(text):
            continue
        fact_types = _fact_types(chunk)
        result.append(
            {
                **dict(chunk),
                "text": text,
                "title": _clean_text(chunk.get("title") or chunk.get("source_title") or "База знаний"),
                "fact_types": list(fact_types),
                "freshness_status": status,
            }
        )
    return result


def _select_confirmed_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    required_fact_types: set[str],
    required_fact_keys: Sequence[str] = (),
    active_topics: Sequence[str] = (),
    topic_id: str,
    query: str,
    active_brand: str = "unknown",
    fact_scope: str = "",
    blocked_neighbor_scopes: Sequence[str] = (),
) -> dict[str, str]:
    candidates: list[Mapping[str, Any]] = []
    for index, fact in enumerate(facts):
        if not _usable_for_precise_answer(fact, active_brand=active_brand):
            continue
        if not _record_matches_scope_only(
            fact,
            fact_scope=fact_scope,
            blocked_neighbor_scopes=blocked_neighbor_scopes,
        ):
            continue
        fact_id = _clean_text(fact.get("fact_id") or fact.get("id") or f"fact:{index + 1}", max_chars=120)
        fact_key = _clean_text(fact.get("fact_key") or fact_id, max_chars=160)
        is_required_answer = any(
            key_matches(required_key, fact_key) or key_matches(required_key, fact_id)
            for required_key in required_fact_keys
        )
        if is_required_answer and not _record_matches_retrieval_core(
            fact,
            query=query,
            fact_scope=fact_scope,
            blocked_neighbor_scopes=blocked_neighbor_scopes,
            allow_objection_pattern=any(
                str(required_key or "").split(".", 1)[0] == "schedule_weekend"
                for required_key in required_fact_keys
            ),
        ):
            continue
        if not is_required_answer and not _record_matches_context(
            fact,
            required_fact_types=required_fact_types,
            topic_id=topic_id,
            query=query,
            fact_scope=fact_scope,
            blocked_neighbor_scopes=blocked_neighbor_scopes,
        ):
            continue
        score = _fact_match_score(fact, required_fact_types=required_fact_types, topic_id=topic_id, query=query)
        scopes = _record_fact_scopes(
            _scope_match_text(fact),
            fact_types=set(_fact_types(fact)),
        )
        candidates.append(
            {
                "__fact": fact,
                "__score": score,
                "__index": index,
                "brand": _clean_text(fact.get("brand")),
                "fact_key": fact_key,
                "scopes": scopes,
            }
        )

    confirmed: dict[str, str] = {}
    selected = select_recall_confirmed_facts(
        sorted(candidates, key=lambda item: (int(item.get("__score") or 0), -int(item.get("__index") or 0)), reverse=True),
        active_brand=active_brand,
        required_fact_keys=required_fact_keys,
        active_topics=active_topics,
        blocked_scopes=_effective_recall_blocked_scopes(
            blocked_neighbor_scopes,
            required_fact_keys=required_fact_keys,
            fact_scope=fact_scope,
            active_brand=active_brand,
        ),
        k=10,
    )
    for candidate in selected:
        raw_fact = candidate.get("__fact")
        if not isinstance(raw_fact, Mapping):
            continue
        fact = raw_fact
        text = _fact_text(fact)
        if not text:
            continue
        fact_id = _clean_text(fact.get("fact_id") or fact.get("id") or f"fact:{len(confirmed) + 1}", max_chars=120)
        confirmed[fact_id] = text
        if len(confirmed) >= 10:
            break
    return confirmed


def _effective_recall_blocked_scopes(
    blocked_neighbor_scopes: Sequence[str],
    *,
    required_fact_keys: Sequence[str],
    fact_scope: str = "",
    active_brand: str = "",
) -> tuple[str, ...]:
    required = {str(item or "").split(".", 1)[0] for item in required_fact_keys if str(item or "").strip()}
    blocked = [str(item) for item in blocked_neighbor_scopes if str(item).strip()]
    normalized_brand = _normalize_active_brand(active_brand)
    scope = str(fact_scope or "").strip()
    compatible = FACT_COMPATIBLE_NEIGHBOR_SCOPES.get(scope, frozenset())
    if scope == "trial_offline" and normalized_brand != "foton":
        compatible = frozenset()
    if compatible:
        blocked = [scope for scope in blocked if scope not in compatible]
    if "trial_online_fragment" in required and normalized_brand == "foton":
        blocked = [scope for scope in blocked if scope != "trial_online_fragment"]
    return tuple(blocked)


def _missing_fact_keys(
    *,
    required_fact_keys: Sequence[str],
    required_fact_types: set[str],
    facts: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    selected_chunk_fact_types: set[str],
    confirmed_facts: Mapping[str, Any],
) -> tuple[tuple[str, ...], bool]:
    if not required_fact_keys:
        return (), False
    confirmed_fact_types = {
        fact_type
        for fact in facts
        if _clean_text(fact.get("fact_id") or fact.get("id")) in confirmed_facts
        for fact_type in _fact_types(fact)
    }
    confirmed_records = [
        fact
        for fact in facts
        if _clean_text(fact.get("fact_id") or fact.get("id")) in confirmed_facts
    ]
    missing: list[str] = []
    stale_or_blocked = False
    for fact_key in required_fact_keys:
        if any(
            key_matches(fact_key, _clean_text(record.get("fact_key") or record.get("fact_id") or record.get("id")))
            for record in confirmed_records
        ):
            continue
        fact_type = fact_type_from_key(fact_key)
        acceptable_fact_types = _expand_required_fact_types({fact_type}, topic_id="", query="")
        if fact_type == "location" and required_fact_types & {"camp_lvsh", "camp_city"}:
            acceptable_fact_types.update({"camp_lvsh", "camp_city"})
        if fact_type == "schedule":
            acceptable_fact_types.update({"course_parameter", "program"})
        if confirmed_fact_types & acceptable_fact_types:
            continue
        candidate_statuses = [
            _stable_status(record.get("freshness_status"))
            for record in (*facts, *sources)
            if fact_type in _fact_types(record)
        ]
        if fact_type in selected_chunk_fact_types and not candidate_statuses:
            candidate_statuses.append("unknown")
        if any(status in _BLOCKING_STATUSES for status in candidate_statuses):
            stale_or_blocked = True
        missing.append(fact_key)
    return tuple(_dedupe(missing)), stale_or_blocked


def _knowledge_snippets(selected_chunks: Sequence[Any]) -> tuple[str, ...]:
    snippets: list[str] = []
    for chunk in selected_chunks:
        title = _clean_text(getattr(chunk, "title", ""), max_chars=120) or "База знаний"
        text = _clean_text(getattr(chunk, "text", ""), max_chars=MAX_KNOWLEDGE_SNIPPET_CHARS)
        if not text:
            continue
        prefix = f"{title}: "
        snippets.append(_clip_text(f"{prefix}{text}", MAX_KNOWLEDGE_SNIPPET_CHARS))
    return tuple(_dedupe(snippets))


def _manager_pattern_snippets(
    snapshot: Mapping[str, Any],
    *,
    topic_id: str,
    active_brand: str = "unknown",
    fact_scope: str = "",
    blocked_neighbor_scopes: Sequence[str] = (),
) -> tuple[str, ...]:
    snippets: list[str] = []
    for pattern in _records(snapshot.get("manager_answer_patterns")):
        if not _record_allowed_for_active_brand(pattern, active_brand=active_brand):
            continue
        if not _record_matches_scope_only(pattern, fact_scope=fact_scope, blocked_neighbor_scopes=blocked_neighbor_scopes):
            continue
        if topic_id and topic_id not in _text_list(pattern.get("related_theme_ids") or pattern.get("theme_ids")):
            continue
        text = _clean_text(
            pattern.get("client_safe_text")
            or pattern.get("safe_pattern")
            or pattern.get("pattern_summary")
            or pattern.get("manager_safe_text"),
            max_chars=560,
        )
        if not text:
            continue
        snippets.append(_clip_text(f"[Прием менеджера, не факт] {text}", MAX_KNOWLEDGE_SNIPPET_CHARS))
        if len(snippets) >= 2:
            break
    return tuple(_dedupe(snippets))


def _chunk_matches_context(
    chunk: Any,
    *,
    required_fact_types: set[str],
    topic_id: str,
    query: str,
    fact_scope: str = "",
    blocked_neighbor_scopes: Sequence[str] = (),
) -> bool:
    record = {
        "title": getattr(chunk, "title", ""),
        "text": getattr(chunk, "text", ""),
        "fact_types": list(getattr(chunk, "fact_types", ()) or ()),
    }
    return _record_matches_context(
        record,
        required_fact_types=required_fact_types,
        topic_id=topic_id,
        query=query,
        fact_scope=fact_scope,
        blocked_neighbor_scopes=blocked_neighbor_scopes,
    )


def _chunk_matches_scope_only(
    chunk: Any,
    *,
    fact_scope: str = "",
    blocked_neighbor_scopes: Sequence[str] = (),
) -> bool:
    if isinstance(chunk, Mapping):
        record = {
            "title": chunk.get("title", ""),
            "text": chunk.get("text", ""),
            "fact_types": list(chunk.get("fact_types", ()) or ()),
            "fact_key": chunk.get("fact_key", ""),
            "product": chunk.get("product", ""),
            "source_path": chunk.get("source_path", ""),
            "metadata": chunk.get("metadata") if isinstance(chunk.get("metadata"), Mapping) else {},
        }
        return _record_matches_scope_only(record, fact_scope=fact_scope, blocked_neighbor_scopes=blocked_neighbor_scopes)
    record = {
        "title": getattr(chunk, "title", ""),
        "text": getattr(chunk, "text", ""),
        "fact_types": list(getattr(chunk, "fact_types", ()) or ()),
    }
    return _record_matches_scope_only(record, fact_scope=fact_scope, blocked_neighbor_scopes=blocked_neighbor_scopes)


def _record_matches_scope_only(
    record: Mapping[str, Any],
    *,
    fact_scope: str = "",
    blocked_neighbor_scopes: Sequence[str] = (),
) -> bool:
    text = _scope_match_text(record)
    return _record_matches_fact_scope(
        text,
        fact_types=set(_fact_types(record)),
        fact_scope=fact_scope,
        blocked_neighbor_scopes=blocked_neighbor_scopes,
    )


def _scope_match_text(record: Mapping[str, Any]) -> str:
    text = _normalize_match_text(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "fact_scope",
                "fact_key",
                "title",
                "product",
                "source_path",
                "client_safe_text",
                "manager_display_text",
                "short_fact",
                "text",
            )
        )
    )
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    if metadata:
        text = _normalize_match_text(
            f"{text} {metadata.get('path', '')} {metadata.get('source_path', '')} {metadata.get('product', '')} {metadata.get('fact_key', '')}"
        )
    return text


def _record_matches_context(
    record: Mapping[str, Any],
    *,
    required_fact_types: set[str],
    topic_id: str,
    query: str,
    fact_scope: str = "",
    blocked_neighbor_scopes: Sequence[str] = (),
) -> bool:
    fact_types = set(_fact_types(record))
    text = _normalize_match_text(
        f"{record.get('title', '')} {record.get('fact_key', '')} {_fact_text(record)} {record.get('text', '')}"
    )
    query_text = _normalize_match_text(query)
    if _record_is_objection_pattern(text):
        return False
    if not _record_matches_fact_scope(text, fact_types=fact_types, fact_scope=fact_scope, blocked_neighbor_scopes=blocked_neighbor_scopes):
        return False
    if not _record_matches_product_markers(text, query_text):
        return False
    if "discount" not in fact_types and not _record_matches_requested_format(text, query_text):
        return False
    if not _record_matches_requested_class(text, query_text):
        return False
    if _record_is_unrequested_special_product(text, query_text):
        return False
    if _query_asks_for_date(query_text) and not _record_answers_date_question(text):
        return False
    if required_fact_types and fact_types & required_fact_types:
        return True
    related_theme_ids = set(_text_list(record.get("related_theme_ids") or record.get("theme_ids") or record.get("topics")))
    if topic_id and topic_id in related_theme_ids:
        return True
    if required_fact_types or topic_id:
        return False
    return any(term in text for term in _query_terms(query))


_EXPLICIT_SCOPE_REQUIRED = {
    "city_day_camp",
    "residential_lvsh",
    "trial_offline",
    "trial_online_fragment",
    "offline_recordings",
    "online_recordings",
    "camp_extra_facts",
    "olympiad_online",
}


def _record_matches_retrieval_core(
    record: Mapping[str, Any],
    *,
    query: str,
    fact_scope: str = "",
    blocked_neighbor_scopes: Sequence[str] = (),
    allow_objection_pattern: bool = False,
) -> bool:
    fact_types = set(_fact_types(record))
    text = _normalize_match_text(
        f"{record.get('title', '')} {record.get('fact_key', '')} {_fact_text(record)} {record.get('text', '')}"
    )
    query_text = _normalize_match_text(query)
    if _record_is_objection_pattern(text) and not allow_objection_pattern:
        return False
    if not _record_matches_fact_scope(text, fact_types=fact_types, fact_scope=fact_scope, blocked_neighbor_scopes=blocked_neighbor_scopes):
        return False
    if not _record_matches_product_markers(text, query_text):
        return False
    if not _record_matches_requested_format(text, query_text) and not (
        fact_scope == "trial_offline" and "фрагмент" in text
    ):
        return False
    if not _record_matches_requested_class(text, query_text):
        return False
    if _record_is_unrequested_special_product(text, query_text):
        return False
    if _query_asks_for_date(query_text) and not _record_answers_date_question(text):
        return False
    return True


def _record_matches_fact_scope(
    text: str,
    *,
    fact_types: set[str],
    fact_scope: str = "",
    blocked_neighbor_scopes: Sequence[str] = (),
) -> bool:
    requested = _clean_text(fact_scope)
    blocked = {str(item) for item in blocked_neighbor_scopes if str(item).strip()}
    if not requested and not blocked:
        return True
    record_scopes = _record_fact_scopes(text, fact_types=fact_types)
    if not fact_scopes_allowed(record_scopes, requested_scope=requested, blocked_neighbor_scopes=blocked):
        return False
    if not requested:
        return True
    if not record_scopes:
        if requested in _EXPLICIT_SCOPE_REQUIRED:
            return False
        return True
    return True


def _record_fact_scopes(text: str, *, fact_types: set[str]) -> set[str]:
    return detect_fact_scopes(text, fact_types=tuple(fact_types))


def _record_matches_product_markers(text: str, query_text: str) -> bool:
    if "выездн" in query_text and "городск" in text:
        return False
    marker_groups = (
        ("лвш", "менделеево", "выездн"),
        ("звш", "зимн"),
        ("городск",),
        ("интенсив",),
        ("индивидуаль",),
        ("маткап", "материн"),
        ("налог", "вычет"),
    )
    for group in marker_groups:
        if any(marker in query_text for marker in group) and not any(marker in text for marker in group):
            return False
    return True


def _record_is_objection_pattern(text: str) -> bool:
    return "objection_responses" in text or "возражение" in text or "черновик для ситуации" in text


def _record_is_unrequested_special_product(text: str, query_text: str) -> bool:
    asks_city_day_camp = (
        ("городск" in query_text and ("лагер" in query_text or "школ" in query_text))
        or "без прожив" in query_text
        or ("летн" in query_text and "школ" in query_text)
        or "лш москва" in query_text
    )
    special_markers = (
        ("индивидуаль",),
        ("интенсив",),
        ("лвш", "менделеево", "выездн"),
        ("звш",),
        ("городск", "лагер"),
    )
    for group in special_markers:
        if group == ("городск", "лагер") and asks_city_day_camp:
            continue
        if any(marker in text for marker in group) and not any(marker in query_text for marker in group):
            return True
    return False


def _record_matches_requested_format(text: str, query_text: str) -> bool:
    asks_online = "онлайн" in query_text or "online" in query_text or "дистанц" in query_text
    asks_offline = "очно" in query_text or "очный" in query_text or "офлайн" in query_text or "offline" in query_text
    if asks_online and ("очно" in text or "очный" in text or "офлайн" in text) and not (
        "онлайн" in text or "online" in text or "дистанц" in text
    ):
        return False
    if asks_offline and ("онлайн" in text or "online" in text or "дистанц" in text) and not (
        "очно" in text or "очный" in text or "офлайн" in text
    ):
        return False
    return True


def _record_matches_requested_class(text: str, query_text: str) -> bool:
    numbers = _query_class_numbers(query_text)
    if not numbers or "класс" not in text:
        return True
    if (
        "для других класс" in text
        or "других классов" in text
        or "другим класс" in text
        or "другому классу" in text
    ):
        return True
    return any(_record_mentions_number_or_range(text, number) for number in numbers)


def _query_class_numbers(query_text: str) -> tuple[str, ...]:
    numbers: list[str] = []
    for match in re.findall(r"(?<!\d)(\d{1,2})(?:\s*[-–]\s*\d{1,2})?\s*(?:класс|кл\b)", query_text):
        numbers.append(match)
    for match in re.findall(r"(?:для|в)\s+(\d{1,2})(?:\s*(?:го|ого|й|ой))?\s*(?:класс|кл\b)", query_text):
        numbers.append(match)
    return tuple(_dedupe(numbers))


def _query_asks_for_date(query_text: str) -> bool:
    return "когда" in query_text or "дат" in query_text or "распис" in query_text


def _record_answers_date_question(text: str) -> bool:
    return (
        _looks_like_date_fact(text)
        or "дат" in text
        or "распис" in text
        or "лист ожид" in text
        or "ждем распис" in text
        or "раз в неделю" in text
        or "записи занятий" in text
        or "записи уроков" in text
    )


def _expand_required_fact_types(required_fact_types: set[str], *, topic_id: str, query: str) -> set[str]:
    expanded = set(required_fact_types)
    query_text = f"{topic_id} {query}".casefold()
    if "schedule" in expanded:
        expanded.update({"deadline", "camp_lvsh", "camp_city", "course_parameter", "program"})
    if "program" in expanded:
        expanded.update({"course_parameter", "program", "intensive", "camp_lvsh", "camp_city", "teacher"})
    if "documents" in expanded:
        expanded.update({"documents", "tax", "matkap"})
    if "location" in expanded:
        expanded.update({"location", "contact"})
    if "installment" in expanded:
        expanded.update({"installment", "payment_methods", "discount", "course_parameter"})
    if "лагер" in query_text or "лвш" in query_text or "лш" in query_text or "менделеево" in query_text:
        expanded.update({"camp_lvsh", "camp_city", "deadline", "price", "location", "program"})
    if any(marker in query_text for marker in ("трансфер", "добир", "дорог", "заезд", "из москв", "самостоятельн")):
        expanded.update({"location", "camp_lvsh", "program"})
    return expanded


def _fact_match_score(
    record: Mapping[str, Any],
    *,
    required_fact_types: set[str],
    topic_id: str,
    query: str,
) -> int:
    text = _normalize_match_text(
        " ".join(
            str(record.get(key) or "")
            for key in ("fact_key", "title", "product", "client_safe_text", "manager_display_text", "short_fact")
        )
    )
    query_text = _normalize_match_text(query)
    score = 0
    fact_types = set(_fact_types(record))
    if required_fact_types & fact_types:
        score += 20
    related_theme_ids = set(_text_list(record.get("related_theme_ids") or record.get("theme_ids") or record.get("topics")))
    if topic_id and topic_id in related_theme_ids:
        score += 12
    for term in _query_terms(query):
        if term in text:
            score += 2
    for number in re.findall(r"\d+", query_text):
        if _record_mentions_number_or_range(text, number):
            score += 8
    if ("онлайн" in query_text or "online" in query_text) and ("онлайн" in text or "online" in text):
        score += 8
    if ("очно" in query_text or "очный" in query_text or "офлайн" in query_text or "offline" in query_text) and (
        "очно" in text or "очный" in text or "offline" in text
    ):
        score += 8
    if ("когда" in query_text or "дат" in query_text or "распис" in query_text) and "deadline" in fact_types:
        score += 30
    if ("когда" in query_text or "дат" in query_text or "распис" in query_text) and _looks_like_date_fact(text):
        score += 50
    if ("где" in query_text or "адрес" in query_text or "площадк" in query_text) and "location" in fact_types:
        score += 30
    if any(marker in query_text for marker in ("трансфер", "добир", "дорог", "заезд", "из москв", "самостоятельн")):
        if any(marker in text for marker in ("трансфер", "добир", "дорог", "заезд", "из москв", "ховрино", "самостоятельн")):
            score += 80
        else:
            score -= 10
    if any(marker in query_text for marker in ("прожив", "питан", "что входит")):
        if any(marker in text for marker in ("прожив", "питан", "5-раз", "шведск", "размещ")):
            score += 50
    for marker in ("лвш", "менделеево", "лагер", "интенсив", "пробн", "маткап", "налог", "скид"):
        if marker in query_text and marker in text:
            score += 20
        if marker in query_text and marker not in text:
            score -= 20
    return score


def _record_mentions_number_or_range(text: str, number: str) -> bool:
    if not number:
        return False
    if re.search(rf"(?<!\d){re.escape(number)}(?!\d)", text):
        return True
    value = int(number)
    for start, end in re.findall(r"(?<!\d)(\d{1,2})\s*[-–]\s*(\d{1,2})(?!\d)", text):
        if int(start) <= value <= int(end):
            return True
    return False


def _looks_like_date_fact(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d{1,2}\s*[-–]\s*\d{1,2}\s*(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)"
            r"|\b\d{1,2}\s+(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)",
            text,
        )
    )


def _normalize_match_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").replace("\u00a0", " ").split())


def _usable_for_precise_answer(record: Mapping[str, Any], *, active_brand: str = "unknown") -> bool:
    return (
        _stable_status(record.get("freshness_status")) in _FRESH_STATUSES
        and _truthy(record.get("usable_for_precise_answer"))
        and _truthy(record.get("allowed_for_client_answer"))
        and not _truthy(record.get("requires_manager_confirmation"))
        and not _truthy(record.get("forbidden_for_client"))
        and _record_allowed_for_active_brand(record, active_brand=active_brand)
    )


def _record_allowed_for_active_brand(record: Mapping[str, Any], *, active_brand: str = "unknown") -> bool:
    if _truthy(record.get("forbidden_for_client")) or _truthy(record.get("internal_only")):
        return False
    if _truthy(record.get("cross_brand_mixed")) or _clean_text(record.get("cross_brand_policy")) == "forbidden_for_client":
        return False
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    has_brand_field = bool(record.get("brand") or metadata.get("brand"))
    brand = _clean_text(record.get("brand") or metadata.get("brand") or "brand_neutral").casefold()
    active = _normalize_active_brand(active_brand)
    text = f"{record.get('title', '')} {_fact_text(record)} {record.get('text', '')}"
    if brand in {"", "unknown", "brand_neutral", "both"}:
        if not has_brand_field:
            return True
        return _brand_neutral_text_is_safe(text)
    if active == "unknown":
        return False
    return brand == active


def _fact_text(record: Mapping[str, Any]) -> str:
    return _clean_text(
        record.get("client_safe_text")
        or record.get("short_fact")
        or record.get("manager_text")
        or record.get("text"),
        max_chars=300,
    )


def _fact_types(record: Mapping[str, Any]) -> tuple[str, ...]:
    values = _text_list(record.get("fact_types"))
    if not values:
        values = _text_list(record.get("fact_type"))
    if values:
        return tuple(_dedupe(fact_type_from_key(value) for value in values))
    return classify_fact_types(f"{record.get('title', '')} {record.get('text', '')} {_fact_text(record)}")


def _topic_id(
    *,
    theme: Mapping[str, Any] | str | None,
    rop_policy: Mapping[str, Any] | None,
) -> str:
    if isinstance(rop_policy, Mapping):
        value = rop_policy.get("topic_id") or rop_policy.get("theme_id")
        if value:
            return _clean_text(value)
    if isinstance(theme, Mapping):
        return _clean_text(theme.get("topic_id") or theme.get("theme_id"))
    return _clean_text(theme)


def _records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("items"), Sequence) and not isinstance(value.get("items"), (str, bytes, bytearray)):
            return _records(value.get("items"))
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        if "," in value:
            return [_clean_text(part) for part in value.split(",") if _clean_text(part)]
        return [_clean_text(value)] if _clean_text(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_clean_text(item) for item in value if _clean_text(item)]
    return [_clean_text(value)] if _clean_text(value) else []


def _query_terms(query: str) -> set[str]:
    return {part for part in re.findall(r"[0-9A-Za-zА-Яа-яЁё]{4,}", query.casefold().replace("ё", "е"))}


def _has_precise_claim(text: str) -> bool:
    pattern = (
        r"\b\d[\d\s]*(?:руб|₽|%|процент|январ|феврал|март|апрел|ма[йя]|июн|июл|август|"
        r"сентябр|октябр|ноябр|декабр)"
    )
    return bool(re.search(pattern, text.casefold()))


def _stable_fact_key(value: Any) -> str:
    text = _clean_text(value)
    if "." in text:
        return text
    return f"{text}.current"


def _stable_status(value: Any) -> str:
    return _clean_text(value).casefold() or "unknown"


def _clean_text(value: Any, max_chars: int = 240) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:max_chars]


def _clip_text(value: str, max_chars: int) -> str:
    text = _clean_text(value, max_chars=max_chars)
    if len(text) < max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да", "истина", "есть"}


def _normalize_active_brand(value: Any) -> str:
    text = str(value or "unknown").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return "unknown"


def _brand_neutral_text_is_safe(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    forbidden = (
        "фотон",
        "унпк",
        "мфти",
        "цдпо",
        "црдо",
        "ано дпо",
        "kmipt",
        "cdpofoton",
        "т-банк",
        "долями",
    )
    precise = _has_precise_claim(value)
    return not precise and not any(marker in value for marker in forbidden)


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
