from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mango_mvp.channels.contracts import ChannelMessage


PILOT_CONTEXT_SCHEMA_VERSION = "telegram_pilot_context_v1_2026_05_17"
DIRECT_PATH_PILOT_CONFIG_ENV = "TELEGRAM_DIRECT_PATH_PILOT_CONFIG"
DIRECT_PATH_PILOT_CONFIG_VERSION = "pilot_gold_v1"
MEMORY_PROVENANCE_COMPACT_ENV = "TELEGRAM_MEMORY_PROVENANCE_COMPACT"
PILOT_CONTEXT_PROFILE_DEFAULT_ON_FLAGS: tuple[str, ...] = ()


@dataclass(frozen=True)
class PilotContextQuality:
    customer_identity_found: bool = False
    phone_found: bool = False
    amo_context_found: bool = False
    tallanto_context_found: bool = False
    timeline_context_found: bool = False
    amo_tallanto_conflict: bool = False
    family_phone: bool = False
    multiple_students: bool = False
    multiple_deals: bool = False
    facts_stale: bool = False
    facts_missing: bool = False

    @property
    def context_found_rate(self) -> float:
        checks = (
            self.customer_identity_found,
            self.amo_context_found,
            self.tallanto_context_found,
            self.timeline_context_found,
        )
        return round(sum(1 for item in checks if item) / len(checks), 3)

    @property
    def requires_manager_review(self) -> bool:
        return any(
            (
                self.amo_tallanto_conflict,
                self.family_phone,
                self.multiple_students,
                self.multiple_deals,
                self.facts_stale,
                self.facts_missing,
            )
        )

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": PILOT_CONTEXT_SCHEMA_VERSION,
            "customer_identity_found": self.customer_identity_found,
            "phone_found": self.phone_found,
            "amo_context_found": self.amo_context_found,
            "tallanto_context_found": self.tallanto_context_found,
            "timeline_context_found": self.timeline_context_found,
            "amo_tallanto_conflict": self.amo_tallanto_conflict,
            "family_phone": self.family_phone,
            "multiple_students": self.multiple_students,
            "multiple_deals": self.multiple_deals,
            "facts_stale": self.facts_stale,
            "facts_missing": self.facts_missing,
            "context_found_rate": self.context_found_rate,
            "requires_manager_review": self.requires_manager_review,
        }


@dataclass(frozen=True)
class PilotContext:
    current_message: str
    active_brand: str = "unknown"
    brand_policy: Mapping[str, Any] = field(default_factory=dict)
    payment_context: Mapping[str, Any] = field(default_factory=dict)
    recent_messages: Sequence[str] = field(default_factory=tuple)
    client_identity: Mapping[str, Any] = field(default_factory=dict)
    customer_summary: str = ""
    amo_context: Mapping[str, Any] = field(default_factory=dict)
    tallanto_context: Mapping[str, Any] = field(default_factory=dict)
    timeline_context: Mapping[str, Any] = field(default_factory=dict)
    rop_policy: Mapping[str, Any] = field(default_factory=dict)
    facts_context: Mapping[str, Any] = field(default_factory=dict)
    confirmed_facts: Mapping[str, Any] = field(default_factory=dict)
    missing_facts: Sequence[str] = field(default_factory=tuple)
    required_fact_keys: Sequence[str] = field(default_factory=tuple)
    knowledge_snippets: Sequence[str] = field(default_factory=tuple)
    context_warnings: Sequence[str] = field(default_factory=tuple)
    knowledge_base_version: str = ""
    risk_flags: Sequence[str] = field(default_factory=tuple)
    dialogue_memory_view: Mapping[str, Any] = field(default_factory=dict)
    dialogue_memory_state: Mapping[str, Any] = field(default_factory=dict)
    conversation_intent_plan: Mapping[str, Any] = field(default_factory=dict)
    answer_contract: Mapping[str, Any] = field(default_factory=dict)
    gold_answers_v3: Mapping[str, Any] = field(default_factory=dict)
    gold_answer_context: Mapping[str, Any] = field(default_factory=dict)
    answer_quality_reference: Mapping[str, Any] = field(default_factory=dict)
    few_shot_style_examples: Sequence[str] = field(default_factory=tuple)
    few_shot_correction_examples: Sequence[str] = field(default_factory=tuple)
    context_quality: PilotContextQuality = field(default_factory=PilotContextQuality)

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_message", clean_text(self.current_message, max_chars=1200))
        object.__setattr__(self, "active_brand", normalize_active_brand(self.active_brand))
        object.__setattr__(self, "brand_policy", compact_mapping(self.brand_policy, max_items=12, max_chars=240))
        object.__setattr__(self, "payment_context", compact_mapping(self.payment_context, max_items=12, max_chars=240))
        object.__setattr__(
            self,
            "recent_messages",
            tuple(clean_text(item, max_chars=500) for item in self.recent_messages if clean_text(item, max_chars=500)),
        )
        object.__setattr__(self, "client_identity", compact_mapping(self.client_identity, max_items=12, max_chars=200))
        object.__setattr__(self, "customer_summary", clean_text(self.customer_summary, max_chars=900))
        object.__setattr__(self, "amo_context", compact_mapping(self.amo_context, max_items=16, max_chars=300))
        object.__setattr__(self, "tallanto_context", compact_mapping(self.tallanto_context, max_items=16, max_chars=300))
        object.__setattr__(self, "timeline_context", compact_mapping(self.timeline_context, max_items=16, max_chars=300))
        object.__setattr__(self, "rop_policy", compact_rop_policy(self.rop_policy, max_items=16, max_chars=300))
        object.__setattr__(self, "facts_context", compact_mapping(self.facts_context, max_items=16, max_chars=300))
        object.__setattr__(self, "confirmed_facts", compact_mapping(self.confirmed_facts, max_items=16, max_chars=300))
        object.__setattr__(
            self,
            "missing_facts",
            tuple(dedupe(clean_text(item, max_chars=160) for item in self.missing_facts)),
        )
        object.__setattr__(
            self,
            "required_fact_keys",
            tuple(dedupe(clean_text(item, max_chars=120) for item in self.required_fact_keys)),
        )
        object.__setattr__(
            self,
            "knowledge_snippets",
            tuple(dedupe(clean_text(item, max_chars=700) for item in self.knowledge_snippets))[:8],
        )
        object.__setattr__(
            self,
            "context_warnings",
            tuple(dedupe(clean_text(item, max_chars=120) for item in self.context_warnings)),
        )
        object.__setattr__(self, "knowledge_base_version", clean_text(self.knowledge_base_version, max_chars=160))
        object.__setattr__(self, "risk_flags", tuple(dedupe(clean_text(item, max_chars=120) for item in self.risk_flags)))
        object.__setattr__(
            self,
            "dialogue_memory_view",
            compact_dialogue_memory_view(self.dialogue_memory_view),
        )
        object.__setattr__(
            self,
            "conversation_intent_plan",
            compact_mapping(self.conversation_intent_plan, max_items=36, max_chars=500),
        )
        object.__setattr__(
            self,
            "answer_contract",
            compact_mapping(self.answer_contract, max_items=28, max_chars=500),
        )
        object.__setattr__(
            self,
            "gold_answers_v3",
            compact_mapping(self.gold_answers_v3, max_items=12, max_chars=700),
        )
        object.__setattr__(
            self,
            "gold_answer_context",
            compact_mapping(self.gold_answer_context, max_items=20, max_chars=700),
        )
        object.__setattr__(
            self,
            "answer_quality_reference",
            compact_mapping(self.answer_quality_reference, max_items=16, max_chars=700),
        )
        object.__setattr__(
            self,
            "few_shot_style_examples",
            tuple(dedupe(clean_text(item, max_chars=900) for item in self.few_shot_style_examples))[:6],
        )
        object.__setattr__(
            self,
            "few_shot_correction_examples",
            tuple(dedupe(clean_text(item, max_chars=900) for item in self.few_shot_correction_examples))[:4],
        )

    def to_prompt_context(self) -> Mapping[str, Any]:
        merged_context_warnings = dedupe((*self.context_warnings, *_context_warnings_for_quality(self.context_quality)))
        payload: dict[str, Any] = {
            "schema_version": PILOT_CONTEXT_SCHEMA_VERSION,
            "active_brand": self.active_brand,
            "brand_policy": dict(self.brand_policy),
            "payment_context": dict(self.payment_context),
            "recent_messages": list(self.recent_messages),
            "client_identity": dict(self.client_identity),
            "customer_context_summary": self.customer_summary,
            "amo_context": dict(self.amo_context),
            "tallanto_context": dict(self.tallanto_context),
            "timeline_context": dict(self.timeline_context),
            "rop_policy": dict(self.rop_policy),
            "facts_context": dict(self.facts_context),
            "confirmed_facts": dict(self.confirmed_facts),
            "missing_facts": list(self.missing_facts),
            "required_fact_keys": list(self.required_fact_keys),
            "knowledge_snippets": list(self.knowledge_snippets),
            "knowledge_base_version": self.knowledge_base_version,
            "risk_flags": list(self.risk_flags),
            "dialogue_memory_view": dict(self.dialogue_memory_view),
            "dialogue_memory_state": dict(self.dialogue_memory_state),
            "conversation_intent_plan": dict(self.conversation_intent_plan),
            "answer_contract": dict(self.answer_contract),
            "gold_answers_v3": dict(self.gold_answers_v3),
            "gold_answer_context": dict(self.gold_answer_context),
            "answer_quality_reference": dict(self.answer_quality_reference),
            "few_shot_style_examples": list(self.few_shot_style_examples),
            "few_shot_correction_examples": list(self.few_shot_correction_examples),
            "context_quality": self.context_quality.to_json_dict(),
            "context_warnings": list(merged_context_warnings),
            "pilot_context_safety": pilot_context_safety_contract(),
        }
        return {key: value for key, value in payload.items() if value not in ({}, [], "", None)}

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": PILOT_CONTEXT_SCHEMA_VERSION,
            "current_message": self.current_message,
            **dict(self.to_prompt_context()),
        }


def build_pilot_context(
    message: ChannelMessage | str,
    *,
    active_brand: str = "unknown",
    brand_policy: Mapping[str, Any] | None = None,
    payment_context: Mapping[str, Any] | None = None,
    recent_messages: Sequence[str] = (),
    client_identity: Mapping[str, Any] | None = None,
    customer_summary: str = "",
    amo_context: Mapping[str, Any] | None = None,
    tallanto_context: Mapping[str, Any] | None = None,
    timeline_context: Mapping[str, Any] | None = None,
    rop_policy: Mapping[str, Any] | None = None,
    facts_context: Mapping[str, Any] | None = None,
    confirmed_facts: Mapping[str, Any] | None = None,
    missing_facts: Sequence[str] = (),
    required_fact_keys: Sequence[str] = (),
    knowledge_snippets: Sequence[str] = (),
    context_warnings: Sequence[str] = (),
    knowledge_base_version: str = "",
    risk_flags: Sequence[str] = (),
    dialogue_memory_view: Mapping[str, Any] | None = None,
    dialogue_memory_state: Mapping[str, Any] | None = None,
    conversation_intent_plan: Mapping[str, Any] | None = None,
    answer_contract: Mapping[str, Any] | None = None,
    gold_answers_v3: Mapping[str, Any] | None = None,
    gold_answer_context: Mapping[str, Any] | None = None,
    answer_quality_reference: Mapping[str, Any] | None = None,
    few_shot_style_examples: Sequence[str] = (),
    few_shot_correction_examples: Sequence[str] = (),
) -> PilotContext:
    current_message = message.text if isinstance(message, ChannelMessage) else str(message or "")
    client = dict(client_identity or {})
    if isinstance(message, ChannelMessage):
        client.setdefault("channel", message.channel)
        client.setdefault("channel_thread_id", message.channel_thread_id)
        client.setdefault("channel_user_id", message.channel_user_id)
    amo = dict(amo_context or {})
    tallanto = dict(tallanto_context or {})
    timeline = dict(timeline_context or {})
    facts = dict(facts_context or {})
    missing_fact_items = tuple(dedupe(clean_text(item, max_chars=160) for item in missing_facts))
    explicit_warnings = tuple(dedupe(clean_text(item, max_chars=120) for item in context_warnings))
    quality = PilotContextQuality(
        customer_identity_found=bool(client.get("phone") or client.get("channel_user_id") or client.get("customer_id")),
        phone_found=bool(client.get("phone")),
        amo_context_found=bool(amo),
        tallanto_context_found=bool(tallanto),
        timeline_context_found=bool(timeline.get("found") or timeline.get("summary")),
        amo_tallanto_conflict=truthy(amo.get("tallanto_conflict") or tallanto.get("amo_tallanto_conflict")),
        family_phone=truthy(amo.get("family_phone") or tallanto.get("family_phone")),
        multiple_students=int_or_zero(tallanto.get("students_count") or tallanto.get("student_count")) > 1,
        multiple_deals=int_or_zero(amo.get("deals_count") or amo.get("deal_count")) > 1,
        facts_stale=truthy(facts.get("stale") or facts.get("facts_stale")) or "facts_stale" in explicit_warnings,
        facts_missing=bool(missing_fact_items) or truthy(facts.get("missing") or facts.get("facts_missing")),
    )
    merged_context_warnings = tuple(dedupe((*explicit_warnings, *_context_warnings_for_quality(quality))))
    merged_risks = list(risk_flags)
    merged_risks.extend(merged_context_warnings)
    return PilotContext(
        current_message=current_message,
        active_brand=active_brand,
        brand_policy=brand_policy or {},
        payment_context=payment_context or {},
        recent_messages=recent_messages,
        client_identity=client,
        customer_summary=customer_summary,
        amo_context=amo,
        tallanto_context=tallanto,
        timeline_context=timeline,
        rop_policy=rop_policy or {},
        facts_context=facts,
        confirmed_facts=confirmed_facts or {},
        missing_facts=missing_fact_items,
        required_fact_keys=required_fact_keys,
        knowledge_snippets=knowledge_snippets,
        context_warnings=merged_context_warnings,
        knowledge_base_version=knowledge_base_version,
        risk_flags=tuple(merged_risks),
        dialogue_memory_view=dialogue_memory_view or {},
        dialogue_memory_state=dialogue_memory_state or {},
        conversation_intent_plan=conversation_intent_plan or {},
        answer_contract=answer_contract or {},
        gold_answers_v3=gold_answers_v3 or {},
        gold_answer_context=gold_answer_context or {},
        answer_quality_reference=answer_quality_reference or {},
        few_shot_style_examples=few_shot_style_examples,
        few_shot_correction_examples=few_shot_correction_examples,
        context_quality=quality,
    )


def context_warnings(quality: PilotContextQuality) -> tuple[str, ...]:
    return _context_warnings_for_quality(quality)


def _context_warnings_for_quality(quality: PilotContextQuality) -> tuple[str, ...]:
    warnings: list[str] = []
    if quality.amo_tallanto_conflict:
        warnings.append("amo_tallanto_conflict")
    if quality.family_phone:
        warnings.append("family_phone")
    if quality.multiple_students:
        warnings.append("multiple_students")
    if quality.multiple_deals:
        warnings.append("multiple_deals")
    if quality.facts_stale:
        warnings.append("facts_stale")
    if quality.facts_missing:
        warnings.append("facts_missing")
    return tuple(warnings)


def pilot_context_safety_contract() -> Mapping[str, bool]:
    return {
        "read_amo": True,
        "read_tallanto": True,
        "read_customer_timeline": True,
        "write_amo": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_customer_timeline_db": False,
        "send_client_message": False,
        "run_asr": False,
        "run_ra": False,
        "requires_manager_approval": True,
    }


def compact_mapping(value: Mapping[str, Any] | None, *, max_items: int, max_chars: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        clean_key = clean_text(key, max_chars=80)
        if not clean_key:
            continue
        if isinstance(item, Mapping):
            result[clean_key] = compact_mapping(item, max_items=8, max_chars=max_chars)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            compact_items: list[Any] = []
            for part in item:
                if isinstance(part, Mapping):
                    compact_part = compact_mapping(part, max_items=8, max_chars=max_chars)
                    if compact_part:
                        compact_items.append(compact_part)
                    continue
                text = clean_text(part, max_chars=max_chars)
                if text:
                    compact_items.append(text)
                if len(compact_items) >= 8:
                    break
            result[clean_key] = compact_items
        elif isinstance(item, (str, int, float, bool)) or item is None:
            result[clean_key] = clean_text(item, max_chars=max_chars) if isinstance(item, str) else item
        if len(result) >= max_items:
            break
    return result


def compact_dialogue_memory_view(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    result = dict(compact_mapping(value, max_items=20, max_chars=700))
    if _pilot_context_flag_enabled(MEMORY_PROVENANCE_COMPACT_ENV):
        for key, max_items, max_chars in (
            ("slot_sources", 18, 120),
            ("client_confirmed_slots", 18, 160),
            ("slot_provenance", 18, 300),
        ):
            mapped_value = value.get(key)
            if isinstance(mapped_value, Mapping):
                result[key] = compact_mapping(mapped_value, max_items=max_items, max_chars=max_chars)
    turns = value.get("recent_turns")
    if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes, bytearray)):
        clean_turns: list[Mapping[str, str]] = []
        for raw in turns:
            if not isinstance(raw, Mapping):
                continue
            role = clean_text(raw.get("role"), max_chars=20)
            text = clean_text(raw.get("text"), max_chars=500)
            if role and text:
                clean_turns.append({"role": role, "text": text})
            if len(clean_turns) >= 6:
                break
        result["recent_turns"] = clean_turns
    open_question = value.get("open_question")
    if isinstance(open_question, Mapping):
        result["open_question"] = compact_mapping(open_question, max_items=6, max_chars=260)
    held_state = value.get("held_state")
    if isinstance(held_state, Mapping):
        result["held_state"] = compact_mapping(held_state, max_items=16, max_chars=180)
    topic_focus = value.get("topic_focus")
    if isinstance(topic_focus, Mapping):
        result["topic_focus"] = compact_mapping(topic_focus, max_items=10, max_chars=180)
    safe_answered_parts = value.get("safe_answered_parts")
    if isinstance(safe_answered_parts, Sequence) and not isinstance(safe_answered_parts, (str, bytes, bytearray)):
        result["safe_answered_parts"] = [
            clean_text(item, max_chars=160) for item in safe_answered_parts if clean_text(item, max_chars=160)
        ][:8]
    known_slots = value.get("known_slots")
    if isinstance(known_slots, Mapping):
        result["known_slots"] = compact_mapping(known_slots, max_items=16, max_chars=120)
    do_not_ask_again = value.get("do_not_ask_again") or value.get("do_not_reask_slots")
    if isinstance(do_not_ask_again, Sequence) and not isinstance(do_not_ask_again, (str, bytes, bytearray)):
        result["do_not_ask_again"] = [
            clean_text(item, max_chars=80) for item in do_not_ask_again if clean_text(item, max_chars=80)
        ][:16]
    return result


def _pilot_context_flag_enabled(env_name: str) -> bool:
    raw = os.getenv(env_name)
    if raw is not None:
        return _truthy_setting(raw)
    return (
        env_name in PILOT_CONTEXT_PROFILE_DEFAULT_ON_FLAGS
        and str(os.getenv(DIRECT_PATH_PILOT_CONFIG_ENV) or "").strip() == DIRECT_PATH_PILOT_CONFIG_VERSION
    )


def _truthy_setting(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да", "on"}


def compact_rop_policy(value: Mapping[str, Any] | None, *, max_items: int, max_chars: int) -> Mapping[str, Any]:
    result = dict(compact_mapping(value, max_items=max_items, max_chars=max_chars))
    if not isinstance(value, Mapping):
        return result
    autonomy = value.get("autonomy_policy")
    if not isinstance(autonomy, Mapping):
        return result
    compact_autonomy = dict(compact_mapping(autonomy, max_items=16, max_chars=max_chars))
    for key in ("allowed_topic_ids", "autonomous_topic_ids", "topic_ids"):
        topics = autonomy.get(key)
        if isinstance(topics, Sequence) and not isinstance(topics, (str, bytes, bytearray)):
            compact_autonomy[key] = [clean_text(item, max_chars=120) for item in topics if clean_text(item, max_chars=120)][:64]
        elif topics is not None:
            compact_autonomy[key] = clean_text(topics, max_chars=120)
    result["autonomy_policy"] = compact_autonomy
    return result


def clean_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").strip().split())
    return text[:max_chars]


def normalize_active_brand(value: Any) -> str:
    text = str(value or "unknown").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return "unknown"


def dedupe(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да", "истина", "есть"}


def int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
