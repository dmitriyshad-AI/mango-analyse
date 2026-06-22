from __future__ import annotations

import json
import os
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.dialogue_contract_pipeline import concrete_anchors as dialogue_contract_concrete_anchors


OUTPUT_SANITIZER_ENV = "TELEGRAM_OUTPUT_SANITIZER"


SEMANTIC_OUTPUT_VERIFIER_ENV = "TELEGRAM_SEMANTIC_OUTPUT_VERIFIER"


VERIFIER_HANDOFF_CLAIMS_ENV = "TELEGRAM_VERIFIER_HANDOFF_CLAIMS"


NUMBER_GATE_SCOPE_AWARE_ENV = "TELEGRAM_NUMBER_GATE_SCOPE_AWARE"


DIRECT_PATH_ENV = "TELEGRAM_DIRECT_PATH"


LLM_RETRIEVE_ENV = "TELEGRAM_LLM_RETRIEVE"


TEMPLATE_FROM_KB_ENV = "TELEGRAM_TEMPLATE_FROM_KB"


ROUTE_RUBRIC_ENV = "TELEGRAM_ROUTE_RUBRIC"


TONE_RICH_FORMAT_ENV = "TELEGRAM_TONE_RICH_FORMAT"


A_RICH_FORMAT_ENV = "TELEGRAM_A_RICH_FORMAT"


BOT_GOLD_REAL_ENV = "TELEGRAM_BOT_GOLD_REAL"


PRESALE_SAFETY_ENV = "TELEGRAM_PRESALE_SAFETY"


PRESALE_PII_MEMORY_ENV = "TELEGRAM_PRESALE_PII_MEMORY"


PRESALE_VERIFIER_FAILSOFT_ENV = "TELEGRAM_PRESALE_VERIFIER_FAILSOFT"


PRESALE_META_RU_ENV = "TELEGRAM_PRESALE_META_RU"


PRESALE_SOURCE_ID_ENV = "TELEGRAM_PRESALE_SOURCE_ID"


MEMORY_PROVENANCE_ENV = "TELEGRAM_MEMORY_PROVENANCE"


MEMORY_PROVENANCE_COMPACT_ENV = "TELEGRAM_MEMORY_PROVENANCE_COMPACT"


PII_RELATION_STOPWORDS_ENV = "TELEGRAM_PII_RELATION_STOPWORDS"


MEMORY_CHILD_ELLIPSIS_ENV = "TELEGRAM_MEMORY_CHILD_ELLIPSIS"


DEAL_ACTION_DECISION_ENV = "TELEGRAM_DEAL_ACTION_DECISION"


DIRECT_PATH_MODEL_P0_ENV = "TELEGRAM_DIRECT_PATH_MODEL_P0"


P0_MODEL_LED_ENV = "TELEGRAM_P0_MODEL_LED"


DIRECT_DEFAULT_MANAGER_ENV = "TELEGRAM_DIRECT_DEFAULT_MANAGER"


PRICE_AXES_SELECTOR_ENV = "TELEGRAM_PRICE_AXES_SELECTOR"


PRICE_AXES_CLEAN_DEFER_ENV = "TELEGRAM_PRICE_AXES_CLEAN_DEFER"


ANSWERABILITY_SHADOW_ENV = "TELEGRAM_ANSWERABILITY_SHADOW"


DIRECT_PATH_PILOT_CONFIG_ENV = "TELEGRAM_DIRECT_PATH_PILOT_CONFIG"


DIRECT_PATH_PILOT_CONFIG_VERSION = "pilot_gold_v1"


DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS = (
    DIRECT_PATH_ENV,
    BOT_GOLD_REAL_ENV,
    ANSWERABILITY_SHADOW_ENV,
    DEAL_ACTION_DECISION_ENV,
    DIRECT_PATH_MODEL_P0_ENV,
    SEMANTIC_OUTPUT_VERIFIER_ENV,
    OUTPUT_SANITIZER_ENV,
    ROUTE_RUBRIC_ENV,
    TONE_RICH_FORMAT_ENV,
    A_RICH_FORMAT_ENV,
    LLM_RETRIEVE_ENV,
    NUMBER_GATE_SCOPE_AWARE_ENV,
    VERIFIER_HANDOFF_CLAIMS_ENV,
    TEMPLATE_FROM_KB_ENV,
    PRESALE_SAFETY_ENV,
    PRESALE_PII_MEMORY_ENV,
    PRESALE_VERIFIER_FAILSOFT_ENV,
    PRESALE_META_RU_ENV,
    PRESALE_SOURCE_ID_ENV,
    MEMORY_PROVENANCE_ENV,
    MEMORY_PROVENANCE_COMPACT_ENV,
    PII_RELATION_STOPWORDS_ENV,
    MEMORY_CHILD_ELLIPSIS_ENV,
    PRICE_AXES_SELECTOR_ENV,
    PRICE_AXES_CLEAN_DEFER_ENV,
)


def _direct_path_pilot_config(context: Optional[Mapping[str, Any]] = None) -> str:
    if isinstance(context, Mapping):
        for key in (DIRECT_PATH_PILOT_CONFIG_ENV, "direct_path_pilot_config", "pilot_config"):
            value = str(context.get(key) or "").strip()
            if value:
                return value
    return str(os.getenv(DIRECT_PATH_PILOT_CONFIG_ENV) or "").strip()


def _answerability_shadow_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _pilot_profile_default_on_flag_enabled(
        context,
        ANSWERABILITY_SHADOW_ENV,
        aliases=("answerability_shadow", "answerability_shadow_enabled"),
    )


def _presale_prompt_child_name_value(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("value")
    text = " ".join(str(value or "").split()).strip(" ,.;:!?")
    if not text or _A2_PHONE_RE.search(text) or _CLIENT_EMAIL_RE.search(text):
        return ""
    parts = [part for part in text.split() if re.match(r"^[А-ЯЁ][а-яё]{2,}$", part)]
    if not parts:
        return ""
    if len(parts) >= 2 and _looks_like_russian_surname(parts[0]):
        return parts[1]
    return parts[0]


_A2_PHONE_RE = re.compile(r"(?:\+7|8|7)?[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")


_CLIENT_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+", re.I)


def _looks_like_russian_surname(word: str) -> bool:
    normalized = str(word or "").casefold().replace("ё", "е")
    return normalized.endswith(
        (
            "ов",
            "ова",
            "ову",
            "ев",
            "ева",
            "еву",
            "ин",
            "ина",
            "ину",
            "ский",
            "ская",
            "ского",
            "скую",
            "цкий",
            "цкая",
            "цкого",
            "цкую",
            "енко",
            "ко",
        )
    )


def _fresh_fact_texts(context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    facts_context = context.get("facts_context")
    facts_mapping = facts_context if isinstance(facts_context, Mapping) else {}
    context_quality = context.get("context_quality")
    quality_mapping = context_quality if isinstance(context_quality, Mapping) else {}

    stale = (
        _truthy_value(context.get("facts_stale"))
        or _truthy_value(facts_mapping.get("stale"))
        or _truthy_value(facts_mapping.get("facts_stale"))
        or _truthy_value(quality_mapping.get("facts_stale"))
    )
    fresh = (
        context.get("facts_fresh") is True
        or facts_mapping.get("fresh") is True
        or facts_mapping.get("facts_fresh") is True
        or facts_mapping.get("fresh_facts") is True
    )
    verified = (
        context.get("client_safe_fact_verified") is True
        or facts_mapping.get("client_safe_fact_verified") is True
        or _has_dialogue_contract_retrieved_facts(context)
    )
    if stale and not (fresh or verified):
        return ()
    if not (fresh or verified):
        return ()

    texts: list[str] = []
    for key in ("confirmed_facts", "facts_context"):
        _append_fact_texts(texts, context.get(key))
    pipeline = context.get("dialogue_contract_pipeline") if isinstance(context.get("dialogue_contract_pipeline"), Mapping) else {}
    if isinstance(pipeline.get("retrieved_facts"), Mapping):
        _append_fact_texts(texts, pipeline.get("retrieved_facts"))
    _append_fact_texts(texts, context.get("knowledge_snippets"))
    return tuple(text for text in texts if text)


def _has_dialogue_contract_retrieved_facts(context: Mapping[str, Any]) -> bool:
    pipeline = context.get("dialogue_contract_pipeline") if isinstance(context.get("dialogue_contract_pipeline"), Mapping) else {}
    retrieved = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    return any(str(key).strip() and str(value).strip() for key, value in retrieved.items())


def _append_fact_texts(result: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        cleaned = " ".join(value.split())
        if cleaned:
            result.append(cleaned)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().casefold() in {
                "missing",
                "facts_missing",
                "stale",
                "facts_stale",
                "fresh",
                "facts_fresh",
                "fresh_facts",
                "client_safe_fact_verified",
            }:
                continue
            _append_fact_texts(result, item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _append_fact_texts(result, item)
        return
    if isinstance(value, (int, float)):
        result.append(str(value))


def _claim_supported_by_facts(claim: str, fact_texts: Sequence[str]) -> bool:
    normalized_claim = _normalize_fact_match_text(claim)
    if not normalized_claim:
        return False
    normalized_facts = [_normalize_fact_match_text(text) for text in fact_texts]
    if normalized_claim == "до 1 июля" and any(
        "before_2026_07_01" in text or "до 1 июля" in text or "ранн" in text for text in normalized_facts
    ):
        return True
    if normalized_claim == "до 1 июня" and any(
        "before_2026_06_01" in text or "до 1 июня" in text or "ранн" in text for text in normalized_facts
    ):
        return True
    if any(normalized_claim in text for text in normalized_facts):
        return True
    claim_anchors = _fact_match_anchors(claim)
    if not claim_anchors:
        return False
    return any(claim_anchors <= _fact_match_anchors(text) for text in fact_texts)


def _keep_answer_supported(claim: str, fact_texts: Sequence[str]) -> bool:
    normalized_claim = _normalize_fact_match_text(claim)
    if not normalized_claim:
        return False
    normalized_facts = [_normalize_fact_match_text(text) for text in fact_texts if _normalize_fact_match_text(text)]
    if not normalized_facts:
        return False
    hard_claim_anchors = _keep_answer_hard_anchors(claim)
    if not hard_claim_anchors:
        return True
    fact_hard_anchors: set[str] = set()
    for text in fact_texts:
        fact_hard_anchors.update(_keep_answer_hard_anchors(text))
    return hard_claim_anchors <= fact_hard_anchors


def _keep_answer_hard_anchors(text: Any) -> set[str]:
    result: set[str] = set()
    for anchor in _fact_match_anchors(text):
        value = str(anchor or "")
        if value.startswith(("brand:", "unit:", "deadline:")):
            result.add(value)
        elif re.search(r"\d", value):
            result.add(value)
    return result


def _fact_match_anchors(text: Any) -> set[str]:
    source = str(text or "")
    low = source.casefold().replace("ё", "е").replace("\u00a0", " ")
    anchors = set(dialogue_contract_concrete_anchors(source))
    anchors.update(_fact_match_unit_anchors(source))
    anchors.update(_fact_match_schedule_condition_anchors(low))
    if re.search(r"\bфотон\b|цдпо|црдо|cdpofoton", low, re.I):
        anchors.add("brand:foton")
    if re.search(r"\bунпк\b|унпк\s+мфти|kmipt", low, re.I):
        anchors.add("brand:unpk")
    if re.search(r"\bсегодня\b", low, re.I):
        anchors.add("deadline:today")
    if re.search(r"\bзавтра\b|до\s+завтра", low, re.I):
        anchors.add("deadline:tomorrow")
    if re.search(r"до\s+вечера|к\s+вечеру", low, re.I):
        anchors.add("deadline:evening")
    if re.search(r"в\s+течение\s+\d+\s*(?:минут|час|часов|дн|дней|суток|сутки)", low, re.I):
        anchors.add("deadline:relative_period")
    return anchors


def _fact_match_unit_anchors(text: Any) -> set[str]:
    source = str(text or "").replace("\u00a0", " ")
    anchors: set[str] = set()
    if re.search(r"\b\d{1,3}(?:[,.]\d{1,2})?\s*(?:%|процент\w*)", source, re.I):
        anchors.add("unit:percent")
    if re.search(r"\b\d[\d\s]{1,9}\s*(?:руб(?:\.|лей|ля|ль)?|₽|р\.)", source, re.I):
        anchors.add("unit:money")
    if re.search(r"\b\d{1,3}\+?\s*балл\w*", source, re.I):
        anchors.add("unit:points")
    return anchors


def _fact_match_schedule_condition_anchors(low_text: str) -> set[str]:
    anchors: set[str] = set()
    if re.search(r"\bвечерн\w*", low_text, re.I):
        anchors.add("condition:evening")
    if re.search(r"\bутренн\w*", low_text, re.I):
        anchors.add("condition:morning")
    if re.search(r"\bдневн\w*", low_text, re.I):
        anchors.add("condition:day")
    if re.search(r"\b(?:выходн|суббот|воскресен)\w*", low_text, re.I):
        anchors.add("condition:weekend")
    if re.search(r"\b(?:будн|буден)\w*", low_text, re.I):
        anchors.add("condition:weekday")
    return anchors


def _normalize_fact_match_text(text: Any) -> str:
    value = str(text or "").casefold().replace("ё", "е").replace("\u00a0", " ")
    return " ".join(value.split())


def _truthy_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


def _explicit_truthy_setting(
    context: Optional[Mapping[str, Any]],
    env_name: str,
    *,
    aliases: Sequence[str] = (),
) -> Optional[bool]:
    if isinstance(context, Mapping):
        for key in (env_name, *aliases):
            if key in context:
                return _truthy_value(context.get(key))
    if env_name in os.environ:
        return _truthy_value(os.getenv(env_name))
    return None


def _pilot_gold_profile_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _direct_path_pilot_config(context) == DIRECT_PATH_PILOT_CONFIG_VERSION


def _pilot_profile_flag_enabled(
    context: Optional[Mapping[str, Any]],
    env_name: str,
    *,
    aliases: Sequence[str] = (),
) -> bool:
    explicit = _explicit_truthy_setting(context, env_name, aliases=aliases)
    if explicit is not None:
        return explicit
    return _pilot_gold_profile_enabled(context)


def _pilot_profile_default_on_flag_enabled(
    context: Optional[Mapping[str, Any]],
    env_name: str,
    *,
    aliases: Sequence[str] = (),
) -> bool:
    explicit = _explicit_truthy_setting(context, env_name, aliases=aliases)
    if explicit is not None:
        return explicit
    return env_name in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS and _pilot_gold_profile_enabled(context)


def _deal_action_decision_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    explicit = _explicit_truthy_setting(
        context,
        DEAL_ACTION_DECISION_ENV,
        aliases=("deal_action_decision_enabled", "action_decision_enabled"),
    )
    if explicit is not None:
        return explicit
    return _pilot_profile_default_on_flag_enabled(
        context,
        DEAL_ACTION_DECISION_ENV,
        aliases=("deal_action_decision_enabled", "action_decision_enabled"),
    )


def _direct_path_model_p0_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if _p0_model_led_enabled(context):
        return True
    explicit = _explicit_truthy_setting(
        context,
        DIRECT_PATH_MODEL_P0_ENV,
        aliases=("direct_path_model_p0", "direct_path_model_p0_enabled", "model_p0_enabled"),
    )
    if explicit is not None:
        return explicit
    return _pilot_profile_default_on_flag_enabled(
        context,
        DIRECT_PATH_MODEL_P0_ENV,
        aliases=("direct_path_model_p0", "direct_path_model_p0_enabled", "model_p0_enabled"),
    )


def _p0_model_led_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    explicit = _explicit_truthy_setting(
        context,
        P0_MODEL_LED_ENV,
        aliases=("p0_model_led", "p0_model_led_enabled"),
    )
    return bool(explicit) if explicit is not None else False


_P0_MODEL_LED_NON_COMPLAINT_HARD_CODES = frozenset({"refund", "legal", "payment_dispute"})


_P0_MODEL_LED_COMPLAINT_BACKSTOP_RE = re.compile(
    r"(?iu)"
    r"(?:"
    r"\bжалоб\w*"
    r"|безобрази\w*"
    r"|отвратительн\w*"
    r"|накричал\w*\s+на\s+реб[её]нк\w*"
    r"|(?:унизил\w*|оскорбил\w*|высмеял\w*|издевал\w*)\s+.*?реб[её]нк\w*"
    r"|реб[её]нок\s+один\s+остал\w*"
    r"|никто\s+не\s+подош[её]л"
    r"|напишу\s+везде\s+какие\s+вы"
    r")"
)


def _p0_model_led_complaint_backstop(client_message: str) -> bool:
    return bool(_P0_MODEL_LED_COMPLAINT_BACKSTOP_RE.search(str(client_message or "")))


def _p0_model_led_filter_high_risk_codes(
    codes: Sequence[str],
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    unique_codes = tuple(dict.fromkeys(str(code) for code in codes if str(code).strip()))
    if not _p0_model_led_enabled(context):
        return unique_codes
    if any(code in _P0_MODEL_LED_NON_COMPLAINT_HARD_CODES for code in unique_codes):
        return unique_codes
    filtered = tuple(code for code in unique_codes if code != "complaint")
    if _p0_model_led_complaint_backstop(client_message):
        filtered = (*filtered, "complaint")
    return tuple(dict.fromkeys(filtered))


def _direct_default_manager_enabled() -> bool:
    return _truthy_value(os.getenv(DIRECT_DEFAULT_MANAGER_ENV))


def _pilot_profile_overrides(context: Optional[Mapping[str, Any]]) -> dict[str, str]:
    if not _pilot_gold_profile_enabled(context):
        return {}
    aliases: Mapping[str, tuple[str, ...]] = {
        DIRECT_PATH_ENV: ("direct_path_enabled",),
        BOT_GOLD_REAL_ENV: ("bot_gold_real", "direct_path_gold_real"),
        OUTPUT_SANITIZER_ENV: ("output_sanitizer_enabled",),
        SEMANTIC_OUTPUT_VERIFIER_ENV: ("semantic_output_verifier_enabled",),
        ROUTE_RUBRIC_ENV: ("route_rubric_enabled",),
        LLM_RETRIEVE_ENV: ("llm_retrieve_enabled",),
        VERIFIER_HANDOFF_CLAIMS_ENV: ("verifier_handoff_claims_enabled",),
        TEMPLATE_FROM_KB_ENV: ("template_from_kb",),
    }
    result: dict[str, str] = {}
    for env_name in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS:
        explicit = _explicit_truthy_setting(context, env_name, aliases=aliases.get(env_name, ()))
        if explicit is False:
            result[env_name] = "0"
    return result


def _template_from_kb_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (TEMPLATE_FROM_KB_ENV, "template_from_kb"):
            if key in context:
                return _truthy_value(context.get(key))
    if TEMPLATE_FROM_KB_ENV in os.environ:
        return _truthy_value(os.getenv(TEMPLATE_FROM_KB_ENV))
    return _pilot_gold_profile_enabled(context)

def _direct_path_snapshot_path_from_context(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return ""
    for key in ("snapshot_path", "knowledge_snapshot_path", "kb_snapshot_path"):
        value = context.get(key)
        if value:
            return str(value)
    return ""

@lru_cache(maxsize=8)
def _direct_path_load_snapshot(path_text: str) -> Mapping[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, Mapping) else {}

def _direct_path_snapshot_facts(snapshot: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    facts = snapshot.get("facts") if isinstance(snapshot, Mapping) else None
    if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in facts if isinstance(item, Mapping))

def _direct_path_valid_until_ok(value: Any, *, today: Optional[date] = None) -> bool:
    raw = str(value or "").strip()
    if not raw:
        return True
    try:
        valid_until = date.fromisoformat(raw[:10])
    except ValueError:
        return False
    return valid_until >= (today or date.today())

def _direct_path_client_safe_snapshot_fact(fact: Mapping[str, Any], *, active_brand: str) -> bool:
    brand = str(fact.get("brand") or fact.get("active_brand") or "").strip().casefold()
    return (
        brand == str(active_brand or "").strip().casefold()
        and fact.get("allowed_for_client_answer") is True
        and fact.get("forbidden_for_client") is not True
        and fact.get("internal_only") is not True
        and _direct_path_valid_until_ok(fact.get("valid_until"))
        and bool(str(fact.get("client_safe_text") or fact.get("fact_text") or "").strip())
    )

def _direct_path_snapshot_fact_text(fact: Mapping[str, Any]) -> str:
    for key in ("client_safe_text", "fact_text"):
        text = str(fact.get(key) or "").strip()
        if text:
            return _client_clean_fact_text(text)
    return ""

def _direct_path_fact_by_brand_key(
    snapshot: Mapping[str, Any],
    *,
    active_brand: str,
    fact_key: str,
) -> Optional[Mapping[str, Any]]:
    key = str(fact_key or "").strip()
    brand = str(active_brand or "").strip().casefold()
    if not key or brand not in {"foton", "unpk"}:
        return None
    for fact in _direct_path_snapshot_facts(snapshot):
        if str(fact.get("fact_key") or "") != key:
            continue
        if str(fact.get("brand") or "").casefold() != brand:
            continue
        if _direct_path_client_safe_snapshot_fact(fact, active_brand=brand):
            return fact
    return None

def _direct_path_fact_value(text: str) -> str:
    value = _client_clean_fact_text(text)
    if "—" in value:
        value = value.rsplit("—", 1)[-1].strip()
    return value.rstrip(" .")

def _direct_path_template_from_fact(
    *,
    active_brand: str,
    fact_key: str,
    literal_text: str,
    neutral_fallback: str,
    context: Optional[Mapping[str, Any]] = None,
    render: Optional[Callable[[str], str]] = None,
) -> str:
    if not _template_from_kb_enabled(context):
        return literal_text
    snapshot = _direct_path_load_snapshot(_direct_path_snapshot_path_from_context(context))
    fact = _direct_path_fact_by_brand_key(snapshot, active_brand=active_brand, fact_key=fact_key)
    if not isinstance(fact, Mapping):
        _template_from_kb_trace_event(context, {"fact_key": fact_key, "outcome": "fallback", "reason": "missing"})
        return neutral_fallback
    text = _direct_path_snapshot_fact_text(fact)
    if render is not None:
        text = render(text)
    text = str(text or "").strip()
    if text:
        _template_from_kb_trace_event(context, {"fact_key": fact_key, "outcome": "hit"})
        return text
    _template_from_kb_trace_event(context, {"fact_key": fact_key, "outcome": "fallback", "reason": "empty"})
    return neutral_fallback

def _template_from_kb_trace_event(context: Optional[Mapping[str, Any]], payload: Mapping[str, Any]) -> None:
    event = dict(payload)
    trace_event(context, "template_from_kb", event)
    if isinstance(context, dict):
        existing = context.setdefault("template_from_kb_trace", [])
        if isinstance(existing, list):
            existing.append(event)

def _direct_path_template_fact_text(
    *,
    active_brand: str,
    fact_key: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    snapshot = _direct_path_load_snapshot(_direct_path_snapshot_path_from_context(context))
    fact = _direct_path_fact_by_brand_key(snapshot, active_brand=active_brand, fact_key=fact_key)
    return _direct_path_snapshot_fact_text(fact) if isinstance(fact, Mapping) else ""

def _client_clean_fact_text(value: object) -> str:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"^(?P<brand>[^:]{1,40}:\s*)?черновик\s+для\s+ситуации\s+«[^»]+»\s*:\s*",
        lambda match: str(match.group("brand") or ""),
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"^(?P<brand>[^:]{1,40}:\s*)?черновик\s+для\s+ситуации\s+\"[^\"]+\"\s*:\s*",
        lambda match: str(match.group("brand") or ""),
        cleaned,
        flags=re.I,
    )
    return cleaned.strip()

def _active_brand(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return "unknown"
    value = context.get("active_brand")
    if not value and isinstance(context.get("facts_context"), Mapping):
        value = context["facts_context"].get("active_brand")  # type: ignore[index]
    text = str(value or "unknown").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return "unknown"
