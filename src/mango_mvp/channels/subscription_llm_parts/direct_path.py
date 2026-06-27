from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import replace
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import yaml

from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.subscription_llm_parts.codex_exec import extract_json_object
from mango_mvp.channels.subscription_llm_parts.contracts import (
    SubscriptionDraftResult,
    _clamp_float,
    _normalize_output_sanitizer_text,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    BOT_GOLD_REAL_ENV,
    DIRECT_PATH_ENV,
    DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    LLM_RETRIEVE_ENV,
    MEMORY_PROVENANCE_ENV,
    PRESALE_PII_MEMORY_ENV,
    PRESALE_SAFETY_ENV,
    ROUTE_RUBRIC_ENV,
    TEMPLATE_FROM_KB_ENV,
    _A2_PHONE_RE,
    _CLIENT_EMAIL_RE,
    _active_brand,
    _answerability_shadow_enabled,
    _client_clean_fact_text,
    _deal_action_decision_enabled,
    _direct_path_model_p0_enabled,
    _intent_model_led_enabled,
    _p0_model_led_enabled,
    _prose_model_led_enabled,
    _direct_path_client_safe_snapshot_fact,
    _direct_path_fact_by_brand_key,
    _direct_path_fact_value,
    _direct_path_load_snapshot,
    _direct_path_snapshot_fact_text,
    _direct_path_snapshot_facts,
    _direct_path_snapshot_path_from_context,
    _direct_path_template_fact_text,
    _direct_path_template_from_fact,
    _direct_path_valid_until_ok,
    _normalize_fact_match_text,
    _pilot_gold_profile_enabled,
    _pilot_profile_default_on_flag_enabled,
    _pilot_profile_flag_enabled,
    _pilot_profile_overrides,
    _presale_prompt_child_name_value,
    _template_from_kb_enabled,
    _template_from_kb_trace_event,
    _truthy_value,
)

BOT_GOLD_REAL_PACK_ENV = "TELEGRAM_BOT_GOLD_REAL_PACK"

DIRECT_PATH_SCHEMA_VERSION = "direct_path_v1_2026_06_08"

DIRECT_PATH_WIDE_FACT_PACK_SCHEMA_VERSION = "direct_path_wide_fact_pack_v1_2026_06_08"

DIRECT_PATH_WIDE_FACT_LIMIT = 60

DIRECT_PATH_WIDE_FACT_CHAR_LIMIT = 10_000

WAPPI_DIRECT_PATH_RAW_HISTORY_LIMIT = 15
WAPPI_OLDER_DIALOGUE_SUMMARY_PREFIX = "Ранее в диалоге:"

RETRIEVER_NEED_SHADOW_ENV = "TELEGRAM_RETRIEVER_NEED_SHADOW"

RETRIEVER_MODEL_DRIVEN_ENV = "TELEGRAM_RETRIEVER_MODEL_DRIVEN"

ASSUMED_SCOPE_GUARD_ENV = "TELEGRAM_ASSUMED_SCOPE_GUARD"

DIRECT_PLAN_KNOWN_SLOTS_ENV = "TELEGRAM_DIRECT_PLAN_KNOWN_SLOTS"

DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV = "TELEGRAM_DIRECT_KEYWORD_FALLBACK_RELEVANCE"

DIRECT_SLOT_TOPIC_SHADOW_ENV = "TELEGRAM_DIRECT_SLOT_TOPIC_SHADOW"

DIRECT_P0_TEXT_HYGIENE_ENV = "TELEGRAM_DIRECT_P0_TEXT_HYGIENE"

BOT_SAFE_CRM_CONTEXT_ENV = "TELEGRAM_BOT_SAFE_CRM_CONTEXT"

RETRIEVER_NEED_DECLARATION_SCHEMA_VERSION = "retriever_need_declaration_v1_2026_06_15"

DIRECT_SLOT_TOPIC_SHADOW_SCHEMA_VERSION = "direct_slot_topic_shadow_v1_2026_06_17"

_BOT_SAFE_SERVICE_ID_RE = re.compile(
    r"\b(?:customer:[a-f0-9]{16,}|timeline_event:[a-f0-9]{16,}|bot_context_chunk:[a-f0-9]{16,}|botsafe:[^\s,;]+)\b",
    re.I,
)
_BOT_SAFE_MEMORY_EXACT_DETAIL_RE = re.compile(
    r"(?:"
    r"\b20\d{2}\s*/\s*\d{2}\b"
    r"|\b\d{1,2}:\d{2}\s*[-–—]\s*\d{1,2}:\d{2}\b"
    r"|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"
    r"|\b\d{1,3}(?:[\s\u00a0]\d{3})+(?:\s*(?:₽|руб\.?|рублей|рубля))?"
    r"|\b\d+(?:[,.]\d+)?\s*%"
    r"|\b\d+\s*(?:₽|руб\.?|рублей|рубля)\b"
    r")",
    re.I,
)
_BOT_SAFE_PERSON_CONTEXT_RE = re.compile(
    r"\b(?:менеджер|куратор|преподаватель|реб[её]н(?:ок|ка|ку)?|сын(?:а)?|доч(?:ь|ка|ку|ери)?|"
    r"ученик(?:а)?|ученица|фио|зовут|имя)\s*[:—-]?\s*"
    r"[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){0,2}",
    re.I,
)

DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH = (
    Path(__file__).resolve().parents[4]
    / "product_data"
    / "bot_improvement_candidates_20260523"
    / "01_gold_and_few_shot"
    / "real_manager_gold_2026-06-08.yaml"
)

DIRECT_PATH_REAL_MANAGER_GOLD_PACK_VERSION = "real_manager_gold_2026-06-08"

DIRECT_PATH_MISSION_TEMPLATE = (
    "Ты — менеджер-консультант учебного центра {brand}. Тебе пишет родитель с задачей\n"
    "про ребёнка. Твоя цель — реально помочь разобраться и довести до записи на\n"
    "подходящий курс. Продажа — это помощь: польза с первого ответа, предугадывай\n"
    "следующий вопрос, веди к понятному шагу. Не дави: честность важнее сделки.\n"
    "Числа, даты и условия — только из фактов; чего нет в фактах — скажи честно\n"
    "и предложи шаг. Если правило безопасности или передача менеджеру противоречат\n"
    "записи — правило важнее. Не обещай действия и сроки от имени менеджера: можно\n"
    "написать «менеджер свяжется» без срока, но нельзя «свяжется завтра/утром/в течение N»\n"
    "или гарантировать действие. Не утверждай, что телефон или контакт уже есть у центра,\n"
    "если это не подтверждено в памяти или фактах. Имя ребёнка можно использовать, если\n"
    "клиент сам его назвал; телефон или ФИО целиком не дублируй."
)

DIRECT_PATH_MISSION_ROUTE_RUBRIC_SCOPE_REPLACEMENT = (
    "написать «менеджер свяжется» без срока только в черновике для менеджера, "
    "но нельзя «свяжется завтра/утром/в течение N»"
)

DIRECT_PATH_ROUTE_RUBRIC_BLOCK = (
    'Выбор маршрута:\n'
    '- "bot_answer_self_for_pilot" — когда факты из блока «Факты по вашему вопросу» покрывают вопрос клиента '
    'и не требуется действие менеджера. Отвечай по фактам уверенно и не обещай, что «менеджер свяжется», '
    '— ты уже отвечаешь. Смежные факты покрытием НЕ считаются: на их основе самостоятельный ответ не выбирай.\n'
    '- "draft_for_manager" — когда фактов не хватает, нужно ДЕЙСТВИЕ или проверка менеджера '
    '(оформить запись, отправить документы, проверить оплату, персональные данные) или вопрос требует личной оценки. '
    'Обязательно заполни missing_facts: какого факта или какой проверки не хватает. В черновике пиши содержательный '
    'ответ по фактам для менеджера — а не «передам менеджеру» как весь текст.\n'
    'Развилка по процессам: РАССКАЗАТЬ, как устроен процесс (как проходит запись, что после оплаты, есть лист ожидания), '
    '— это самостоятельный ответ по факту процесса. ВЫПОЛНИТЬ действие по просьбе клиента («запишите меня», '
    '«пришлите договор», «проверьте оплату») — это draft_for_manager.\n'
    'Запрещено вычислять новые числа: не выводи проценты, скидки, суммы и итоги из других цен '
    '(«за два предмета выйдет…», «это получается N%»). Называй только числа, которые есть в фактах дословно '
    'или назвал сам клиент. Не подтверждай расчёты клиента («у меня выходит N, верно?») — точный расчёт '
    'и итог по нескольким предметам или со скидками подтвердит менеджер.\n'
    'Избегай сравнительных оценок форматов/программ без факта («очно удобнее…») — вместо этого предложи '
    'признак выбора вопросом.\n'
    'Запрещено: выбирать "draft_for_manager" на всякий случай при полных фактах.'
)

def _direct_path_mission_text(*, brand_label: str, context: Optional[Mapping[str, Any]]) -> str:
    mission = DIRECT_PATH_MISSION_TEMPLATE.format(brand=brand_label)
    if not _route_rubric_enabled(context):
        return mission
    return mission.replace(
        "написать «менеджер свяжется» без срока, но нельзя «свяжется завтра/утром/в течение N»",
        DIRECT_PATH_MISSION_ROUTE_RUBRIC_SCOPE_REPLACEMENT,
    )

def _direct_path_route_rubric_block(context: Optional[Mapping[str, Any]]) -> str:
    return f"{DIRECT_PATH_ROUTE_RUBRIC_BLOCK}\n\n" if _route_rubric_enabled(context) else ""


DIRECT_PATH_PROSE_MODEL_LED_BLOCK = (
    "Качество текста:\n"
    "- Пиши клиентский текст сам, естественно и по-разному на повторах; не копируй служебные шаблоны.\n"
    "- Не начинай с казённых фраз вроде «Да, сориентирую по проверенной информации/условиям».\n"
    "- Если вопрос про наличие мест, бронь, запись на группу или смену: ответь по известным фактам, но не обещай место. "
    "Сформулируй живо: что уже понятно и что менеджер должен проверить по конкретной группе/смене.\n"
    "- Если приходится передать менеджеру, не повторяй дословно предыдущий ответ и не делай весь текст одной фразой «передам менеджеру».\n"
    "- Не пиши «в фактах нет», «по фактам не вижу», «у меня нет данных» клиенту. Скажи по-человечески: эту деталь нужно проверить у менеджера.\n"
    "- Не пиши «прикрепляю», «присылаю», «отправляю», «скину», «дам ссылку/фрагмент/инструкцию», если ты реально не отправляешь файл или ссылку. "
    "Можно написать, что менеджер проверит и пришлёт материал/ссылку.\n"
    "- По адресам: общий действующий адрес можно назвать. Но если клиент спрашивает, куда ехать на конкретное занятие/группу, "
    "не привязывай группу к адресу без точного факта расписания; скажи, что площадку конкретной группы подтвердит менеджер.\n"
    "- Не выводи клиенту внутренние плейсхолдеры в квадратных скобках, включая «[данные у менеджера]»."
)


def _direct_path_prose_model_led_block(context: Optional[Mapping[str, Any]]) -> str:
    return f"{DIRECT_PATH_PROSE_MODEL_LED_BLOCK}\n\n" if _prose_model_led_enabled(context) else ""


def _direct_path_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (DIRECT_PATH_ENV, "direct_path_enabled"):
            if key in context:
                return _truthy_value(context.get(key))
    if DIRECT_PATH_ENV in os.environ:
        return _truthy_value(os.getenv(DIRECT_PATH_ENV))
    return _pilot_gold_profile_enabled(context)

def _llm_retrieve_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _pilot_profile_flag_enabled(context, LLM_RETRIEVE_ENV, aliases=("llm_retrieve_enabled",))

def _default_off_flag_enabled(
    context: Optional[Mapping[str, Any]],
    env_name: str,
    *,
    aliases: Sequence[str] = (),
) -> bool:
    if isinstance(context, Mapping):
        for key in (env_name, *aliases):
            if key in context:
                return _truthy_value(context.get(key))
    if env_name in os.environ:
        return _truthy_value(os.getenv(env_name))
    return False

def _retriever_need_shadow_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _default_off_flag_enabled(
        context,
        RETRIEVER_NEED_SHADOW_ENV,
        aliases=("retriever_need_shadow", "retriever_need_shadow_enabled"),
    )

def _assumed_scope_guard_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _default_off_flag_enabled(
        context,
        ASSUMED_SCOPE_GUARD_ENV,
        aliases=("assumed_scope_guard", "assumed_scope_guard_enabled"),
    )

def _retriever_model_driven_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _assumed_scope_guard_enabled(context) and _default_off_flag_enabled(
        context,
        RETRIEVER_MODEL_DRIVEN_ENV,
        aliases=("retriever_model_driven", "retriever_model_driven_enabled"),
    )

def _direct_plan_known_slots_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _default_off_flag_enabled(
        context,
        DIRECT_PLAN_KNOWN_SLOTS_ENV,
        aliases=("direct_plan_known_slots", "direct_plan_known_slots_enabled"),
    )

def _direct_keyword_fallback_relevance_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _default_off_flag_enabled(
        context,
        DIRECT_KEYWORD_FALLBACK_RELEVANCE_ENV,
        aliases=("direct_keyword_fallback_relevance", "direct_keyword_fallback_relevance_enabled"),
    )

def _direct_slot_topic_shadow_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _default_off_flag_enabled(
        context,
        DIRECT_SLOT_TOPIC_SHADOW_ENV,
        aliases=("direct_slot_topic_shadow", "direct_slot_topic_shadow_enabled"),
    )

def _direct_p0_text_hygiene_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _default_off_flag_enabled(
        context,
        DIRECT_P0_TEXT_HYGIENE_ENV,
        aliases=("direct_p0_text_hygiene", "direct_p0_text_hygiene_enabled"),
    )

def _direct_path_known_slots_next_step_prompt_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _pilot_profile_default_on_flag_enabled(
        context,
        DIRECT_PATH_KNOWN_SLOTS_NEXT_STEP_PROMPT_ENV,
        aliases=(
            "direct_path_known_slots_next_step_prompt",
            "known_slots_next_step_prompt",
            "known_slots_no_reask_prompt",
        ),
    )

def _direct_path_answerability_shadow_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _answerability_shadow_enabled(context)


def _retriever_need_declaration_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _retriever_need_shadow_enabled(context) or _retriever_model_driven_enabled(context)


def _bot_safe_crm_context_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _default_off_flag_enabled(
        context,
        BOT_SAFE_CRM_CONTEXT_ENV,
        aliases=(
            "bot_safe_crm_context",
            "bot_safe_crm_context_enabled",
            "bot_safe_summary_context",
            "bot_safe_summary_context_enabled",
        ),
    )

def _route_rubric_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _pilot_profile_flag_enabled(context, ROUTE_RUBRIC_ENV, aliases=("route_rubric_enabled",))


def _presale_safety_enabled(context: Optional[Mapping[str, Any]] = None, *, subflag: str = "") -> bool:
    if isinstance(context, Mapping):
        if subflag and subflag in context:
            return _truthy_value(context.get(subflag))
        if PRESALE_SAFETY_ENV in context:
            return _truthy_value(context.get(PRESALE_SAFETY_ENV))
    if subflag and subflag in os.environ:
        return _truthy_value(os.getenv(subflag))
    if PRESALE_SAFETY_ENV in os.environ:
        return _truthy_value(os.getenv(PRESALE_SAFETY_ENV))
    return _pilot_gold_profile_enabled(context)

def _direct_path_brand_label(active_brand: str) -> str:
    brand = str(active_brand or "").strip().casefold()
    if brand == "foton":
        return "Фотон"
    if brand == "unpk":
        return "УНПК МФТИ"
    return "текущего учебного центра"






def _direct_path_snapshot_fact_key(fact: Mapping[str, Any]) -> str:
    return str(fact.get("fact_key") or fact.get("fact_id") or fact.get("id") or "").strip()





def _template_from_kb_context_trace(context: Optional[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(context, Mapping):
        return ()
    value = context.get("template_from_kb_trace")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))



def _direct_path_fact_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _client_clean_fact_text(value)
    if isinstance(value, Mapping):
        for key in ("client_safe_text", "fact_text", "manager_display_text", "text", "answer", "draft_text"):
            text = str(value.get(key) or "").strip()
            if text:
                return _client_clean_fact_text(text)
        return ""
    return _client_clean_fact_text(str(value))

def _direct_path_add_fact(items: dict[str, str], key: str, value: Any) -> None:
    fact_key = str(key or "").strip()
    text = _direct_path_fact_text(value)
    if fact_key and text:
        items.setdefault(fact_key, text)

def _direct_path_legacy_context_fact_allowed(value: Any, *, active_brand: str) -> bool:
    if not isinstance(value, Mapping):
        return True
    brand = str(value.get("brand") or value.get("active_brand") or "").strip().casefold()
    if brand and brand != str(active_brand or "").strip().casefold():
        return False
    if "allowed_for_client_answer" in value and value.get("allowed_for_client_answer") is not True:
        return False
    if "client_safe" in value and value.get("client_safe") is not True:
        return False
    if value.get("forbidden_for_client") is True or value.get("internal_only") is True:
        return False
    if "valid_until" in value and not _direct_path_valid_until_ok(value.get("valid_until")):
        return False
    return True

def _direct_path_add_legacy_fact(items: dict[str, str], key: str, value: Any, *, active_brand: str) -> None:
    if _direct_path_legacy_context_fact_allowed(value, active_brand=active_brand):
        _direct_path_add_fact(items, key, value)

def _direct_path_legacy_context_fact_items(context: Optional[Mapping[str, Any]], *, limit: int = 18) -> dict[str, str]:
    items: dict[str, str] = {}
    if not isinstance(context, Mapping):
        return items
    active_brand = _active_brand(context)
    confirmed = context.get("confirmed_facts")
    if isinstance(confirmed, Mapping):
        for key, value in confirmed.items():
            _direct_path_add_legacy_fact(items, str(key), value, active_brand=active_brand)
    facts_context = context.get("facts_context")
    if isinstance(facts_context, Mapping):
        confirmed_context = facts_context.get("confirmed_facts")
        if isinstance(confirmed_context, Mapping):
            for key, value in confirmed_context.items():
                _direct_path_add_legacy_fact(items, str(key), value, active_brand=active_brand)
    pipeline = context.get("dialogue_contract_pipeline")
    if isinstance(pipeline, Mapping) and isinstance(pipeline.get("retrieved_facts"), Mapping):
        for key, value in pipeline["retrieved_facts"].items():
            _direct_path_add_legacy_fact(items, str(key), value, active_brand=active_brand)
    snippets = context.get("knowledge_snippets")
    if isinstance(snippets, Mapping):
        for key, value in snippets.items():
            _direct_path_add_legacy_fact(items, f"snippet:{key}", value, active_brand=active_brand)
    elif isinstance(snippets, Sequence) and not isinstance(snippets, (str, bytes, bytearray)):
        for idx, value in enumerate(snippets, 1):
            _direct_path_add_legacy_fact(items, f"snippet:{idx}", value, active_brand=active_brand)
    return dict(list(items.items())[:limit])


def _direct_path_bot_safe_context_items(context: Optional[Mapping[str, Any]], *, limit: int = 3) -> tuple[Mapping[str, Any], ...]:
    if not _bot_safe_crm_context_enabled(context) or not isinstance(context, Mapping):
        return ()
    active_brand = _active_brand(context)
    if active_brand not in {"foton", "unpk"}:
        return ()
    containers: list[Any] = []
    timeline_context = context.get("timeline_context")
    if isinstance(timeline_context, Mapping):
        containers.append(timeline_context)
    read_only_context = context.get("read_only_customer_context")
    if isinstance(read_only_context, Mapping):
        nested_timeline = read_only_context.get("timeline_context")
        if isinstance(nested_timeline, Mapping):
            containers.append(nested_timeline)
        containers.append(read_only_context)
    result: list[Mapping[str, Any]] = []
    for container in containers:
        bot_context = container.get("bot_context") if isinstance(container, Mapping) else None
        if not isinstance(bot_context, Mapping):
            continue
        if bot_context.get("allowed_only") is not True:
            continue
        raw_items = bot_context.get("items")
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
            continue
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            if item.get("allowed_for_bot") is not True or item.get("requires_manager_review") is True:
                continue
            if str(item.get("chunk_type") or "").strip().casefold() != "bot_safe_summary":
                continue
            tags = {str(tag or "").strip().casefold() for tag in item.get("relevance_tags") or ()}
            if not _direct_path_bot_safe_item_visible(tags, active_brand=active_brand):
                continue
            text = str(item.get("summary") or item.get("text") or "").strip()
            if not text or _direct_path_bot_safe_text_has_pii(text):
                continue
            result.append(
                {
                    "chunk_type": "bot_safe_summary",
                    "text": _direct_path_trim_context_text(text, 700),
                    "event_at": str(item.get("event_at") or "").strip(),
                    "next_step_status": _direct_path_bot_safe_next_step_status(item),
                    "relevance_tags": [tag for tag in ("bot_safe", "structured", active_brand, "unknown") if tag in tags],
                }
            )
            if len(result) >= max(1, int(limit or 3)):
                return tuple(result)
    return tuple(result)


def _direct_path_bot_safe_next_step_status(item: Mapping[str, Any]) -> str:
    status = str(item.get("next_step_status") or "").strip().casefold()
    if not status:
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            next_step = metadata.get("next_step")
            if isinstance(next_step, Mapping):
                status = str(next_step.get("status") or "").strip().casefold()
    return status if status in {"active", "needs_manager_review", "empty"} else ""


def _direct_path_bot_safe_item_visible(tags: set[str], *, active_brand: str) -> bool:
    if "bot_safe" not in tags:
        return False
    known_brand_tags = tags & {"foton", "unpk"}
    if known_brand_tags - {active_brand}:
        return False
    return active_brand in tags or "unknown" in tags


def _direct_path_bot_safe_text_has_pii(text: str) -> bool:
    return bool(
        _A2_PHONE_RE.search(text)
        or _CLIENT_EMAIL_RE.search(text)
        or _BOT_SAFE_SERVICE_ID_RE.search(text)
        or _BOT_SAFE_PERSON_CONTEXT_RE.search(text)
    )


def _direct_path_trim_context_text(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split()).strip()
    return value if len(value) <= limit else value[: max(0, limit - 1)].rstrip() + "…"


def _direct_path_bot_safe_memory_prompt_text(text: str) -> str:
    value = _direct_path_trim_context_text(text, 700)
    return _BOT_SAFE_MEMORY_EXACT_DETAIL_RE.sub("<точная деталь из памяти скрыта>", value)


def _direct_path_bot_safe_context_prompt_block(context: Optional[Mapping[str, Any]]) -> str:
    if not _bot_safe_crm_context_enabled(context):
        return ""
    items = _direct_path_bot_safe_context_items(context)
    if not items:
        return ""
    statuses = {str(item.get("next_step_status") or "").strip().casefold() for item in items}
    has_unconfirmed_step = bool(statuses & {"needs_manager_review", "empty"})
    lines = [
        "Безопасная выжимка клиента: это разрешённая выжимка истории по активному бренду. "
        "Используй её только для продолжения диалога, понимания уже обсуждённого и следующего шага. "
        "Цены, даты и условия называй только из блока «Факты по вашему вопросу». "
        "Числа, даты, проценты, цены, расписание и адреса из этой выжимки НЕ называй клиенту как факт: "
        "если такая деталь нужна, бери её только из блока «Факты по вашему вопросу», а если её там нет — предложи уточнить. "
        "Память используй как нить разговора: «обсуждали расписание», без точных чисел из памяти. "
        "Не раскрывай клиенту, что данные взяты из CRM/истории/базы.",
    ]
    if "active" in statuses:
        lines.append("Если статус следующего шага «active», продолжай эту нить и называй шаг без лишних оговорок.")
    if has_unconfirmed_step:
        lines.append(
            "Если статус следующего шага «needs_manager_review» или «empty», следующий шаг НЕ подтверждён: "
            "не утверждай его клиенту, предложи уточнить с менеджером. "
            "Датированную историю с таким статусом подавай как прежние заметки: «по прежним заметкам, актуальность уточню»."
        )
    for idx, item in enumerate(items, 1):
        text = _direct_path_bot_safe_memory_prompt_text(str(item.get("text") or "").strip())
        event_at = str(item.get("event_at") or "").strip()
        suffix = f" ({event_at[:10]})" if event_at else ""
        status = str(item.get("next_step_status") or "").strip().casefold()
        status_suffix = f" [статус следующего шага: {status}]" if status else ""
        lines.append(f"{idx}. {text}{suffix}{status_suffix}")
    return "\n".join(lines)


def _direct_path_bot_safe_context_trace(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not _bot_safe_crm_context_enabled(context):
        return {"enabled": False, "reason": "bot_safe_crm_context_flag_off"}
    items = _direct_path_bot_safe_context_items(context)
    return {
        "enabled": True,
        "visible_items": len(items),
        "active_brand": _active_brand(context),
        "source": "read_only_customer_context.timeline_context.bot_context",
        "next_step_statuses": [str(item.get("next_step_status") or "") for item in items if str(item.get("next_step_status") or "")],
    }

DIRECT_PATH_CATEGORY_ALIASES: Mapping[str, tuple[str, ...]] = {
    "pricing": ("pricing", "price", "стоим", "цен", "дорог", "оплат", "рассроч", "долями", "скидк", "помесяч"),
    "schedule": ("schedule", "распис", "дни", "время", "когда", "старт", "начал", "будни", "выходн"),
    "camp": ("camp", "лагер", "летн", "смен", "лш", "лвш", "менделеево", "пацаева"),
    "documents": ("document", "documents", "certificate", "tax", "matkap", "документ", "справк", "вычет", "маткап"),
    "format": ("format", "platform", "recording", "онлайн", "очно", "платформ", "запис", "формат"),
    "enrollment": ("enrollment", "trial", "запис", "оформ", "пробн", "вступ", "тест"),
    "contact": ("contact", "contacts", "phone", "email", "e-mail", "mail", "контакт", "телефон", "номер", "почт", "связаться"),
    "address": ("address", "location", "transport", "адрес", "где", "дорог", "добир", "трансфер"),
    "course": ("teacher", "program", "homework", "materials", "level", "преподав", "программ", "дз", "материал", "уров"),
}

def _direct_path_fact_categories(fact: Mapping[str, Any]) -> frozenset[str]:
    key = _direct_path_snapshot_fact_key(fact).casefold()
    fact_type = str(fact.get("fact_type") or "").casefold()
    product = str(fact.get("product") or "").casefold()
    text = _normalize_fact_match_text(f"{key} {fact_type} {product} {_direct_path_snapshot_fact_text(fact)}")
    haystack = f"{key} {fact_type} {product} {text}"
    categories: set[str] = set()
    if fact_type in {"price", "discount", "installment", "payment", "payment_method"} or re.search(r"₽|руб|%|скид|рассроч|долями|помесяч", haystack):
        categories.add("pricing")
    if fact_type in {"schedule", "deadline"} or re.search(r"распис|старт|начал|дни|время|будни|выходн|дедлайн", haystack):
        categories.add("schedule")
    if "camp" in fact_type or re.search(r"лагер|летн|смен|лш|лвш|менделеево|пацаева|city_camp", haystack):
        categories.add("camp")
    if fact_type in {"documents", "tax", "matkap", "certificate"} or re.search(r"документ|справк|вычет|маткап|лиценз", haystack):
        categories.add("documents")
    if fact_type in {"format", "platform", "recording"} or re.search(r"онлайн|очно|платформ|запис[ьи] занят|формат|методич", haystack):
        categories.add("format")
    if fact_type in {"trial", "enrollment"} or re.search(r"пробн|записат|записаться|оформ|вступительн|тест", haystack):
        categories.add("enrollment")
    if (
        fact_type == "contact"
        or "contacts_" in haystack
        or re.search(r"контакт|телефон|phone|toll_free|email|e-mail|почт", haystack)
    ):
        categories.add("contact")
    if fact_type in {"address", "location", "transport"} or re.search(r"адрес|локац|москва|долгопруд|дорог|добир|трансфер", haystack):
        categories.add("address")
    if fact_type in {"teacher", "program", "homework", "materials", "level", "course_parameter"} or re.search(r"программ|преподав|домашн|дз|материал|уров|заняти|ак\.ч", haystack):
        categories.add("course")
    return frozenset(categories or {"course"})

def _direct_path_category_from_hint(value: Any) -> str:
    text = _normalize_fact_match_text(value)
    if not text:
        return ""
    if text in {"pricing", "price", "discount", "installment", "payment_method", "payment_status"}:
        return "pricing"
    if text in {"schedule", "start", "when_start"}:
        return "schedule"
    if text in {"camp", "camp_lvsh", "camp_city", "residential_lvsh"}:
        return "camp"
    if text in {"document", "documents", "tax", "matkap", "certificate"}:
        return "documents"
    if text in {"format", "platform", "recording", "materials"}:
        return "format"
    if text in {"enrollment", "trial", "readiness"}:
        return "enrollment"
    if text in {"contact", "contacts", "phone", "email", "mail"}:
        return "contact"
    if text in {"transport", "logistics", "travel_time", "route_logistics", "address"}:
        return "address"
    if text in {"teacher", "program", "homework", "level", "value", "course_pick"}:
        return "course"
    for category, aliases in DIRECT_PATH_CATEGORY_ALIASES.items():
        if any(alias in text for alias in aliases):
            return category
    return ""

def _direct_path_selected_categories(client_message: str, context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    values: list[Any] = []
    if isinstance(context, Mapping):
        for container_key in ("conversation_intent_plan", "dialogue_memory_view", "answer_contract", "facts_context"):
            container = context.get(container_key)
            if not isinstance(container, Mapping):
                continue
            for key in ("primary_intent", "topic_id", "question_kind", "fact_scope", "product_family"):
                values.append(container.get(key))
            for key in ("answer_topics", "topic_roles", "active_topics", "required_fact_keys"):
                seq = container.get(key)
                if isinstance(seq, Sequence) and not isinstance(seq, (str, bytes, bytearray)):
                    values.extend(seq)
            held = container.get("held_state") if isinstance(container.get("held_state"), Mapping) else {}
            if held:
                values.extend(held.get("active_topics") or ())
                values.extend(held.get("required_fact_keys") or ())
            focus = container.get("topic_focus") if isinstance(container.get("topic_focus"), Mapping) else {}
            if focus:
                values.extend(focus.values())
    values.append(client_message)
    categories: list[str] = []
    for value in values:
        category = _direct_path_category_from_hint(value)
        if category and category not in categories:
            categories.append(category)
    client_category = _direct_path_category_from_hint(client_message)
    if client_category in {"contact", "address"}:
        categories = [client_category, *[item for item in categories if item != client_category]]
    return tuple(categories[:2])

_ASSUMED_SCOPE_KEYS = frozenset(
    {
        "format",
        "training_format",
        "grade",
        "class",
        "subject",
        "course_subject",
        "product",
        "product_family",
    }
)

_CONFIRMED_SLOT_SOURCES = {"dialogue_memory", "memory_provenance"}


def _direct_path_add_slot_provenance(
    result: dict[str, dict[str, Any]],
    key: Any,
    value: Any,
    *,
    source: str,
    quote: str = "",
    confirmed: bool = False,
) -> None:
    normalized_key = str(key or "").strip()
    text = " ".join(str(value or "").split())
    if not normalized_key or not text:
        return
    existing = result.get(normalized_key)
    confirmed = bool(confirmed)
    if existing and existing.get("confirmed") and not confirmed:
        return
    result[normalized_key] = {
        "value": text,
        "source": str(source or "unknown"),
        "quote": str(quote or "").strip()[:160],
        "confirmed": confirmed,
        "status": "confirmed_by_client" if confirmed else "assumed_from_context",
    }


def _direct_path_slot_provenance(context: Optional[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(context, Mapping):
        return result
    containers: list[Mapping[str, Any]] = [context]
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        containers.insert(0, memory)

    for container in containers:
        provenance = container.get("slot_provenance")
        if isinstance(provenance, Mapping):
            for key, raw in provenance.items():
                if not isinstance(raw, Mapping):
                    continue
                source = str(raw.get("source") or "").strip()
                quote = str(raw.get("quote") or "").strip()
                _direct_path_add_slot_provenance(
                    result,
                    key,
                    raw.get("value"),
                    source=source or "slot_provenance",
                    quote=quote,
                    confirmed=bool(quote and source in _CONFIRMED_SLOT_SOURCES),
                )

    for container in containers:
        slot_history = container.get("slot_history")
        if not isinstance(slot_history, Sequence) or isinstance(slot_history, (str, bytes, bytearray)):
            continue
        for item in slot_history:
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source") or "").strip()
            quote = str(item.get("quote") or "").strip()
            if not quote or source not in _CONFIRMED_SLOT_SOURCES:
                continue
            _direct_path_add_slot_provenance(
                result,
                item.get("slot") or item.get("key") or item.get("name"),
                item.get("value"),
                source=source,
                quote=quote,
                confirmed=True,
            )

    for container in containers:
        for source_key, source in (
            ("client_confirmed_slots", "client_confirmed_slots"),
            ("crm_known_slots", "crm_known_slots"),
            ("bot_inferred_slots", "bot_inferred_slots"),
            ("known_slots", "known_slots"),
            ("known_dialog_fields", "known_dialog_fields"),
        ):
            slots = container.get(source_key)
            if not isinstance(slots, Mapping):
                continue
            for key, value in slots.items():
                existing = result.get(str(key))
                if existing and existing.get("confirmed"):
                    continue
                _direct_path_add_slot_provenance(
                    result,
                    key,
                    value,
                    source=source,
                    quote=str(existing.get("quote") or "") if existing else "",
                    confirmed=bool(existing and existing.get("confirmed")),
                )

    return result


def _direct_path_all_slot_scope(context: Optional[Mapping[str, Any]]) -> Mapping[str, str]:
    slots = _direct_path_known_slots(context)
    focus: Mapping[str, Any] = {}
    if isinstance(context, Mapping) and isinstance(context.get("dialogue_memory_view"), Mapping):
        memory = context["dialogue_memory_view"]
        if isinstance(memory.get("topic_focus"), Mapping):
            focus = memory["topic_focus"]  # type: ignore[assignment]
    merged = {**dict(focus), **slots}
    result: dict[str, str] = {}
    for key in ("format", "training_format", "grade", "class", "product", "product_family"):
        value = str(merged.get(key) or "").strip()
        if value:
            result[key] = value
    return result


def _direct_path_confirmed_slot_scope(context: Optional[Mapping[str, Any]]) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for key, data in _direct_path_slot_provenance(context).items():
        if key not in _ASSUMED_SCOPE_KEYS or not data.get("confirmed"):
            continue
        value = str(data.get("value") or "").strip()
        if value:
            result[key] = value
    return result


def _direct_path_soft_slot_scope(context: Optional[Mapping[str, Any]]) -> Mapping[str, str]:
    if not _assumed_scope_guard_enabled(context):
        return _direct_path_all_slot_scope(context)
    result = dict(_direct_path_all_slot_scope(context))
    for key, data in _direct_path_slot_provenance(context).items():
        if key in _ASSUMED_SCOPE_KEYS and str(data.get("value") or "").strip():
            result[key] = str(data.get("value") or "").strip()
    return result


def _direct_path_slot_scope(context: Optional[Mapping[str, Any]]) -> Mapping[str, str]:
    if _assumed_scope_guard_enabled(context):
        return _direct_path_confirmed_slot_scope(context)
    return _direct_path_all_slot_scope(context)


def _direct_path_format_scope(value: str) -> str:
    text = _normalize_fact_match_text(value)
    if "онлайн" in text or "online" in text:
        return "online"
    if "очно" in text or "offline" in text or "москва" in text or "долгопруд" in text:
        return "offline"
    return ""

def _direct_path_grade_in_fact(grade: str, fact_text: str) -> bool:
    if not grade.isdigit():
        return True
    value = int(grade)
    text = _normalize_fact_match_text(fact_text)
    ranges = [(int(a), int(b)) for a, b in re.findall(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(?:класс|кл)", text)]
    singles = [int(item) for item in re.findall(r"\b(\d{1,2})\s*(?:класс|кл)\b", text)]
    if ranges:
        return any(start <= value <= end for start, end in ranges)
    if singles:
        return value in singles
    return True

def _direct_path_fact_conflicts_slots(fact: Mapping[str, Any], slots: Mapping[str, str]) -> bool:
    haystack = f"{_direct_path_snapshot_fact_key(fact)} {_direct_path_snapshot_fact_text(fact)} {fact.get('product') or ''}"
    slot_format = _direct_path_format_scope(slots.get("format") or slots.get("training_format") or "")
    fact_format = _direct_path_format_scope(haystack)
    if slot_format and fact_format and slot_format != fact_format:
        return True
    grade = re.sub(r"\D+", "", str(slots.get("grade") or slots.get("class") or ""))
    if grade and not _direct_path_grade_in_fact(grade, haystack):
        return True
    family = _normalize_fact_match_text(slots.get("product_family") or slots.get("product") or "")
    fact_text = _normalize_fact_match_text(haystack)
    fact_is_camp = bool(re.search(r"лагер|летн|смен|лш|лвш|camp", fact_text))
    if family in {"regular_course", "regular"} and fact_is_camp:
        return True
    if family == "camp" and not fact_is_camp and any(marker in fact_text for marker in ("курс", "учебный год", "семестр")):
        return True
    return False

def _direct_path_fact_relevance_score(
    fact: Mapping[str, Any],
    *,
    client_message: str,
    categories: Sequence[str],
    slots: Mapping[str, str],
    soft_slots: Optional[Mapping[str, str]] = None,
) -> int:
    haystack = _normalize_fact_match_text(
        f"{_direct_path_snapshot_fact_key(fact)} {fact.get('fact_type') or ''} {fact.get('product') or ''} {_direct_path_snapshot_fact_text(fact)}"
    )
    score = 0
    if _direct_path_fact_categories(fact).intersection(categories):
        score += 30
    if not _direct_path_fact_conflicts_slots(fact, slots):
        score += 20
    boost_slots = soft_slots if soft_slots is not None else slots
    for value in boost_slots.values():
        normalized = _normalize_fact_match_text(value)
        if normalized and normalized in haystack:
            score += 8
    for token in re.findall(r"[a-zа-яё0-9]{4,}", _normalize_fact_match_text(client_message)):
        if token in haystack:
            score += 2
    if str(fact.get("bot_template_required") or "").casefold() == "true":
        score += 1
    return score

def _direct_path_fact_has_positive_question_relevance(
    fact: Mapping[str, Any],
    *,
    client_message: str,
    categories: Sequence[str],
    slots: Mapping[str, str],
    soft_slots: Optional[Mapping[str, str]] = None,
) -> bool:
    if _direct_path_fact_categories(fact).intersection(categories):
        return True
    haystack = _normalize_fact_match_text(
        f"{_direct_path_snapshot_fact_key(fact)} {fact.get('fact_type') or ''} {fact.get('product') or ''} {_direct_path_snapshot_fact_text(fact)}"
    )
    boost_slots = soft_slots if soft_slots is not None else slots
    for value in boost_slots.values():
        normalized = _normalize_fact_match_text(value)
        if normalized and normalized in haystack:
            return True
    for token in re.findall(r"[a-zа-яё0-9]{4,}", _normalize_fact_match_text(client_message)):
        if token in haystack:
            return True
    return str(fact.get("bot_template_required") or "").casefold() == "true"

def _direct_path_known_grade_subject(context: Optional[Mapping[str, Any]]) -> tuple[str, str]:
    known: Mapping[str, Any] = _direct_path_slot_scope(context) if _assumed_scope_guard_enabled(context) else _direct_path_known_slots(context)
    grade = re.sub(r"\D+", "", str(known.get("grade") or known.get("class") or ""))[:2]
    subject = _normalize_fact_match_text(known.get("subject") or known.get("course_subject") or "")
    return grade, subject

def _direct_path_subject_matches_fact(subject: str, fact_text: str) -> bool:
    if not subject:
        return False
    text = _normalize_fact_match_text(fact_text)
    subject_markers = (
        ("физик", ("физик", "physics")),
        ("математ", ("математ", "math")),
        ("информат", ("информат", "программирован", "informatics", "programming")),
        ("русск", ("русск", "russian")),
        ("англий", ("англий", "english")),
        ("хими", ("хими", "chemistry")),
        ("биолог", ("биолог", "biology")),
    )
    for marker, aliases in subject_markers:
        if marker in subject:
            return any(alias in text for alias in aliases)
    return subject in text

def _direct_path_regular_course_price_fact(fact: Mapping[str, Any], fact_text: str) -> bool:
    product = _normalize_fact_match_text(fact.get("product") or "")
    text = _normalize_fact_match_text(fact_text)
    if any(marker in text for marker in ("лагер", "лвш", "смен", "интенсив", "огэ интенсив", "егэ интенсив")):
        return False
    return "regular" in product or "regular_courses" in text or "учебный год" in text or "онлайн" in text or "очно" in text

def _direct_path_course_fact_supplements(
    records: Sequence[Mapping[str, Any]],
    *,
    context: Optional[Mapping[str, Any]],
    slots: Mapping[str, str],
    existing_keys: set[str],
) -> tuple[Mapping[str, Any], ...]:
    grade, subject = _direct_path_known_grade_subject(context)
    if not grade or not subject:
        return ()
    result: list[Mapping[str, Any]] = []
    for fact in records:
        key = _direct_path_snapshot_fact_key(fact)
        if not key or key in existing_keys:
            continue
        categories = _direct_path_fact_categories(fact)
        if "pricing" not in categories and "schedule" not in categories:
            continue
        haystack = f"{key} {fact.get('fact_type') or ''} {fact.get('product') or ''} {_direct_path_snapshot_fact_text(fact)}"
        if not _direct_path_grade_in_fact(grade, haystack):
            continue
        if _direct_path_fact_conflicts_slots(fact, slots):
            continue
        fact_type = str(fact.get("fact_type") or "").strip().casefold()
        if "schedule" in categories:
            if not _direct_path_subject_matches_fact(subject, haystack):
                continue
        elif "pricing" in categories:
            if fact_type != "price":
                continue
            if not _direct_path_regular_course_price_fact(fact, haystack):
                continue
        result.append(fact)
    return tuple(result)

def _direct_path_render_fact_line(key: str, text: str, meta: Mapping[str, str]) -> str:
    fact_type = str(meta.get("fact_type") or "").strip()
    product = str(meta.get("product") or "").strip()
    suffix = "; ".join(part for part in (f"fact_type={fact_type}" if fact_type else "", f"product={product}" if product else "") if part)
    return f"- {key}" + (f" ({suffix})" if suffix else "") + f": {text}"

def _direct_path_render_fact_block(
    facts: Mapping[str, str],
    *,
    fact_metadata: Mapping[str, Mapping[str, str]],
    keys: Sequence[str],
) -> str:
    lines = [
        _direct_path_render_fact_line(str(key), str(facts.get(str(key)) or ""), fact_metadata.get(str(key), {}))
        for key in keys
        if str(key).strip() and str(facts.get(str(key)) or "").strip()
    ]
    return "\n".join(lines) or "(нет подтверждённых фактов в этом блоке)"

def _direct_path_fact_pack_char_count(facts: Mapping[str, str], meta: Mapping[str, Mapping[str, str]], keys: Sequence[str]) -> int:
    return sum(len(_direct_path_render_fact_line(key, facts.get(key, ""), meta.get(key, {}))) + 1 for key in keys)

def _direct_path_core_fact(fact: Mapping[str, Any]) -> bool:
    key = _direct_path_snapshot_fact_key(fact).casefold()
    text = _normalize_fact_match_text(f"{key} {_direct_path_snapshot_fact_text(fact)} {fact.get('fact_type') or ''} {fact.get('product') or ''}")
    return bool(
        re.search(
            r"цен|стоим|₽|руб|скид|рассроч|долями|формат|онлайн|очно|старт|адрес|пробн|запис|учебный год|заняти",
            text,
            re.I,
        )
    )

def _direct_path_empty_fact_pack(active_brand: str, *, selected_category: str = "empty") -> Mapping[str, Any]:
    return {
        "schema_version": DIRECT_PATH_WIDE_FACT_PACK_SCHEMA_VERSION,
        "facts": {},
        "exact_keys": [],
        "adjacent_keys": [],
        "selected_category": selected_category,
        "fact_metadata": {},
    }

def _direct_path_records_to_fact_pack(
    *,
    active_brand: str,
    legacy: Mapping[str, str],
    exact_records: Sequence[Mapping[str, Any]],
    adjacent_records: Sequence[Mapping[str, Any]],
    selected_category: str,
    max_facts: int,
    max_chars: int,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    facts: dict[str, str] = {}
    meta: dict[str, dict[str, str]] = {}

    def add_record(fact: Mapping[str, Any], *, fact_limit: int = max_facts, char_limit: int = max_chars) -> bool:
        key = _direct_path_snapshot_fact_key(fact)
        text = _direct_path_snapshot_fact_text(fact)
        if not key or not text or key in facts:
            return False
        prospective = {**facts, key: text}
        prospective_meta = {
            **meta,
            key: {
                "brand": str(fact.get("brand") or ""),
                "fact_type": str(fact.get("fact_type") or ""),
                "product": str(fact.get("product") or ""),
            },
        }
        if len(prospective) > fact_limit:
            return False
        if _direct_path_fact_pack_char_count(prospective, prospective_meta, list(prospective.keys())) > char_limit:
            return False
        facts[key] = text
        meta[key] = prospective_meta[key]
        return True

    adjacent_reserve = min(8, len(adjacent_records)) if adjacent_records else 0
    exact_fact_limit = max(1, max_facts - adjacent_reserve)
    exact_char_limit = max(2000, max_chars - (1200 if adjacent_records else 0))
    for fact in exact_records:
        add_record(fact, fact_limit=exact_fact_limit, char_limit=exact_char_limit)
    exact_keys = list(facts.keys())
    for fact in adjacent_records:
        add_record(fact)
    adjacent_keys = [key for key in facts if key not in set(exact_keys)]

    if not facts:
        facts = dict(legacy)
        exact_keys = list(facts.keys())
        adjacent_keys = []
        meta = {key: {"brand": active_brand, "fact_type": "", "product": ""} for key in facts}
        selected_category = "legacy_context"

    result: dict[str, Any] = {
        "schema_version": DIRECT_PATH_WIDE_FACT_PACK_SCHEMA_VERSION,
        "facts": facts,
        "exact_keys": exact_keys,
        "adjacent_keys": adjacent_keys,
        "selected_category": selected_category,
        "fact_metadata": meta,
    }
    if extra_metadata:
        result.update(dict(extra_metadata))
    return result

def _direct_path_keyword_fact_pack_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    legacy: Mapping[str, str],
    active_brand: str,
    context: Optional[Mapping[str, Any]],
    client_message: str,
    max_facts: int,
    max_chars: int,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    categories = _direct_path_selected_categories(client_message, context)
    selected_category = "+".join(categories) if categories else "fallback_core"
    if _direct_keyword_fallback_relevance_enabled(context) and not categories:
        candidates = list(records)
        selected_category = "fallback_relevance"
    else:
        candidates = [
            fact
            for fact in records
            if (_direct_path_core_fact(fact) if not categories else bool(_direct_path_fact_categories(fact).intersection(categories)))
        ]
    if not candidates:
        if _direct_keyword_fallback_relevance_enabled(context):
            candidates = []
            selected_category = "fallback_relevance"
        else:
            candidates = [fact for fact in records if _direct_path_core_fact(fact)]
            selected_category = "fallback_core"
    if not candidates:
        if _direct_keyword_fallback_relevance_enabled(context):
            candidates = list(records)
            selected_category = "fallback_relevance"
        else:
            candidates = list(records)[:max_facts]
            selected_category = "fallback_core"

    slots = _direct_path_slot_scope(context)
    soft_slots = _direct_path_soft_slot_scope(context)
    scored = [
        (
            _direct_path_fact_relevance_score(
                fact,
                client_message=client_message,
                categories=categories or ("pricing", "format", "schedule", "address", "course"),
                slots=slots,
                soft_slots=soft_slots,
            ),
            idx,
            fact,
        )
        for idx, fact in enumerate(candidates)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    if _direct_keyword_fallback_relevance_enabled(context) and selected_category == "fallback_relevance":
        scored = [
            item
            for item in scored
            if _direct_path_fact_has_positive_question_relevance(
                item[2],
                client_message=client_message,
                categories=categories or ("pricing", "format", "schedule", "address", "course"),
                slots=slots,
                soft_slots=soft_slots,
            )
        ]
    ordered = [fact for _, _, fact in scored]

    has_scope_slots = bool(slots)
    exact_records: list[Mapping[str, Any]] = []
    adjacent_records: list[Mapping[str, Any]] = []
    for fact in ordered:
        conflicts = _direct_path_fact_conflicts_slots(fact, slots)
        if not has_scope_slots and selected_category == "pricing":
            exact_records.append(fact)
        elif conflicts:
            adjacent_records.append(fact)
        else:
            exact_records.append(fact)
    if not exact_records and ordered:
        exact_records = [ordered[0]]
        adjacent_records = ordered[1:]

    return _direct_path_records_to_fact_pack(
        active_brand=active_brand,
        legacy=legacy,
        exact_records=exact_records,
        adjacent_records=adjacent_records,
        selected_category=selected_category,
        max_facts=max_facts,
        max_chars=max_chars,
        extra_metadata=extra_metadata,
    )

def _direct_path_retriever_candidate_summary(fact: Mapping[str, Any]) -> str:
    text = _direct_path_snapshot_fact_text(fact)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 220:
        text = text[:217].rstrip() + "..."
    fact_type = str(fact.get("fact_type") or "").strip()
    product = str(fact.get("product") or "").strip()
    prefix = "; ".join(item for item in (f"fact_type={fact_type}" if fact_type else "", f"product={product}" if product else "") if item)
    return f"{prefix}: {text}" if prefix else text

def _direct_path_required_fact_keys(context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    values: list[str] = []

    def add_many(raw: Any) -> None:
        if isinstance(raw, str):
            seq: Sequence[Any] = [raw]
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
            seq = raw
        else:
            return
        for item in seq:
            key = str(item or "").strip()
            if key and key not in values:
                values.append(key)

    add_many(context.get("required_fact_keys"))
    plan = context.get("conversation_intent_plan")
    if isinstance(plan, Mapping):
        add_many(plan.get("required_fact_keys"))
    facts_context = context.get("facts_context")
    if isinstance(facts_context, Mapping):
        add_many(facts_context.get("required_fact_keys"))
    return tuple(values)

def _direct_path_retriever_mode(context: Optional[Mapping[str, Any]]) -> str:
    if _retriever_model_driven_enabled(context):
        return "model_driven"
    if _retriever_need_shadow_enabled(context):
        return "need_shadow"
    return "id_only"

def _compact_retriever_text(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > max_chars:
        return text[: max_chars - 1].rstrip() + "…"
    return text

def _direct_path_needed_fact_declaration(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("needed_facts") or payload.get("needed_fact_requests") or payload.get("facts_needed")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    result: list[dict[str, str]] = []
    allowed_keys = (
        "theme",
        "fact_type",
        "brand",
        "grade",
        "subject",
        "format",
        "product",
        "why_needed",
        "importance",
    )
    for item in raw[:20]:
        if not isinstance(item, Mapping):
            continue
        normalized = {
            key: _compact_retriever_text(item.get(key), max_chars=260 if key == "why_needed" else 80)
            for key in allowed_keys
            if _compact_retriever_text(item.get(key), max_chars=260 if key == "why_needed" else 80)
        }
        if normalized:
            result.append(normalized)
    return result

def _direct_path_fact_type_root(value: str) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    return re.split(r"[.:/\s_-]+", text, maxsplit=1)[0]

def _direct_path_declaration_comparison(
    *,
    keyword_required_fact_keys: Sequence[str],
    needed_facts: Sequence[Mapping[str, str]],
) -> Mapping[str, Any]:
    keyword_types = sorted(
        {
            _direct_path_fact_type_root(key)
            for key in keyword_required_fact_keys
            if _direct_path_fact_type_root(key)
        }
    )
    model_types = sorted(
        {
            _direct_path_fact_type_root(str(item.get("fact_type") or ""))
            for item in needed_facts
            if _direct_path_fact_type_root(str(item.get("fact_type") or ""))
        }
    )
    keyword_set = set(keyword_types)
    model_set = set(model_types)
    return {
        "keyword_fact_types": keyword_types,
        "model_fact_types": model_types,
        "model_only_fact_types": sorted(model_set - keyword_set),
        "keyword_only_fact_types": sorted(keyword_set - model_set),
    }

def build_direct_path_llm_retriever_prompt(
    client_message: str,
    *,
    context: Optional[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> str:
    recent = "\n".join(_direct_path_recent_messages(context, limit=6)) or "(нет истории)"
    slots = json.dumps(_direct_path_prompt_known_slots(context), ensure_ascii=False, sort_keys=True)
    need_declaration = _retriever_need_declaration_enabled(context)
    model_driven = _retriever_model_driven_enabled(context)
    plan = {}
    if isinstance(context, Mapping) and isinstance(context.get("conversation_intent_plan"), Mapping):
        source_plan = context["conversation_intent_plan"]
        plan_keys = ("primary_intent", "answer_topics", "planner_slots", "planner_confidence")
        if not model_driven:
            plan_keys = ("primary_intent", "answer_topics", "required_fact_keys", "planner_slots", "planner_confidence")
        plan = {
            key: source_plan.get(key)
            for key in plan_keys
            if key in source_plan
        }
    plan_json = json.dumps(plan, ensure_ascii=False, sort_keys=True)
    lines = []
    for fact in candidates:
        key = _direct_path_snapshot_fact_key(fact)
        if not key:
            continue
        lines.append(f"- {key}: {_direct_path_retriever_candidate_summary(fact)}")
    candidate_block = "\n".join(lines) or "(нет кандидатов)"
    declaration_instruction = ""
    json_schema = '{"exact_ids":["fact.id"],"adjacent_ids":["fact.id"]}'
    if need_declaration:
        driver_line = (
            "В этом режиме сам по смыслу определи, какие факты нужны для ответа; "
            "не жди внешней подсказки с готовыми ключами фактов.\n"
            if model_driven
            else "Наличие needed_facts не должно менять exact_ids и adjacent_ids: сначала выбери id как в обычном режиме, затем опиши нужные факты.\n"
        )
        declaration_instruction = (
            "\nДополнительно верни needed_facts — структурированную декларацию того, какие факты нужны клиенту.\n"
            f"Версия схемы декларации: {RETRIEVER_NEED_DECLARATION_SCHEMA_VERSION}.\n"
            f"{driver_line}"
            "Каждый элемент needed_facts: theme, fact_type, brand, grade, subject, format, product, "
            "why_needed, importance. importance только required или helpful. Если нужных фактов нет, верни пустой список.\n"
        )
        json_schema = (
            '{"needed_facts":[{"theme":"pricing","fact_type":"price","brand":"foton",'
            '"grade":"9","subject":"физика","format":"онлайн","product":"regular_course",'
            '"why_needed":"клиент спрашивает стоимость","importance":"required"}],'
            '"exact_ids":["fact.id"],"adjacent_ids":["fact.id"]}'
        )
    return (
        "Ты выбираешь факты для черновика ответа учебного центра.\n"
        "Твоя задача — выбрать id фактов из списка кандидатов. Не пиши клиентский текст.\n"
        "Выбирай ВСЕ факты, которые могут помочь ответить на вопрос, включая смысловые связи и следующий шаг; "
        "не ограничивайся дословными совпадениями.\n"
        "Если текущий вопрос неполный («а по физике?», «а очно?») — восстанови его по последним репликам диалога "
        "и подбирай факты для восстановленного вопроса.\n"
        "exact_ids — факты, которые прямо отвечают на вопрос или его часть. adjacent_ids — смежные полезные факты.\n"
        "Нельзя выдумывать id: используй только id из списка кандидатов.\n\n"
        f"Вопрос клиента:\n{client_message}\n\n"
        f"Последние реплики:\n{recent}\n\n"
        f"Известные слоты: {slots}\n"
        f"План/интент: {plan_json}\n\n"
        f"Кандидаты:\n{candidate_block}\n\n"
        f"{declaration_instruction}"
        f"Верни строго JSON: {json_schema}"
    )

_DIRECT_SLOT_TOPIC_SHADOW_SLOT_KEYS = frozenset(
    {"grade", "class", "subject", "course_subject", "format", "training_format", "product", "product_family", "level"}
)

def build_direct_path_slot_topic_shadow_prompt(
    client_message: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    plan = context.get("conversation_intent_plan") if isinstance(context, Mapping) and isinstance(context.get("conversation_intent_plan"), Mapping) else {}
    recent = "\n".join(_direct_path_recent_messages(context, limit=6)) or "(диалог только начался)"
    slots = json.dumps(_direct_path_prompt_known_slots(context), ensure_ascii=False, sort_keys=True)
    return (
        "Ты делаешь только теневой анализ для аудита. Не пиши ответ клиенту и не выбирай маршрут.\n"
        "Извлеки из текущего сообщения и последних реплик возможные учебные слоты и тему вопроса. "
        "Если не уверен — оставь поле пустым и снизь confidence. Не выдумывай данные.\n\n"
        f"Текущее сообщение:\n{client_message}\n\n"
        f"Последние реплики:\n{recent}\n\n"
        f"Уже известные слоты:\n{slots}\n\n"
        f"План/интент регулярного слоя: {json.dumps(plan, ensure_ascii=False, sort_keys=True)[:1800]}\n\n"
        "Верни строго JSON без Markdown:\n"
        "{\n"
        '  "model_slots": {"grade": "", "subject": "", "format": "", "product": "", "product_family": ""},\n'
        '  "model_topic": "",\n'
        '  "evidence_quote": "",\n'
        '  "confidence": 0.0\n'
        "}\n"
    )

def _direct_path_shadow_safe_slots(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, raw in value.items():
        normalized_key = str(key or "").strip()
        if normalized_key not in _DIRECT_SLOT_TOPIC_SHADOW_SLOT_KEYS:
            continue
        safe_value = _presale_prompt_safe_slot_value(normalized_key, raw)
        if safe_value not in ("", {}, []):
            result[normalized_key] = str(safe_value)
    return result

def build_direct_path_slot_topic_shadow_metadata(
    client_message: str,
    *,
    context: Optional[Mapping[str, Any]],
    shadow_fn: Optional[Callable[[str], Mapping[str, Any] | str]],
) -> Optional[Mapping[str, Any]]:
    if not _direct_slot_topic_shadow_enabled(context):
        return None
    plan = context.get("conversation_intent_plan") if isinstance(context, Mapping) and isinstance(context.get("conversation_intent_plan"), Mapping) else {}
    metadata: dict[str, Any] = {
        "schema_version": DIRECT_SLOT_TOPIC_SHADOW_SCHEMA_VERSION,
        "enabled": True,
        "used": False,
        "fallback_reason": "",
        "model_slots": {},
        "model_topic": "",
        "evidence_quote": "",
        "confidence": 0.0,
        "direct_known_slots": _presale_prompt_safe_mapping(_direct_path_known_slots(context)),
        "prompt_known_slots": _direct_path_prompt_known_slots(context),
        "conversation_intent_plan_known_slots": _presale_prompt_safe_mapping(plan.get("known_slots") if isinstance(plan.get("known_slots"), Mapping) else {}),
        "plan_primary_intent": str(plan.get("primary_intent") or ""),
        "plan_answer_topics": [str(item) for item in (plan.get("answer_topics") or ()) if str(item).strip()]
        if isinstance(plan.get("answer_topics"), Sequence) and not isinstance(plan.get("answer_topics"), (str, bytes, bytearray))
        else [],
    }
    if shadow_fn is None:
        metadata["fallback_reason"] = "shadow_fn_missing"
        return metadata
    prompt = build_direct_path_slot_topic_shadow_prompt(client_message, context=context)
    try:
        raw_payload = shadow_fn(prompt)
    except subprocess.TimeoutExpired:
        metadata["fallback_reason"] = "timeout"
        return metadata
    except Exception as exc:  # noqa: BLE001
        metadata.update({"fallback_reason": "runtime_error", "error": str(exc)[:300]})
        return metadata
    try:
        payload = extract_json_object(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
    except Exception as exc:  # noqa: BLE001
        metadata.update({"fallback_reason": "invalid_json", "error": str(exc)[:300]})
        return metadata
    metadata.update(
        {
            "used": True,
            "model_slots": _direct_path_shadow_safe_slots(payload.get("model_slots") or payload.get("slots")),
            "model_topic": _presale_prompt_safe_value(payload.get("model_topic") or payload.get("topic")),
            "evidence_quote": _presale_prompt_safe_value(payload.get("evidence_quote") or payload.get("quote")),
            "confidence": _clamp_float(payload.get("confidence")),
        }
    )
    return metadata

def _direct_path_retriever_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        seq: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        seq = value
    else:
        return ()
    result: list[str] = []
    for item in seq:
        key = str(item or "").strip()
        if key and key not in result:
            result.append(key)
    return tuple(result)

def _direct_path_llm_retrieve_fact_pack(
    records: Sequence[Mapping[str, Any]],
    *,
    legacy: Mapping[str, str],
    active_brand: str,
    context: Optional[Mapping[str, Any]],
    client_message: str,
    max_facts: int,
    max_chars: int,
    retriever_fn: Optional[Callable[[str], Mapping[str, Any] | str]],
) -> tuple[Optional[Mapping[str, Any]], Mapping[str, Any]]:
    need_declaration = _retriever_need_declaration_enabled(context)
    model_driven = _retriever_model_driven_enabled(context)
    keyword_required_fact_keys = _direct_path_required_fact_keys(context)
    candidate_by_key = {
        _direct_path_snapshot_fact_key(fact): fact
        for fact in records
        if _direct_path_snapshot_fact_key(fact)
    }
    metadata: dict[str, Any] = {
        "schema_version": "llm_retrieve_v2_2026_06_15",
        "enabled": True,
        "used": False,
        "fallback": False,
        "fallback_reason": "",
        "mode": _direct_path_retriever_mode(context),
        "need_shadow_enabled": _retriever_need_shadow_enabled(context),
        "model_driven": model_driven,
        "need_declaration_schema_version": RETRIEVER_NEED_DECLARATION_SCHEMA_VERSION if need_declaration else "",
        "keyword_required_fact_keys": list(keyword_required_fact_keys),
        "needed_facts": [],
        "needed_fact_declaration_missing": False,
        "declaration_comparison": _direct_path_declaration_comparison(
            keyword_required_fact_keys=keyword_required_fact_keys,
            needed_facts=(),
        ),
        "candidate_count": len(candidate_by_key),
        "selected_exact_ids": [],
        "selected_adjacent_ids": [],
        "model_selected_exact_ids": [],
        "model_selected_adjacent_ids": [],
        "invalid_ids": [],
        "discarded_ids": [],
        "scope_demoted_ids": [],
        "active_brand": str(active_brand or ""),
    }
    if not candidate_by_key:
        metadata.update({"fallback": True, "fallback_reason": "no_candidates"})
        return None, metadata
    if retriever_fn is None:
        metadata.update({"fallback": True, "fallback_reason": "retriever_fn_missing"})
        return None, metadata
    prompt = build_direct_path_llm_retriever_prompt(client_message, context=context, candidates=records)
    try:
        raw_payload = retriever_fn(prompt)
    except subprocess.TimeoutExpired:
        metadata.update({"fallback": True, "fallback_reason": "timeout"})
        return None, metadata
    except Exception as exc:  # noqa: BLE001
        metadata.update({"fallback": True, "fallback_reason": "runtime_error", "error": str(exc)[:300]})
        return None, metadata
    try:
        payload = extract_json_object(raw_payload) if isinstance(raw_payload, str) else dict(raw_payload)
    except Exception as exc:  # noqa: BLE001
        metadata.update({"fallback": True, "fallback_reason": "invalid_json", "error": str(exc)[:300]})
        return None, metadata
    if need_declaration:
        needed_facts = _direct_path_needed_fact_declaration(payload)
        metadata["needed_facts"] = needed_facts
        metadata["needed_fact_declaration_missing"] = not bool(needed_facts)
        metadata["declaration_comparison"] = _direct_path_declaration_comparison(
            keyword_required_fact_keys=keyword_required_fact_keys,
            needed_facts=needed_facts,
        )
        if model_driven and not needed_facts:
            metadata.update({"fallback": True, "fallback_reason": "missing_needed_facts"})
            return None, metadata
    exact_raw = _direct_path_retriever_ids(payload.get("exact_ids") or payload.get("exact") or payload.get("exact_fact_ids"))
    adjacent_raw = _direct_path_retriever_ids(payload.get("adjacent_ids") or payload.get("adjacent") or payload.get("adjacent_fact_ids"))
    selected_exact: list[str] = []
    selected_adjacent: list[str] = []
    invalid: list[str] = []
    for key in (*exact_raw, *adjacent_raw):
        if key not in candidate_by_key:
            if key not in invalid:
                invalid.append(key)
            continue
        if key in selected_exact or key in selected_adjacent:
            continue
        if key in exact_raw:
            selected_exact.append(key)
        else:
            selected_adjacent.append(key)
    metadata["invalid_ids"] = invalid
    metadata["discarded_ids"] = list(invalid)
    metadata["model_selected_exact_ids"] = list(selected_exact)
    metadata["model_selected_adjacent_ids"] = list(selected_adjacent)
    if not selected_exact and not selected_adjacent:
        metadata.update({"fallback": True, "fallback_reason": "empty_selection"})
        return None, metadata

    slots = _direct_path_slot_scope(context)
    exact_records: list[Mapping[str, Any]] = []
    adjacent_records: list[Mapping[str, Any]] = []
    final_exact_ids: list[str] = []
    final_adjacent_ids: list[str] = []
    scope_demoted_ids: list[str] = []
    for key in selected_exact:
        fact = candidate_by_key[key]
        if _direct_path_fact_conflicts_slots(fact, slots):
            adjacent_records.append(fact)
            scope_demoted_ids.append(key)
            if key not in final_adjacent_ids:
                final_adjacent_ids.append(key)
        else:
            exact_records.append(fact)
            final_exact_ids.append(key)
    for key in selected_adjacent:
        adjacent_records.append(candidate_by_key[key])
        if key not in final_adjacent_ids:
            final_adjacent_ids.append(key)
    supplemented_exact: list[str] = []
    for fact in _direct_path_course_fact_supplements(
        records,
        context=context,
        slots=slots,
        existing_keys=set(selected_exact),
    ):
        key = _direct_path_snapshot_fact_key(fact)
        if not key or key in selected_exact:
            continue
        if key in final_adjacent_ids:
            final_adjacent_ids.remove(key)
            adjacent_records = [item for item in adjacent_records if _direct_path_snapshot_fact_key(item) != key]
        if key not in final_exact_ids:
            final_exact_ids.append(key)
        exact_records.append(fact)
        supplemented_exact.append(key)
    metadata.update(
        {
            "used": True,
            "selected_exact_ids": list(final_exact_ids),
            "selected_adjacent_ids": list(final_adjacent_ids),
            "scope_demoted_ids": scope_demoted_ids,
            "supplemented_exact_ids": supplemented_exact,
        }
    )
    pack = _direct_path_records_to_fact_pack(
        active_brand=active_brand,
        legacy=legacy,
        exact_records=exact_records,
        adjacent_records=adjacent_records,
        selected_category="llm_retrieve",
        max_facts=max_facts,
        max_chars=max_chars,
        extra_metadata={"llm_retrieve": metadata},
    )
    return pack, metadata

def _direct_path_wide_fact_pack(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    max_facts: int = DIRECT_PATH_WIDE_FACT_LIMIT,
    max_chars: int = DIRECT_PATH_WIDE_FACT_CHAR_LIMIT,
    retriever_fn: Optional[Callable[[str], Mapping[str, Any] | str]] = None,
) -> Mapping[str, Any]:
    legacy = _direct_path_legacy_context_fact_items(context, limit=18)
    active_brand = _active_brand(context)
    snapshot_path = _direct_path_snapshot_path_from_context(context)
    snapshot = _direct_path_load_snapshot(snapshot_path)
    records = [
        fact
        for fact in _direct_path_snapshot_facts(snapshot)
        if _direct_path_client_safe_snapshot_fact(fact, active_brand=active_brand)
    ]
    if not records:
        return {
            "schema_version": DIRECT_PATH_WIDE_FACT_PACK_SCHEMA_VERSION,
            "facts": legacy,
            "exact_keys": list(legacy.keys()),
            "adjacent_keys": [],
            "selected_category": "legacy_context",
            "fact_metadata": {key: {"brand": active_brand, "fact_type": "", "product": ""} for key in legacy},
        }

    llm_retrieve_metadata: Optional[Mapping[str, Any]] = None
    if _llm_retrieve_enabled(context):
        llm_pack, llm_retrieve_metadata = _direct_path_llm_retrieve_fact_pack(
            records,
            legacy=legacy,
            active_brand=active_brand,
            context=context,
            client_message=client_message,
            max_facts=max_facts,
            max_chars=max_chars,
            retriever_fn=retriever_fn,
        )
        if llm_pack is not None:
            return llm_pack

    return _direct_path_keyword_fact_pack_from_records(
        records,
        legacy=legacy,
        active_brand=active_brand,
        context=context,
        client_message=client_message,
        max_facts=max_facts,
        max_chars=max_chars,
        extra_metadata={"llm_retrieve": llm_retrieve_metadata} if llm_retrieve_metadata is not None else None,
    )

def _direct_path_context_fact_pack(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    limit: int = DIRECT_PATH_WIDE_FACT_LIMIT,
    retriever_fn: Optional[Callable[[str], Mapping[str, Any] | str]] = None,
) -> Mapping[str, Any]:
    pack = _direct_path_wide_fact_pack(context, client_message=client_message, max_facts=limit, retriever_fn=retriever_fn)
    facts = pack.get("facts")
    if not isinstance(facts, Mapping):
        return {
            "schema_version": DIRECT_PATH_WIDE_FACT_PACK_SCHEMA_VERSION,
            "facts": {},
            "exact_keys": [],
            "adjacent_keys": [],
            "selected_category": "empty",
            "fact_metadata": {},
        }
    return pack

def _direct_path_recent_messages(context: Optional[Mapping[str, Any]], *, limit: int = 8) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    value = context.get("recent_messages")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    items = tuple(str(item or "").strip() for item in value if str(item or "").strip())
    client_identity = context.get("client_identity") if isinstance(context.get("client_identity"), Mapping) else {}
    channel = str(client_identity.get("channel") if isinstance(client_identity, Mapping) else "").strip().casefold()
    if not channel.startswith("wappi_"):
        return items[-limit:]
    raw_limit = max(WAPPI_DIRECT_PATH_RAW_HISTORY_LIMIT, int(limit or 0))
    older_summary = tuple(item for item in items if item.startswith(WAPPI_OLDER_DIALOGUE_SUMMARY_PREFIX))[:1]
    raw_history = tuple(item for item in items if not item.startswith(WAPPI_OLDER_DIALOGUE_SUMMARY_PREFIX))[-raw_limit:]
    return (*older_summary, *raw_history)

def _direct_path_known_slots(context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(context, Mapping):
        return result
    for key in ("known_slots", "known_dialog_fields"):
        value = context.get(key)
        if isinstance(value, Mapping):
            result.update({str(k): v for k, v in value.items() if str(k).strip() and str(v).strip()})
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        slots = memory.get("known_slots")
        if isinstance(slots, Mapping):
            result.update({str(k): v for k, v in slots.items() if str(k).strip() and str(v).strip()})
    plan = context.get("conversation_intent_plan")
    if isinstance(plan, Mapping):
        if _direct_plan_known_slots_enabled(context):
            slots = plan.get("known_slots")
            if not isinstance(slots, Mapping):
                slots = plan.get("slots")
        else:
            slots = plan.get("slots")
        if isinstance(slots, Mapping):
            result.update({str(k): v for k, v in slots.items() if str(k).strip() and str(v).strip()})
    return result

PRESALE_PROMPT_SAFE_SLOT_KEYS = frozenset(
    {
        "active_brand",
        "brand",
        "campus",
        "city",
        "class",
        "course",
        "exam",
        "format",
        "grade",
        "intent",
        "level",
        "learning_goal",
        "message_type",
        "modality",
        "platform",
        "primary_intent",
        "product",
        "schedule",
        "subject",
        "topic",
        "topic_focus",
        "topic_id",
        "training_format",
    }
)

PRESALE_PROMPT_SENSITIVE_KEY_RE = re.compile(
    r"(?:phone|телефон|contact|контакт|email|mail|почт|name|имя|фио|fio|identity|client|parent|mother|father|мам|пап|родител|реб[её]н|child|student|ученик)",
    re.I,
)

PRESALE_PROMPT_CHILD_NAME_KEY_RE = re.compile(r"(?:child|student|реб[её]н|ученик|доч|сын)", re.I)

PRESALE_PROMPT_PARENT_NAME_KEY_RE = re.compile(r"(?:client|parent|mother|father|мам|пап|родител)", re.I)

def _presale_prompt_safe_key(key: object) -> bool:
    normalized = str(key or "").strip().casefold()
    if not normalized or PRESALE_PROMPT_SENSITIVE_KEY_RE.search(normalized):
        return False
    return normalized in PRESALE_PROMPT_SAFE_SLOT_KEYS

def _presale_prompt_safe_slot_value(key: object, value: Any) -> Any:
    key_text = str(key or "")
    if PRESALE_PROMPT_CHILD_NAME_KEY_RE.search(key_text):
        return _presale_prompt_child_name_value(value)
    if PRESALE_PROMPT_PARENT_NAME_KEY_RE.search(key_text):
        return ""
    if PRESALE_PROMPT_SENSITIVE_KEY_RE.search(key_text):
        return ""
    return _presale_prompt_safe_value(value)

def _presale_prompt_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    filtered = {
        str(key): _presale_prompt_safe_slot_value(key, item)
        for key, item in value.items()
        if str(key or "").strip() and str(item or "").strip()
    }
    return {key: item for key, item in filtered.items() if item not in ("", {}, [])}

def _presale_prompt_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _presale_prompt_safe_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [_presale_prompt_safe_value(item) for item in value[:8]]
        return [item for item in items if item not in ("", {}, [])]
    text = " ".join(str(value or "").split())
    if not text or _A2_PHONE_RE.search(text) or _CLIENT_EMAIL_RE.search(text):
        return ""
    return text[:220]

def _direct_path_prompt_known_slots(context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if _assumed_scope_guard_enabled(context):
        result: dict[str, Any] = {}
        for key, data in _direct_path_slot_provenance(context).items():
            if key not in _ASSUMED_SCOPE_KEYS or not _presale_prompt_safe_key(key):
                continue
            safe_value = _presale_prompt_safe_slot_value(key, data.get("value"))
            if safe_value in ("", {}, []):
                continue
            result[key] = {
                "value": safe_value,
                "status": str(data.get("status") or "assumed_from_context"),
            }
        return result
    slots = _direct_path_known_slots(context)
    if not (
        _presale_safety_enabled(context, subflag=PRESALE_PII_MEMORY_ENV)
        or _direct_path_known_slots_next_step_prompt_enabled(context)
    ):
        return slots
    return _presale_prompt_safe_mapping(slots)

def _direct_path_prompt_memory_view(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping) or not isinstance(context.get("dialogue_memory_view"), Mapping):
        return {}
    memory = context["dialogue_memory_view"]
    if not _presale_safety_enabled(context, subflag=PRESALE_PII_MEMORY_ENV):
        return memory
    result: dict[str, Any] = {}
    for key in ("known_slots", "client_confirmed_slots", "crm_known_slots", "topic_focus"):
        value = memory.get(key)
        if isinstance(value, Mapping):
            filtered = _presale_prompt_safe_mapping(value)
            if filtered:
                result[key] = filtered
    for key in ("topic", "topic_id", "primary_intent", "message_type"):
        value = _presale_prompt_safe_value(memory.get(key))
        if value:
            result[key] = value
    return result

def _presale_prompt_safe_dialogue_text(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    value = _replace_echoed_phone(value, _a2_extract_phone(value)) if _a2_extract_phone(value) else value
    value = _CLIENT_EMAIL_RE.sub("[данные у менеджера]", value)
    value = _PARTIAL_PHONE_CONTEXT_RE.sub(lambda m: f"{m.group('label')} [данные у менеджера]", value)
    value = _CLIENT_CHILD_IDENTITY_PROMPT_RE.sub(
        lambda m: f"{m.group('prefix')}{_presale_prompt_child_name_value(m.group('name')) or '[данные у менеджера]'}",
        value,
    )
    value = _CLIENT_PARENT_IDENTITY_PROMPT_RE.sub(lambda m: f"{m.group('prefix')}[данные у менеджера]", value)
    return _normalize_output_sanitizer_text(value)

_DIRECT_PATH_QUALIFICATION_SLOT_LABELS: Mapping[str, str] = {
    "class": "класс",
    "course_subject": "предмет",
    "format": "формат",
    "grade": "класс",
    "learning_goal": "цель",
    "level": "уровень",
    "modality": "формат",
    "product": "продукт",
    "product_family": "продукт",
    "subject": "предмет",
    "training_format": "формат",
}

_DIRECT_PATH_QUALIFICATION_SLOT_CANONICAL: Mapping[str, str] = {
    "class": "grade",
    "course_subject": "subject",
    "modality": "format",
    "training_format": "format",
}

_DIRECT_PATH_QUALIFICATION_SLOTS = frozenset({"grade", "subject", "format"})
_DIRECT_PATH_QUESTIONNAIRE_GOLD_TOPICS = frozenset({"course_pick"})

def _direct_path_canonical_slot_key(key: object) -> str:
    normalized = str(key or "").strip().casefold()
    return _DIRECT_PATH_QUALIFICATION_SLOT_CANONICAL.get(normalized, normalized)


def _direct_path_slot_label(key: str) -> str:
    return _DIRECT_PATH_QUALIFICATION_SLOT_LABELS.get(key, key)


def _direct_path_safe_slot_value_for_instruction(key: object, value: Any) -> str:
    if isinstance(value, Mapping) and "value" in value:
        value = value.get("value")
    safe_value = _presale_prompt_safe_slot_value(key, value)
    if isinstance(safe_value, (Mapping, list, tuple, set)):
        return ""
    return " ".join(str(safe_value or "").split()).strip()


def _direct_path_merge_instruction_slots(target: dict[str, tuple[str, str]], source: Any) -> None:
    if not isinstance(source, Mapping):
        return
    for raw_key, raw_value in source.items():
        key = _direct_path_canonical_slot_key(raw_key)
        if not key or not _presale_prompt_safe_key(key):
            continue
        value = _direct_path_safe_slot_value_for_instruction(key, raw_value)
        if not value:
            continue
        target[key] = (_direct_path_slot_label(key), value)


def _direct_path_prompt_instruction_slot_map(context: Optional[Mapping[str, Any]]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    _direct_path_merge_instruction_slots(result, _direct_path_prompt_known_slots(context))
    if not isinstance(context, Mapping):
        return result
    containers: list[Any] = [
        context.get("conversation_intent_plan"),
        context.get("planner_intent"),
        context.get("answer_contract"),
        context.get("dialogue_memory_view"),
    ]
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        _direct_path_merge_instruction_slots(result, container.get("known_slots"))
        _direct_path_merge_instruction_slots(result, container.get("slots"))
    return result


def _direct_path_prompt_do_not_reask_keys(context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    keys: list[str] = []
    for container in (
        context,
        context.get("conversation_intent_plan"),
        context.get("planner_intent"),
        context.get("answer_contract"),
        context.get("dialogue_memory_view"),
    ):
        if not isinstance(container, Mapping):
            continue
        raw = container.get("do_not_reask_slots") or container.get("do_not_ask_again")
        if isinstance(raw, str):
            keys.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
            keys.extend(str(item or "") for item in raw)
    result: list[str] = []
    for key in keys:
        canonical = _direct_path_canonical_slot_key(key)
        if canonical and _presale_prompt_safe_key(canonical) and canonical not in result:
            result.append(canonical)
    return tuple(result)


def _direct_path_known_slots_instruction_line(context: Optional[Mapping[str, Any]]) -> str:
    slots = _direct_path_prompt_instruction_slot_map(context)
    do_not_reask = _direct_path_prompt_do_not_reask_keys(context)
    for key in do_not_reask:
        slots.setdefault(key, (_direct_path_slot_label(key), ""))
    if not slots:
        return ""
    ordered_keys = [key for key in ("grade", "subject", "format", "learning_goal", "level", "product", "product_family") if key in slots]
    ordered_keys.extend(key for key in sorted(slots) if key not in ordered_keys)
    parts = []
    for key in ordered_keys:
        label, value = slots[key]
        parts.append(f"{label}: {value}" if value else label)
    return "эти параметры клиент уже назвал — НЕ переспрашивай: " + "; ".join(parts) + "."


def _direct_path_has_known_qualification_slot(context: Optional[Mapping[str, Any]]) -> bool:
    slot_keys = set(_direct_path_prompt_instruction_slot_map(context))
    slot_keys.update(_direct_path_prompt_do_not_reask_keys(context))
    return bool(slot_keys & _DIRECT_PATH_QUALIFICATION_SLOTS)


def _direct_path_context_next_step_statuses(context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    statuses: list[str] = []

    def add(value: Any) -> None:
        status = str(value or "").strip().casefold()
        if status in {"active", "needs_manager_review", "empty", "closed"} and status not in statuses:
            statuses.append(status)

    for item in _direct_path_bot_safe_context_items(context):
        add(item.get("next_step_status"))
    if isinstance(context, Mapping):
        for container in (
            context,
            context.get("timeline_context"),
            context.get("read_only_customer_context"),
        ):
            if not isinstance(container, Mapping):
                continue
            add(container.get("next_step_status"))
            raw_next_step = container.get("next_step_resolution") or container.get("next_step")
            if isinstance(raw_next_step, Mapping):
                add(raw_next_step.get("status"))
    return tuple(statuses)


def _direct_path_has_active_next_step(context: Optional[Mapping[str, Any]]) -> bool:
    return "active" in _direct_path_context_next_step_statuses(context)


def _direct_path_suppress_questionnaire_gold(context: Optional[Mapping[str, Any]]) -> bool:
    return _direct_path_known_slots_next_step_prompt_enabled(context) and (
        _direct_path_has_active_next_step(context) or _direct_path_has_known_qualification_slot(context)
    )


def _direct_path_known_slots_next_step_prompt_block(context: Optional[Mapping[str, Any]]) -> str:
    if not _direct_path_known_slots_next_step_prompt_enabled(context):
        return ""
    lines = ["Приоритет уже известного контекста:"]
    known_line = _direct_path_known_slots_instruction_line(context)
    if known_line:
        lines.append(f"- {known_line}")
    lines.append(
        "- Вопрос про класс/предмет/формат задавай ТОЛЬКО если он реально неизвестен И нет активного следующего шага. "
        "Если параметр уже известен, анкета — ошибка: продвигай разговор по сути."
    )
    if _direct_path_has_active_next_step(context):
        lines.append(
            "- Если статус next_step active — ответ ДОЛЖЕН продвигать шаг ИЛИ прямо отвечать на вопрос клиента; "
            "НЕ задавай квалифицирующих вопросов, если шаг известен."
        )
    else:
        lines.append(
            "- Если класс/предмет/формат действительно неизвестны и без них нельзя помочь, допустим один короткий уточняющий вопрос."
        )
    return "\n".join(lines)


def _direct_path_known_slots_next_step_prompt_trace(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not _direct_path_known_slots_next_step_prompt_enabled(context):
        return {"enabled": False}
    slots = _direct_path_prompt_instruction_slot_map(context)
    return {
        "enabled": True,
        "known_slot_keys": sorted(slots),
        "do_not_reask_slots": list(_direct_path_prompt_do_not_reask_keys(context)),
        "active_next_step": _direct_path_has_active_next_step(context),
        "next_step_statuses": list(_direct_path_context_next_step_statuses(context)),
        "questionnaire_gold_suppressed": _direct_path_suppress_questionnaire_gold(context),
    }

DIRECT_PATH_GOLD_TOPIC_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "camp": ("лагер", "лш", "лвш", "смен", "летн"),
    "close": ("спасибо", "подума", "понятно", "вернем", "вернём"),
    "course_pick": ("курс", "заняти", "групп", "подготов", "услов"),
    "docs": ("договор", "документ", "справк"),
    "enrollment": ("запис", "брон", "оформ"),
    "format": ("онлайн", "очно", "формат", "платформ", "программирован"),
    "join_mid": ("присоедин", "войти", "середин", "идет", "идёт"),
    "payment_flex": ("част", "доплат", "внес", "остаток", "сегодня"),
    "price": ("стоим", "цен", "рассроч", "оплат", "дорог"),
    "value": ("школ", "институт", "ценност", "уров", "польз"),
}

def _direct_path_gold_real_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (BOT_GOLD_REAL_ENV, "bot_gold_real", "direct_path_gold_real"):
            if key in context:
                return _truthy_value(context.get(key))
    if BOT_GOLD_REAL_ENV in os.environ:
        return _truthy_value(os.getenv(BOT_GOLD_REAL_ENV))
    return _pilot_gold_profile_enabled(context)

def _direct_path_gold_pack_path() -> Path:
    override = os.getenv(BOT_GOLD_REAL_PACK_ENV)
    if override:
        return Path(override).expanduser()
    return DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH

def _direct_path_gold_pack_version() -> str:
    override = os.getenv(BOT_GOLD_REAL_PACK_ENV)
    if override:
        return Path(override).expanduser().stem
    return DIRECT_PATH_REAL_MANAGER_GOLD_PACK_VERSION

def _load_direct_path_gold_real_examples(path: Optional[Path] = None) -> tuple[Mapping[str, Any], ...]:
    pack_path = path or _direct_path_gold_pack_path()
    if not pack_path.exists():
        return ()
    payload = yaml.safe_load(pack_path.read_text(encoding="utf-8")) or {}
    examples = payload.get("examples") if isinstance(payload, Mapping) else None
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes, bytearray)):
        return ()
    result: list[Mapping[str, Any]] = []
    for item in examples:
        if not isinstance(item, Mapping):
            continue
        if not _truthy_value(item.get("mission_gold")):
            continue
        result.append(dict(item))
    return tuple(result)

def _direct_path_topic_hints(client_message: str, context: Optional[Mapping[str, Any]]) -> set[str]:
    hints: set[str] = set()
    if isinstance(context, Mapping):
        for container_key in ("conversation_intent_plan", "planner_intent", "dialogue_contract_pipeline"):
            container = context.get(container_key)
            if not isinstance(container, Mapping):
                continue
            for key in ("primary_intent", "planner_intent", "intent", "topic", "subvariant"):
                value = str(container.get(key) or "").strip().casefold()
                if value:
                    hints.add(value)
            required = container.get("required_fact_keys")
            if isinstance(required, Sequence) and not isinstance(required, (str, bytes, bytearray)):
                hints.update(str(item or "").casefold() for item in required if str(item or "").strip())
    lowered = str(client_message or "").casefold()
    for topic, markers in DIRECT_PATH_GOLD_TOPIC_KEYWORDS.items():
        if any(marker in lowered for marker in markers):
            hints.add(topic)
    return hints

def _direct_path_select_gold_real_examples(
    client_message: str,
    *,
    context: Optional[Mapping[str, Any]],
    active_brand: str,
    limit: int = 4,
) -> tuple[Mapping[str, Any], ...]:
    if not _direct_path_gold_real_enabled(context):
        return ()
    brand = str(active_brand or "").strip().casefold()
    examples = [item for item in _load_direct_path_gold_real_examples() if str(item.get("brand") or "").casefold() == brand]
    if not examples:
        return ()
    if _direct_path_suppress_questionnaire_gold(context):
        examples = [
            item
            for item in examples
            if str(item.get("topic") or "").strip().casefold() not in _DIRECT_PATH_QUESTIONNAIRE_GOLD_TOPICS
        ]
        if not examples:
            return ()
    hints = _direct_path_topic_hints(client_message, context)
    scored: list[tuple[int, str, Mapping[str, Any]]] = []
    for item in examples:
        topic = str(item.get("topic") or "").strip().casefold()
        score = 2 if topic and topic in hints else 0
        if topic == "course_pick" and any(hint in {"pricing", "schedule", "teacher"} for hint in hints):
            score = max(score, 1)
        scored.append((score, str(item.get("id") or ""), item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    selected = [item for score, _, item in scored if score > 0][:limit]
    if not selected:
        selected = [item for _, _, item in scored[:2]]
    elif len(selected) < min(2, limit):
        selected_ids = {str(item.get("id") or "") for item in selected}
        for _, _, item in scored:
            if str(item.get("id") or "") in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= min(2, limit):
                break
    return tuple(selected)

def _direct_path_gold_prompt_block(examples: Sequence[Mapping[str, Any]]) -> str:
    if not examples:
        return ""
    lines = [
        "Живые образцы менеджерского стиля. Это НЕ источник фактов: маски в квадратных скобках заменяй только подтверждёнными фактами текущего хода или опускай.",
    ]
    for idx, item in enumerate(examples, 1):
        client = str(item.get("client") or "").strip()
        answer = str(item.get("manager_response_masked") or "").strip()
        note = str(item.get("prompt_example") or "").strip()
        if not client or not answer:
            continue
        lines.append(f"{idx}. Тема: {item.get('topic')}.")
        lines.append(f"   Клиент: {client}")
        lines.append(f"   Хороший стиль: {answer}")
        if note:
            lines.append(f"   Принцип: {note}")
    return "\n".join(lines)

def _build_direct_path_prompt(
    client_message: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
    facts: Optional[Mapping[str, str]] = None,
    fact_pack: Optional[Mapping[str, Any]] = None,
    gold_examples: Sequence[Mapping[str, Any]] = (),
) -> str:
    active_brand = _active_brand(context)
    brand_label = _direct_path_brand_label(active_brand)
    pack = fact_pack if isinstance(fact_pack, Mapping) else _direct_path_context_fact_pack(context, client_message=client_message)
    fact_items = dict(facts or pack.get("facts") or {})
    fact_metadata = pack.get("fact_metadata") if isinstance(pack.get("fact_metadata"), Mapping) else {}
    exact_keys = [str(key) for key in (pack.get("exact_keys") or fact_items.keys()) if str(key).strip()]
    adjacent_keys = [str(key) for key in (pack.get("adjacent_keys") or ()) if str(key).strip()]
    exact_block = _direct_path_render_fact_block(fact_items, fact_metadata=fact_metadata, keys=exact_keys)
    adjacent_block = _direct_path_render_fact_block(fact_items, fact_metadata=fact_metadata, keys=adjacent_keys)
    gold_block = _direct_path_gold_prompt_block(gold_examples)
    recent_messages = _direct_path_recent_messages(context)
    if _presale_safety_enabled(context, subflag=PRESALE_PII_MEMORY_ENV):
        recent_messages = tuple(
            item
            for item in (_presale_prompt_safe_dialogue_text(message) for message in recent_messages)
            if item
        )
    recent_block = "\n".join(recent_messages) or "(диалог только начался)"
    prompt_client_message = (
        _presale_prompt_safe_dialogue_text(client_message)
        if _presale_safety_enabled(context, subflag=PRESALE_PII_MEMORY_ENV)
        else client_message
    )
    slots = _direct_path_prompt_known_slots(context)
    slots_block = json.dumps(slots, ensure_ascii=False, indent=2) if slots else "{}"
    memory = _direct_path_prompt_memory_view(context)
    memory_block = json.dumps(memory, ensure_ascii=False, indent=2)[:2400] if memory else "{}"
    known_slots_next_step_block = _direct_path_known_slots_next_step_prompt_block(context)
    bot_safe_context_block = _direct_path_bot_safe_context_prompt_block(context)
    action_proposal_instruction = ""
    action_proposal_field = ""
    p0_instruction = ""
    p0_fields = ""
    intent_instruction = ""
    intent_field = ""
    assumed_scope_instruction = ""
    route_choices = '"bot_answer_self_for_pilot" | "draft_for_manager"'
    if _direct_path_model_p0_enabled(context):
        route_choices = '"bot_answer_self_for_pilot" | "draft_for_manager" | "manager_only"'
        p0_instruction = (
            "Срочные обращения/P0: если клиент пишет про спорную оплату, списание/платёж, возврат, жалобу, "
            "юридическую угрозу, претензию или конфликтную ситуацию, поставь is_p0=true, risk_level=\"high\", "
            "route=\"manager_only\". В p0_kind выбери одно: payment_dispute, refund, complaint, legal_threat. "
            "Модель может только добавить срочность; если это обычное возражение «дорого/подумаю» или "
            "гипотетический вопрос про правила возврата без претензии, is_p0=false.\n\n"
        )
        if _p0_model_led_enabled(context):
            p0_instruction += (
                "Для p0_kind=complaint отличай реальную жалобу от растерянности. Реальная жалоба/претензия: "
                "клиент недоволен действиями школы или сотрудника, пишет «жалоба», «безобразие», "
                "«накричали/унизили/оскорбили ребёнка», «ребёнок один остался», «напишу везде какие вы» — "
                "тогда is_p0=true, p0_kind=\"complaint\", route=\"manager_only\". "
                "Растерянность, уточнение порядка или тревога без претензии — «не понимаю», «как дальше», "
                "«ребёнок в 6 классе», «сначала тест или группа», «вдруг не потянет» — это НЕ complaint: "
                "ставь is_p0=false и отвечай полезно по фактам.\n\n"
            )
        if _direct_p0_text_hygiene_enabled(context):
            p0_instruction += (
                "P0-гигиена текста: если тема возврата, претензии, договора или спорной оплаты — "
                "в draft_text не обещай исход возврата, не называй сумму, не продавай и не подталкивай к оплате. "
                "Дай короткую нейтральную оговорку и передай вопрос менеджеру.\n\n"
            )
        p0_fields = (
            '  "is_p0": false,\n'
            '  "risk_level": "low|high",\n'
            '  "p0_kind": "none|payment_dispute|refund|complaint|legal_threat",\n'
            '  "model_reason": "кратко, почему это P0 или почему нет",\n'
        )
    if _intent_model_led_enabled(context):
        intent_instruction = (
            "Смысловой intent_model_led: отдельные слова клиента — только сигналы. "
            "Классифицируй реальный смысл текущей реплики в поле model_intent. "
            "primary_intent выбери из: live_availability, schedule, address, camp, price_fix, other. "
            "live_availability ставь только для настоящего вопроса о наличии мест/броней/свободной группе; "
            "«место» как территория/площадка/место занятий, «привезу на место», «в одном месте» — это НЕ live_availability. "
            "schedule ставь только для вопроса о расписании/времени занятий; «когда привезу/подъеду» — other. "
            "address ставь только для вопроса о локации/адресе/площадке; бытовое «где-то/негде/живём рядом» — other. "
            "camp ставь только если вопрос реально про лагерь/ЛВШ/смену как продукт; бытовое «смена настроения/где живём» — other. "
            "price_fix ставь только если клиент хочет зафиксировать цену/условия; «закрепить материал/навык» — other. "
            "sense кратко укажи смысл: seats, venue, schedule, address, camp_product, price_terms, learning, logistics, other. "
            "confidence — число 0..1.\n\n"
        )
        intent_field = (
            '  "model_intent": {"primary_intent": "live_availability|schedule|address|camp|price_fix|other", "scope": "", "sense": "", "confidence": 0.0, "reason": "кратко"},\n'
        )
    if _deal_action_decision_enabled(context):
        action_proposal_instruction = (
            "Предложи одно следующее действие для менеджера в поле action_proposal из закрытого списка: "
            "answer_only, send_schedule, send_materials, send_crm_data, capture_lead, schedule_followup, "
            "send_payment_link, send_document, advance_stage, handoff_manager, unknown. "
            "Это только предложение: не исполняй действие и не обещай его клиенту. Если не уверен — unknown.\n\n"
        )
        action_proposal_field = (
            '  "action_proposal": {"action": "answer_only|send_schedule|send_materials|send_crm_data|capture_lead|schedule_followup|send_payment_link|send_document|advance_stage|handoff_manager|unknown", "confidence": 0.0, "reason": "кратко"},\n'
        )
    if _assumed_scope_guard_enabled(context):
        assumed_scope_instruction = (
            "Правило неподтверждённых параметров: в «Известных слотах» status=confirmed_by_client означает, "
            "что клиент сам подтвердил параметр в диалоге. status=assumed_from_context означает CRM/контекстную "
            "догадку. Не представляй такие класс, предмет, формат или продукт как подтверждённые клиентом. "
            "Не называй итоговые цены, даты или расписание, если число зависит только от assumed_from_context. "
            "В такой ситуации мягко задай один уточняющий вопрос или ответь без привязки к неподтверждённому параметру.\n\n"
        )
    return (
        f"{_direct_path_mission_text(brand_label=brand_label, context=context)}\n\n"
        f"{_direct_path_prose_model_led_block(context)}"
        f"{_direct_path_route_rubric_block(context)}"
        "Дополнение к числам: каждую цену, дату, процент, длительность и количество называй вместе с форматом,\n"
        "классом или продуктом того факта, из которого взял число. Если скоуп факта не совпадает с вопросом — не называй число.\n\n"
        f"{p0_instruction}"
        f"{intent_instruction}"
        f"{action_proposal_instruction}"
        f"{assumed_scope_instruction}"
        f"Активный бренд: {brand_label} ({active_brand}).\n"
        f"Текущее сообщение клиента:\n{prompt_client_message}\n\n"
        + (f"{known_slots_next_step_block}\n\n" if known_slots_next_step_block else "")
        + (f"{gold_block}\n\n" if gold_block else "")
        +
        "Факты по вашему вопросу:\n"
        f"{exact_block}\n\n"
        "Смежные факты — используй только если вопрос реально про это:\n"
        f"{adjacent_block}\n\n"
        "Память диалога:\n"
        f"{memory_block}\n\n"
        + (f"{bot_safe_context_block}\n\n" if bot_safe_context_block else "")
        +
        "Известные слоты:\n"
        f"{slots_block}\n\n"
        "Последние реплики:\n"
        f"{recent_block}\n\n"
        "Верни только JSON без Markdown и без комментариев:\n"
        "{\n"
        f'  "route": {route_choices},\n'
        '  "draft_text": "текст для клиента",\n'
        f"{p0_fields}"
        f"{intent_field}"
        f"{action_proposal_field}"
        '  "manager_checklist": [],\n'
        '  "missing_facts": [],\n'
        '  "context_used": []\n'
        "}\n"
    )

def _direct_path_metadata(
    *,
    attempted: bool,
    model_called: bool,
    facts: Mapping[str, str],
    fact_pack: Optional[Mapping[str, Any]] = None,
    gold_examples: Sequence[Mapping[str, Any]] = (),
    preblocked: bool = False,
    preblock_reason: str = "",
    reason_class: str = "",
    reason_evidence: Optional[Mapping[str, Any]] = None,
    pilot_config: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    gold_ids = [str(item.get("id") or "").strip() for item in gold_examples if str(item.get("id") or "").strip()]
    pack = fact_pack if isinstance(fact_pack, Mapping) else {}
    fact_meta = pack.get("fact_metadata") if isinstance(pack.get("fact_metadata"), Mapping) else {}
    exact_keys = [str(key) for key in (pack.get("exact_keys") or ()) if str(key).strip()]
    adjacent_keys = [str(key) for key in (pack.get("adjacent_keys") or ()) if str(key).strip()]
    metadata = {
        "schema_version": DIRECT_PATH_SCHEMA_VERSION,
        "enabled": True,
        "pilot_config": str(pilot_config or ""),
        "pilot_config_version": DIRECT_PATH_PILOT_CONFIG_VERSION if str(pilot_config or "") == DIRECT_PATH_PILOT_CONFIG_VERSION else "",
        "pilot_profile_overrides": _pilot_profile_overrides(context),
        "attempted": bool(attempted),
        "model_called": bool(model_called),
        "preblocked": bool(preblocked),
        "preblock_reason": str(preblock_reason or ""),
        "retrieved_fact_keys": list(facts.keys()),
        "retrieved_facts": dict(facts),
        "wide_facts_count": len(facts),
        "wide_fact_keys": list(facts.keys()),
        "selected_category": str(pack.get("selected_category") or ""),
        "wide_fact_exact_keys": exact_keys,
        "wide_fact_adjacent_keys": adjacent_keys,
        "wide_fact_metadata": {str(key): dict(value) for key, value in fact_meta.items() if str(key).strip() and isinstance(value, Mapping)},
        "gold_real_enabled": bool(gold_ids),
        "gold_pack_version": _direct_path_gold_pack_version() if gold_ids else "",
        "gold_real_example_ids": gold_ids,
        "text_composition_source": "direct_path_model" if model_called else "deterministic_preblock",
        "direct_path_attempted": bool(attempted),
        "direct_path_downgraded": False,
        "direct_path_regenerated": False,
        "rubric_enabled": _route_rubric_enabled(context),
        "rubric_regenerated": False,
        "rubric_reason": "",
        "known_slots_next_step_prompt": dict(_direct_path_known_slots_next_step_prompt_trace(context)),
        "bot_safe_crm_context": dict(_direct_path_bot_safe_context_trace(context)),
        "reason_class": str(reason_class or ""),
        "reason_evidence": dict(reason_evidence or {}),
        "is_manager_deferral": bool(reason_class),
    }
    if _template_from_kb_enabled(context) and facts:
        metadata["template_from_kb_trace"] = [
            {
                "fact_key": "direct_path.wide_fact_pack",
                "outcome": "hit",
                "selected_category": str(pack.get("selected_category") or ""),
                "fact_count": len(facts),
                "exact_keys": exact_keys[:20],
            }
        ]
    if isinstance(pack.get("llm_retrieve"), Mapping):
        metadata["llm_retrieve"] = dict(pack["llm_retrieve"])  # type: ignore[index]
    if isinstance(pack.get("slot_topic_shadow"), Mapping):
        metadata["slot_topic_shadow"] = dict(pack["slot_topic_shadow"])  # type: ignore[index]
    if _assumed_scope_guard_enabled(context):
        metadata["assumed_scope_guard"] = {
            "enabled": True,
            "slot_provenance": {
                key: {
                    "value": str(data.get("value") or ""),
                    "status": str(data.get("status") or ""),
                    "source": str(data.get("source") or ""),
                    "confirmed": bool(data.get("confirmed")),
                }
                for key, data in _direct_path_slot_provenance(context).items()
                if key in _ASSUMED_SCOPE_KEYS
            },
            "confirmed_slot_scope": dict(_direct_path_confirmed_slot_scope(context)),
            "soft_slot_scope": dict(_direct_path_soft_slot_scope(context)),
        }
    return metadata

def _direct_path_merge_metadata(result: SubscriptionDraftResult, direct_meta: Mapping[str, Any]) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    metadata["direct_path"] = dict(direct_meta)
    metadata["text_composition_source"] = direct_meta.get("text_composition_source") or "direct_path_model"
    if direct_meta.get("reason_class"):
        metadata["reason_class"] = str(direct_meta.get("reason_class") or "")
        metadata["is_manager_deferral"] = bool(direct_meta.get("is_manager_deferral"))
    return replace(result, metadata=metadata)


def _direct_path_assumed_scope_p0_active(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
) -> bool:
    if str(result.risk_level or "").strip().casefold() in {"high", "p0", "critical", "high_risk"}:
        return True
    if any(re.search(r"p0|payment_dispute|refund|complaint|legal|high_risk", flag, re.I) for flag in result.safety_flags):
        return True
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if isinstance(metadata.get("direct_path_model_p0"), Mapping):
        return True
    if isinstance(context, Mapping):
        memory = context.get("dialogue_memory_view") if isinstance(context.get("dialogue_memory_view"), Mapping) else context
        latch = memory.get("p0_latch") if isinstance(memory, Mapping) and isinstance(memory.get("p0_latch"), Mapping) else {}
        if latch and (latch.get("active") or latch.get("had_hard_p0_claim")):
            return True
        risk_flags = memory.get("risk_flags") if isinstance(memory, Mapping) else ()
        if isinstance(risk_flags, Sequence) and not isinstance(risk_flags, (str, bytes, bytearray)):
            return any(re.search(r"p0|payment_dispute|refund|complaint|legal|high_risk", str(flag), re.I) for flag in risk_flags)
    return False


def _direct_path_do_not_reask_slots(context: Optional[Mapping[str, Any]]) -> set[str]:
    if not isinstance(context, Mapping):
        return set()
    values: list[Any] = []
    for container in (context, context.get("dialogue_memory_view")):
        if not isinstance(container, Mapping):
            continue
        raw = container.get("do_not_reask_slots")
        if isinstance(raw, str):
            values.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
            values.extend(raw)
    result = {str(item or "").strip() for item in values if str(item or "").strip()}
    if "grade" in result:
        result.add("class")
    if "class" in result:
        result.add("grade")
    if "format" in result:
        result.add("training_format")
    if "training_format" in result:
        result.add("format")
    if "subject" in result:
        result.add("course_subject")
    if "course_subject" in result:
        result.add("subject")
    return result


def _direct_path_assumed_scope_asserted(text: str, key: str, value: str) -> bool:
    if not value:
        return False
    draft = str(text or "")
    normalized_draft = _normalize_fact_match_text(draft)
    normalized_value = _normalize_fact_match_text(value)
    if key in {"grade", "class"}:
        grade = re.sub(r"\D+", "", value)[:2]
        return bool(grade and re.search(rf"\b{re.escape(grade)}\s*(?:-|–)?\s*(?:класс\w*|кл)\b", draft, re.I))
    if key in {"format", "training_format"}:
        marker = _direct_path_format_scope(value)
        if marker == "online":
            return bool(re.search(r"\bонлайн\b|\bonline\b", draft, re.I))
        if marker == "offline":
            return bool(re.search(r"\bочно\b|\boffline\b|долгопруд|москв|красносель", draft, re.I))
        return False
    if key in {"subject", "course_subject"}:
        return bool(normalized_value and len(normalized_value) >= 4 and normalized_value in normalized_draft)
    if key in {"product", "product_family"}:
        product_markers = {
            "regular_course": ("курс", "учебный год", "семестр"),
            "regular": ("курс", "учебный год", "семестр"),
            "camp": ("лагер", "смен", "лвш", "летн"),
            "trial": ("пробн",),
        }
        markers = product_markers.get(normalized_value, (normalized_value,))
        return any(marker and marker in normalized_draft for marker in markers)
    return False


def _direct_path_assumed_scope_reask_text(slots: Sequence[Mapping[str, str]]) -> str:
    first = slots[0] if slots else {}
    key = str(first.get("key") or "")
    value = str(first.get("value") or "").strip()
    if key in {"grade", "class"}:
        grade = re.sub(r"\D+", "", value)[:2] or value
        detail = f"про {grade} класс" if grade else "про этот класс"
    elif key in {"subject", "course_subject"}:
        detail = f"про предмет «{value}»" if value else "про этот предмет"
    elif key in {"format", "training_format"}:
        detail = f"про формат «{value}»" if value else "про этот формат"
    else:
        detail = f"про «{value}»" if value else "про этот параметр"
    return (
        f"Правильно ли я понимаю, что вопрос {detail}? "
        "Подтвердите, пожалуйста, и я подскажу условия без риска ошибиться."
    )


def apply_assumed_scope_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    if not _assumed_scope_guard_enabled(context):
        return result
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    trace: dict[str, Any] = {
        "schema_version": "assumed_scope_guard_v1_2026_06_16",
        "enabled": True,
        "action": "pass",
        "asserted_assumed_slots": [],
    }
    provenance = _direct_path_slot_provenance(context)
    assumed_slots = [
        {"key": key, "value": str(data.get("value") or "")}
        for key, data in provenance.items()
        if key in _ASSUMED_SCOPE_KEYS and not data.get("confirmed") and str(data.get("value") or "").strip()
    ]
    trace["assumed_slots"] = assumed_slots
    if _direct_path_assumed_scope_p0_active(result, context=context):
        trace["action"] = "skipped_p0_or_risk"
    elif result.route not in {"bot_answer_self", "bot_answer_self_for_pilot"}:
        trace["action"] = "skipped_non_self_route"
    else:
        do_not_reask = _direct_path_do_not_reask_slots(context)
        asserted = [
            slot
            for slot in assumed_slots
            if slot["key"] not in do_not_reask and _direct_path_assumed_scope_asserted(result.draft_text, slot["key"], slot["value"])
        ]
        trace["asserted_assumed_slots"] = asserted
        if asserted:
            trace["action"] = "reask_assumed_parameter"
            metadata["assumed_scope_guard"] = trace
            direct["assumed_scope_guard"] = trace
            metadata["direct_path"] = direct
            flags = tuple(dict.fromkeys((*result.safety_flags, "assumed_scope_guard_reask")))
            context_used = tuple(dict.fromkeys((*result.context_used, "assumed_scope_guard")))
            missing = tuple(dict.fromkeys((*result.missing_facts, "подтвердить параметр из контекста")))
            return replace(
                result,
                draft_text=_direct_path_assumed_scope_reask_text(asserted),
                missing_facts=missing,
                safety_flags=flags,
                context_used=context_used,
                metadata=metadata,
            )
    metadata["assumed_scope_guard"] = trace
    direct["assumed_scope_guard"] = trace
    metadata["direct_path"] = direct
    return replace(result, metadata=metadata)


def _direct_path_route_rubric_should_regenerate(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    facts: Mapping[str, str],
    model_called: bool,
    fact_pack: Optional[Mapping[str, Any]] = None,
) -> bool:
    if not _route_rubric_enabled(context):
        return False
    if not model_called or result.route != "draft_for_manager":
        return False
    if result.missing_facts:
        return False
    if _direct_keyword_fallback_relevance_enabled(context) and _direct_path_fallback_open_question(fact_pack, context):
        return True
    return bool(facts)

def _direct_path_fallback_open_question(
    fact_pack: Optional[Mapping[str, Any]],
    context: Optional[Mapping[str, Any]],
) -> bool:
    if not isinstance(fact_pack, Mapping):
        return False
    llm_retrieve = fact_pack.get("llm_retrieve")
    if not isinstance(llm_retrieve, Mapping):
        return False
    if str(llm_retrieve.get("fallback_reason") or "") not in {"empty_selection", "timeout"}:
        return False
    if not isinstance(context, Mapping):
        return False
    plan = context.get("conversation_intent_plan")
    if isinstance(plan, Mapping) and str(plan.get("direct_question") or "").strip():
        return True
    memory = context.get("dialogue_memory_view")
    open_question = memory.get("open_question") if isinstance(memory, Mapping) and isinstance(memory.get("open_question"), Mapping) else {}
    return bool(str(open_question.get("text") or "").strip() or str(open_question.get("kind") or "").strip())

def _direct_path_fallback_reask_text(context: Optional[Mapping[str, Any]]) -> str:
    requested: list[str] = []
    if isinstance(context, Mapping):
        plan = context.get("conversation_intent_plan")
        raw = plan.get("requested_slots") if isinstance(plan, Mapping) else ()
        if isinstance(raw, str):
            requested.append(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
            requested.extend(str(item or "") for item in raw)
    do_not_reask = _direct_path_do_not_reask_slots(context)
    for slot in requested:
        normalized = str(slot or "").strip()
        if not normalized or normalized in do_not_reask:
            continue
        if normalized in {"grade", "class"}:
            return "Подскажите, пожалуйста, класс ученика — тогда сориентирую точнее."
        if normalized in {"subject", "course_subject"}:
            return "Подскажите, пожалуйста, предмет — тогда сориентирую точнее."
        if normalized in {"format", "training_format"}:
            return "Подскажите, пожалуйста, какой формат удобнее: очно или онлайн?"
    return "Уточните, пожалуйста, класс, предмет или формат — тогда сориентирую без риска ошибиться."

def apply_direct_keyword_fallback_reask_layer(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    if not _direct_keyword_fallback_relevance_enabled(context):
        return result
    if result.route != "draft_for_manager" or result.missing_facts:
        return result
    metadata = dict(result.metadata)
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    if not _direct_path_fallback_open_question(direct, context):
        return result
    if str(result.risk_level or "").strip().casefold() in {"high", "p0", "critical", "high_risk"}:
        return result
    if any(re.search(r"p0|payment_dispute|refund|complaint|legal|high_risk", flag, re.I) for flag in result.safety_flags):
        return result
    llm_retrieve = direct.get("llm_retrieve") if isinstance(direct.get("llm_retrieve"), Mapping) else {}
    trace = {
        "schema_version": "direct_keyword_fallback_reask_v1_2026_06_17",
        "enabled": True,
        "status": "fired",
        "fallback_reason": str(llm_retrieve.get("fallback_reason") or ""),
    }
    direct_copy = dict(direct)
    direct_copy["keyword_fallback_reask"] = trace
    direct_copy["route_after"] = "bot_answer_self_for_pilot"
    direct_copy["is_manager_deferral"] = False
    direct_copy["reason_class"] = ""
    direct_copy["reason_evidence"] = {}
    metadata["direct_path"] = direct_copy
    metadata["keyword_fallback_reask"] = trace
    metadata["text_composition_source"] = "deterministic_clarification_question"
    metadata.pop("reason_class", None)
    metadata["is_manager_deferral"] = False
    safety_flags = tuple(flag for flag in result.safety_flags if flag != "manager_approval_required")
    return replace(
        result,
        route="bot_answer_self_for_pilot",
        draft_text=_direct_path_fallback_reask_text(context),
        manager_followup_required=False,
        safety_flags=tuple(dict.fromkeys((*safety_flags, "keyword_fallback_reask"))),
        context_used=tuple(dict.fromkeys((*result.context_used, "keyword_fallback_reask"))),
        missing_facts=("уточнить параметры вопроса",),
        metadata=metadata,
    )

def _build_direct_path_route_rubric_regen_prompt(prompt: str, first_result: SubscriptionDraftResult) -> str:
    previous_json = json.dumps(first_result.to_json_dict(include_raw_response=False), ensure_ascii=False, indent=2)
    return (
        f"{str(prompt or '').rstrip()}\n\n"
        "Предыдущий JSON-ответ модели:\n"
        f"{previous_json}\n\n"
        'В предыдущем ответе выбран "draft_for_manager", но missing_facts пуст. '
        "Если факты по вопросу есть — ответь самостоятельно по фактам. Если фактов нет или они недостаточны, "
        "задай один короткий уточняющий вопрос клиенту или заполни missing_facts конкретным недостающим фактом "
        "или нужной проверкой менеджера.\n"
        "Верни только JSON без Markdown и без комментариев."
    )

def _a2_extract_phone(text: str) -> str:
    match = _A2_PHONE_RE.search(str(text or ""))
    return match.group(0).strip() if match else ""

_PARTIAL_PHONE_CONTEXT_RE = re.compile(
    r"(?P<label>\b(?:тел(?:ефон)?|номер|контакт)\b)\s*[:—-]?\s*(?P<value>(?:\+?7|8)?[\d\s().-]{3,}\.{0,3})",
    re.I,
)

_CLIENT_CHILD_IDENTITY_PROMPT_RE = re.compile(
    r"(?P<prefix>\b(?:записыва(?:й(?:те)?|ю|ем)|запиш(?:и(?:те)?|у|ем)(?:\s+нас)?|"
    r"реб[её]н(?:ок|ка|ку)?|сын(?:а)?|доч(?:ь|ка|ку|ери)?|ученик(?:а)?|ученица|фио|зовут|имя|"
    r"справк\w*\s+на)\s*[:—-]?\s*)"
    r"(?P<name>[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2})",
    re.I,
)

_CLIENT_PARENT_IDENTITY_PROMPT_RE = re.compile(
    r"(?P<prefix>\b(?:родител[ья]|мама|папа|меня\s+зовут|я)\s*[:—-]?\s*)"
    r"(?P<name>[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){0,2})",
    re.I,
)

def _replace_echoed_phone(text: str, phone: str) -> str:
    digits = re.sub(r"\D+", "", str(phone or ""))
    if len(digits) < 7:
        return str(text or "")
    chunks: list[str] = []
    last = 0
    for match in _A2_PHONE_RE.finditer(str(text or "")):
        candidate_digits = re.sub(r"\D+", "", match.group(0))
        if candidate_digits and (candidate_digits in digits or digits in candidate_digits):
            chunks.append(str(text or "")[last : match.start()])
            chunks.append("[данные у менеджера]")
            last = match.end()
    chunks.append(str(text or "")[last:])
    return "".join(chunks)
