from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import yaml

from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.fact_scope_spec import answer_scopes_allowed, detect_fact_scopes
from mango_mvp.channels.output_verification_floor import (
    _GENERIC_HANDOFF_TEXTS as dialogue_contract_generic_handoff_texts,
    _HANDOFF_EXHAUSTED_TEXTS as dialogue_contract_handoff_exhausted_texts,
    _handoff_factual_claim_text as dialogue_contract_handoff_factual_claim_text,
    _is_pure_handoff_text as dialogue_contract_is_pure_handoff_text,
    has_meta_leak,
    is_near_repeat,
    p0_pre_gate as dialogue_contract_p0_pre_gate,
    verify_output as verify_dialogue_contract_output,
)
from mango_mvp.channels.semantic_roles import tag_message_roles
from mango_mvp.channels.text_signals import has_any_marker, has_marker
from mango_mvp.channels.tone_block import (
    TONE_RICH_FORMAT_ENV,
    tone_rich_format_enabled,
)
from mango_mvp.channels.draft_prompt_builder import (
    IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES,
    build_draft_prompt,
    safe_schedule_template,
    should_force_manager_only,
)
from mango_mvp.insights.sanitizers import sanitize_answer
from mango_mvp.question_catalog.classifier import load_valid_theme_and_service_ids


from mango_mvp.channels.subscription_llm_parts.codex_exec import (
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REASONING_EFFORT,
    _RETRYABLE_MARKERS,
    build_codex_exec_command,
    codex_isolation_cwd,
    _with_codex_exec_metadata,
    build_codex_exec_env,
    CodexExecConfig,
    extract_json_object,
    _cache_key,
    _guard_cache_dir,
    _is_retryable,
    _CodexRetryableError,
    _PromptProviderError,
)
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    SemanticReading,
    append_reading_trace_record,
    reading_class_enabled,
    semantic_frame_from_metadata,
    semantic_reading_trace_record,
    semantic_reading_transition_metadata,
)

from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_ENV,
    LLM_RETRIEVE_ENV,
    TEMPLATE_FROM_KB_ENV,
    ROUTE_RUBRIC_ENV,
    BOT_GOLD_REAL_ENV,
    PRESALE_SAFETY_ENV,
    PRESALE_PII_MEMORY_ENV,
    PRESALE_VERIFIER_FAILSOFT_ENV,
    PRESALE_META_RU_ENV,
    PRESALE_SOURCE_ID_ENV,
    MEMORY_PROVENANCE_ENV,
    MEMORY_PROVENANCE_COMPACT_ENV,
    PII_RELATION_STOPWORDS_ENV,
    MEMORY_CHILD_ELLIPSIS_ENV,
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS,
    SEMANTIC_OUTPUT_VERIFIER_ENV,
    NUMBER_GATE_SCOPE_AWARE_ENV,
    VERIFIER_HANDOFF_CLAIMS_ENV,
    OUTPUT_SANITIZER_ENV,
    _A2_PHONE_RE,
    _CLIENT_EMAIL_RE,
    _active_brand,
    _answerability_shadow_enabled,
    _client_clean_fact_text,
    _direct_path_client_safe_snapshot_fact,
    _direct_path_pilot_config,
    _direct_path_fact_by_brand_key,
    _direct_path_fact_value,
    _direct_path_load_snapshot,
    _direct_path_model_p0_enabled,
    _direct_path_snapshot_fact_text,
    _direct_path_snapshot_facts,
    _direct_path_snapshot_path_from_context,
    _direct_path_template_fact_text,
    _direct_path_template_from_fact,
    _direct_path_valid_until_ok,
    _presale_prompt_child_name_value,
    _looks_like_russian_surname,
    _fresh_fact_texts,
    _has_dialogue_contract_retrieved_facts,
    _append_fact_texts,
    _claim_supported_by_facts,
    _fact_match_anchors,
    _fact_match_unit_anchors,
    _fact_match_schedule_condition_anchors,
    _normalize_fact_match_text,
    _truthy_value,
    _explicit_truthy_setting,
    _pilot_gold_profile_enabled,
    _pilot_profile_flag_enabled,
    _pilot_profile_default_on_flag_enabled,
    _pilot_profile_overrides,
    _p0_model_led_enabled,
    _prose_model_led_enabled,
    _template_from_kb_enabled,
    _template_from_kb_trace_event,
)

from mango_mvp.channels.subscription_llm_parts.contracts import (
    SUBSCRIPTION_LLM_SCHEMA_VERSION,
    SAFE_FALLBACK_DRAFT_TEXT,
    INTERNAL_SERVICE_MARKER_RE,
    INTERNAL_SERVICE_TOKEN_RE,
    INTERNAL_SCAFFOLD_PREFIX_RE,
    INTERNAL_PROMPT_DIRECTIVE_PREFIX_RE,
    INTERNAL_PROMPT_DIRECTIVE_ANYWHERE_RE,
    INTERNAL_CLIENT_SAFE_JARGON_RE,
    INTERNAL_RUNTIME_LIMIT_JARGON_RE,
    INTERNAL_REGEN_EDIT_COMMENT_RE,
    INTERNAL_CLIENT_INSTRUCTION_RE,
    INTERNAL_MANAGER_DRAFT_RE,
    INTERNAL_SAFE_VARIANT_RE,
    ALLOWED_ROUTES,
    ALLOWED_MESSAGE_TYPES,
    BASE_SAFETY_FLAGS,
    SubscriptionDraftResult,
    _normalize_output_sanitizer_text,
    strip_internal_service_markers,
    _clean_list,
    _clean_crm_recommendations,
    _clamp_float,
)

from mango_mvp.channels.subscription_llm_parts.direct_path import (
    BOT_GOLD_REAL_PACK_ENV,
    DIRECT_PATH_SCHEMA_VERSION,
    DIRECT_PATH_WIDE_FACT_PACK_SCHEMA_VERSION,
    DIRECT_PATH_WIDE_FACT_LIMIT,
    DIRECT_PATH_WIDE_FACT_CHAR_LIMIT,
    DIRECT_PATH_REAL_MANAGER_GOLD_PACK_PATH,
    DIRECT_PATH_REAL_MANAGER_GOLD_PACK_VERSION,
    DIRECT_PATH_MISSION_TEMPLATE,
    DIRECT_PATH_MISSION_ROUTE_RUBRIC_SCOPE_REPLACEMENT,
    DIRECT_PATH_ROUTE_RUBRIC_BLOCK,
    PRESALE_PROMPT_SAFE_SLOT_KEYS,
    PRESALE_PROMPT_SENSITIVE_KEY_RE,
    PRESALE_PROMPT_CHILD_NAME_KEY_RE,
    PRESALE_PROMPT_PARENT_NAME_KEY_RE,
    DIRECT_PATH_CATEGORY_ALIASES,
    _PARTIAL_PHONE_CONTEXT_RE,
    _CLIENT_CHILD_IDENTITY_PROMPT_RE,
    _CLIENT_PARENT_IDENTITY_PROMPT_RE,
    _direct_path_mission_text,
    _direct_path_route_rubric_block,
    _direct_path_enabled,
    _llm_retrieve_enabled,
    _route_rubric_enabled,
    _presale_safety_enabled,
    _direct_path_brand_label,
    _direct_path_snapshot_fact_key,
    _template_from_kb_context_trace,
    _direct_path_fact_text,
    _direct_path_add_fact,
    _direct_path_legacy_context_fact_allowed,
    _direct_path_add_legacy_fact,
    _direct_path_legacy_context_fact_items,
    _direct_path_fact_categories,
    _direct_path_category_from_hint,
    _direct_path_selected_categories,
    _direct_path_slot_scope,
    _direct_path_format_scope,
    _direct_path_grade_in_fact,
    _direct_path_fact_conflicts_slots,
    _direct_path_fact_relevance_score,
    _direct_path_render_fact_line,
    _direct_path_render_fact_block,
    _direct_path_fact_pack_char_count,
    _direct_path_core_fact,
    _direct_path_empty_fact_pack,
    _direct_path_records_to_fact_pack,
    _direct_path_keyword_fact_pack_from_records,
    _direct_path_retriever_candidate_summary,
    build_direct_path_llm_retriever_prompt,
    _direct_path_retriever_ids,
    _direct_path_llm_retrieve_fact_pack,
    _direct_path_wide_fact_pack,
    _direct_path_context_fact_pack,
    _direct_path_recent_messages,
    _direct_path_known_slots,
    _presale_prompt_safe_key,
    _presale_prompt_safe_slot_value,
    _presale_prompt_safe_mapping,
    _presale_prompt_safe_value,
    _direct_path_prompt_known_slots,
    _direct_path_prompt_memory_view,
    _presale_prompt_safe_dialogue_text,
    _direct_path_gold_real_enabled,
    _direct_path_gold_pack_path,
    _load_direct_path_gold_real_examples,
    _direct_path_topic_hints,
    _direct_path_select_gold_real_examples,
    _direct_path_gold_prompt_block,
    _build_direct_path_prompt,
    _direct_path_metadata,
    _direct_path_merge_metadata,
    _direct_path_route_rubric_should_regenerate,
    _build_direct_path_route_rubric_regen_prompt,
    _a2_extract_phone,
    _replace_echoed_phone,
)

from mango_mvp.channels.subscription_llm_parts.reliable_answerer import (
    reliable_answerer_step1_bypass_reason,
    reliable_answerer_step1_enabled,
)


from mango_mvp.channels.subscription_llm_parts.policy_routing import (
    ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
    ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
    ADDRESS_UNPK_SAFE_TEXT,
    ADMISSION_GUARANTEE_INPUT_RE,
    ADMISSION_GUARANTEE_SAFE_TEXT,
    ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION_ENV,
    AUTONOMOUS_ROUTES,
    AUTONOMY_MATRIX_SAFE_TOPIC_IDS,
    A_THREAD_ENV,
    BRAND_FORBIDDEN_TERMS,
    BRAND_LOYALTY_FOTON_TEXT,
    BRAND_LOYALTY_UNPK_TEXT,
    COMBINED_NON_RISK_INPUT_RE,
    COMPLAINT_SAFE_TEXT,
    CONCRETE_FACT_RE,
    CONTACT_FOTON_SAFE_TEXT,
    CONTACT_UNPK_SAFE_TEXT,
    CONTRACT_ENTITY_SAFE_TEXT,
    CROSS_BRAND_GENERIC_SAFE_TEXT,
    CROSS_BRAND_LICENSE_SAFE_TEXT,
    CROSS_BRAND_PLATFORM_SAFE_TEXT,
    DISCOUNT_STACKING_SAFE_TEXT,
    EMPLOYEE_PRIVACY_SAFE_TEXT,
    FALSE_INFO_SAFE_TEXT,
    FOTON_CAMP_INSTALLMENT_SAFE_TEXT,
    FOTON_CAMP_OVERVIEW_SAFE_TEXT,
    FOTON_CITY_CAMP_AUGUST_SAFE_TEXT,
    FOTON_DOLYAMI_SAFE_TEXT,
    FOTON_INSTALLMENT_SAFE_TEXT,
    FOTON_LVSH_DATES_SAFE_TEXT,
    FOTON_LVSH_PRICE_SAFE_TEXT,
    FOTON_OFFLINE_FREE_TRIAL_GUARD_TEXT,
    FOTON_ONLINE_TRIAL_SAFE_TEXT,
    FOTON_SECOND_SUBJECT_DISCOUNT_TEXT,
    FUTURE_PRICE_INPUT_RE,
    HIGH_RISK_MARKERS,
    HIGH_RISK_THEME_IDS,
    IDENTITY_FOTON_SAFE_TEXT,
    IDENTITY_PROMPT_SAFE_TEXT,
    IDENTITY_UNPK_SAFE_TEXT,
    INDIVIDUAL_HANDOFF_SAFE_TEXT,
    LEGAL_THREAT_PII_SAFE_TEXT,
    LEGAL_THREAT_SAFE_TEXT,
    MATKAP_FEDERAL_TIMING_SAFE_TEXT,
    MATKAP_REGIONAL_SAFE_TEXT,
    MATKAP_SFR_REVIEW_SAFE_TEXT,
    MISSING_CAMP_HELPFUL_TEXT,
    MISSING_DISCOUNT_HELPFUL_TEXT,
    MISSING_DOCS_HELPFUL_TEXT,
    MISSING_GENERAL_HELPFUL_TEXT,
    MISSING_INSTALLMENT_HELPFUL_TEXT,
    MISSING_INTENSIVE_PRICE_HELPFUL_TEXT,
    MISSING_PRICE_HELPFUL_TEXT,
    MISSING_PROGRAM_HELPFUL_TEXT,
    MISSING_SCHEDULE_HELPFUL_TEXT,
    MULTICHILD_DISCOUNT_TEXT,
    OFF_TOPIC_FOTON_SAFE_TEXT,
    OFF_TOPIC_GENERIC_SAFE_TEXT,
    OFF_TOPIC_UNPK_SAFE_TEXT,
    OLD_TERM_SAFE_TEXT,
    PAYMENT_CONFIRMATION_RE,
    PAYMENT_DISPUTE_SAFE_TEXT,
    PAYMENT_LINK_SAFE_TEXT,
    PH2_ANXIETY_ENV,
    PH2_OBJECTION_ENV,
    PRECISE_CONDITION_RE,
    PROGRAM_HANDOFF_SAFE_TEXT,
    PROMOCODE_DRAFT_RE,
    PROMOCODE_SAFE_TEXT,
    QUITTANCE_SAFE_TEXT,
    REFUND_ZERO_COLLECT_SAFE_TEXT,
    RESULT_GUARANTEE_INPUT_RE,
    RESULT_GUARANTEE_SAFE_TEXT,
    RouteDecision,
    SCOPE_FACT_GUARD_ENV,
    SOFT_NEGATIVE_HANDOFF_SAFE_TEXT,
    SUBJECT_GUARD_MARKERS,
    TAX_AMOUNT_SAFE_TEXT,
    TAX_FNS_REVIEW_SAFE_TEXT,
    TAX_LICENSE_SAFE_TEXT,
    TAX_ONLINE_FORM_SAFE_TEXT,
    THIRD_PARTY_PRIVACY_SAFE_TEXT,
    UNKNOWN_TOPIC_FALLBACK_ID,
    UNPK_CAMP_ONLINE_FORMAT_SAFE_TEXT,
    UNPK_CAMP_OVERVIEW_SAFE_TEXT,
    UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT,
    UNPK_LVSH_DATES_SAFE_TEXT,
    UNPK_LVSH_GRADE_11_PRICE_DETAILS_SAFE_TEXT,
    UNPK_LVSH_GRADE_11_SAFE_TEXT,
    UNPK_LVSH_LIVING_TRANSFER_SAFE_TEXT,
    UNPK_LVSH_PRICE_DETAILS_SAFE_TEXT,
    UNPK_LVSH_PRICE_SAFE_TEXT,
    UNPK_LVSH_SEATS_SAFE_TEXT,
    UNPK_MONTHLY_SEMESTER_DISCOUNT_TEXT,
    UNPK_SECOND_SUBJECT_DISCOUNT_TEXT,
    UNPK_TRIAL_SAFE_TEXT,
    UNPK_ZVSH_WAITLIST_SAFE_TEXT,
    UNSUPPORTED_PROMISE_PATTERNS,
    _BARE_N_POINTS_RE,
    _COMPLAINT_SAFE_VARIANTS,
    _LEGAL_SAFE_VARIANTS,
    _N_POINTS_PROMISE_CONTEXT_RE,
    _PAYMENT_DISPUTE_VARIANTS,
    _REFUND_ZERO_COLLECT_VARIANTS,
    _SAFE_TEMPLATE_DISPATCHER_RECONSIDER_BLOCKING_FLAGS,
    _allowed_subjects_from_context,
    _answer_fact_scopes,
    _autonomy_enabled,
    _autonomy_policy,
    _autonomy_topic_allowed,
    _context_has_missing_fact_signal,
    _context_with_dialogue_contract_retrieved_facts,
    _conversation_intent_plan,
    _dedupe_sentence,
    _dialog_context_haystack,
    _draft_confirms_payment,
    _extract_numeric_promise_claims,
    _fact_key_root,
    _has_client_safe_current_fact,
    _has_missing_fact_signal,
    _humanity_previous_bot_texts,
    _is_combined_high_risk_case,
    _is_verified_safe_numeric_template,
    _known_fields_from_text,
    _mapping_has_client_safe_current_fact,
    _mentioned_subjects,
    _merge_known_context_fields,
    _metadata_with_guarded_original_text,
    _p0_text_with_antirepeat,
    _payment_context,
    _payment_guarded_result,
    _payment_status,
    _pipeline_contract,
    _pipeline_fact_texts,
    _retrieved_fact_matches_active_brand,
    _scope_fact_detail_label,
    _scope_fact_narrow_handoff_text,
    _scope_guard_has_foreign_concrete_fact,
    _scope_guard_has_missing_intent_fact,
    _scope_guard_missing_fact_keys,
    _scope_guard_required_fact_keys,
    _select_nonrepeating_text,
    _semantic_haystack,
    _subjects_from_retrieved_facts,
    _unstated_subject_safe_text,
    apply_payment_confirmation_guard,
    apply_subscription_policy_guards,
    apply_taxonomy_topic_guard,
    apply_unstated_subject_guard,
    find_unsupported_numeric_promises,
    is_high_risk_result,
    known_context_fields,
)

A_PROACTIVE_ENV = "TELEGRAM_A_PROACTIVE"


A_RICH_FORMAT_ENV = "TELEGRAM_A_RICH_FORMAT"


SEMANTIC_OUTPUT_VERIFIER_MODEL_ENV = "TELEGRAM_SEMANTIC_VERIFIER_MODEL"


SEMANTIC_OUTPUT_VERIFIER_REASONING_ENV = "TELEGRAM_SEMANTIC_VERIFIER_REASONING"


SEMANTIC_OUTPUT_VERIFIER_TIMEOUT_ENV = "TELEGRAM_SEMANTIC_VERIFIER_TIMEOUT_SEC"


SEMANTIC_OUTPUT_VERIFIER_MAX_ATTEMPTS_SETTING = "MANGO_EVAL_SEMANTIC_VERIFIER_MAX_ATTEMPTS"


LLM_RETRIEVE_MODEL_ENV = "TELEGRAM_LLM_RETRIEVE_MODEL"


LLM_RETRIEVE_REASONING_ENV = "TELEGRAM_LLM_RETRIEVE_REASONING"


LLM_RETRIEVE_TIMEOUT_ENV = "TELEGRAM_LLM_RETRIEVE_TIMEOUT_SEC"


NIGHT_HOURS_NOTE_ENV = "TELEGRAM_NIGHT_HOURS_NOTE"


AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION = "authoritative_output_gate_v1_2026_06_02"


SEMANTIC_OUTPUT_VERIFIER_SCHEMA_VERSION = "semantic_output_verifier_v1_2026_06_06"


SEMANTIC_VERIFIER_DOWNGRADE_REASON = "semantic_verifier_downgrade"


SEMANTIC_VERIFIER_UNAVAILABLE_REASON = "semantic_verifier_unavailable"


NIGHT_HOURS_NOTE_TEXT = "Сейчас нерабочее время — менеджер ответит ежедневно с 10:00 до 18:00 по Москве."


_MANAGER_CONTACT_PROMISE_PATTERNS = (
    re.compile(r"\b(?:менеджер\w*|сотрудник\w*)\b[^.!?\n]{0,80}\b(?:верн[её]тся|свяжется|подключится|ответит)\b", re.I),
    re.compile(r"\b(?:верн[её]тся|свяжется|подключится|ответит)\b[^.!?\n]{0,80}\b(?:менеджер\w*|сотрудник\w*)\b", re.I),
    re.compile(r"\bпередам\b[^.!?\n]{0,40}\b(?:вопрос\s+|вас\s+)?менеджер\w*\b", re.I),
)


_HUMANE_GENERIC_HANDOFF_TEXTS: tuple[str, ...] = (
    SAFE_FALLBACK_DRAFT_TEXT,
    "Не хочу гадать по неподтверждённому пункту: менеджер проверит его и вернётся с ответом.",
    "Здесь лучше сверить условия: передам вопрос менеджеру, он ответит по точным данным.",
    "Передам этот пункт менеджеру, чтобы он проверил его по актуальным данным и ответил вам.",
)


_HUMANE_DETAIL_HANDOFF_TEXTS: tuple[str, ...] = (
    "Чтобы не ошибиться, менеджер уточнит именно про {detail} и вернётся с ответом.",
    "Не хочу гадать по неподтверждённому пункту: менеджер проверит именно {detail} и ответит вам.",
    "По пункту «{detail}» нужна точная сверка — передам его менеджеру.",
    "Передам менеджеру именно вопрос про {detail}, чтобы он проверил актуальные условия.",
)


PRICE_FIX_PROCESS_SAFE_TEXT = (
    "Вы спрашиваете именно про оформление по текущим условиям. Я не буду выдумывать, достаточно ли одной заявки "
    "или нужна оплата: это проверяет менеджер по выбранному курсу. Следующий шаг простой — передам менеджеру ваш запрос, "
    "он подтвердит, как оформить по текущей цене и что нужно сделать дальше."
)


MANAGER_HANDOFF_REQUEST_SAFE_TEXT = (
    "Да, передам менеджеру: он подтвердит деталь, которую нужно проверить. "
    "Чтобы он сразу был в теме, передам ему контекст диалога: класс, предмет, формат и ваш вопрос. "
    "Повторно писать уже известные данные не нужно."
)


UNSUPPORTED_FOLLOWUP_DEADLINE_SAFE_TEXT = (
    "Передам вопрос менеджеру: он проверит детали и вернётся с ответом в рабочее время."
)


UNSUPPORTED_SCHEDULE_ASSUMPTION_SAFE_TEXT = (
    "Точное расписание зависит от класса, предмета, формата и площадки; без проверки конкретной группы не буду называть дни как факт. "
    "Передам менеджеру проверить именно ваш вариант по указанным параметрам."
)


UNSUPPORTED_OFFLINE_VISIT_INVITATION_SAFE_TEXT = (
    "Запись и оформление проходят дистанционно, приезжать не нужно. Если вам удобнее очная встреча — напишите, менеджер отдельно проверит такую возможность."
)


PRESALE_SOURCE_ID_TOKEN_PATTERN = (
    r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*_facts_\d{4}_\d{2}_\d{2}(?:[._][a-z0-9]+)*"
    r"|source_coverage_audit_\d{4}_\d{2}_\d{2}(?:[._][a-z0-9]+)*"
    r"|prices_regular_\d{4}_\d{2}(?:[._][a-z0-9]+)*"
    r"|customer:[a-f0-9]{16,}"
    r"|timeline_event:[a-f0-9]{16,}"
    r"|bot_context_chunk:[a-f0-9]{16,}"
    r"|botsafe:[^\s,;]+"
)


PRESALE_SOURCE_ID_PHRASE_RE = re.compile(
    rf"(?<![\w/.-])(?:по\s+факту|факт|источник|source|source_id|fact_id)\s+"
    rf"(?:{PRESALE_SOURCE_ID_TOKEN_PATTERN})(?![\w/.-])\s*[:;,.—-]?\s*",
    re.I,
)


PRESALE_SOURCE_ID_TOKEN_RE = re.compile(
    rf"(?<![\w/.-])(?:{PRESALE_SOURCE_ID_TOKEN_PATTERN})(?![\w/.-])",
    re.I,
)


DRAFT_PLACEHOLDER_RE = re.compile(
    r"\[(?:[^\]\n]{0,80})?(?:вставить|указать|подставить|TODO|проверенн\w+\s+ссылк|актуальн\w+\s+ссылк)(?:[^\]\n]{0,120})?\]",
    re.I,
)


OUTPUT_SANITIZER_CLIENT_TEXT_RE = re.compile(
    r"(?:^|\n)\s*(?:черновик|ответ|сообщение)\s+клиенту\s*:\s*|(?:^|\n)\s*клиенту\s*:\s*",
    re.I,
)


OUTPUT_SANITIZER_META_LINE_RE = re.compile(
    r"(?:изуча\w+\s+задач\w+|созда\w+\s+план|что\s+вижу\s*:|вопрос\s+к\s+тебе\s*:|"
    r"прежде\s+чем\s+дать\s+черновик|проблема\s+с\s+данными|инструкци\w+\s+шаг\w+\s+требу\w+|"
    r"правил\w+\s+шаг\w+\s+требу\w+|оформ\w+[^.\n]{0,120}audits/_inbox|audits/_inbox)",
    re.I,
)


PRESALE_RU_META_LINE_RE = re.compile(
    r"(?:(?:этого|этой\s+информации|такого|таких\s+данных)?\s*нет\s+в\s+подтвержд[её]нных\s+фактах|"
    r"в\s+фактах\s+нет\s+подтверждени[яе]|"
    r"не\s+подтвержд[её]н[оа]?\s+фактами|"
    r"отсутствует\s+в\s+подтвержд[её]нных\s+(?:фактах|данных))",
    re.I,
)


OUTPUT_SANITIZER_OPTION_LINE_RE = re.compile(r"^\s*(?:[A-CА-В]\)|[A-CА-В]\.)\s+", re.I)


OUTPUT_SANITIZER_PLACEHOLDER_RE = re.compile(
    r"\bуточнен\w+\s+по\s+текущей\s+теме\s*\.\s*тема\s*:\s*[^.?!\n]*(?:[.?!]|$)",
    re.I,
)


OUTPUT_SANITIZER_RAW_DETAIL_HANDOFF_RE = re.compile(
    r"(?:чтобы\s+не\s+ошибиться,\s*)?менеджер\s+уточнит\s+именно\s+про\s+(?P<detail>[^.?!\n]{20,220}?)(?=\s+и\s+верн[её]тся\s+с\s+ответом|[.?!]|$)(?:\s+и\s+верн[её]тся\s+с\s+ответом)?[.?!]?"
    r"|не\s+хочу\s+гадать\s+по\s+неподтвержд[её]нному\s+пункту:\s*менеджер\s+проверит\s+именно\s+(?P<detail2>[^.?!\n]{20,220}?)(?=\s+и\s+ответит\s+вам|[.?!]|$)(?:\s+и\s+ответит\s+вам)?[.?!]?"
    r"|передам\s+менеджеру\s+именно\s+вопрос\s+про\s+(?P<detail3>[^.?!\n]{20,220}?)(?=,\s*чтобы\s+он\s+проверил\s+актуальные\s+условия|[.?!]|$)(?:,\s*чтобы\s+он\s+проверил\s+актуальные\s+условия)?[.?!]?",
    re.I,
)


OUTPUT_SANITIZER_MANAGER_TAG_RE = re.compile(r"\[/?manager\]\s*", re.I)


OUTPUT_SANITIZER_MANAGER_TAG_INSTRUCTION_RE = re.compile(
    r"^(?=.*\[/?manager\])(?=.*(?:интерпретир\w+|служебн\w+\s+тег|тег\s+\[/?manager\])).*$",
    re.I,
)


OUTPUT_SANITIZER_SEPARATOR_LINE_RE = re.compile(r"^\s*[-–—_*]{3,}\s*$")


OUTPUT_SANITIZER_BAD_TONE_PHRASE_RE = re.compile(
    r"\bздравствующ\w*(?:\s+момент)?[,.!:;—-]*\s*|\bникакого\s+спешки\b",
    re.I,
)


COSMETIC_OPENING_RE = re.compile(
    r"^\s*(?:здравствуйте[!.]?\s*|да,\s*(?:сориентирую|подскажу|понимаю|конечно)[,!.]?\s*|"
    r"понимаю[,.]?\s*|спасибо(?:\s+за\s+сообщение|\s+за\s+вопрос)?[,.]?\s*)",
    re.I,
)


MANAGER_ACTION_PROMISE_ACTOR_RE = re.compile(r"\b(?:менеджер|сотрудник|специалист|куратор)\b", re.I)


MANAGER_ACTION_PROMISE_ACTION_RE = re.compile(
    r"\b(?:свяж(?:ется|утся)|позвон(?:ит|ят)|напиш(?:ет|ут)|ответ(?:ит|ят)|верн[её]тся|провер(?:ит|ят)|уточн(?:ит|ят))\b",
    re.I,
)


MANAGER_ACTION_PROMISE_DEADLINE_RE = re.compile(
    r"\b(?:сегодня|завтра|утром|вечером|дн[её]м|после\s+обеда|"
    r"в\s+течение\s+\d+\s*(?:минут|час(?:а|ов)?|дн(?:я|ей)?|сут(?:ок|ки)?)|"
    r"до\s+\d{1,2}(?::\d{2})?|к\s+\d{1,2}(?::\d{2})?)\b",
    re.I,
)


DERIVED_PRODUCT_NUMBER_RE = re.compile(
    r"\b\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:₽|руб(?:\.|лей|ля|ль)?|р\.)(?=$|[\s,.;:!?])|"
    r"\b(?:\d+(?:[.,]\d+)?\s*/\s*)*\d+(?:[.,]\d+)?\s*(?:%|процент(?:ов|а)?)(?=$|[\s,.;:!?])",
    re.I,
)


GATE_BLOCKING_CODES: Mapping[str, str] = {
    "hard_p0": "block",
    "zero_collect_required": "block",
    "brand_leak": "block",
    "cross_brand": "block",
    "meta_leak": "block",
    "ai_disclosure": "block",
    "identity_disclosure": "block",
    "draft_placeholder": "block",
    "promocode_leak": "block",
    "p0_promise": "block",
    "p0_money_promise": "block",
    "unsupported_promise": "block",
    "unsupported_product_claim": "block",
    "unsupported_product_number": "block",
    "fact_grounding": "downgrade",
    "general_number_without_marker": "downgrade",
    "estimate_without_uncertainty_marker": "downgrade",
    "estimate_individual_child_advice": "downgrade",
    "estimate_general_advice_risk": "downgrade",
    "unsupported_entity": "downgrade",
    "forbidden_scope": "downgrade",
    "preemptive_format": "downgrade",
    "unconfirmed_schedule": "downgrade",
    "self_contradiction": "downgrade",
    "wrong_scope": "downgrade",
    "unsupported_followup_deadline": "downgrade",
    "unsupported_manager_deadline_promise": "downgrade_keep_text",
    "unsupported_schedule_assumption": "downgrade",
    "unsupported_offline_visit_invitation": "downgrade",
    "unsupported_content_delivery_action": "downgrade",
    "unconfirmed_operational_specificity": "downgrade",
    "fake_enrollment_claim": "block",
    "proactive_pii_echo": "block",
    "proactive_too_many_questions": "downgrade",
    "proactive_emoji_overuse": "downgrade",
    "derived_product_number": "downgrade_keep_text",
    "derived_product_claim": "downgrade_keep_text",
    "individual_diagnosis": "downgrade_keep_text",
    "irrelevant_to_question": "annotate",
    "unsafe_future_commitment": "downgrade_keep_text",
    "invented_generalization": "annotate",
}


DIRECT_PATH_REPLACE_TEXT_GATE_CODES = frozenset(
    {
        "hard_p0",
        "zero_collect_required",
        "p0_promise",
        "p0_money_promise",
        "brand_leak",
        "cross_brand",
        "irrelevant_to_question",
        "unsupported_product_number",
    }
)


FOLLOWUP_DEADLINE_RE = re.compile(
    r"(?:"
    r"\b(?:менеджер|ответственн\w+\s+сотрудник|сотрудник|специалист|мы|я)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:свяж\w*|ответ\w*|напиш\w*|перезвон\w*|верн\w*)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:сегодня|завтра|послезавтра|до\s+вечера|к\s+вечеру|до\s+завтра|в\s+течение\s+(?:(?:\d+\s+)?(?:минут|час|часов|дн|дней|суток|сутки)|дня)|"
    r"не\s+позднее\s+[^.!?\n]{0,40}|до\s+\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))\b"
    r"|"
    r"\b(?:ориентир|срок)\b[^.!?\n]{0,80}\b(?:ответ[а-я]*|связ[а-я]*|менеджер[а-я]*)\b"
    r"[^.!?\n]{0,80}\bв\s+течение\s+(?:(?:\d+\s+)?(?:минут|час|часов|дн|дней|суток|сутки)|дня)\b"
    r")",
    re.I,
)


SCHEDULE_ASSUMPTION_RE = re.compile(
    r"\b(?:чаще|обычно|как\s+правило|скорее\s+всего|часто)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:выходн\w*|суббот\w*|воскресень\w*|вечер\w*|будн\w*)\b"
    r"|\b(?:есть|подбираем|подбер[её]м|можно\s+подобрать)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:групп\w*|заняти\w*|расписани\w*)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:выходн\w*|суббот\w*|воскресень\w*|вечер\w*|будн\w*)\b",
    re.I,
)


OFFLINE_VISIT_INVITATION_RE = re.compile(
    r"\b(?:приезж\w*|подъезж\w*|приход\w*|жд[её]м\s+вас|можете\s+прийти|можно\s+прийти)\b"
    r"[^.!?\n]{0,140}?"
    r"\b(?:познаком\w*|посмотр\w*|оформ\w*|запис\w*|встреч\w*|на\s+площадк\w*|в\s+офис\w*)\b",
    re.I,
)


CONTENT_DELIVERY_ACTION_RE = re.compile(
    r"\b(?:я\s+)?(?:пришл[юеё]м?|отправл[юеё]м?|дам|скин[уe]|подготовл[юеё]м?)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:фрагмент|ссылк\w*|запис[ьи]\w*|доступ)\b",
    re.I,
)


BOT_SAFE_CRM_CONTEXT_ENV = "TELEGRAM_BOT_SAFE_CRM_CONTEXT"
TIMELINE_MEMORY_IN_PROMPT_ENV = "TELEGRAM_TIMELINE_MEMORY_IN_PROMPT"
BOT_SAFE_MEMORY_STEP_GUARD_ENV = "TELEGRAM_BOT_SAFE_MEMORY_STEP_GUARD"
BOT_SAFE_MEMORY_STEP_GUARD_FLAG = "bot_safe_memory_unconfirmed_step_detected"
BOT_SAFE_MEMORY_VALID_NEXT_STEP_STATUSES = frozenset({"active", "needs_manager_review", "empty"})
BOT_SAFE_MEMORY_REVIEW_NEXT_STEP_STATUSES = frozenset({"needs_manager_review", "empty"})
BOT_SAFE_MEMORY_CONCRETE_STEP_RE = re.compile(
    r"(?:"
    r"\b(?:место|бронь|запис[ьи]\w*|заявк\w*|групп\w*)\b"
    r"[^.!?\n]{0,90}?"
    r"\b(?:заброниров\w*|закреп\w*|сохран\w*|подтвержд\w*|оформ\w*|зачисл\w*)\b"
    r"|"
    r"\b(?:заброниров\w*|закреп\w*|сохран\w*|подтвержд\w*|оформ\w*|зачисл\w*)\b"
    r"[^.!?\n]{0,90}?"
    r"\b(?:место|бронь|запис[ьи]\w*|заявк\w*|групп\w*)\b"
    r"|"
    r"\b(?:зачислен\w*|зачислил[аи]?|зачислим|зачислить)\b"
    r"[^.!?\n]{0,90}?"
    r"\b(?:курс|групп\w*|поток|программ\w*)\b"
    r"|"
    r"\b(?:гарантир\w*|точно\s+(?:будет|получится)|место\s+(?:будет|оста[её]тся)\s+за\s+вами)\b"
    r"[^.!?\n]{0,90}?"
    r"|"
    r"\b(?:перевед\w*|продвин\w*|постав\w*)\b"
    r"[^.!?\n]{0,90}?"
    r"\b(?:этап|статус|воронк\w*|сделк\w*)\b"
    r")",
    re.I,
)
NEXT_STEP_SOFT_ACTION_RE_FRAGMENT = (
    r"(?:уточн\w*|указ\w*|спрос\w*|узна\w*|выясн\w*|поня\w*|"
    r"подтверд\w*|подбер\w*|подобр\w*|провер\w*|напиш\w*|сообщ\w*)"
)
BOT_SAFE_MEMORY_SOFT_NEXT_STEP_FRAME_RE = re.compile(
    r"(?:"
    r"\bследующ(?:ий|им)\s+шаг(?:ом)?(?:\s+\w+){0,2}\s*(?:[:—-]|\b(?:будет|это)\b)\s*"
    r"|"
    r"\bдальше\s+(?:нужно|по\s+плану)\s+"
    r")"
    r"[^.!?\n]{0,120}?"
    rf"\b{NEXT_STEP_SOFT_ACTION_RE_FRAGMENT}\b"
    r"[^.!?\n]{0,120}(?:[.!?]|$)",
    re.I,
)
UNSAFE_FUTURE_COMMITMENT_RE = re.compile(
    r"(?:"
    r"\b(?:верн(?:[её]м|у)|возврат(?:им|у)|оформ(?:им|лю)\s+возврат|перевед(?:[её]м|у)|компенсир(?:уем|ую))\b"
    r"[^.!?\n]{0,100}?\b(?:деньг\w*|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*|руб|₽)\b"
    r"|"
    r"\b(?:пришл(?:ю|[её]м)|отправ(?:лю|им)|выстав(?:лю|им)|подготов(?:лю|им))\b"
    r"[^.!?\n]{0,90}?\b(?:ссылк\w*\s+на\s+оплат|сч[её]т|квитанц\w*)\b"
    r"|"
    r"\b(?:дад(?:им|у)|сдела(?:ем|ю)|оформ(?:им|лю)|закреп(?:им|лю)|примен(?:им|ю))\b"
    r"[^.!?\n]{0,80}?\b(?:скидк\w*|\d+\s*%)\b"
    r"|"
    r"\b(?:забронир(?:уем|ую)|закреп(?:им|лю)|сохран(?:им|ю)|оформ(?:им|лю)|запиш(?:ем|у)|зачисл(?:им|ю))\b"
    r"[^.!?\n]{0,90}?\b(?:место|брон\w*|заявк\w*|запис\w*|групп\w*)\b"
    r"|"
    r"\b(?:место|брон\w*|заявк\w*|запис\w*|групп\w*)\b"
    r"[^.!?\n]{0,90}?\b(?:забронир(?:уем|ую)|закреп(?:им|лю)|сохран(?:им|ю)|оформ(?:им|лю)|запиш(?:ем|у)|зачисл(?:им|ю))\b"
    r")",
    re.I,
)
SAFE_FUTURE_COMMITMENT_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"не\s+(?:обещаю|гарантирую|могу\s+обещать|буду\s+обещать)"
    r"|без\s+проверки"
    r"|после\s+проверки"
    r"|сначала\s+менеджер\s+проверит"
    r"|если\s+(?:место|группа|вариант)\s+(?:есть|будет)"
    r"|менеджер\s+(?:проверит|уточнит|подскажет|сверит)"
    r"|передам\s+(?:вопрос\s+)?менеджеру"
    r")\b",
    re.I,
)
BOT_SAFE_MEMORY_RISKY_NEXT_STEP_FRAME_RE = re.compile(
    r"(?:"
    r"\bследующ(?:ий|им)\s+шаг(?:ом)?(?:\s+\w+){0,2}\s*(?:[:—-]|\b(?:будет|это)\b)\s*"
    r"|"
    r"\bдальше\s+(?:нужно|по\s+плану)\s+"
    r"|"
    r"\bлучше\s+начать\s+с\s+"
    r")"
    r"[^.!?\n]{0,160}?"
    r"\b(?:верн\w*|возврат\w*|компенс\w*|спиш\w*|зачисл\w*|перевед\w*|подтверд\w*\s+оплат\w*)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:\d{3,}(?:[\s\u00a0]\d{3})*|руб\.?|рублей|рубля|₽|оплат\w*|деньг\w*|возврат\w*)\b"
    r"[^.!?\n]{0,80}(?:[.!?]|$)",
    re.I,
)


def _direct_path_p0_text(reason: str, context: Optional[Mapping[str, Any]]) -> tuple[str, str]:
    lowered = str(reason or "").casefold()
    if "payment_dispute" in lowered or "payment dispute" in lowered or "спис" in lowered or "chargeback" in lowered:
        return _p0_text_with_antirepeat("payment_dispute", PAYMENT_DISPUTE_SAFE_TEXT, context), "payment_dispute"
    if "refund" in lowered or "возврат" in lowered:
        return _p0_text_with_antirepeat("refund", REFUND_ZERO_COLLECT_SAFE_TEXT, context), "refund"
    if "complaint" in lowered or "жалоб" in lowered or "претенз" in lowered:
        return _p0_text_with_antirepeat("complaint", COMPLAINT_SAFE_TEXT, context), "complaint"
    return _p0_text_with_antirepeat("legal", LEGAL_THREAT_SAFE_TEXT, context), "legal"


def _direct_path_preblocked_result(
    client_message: str,
    *,
    context: Optional[Mapping[str, Any]],
    facts: Mapping[str, str],
    fact_pack: Optional[Mapping[str, Any]] = None,
) -> Optional[SubscriptionDraftResult]:
    pilot_config = _direct_path_pilot_config(context)
    p0_reason = None if _p0_model_led_enabled(context) else dialogue_contract_p0_pre_gate(client_message, context=context)
    if p0_reason:
        text, kind = _direct_path_p0_text(p0_reason, context)
        p0_guard_key = {
            "payment_dispute": "payment_dispute_manager_only",
            "refund": "zero_collect_refund_guarded",
            "complaint": "complaint_apology_guarded",
            "legal": "zero_collect_legal_guarded",
        }.get(kind, "zero_collect_legal_guarded")
        meta = _direct_path_metadata(
            attempted=True,
            model_called=False,
            facts=facts,
            fact_pack=fact_pack,
            preblocked=True,
            pilot_config=pilot_config,
            context=context,
            preblock_reason="p0_pre_gate",
            reason_class="p0_deferral",
            reason_evidence={"p0_reason": p0_reason, "p0_kind": kind},
        )
        return SubscriptionDraftResult(
            message_type="manager_only",
            broad_group="direct_path",
            route="manager_only",
            draft_text=text,
            risk_level="high",
            safety_flags=(*BASE_SAFETY_FLAGS, "direct_path_preblocked_p0", p0_guard_key, "manager_approval_required", "no_auto_send"),
            manager_checklist=("P0/high-risk: прямой путь не вызывался, отвечает менеджер.",),
            metadata={"direct_path": meta, "reason_class": "p0_deferral", "is_manager_deferral": True, p0_guard_key: True},
        )
    if reliable_answerer_step1_enabled(context):
        reliable_bypass_reason = reliable_answerer_step1_bypass_reason(client_message, context=context)
        if reliable_bypass_reason == "p0" and not _p0_model_led_enabled(context):
            text, kind = _direct_path_p0_text("payment_dispute", context)
            meta = _direct_path_metadata(
                attempted=True,
                model_called=False,
                client_message=client_message,
                facts=facts,
                fact_pack=fact_pack,
                preblocked=True,
                pilot_config=pilot_config,
                context=context,
                preblock_reason="reliable_answerer_p0_bypass",
                reason_class="p0_deferral",
                reason_evidence={"source": "reliable_answerer_bypass", "p0_kind": kind},
            )
            return SubscriptionDraftResult(
                message_type="manager_only",
                broad_group="direct_path",
                route="manager_only",
                draft_text=text,
                risk_level="high",
                safety_flags=(
                    *BASE_SAFETY_FLAGS,
                    "direct_path_preblocked_p0",
                    f"direct_path_reliable_answerer_bypass_{kind}",
                    "manager_approval_required",
                    "no_auto_send",
                ),
                manager_checklist=("P0/high-risk: прямой путь не вызывался, отвечает менеджер.",),
                metadata={
                    "direct_path": meta,
                    "reason_class": "p0_deferral",
                    "is_manager_deferral": True,
                    "reliable_answerer_bypassed_reason": "p0",
                },
            )
        if reliable_bypass_reason == "cross_brand":
            meta = _direct_path_metadata(
                attempted=True,
                model_called=False,
                client_message=client_message,
                facts=facts,
                fact_pack=fact_pack,
                preblocked=True,
                pilot_config=pilot_config,
                context=context,
                preblock_reason="reliable_answerer_cross_brand_bypass",
                reason_class="cross_brand",
                reason_evidence={"source": "reliable_answerer_bypass"},
            )
            return SubscriptionDraftResult(
                message_type="manager_only",
                broad_group="direct_path",
                route="manager_only",
                draft_text=CROSS_BRAND_GENERIC_SAFE_TEXT,
                safety_flags=(
                    *BASE_SAFETY_FLAGS,
                    "direct_path_preblocked_cross_brand",
                    "cross_brand_safe_template_applied",
                    "manager_approval_required",
                    "no_auto_send",
                ),
                manager_checklist=("Cross-brand: прямой путь не вызывался, отвечает менеджер.",),
                metadata={
                    "direct_path": meta,
                    "reason_class": "cross_brand",
                    "is_manager_deferral": True,
                    "reliable_answerer_bypassed_reason": "cross_brand",
                    "cross_brand_safe_template_applied": True,
                },
            )
    if should_force_manager_only(context):
        meta = _direct_path_metadata(
            attempted=True,
            model_called=False,
            facts=facts,
            fact_pack=fact_pack,
            preblocked=True,
            pilot_config=pilot_config,
            context=context,
            preblock_reason="force_manager_only",
            reason_class="policy_permission",
            reason_evidence={"source": "rop_policy"},
        )
        return SubscriptionDraftResult(
            message_type="manager_only",
            broad_group="direct_path",
            route="manager_only",
            draft_text=SAFE_FALLBACK_DRAFT_TEXT,
            safety_flags=(*BASE_SAFETY_FLAGS, "direct_path_preblocked_policy", "manager_approval_required", "no_auto_send"),
            manager_checklist=("Политика ROP требует менеджера: прямой путь не вызывался.",),
            metadata={"direct_path": meta, "reason_class": "policy_permission", "is_manager_deferral": True},
        )
    if _active_brand(context) == "unknown":
        meta = _direct_path_metadata(
            attempted=True,
            model_called=False,
            facts=facts,
            fact_pack=fact_pack,
            preblocked=True,
            pilot_config=pilot_config,
            context=context,
            preblock_reason="unknown_brand",
            reason_class="policy_permission",
            reason_evidence={"active_brand": "unknown"},
        )
        return SubscriptionDraftResult(
            message_type="manager_only",
            broad_group="direct_path",
            route="draft_for_manager",
            draft_text=SAFE_FALLBACK_DRAFT_TEXT,
            safety_flags=(*BASE_SAFETY_FLAGS, "direct_path_preblocked_unknown_brand", "manager_approval_required", "no_auto_send"),
            manager_checklist=("Активный бренд не определён: прямой путь не вызывался.",),
            metadata={"direct_path": meta, "reason_class": "policy_permission", "is_manager_deferral": True},
        )
    return None


def _direct_path_prepare_model_result(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    return replace(
        result,
        context_used=tuple(dict.fromkeys([*result.context_used, "direct_path", "client_safe_facts"])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "direct_path_model", "draft_only"])),
    )



def _answerability_gate_findings(gate: Mapping[str, Any]) -> list[dict[str, str]]:
    findings = gate.get("findings")
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes)):
        return []
    result: list[dict[str, str]] = []
    for item in findings:
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        result.append(
            {
                "code": code[:120],
                "source": str(item.get("source") or item.get("layer") or "").strip()[:120],
                "detail": str(item.get("detail") or item.get("message") or "").strip()[:240],
            }
        )
    return result


def _answerability_semantic_codes(verifier: Mapping[str, Any]) -> list[str]:
    codes = verifier.get("finding_codes")
    if not codes:
        findings = verifier.get("findings")
        if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)):
            codes = [
                item.get("code")
                for item in findings
                if isinstance(item, Mapping) and str(item.get("code") or "").strip()
            ]
    return list(_clean_list(codes, max_items=16, max_chars=120))


def _direct_path_answerability_trace(
    *,
    direct: Mapping[str, Any],
    gate: Mapping[str, Any],
    verifier: Mapping[str, Any],
    answerability_self: Mapping[str, Any],
    before_gate_route: str,
    final_route: str,
    gate_action: str,
    downgraded: bool,
    regenerated: bool,
    reason_class: str,
    reason_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    preblocked = bool(direct.get("preblocked"))
    lowering_layers: list[str] = []
    if preblocked:
        lowering_layers.append("preblock")
    semantic_action = str(verifier.get("action") or "").strip()
    if semantic_action in {"block", "downgrade", "downgrade_keep_text", "regenerate"} or regenerated:
        lowering_layers.append("semantic_output_verifier")
    if gate_action in {"block", "downgrade", "downgrade_keep_text"} or downgraded:
        lowering_layers.append("authoritative_output_gate")
    if final_route not in AUTONOMOUS_ROUTES and not lowering_layers:
        lowering_layers.append("direct_path_policy")
    return {
        "schema_version": "answerability_trace_v1_2026_06_15",
        "enabled": True,
        "route_before_gate": str(before_gate_route or ""),
        "route_after": str(final_route or ""),
        "lowering_layers": lowering_layers,
        "preblock": {
            "preblocked": preblocked,
            "reason": str(direct.get("preblock_reason") or "").strip(),
        },
        "direct_path": {
            "reason_class": str(direct.get("reason_class") or reason_class or "").strip(),
            "reason_evidence": dict(direct.get("reason_evidence") or reason_evidence or {}),
            "model_called": bool(direct.get("model_called")),
            "text_composition_source": str(direct.get("text_composition_source") or "").strip(),
        },
        "semantic_output_verifier": {
            "action": semantic_action,
            "finding_codes": _answerability_semantic_codes(verifier),
            "fallback_reason": str(verifier.get("fallback_reason") or "").strip()[:240],
        },
        "authoritative_output_gate": {
            "action": gate_action,
            "route_before": str(gate.get("route_before") or "").strip(),
            "route_after": str(gate.get("route_after") or "").strip(),
            "findings": _answerability_gate_findings(gate),
        },
        "answerability_self": dict(answerability_self),
        "final": {
            "reason_class": reason_class,
            "reason_evidence": dict(reason_evidence or {}),
            "is_manager_deferral": final_route not in AUTONOMOUS_ROUTES,
        },
    }


def _direct_path_finalize_metadata(
    result: SubscriptionDraftResult,
    *,
    before_gate_route: str,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    recorded_before_gate_route = str(direct.get("route_before_gate") or before_gate_route)
    gate = metadata.get("authoritative_output_gate") if isinstance(metadata.get("authoritative_output_gate"), Mapping) else {}
    verifier = metadata.get("semantic_output_verifier") if isinstance(metadata.get("semantic_output_verifier"), Mapping) else {}
    gate_action = str(gate.get("action") or "").strip()
    downgraded = gate_action in {"block", "downgrade", "downgrade_keep_text"} or (
        before_gate_route in AUTONOMOUS_ROUTES and result.route not in AUTONOMOUS_ROUTES
    )
    regenerated = bool(verifier.get("regen_attempted") or verifier.get("regen_accepted"))
    reason_class = ""
    reason_evidence: dict[str, Any] = {}
    if result.route not in AUTONOMOUS_ROUTES:
        if downgraded:
            reason_class = "output_safety"
            findings = gate.get("findings") if isinstance(gate.get("findings"), Sequence) else ()
            reason_evidence["gate_findings"] = [
                str(item.get("code") or "")
                for item in findings
                if isinstance(item, Mapping) and str(item.get("code") or "").strip()
            ]
        else:
            reason_class = str(direct.get("reason_class") or "policy_permission")
            reason_evidence = dict(direct.get("reason_evidence") or {})
    direct.update(
        {
            "route_before_gate": recorded_before_gate_route,
            "route_after": result.route,
            "authoritative_gate_action": gate_action,
            "direct_path_downgraded": downgraded,
            "downgraded": downgraded,
            "direct_path_regenerated": regenerated,
            "regenerated": regenerated,
            "deferral_text_in_self": bool(result.route in AUTONOMOUS_ROUTES and _has_manager_contact_promise(result.draft_text)),
            "is_manager_deferral": result.route not in AUTONOMOUS_ROUTES,
            "reason_class": reason_class,
            "reason_evidence": reason_evidence,
        }
    )
    template_trace = [
        dict(item)
        for item in (direct.get("template_from_kb_trace") or ())
        if isinstance(item, Mapping)
    ]
    for item in _template_from_kb_context_trace(context):
        record = dict(item)
        if record not in template_trace:
            template_trace.append(record)
    if template_trace:
        direct["template_from_kb_trace"] = template_trace
        metadata["template_from_kb_trace"] = template_trace
    if _answerability_shadow_enabled(context):
        answerability_self = metadata.get("answerability_self") if isinstance(metadata.get("answerability_self"), Mapping) else {}
        metadata["answerability_trace"] = _direct_path_answerability_trace(
            direct=direct,
            gate=gate,
            verifier=verifier,
            answerability_self=answerability_self,
            before_gate_route=recorded_before_gate_route,
            final_route=result.route,
            gate_action=gate_action,
            downgraded=downgraded,
            regenerated=regenerated,
            reason_class=reason_class,
            reason_evidence=reason_evidence,
        )
    metadata["direct_path"] = direct
    metadata["text_composition_source"] = direct.get("text_composition_source") or metadata.get("text_composition_source")
    metadata["is_manager_deferral"] = bool(direct["is_manager_deferral"])
    metadata["reason_class"] = reason_class
    return replace(result, metadata=metadata)


_INTERNAL_CLIENT_PLACEHOLDER_RE = re.compile(r"\s*\[(?:\s*данные\s+у\s+менеджера\s*|\.{3}|…)\]\s*", re.I)
def _sanitize_internal_client_placeholders(text: str) -> tuple[str, bool]:
    raw = str(text or "")
    if not raw:
        return "", False
    value, removed = _INTERNAL_CLIENT_PLACEHOLDER_RE.subn(" ", raw)
    if not removed:
        return raw, False
    return _normalize_output_sanitizer_text(value), True


def _prose_model_led_protected_result(result: SubscriptionDraftResult) -> bool:
    flags = {str(flag) for flag in result.safety_flags}
    if result.route == "manager_only" or result.topic_id in HIGH_RISK_THEME_IDS:
        return True
    protected_flags = {
        "complaint_apology_guarded",
        "payment_dispute_manager_only",
        "zero_collect_refund_guarded",
        "zero_collect_legal_guarded",
        "direct_path_preblocked_p0",
        "high_risk_manager_only",
        BOT_SAFE_MEMORY_STEP_GUARD_FLAG,
    }
    if flags & protected_flags:
        return True
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    return bool(metadata.get("final_p0_text_override") or metadata.get("forced_route_high_risk"))


def apply_prose_model_led_quality_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _prose_model_led_enabled(context):
        return result
    text = str(result.draft_text or "")
    cleaned, placeholder_removed = _sanitize_internal_client_placeholders(text)
    reasons: list[str] = []
    if placeholder_removed:
        reasons.append("internal_client_placeholder")
    protected = _prose_model_led_protected_result(result)
    repeated = False
    if not protected:
        previous = _humanity_previous_bot_texts(context)
        repeated = bool(previous and is_near_repeat(cleaned, previous[-6:], threshold=0.88))
        if repeated:
            reasons.append("near_repeat_detected")
    if cleaned == text and not reasons:
        return result
    flags = [*result.safety_flags]
    if reasons:
        flags.extend("prose_model_led:" + reason for reason in reasons)
    metadata = dict(result.metadata)
    metadata["prose_model_led"] = {
        "enabled": True,
        "applied": bool(reasons),
        "protected": protected,
        "placeholder_removed": placeholder_removed,
        "near_repeat": repeated,
        "reasons": list(dict.fromkeys(reasons)),
    }
    if placeholder_removed:
        metadata = _metadata_with_guarded_original_text(metadata, text, guard="prose_model_led")
    return replace(
        result,
        draft_text=cleaned,
        safety_flags=tuple(dict.fromkeys(flags)),
        metadata=metadata,
    )


_A2_FAKE_DONE_RE = re.compile(
    r"я\s+(?:вас\s+)?записал|вы\s+записаны|запись\s+оформлена|оформил\s+запись|записал\s+на\s+курс",
    re.I,
)


_A2_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]")


_A2_SERIOUS_TAGS = {"p0", "refund", "complaint", "manager_only", "legal", "guarantee"}


def _a2_proactive_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        for key in ("a_proactive_enabled", "proactive_enabled", A_PROACTIVE_ENV):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(A_PROACTIVE_ENV))


def _a2_rich_format_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        for key in ("a_rich_format_enabled", "rich_format_enabled", A_RICH_FORMAT_ENV, TONE_RICH_FORMAT_ENV):
            if key in context:
                return _truthy_value(context.get(key))
        if tone_rich_format_enabled(context):
            return True
    if tone_rich_format_enabled(context):
        return True
    return _truthy_value(os.getenv(A_RICH_FORMAT_ENV))


def _a2_context_tag(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    if result.route == "manager_only":
        return "manager_only"
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    for tag in ("complaint", "refund", "legal", "guarantee"):
        if tag in flags:
            return tag
    safety = classify_answer_safety(
        client_message=client_message,
        context=context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
        include_text_signals=not _p0_model_led_enabled(context),
    )
    if safety.p0_required and not safety.semantic_non_p0:
        return "p0"
    return "warm" if "a2_proactive" in result.metadata or any("a2_proactive" in flag for flag in result.safety_flags) else "neutral"


def _a2_enforce_emoji_limit(text: str, *, context_tag: str, max_emoji: int = 1) -> str:
    if context_tag in _A2_SERIOUS_TAGS:
        return _A2_EMOJI_RE.sub("", str(text or "")).strip()
    count = 0
    chars: list[str] = []
    for char in str(text or ""):
        if _A2_EMOJI_RE.match(char):
            count += 1
            if count > max_emoji:
                continue
        chars.append(char)
    return "".join(chars).strip()


def apply_authoritative_output_gate(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    """Final safety gate over every provider output.

    The gate composes existing verifiers/guards and only downgrades unsafe output.
    It is intentionally not a quality improver: it never promotes a route and never
    invents replacement facts.
    """

    result = apply_output_sanitizer(result, context=context, client_message=client_message)
    result = apply_prose_model_led_quality_guard(result, context=context, client_message=client_message)
    previous_gate = result.metadata.get("authoritative_output_gate") if isinstance(result.metadata, Mapping) else {}
    previous_findings = previous_gate.get("findings") if isinstance(previous_gate, Mapping) else ()
    findings = _dedupe_gate_findings(
        [
            *(item for item in previous_findings if isinstance(item, Mapping)),
            *_authoritative_gate_findings(result, client_message=client_message, context=context),
        ]
    )
    actions = tuple(_authoritative_gate_action(finding["code"]) for finding in findings)
    direct_path_keep_text = _authoritative_gate_direct_path_keep_text(result, findings)
    actionable = [finding for finding, action in zip(findings, actions) if action in {"block", "downgrade", "downgrade_keep_text"}]
    gate_action = (
        "downgrade_keep_text"
        if direct_path_keep_text
        else
        "block"
        if "block" in actions
        else "downgrade"
        if "downgrade" in actions
        else "downgrade_keep_text"
        if "downgrade_keep_text" in actions
        else "annotate"
        if "annotate" in actions
        else "pass"
    )
    metadata = dict(result.metadata)
    metadata["authoritative_output_gate"] = {
        "schema_version": AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION,
        "checked": True,
        "action": gate_action,
        "findings": findings,
        "route_before": result.route,
        "route_after": result.route,
    }
    if gate_action == "annotate":
        checklist = tuple(dict.fromkeys([*result.manager_checklist, _semantic_output_manager_note(findings)]))
        return apply_night_hours_note(replace(result, manager_checklist=checklist, metadata=metadata), context=context)
    if not actionable:
        if direct_path_keep_text:
            actionable = list(findings)
        else:
            return apply_night_hours_note(replace(result, metadata=metadata), context=context)
    if not actionable:
        return apply_night_hours_note(replace(result, metadata=metadata), context=context)

    route = (
        result.route
        if direct_path_keep_text and result.route == "manager_only"
        else "draft_for_manager"
        if direct_path_keep_text
        else _authoritative_gate_downgraded_route(result.route, actions)
    )
    metadata["authoritative_output_gate"]["route_after"] = route
    codes = tuple(dict.fromkeys(str(item["code"]) for item in actionable))
    flags = tuple(
        dict.fromkeys(
            [
                *result.safety_flags,
                "authoritative_output_gate_blocked",
                *[f"authoritative_gate:{code}" for code in codes],
                *(("direct_path_gate_text_preserved",) if direct_path_keep_text else ()),
                "manager_approval_required",
                "no_auto_send",
            ]
        )
    )
    semantic_note = _semantic_output_manager_note(actionable)
    derived_number_notes = _derived_product_number_manager_notes(actionable)
    checklist_items = [
        *result.manager_checklist,
        (
            "Финальный safety gate перевёл прямой путь в менеджерский черновик: проверить findings перед отправкой."
            if direct_path_keep_text
            else "Финальный safety gate заблокировал клиентский текст: не отправлять без ручной проверки."
        ),
    ]
    checklist_items.extend(derived_number_notes)
    has_semantic_finding = any(
        str(item.get("source") or "") == "semantic_output_verifier" or str(item.get("code") or "") in _SEMANTIC_OUTPUT_VERIFIER_CODES
        for item in actionable
    )
    if "downgrade_keep_text" in actions and (has_semantic_finding or not derived_number_notes):
        checklist_items.append(semantic_note)
    checklist = tuple(
        dict.fromkeys(
            checklist_items
        )
    )
    forbidden = tuple(dict.fromkeys([*result.forbidden_promises_detected, *codes]))
    keep_text_only = direct_path_keep_text or (
        "block" not in actions and "downgrade" not in actions and "downgrade_keep_text" in actions
    )
    if keep_text_only:
        semantic_meta = dict(metadata.get("semantic_output_verifier") or {})
        if semantic_meta:
            semantic_meta["fallback_reason"] = semantic_meta.get("fallback_reason") or SEMANTIC_VERIFIER_DOWNGRADE_REASON
            metadata["semantic_output_verifier"] = semantic_meta
        return apply_night_hours_note(
            replace(
                result,
                route=route,
                safety_flags=flags,
                manager_checklist=checklist,
                forbidden_promises_detected=forbidden,
                metadata=metadata,
                error=result.error,
            ),
            context=context,
        )
    return apply_night_hours_note(
        replace(
            result,
            route=route,
            draft_text=_direct_path_generic_replacement_text(context)
            if _truthy_value((result.metadata.get("direct_path") or {}).get("direct_path_attempted") if isinstance(result.metadata.get("direct_path"), Mapping) else False)
            else SAFE_FALLBACK_DRAFT_TEXT,
            safety_flags=flags,
            manager_checklist=checklist,
            forbidden_promises_detected=forbidden,
            metadata=metadata,
            error=result.error or "authoritative_output_gate_blocked",
        ),
        context=context,
    )


def apply_night_hours_note(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _night_hours_note_enabled(context):
        return result
    text = str(result.draft_text or "").strip()
    if not text or NIGHT_HOURS_NOTE_TEXT in text:
        return result
    if not _has_manager_contact_promise(text):
        return result
    if not _outside_moscow_work_hours(context):
        return result
    flags = tuple(dict.fromkeys([*result.safety_flags, "night_hours_note_applied"]))
    metadata = {
        **dict(result.metadata),
        "night_hours_note": {
            "applied": True,
            "hour_msk": _current_moscow_hour(context),
            "window": "10:00-18:00",
        },
    }
    return replace(result, draft_text=f"{text} {NIGHT_HOURS_NOTE_TEXT}", safety_flags=flags, metadata=metadata)


def _night_hours_note_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (NIGHT_HOURS_NOTE_ENV, "night_hours_note"):
            if key in context:
                return _truthy_value(context.get(key))
    if NIGHT_HOURS_NOTE_ENV in os.environ:
        return _truthy_value(os.getenv(NIGHT_HOURS_NOTE_ENV))
    return False


def _has_manager_contact_promise(text: str) -> bool:
    return any(pattern.search(str(text or "")) for pattern in _MANAGER_CONTACT_PROMISE_PATTERNS)


def _current_moscow_hour(context: Optional[Mapping[str, Any]] = None) -> int:
    if isinstance(context, Mapping):
        for key in ("now_msk_hour", "current_msk_hour", "moscow_hour", "hour_msk"):
            if key in context:
                try:
                    return int(float(str(context.get(key)))) % 24
                except Exception:
                    break
    return datetime.now(ZoneInfo("Europe/Moscow")).hour


def _outside_moscow_work_hours(context: Optional[Mapping[str, Any]] = None) -> bool:
    hour = _current_moscow_hour(context)
    return hour < 10 or hour >= 18


def _authoritative_gate_direct_path_keep_text(
    result: SubscriptionDraftResult,
    findings: Sequence[Mapping[str, Any]],
) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    if not _truthy_value(direct.get("enabled") or direct.get("attempted") or direct.get("direct_path_attempted")):
        return False
    codes = {str(item.get("code") or "").strip() for item in findings if isinstance(item, Mapping)}
    if not codes:
        return False
    return not bool(codes & DIRECT_PATH_REPLACE_TEXT_GATE_CODES)


def _direct_path_generic_replacement_text(context: Optional[Mapping[str, Any]]) -> str:
    previous = _humanity_previous_bot_texts(context)
    return _select_nonrepeating_text(
        _HUMANE_GENERIC_HANDOFF_TEXTS,
        previous,
        fallback="Передам этот пункт менеджеру, чтобы он проверил актуальные условия и ответил вам.",
    )


def apply_output_sanitizer(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    client_message: str = "",
) -> SubscriptionDraftResult:
    sanitizer_enabled = _output_sanitizer_enabled(context)
    client_pii_deecho_allowed = not _a2_is_proactive_result(result)
    pii_client_message = (
        _client_pii_echo_context(client_message=client_message, context=context)
        if client_pii_deecho_allowed
        else ""
    )
    pii_allowed_dialogue = (
        _client_pii_echo_context(client_message=client_message, context=context, include_slot_context=False)
        if client_pii_deecho_allowed
        else ""
    )
    pii_names_for_checklist: tuple[str, ...] = ()
    if pii_client_message:
        pii_names_for_checklist = tuple(
            dict.fromkeys(
                [
                    *_client_name_echoes(
                        pii_client_message,
                        result.draft_text,
                        allowed_client_message=pii_allowed_dialogue,
                    ),
                    *_unexpected_client_name_echoes(
                        result.draft_text,
                        allowed_client_message=pii_allowed_dialogue,
                    ),
                ]
            )
        )
    if sanitizer_enabled:
        cleaned, reasons = _sanitize_output_client_text(
            result.draft_text,
            client_message=pii_client_message,
            allowed_client_message=pii_allowed_dialogue,
            presale_ru_meta=_presale_safety_enabled(context, subflag=PRESALE_META_RU_ENV),
            presale_source_id=_presale_safety_enabled(context, subflag=PRESALE_SOURCE_ID_ENV),
        )
    else:
        cleaned, reasons = _sanitize_client_pii_echo(
            result.draft_text,
            client_message=pii_client_message,
            allowed_client_message=pii_allowed_dialogue,
        )
        if cleaned != result.draft_text and "client_pii_echo" not in reasons:
            reasons = (*reasons, "client_pii_echo")
        if _presale_safety_enabled(context, subflag=PRESALE_META_RU_ENV):
            cleaned, meta_reasons = _sanitize_presale_ru_meta_lines(cleaned)
            reasons = (*reasons, *meta_reasons)
        if _presale_safety_enabled(context, subflag=PRESALE_SOURCE_ID_ENV):
            cleaned, source_id_reasons = _sanitize_presale_source_id_text(cleaned)
            reasons = (*reasons, *source_id_reasons)
    if not reasons and cleaned == result.draft_text:
        return result

    fallback = not cleaned.strip()
    route = result.route
    flags = [*result.safety_flags, "output_sanitizer_applied", *[f"output_sanitizer:{reason}" for reason in reasons]]
    checklist = list(result.manager_checklist)
    pii_manager_items = _client_pii_manager_items(pii_client_message)
    if "client_name_echo" in reasons and pii_names_for_checklist:
        checklist.append(
            "Проверьте имя в черновике: "
            + ", ".join(pii_names_for_checklist[:4])
            + " не было разрешено текущим диалогом или было ФИО целиком; в тексте замаскировано."
        )
    if pii_manager_items and any(reason in reasons for reason in ("client_name_echo", "client_phone_echo", "client_email_echo")):
        checklist.append("ПДн из диалога для менеджера: " + "; ".join(pii_manager_items[:8]) + ".")
    if fallback:
        cleaned = SAFE_FALLBACK_DRAFT_TEXT
        if route != "manager_only":
            route = "draft_for_manager"
        flags.extend(["manager_approval_required", "no_auto_send"])
        checklist.append("Output sanitizer удалил внутренний текст целиком: не отправлять без ручной проверки.")
    metadata = dict(result.metadata)
    metadata = _metadata_with_guarded_original_text(metadata, result.draft_text, guard="output_sanitizer")
    metadata["output_sanitizer"] = {
        "enabled": sanitizer_enabled,
        "applied": True,
        "fallback": fallback,
        "reasons": list(reasons),
        "route_before": result.route,
        "route_after": route,
        "text_before_len": len(str(result.draft_text or "")),
        "text_after_len": len(cleaned),
    }
    return replace(
        result,
        route=route,
        draft_text=cleaned,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
        error=result.error or ("output_sanitizer_fallback" if fallback else result.error),
    )


def _sanitize_output_client_text(
    text: str,
    *,
    client_message: str = "",
    allowed_client_message: Optional[str] = None,
    presale_ru_meta: bool = False,
    presale_source_id: bool = False,
) -> tuple[str, tuple[str, ...]]:
    raw = str(text or "")
    if not raw:
        return "", ()

    value = raw
    reasons: list[str] = []
    marker_matches = list(OUTPUT_SANITIZER_CLIENT_TEXT_RE.finditer(value))
    if marker_matches:
        tail = value[marker_matches[-1].end() :].strip()
        if tail:
            value = tail
            reasons.append("client_text_marker")

    plan_context = bool(
        OUTPUT_SANITIZER_META_LINE_RE.search(raw)
        or re.search(r"^\s*(?:[A-CА-В]\)|[A-CА-В]\.)\s+", raw, flags=re.I | re.M)
    )
    value, placeholder_removed = OUTPUT_SANITIZER_PLACEHOLDER_RE.subn(" ", value)
    if placeholder_removed:
        reasons.append("topic_placeholder")
    value, raw_detail_removed = _sanitize_raw_detail_handoff_text(value)
    if raw_detail_removed:
        reasons.append("raw_detail_handoff")
    value, regen_edit_removed = INTERNAL_REGEN_EDIT_COMMENT_RE.subn(" ", value)
    if regen_edit_removed:
        reasons.append("regen_edit_comment")

    kept_lines: list[str] = []
    for line in value.splitlines() or [value]:
        stripped = line.strip()
        if not stripped:
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue
        if OUTPUT_SANITIZER_SEPARATOR_LINE_RE.fullmatch(stripped):
            reasons.append("tone_separator")
            continue
        if OUTPUT_SANITIZER_MANAGER_TAG_INSTRUCTION_RE.search(stripped):
            reasons.append("manager_tag_instruction")
            continue
        if OUTPUT_SANITIZER_META_LINE_RE.search(stripped):
            reasons.append("meta_process_line")
            continue
        if presale_ru_meta and PRESALE_RU_META_LINE_RE.search(stripped):
            reasons.append("presale_ru_meta_line")
            continue
        if plan_context and OUTPUT_SANITIZER_OPTION_LINE_RE.search(stripped):
            reasons.append("plan_option_line")
            continue
        kept_lines.append(stripped)
    value = "\n".join(kept_lines)

    if presale_source_id:
        value, source_id_reasons = _sanitize_presale_source_id_text(value)
        reasons.extend(source_id_reasons)

    value, bad_tone_removed = OUTPUT_SANITIZER_BAD_TONE_PHRASE_RE.subn("", value)
    if bad_tone_removed:
        reasons.append("bad_tone_phrase")

    value, tag_removed = OUTPUT_SANITIZER_MANAGER_TAG_RE.subn("", value)
    if tag_removed:
        reasons.append("manager_tag")

    stripped = strip_internal_service_markers(value)
    if stripped != value:
        value = stripped
        reasons.append("internal_service_marker")

    value, pii_reasons = _sanitize_client_pii_echo(
        value,
        client_message=client_message,
        allowed_client_message=allowed_client_message,
    )
    reasons.extend(pii_reasons)

    value = _normalize_output_sanitizer_text(value)
    if _output_sanitizer_degenerate(value):
        reasons.append("degenerate_output")
        return "", tuple(dict.fromkeys(reasons))
    if value != raw and not reasons:
        reasons.append("normalized")
    return value, tuple(dict.fromkeys(reasons))


def _sanitize_presale_ru_meta_lines(text: str) -> tuple[str, tuple[str, ...]]:
    raw = str(text or "")
    if not raw or not PRESALE_RU_META_LINE_RE.search(raw):
        return raw, ()
    kept: list[str] = []
    removed = False
    for line in raw.splitlines() or [raw]:
        stripped = line.strip()
        if stripped and PRESALE_RU_META_LINE_RE.search(stripped):
            removed = True
            continue
        kept.append(line)
    value = _normalize_output_sanitizer_text("\n".join(kept))
    return value, ("presale_ru_meta_line",) if removed else ()


def _sanitize_presale_source_id_text(text: str) -> tuple[str, tuple[str, ...]]:
    raw = str(text or "")
    if not raw or not PRESALE_SOURCE_ID_TOKEN_RE.search(raw):
        return raw, ()
    value, phrase_removed = PRESALE_SOURCE_ID_PHRASE_RE.subn(" ", raw)
    value, token_removed = PRESALE_SOURCE_ID_TOKEN_RE.subn(" ", value)
    if not phrase_removed and not token_removed:
        return raw, ()
    return _normalize_output_sanitizer_text(value), ("presale_source_id",)


_CLIENT_NAME_PAIR_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2}\b")


_CLIENT_NAME_MARKER_RE = re.compile(
    r"(?i:(?:записыва(?:й(?:те)?|ю|ем)|запиш(?:и(?:те)?|у|ем)(?:\s+нас)?|реб[её]н(?:ок|ка|ку)?|сын(?:а)?|доч(?:ь|ка|ку|ери)?|"
    r"ученик(?:а)?|ученица|фио|зовут|имя))\s*[:—-]?\s*"
    r"(?P<name>[А-ЯЁ][а-яё]{2,}(?:[ \t]+[А-ЯЁ][а-яё]{2,}){0,2})",
)


_CLIENT_SELF_NAME_MARKER_RE = re.compile(
    r"(?i:(?:\bя\b|меня|мама|папа|родител[ья]))\s*[:—-]?\s*"
    r"(?P<name>[А-ЯЁ][а-яё]{2,}(?:[ \t]+[А-ЯЁ][а-яё]{2,}){0,1})",
)


_CLIENT_NAME_STOPWORDS = {
    "добрый",
    "добрая",
    "вечер",
    "день",
    "утро",
    "здравствуйте",
    "привет",
    "фотон",
    "унпк",
    "мфти",
    "москва",
    "сретенка",
    "красносельская",
    "менеджер",
}


_CLIENT_RELATION_NAME_STOPWORDS = {
    "сын",
    "сына",
    "сыну",
    "сыном",
    "сыне",
    "дочь",
    "дочку",
    "дочка",
    "дочке",
    "дочки",
    "дочери",
    "дочерью",
    "ребенок",
    "ребенка",
    "ребенку",
    "ребенком",
    "ребенке",
    "ребёнок",
    "ребёнка",
    "ребёнку",
    "ребёнком",
    "ребёнке",
    "мальчик",
    "мальчика",
    "мальчику",
    "мальчиком",
    "мальчике",
    "девочка",
    "девочку",
    "девочке",
    "девочки",
    "девочкой",
}


_CLIENT_PII_CONFIRMATION_RE = re.compile(
    r"\b(?:принял[аи]?|записал[аи]?|передам|менеджер|свяжется|контакт|телефон|номер|заявк[ауи])\b",
    re.I,
)


def _sanitize_client_pii_echo(
    text: str,
    *,
    client_message: str = "",
    allowed_client_message: Optional[str] = None,
) -> tuple[str, tuple[str, ...]]:
    value = str(text or "")
    client = str(client_message or "")
    if not value or not client:
        return value, ()
    allowed = client if allowed_client_message is None else str(allowed_client_message or "")
    phone = _a2_extract_phone(client)
    phone_echoed = bool(phone and _a2_phone_echoed(phone, value))
    email_echoes = tuple(dict.fromkeys(match.group(0) for match in _CLIENT_EMAIL_RE.finditer(client) if match.group(0) in value))
    echoed_names = tuple(
        dict.fromkeys(
            [
                *_client_name_echoes(client, value, allowed_client_message=allowed),
                *_unexpected_client_name_echoes(value, allowed_client_message=allowed),
            ]
        )
    )
    if not phone_echoed and not email_echoes and not echoed_names:
        return value, ()

    reasons: list[str] = []
    if phone_echoed:
        reasons.append("client_phone_echo")
    if email_echoes:
        reasons.append("client_email_echo")
    if echoed_names:
        reasons.append("client_name_echo")

    child_first_names = _client_dialogue_child_first_names(client)
    parent_names = _client_dialogue_parent_names(client)
    safe_child_replacements = {
        name: first
        for name in echoed_names
        for first in (_presale_prompt_child_name_value(name),)
        if first and len(str(name).split()) >= 2 and _client_name_allowed(first, child_first_names)
    }
    whole_identity_echoed = any(" " in name and name not in safe_child_replacements for name in echoed_names)
    if _CLIENT_PII_CONFIRMATION_RE.search(value) and (phone_echoed or email_echoes or whole_identity_echoed) and not safe_child_replacements:
        return "Записала, передам менеджеру — он свяжется с вами.", tuple(reasons)

    if phone_echoed:
        value = _replace_echoed_phone(value, phone)
    for email in email_echoes:
        value = value.replace(email, "[данные у менеджера]")
    for name in echoed_names:
        if " " not in name and _client_name_allowed(name, child_first_names):
            continue
        replacement = safe_child_replacements.get(name)
        if replacement is None:
            if " " in name or _client_name_allowed(name, parent_names):
                replacement = "[данные у менеджера]"
            elif _looks_like_proper_person_name(name):
                replacement = "данные ребёнка"
            else:
                replacement = "[данные у менеджера]"
        value = re.sub(_flexible_name_pattern(name), replacement, value, flags=re.I)
    return value, tuple(reasons)


def _client_pii_echo_context(
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    include_slot_context: bool = True,
) -> str:
    items: list[str] = []
    if isinstance(context, Mapping):
        memory = context.get("dialogue_memory_view")
        if isinstance(memory, Mapping):
            turns = memory.get("recent_turns")
            if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes, bytearray)):
                for item in turns:
                    if isinstance(item, Mapping) and str(item.get("role") or "").casefold() in {"client", "user"}:
                        text = str(item.get("text") or "").strip()
                        if text:
                            items.append(text)
            if include_slot_context and _presale_safety_enabled(context, subflag=PRESALE_PII_MEMORY_ENV):
                items.extend(_client_pii_slot_context_lines(memory))
        recent = context.get("recent_messages")
        if isinstance(recent, Sequence) and not isinstance(recent, (str, bytes, bytearray)):
            for item in recent:
                text = str(item or "").strip()
                if text.casefold().startswith(("клиент:", "client:", "user:")):
                    value = text.split(":", 1)[-1].strip()
                    if value:
                        items.append(value)
        if include_slot_context and _presale_safety_enabled(context, subflag=PRESALE_PII_MEMORY_ENV):
            items.extend(_client_pii_slot_context_lines(context))
    current = str(client_message or "").strip()
    if current:
        items.append(current)
    deduped = tuple(dict.fromkeys(item for item in items if item))
    return "\n".join(deduped[-8:])


PRESALE_PII_NAME_KEY_RE = re.compile(r"(?:name|имя|фио|fio|parent|mother|father|мам|пап|родител|client)", re.I)


PRESALE_PII_CHILD_NAME_KEY_RE = re.compile(r"(?:child|student|реб[её]н|ученик|доч|сын)", re.I)


PRESALE_PII_PHONE_KEY_RE = re.compile(r"(?:phone|телефон|contact|контакт)", re.I)


def _client_pii_slot_context_lines(source: Mapping[str, Any]) -> list[str]:
    containers: list[Mapping[str, Any]] = []
    for key in ("known_slots", "known_dialog_fields", "known_client_fields", "client_identity", "crm_known_slots", "client_confirmed_slots"):
        value = source.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    memory = source.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        containers.extend(_client_pii_slot_context_lines_as_containers(memory))
    lines: list[str] = []
    for container in containers:
        for key, raw in container.items():
            value = raw.get("value") if isinstance(raw, Mapping) else raw
            text = " ".join(str(value or "").split()).strip(" ,.;:!?")
            if not text:
                continue
            key_text = str(key or "")
            if PRESALE_PII_PHONE_KEY_RE.search(key_text) or _A2_PHONE_RE.fullmatch(text):
                lines.append(f"телефон {text}")
            elif PRESALE_PII_CHILD_NAME_KEY_RE.search(key_text):
                lines.append(f"ребёнок {text}")
            elif PRESALE_PII_NAME_KEY_RE.search(key_text):
                lines.append(f"меня зовут {text}")
    return lines


def _client_pii_manager_items(client_context: str) -> tuple[str, ...]:
    text = " ".join(str(client_context or "").split())
    if not text:
        return ()
    items: list[str] = []
    for match in _CLIENT_NAME_MARKER_RE.finditer(text):
        name = " ".join(str(match.group("name") or "").split()).strip(" ,.;:!?")
        if name and len(name.split()) >= 2:
            items.append(f"ФИО/имя: {name}")
    for match in _CLIENT_SELF_NAME_MARKER_RE.finditer(text):
        name = " ".join(str(match.group("name") or "").split()).strip(" ,.;:!?")
        if name:
            items.append(f"ФИО/имя родителя: {name}")
    for match in _CLIENT_NAME_PAIR_RE.finditer(text):
        name = " ".join(match.group(0).split())
        words = [word.casefold().replace("ё", "е") for word in name.split()]
        if any(word in _client_name_stopwords() for word in words):
            continue
        items.append(f"ФИО/имя: {name}")
    for match in _A2_PHONE_RE.finditer(text):
        items.append(f"телефон: {match.group(0).strip()}")
    for match in _CLIENT_EMAIL_RE.finditer(text):
        items.append(f"email: {match.group(0).strip()}")
    return tuple(dict.fromkeys(items))


def _client_name_stopwords() -> set[str]:
    result = set(_CLIENT_NAME_STOPWORDS)
    if _pilot_profile_default_on_flag_enabled(None, PII_RELATION_STOPWORDS_ENV):
        result.update(item.replace("ё", "е") for item in _CLIENT_RELATION_NAME_STOPWORDS)
    return result


def _client_pii_slot_context_lines_as_containers(source: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = []
    for key in ("known_slots", "client_confirmed_slots", "crm_known_slots", "client_identity", "known_client_fields"):
        value = source.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    return containers


def _client_name_echoes(
    client_message: str,
    bot_text: str,
    *,
    allowed_client_message: Optional[str] = None,
) -> tuple[str, ...]:
    candidates: list[str] = []
    client = str(client_message or "").strip()
    allowed_names = _client_dialogue_allowed_names(client if allowed_client_message is None else str(allowed_client_message or ""))
    phone = _a2_extract_phone(client)
    for match in _CLIENT_NAME_MARKER_RE.finditer(client):
        name = match.group("name")
        candidates.append(name)
        parts = [part for part in str(name or "").split() if part]
        if len(parts) >= 2:
            candidates.append(parts[-1])
    for match in _CLIENT_SELF_NAME_MARKER_RE.finditer(client):
        name = match.group("name")
        candidates.append(name)
        parts = [part for part in str(name or "").split() if part]
        proper_parts = [part for part in parts if re.match(r"^[А-ЯЁ]", part)]
        if len(proper_parts) >= 1:
            candidates.append(proper_parts[0])
        if len(proper_parts) >= 2:
            candidates.append(proper_parts[-1])
    if phone:
        phone_pos = client.find(phone)
        for match in _CLIENT_NAME_PAIR_RE.finditer(client):
            if phone_pos >= 0 and abs(match.start() - phone_pos) > 140:
                continue
            candidates.append(match.group(0))
    for match in _CLIENT_CHILD_IDENTITY_PROMPT_RE.finditer(client):
        candidates.append(match.group("name"))
    result: list[str] = []
    for raw in candidates:
        name = " ".join(str(raw or "").split()).strip(" ,.;:!?")
        words = [word.casefold().replace("ё", "е") for word in name.split()]
        if not words or any(word in _client_name_stopwords() for word in words):
            continue
        if len(name.split()) == 1 and _client_name_allowed(name, allowed_names):
            continue
        if _client_name_echoed(name, bot_text) and name not in result:
            result.append(name)
    return tuple(result)


_DRAFT_PERSON_NAME_CONTEXT_RE = re.compile(
    r"(?:(?i:спасибо,|здравствуйте,|добрый\s+(?:день|вечер),|доброе\s+утро,|"
    r"записал[аи]?|запишем|передайте|по\s+сыну|по\s+дочери|для|"
    r"сын[ау]?|доч(?:ь|ку|ери)?|реб[её]н(?:ок|ка|ку)?|ученик(?:а)?|ученица))\s+"
    r"(?P<name>[А-ЯЁ][а-яё]{2,})"
)


def _unexpected_client_name_echoes(bot_text: str, *, allowed_client_message: str = "") -> tuple[str, ...]:
    allowed_names = _client_dialogue_allowed_names(allowed_client_message)
    result: list[str] = []
    for match in _DRAFT_PERSON_NAME_CONTEXT_RE.finditer(str(bot_text or "")):
        name = " ".join(str(match.group("name") or "").split()).strip(" ,.;:!?")
        if not name:
            continue
        normalized = name.casefold().replace("ё", "е")
        if normalized in _client_name_stopwords() or _client_name_allowed(name, allowed_names):
            continue
        if name not in result:
            result.append(name)
    return tuple(result)


def _client_dialogue_allowed_names(client_message: str) -> tuple[str, ...]:
    candidates: list[str] = []
    client = " ".join(str(client_message or "").split())
    for match in _CLIENT_NAME_MARKER_RE.finditer(client):
        name = " ".join(str(match.group("name") or "").split()).strip(" ,.;:!?")
        if name:
            candidates.append(name)
            parts = [part for part in name.split() if part]
            if len(parts) >= 2:
                candidates.append(parts[-1])
    for match in _CLIENT_SELF_NAME_MARKER_RE.finditer(client):
        name = " ".join(str(match.group("name") or "").split()).strip(" ,.;:!?")
        if name:
            parts = [part for part in name.split() if part]
            proper_parts = [part for part in parts if re.match(r"^[А-ЯЁ]", part)]
            candidates.extend(proper_parts[:2])
    result: list[str] = []
    for raw in candidates:
        words = [word.casefold().replace("ё", "е") for word in str(raw or "").split()]
        if not words or any(word in _client_name_stopwords() for word in words):
            continue
        value = " ".join(str(raw or "").split())
        if value and value not in result:
            result.append(value)
    return tuple(result)


def _client_dialogue_child_first_names(client_message: str) -> tuple[str, ...]:
    result: list[str] = []
    for line in _client_pii_context_lines(client_message):
        for match in _CLIENT_CHILD_IDENTITY_PROMPT_RE.finditer(line):
            if not _child_identity_match_has_child_context(line, match):
                continue
            first = _presale_prompt_child_name_value(match.group("name"))
            if first and first not in result:
                result.append(first)
        for match in _CLIENT_NAME_MARKER_RE.finditer(line):
            if not _child_identity_match_has_child_context(line, match):
                continue
            first = _presale_prompt_child_name_value(match.group("name"))
            if first and first not in result:
                result.append(first)
    return tuple(result)


def _client_dialogue_parent_names(client_message: str) -> tuple[str, ...]:
    result: list[str] = []
    for client in _client_pii_context_lines(client_message):
        for pattern in (_CLIENT_PARENT_IDENTITY_PROMPT_RE, _CLIENT_SELF_NAME_MARKER_RE):
            for match in pattern.finditer(client):
                name = " ".join(str(match.group("name") or "").split()).strip(" ,.;:!?")
                for part in ([name] + [item for item in name.split() if item]):
                    if part and _looks_like_proper_person_name(part) and part not in result:
                        result.append(part)
    return tuple(result)


_child_identity_context_markers = (
    "записыва",
    "запиш",
    "ребен",
    "сын",
    "доч",
    "ученик",
    "ученица",
    "справк",
)


def _client_pii_context_lines(client_message: str) -> tuple[str, ...]:
    return tuple(
        line
        for line in (" ".join(raw.split()).strip() for raw in str(client_message or "").splitlines())
        if line
    )


def _child_identity_match_has_child_context(line: str, match: re.Match[str]) -> bool:
    matched = str(match.group(0) or "")
    name = str(match.group("name") or "")
    prefix = matched[: max(0, len(matched) - len(name))]
    if _has_child_identity_context_marker(prefix):
        return True
    before = str(line or "")[max(0, match.start() - 32) : match.start()]
    return _has_child_identity_context_marker(before)


def _has_child_identity_context_marker(value: str) -> bool:
    normalized = str(value or "").casefold().replace("ё", "е")
    return any(marker in normalized for marker in _child_identity_context_markers)


def _looks_like_proper_person_name(value: str) -> bool:
    words = [word for word in str(value or "").split() if word]
    return bool(words) and all(_looks_like_capitalized_cyrillic_word(word) for word in words)


def _looks_like_capitalized_cyrillic_word(value: str) -> bool:
    word = str(value or "")
    if len(word) < 3:
        return False
    first, rest = word[0], word[1:]
    return first in "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" and all(
        char in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for char in rest
    )


def _client_name_allowed(name: str, allowed_names: Sequence[str]) -> bool:
    value = str(name or "").strip()
    if not value:
        return False
    return any(_client_name_echoed(allowed, value) or _client_name_echoed(value, allowed) for allowed in allowed_names)


def _client_name_echoed(name: str, text: str) -> bool:
    return bool(re.search(_flexible_name_pattern(name), str(text or ""), flags=re.I))


def _flexible_name_pattern(name: str) -> str:
    parts = [_name_word_pattern(part) for part in str(name or "").split() if part]
    if not parts:
        return r"(?!)"
    return r"\b" + r"\s+".join(parts) + r"\b"


def _name_word_pattern(word: str) -> str:
    text = str(word or "").strip()
    if not text:
        return r"(?!)"
    normalized = text.casefold().replace("ё", "е")
    if normalized == "петр":
        return r"п[её]тр(?:а|у|ом|е)?"
    escaped = re.escape(text).replace("ё", "[её]").replace("Ё", "[ЕЁ]")
    if re.search(r"[бвгджзклмнпрстфхцчшщ]$", normalized, re.I):
        return escaped + r"(?:а|у|ом|е)?"
    if normalized.endswith("й"):
        return re.escape(text[:-1]).replace("ё", "[её]").replace("Ё", "[ЕЁ]") + r"(?:й|я|ю|ем|е)"
    if normalized.endswith(("а", "я")) and len(normalized) > 3:
        stem = re.escape(text[:-1]).replace("ё", "[её]").replace("Ё", "[ЕЁ]")
        return stem + r"(?:а|я|ы|и|е|у|ю|ой|ей)?"
    return escaped


def _sanitize_raw_detail_handoff_text(text: str) -> tuple[str, bool]:
    changed = False

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        replacement = _sanitize_raw_detail_handoff_match(match)
        if replacement != match.group(0):
            changed = True
        return replacement

    return OUTPUT_SANITIZER_RAW_DETAIL_HANDOFF_RE.sub(repl, str(text or "")), changed


def _sanitize_raw_detail_handoff_match(match: re.Match[str]) -> str:
    detail = next((str(item or "") for item in match.groups() if item), "")
    if not _raw_detail_handoff_looks_like_question(detail):
        return match.group(0)
    return SAFE_FALLBACK_DRAFT_TEXT


def _raw_detail_handoff_looks_like_question(detail: str) -> bool:
    value = " ".join(str(detail or "").split())
    low = value.casefold().replace("ё", "е")
    if len(value) >= 55:
        return True
    return bool(
        re.search(
            r"\b(?:сможет|можно|есть|будет|получится|подойдет|подойд[её]т|оценить|сколько|когда|как|где)\s+ли\b|"
            r"\b(?:сын|дочк|дочь|реб[её]н|школьник|ученик)\b|\?$",
            low,
            re.I,
        )
    )


def _output_sanitizer_degenerate(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if OUTPUT_SANITIZER_META_LINE_RE.search(value) or OUTPUT_SANITIZER_MANAGER_TAG_RE.search(value):
        return True
    if re.fullmatch(r"(?:[A-CА-В][).]\s*[^.?!\n]{1,120}\s*)+", value, flags=re.I):
        return True
    if not re.search(r"[а-яёa-z]", value, flags=re.I):
        return True
    return False


def _authoritative_gate_action(code: str) -> str:
    return str(GATE_BLOCKING_CODES.get(str(code or ""), "warn") or "warn")


def _authoritative_gate_downgraded_route(route: str, actions: Sequence[str]) -> str:
    current = str(route or "manager_only")
    if "block" in set(actions):
        return "manager_only"
    if current in AUTONOMOUS_ROUTES:
        return "draft_for_manager"
    return current


def _authoritative_gate_finding(code: str, *, detail: str = "", source: str = "", **extra: str) -> dict[str, str]:
    finding = {
        "code": str(code or "").strip(),
        "detail": " ".join(str(detail or "").split())[:240],
        "source": str(source or "authoritative_output_gate").strip(),
        "policy": _authoritative_gate_action(code),
    }
    for key, value in extra.items():
        normalized = " ".join(str(value or "").split())[:240]
        if normalized:
            finding[str(key)] = normalized
    return finding


def _authoritative_gate_findings(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    text_only = not client_message and context is None and not _pipeline_fact_texts(result)

    findings.extend(_authoritative_gate_text_guard_findings(result))
    findings.extend(_authoritative_gate_a2_findings(result, client_message=client_message, context=context))
    findings.extend(_authoritative_gate_semantic_output_findings(result))
    if text_only:
        return _dedupe_gate_findings(findings)

    gate_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    facts = _authoritative_gate_fact_texts(result, gate_context)
    findings.extend(
        _authoritative_gate_derived_product_number_findings(
            result,
            client_message=client_message,
            context=gate_context,
            facts=facts,
        )
    )
    findings.extend(_authoritative_gate_unsafe_future_commitment_findings(result))
    contract = _pipeline_contract(result, active_brand=_active_brand(gate_context), fact_keys=tuple(facts.keys()))
    previous_bot_texts = _humanity_previous_bot_texts(gate_context)
    p0_already_guarded = _authoritative_gate_p0_already_guarded(result)
    model_p0 = result.metadata.get("direct_path_model_p0") if isinstance(result.metadata, Mapping) else {}
    if (
        not p0_already_guarded
        and _direct_path_model_p0_enabled(context)
        and isinstance(model_p0, Mapping)
        and model_p0.get("is_p0") is True
        and model_p0.get("route_applied") is True
    ):
        findings.append(
            _authoritative_gate_finding(
                "hard_p0",
                detail=str(model_p0.get("p0_kind") or "model_p0"),
                source="direct_path_model_p0",
            )
        )
    has_pipeline = _authoritative_gate_has_pipeline(result)
    semantic_verifier = result.metadata.get("semantic_output_verifier") if isinstance(result.metadata, Mapping) else {}
    semantic_relevance_checked = bool(
        isinstance(semantic_verifier, Mapping)
        and semantic_verifier.get("checked") is True
        and not semantic_verifier.get("skipped")
        and not semantic_verifier.get("unavailable")
    )
    for finding in verify_dialogue_contract_output(
        result.draft_text,
        facts=facts,
        active_brand=_active_brand(gate_context),
        contract=contract,
        client_message=client_message,
        context=gate_context,
        previous_bot_texts=previous_bot_texts,
    ):
        if not has_pipeline and finding.code not in {"brand_leak", "meta_leak", "ai_disclosure", "p0_promise"}:
            continue
        if finding.code == "wrong_intent_fact" and semantic_relevance_checked:
            continue
        if finding.code == "p0_promise" and _authoritative_gate_verified_content_flag(result):
            continue
        if p0_already_guarded and finding.code == "p0_promise":
            continue
        if _authoritative_gate_skip_backed_finding(
            finding.code,
            detail=finding.detail,
            result=result,
            client_message=client_message,
            facts=facts,
        ):
            continue
        findings.append(_authoritative_gate_finding(finding.code, detail=finding.detail, source="verify_output"))

    safety = classify_answer_safety(
        client_message=client_message,
        context=gate_context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
        include_text_signals=not _p0_model_led_enabled(gate_context),
    )
    safety_from_latch = "p0_latch" in (safety.evidence or {})
    safety_authoritative = not _p0_model_led_enabled(gate_context) or safety_from_latch
    hard_codes = tuple(code for code in safety.risk_codes if code in {"refund", "legal", "complaint", "payment_dispute"}) if safety_authoritative else ()
    safety_p0_blocking = bool(safety_authoritative and safety.p0_required and not safety.semantic_non_p0)
    if not p0_already_guarded and (hard_codes or safety_p0_blocking):
        detail = ",".join(dict.fromkeys(hard_codes))
        findings.append(_authoritative_gate_finding("hard_p0", detail=detail or safety.primary_risk, source="answer_safety"))
    if safety_authoritative and safety.zero_collect_required and not p0_already_guarded and (safety.p0_required or hard_codes):
        findings.append(_authoritative_gate_finding("zero_collect_required", detail=safety.primary_risk, source="answer_safety"))

    findings.extend(
        _authoritative_gate_existing_guard_findings(
            result,
            client_message=client_message,
            context=gate_context,
            facts=facts,
        )
    )
    return _dedupe_gate_findings(findings)


def _authoritative_gate_text_guard_findings(result: SubscriptionDraftResult) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    guarded = guard_identity_disclosure(result)
    if guarded is not result and guarded.draft_text != result.draft_text:
        findings.append(_authoritative_gate_finding("identity_disclosure", source="guard_identity_disclosure"))
    guarded = guard_draft_placeholder(result)
    if guarded is not result and guarded.draft_text != result.draft_text:
        findings.append(_authoritative_gate_finding("draft_placeholder", source="guard_draft_placeholder"))
    guarded = guard_promocode_leak(result)
    if guarded is not result and guarded.draft_text != result.draft_text:
        findings.append(_authoritative_gate_finding("promocode_leak", source="guard_promocode_leak"))
    manager_deadline = _manager_deadline_promise_detail(result.draft_text)
    if manager_deadline:
        findings.append(
            _authoritative_gate_finding(
                "unsupported_manager_deadline_promise",
                detail=manager_deadline,
                source="manager_deadline_promise_guard",
            )
        )
    return findings


def _manager_deadline_promise_detail(text: str) -> str:
    for sentence in re.split(r"(?<=[.?!])\s+|\n+", str(text or "")):
        value = " ".join(sentence.split())
        if not value:
            continue
        if (
            MANAGER_ACTION_PROMISE_ACTOR_RE.search(value)
            and MANAGER_ACTION_PROMISE_ACTION_RE.search(value)
            and MANAGER_ACTION_PROMISE_DEADLINE_RE.search(value)
        ):
            return value[:240]
    return ""


def _authoritative_gate_unsafe_future_commitment_findings(result: SubscriptionDraftResult) -> list[dict[str, str]]:
    details = _unsafe_future_commitment_details(result.draft_text)
    return [
        _authoritative_gate_finding(
            "unsafe_future_commitment",
            detail=detail,
            source="draft_future_commitment_gate",
        )
        for detail in details
    ]


def _unsafe_future_commitment_details(text: str) -> tuple[str, ...]:
    details: list[str] = []
    for sentence in _guard_sentences(text):
        if not UNSAFE_FUTURE_COMMITMENT_RE.search(sentence):
            continue
        if SAFE_FUTURE_COMMITMENT_CONTEXT_RE.search(sentence):
            continue
        details.append(sentence[:240])
    return tuple(dict.fromkeys(details))


def _guard_sentences(text: str) -> tuple[str, ...]:
    return tuple(
        " ".join(part.split())
        for part in re.split(r"(?<=[.?!])\s+|\n+", str(text or ""))
        if part.strip()
    )


def _authoritative_gate_a2_findings(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    text = str(result.draft_text or "")
    proactive_active = _a2_proactive_enabled(context) or _a2_is_proactive_result(result)
    if proactive_active:
        if _A2_FAKE_DONE_RE.search(text):
            findings.append(_authoritative_gate_finding("fake_enrollment_claim", source="a2_proactive_gate"))
        phone = _a2_extract_phone(client_message)
        if phone and _a2_phone_echoed(phone, text):
            findings.append(_authoritative_gate_finding("proactive_pii_echo", source="a2_proactive_gate"))
        if _a2_is_proactive_result(result) and text.count("?") > 1:
            findings.append(
                _authoritative_gate_finding("proactive_too_many_questions", detail="more_than_one_question", source="a2_proactive_gate")
            )
    if _a2_rich_format_enabled(context):
        context_tag = _a2_context_tag(result, client_message=client_message, context=context)
        cleaned = _a2_enforce_emoji_limit(text, context_tag=context_tag)
        if cleaned != text:
            findings.append(_authoritative_gate_finding("proactive_emoji_overuse", detail="emoji_guard_not_applied", source="a2_rich_format_gate"))
    return findings


def _authoritative_gate_semantic_output_findings(result: SubscriptionDraftResult) -> list[dict[str, str]]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    verifier = metadata.get("semantic_output_verifier") if isinstance(metadata.get("semantic_output_verifier"), Mapping) else {}
    raw_findings = verifier.get("findings") if isinstance(verifier, Mapping) else ()
    if not isinstance(raw_findings, Sequence) or isinstance(raw_findings, (str, bytes, bytearray)):
        return []
    findings: list[dict[str, str]] = []
    for raw in raw_findings:
        if not isinstance(raw, Mapping):
            continue
        code = str(raw.get("code") or "").strip()
        if code not in _SEMANTIC_OUTPUT_VERIFIER_CODES:
            continue
        detail = _semantic_output_finding_detail(raw)
        findings.append(
            _authoritative_gate_finding(
                code,
                detail=detail,
                source="semantic_output_verifier",
                relation_to_base=str(raw.get("relation_to_base") or ""),
                nearest_fact_key=str(raw.get("nearest_fact_key") or ""),
                evidence=str(raw.get("evidence") or ""),
                missing_fact=str(raw.get("missing_fact") or ""),
            )
        )
    return findings


def _semantic_output_finding_detail(item: Mapping[str, Any]) -> str:
    parts = [
        str(item.get("span") or "").strip(),
        str(item.get("relation_to_base") or "").strip(),
        str(item.get("nearest_fact_key") or "").strip(),
        str(item.get("missing_fact") or "").strip(),
        str(item.get("evidence") or "").strip(),
    ]
    return " | ".join(part for part in parts if part)[:240]


def _semantic_output_manager_note(findings: Sequence[Mapping[str, Any]]) -> str:
    semantic = [item for item in findings if str(item.get("source") or "") == "semantic_output_verifier" or str(item.get("code") or "") in _SEMANTIC_OUTPUT_VERIFIER_CODES]
    if not semantic:
        return "Смысловой верификатор: проверить черновик перед отправкой."
    samples: list[str] = []
    for item in semantic[:2]:
        code = str(item.get("code") or "")
        relation = str(item.get("relation_to_base") or "")
        nearest = str(item.get("nearest_fact_key") or "")
        span = str(item.get("span") or item.get("detail") or "").strip()
        if relation == "contradicts" and nearest:
            samples.append(f"{code}: противоречит факту {nearest} ({span})")
        elif relation == "adjacent" and nearest:
            samples.append(f"{code}: рядом с фактом {nearest}, но не подтверждено ({span})")
        else:
            samples.append(f"{code}: в базе нет подтверждения ({span})")
    suffix = f"; и ещё {len(semantic) - 2}" if len(semantic) > 2 else ""
    return "Смысловой верификатор: " + "; ".join(samples)[:200] + suffix


def _derived_product_number_manager_notes(findings: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    notes: list[str] = []
    for item in findings:
        if str(item.get("code") or "") != "derived_product_number":
            continue
        span = str(item.get("span") or item.get("detail") or "").strip()
        if not span:
            continue
        notes.append(f"Проверьте {span} — вычислено ботом, в прайсе нет.")
    return tuple(dict.fromkeys(notes))


def _a2_is_proactive_result(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    a2 = metadata.get("a2_proactive") if isinstance(metadata.get("a2_proactive"), Mapping) else {}
    selling = metadata.get("selling") if isinstance(metadata.get("selling"), Mapping) else {}
    rules = metadata.get("rules_engine") if isinstance(metadata.get("rules_engine"), Mapping) else {}
    rules_selling = rules.get("selling") if isinstance(rules.get("selling"), Mapping) else {}
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    return bool(
        a2.get("step")
        or selling.get("proactive")
        or rules_selling.get("proactive")
        or "a2_proactive" in flags
        or "offer_callback" in flags
    )


def _a2_phone_echoed(phone: str, text: str) -> bool:
    digits = re.sub(r"\D+", "", str(phone or ""))
    if len(digits) < 7:
        return False
    haystack = re.sub(r"\D+", "", str(text or ""))
    return bool(haystack and digits in haystack)


def _authoritative_gate_existing_guard_findings(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
    facts: Mapping[str, str],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    guard_checks: tuple[tuple[str, str, Callable[[SubscriptionDraftResult], SubscriptionDraftResult]], ...] = (
        ("unsupported_promise", "apply_unsupported_promise_guard", lambda item: apply_unsupported_promise_guard(item, context=context)),
        (
            "unconfirmed_operational_specificity",
            "apply_unconfirmed_operational_specificity_guard",
            lambda item: apply_unconfirmed_operational_specificity_guard(item, context=context),
        ),
    )
    for code, source, guard_fn in guard_checks:
        if code == "unsupported_promise" and _authoritative_gate_verified_content_flag(result):
            continue
        guarded = guard_fn(result)
        if _authoritative_guard_changed(result, guarded):
            added_flags = sorted(set(guarded.safety_flags) - set(result.safety_flags))
            detail = ",".join(added_flags) or guarded.error or guarded.route
            if _authoritative_gate_skip_backed_finding(
                code,
                detail=detail,
                result=result,
                client_message=client_message,
                facts=facts,
            ):
                continue
            findings.append(_authoritative_gate_finding(code, detail=detail, source=source))
    specificity_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    for code, fn in (
        ("unsupported_followup_deadline", find_unsupported_followup_deadline_claims),
        ("unsupported_schedule_assumption", find_unsupported_schedule_assumption_claims),
        ("unsupported_offline_visit_invitation", find_unsupported_offline_visit_invitation_claims),
        ("unsupported_content_delivery_action", find_unsupported_content_delivery_action_claims),
    ):
        claims = fn(result.draft_text, context=specificity_context)
        if claims:
            if _authoritative_gate_skip_backed_finding(
                code,
                detail="; ".join(claims),
                result=result,
                client_message=client_message,
                facts=facts,
            ):
                continue
            findings.append(_authoritative_gate_finding(code, detail="; ".join(claims), source=fn.__name__))
    return findings


def _authoritative_gate_derived_product_number_findings(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
    facts: Mapping[str, str],
) -> list[dict[str, str]]:
    if _authoritative_gate_verified_content_flag(result):
        return []
    draft_claims = _derived_product_number_claims(result.draft_text)
    if not draft_claims:
        return []
    fact_surfaces = {
        normalized
        for value in facts.values()
        for _span, normalized in _derived_product_number_claims(str(value or ""))
    }
    client_context = _client_pii_echo_context(client_message=client_message, context=context)
    client_surfaces = {normalized for _span, normalized in _derived_product_number_claims(client_context)}
    findings: list[dict[str, str]] = []
    seen: set[str] = set()
    for span, normalized in draft_claims:
        if normalized in fact_surfaces or normalized in client_surfaces or normalized in seen:
            continue
        seen.add(normalized)
        findings.append(
            _authoritative_gate_finding(
                "derived_product_number",
                detail=span,
                source="derived_product_number_gate",
                span=span,
                evidence=normalized,
            )
        )
    return findings


def _derived_product_number_claims(text: str) -> tuple[tuple[str, str], ...]:
    claims: list[tuple[str, str]] = []
    for match in DERIVED_PRODUCT_NUMBER_RE.finditer(str(text or "")):
        span = " ".join(match.group(0).replace("\u00a0", " ").split())
        if not span:
            continue
        is_percent = bool(re.search(r"%|процент", span, flags=re.I))
        numbers = re.findall(r"\d+(?:[.,]\d+)?", span)
        if is_percent:
            for raw in numbers:
                normalized = _normalize_derived_number_surface(raw)
                if normalized:
                    claims.append((span, f"{normalized}%"))
            continue
        normalized = _normalize_derived_number_surface("".join(numbers))
        if normalized:
            claims.append((span, normalized))
    return tuple(claims)


def _normalize_derived_number_surface(value: str) -> str:
    normalized = str(value or "").replace("\u00a0", " ").replace(" ", "").replace(",", ".").strip()
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _authoritative_guard_changed(before: SubscriptionDraftResult, after: SubscriptionDraftResult) -> bool:
    return (
        before.route != after.route
        or before.draft_text != after.draft_text
        or set(after.safety_flags) != set(before.safety_flags)
        or set(after.forbidden_promises_detected) != set(before.forbidden_promises_detected)
    )


def _authoritative_gate_fact_texts(
    result: SubscriptionDraftResult,
    context: Optional[Mapping[str, Any]],
) -> dict[str, str]:
    facts = dict(_pipeline_fact_texts(result))
    if facts:
        return facts
    if isinstance(context, Mapping):
        confirmed = context.get("confirmed_facts")
        if isinstance(confirmed, Mapping):
            facts.update({str(key): str(value) for key, value in confirmed.items() if str(key).strip() and str(value).strip()})
        facts_context = context.get("facts_context")
        if isinstance(facts_context, Mapping):
            confirmed_context = facts_context.get("confirmed_facts")
            if isinstance(confirmed_context, Mapping):
                facts.update(
                    {str(key): str(value) for key, value in confirmed_context.items() if str(key).strip() and str(value).strip()}
                )
        known_slots = context.get("known_slots")
        if isinstance(known_slots, Mapping):
            for key, value in known_slots.items():
                text = _authoritative_gate_slot_text(str(key), value)
                if text:
                    facts[f"_known_slot:{key}"] = text
    return facts


def _authoritative_gate_skip_backed_finding(
    code: str,
    *,
    detail: str = "",
    result: SubscriptionDraftResult,
    client_message: str,
    facts: Mapping[str, str],
) -> bool:
    code_text = str(code or "")
    combined = " ".join([str(detail or ""), str(result.draft_text or ""), str(client_message or "")]).casefold().replace("ё", "е")
    fact_text = " ".join(str(value or "") for value in facts.values()).casefold().replace("ё", "е")
    if code_text in {
        "unconfirmed_operational_specificity",
        "unsupported_schedule_assumption",
    }:
        schedule_markers = ("выходн", "суббот", "воскрес", "будн", "вечер", "утрен", "дневн")
        return any(marker in combined and marker in fact_text for marker in schedule_markers)
    if code_text in {"fact_grounding", "unsupported_entity"} and _authoritative_gate_verified_content_flag(result):
        return True
    if code_text == "unsupported_entity" and "address:generic" in str(detail or ""):
        asks_address = has_any_marker(combined, ("адрес", "сретенк", "скорняжн", "москва", "метро", "где находит"))
        has_address_fact = has_any_marker(fact_text, ("адрес", "сретенк", "скорняжн", "москва", "метро", "чистые пруды"))
        return asks_address and has_address_fact
    return False


def _authoritative_gate_verified_content_flag(result: SubscriptionDraftResult) -> bool:
    flags = tuple(str(flag or "") for flag in result.safety_flags)
    if any(flag.endswith("_safe_template_applied") or flag.endswith("_fallback_applied") for flag in flags):
        return True
    return any(
        flag
        in {
            "safe_template_yielded_to_verified_answer",
            "humanity_block_a_direct_answer_applied",
            "cite_only_recover_at_guardchain",
        }
        for flag in flags
    )


def _authoritative_gate_has_pipeline(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    return isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) or isinstance(metadata.get("direct_path"), Mapping)


def _authoritative_gate_slot_text(key: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized_key = str(key or "").strip()
    if normalized_key == "grade" and text.isdigit():
        return f"{text} класс"
    return f"{normalized_key}: {text}" if normalized_key else text


def _authoritative_gate_p0_already_guarded(result: SubscriptionDraftResult) -> bool:
    if result.route != "manager_only":
        return False
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    return bool(
        metadata.get("final_p0_text_override")
        or metadata.get("zero_collect_legal_guarded")
        or metadata.get("zero_collect_refund_guarded")
        or metadata.get("complaint_apology_guarded")
        or metadata.get("payment_dispute_manager_only")
        or any(
            marker in flags
            for marker in (
                "zero_collect_legal_guarded",
                "zero_collect_refund_guarded",
                "complaint_apology_guarded",
                "payment_dispute_manager_only",
            )
        )
    )


def _dedupe_gate_findings(findings: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, str]] = []
    for item in findings:
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        source = str(item.get("source") or "")
        detail = str(item.get("detail") or "")
        key = (code, source, detail)
        if key in seen:
            continue
        seen.add(key)
        compact = {
            "code": code,
            "detail": detail,
            "source": source,
            "policy": _authoritative_gate_action(code),
        }
        for extra_key in ("relation_to_base", "nearest_fact_key", "evidence", "missing_fact", "span"):
            extra_value = str(item.get(extra_key) or "").strip()
            if extra_value:
                compact[extra_key] = extra_value
        result.append(compact)
    return result


def draft_has_internal_service_markers(text: str) -> bool:
    value = str(text or "")
    return bool(
        INTERNAL_SERVICE_MARKER_RE.search(value)
        or INTERNAL_SERVICE_TOKEN_RE.search(value)
        or INTERNAL_SCAFFOLD_PREFIX_RE.search(value)
        or INTERNAL_PROMPT_DIRECTIVE_PREFIX_RE.search(value)
        or INTERNAL_PROMPT_DIRECTIVE_ANYWHERE_RE.search(value)
        or INTERNAL_REGEN_EDIT_COMMENT_RE.search(value)
        or INTERNAL_CLIENT_SAFE_JARGON_RE.search(value)
        or INTERNAL_RUNTIME_LIMIT_JARGON_RE.search(value)
        or INTERNAL_CLIENT_INSTRUCTION_RE.search(value)
        or INTERNAL_MANAGER_DRAFT_RE.search(value)
    )




def find_identity_disclosure_phrases(text: str) -> tuple[str, ...]:
    lowered = str(text or "").casefold()
    return tuple(phrase for phrase in IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES if _identity_phrase_present(lowered, phrase))


def draft_has_identity_disclosure(text: str) -> bool:
    return bool(find_identity_disclosure_phrases(text))


def _identity_phrase_present(lowered_text: str, phrase: str) -> bool:
    value = str(phrase or "").casefold().strip()
    if not value:
        return False
    if value == "gpt":
        pattern = r"(?:chat\s*)?gpt"
    else:
        pattern = r"\s+".join(re.escape(part) for part in value.split())
    return bool(re.search(rf"(?<!\w){pattern}(?!\w)", lowered_text, flags=re.I))


def guard_identity_disclosure(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    phrases = find_identity_disclosure_phrases(result.draft_text)
    if not phrases:
        return result
    metadata = _metadata_with_guarded_original_text(result.metadata, result.draft_text, guard="identity_disclosure")
    return replace(
        result,
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *phrases])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "identity_disclosure_guarded", "bot_identity_disclosure", "llm_fallback"])),
        metadata=metadata,
        error=result.error or "identity_disclosure_guarded",
    )


def guard_draft_placeholder(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    if not DRAFT_PLACEHOLDER_RE.search(result.draft_text):
        return result
    return replace(
        result,
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, "placeholder_in_draft"])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "placeholder_in_draft", "llm_fallback"])),
        manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, "Черновик содержит placeholder: заменить вручную."])),
        error=result.error or "placeholder_in_draft",
    )


def guard_promocode_leak(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    if not PROMOCODE_DRAFT_RE.search(result.draft_text):
        return result
    return replace(
        result,
        route="manager_only",
        draft_text=PROMOCODE_SAFE_TEXT,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, "promocode_in_draft"])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "promocode_in_draft_guarded", "manager_approval_required", "no_auto_send"])),
        manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, "Не повторять промокод клиенту до проверки условий акции."])),
        error=result.error,
    )


def apply_unsupported_promise_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if result.draft_text == UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT:
        trace_event(
            context,
            "apply_unsupported_promise_guard",
            {
                "skipped": "verified_installment_fallback",
                "route": result.route,
            },
        )
        return result
    promise_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    claims = find_unsupported_numeric_promises(result.draft_text, context=promise_context)
    if not claims:
        trace_event(
            context,
            "apply_unsupported_promise_guard",
            {
                "claims": (),
                "route_before": result.route,
                "route_after": result.route,
                "blocked": False,
            },
        )
        return result
    flags = tuple(dict.fromkeys([*result.safety_flags, "unsupported_promise_detected"]))
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "Черновик содержит конкретную цифру, сумму, процент или срок без подтвержденного свежего факта: проверить вручную.",
            ]
        )
    )
    guarded = replace(
        result,
        route="manager_only",
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *claims])),
        safety_flags=flags,
        manager_checklist=checklist,
        metadata={**dict(result.metadata), "unsupported_promises": list(claims)},
    )
    trace_event(
        context,
        "apply_unsupported_promise_guard",
        {
            "claims": claims,
            "route_before": result.route,
            "route_after": guarded.route,
            "blocked": True,
            "safety_flags": guarded.safety_flags,
        },
    )
    return guarded


def apply_unconfirmed_operational_specificity_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    specificity_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    followup_claims = find_unsupported_followup_deadline_claims(result.draft_text, context=specificity_context)
    if followup_claims:
        return _operational_specificity_guarded_result(
            result,
            draft_text=UNSUPPORTED_FOLLOWUP_DEADLINE_SAFE_TEXT,
            flag="unsupported_followup_deadline_detected",
            claims=followup_claims,
            checklist_item="Не называть конкретную дату или срок связи менеджера без подтверждённого факта.",
        )

    schedule_claims = find_unsupported_schedule_assumption_claims(result.draft_text, context=specificity_context)
    if schedule_claims:
        return _operational_specificity_guarded_result(
            result,
            draft_text=UNSUPPORTED_SCHEDULE_ASSUMPTION_SAFE_TEXT,
            flag="unsupported_schedule_assumption_detected",
            claims=schedule_claims,
            checklist_item="Не делать догадки по расписанию без подтверждённого факта.",
        )

    visit_claims = find_unsupported_offline_visit_invitation_claims(result.draft_text, context=specificity_context)
    if visit_claims:
        return _operational_specificity_guarded_result(
            result,
            draft_text=UNSUPPORTED_OFFLINE_VISIT_INVITATION_SAFE_TEXT,
            flag="unsupported_offline_visit_invitation_detected",
            claims=visit_claims,
            checklist_item="Запись и оформление по умолчанию дистанционные; очную встречу не предлагать без согласования.",
        )

    delivery_claims = find_unsupported_content_delivery_action_claims(result.draft_text, context=specificity_context)
    if delivery_claims:
        return _operational_specificity_guarded_result(
            result,
            draft_text=(
                "Фрагмент занятия можно прислать для знакомства, но точный способ доступа — ссылка, запись или регистрация — "
                "нужно подтвердить у менеджера. Передам ему ваш запрос; класс, предмет и онлайн-формат уже вижу."
            ),
            flag="unsupported_content_delivery_action_detected",
            claims=delivery_claims,
            checklist_item="Не обещать от лица бота отправить ссылку/фрагмент/запись без подтверждённого способа доступа.",
            route="draft_for_manager",
        )
    return result


def find_unsupported_followup_deadline_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=FOLLOWUP_DEADLINE_RE, context=context)


def find_unsupported_schedule_assumption_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=SCHEDULE_ASSUMPTION_RE, context=context)


def find_unsupported_offline_visit_invitation_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=OFFLINE_VISIT_INVITATION_RE, context=context)


def find_unsupported_content_delivery_action_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=CONTENT_DELIVERY_ACTION_RE, context=context)


def _unsupported_claims_by_pattern(
    draft_text: str,
    *,
    pattern: re.Pattern[str],
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    source = str(draft_text or "")
    claims = tuple(dict.fromkeys(" ".join(match.group(0).split()) for match in pattern.finditer(source) if match.group(0).strip()))
    if not claims:
        return ()
    fact_texts = _fresh_fact_texts(context)
    return tuple(claim for claim in claims if not _claim_supported_by_facts(claim, fact_texts))


def _operational_specificity_guarded_result(
    result: SubscriptionDraftResult,
    *,
    draft_text: str,
    flag: str,
    claims: Sequence[str],
    checklist_item: str,
    route: str = "manager_only",
) -> SubscriptionDraftResult:
    return replace(
        result,
        route=route,
        draft_text=draft_text,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *claims])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, flag, "manager_approval_required", "no_auto_send"])),
        manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, checklist_item])),
        metadata={**dict(result.metadata), flag: True, "unsupported_operational_claims": list(claims)},
    )


def apply_bot_safe_memory_step_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    try:
        if not _bot_safe_memory_step_guard_enabled(context):
            return result
        statuses = _bot_safe_memory_next_step_statuses(result, context)
        review_statuses = tuple(status for status in statuses if status in BOT_SAFE_MEMORY_REVIEW_NEXT_STEP_STATUSES)
        if not review_statuses:
            return result
        guard_context = _context_with_dialogue_contract_retrieved_facts(context, result)
        hard_claims = tuple(
            dict.fromkeys(
                [
                    *_bot_safe_memory_hard_step_claims(result.draft_text, context=guard_context),
                    *_bot_safe_memory_risky_step_claims(result.draft_text, context=guard_context),
                ]
            )
        )
        soft_claims = _bot_safe_memory_soft_step_claims(result.draft_text, context=guard_context)
        claims = tuple(dict.fromkeys([*hard_claims, *soft_claims]))
        if not claims:
            return result
        metadata = dict(result.metadata)
        metadata["bot_safe_memory_step_guard"] = {
            "applied": True,
            "next_step_statuses": list(statuses),
            "review_statuses": list(review_statuses),
            "claims": list(claims),
            "source": "deterministic_output_guard",
        }
        route = "draft_for_manager" if result.route in AUTONOMOUS_ROUTES else result.route
        return replace(
            result,
            route=route,
            forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *claims])),
            safety_flags=tuple(
                dict.fromkeys([*result.safety_flags, BOT_SAFE_MEMORY_STEP_GUARD_FLAG, "manager_approval_required", "no_auto_send"])
            ),
            manager_checklist=tuple(
                dict.fromkeys(
                    [
                        *result.manager_checklist,
                        "Не утверждать конкретный шаг из памяти: статус next_step требует проверки менеджером.",
                    ]
                )
            ),
            metadata=metadata,
        )
    except Exception:
        return result


UNCONFIRMED_CONTACT_DATA_CLAIM_FLAG = "unconfirmed_contact_data_claim_detected"

CONTACT_DATA_EVIDENCE_RE = re.compile(
    r"(?:\+?\d[\d\s().-]{6,}\d|[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}|"
    r"\b(?:мой|моя|наш|наш[аи]|указал[аи]?|пишу|оставляю)\b[^.?!\n]{0,80}"
    r"\b(?:телефон|номер|почт[ауеы]|email|e-mail|адрес|контакт)\b)",
    re.I,
)

CLIENT_CONTACT_FACT_EVIDENCE_RE = re.compile(
    r"(?:\b(?:клиент|родител[ьяюем]?|мам[аы]?|пап[аы]?|заявител[ьяюем]?|контакт\s+клиент[а-яё]*)\b"
    r"[^.?!\n]{0,120}(?:\+?\d[\d\s().-]{6,}\d|[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}|"
    r"\b(?:телефон|номер|почт[ауеы]|email|e-mail|адрес|контакт)\b)"
    r"|(?:\+?\d[\d\s().-]{6,}\d|[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}|"
    r"\b(?:телефон|номер|почт[ауеы]|email|e-mail|адрес|контакт)\b)"
    r"[^.?!\n]{0,120}\b(?:клиент|родител[ьяюем]?|мам[аы]?|пап[аы]?|заявител[ьяюем]?)\b)",
    re.I,
)

UNCONFIRMED_CONTACT_DATA_CLAIM_RE = re.compile(
    r"(?P<sentence>[^.?!\n]{0,180}(?:"
    r"(?:телефон|номер(?:\s+телефона)?|почт[ауеы]|email|e-mail|адрес|контакт(?:ы|ные\s+данные)?)"
    r"[^.?!\n]{0,180}"
    r"(?:уже\s+(?:есть|вижу|указан[аоы]?)|есть\s+(?:у\s+нас|у\s+центра|в\s+диалоге|в\s+системе)|"
    r"повторно\s+(?:указывать|присылать|отправлять)\s+не\s+нужно|"
    r"(?:указывать|присылать|отправлять)\s+повторно\s+не\s+нужно|"
    r"не\s+нужно\s+(?:повторно\s+)?(?:указывать|присылать|отправлять))"
    r"|"
    r"(?:уже\s+(?:есть|вижу)|есть\s+(?:у\s+нас|у\s+центра|в\s+диалоге|в\s+системе))"
    r"[^.?!\n]{0,180}"
    r"(?:телефон|номер(?:\s+телефона)?|почт[ауеы]|email|e-mail|адрес|контакт(?:ы|ные\s+данные)?)"
    r")[^.?!\n]*(?:[.?!]|$))",
    re.I,
)

NO_MEMORY_STEP_FRAME_GUARD_FLAG = "no_memory_step_frame_detected"

NO_MEMORY_STEP_FRAME_RE = re.compile(
    r"(?P<sentence>[^.?!\n]{0,120}(?:"
    r"следующ(?:ий|им)\s+шаг(?:ом)?\s*(?:[—:-]|\b(?:будет|это)\b)\s*"
    r"|лучше\s+начать\s+с\s+"
    rf"|дальше\s+(?:нужно|по\s+плану)\s+(?=[^.?!\n]{{0,120}}\b{NEXT_STEP_SOFT_ACTION_RE_FRAGMENT}\b)"
    r")"
    r"(?P<body>[^.?!\n]{1,180})(?:[.?!]|$))",
    re.I,
)


def apply_unconfirmed_contact_data_claim_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    claims = find_unconfirmed_contact_data_claims(result.draft_text)
    if not claims or _contact_data_claim_has_evidence(client_message=client_message, context=context):
        return result

    metadata = dict(result.metadata)
    metadata["unconfirmed_contact_data_claim_guard"] = {
        "applied": True,
        "claims": list(claims),
        "source": "deterministic_output_guard",
    }
    route = "draft_for_manager" if result.route in AUTONOMOUS_ROUTES else result.route
    return replace(
        result,
        route=route,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *claims])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, UNCONFIRMED_CONTACT_DATA_CLAIM_FLAG])),
        manager_checklist=tuple(
            dict.fromkeys(
                [
                    *result.manager_checklist,
                    "Не утверждать, что контактные данные уже есть, без client-safe факта или реплики клиента в этом диалоге.",
                ]
            )
        ),
        metadata=metadata,
    )


def find_unconfirmed_contact_data_claims(draft_text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            " ".join(match.group("sentence").split())
            for match in UNCONFIRMED_CONTACT_DATA_CLAIM_RE.finditer(str(draft_text or ""))
            if match.group("sentence").strip()
        )
    )


def _contact_data_claim_has_evidence(
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    evidence_texts: list[str] = []
    if str(client_message or "").strip():
        evidence_texts.append(str(client_message))
    if isinstance(context, Mapping):
        evidence_texts.extend(_client_dialogue_texts_from_context(context))
        if any(CONTACT_DATA_EVIDENCE_RE.search(text) for text in evidence_texts):
            return True
        return any(CLIENT_CONTACT_FACT_EVIDENCE_RE.search(text) for text in _fresh_fact_texts(context))
    return any(CONTACT_DATA_EVIDENCE_RE.search(text) for text in evidence_texts)


def _client_dialogue_texts_from_context(context: Mapping[str, Any]) -> tuple[str, ...]:
    texts: list[str] = []
    for key in ("recent_messages", "conversation", "messages", "dialogue_messages"):
        _append_client_dialogue_texts(texts, context.get(key))
    return tuple(texts)


def _append_client_dialogue_texts(result: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        stripped = value.strip()
        if re.match(r"^(?:клиент|client|user|пользователь|родитель)\s*[:：-]", stripped, re.I):
            result.append(stripped)
        return
    if isinstance(value, Mapping):
        role = str(value.get("role") or value.get("speaker") or value.get("author") or "").strip().casefold()
        if role in {"client", "customer", "user", "parent", "клиент", "родитель", "пользователь"}:
            text = value.get("text") or value.get("content") or value.get("message")
            if str(text or "").strip():
                result.append(str(text))
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _append_client_dialogue_texts(result, item)


def apply_no_memory_step_frame_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    statuses = _bot_safe_memory_next_step_statuses(result, context)
    if "active" in statuses:
        return result
    claims = find_no_memory_step_frame_claims(result.draft_text)
    if not claims:
        return result
    metadata = dict(result.metadata)
    metadata["no_memory_step_frame_guard"] = {
        "applied": True,
        "claims": list(claims),
        "next_step_statuses": list(statuses),
        "source": "deterministic_output_guard",
    }
    return replace(
        result,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, NO_MEMORY_STEP_FRAME_GUARD_FLAG])),
        manager_checklist=tuple(
            dict.fromkeys([*result.manager_checklist, "Не называть уточняющий вопрос «следующим шагом» без active next_step."])
        ),
        metadata=metadata,
    )


def find_no_memory_step_frame_claims(draft_text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            " ".join(match.group("sentence").split())
            for match in NO_MEMORY_STEP_FRAME_RE.finditer(str(draft_text or ""))
            if match.group("sentence").strip()
        )
    )


def find_bot_safe_memory_disputed_step_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *_bot_safe_memory_hard_step_claims(draft_text, context=context),
                *_bot_safe_memory_risky_step_claims(draft_text, context=context),
                *_bot_safe_memory_soft_step_claims(draft_text, context=context),
            ]
        )
    )


def _bot_safe_memory_hard_step_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=BOT_SAFE_MEMORY_CONCRETE_STEP_RE, context=context)


def _bot_safe_memory_risky_step_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return tuple(
        claim
        for claim in _unsupported_claims_by_pattern(
            draft_text,
            pattern=BOT_SAFE_MEMORY_RISKY_NEXT_STEP_FRAME_RE,
            context=context,
        )
        if not _bot_safe_memory_safe_payment_link_step_claim(claim)
    )


def _bot_safe_memory_soft_step_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return tuple(
        claim
        for claim in _unsupported_claims_by_pattern(
            draft_text,
            pattern=BOT_SAFE_MEMORY_SOFT_NEXT_STEP_FRAME_RE,
            context=context,
        )
        if not _bot_safe_memory_safe_payment_link_step_claim(claim)
    )


def _bot_safe_memory_safe_payment_link_step_claim(claim: str) -> bool:
    normalized = str(claim or "").casefold().replace("ё", "е")
    if "оплат" not in normalized or "ссыл" not in normalized:
        return False
    return not re.search(r"\b(?:верн\w*|возврат\w*|деньг\w*|компенс\w*|руб(?:\.|л)|₽|\d{3,})\b", normalized)


def _bot_safe_memory_step_guard_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    crm_enabled = _explicit_truthy_setting(
        context,
        BOT_SAFE_CRM_CONTEXT_ENV,
        aliases=(
            "bot_safe_crm_context",
            "bot_safe_crm_context_enabled",
            "bot_safe_summary_context",
            "bot_safe_summary_context_enabled",
        ),
    )
    guard_enabled = _explicit_truthy_setting(
        context,
        BOT_SAFE_MEMORY_STEP_GUARD_ENV,
        aliases=("bot_safe_memory_step_guard", "bot_safe_memory_step_guard_enabled"),
    )
    return crm_enabled is True and guard_enabled is True


def _bot_safe_memory_next_step_statuses(
    result: SubscriptionDraftResult,
    context: Optional[Mapping[str, Any]],
) -> tuple[str, ...]:
    statuses: list[str] = []

    def add(value: Any) -> None:
        status = str(value or "").strip().casefold()
        if status in BOT_SAFE_MEMORY_VALID_NEXT_STEP_STATUSES and status not in statuses:
            statuses.append(status)

    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    _bot_safe_memory_add_statuses_from_metadata(metadata, add)
    if isinstance(context, Mapping):
        _bot_safe_memory_add_statuses_from_context(context, add)
    return tuple(statuses)


def _bot_safe_memory_add_statuses_from_metadata(metadata: Mapping[str, Any], add: Callable[[Any], None]) -> None:
    direct_path = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    next_step = metadata.get("next_step") if isinstance(metadata.get("next_step"), Mapping) else {}
    add(next_step.get("status"))
    direct_next_step = direct_path.get("next_step") if isinstance(direct_path.get("next_step"), Mapping) else {}
    add(direct_next_step.get("status"))
    for container in (
        metadata.get("bot_safe_crm_context"),
        direct_path.get("bot_safe_crm_context"),
        metadata.get("bot_safe_context"),
        direct_path.get("bot_safe_context"),
    ):
        if not isinstance(container, Mapping):
            continue
        raw_statuses = container.get("next_step_statuses")
        if isinstance(raw_statuses, Sequence) and not isinstance(raw_statuses, (str, bytes, bytearray)):
            for status in raw_statuses:
                add(status)
        add(container.get("next_step_status"))


def _bot_safe_memory_add_statuses_from_context(context: Mapping[str, Any], add: Callable[[Any], None]) -> None:
    _bot_safe_memory_add_statuses_from_bot_context(context.get("bot_context"), add)
    timeline = context.get("timeline_context") if isinstance(context.get("timeline_context"), Mapping) else {}
    _bot_safe_memory_add_statuses_from_bot_context(timeline.get("bot_context"), add)
    customer_context = (
        context.get("read_only_customer_context")
        if isinstance(context.get("read_only_customer_context"), Mapping)
        else {}
    )
    _bot_safe_memory_add_statuses_from_bot_context(customer_context.get("bot_context"), add)
    nested_timeline = (
        customer_context.get("timeline_context")
        if isinstance(customer_context.get("timeline_context"), Mapping)
        else {}
    )
    _bot_safe_memory_add_statuses_from_bot_context(nested_timeline.get("bot_context"), add)


def _bot_safe_memory_add_statuses_from_bot_context(bot_context: Any, add: Callable[[Any], None]) -> None:
    if not isinstance(bot_context, Mapping):
        return
    raw_items = bot_context.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
        return
    for item in raw_items:
        if not isinstance(item, Mapping):
            continue
        chunk_type = str(item.get("chunk_type") or "").strip()
        source_system = str(item.get("source_system") or "").strip()
        if chunk_type not in {"bot_safe_summary", "email_message", "channel_message", "mango_call_summary"}:
            continue
        if chunk_type == "email_message" and source_system != "mail_archive_stage2":
            continue
        if chunk_type == "channel_message" and source_system not in {"telegram_history", "wappi_telegram", "wappi_max"}:
            continue
        if chunk_type == "mango_call_summary" and source_system != "mango_processed_summary":
            continue
        if item.get("allowed_for_bot") is not True:
            continue
        if item.get("requires_manager_review") is True:
            continue
        add(_bot_safe_memory_item_next_step_status(item) or "empty")


def _bot_safe_memory_item_next_step_status(item: Mapping[str, Any]) -> str:
    try:
        from mango_mvp.channels.subscription_llm_parts.direct_path import _direct_path_bot_safe_next_step_status

        status = _direct_path_bot_safe_next_step_status(item)
        if status:
            return status
    except Exception:
        pass
    status = str(item.get("next_step_status") or "").strip().casefold()
    if not status:
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            next_step = metadata.get("next_step")
            if isinstance(next_step, Mapping):
                status = str(next_step.get("status") or "").strip().casefold()
    return status if status in BOT_SAFE_MEMORY_VALID_NEXT_STEP_STATUSES else ""








_SEMANTIC_OUTPUT_VERIFIER_CODES = frozenset(
    {
        "derived_product_claim",
        "invented_generalization",
        "individual_diagnosis",
        "irrelevant_to_question",
        "p0_money_promise",
    }
)


_SEMANTIC_VERIFIER_FOTON_OFFLINE_SEMESTER_FACT_KEY = "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.semester"
_SEMANTIC_VERIFIER_FOTON_OFFLINE_YEAR_FACT_KEY = "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.year"


def _semantic_output_verifier_price_scope_few_shot(context: Optional[Mapping[str, Any]] = None) -> str:
    snapshot = _direct_path_load_snapshot(_direct_path_snapshot_path_from_context(context))
    semester_fact = _direct_path_fact_by_brand_key(
        snapshot,
        active_brand="foton",
        fact_key=_SEMANTIC_VERIFIER_FOTON_OFFLINE_SEMESTER_FACT_KEY,
    )
    year_fact = _direct_path_fact_by_brand_key(
        snapshot,
        active_brand="foton",
        fact_key=_SEMANTIC_VERIFIER_FOTON_OFFLINE_YEAR_FACT_KEY,
    )
    if not isinstance(semester_fact, Mapping) or not isinstance(year_fact, Mapping):
        return (
            "- Факты: «Фотон: очная цена подтверждена только для очного формата; онлайн-цена не указана». "
            "Вопрос: «а онлайн?». Ответ переносит очную цену в онлайн-контекст. "
            "Вердикт: derived_product_claim, relation_to_base=adjacent — цена очного формата не подтверждает онлайн-контекст.\n"
        )
    semester_text = _direct_path_snapshot_fact_text(semester_fact)
    year_text = _direct_path_snapshot_fact_text(year_fact)
    semester_price = _direct_path_fact_value(semester_text)
    year_price = _direct_path_fact_value(year_text)
    if not semester_text or not year_text or not semester_price or not year_price:
        return (
            "- Факты: «Фотон: очная цена подтверждена только для очного формата; онлайн-цена не указана». "
            "Вопрос: «а онлайн?». Ответ переносит очную цену в онлайн-контекст. "
            "Вердикт: derived_product_claim, relation_to_base=adjacent — цена очного формата не подтверждает онлайн-контекст.\n"
        )
    return (
        f"- Факты: «{semester_text}» / «{year_text}». "
        f"Вопрос: «а онлайн?». Ответ: «Стоимость курса — {semester_price} или {year_price}». "
        "Вердикт: derived_product_claim, relation_to_base=adjacent — цена очного формата не подтверждает онлайн-контекст.\n"
    )


def build_semantic_output_verifier_prompt(
    *,
    bot_text: str,
    client_message: str = "",
    facts: Mapping[str, str] | None = None,
    active_brand: str = "",
    route: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    facts_block = "\n".join(f"- {key}: {value}" for key, value in (facts or {}).items()) or "(фактов нет)"
    price_scope_few_shot = _semantic_output_verifier_price_scope_few_shot(context)
    return (
        "Ты — смысловой верификатор финального текста бота учебного центра. "
        "Проверяй только смысловые производные, которые плохо ловятся регулярными правилами. "
        "Не проверяй цены/проценты/бренд/мета и входящий P0: это делает отдельный детерминированный gate. "
        "Единственное выходное P0-правило здесь — обещание денег самим ботом.\n\n"
        "Верни СТРОГО JSON:\n"
        '{"findings":[{"code":"derived_product_claim|invented_generalization|individual_diagnosis|irrelevant_to_question|p0_money_promise",'
        '"span":"цитата из ответа","evidence":"почему это риск","missing_fact":"какого факта не хватает",'
        '"relation_to_base":"contradicts|absent|adjacent","nearest_fact_key":"fact.key или пусто"}]}\n'
        'Если нарушений нет: {"findings":[]}.\n\n'
        "КЛАССЫ:\n"
        "- derived_product_claim: продукту/курсу/процессу приписано то, чего нет в фактах: назначение группы, "
        "уровень курса, порядок записи/оплаты, состав программы, материалы, размер группы, что у нас принято.\n"
        "- invented_generalization: обобщение или соцдоказательство как опора рекомендации клиенту: "
        "«обычно», «большинство», «за год-два», если это не дано в фактах.\n"
        "- individual_diagnosis: бот оценивает конкретного ребёнка: справится/потянет/подойдёт/сможет влиться, "
        "«слишком тяжело быть не должно», «посильный ритм», «подберут под ребёнка» — без хеджа и передачи "
        "менеджеру/преподавателю.\n"
        "- irrelevant_to_question: финальный текст отвечает на другой вопрос или добавляет факт, который не нужен "
        "для прямого вопроса клиента и не является кратким уточнением либо безопасным следующим шагом.\n"
        "- p0_money_promise: бот от лица центра обещает вернуть, возместить, компенсировать, пересчитать в пользу "
        "клиента, отдать оплату или перевести деньги обратно. Это обязательство центра, а не описание порядка.\n\n"
        "НЕ ФЛАГАЙ:\n"
        "- дословный или смысловой пересказ факта;\n"
        "- склейку двух реальных фактов без новой приписки;\n"
        "- каноничную фразу разделения брендов;\n"
        "- общий житейский совет с хеджем, если он не делает продуктовый вывод;\n"
        "- хеджированный ответ по ребёнку с передачей преподавателю/менеджеру;\n"
        "- описание порядка возврата, ссылка на договор или передача денежного вопроса менеджеру;\n"
        "- «вернёмся к вопросу/занятиям/обсуждению» без обещания денег;\n"
        "- сервисное предложение или следующий шаг без новой продуктовой приписки: «Помогу с оформлением», "
        "«помогу записаться к старту», «менеджер сверит/свяжется/проверит наличие мест», "
        "«подберём подходящий вариант/группу»;\n"
        "- «подберём подходящий вариант/группу» без оценки конкретного ребёнка — это НЕ individual_diagnosis.\n"
        "ФЛАГАЙ сервисный шаг только если он обещает конкретный неподтверждённый процесс: сроки, условия, документы, "
        "зачисление/бронь/место, или гарантию результата.\n\n"
        "FEW-SHOT КАЛИБРОВКА:\n"
        "- Факт: «Фотон: курс физики есть онлайн и очно». Ответ: «Очный курс физики есть». "
        'Вердикт: {"findings":[]} — это смысловой пересказ подтверждённого факта, НЕ derived_product_claim.\n'
        "- Факт: «УНПК: олимпиадная физика доступна онлайн и очно». Ответ: «Олимпиадная физика есть онлайн и очно». "
        'Вердикт: {"findings":[]} — это перефраз факта без новой приписки.\n'
        "- Ответ: «Помогу с оформлением» / «помогу записаться к старту» / "
        "«менеджер сверит наличие мест» / «подберём подходящий вариант». "
        'Вердикт: {"findings":[]} — это сервисный следующий шаг, не продуктовый claim и не diagnosis.\n'
        f"{price_scope_few_shot}"
        "- Факт: «Фотон: оформление проходит дистанционно, менеджер помогает с договором». "
        "Ответ: «После оплаты по оферте запись считается подтверждённой». "
        "Вердикт: derived_product_claim, relation_to_base=adjacent — похожий факт есть, но порядок записи не подтверждён.\n"
        "- Факт: «Фотон: очная площадка — Верхняя Красносельская». Ответ: «Забронирую место на Сретенке». "
        "Вердикт: derived_product_claim, relation_to_base=contradicts — локация противоречит факту.\n\n"
        "relation_to_base: contradicts = противоречит факту; absent = в базе нет такого факта; "
        "adjacent = похожий факт есть, но он не подтверждает этот вывод. Для adjacent укажи nearest_fact_key.\n\n"
        f"active_brand: {active_brand}\n"
        f"route: {route}\n"
        f"Факты:\n{facts_block}\n\n"
        f"Вопрос клиента:\n{str(client_message or '').strip()}\n\n"
        f"Финальный текст бота:\n{str(bot_text or '').strip()}\n"
    )


def build_semantic_output_regen_prompt(
    *,
    bot_text: str,
    client_message: str,
    facts: Mapping[str, str],
    findings: Sequence[Mapping[str, Any]],
) -> str:
    findings_block = "\n".join(
        f"- {item.get('code')}: {item.get('span') or item.get('evidence') or item.get('missing_fact')}"
        for item in findings
        if isinstance(item, Mapping)
    )
    facts_block = "\n".join(f"- {key}: {value}" for key, value in facts.items()) or "(фактов нет)"
    return (
        "Перепиши текст бота для менеджерского черновика: убери или захеджируй только указанные смысловые риски. "
        "Не добавляй новых фактов, чисел, брендов, обещаний и внутренних комментариев. "
        "Верни ТОЛЬКО текст ответа клиенту, без Markdown, без пояснений и без комментариев о правках. "
        "Не пиши фразы вроде «Заменяю только этот абзац», «Остальной текст без изменений», "
        "«переписываю фрагмент».\n\n"
        f"Вопрос клиента:\n{client_message}\n\n"
        f"Факты:\n{facts_block}\n\n"
        f"Риски:\n{findings_block}\n\n"
        f"Исходный текст:\n{bot_text}\n"
    )


def apply_semantic_output_verifier(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    verifier_fn: Optional[Callable[[str], object]] = None,
    regen_fn: Optional[Callable[[str], object]] = None,
) -> SubscriptionDraftResult:
    if not _semantic_output_verifier_enabled(context):
        return result
    metadata = dict(result.metadata)
    verifier_meta: dict[str, Any] = {
        "schema_version": SEMANTIC_OUTPUT_VERIFIER_SCHEMA_VERSION,
        "enabled": True,
        "checked": False,
        "skipped": False,
        "findings": [],
        "route_before": result.route,
        "route_after": result.route,
        "regen_attempted": False,
        "fallback_reason": "",
    }
    metadata["semantic_output_verifier"] = verifier_meta

    if _semantic_diagnosis_locked_deferral(result, client_message=client_message):
        verifier_meta["skipped"] = True
        verifier_meta["skip_reason"] = "locked_p0_or_high_risk_deferral"
        return replace(result, metadata=metadata)
    handoff_claim_text = dialogue_contract_handoff_factual_claim_text(result.draft_text)
    pure_handoff = dialogue_contract_is_pure_handoff_text(result.draft_text)
    if pure_handoff and result.route not in AUTONOMOUS_ROUTES and not handoff_claim_text and (
        not _verifier_handoff_claims_enabled(context) or _semantic_verifier_is_whitelisted_pure_handoff(result.draft_text)
    ):
        verifier_meta["skipped"] = True
        verifier_meta["skip_reason"] = "pure_handoff"
        return replace(result, metadata=metadata)

    gate_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    facts = _authoritative_gate_fact_texts(result, gate_context)
    verifier = _semantic_output_verifier_override(context) or verifier_fn
    if verifier is None:
        verifier_meta.update({"unavailable": True, "fallback_reason": SEMANTIC_VERIFIER_UNAVAILABLE_REASON})
        return replace(
            result,
            manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, "Смысловой верификатор недоступен: проверить черновик вручную."])),
            metadata=metadata,
        )

    findings, unavailable_reason = _run_semantic_output_verifier_once(
        verifier,
        result.draft_text,
        client_message=client_message,
        facts=facts,
        active_brand=_active_brand(gate_context),
        route=result.route,
        context=gate_context,
    )
    if unavailable_reason and _semantic_output_verifier_max_attempts() > 1:
        verifier_meta["retry_attempted"] = True
        findings, unavailable_reason = _run_semantic_output_verifier_once(
            verifier,
            result.draft_text,
            client_message=client_message,
            facts=facts,
            active_brand=_active_brand(gate_context),
            route=result.route,
            context=gate_context,
        )
    verifier_meta["checked"] = unavailable_reason == ""
    if unavailable_reason:
        verifier_meta.update({"unavailable": True, "fallback_reason": SEMANTIC_VERIFIER_UNAVAILABLE_REASON, "error": unavailable_reason})
        return replace(
            result,
            manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, "Смысловой верификатор недоступен: проверить черновик вручную."])),
            metadata=metadata,
        )

    findings = _semantic_output_filter_findings(findings, result.draft_text)
    verifier_meta["findings"] = list(findings)
    verifier_meta["finding_codes"] = [str(item.get("code") or "") for item in findings]
    verifier_meta["action"] = _semantic_output_verifier_highest_action(findings)
    if not findings:
        verifier_meta["fallback_reason"] = "ok"
        return replace(result, metadata=metadata)

    needs_regen = any(str(item.get("action") or "") == "downgrade_keep_text" for item in findings)
    if needs_regen and regen_fn is not None:
        verifier_meta["regen_attempted"] = True
        try:
            regen_text = str(
                regen_fn(
                    build_semantic_output_regen_prompt(
                        bot_text=result.draft_text,
                        client_message=client_message,
                        facts=facts,
                        findings=findings,
                    )
                )
                or ""
            ).strip()
        except Exception as exc:  # noqa: BLE001
            verifier_meta["regen_error"] = str(exc)[:200]
            return replace(result, metadata=metadata)
        if regen_text:
            regen_findings, regen_unavailable = _run_semantic_output_verifier_once(
                verifier,
                regen_text,
                client_message=client_message,
                facts=facts,
                active_brand=_active_brand(gate_context),
                route=result.route,
                context=gate_context,
            )
            verifier_meta["regen_checked"] = regen_unavailable == ""
            verifier_meta["regen_findings"] = list(_semantic_output_filter_findings(regen_findings, regen_text))
            if not regen_unavailable and not verifier_meta["regen_findings"]:
                verifier_meta["regen_accepted"] = True
                verifier_meta["findings_before_regen"] = list(findings)
                verifier_meta["findings"] = []
                verifier_meta["finding_codes"] = []
                verifier_meta["action"] = "pass_after_regen"
                verifier_meta["fallback_reason"] = "regenerated"
                route = "draft_for_manager" if result.route in AUTONOMOUS_ROUTES else result.route
                verifier_meta["route_after"] = route
                flags = result.safety_flags
                checklist = result.manager_checklist
                if route != result.route:
                    flags = tuple(dict.fromkeys([*flags, "semantic_output_verifier_regenerated_for_manager"]))
                    checklist = tuple(
                        dict.fromkeys([*checklist, "Смысловой верификатор смягчил текст: оставить как менеджерский черновик."])
                    )
                return replace(result, route=route, draft_text=regen_text, safety_flags=flags, manager_checklist=checklist, metadata=metadata)

    if needs_regen:
        verifier_meta["fallback_reason"] = SEMANTIC_VERIFIER_DOWNGRADE_REASON
    return replace(result, metadata=metadata)


def _verifier_handoff_claims_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _pilot_profile_flag_enabled(context, VERIFIER_HANDOFF_CLAIMS_ENV, aliases=("verifier_handoff_claims_enabled",))


def _semantic_verifier_is_whitelisted_pure_handoff(text: str) -> bool:
    normalized = _normalized_handoff_template_text(text)
    if not normalized:
        return False
    whitelist = {
        _normalized_handoff_template_text(item)
        for item in (
            SAFE_FALLBACK_DRAFT_TEXT,
            *_HUMANE_GENERIC_HANDOFF_TEXTS,
            *dialogue_contract_generic_handoff_texts,
            *dialogue_contract_handoff_exhausted_texts,
        )
    }
    return normalized in whitelist


def _normalized_handoff_template_text(text: str) -> str:
    return " ".join(str(text or "").split()).casefold().replace("ё", "е")


def _run_semantic_output_verifier_once(
    verifier: Callable[[str], object],
    bot_text: str,
    *,
    client_message: str,
    facts: Mapping[str, str],
    active_brand: str,
    route: str,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    prompt = build_semantic_output_verifier_prompt(
        bot_text=bot_text,
        client_message=client_message,
        facts=facts,
        active_brand=active_brand,
        route=route,
        context=context,
    )
    try:
        raw_payload = verifier(prompt)
        payload = extract_json_object(raw_payload) if isinstance(raw_payload, str) else raw_payload
    except subprocess.TimeoutExpired:
        return (), "timeout"
    except Exception as exc:  # noqa: BLE001
        return (), str(exc)[:200] or "verifier_error"
    findings = _semantic_output_findings_from_payload(payload)
    if findings is None:
        return (), "invalid_schema"
    return findings, ""


def _semantic_output_findings_from_payload(
    payload: object,
) -> Optional[tuple[Mapping[str, Any], ...]]:
    if not isinstance(payload, Mapping):
        return None
    raw_findings = payload.get("findings")
    if raw_findings is None and _truthy_value(payload.get("individual_diagnosis")):
        raw_findings = [
            {
                "code": "individual_diagnosis",
                "span": payload.get("span") or "",
                "evidence": payload.get("reason") or "",
            }
        ]
    if not isinstance(raw_findings, Sequence) or isinstance(raw_findings, (str, bytes, bytearray)):
        return None
    findings: list[Mapping[str, Any]] = []
    for raw in raw_findings:
        if not isinstance(raw, Mapping):
            return None
        code = str(raw.get("code") or "").strip()
        if code not in _SEMANTIC_OUTPUT_VERIFIER_CODES:
            return None
        action = _authoritative_gate_action(code)
        if action not in {"annotate", "downgrade_keep_text", "block"}:
            continue
        findings.append(
            {
                "code": code,
                "action": action,
                "span": " ".join(str(raw.get("span") or "").split())[:240],
                "evidence": " ".join(str(raw.get("evidence") or raw.get("reason") or "").split())[:240],
                "missing_fact": " ".join(str(raw.get("missing_fact") or "").split())[:240],
                "relation_to_base": _normalize_semantic_relation(raw.get("relation_to_base")),
                "nearest_fact_key": " ".join(str(raw.get("nearest_fact_key") or raw.get("fact_key") or "").split())[:160],
            }
        )
    return tuple(findings)


def _semantic_output_filter_findings(
    findings: Sequence[Mapping[str, Any]],
    bot_text: str,
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for item in findings:
        code = str(item.get("code") or "")
        if code == "individual_diagnosis" and _has_diagnosis_hedge_and_transfer(bot_text):
            continue
        result.append(dict(item))
    return tuple(result)


def _semantic_output_verifier_highest_action(findings: Sequence[Mapping[str, Any]]) -> str:
    actions = {str(item.get("action") or "") for item in findings if isinstance(item, Mapping)}
    if "block" in actions:
        return "block"
    if "downgrade_keep_text" in actions:
        return "downgrade_keep_text"
    if "annotate" in actions:
        return "annotate"
    return "pass"


def _normalize_semantic_relation(value: object) -> str:
    normalized = str(value or "").strip().casefold()
    if normalized in {"contradicts", "absent", "adjacent"}:
        return normalized
    return "absent"


def _semantic_output_verifier_override(context: Optional[Mapping[str, Any]]) -> Optional[Callable[[str], object]]:
    if not isinstance(context, Mapping):
        return None
    value = context.get("semantic_output_verifier_fn")
    return value if callable(value) else None


def _semantic_output_verifier_timeout_sec() -> int:
    try:
        return max(1, int(float(os.getenv(SEMANTIC_OUTPUT_VERIFIER_TIMEOUT_ENV) or "30")))
    except Exception:
        return 30


def _semantic_output_verifier_max_attempts() -> int:
    try:
        return max(1, int(float(os.getenv(SEMANTIC_OUTPUT_VERIFIER_MAX_ATTEMPTS_SETTING) or "2")))
    except Exception:
        return 2


def _llm_retrieve_timeout_sec() -> int:
    try:
        return max(1, int(float(os.getenv(LLM_RETRIEVE_TIMEOUT_ENV) or "30")))
    except Exception:
        return 30


def _semantic_diagnosis_locked_deferral(result: SubscriptionDraftResult, *, client_message: str = "") -> bool:
    if result.route != "manager_only":
        return False
    if not (
        _humanity_p0_required(result)
        or _semantic_diagnosis_high_risk_flagged(result)
    ):
        return False
    return _semantic_diagnosis_plain_deferral_text(result.draft_text)


def _semantic_diagnosis_high_risk_flagged(result: SubscriptionDraftResult) -> bool:
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    return bool(
        re.search(
            r"high[_-]?risk|p0|refund|complaint|payment[_-]?dispute|legal|zero[_-]?collect|manager[_-]?only",
            flags,
            re.I,
        )
    )


def _semantic_diagnosis_plain_deferral_text(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if not value:
        return True
    low = value.casefold().replace("ё", "е")
    if re.search(
        r"справит|потян|подойдет|тяжело|посильн|влит|догонять|подберут?\s+под\s+реб",
        low,
        re.I,
    ):
        return False
    return bool(re.search(r"передам|верн[её]тся|ответственн|менеджер|сотрудник|сверит|проверит", low, re.I))


def _has_diagnosis_hedge_and_transfer(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    hedge = bool(
        re.search(
            r"заочно|не\s+буду\s+обещ|не\s+возьмусь|лучше\s+(?:сверить|подобрать|оценить)|"
            r"стоит\s+сверить|на\s+пробн|без\s+обещан|уровень\s+лучше",
            value,
            re.I,
        )
    )
    transfer = bool(re.search(r"менеджер|преподавател|педагог|куратор|пробн", value, re.I))
    return hedge and transfer


def _strict_antirepeat_fallback_text(
    context: Optional[Mapping[str, Any]],
    *,
    result: SubscriptionDraftResult,
    client_message: str = "",
) -> str:
    plan = _conversation_intent_plan(context)
    if _scope_guard_has_missing_intent_fact(result, context, plan=plan):
        return _scope_fact_narrow_handoff_text(context, result=result, plan=plan)
    detail = _scope_fact_detail_label(context, result=result, plan=plan)
    if detail == "эту деталь":
        detail = _core_handoff_detail(context, client_message=client_message)
    previous = _humanity_previous_bot_texts(context)
    variants = tuple(item.format(detail=detail) for item in (*_HUMANE_DETAIL_HANDOFF_TEXTS, *_HUMANE_GENERIC_HANDOFF_TEXTS))
    return _select_nonrepeating_text(
        variants,
        previous,
        fallback="Вижу, это важно — отдельно отмечу менеджеру, чтобы он ответил именно по этому пункту.",
    )


def _core_handoff_detail(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> str:
    plan = _conversation_intent_plan(context)
    detail = _scope_fact_detail_label(context, plan=plan)
    if detail and detail != "эту деталь":
        return detail
    text = " ".join(str(client_message or "").split())
    text = re.sub(
        r"^\s*клиент\s+(?:спрашивает|уточняет|интересуется|хочет\s+понять|просит\s+уточнить)\s*(?:,|:|—|-)?\s*",
        "",
        text,
        flags=re.I,
    ).strip(" \t\n\r:;,.—-")
    if text and not text.casefold().startswith("клиент "):
        return text[:90].rstrip() + ("…" if len(text) > 90 else "")
    return "эту деталь"




def _format_choice_is_disjunctive_question(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return bool(
        ("онлайн" in value and has_any_marker(value, ("очно", "офлайн")) and has_marker(value, "или"))
        or ("очно" in value and "онлайн" in value and "?" in value)
    )


def _default_autonomy_flip_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(context, Mapping) or not _autonomy_enabled(context):
        return False
    policy = _autonomy_policy(context)
    for value in (
        context.get("allow_default_autonomy"),
        context.get("default_autonomy_flip_enabled"),
        policy.get("allow_default_autonomy"),
        policy.get("default_autonomy_flip_enabled"),
    ):
        if value is not None:
            return _truthy_value(value)
    return False


def _humanity_p0_required(result: SubscriptionDraftResult) -> bool:
    metadata = dict(result.metadata)
    answer_safety = metadata.get("answer_safety")
    p0_from_safety = bool(isinstance(answer_safety, Mapping) and answer_safety.get("p0_required"))
    return bool(
        p0_from_safety
        or metadata.get("final_p0_text_override")
        or metadata.get("forced_route_high_risk")
        or "high_risk_manager_only" in result.safety_flags
    )












def _trim_repeated_cosmetic_opening(text: str, previous_bot_texts: Sequence[str]) -> str:
    value = str(text or "").strip()
    match = COSMETIC_OPENING_RE.match(value)
    if not match:
        return value
    opening = match.group(0).strip().casefold()
    if not opening:
        return value
    previous_openings = {
        (COSMETIC_OPENING_RE.match(str(item or "").strip()).group(0).strip().casefold())
        for item in previous_bot_texts
        if COSMETIC_OPENING_RE.match(str(item or "").strip())
    }
    if opening not in previous_openings:
        return value
    trimmed = value[match.end() :].lstrip(" ,.!—-")
    if len(trimmed.split()) < 4:
        return value
    return trimmed[:1].upper() + trimmed[1:]






































def _asks_money_price_question(text: str) -> bool:
    normalized = str(text or "").casefold().replace("ё", "е")
    if has_marker(normalized, "процент") and not has_any_marker(normalized, ("стоим", "цена", "цену", "прайс", "руб", "почем", "почём")):
        return False
    return bool(
        re.search(r"\b(?:стоим\w*|цена|цену|цены|ценой|прайс|почем|почём|руб(?:\.|лей|ля|ль)?)\b", normalized)
        or re.search(r"\bсколько\b[^.!?\n]{0,80}\b(?:стоит|стоим|руб|₽)", normalized)
        or re.search(r"\bсколько\b[^.!?\n]{0,80}\b(?:выходит|плат[её]ж|в\s+месяц|за\s+месяц)", normalized)
    )






def _topic_id_from_context(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return UNKNOWN_TOPIC_FALLBACK_ID
    plan = context.get("conversation_intent_plan")
    if isinstance(plan, Mapping) and plan.get("topic_id"):
        return str(plan.get("topic_id") or UNKNOWN_TOPIC_FALLBACK_ID)
    contract = context.get("answer_contract")
    if isinstance(contract, Mapping) and contract.get("topic_id"):
        return str(contract.get("topic_id") or UNKNOWN_TOPIC_FALLBACK_ID)
    return str(context.get("topic_id") or UNKNOWN_TOPIC_FALLBACK_ID)


def _dialogue_contract_tone_guide(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return ""
    examples: list[str] = []
    for key in ("few_shot_style_examples", "few_shot_correction_examples"):
        value = context.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            examples.extend(str(item or "").strip() for item in value if str(item or "").strip())
    gold = context.get("gold_answer_context")
    if isinstance(gold, Mapping):
        for value in gold.values():
            if isinstance(value, str) and value.strip():
                examples.append(value.strip())
            elif isinstance(value, Mapping):
                text = value.get("answer") or value.get("text") or value.get("draft_text")
                if text:
                    examples.append(str(text).strip())
    return " | ".join(dict.fromkeys(examples[:3]))[:1600]


def _dialogue_contract_style_examples(context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    examples: list[str] = []
    for key in ("few_shot_style_examples", "few_shot_correction_examples"):
        value = context.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            examples.extend(str(item or "").strip() for item in value if str(item or "").strip())
    gold = context.get("gold_answer_context")
    if isinstance(gold, Mapping):
        for value in gold.values():
            if isinstance(value, str) and value.strip():
                examples.append(value.strip())
            elif isinstance(value, Mapping):
                text = value.get("answer") or value.get("text") or value.get("draft_text")
                if text:
                    examples.append(str(text).strip())
    return tuple(dict.fromkeys(item[:900] for item in examples if item))[:8]


def _dialogue_contract_safety_flags(pipeline_result: Any) -> list[str]:
    flags = ["dialogue_contract_pipeline", "manager_approval_required", "no_auto_send"]
    if getattr(pipeline_result.contract, "is_p0", False):
        flags.append("dialogue_contract_p0_pregate")
        evidence = getattr(pipeline_result, "reason_evidence", {}) or {}
        if isinstance(evidence, Mapping) and str(evidence.get("p0_handoff_kind") or "") == "payment_dispute":
            flags.append("payment_dispute_manager_only")
    flags.append(
        "dialogue_contract_verified"
        if not pipeline_result.findings and not getattr(pipeline_result, "fallback_reason", "")
        else "dialogue_contract_verification_fallback"
    )
    if getattr(pipeline_result, "unsupported_claims", ()):
        flags.append("dialogue_contract_semantic_fallback")
    if getattr(pipeline_result, "warmed", False):
        flags.append("dialogue_contract_x2_warmth_applied")
    if getattr(pipeline_result, "repaired", False):
        flags.append("dialogue_contract_safety_repair_applied")
    if getattr(pipeline_result, "is_estimate", False):
        flags.append("dialogue_contract_estimate_answer")
    if getattr(pipeline_result, "partial_yield_applied", False):
        flags.append("dialogue_contract_partial_yield_applied")
    if getattr(pipeline_result, "composite_applied", False):
        flags.append("dialogue_contract_composite_applied")
    if getattr(pipeline_result, "next_step_applied", False):
        flags.append("dialogue_contract_next_step_applied")
    return flags


def _sanitize_dialogue_contract_client_text(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    stripped = strip_internal_service_markers(result.draft_text)
    if stripped != result.draft_text:
        flags = tuple(dict.fromkeys([*result.safety_flags, "dialogue_contract_internal_text_sanitized"]))
        metadata = {**dict(result.metadata), "dialogue_contract_internal_text_sanitized": True}
        if not stripped.strip():
            return replace(
                result,
                draft_text=SAFE_FALLBACK_DRAFT_TEXT,
                route="draft_for_manager" if result.route != "manager_only" else result.route,
                safety_flags=tuple(dict.fromkeys([*flags, "manager_approval_required", "no_auto_send"])),
                metadata=metadata,
            )
        result = replace(result, draft_text=stripped, safety_flags=flags, metadata=metadata)
    sanitized = sanitize_answer(result.draft_text, mode="bot")
    blocking_flags = {
        "raw_json_leak",
        "internal_metadata_leak",
        "bot_placeholder_leak",
        "unsafe_placeholder_leak",
        "personal_placeholder_leak",
    }
    blocking_detected = set(sanitized.flags) & blocking_flags
    if not blocking_detected:
        if not sanitized.flags:
            return result
        return replace(
            result,
            safety_flags=tuple(dict.fromkeys([*result.safety_flags, "dialogue_contract_sanitize_checked", *sanitized.flags])),
            metadata={**dict(result.metadata), "dialogue_contract_sanitize_flags": list(sanitized.flags)},
        )
    if sanitized.text == result.draft_text:
        return result
    flags = tuple(dict.fromkeys([*result.safety_flags, "dialogue_contract_sanitize_applied", *sanitized.flags]))
    metadata = {**dict(result.metadata), "dialogue_contract_sanitize_flags": list(sanitized.flags)}
    if not sanitized.text.strip():
        return replace(
            result,
            draft_text=SAFE_FALLBACK_DRAFT_TEXT,
            route="draft_for_manager" if result.route != "manager_only" else result.route,
            safety_flags=tuple(dict.fromkeys([*flags, "manager_approval_required", "no_auto_send"])),
            metadata=metadata,
        )
    return replace(result, draft_text=sanitized.text or SAFE_FALLBACK_DRAFT_TEXT, safety_flags=flags, metadata=metadata)


def _output_sanitizer_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    return _pilot_profile_flag_enabled(context, OUTPUT_SANITIZER_ENV, aliases=("output_sanitizer_enabled",))




def _semantic_output_verifier_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    # In a future autonomous send mode Дмитрий may choose fail-closed when this
    # verifier is unavailable; today it is advisory in draft-only mode.
    return _pilot_profile_flag_enabled(context, SEMANTIC_OUTPUT_VERIFIER_ENV, aliases=("semantic_output_verifier_enabled",))
