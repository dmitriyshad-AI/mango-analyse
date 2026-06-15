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

from mango_mvp.channels.answer_quality_rewriter import (
    AnswerQualityAssessment,
    build_answer_quality_llm_rewrite_prompt,
    apply_answer_quality_rewriter,
)
from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.fact_scope_spec import answer_scopes_allowed, detect_fact_scopes
from mango_mvp.channels.dialogue_contract_pipeline import (
    Toggles as DialogueContractToggles,
    build_conversation as build_dialogue_contract_conversation,
    build_fact_store as build_dialogue_contract_fact_store,
    check_claim_faithfulness as check_dialogue_contract_faithfulness,
    faithfulness_shadow_enabled as dialogue_contract_faithfulness_shadow_enabled,
    faithfulness_shadow_events as dialogue_contract_faithfulness_shadow_events,
    faithfulness_shadow_record as dialogue_contract_faithfulness_shadow_record,
    _GENERIC_HANDOFF_TEXTS as dialogue_contract_generic_handoff_texts,
    _handoff_factual_claim_text as dialogue_contract_handoff_factual_claim_text,
    _HANDOFF_EXHAUSTED_TEXTS as dialogue_contract_handoff_exhausted_texts,
    _is_pure_handoff_text as dialogue_contract_is_pure_handoff_text,
    concrete_anchors as dialogue_contract_concrete_anchors,
    _established_topic_from_context as dialogue_contract_established_topic_from_context,
    new_concrete_anchors as dialogue_contract_new_concrete_anchors,
    parse_contract as parse_dialogue_contract,
    pipeline_enabled as dialogue_contract_pipeline_enabled,
    p0_pre_gate as dialogue_contract_p0_pre_gate,
    run_pipeline as run_dialogue_contract_pipeline,
    verify_output as verify_dialogue_contract_output,
)
from mango_mvp.channels.humanity_guards import (
    has_meta_leak,
    humanity_route_action,
    is_near_repeat,
    meta_markers_present,
    unanswered_direct_question,
)
from mango_mvp.channels.humanity_linter import lint_turn
from mango_mvp.channels.humanity_rewriter import apply_rewrite as apply_humanity_form_rewrite
from mango_mvp.channels.p0_recall_spec import HARD_P0_CODES, codes_from_text, is_benign_hypothetical_refund
from mango_mvp.channels.rules_engine import (
    RuleOutcome,
    apply_rule as apply_migrated_domain_rule,
    load_rules_registry,
    select_rule as select_migrated_domain_rule,
)
from mango_mvp.channels.semantic_roles import tag_message_roles
from mango_mvp.channels.text_signals import has_any_marker, has_marker
from mango_mvp.channels.tone_block import (
    TONE_CLOSE_DETECT_ENV,
    TONE_RICH_FORMAT_ENV,
    TONE_SELL_PROMPT_ENV,
    TONE_WARM_FRAME_ENV,
    apply_warm_frame,
    close_detect_enabled,
    sell_prompt_enabled,
    tone_rich_format_enabled,
)
from mango_mvp.channels.draft_prompt_builder import (
    IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES,
    build_draft_prompt,
    safe_schedule_template,
    should_force_manager_only,
)
from mango_mvp.insights.sanitizers import sanitize_answer
from mango_mvp.insights.phase2_detectors import detect_anxiety, detect_objection
from mango_mvp.insights.tone_score import score_tone
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
    _client_clean_fact_text,
    _deal_action_decision_enabled,
    _direct_path_client_safe_snapshot_fact,
    _direct_path_pilot_config,
    _direct_path_fact_by_brand_key,
    _direct_path_fact_value,
    _direct_path_load_snapshot,
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
    _keep_answer_supported,
    _keep_answer_hard_anchors,
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
    DIRECT_PATH_GOLD_TOPIC_KEYWORDS,
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
    DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY,
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
    KNOWN_CONTEXT_REPAIR_TEXT,
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
    OFF_TOPIC_INPUT_RE,
    OFF_TOPIC_UNPK_SAFE_TEXT,
    OLD_TERM_SAFE_TEXT,
    PAYMENT_CONFIRMATION_RE,
    PAYMENT_DISPUTE_SAFE_TEXT,
    PAYMENT_LINK_SAFE_TEXT,
    PH2_ANXIETY_ENV,
    PH2_OBJECTION_ENV,
    PLANNER_INTENT_CONFIDENCE_THRESHOLD,
    PRECISE_CONDITION_RE,
    PRICE_AMOUNT_RE,
    PROGRAM_HANDOFF_SAFE_TEXT,
    PROMOCODE_DRAFT_RE,
    PROMOCODE_SAFE_TEXT,
    QUITTANCE_SAFE_TEXT,
    REFUND_ZERO_COLLECT_SAFE_TEXT,
    RESULT_GUARANTEE_INPUT_RE,
    RESULT_GUARANTEE_SAFE_TEXT,
    RULES_ENGINE_PLANNER_INTENT_ENV,
    RouteDecision,
    SCOPE_FACT_GUARD_ENV,
    SOFT_NEGATIVE_HANDOFF_SAFE_TEXT,
    STEP4_KEEP_ANSWER_ENV,
    SUBJECT_GUARD_MARKERS,
    SafeTemplateSpec,
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
    _GUARDCHAIN_RECOVERY_BLOCKING_FLAGS,
    _INFORMATIONAL_SAFE_TEMPLATE_NAMES,
    _LEGAL_SAFE_VARIANTS,
    _N_POINTS_PROMISE_CONTEXT_RE,
    _PAYMENT_DISPUTE_VARIANTS,
    _REFUND_ZERO_COLLECT_VARIANTS,
    _SAFE_TEMPLATE_DISPATCHER_RECONSIDER_BLOCKING_FLAGS,
    _a_thread_enabled,
    _allowed_subjects_from_context,
    _answer_contract,
    _answer_contract_green_template_reduction_enabled,
    _answer_fact_scopes,
    _answer_quality_was_rewritten,
    _answers_matkap_scope,
    _answers_tax_deduction_scope,
    _apply_migrated_rules_engine,
    _apply_rules_engine_outcome,
    _apply_safe_template_spec,
    _asks_center_contact,
    _asks_live_status_or_booking_question,
    _asks_non_matkap_document_or_contract,
    _asks_non_tax_document_or_contract,
    _autonomy_enabled,
    _autonomy_policy,
    _autonomy_topic_allowed,
    _brand_guarded_result,
    _client_message_contains_pii,
    _compact_conversation_intent_plan_for_metadata,
    _confirmed_fact_texts,
    _context_has_missing_fact_signal,
    _context_with_dialogue_contract_retrieved_facts,
    _context_with_selling_thread_slots,
    _conversation_intent_plan,
    _conversation_plan_controls_green_templates,
    _conversation_plan_semantic_non_p0,
    _conversation_plan_template_blocked_by_substantive_answer,
    _cross_brand_safe_template,
    _dedupe_sentence,
    _dialog_context_haystack,
    _dialogue_contract_mapping,
    _dialogue_contract_retrieved_facts,
    _draft_addresses_question,
    _draft_confirms_payment,
    _draft_is_low_value_without_exact_fact,
    _ensure_sentence,
    _extract_numeric_promise_claims,
    _fact_key_root,
    _fact_scope_guard_template,
    _float_value,
    _forbidden_pair_guard_template,
    _foton_address_template_from_kb,
    _foton_contact_template_from_kb,
    _foton_offline_free_trial_guard_template,
    _has_client_safe_current_fact,
    _has_informational_safe_template,
    _has_missing_fact_signal,
    _has_presale_refund_policy_context,
    _has_word_marker,
    _humanity_previous_bot_texts,
    _informational_fact_matches_question,
    _informational_yield_has_unbacked_concrete_anchors,
    _is_admission_guarantee_case,
    _is_approved_policy_c_identity_text,
    _is_combined_high_risk_case,
    _is_complaint_case,
    _is_enrollment_signup_question,
    _is_future_price_case,
    _is_informational_terminal_template,
    _is_legal_threat_case,
    _is_lesson_recording_question,
    _is_policy_c_identity_question,
    _is_refund_case,
    _is_reputation_only_case,
    _is_result_guarantee_case,
    _is_template_from_kb_terminal_text,
    _is_terminal_direct_info_template,
    _is_unpk_bank_installment_question,
    _is_unpk_installment_case,
    _is_unpk_zvsh_case,
    _is_verified_client_safe_template,
    _is_verified_safe_numeric_template,
    _known_context_repair_text,
    _known_fields_from_text,
    _live_status_manager_check_text,
    _looks_like_generic_template,
    _looks_like_low_value_handoff_only,
    _manager_only_recovery_yield_allowed,
    _manager_route_migrated_rules_override_allowed,
    _mapping_has_client_safe_current_fact,
    _memory_followup_answered_topic,
    _memory_mentions_different_topic,
    _memory_mentions_focus,
    _memory_norm,
    _memory_short_followup,
    _memory_text_items,
    _memory_topic_aliases,
    _mentioned_subjects,
    _mentions_unbacked_children_rule,
    _merge_known_context_fields,
    _merge_selling_slot_values,
    _merged_selling_signals,
    _metadata_with_guarded_original_text,
    _metadata_with_self_route_deferral_cleared,
    _migrated_rule_intent_from_dialogue_contract,
    _migrated_rules_keep_existing_verified_answer,
    _missing_fact_helpful_template,
    _normalize_for_template_decision,
    _p0_text_with_antirepeat,
    _payment_context,
    _payment_guarded_result,
    _payment_status,
    _phase2_anxiety_enabled,
    _phase2_anxiety_signal,
    _phase2_objection_enabled,
    _phase2_objection_signal,
    _pipeline_contract,
    _pipeline_fact_texts,
    _pipeline_travel_estimate_applied,
    _planner_intent_candidate,
    _policy_c_identity_allowed,
    _prefer_format_facts,
    _presale_refund_policy_template,
    _produce_admission_guarantee_template,
    _produce_cross_brand_template,
    _produce_result_guarantee_template,
    _produce_terminal_template,
    _promoted_verified_fact_text,
    _recovery_candidate_from_informational_facts,
    _remove_repeated_known_data_questions,
    _requested_fact_scope_context,
    _result_has_live_status_missing_fact,
    _retrieved_fact_matches_active_brand,
    _rules_engine_facts,
    _rules_engine_intent_shadow,
    _rules_engine_planner_intent_enabled,
    _safe_template_already_applied,
    _safe_template_applied_name,
    _safe_template_can_yield_to_dispatcher,
    _safe_template_route,
    _safe_template_yield_result,
    _scope_fact_detail_label,
    _scope_fact_guard_enabled,
    _scope_fact_missing_guard_template,
    _scope_fact_narrow_handoff_text,
    _scope_guard_has_foreign_concrete_fact,
    _scope_guard_has_missing_intent_fact,
    _scope_guard_missing_fact_keys,
    _scope_guard_required_fact_keys,
    _select_nonrepeating_text,
    _selling_slots_from_contract_and_text,
    _selling_slots_from_memory,
    _selling_slots_from_text,
    _semantic_haystack,
    _skip_missing_fact_template_by_answer_contract,
    _soften_current_price_deadline_text,
    _step4_keep_answer_enabled,
    _strict_informational_yield_ok,
    _strip_false_p0_flags,
    _subjects_from_retrieved_facts,
    _terminal_safe_template,
    _text_explicitly_mentions_selling_slot,
    _unpk_all_addresses_template_from_kb,
    _unpk_contact_template_from_kb,
    _unpk_moscow_address_template_from_kb,
    _unstated_subject_safe_text,
    _validated_guardchain_recovery_candidate,
    _verified_informational_answer,
    _with_rules_engine_intent_shadow,
    _yield_dispatcher_to_travel_estimate,
    apply_autonomy_matrix_guard,
    apply_brand_separation_guard,
    apply_conversation_intent_plan_guard,
    apply_dialogue_contract_v2_template_dispatcher,
    apply_funnel_policy_guard,
    apply_high_risk_content_guards,
    apply_input_policy_guards,
    apply_known_context_redundant_question_guard,
    apply_payment_confirmation_guard,
    apply_subscription_policy_guards,
    apply_taxonomy_topic_guard,
    apply_unstated_subject_guard,
    decide_route,
    detect_high_risk_input_markers,
    find_redundant_questions_for_known_context,
    find_unsupported_numeric_promises,
    is_high_risk_result,
    known_context_fields,
)

ANSWER_QUALITY_LLM_REWRITE_ENV = "TELEGRAM_ANSWER_QUALITY_LLM_REWRITE"


ANSWER_QUALITY_LLM_REWRITER_ENV = "TELEGRAM_ANSWER_QUALITY_LLM_REWRITER"


ANSWER_QUALITY_LLM_REWRITE_REASONING_ENV = "TELEGRAM_ANSWER_QUALITY_LLM_REWRITE_REASONING"


ANSWER_QUALITY_LLM_REWRITE_MODE_ENV = "TELEGRAM_ANSWER_QUALITY_LLM_REWRITE_MODE"


HUMANITY_BLOCK_A_ROUTE_FIX_ENV = "TELEGRAM_HUMANITY_BLOCK_A_ROUTE_FIX"


HUMANITY_X2_REWRITE_ENV = "TELEGRAM_DRAFT_X2_REWRITE"


HUMANITY_X2_REWRITE_MODE_ENV = "TELEGRAM_DRAFT_X2_REWRITE_MODE"


HUMANITY_X2_REWRITE_MODEL_ENV = "TELEGRAM_DRAFT_X2_REWRITE_MODEL"


HUMANITY_X2_REWRITE_REASONING_ENV = "TELEGRAM_DRAFT_X2_REWRITE_REASONING"


DIALOGUE_CONTRACT_SEMANTIC_MATCH_MODEL_ENV = "TELEGRAM_DIALOGUE_CONTRACT_SEMANTIC_MATCH_MODEL"


DIALOGUE_CONTRACT_SEMANTIC_MATCH_REASONING_ENV = "TELEGRAM_DIALOGUE_CONTRACT_SEMANTIC_MATCH_REASONING"


ANTIREPEAT_STRICT_ENV = "TELEGRAM_ANTIREPEAT_STRICT"


A_PROACTIVE_ENV = "TELEGRAM_A_PROACTIVE"


A_RICH_FORMAT_ENV = "TELEGRAM_A_RICH_FORMAT"


PH2_TONE_ENV = "TELEGRAM_PH2_TONE"


SEMANTIC_DIAGNOSIS_GUARD_ENV = "TELEGRAM_SEMANTIC_DIAGNOSIS_GUARD"


SEMANTIC_DIAGNOSIS_MODEL_ENV = "TELEGRAM_SEMANTIC_DIAGNOSIS_MODEL"


SEMANTIC_DIAGNOSIS_REASONING_ENV = "TELEGRAM_SEMANTIC_DIAGNOSIS_REASONING"


SEMANTIC_OUTPUT_VERIFIER_MODEL_ENV = "TELEGRAM_SEMANTIC_VERIFIER_MODEL"


SEMANTIC_OUTPUT_VERIFIER_REASONING_ENV = "TELEGRAM_SEMANTIC_VERIFIER_REASONING"


SEMANTIC_OUTPUT_VERIFIER_TIMEOUT_ENV = "TELEGRAM_SEMANTIC_VERIFIER_TIMEOUT_SEC"


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


SEMANTIC_DIAGNOSIS_SAFE_TEXT = (
    "Заочно не буду оценивать уровень конкретного ребёнка. Лучше сверить уровень и нагрузку с преподавателем; "
    "менеджер поможет сверить детали и подобрать аккуратный следующий шаг."
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
    "p0_semantic_risk": "block",
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
    "invented_generalization": "annotate",
}


DIRECT_PATH_REPLACE_TEXT_GATE_CODES = frozenset(
    {
        "hard_p0",
        "zero_collect_required",
        "p0_promise",
        "p0_semantic_risk",
        "brand_leak",
        "cross_brand",
        "unsupported_product_number",
    }
)


HIGH_RISK_INPUT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "refund",
        re.compile(
            r"\bвозв?рат(?!\w*\s+к\s+тем)\w*"
            r"|\bвозвращ\w*\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
            r"|\bверн\w*(?:\s+мне|\s+нам|\s+пожалуйста)?\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
            r"|\bвозвратит\w*\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
            r"|\bрасторг\w*\s+договор"
            r"|\bрасторжен\w*\s+договор"
            r"|\bотказ\w*\s+от\s+обучен"
            r"|\bзабрать\s+деньги",
            re.I,
        ),
    ),
    (
        "legal",
        re.compile(
            r"\bсуд\b|\bиск\b|претензи|досудеб|роспотребнадзор|прокуратур"
            r"|наруш\w*\s+прав|расторжен\w*\s+договор|по\s+закону[^.!?\n]{0,80}обязан\w*(?:\s+(?:вернуть|возместить|расторгнуть))?",
            re.I,
        ),
    ),
    (
        "complaint",
        re.compile(
            r"жалоб(?!а\s+на\s+сайт)\w*|жалуюсь|возмущ\w*|недовол\w*|претензи|конфликт"
            r"|обман|ужасн|плохо\s+учит|плохо\s+пров[её]л|некомпетентн\w*",
            re.I,
        ),
    ),
    (
        "reputation_threat",
        re.compile(r"отзыв\w*\s+в\s+интернет|всех\s+предупреж\w*|напиш\w*\s+отзыв|остав\w*\s+отзыв", re.I),
    ),
)


LEGAL_CONTEXT_INPUT_RE = re.compile(
    r"\bсуд\b|\bиск\b|претензи|досудеб|роспотребнадзор|прокуратур|адвокат|юрист"
    r"|прав[ао][^.!?\n]{0,60}потребител|защит[а-яё]*\s+прав\s+потребител"
    r"|наруш\w*\s+прав|расторжен\w*\s+договор|по\s+закону[^.!?\n]{0,80}обязан\w*",
    re.I,
)


ZERO_COLLECT_DRAFT_RE = re.compile(
    r"\b(?:пришлите|напишите|уточните|сообщите|отправьте|предоставьте|нужн[аоы]?|понадоб(?:ит|ят))\b"
    r"[^.!?\n]{0,140}?"
    r"\b(?:фио|имя|фамили[яюи]|договор|номер\s+договора|телефон|email|e-mail|почт[ауеы]|сумм[ауеы]?|"
    r"причин[ауеы]?|подтвержден\w+\s+оплат|чек|квитанц)\b",
    re.I,
)


REFUND_FORBIDDEN_DETAIL_RE = re.compile(
    r"\b(?:фио|имя|фамили[яюи]|договор\w*|номер\s+договора|телефон|email|e-mail|почт[ауеы]|"
    r"сумм[ауеы]?|причин[ауеы]?|оплат\w*|подтвержден\w+\s+оплат|чек|квитанц)\b",
    re.I,
)


COMPLAINT_APOLOGY_RE = re.compile(
    r"\b(?:понимаю|извините|приносим\s+извинения|(?:нам|мне|очень)\s+жаль|сожалеем|неприятно)\b",
    re.I,
)


COMPLAINT_DETAIL_COLLECT_RE = re.compile(
    r"\b(?:уточните|пришлите|напишите|сообщите|предоставьте|подскажите)\b"
    r"[^.!?\n]{0,180}?"
    r"\b(?:дат[ауеы]?|предмет|курс|групп[ауеы]?|имя|фио|ученик[а-я]*|преподавател[яьюе]?|что\s+именно)\b",
    re.I,
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


def _rules_engine_result_applied(metadata: Mapping[str, Any]) -> bool:
    rules = metadata.get("rules_engine") if isinstance(metadata.get("rules_engine"), Mapping) else {}
    applied = str(rules.get("applied") or "").strip()
    if applied:
        return True
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    pipeline_rules = pipeline.get("rules_engine") if isinstance(pipeline.get("rules_engine"), Mapping) else {}
    return bool(str(pipeline_rules.get("applied") or "").strip())


def _direct_path_p0_text(reason: str, context: Optional[Mapping[str, Any]]) -> tuple[str, str]:
    lowered = str(reason or "").casefold()
    if "payment" in lowered or "спис" in lowered or "оплат" in lowered:
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
    p0_reason = dialogue_contract_p0_pre_gate(client_message, context=context)
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
    high_risk = detect_high_risk_input_markers(client_message, context=context)
    if high_risk:
        meta = _direct_path_metadata(
            attempted=True,
            model_called=False,
            facts=facts,
            fact_pack=fact_pack,
            preblocked=True,
            pilot_config=pilot_config,
            context=context,
            preblock_reason="high_risk",
            reason_class="high_risk",
            reason_evidence={"risk_codes": list(high_risk)},
        )
        return SubscriptionDraftResult(
            message_type="manager_only",
            broad_group="direct_path",
            route="manager_only",
            draft_text=SAFE_FALLBACK_DRAFT_TEXT,
            risk_level="high",
            safety_flags=(*BASE_SAFETY_FLAGS, "direct_path_preblocked_high_risk", "manager_approval_required", "no_auto_send"),
            manager_checklist=("High-risk: прямой путь не вызывался, отвечает менеджер.",),
            metadata={"direct_path": meta, "reason_class": "high_risk", "is_manager_deferral": True},
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


DEAL_ACTION_DECISION_SCHEMA_VERSION = "deal_action_decision_v1_2026_06_14"

DEAL_ACTION_UNKNOWN = "unknown"

DEAL_ACTIONS = frozenset(
    {
        "answer_only",
        "send_schedule",
        "send_materials",
        "send_crm_data",
        "capture_lead",
        "schedule_followup",
        "send_payment_link",
        "send_document",
        "advance_stage",
        "handoff_manager",
        DEAL_ACTION_UNKNOWN,
    }
)

_DEAL_ACTION_MANAGER_APPROVAL_ACTIONS = frozenset(
    {
        "handoff_manager",
        "send_crm_data",
        "send_payment_link",
        "send_document",
        "advance_stage",
    }
)

_DEAL_ACTION_PAYMENT_RE = re.compile(
    r"\b(?:беру|оформляйте|оформим|готов[аы]?\s+(?:оплатить|платить|оформить)|давайте\s+(?:оплат|оформ)|ссылк\w+\s+на\s+оплат)\b",
    re.I,
)
_DEAL_ACTION_PAYMENT_QUERY_RE = re.compile(r"\b(?:как|куда|можно\s+ли)\s+(?:оплат|плат)|\b(?:ссылк\w+|реквизит\w*)\b", re.I)
_DEAL_ACTION_PAYMENT_QUESTION_RE = re.compile(r"\?(?:\s*)$|(?:как|куда|можно\s+ли|сколько|какая|какой|что)\b", re.I)
_DEAL_ACTION_SCHEDULE_RE = re.compile(r"\b(?:распис|когда|во\s+сколько|дни|дням|суббот|воскрес|будн|время\s+занят)\b", re.I)
_DEAL_ACTION_MATERIALS_RE = re.compile(r"\b(?:пробн|фрагмент|материал|посмотреть\s+(?:урок|занят)|пример\s+(?:урока|занят))\b", re.I)
_DEAL_ACTION_CRM_DATA_RE = re.compile(
    r"\b(?:баланс|остат\w*|осталось\s+(?:занят\w*|урок\w*|средств\w*)|мои\s+оплат\w*|сколько(?:\s+\w+){0,4}\s+(?:занят\w*|урок\w*|средств\w*))\b",
    re.I,
)
_DEAL_ACTION_LEAD_RE = re.compile(r"\b(?:запишите|записать|оставлю\s+заявк|хочу\s+записаться|интересно|подберите|подобрать\s+групп)\b", re.I)
_DEAL_ACTION_FOLLOWUP_RE = re.compile(r"\b(?:перезвон|позвоните|напишите\s+позже|свяжитесь|напомните|завтра|вечером|утром)\b", re.I)
_DEAL_ACTION_DOCUMENT_RE = re.compile(r"\b(?:договор|оферт|сч[её]т|квитанц|документ|справк|акт)\b", re.I)
_DEAL_ACTION_FACT_QUESTION_RE = re.compile(
    r"\b(?:цена|стоимост|скидк|рассроч|долями|адрес|маткап|материнск|налог|платформ|запис[ьи]\s+урок|вы\s+бот|ты\s+бот|кто\s+вы)\b",
    re.I,
)


def _deal_action_unknown(reason: str, *, proposal: Optional[Mapping[str, Any]] = None, enabled: bool = True) -> dict[str, Any]:
    return {
        "schema_version": DEAL_ACTION_DECISION_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "action": DEAL_ACTION_UNKNOWN,
        "confidence": 0.0,
        "reason": str(reason or "unknown"),
        "source": "deterministic",
        "proposal_action": str((proposal or {}).get("action") or DEAL_ACTION_UNKNOWN),
        "requires_manager_approval": False,
    }


def _deal_action_normalize(value: Any) -> str:
    action = str(value or "").strip().casefold()
    action = action.replace("-", "_").replace(" ", "_")
    if action == "book_trial":
        return "send_materials"
    return action if action in DEAL_ACTIONS else DEAL_ACTION_UNKNOWN


def _deal_action_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    text = str(value or "").strip().replace(",", ".")
    if not text:
        return None
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        return None


def _deal_action_proposal(result: SubscriptionDraftResult) -> dict[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    raw = metadata.get("action_proposal")
    if isinstance(raw, Mapping):
        action = _deal_action_normalize(raw.get("action") or raw.get("name") or raw.get("intent"))
        confidence = _deal_action_float(raw.get("confidence"))
        return {
            "schema_version": DEAL_ACTION_DECISION_SCHEMA_VERSION,
            "action": action,
            "confidence": confidence,
            "reason": " ".join(str(raw.get("reason") or raw.get("rationale") or "").split())[:240],
            "source": str(raw.get("source") or "direct_model"),
        }
    if isinstance(raw, str) and raw.strip():
        return {
            "schema_version": DEAL_ACTION_DECISION_SCHEMA_VERSION,
            "action": _deal_action_normalize(raw),
            "confidence": None,
            "reason": "",
            "source": "direct_model",
        }
    return {
        "schema_version": DEAL_ACTION_DECISION_SCHEMA_VERSION,
        "action": DEAL_ACTION_UNKNOWN,
        "confidence": None,
        "reason": "model_proposal_missing",
        "source": "missing",
    }


def _deal_action_context_mapping(context: Optional[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    value = context.get(key)
    return value if isinstance(value, Mapping) else {}


def _deal_action_intent_plan(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return _deal_action_context_mapping(context, "conversation_intent_plan")


def _deal_action_known_slots(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    slots = _deal_action_context_mapping(context, "known_slots")
    plan_slots = _deal_action_context_mapping(_deal_action_intent_plan(context), "known_slots")
    memory = _deal_action_context_mapping(context, "dialogue_memory_view")
    memory_slots = _deal_action_context_mapping(memory, "known_slots")
    return {**dict(slots), **dict(memory_slots), **dict(plan_slots)}


def _deal_action_texts(result: SubscriptionDraftResult, context: Optional[Mapping[str, Any]]) -> tuple[dict[str, str], tuple[str, ...]]:
    gate_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    facts = _authoritative_gate_fact_texts(result, gate_context)
    exact_keys: list[str] = []
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    direct_facts = direct.get("retrieved_facts") if isinstance(direct.get("retrieved_facts"), Mapping) else {}
    for key, value in direct_facts.items():
        if str(key).strip() and str(value).strip():
            facts[str(key)] = str(value)
    exact_keys.extend(str(key) for key in (direct.get("wide_fact_exact_keys") or ()) if str(key).strip())
    if not exact_keys:
        exact_keys.extend(str(key) for key in facts.keys())
    return facts, tuple(dict.fromkeys(exact_keys))


def _deal_action_final_p0(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> tuple[bool, str]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    gate = metadata.get("authoritative_output_gate") if isinstance(metadata.get("authoritative_output_gate"), Mapping) else {}
    findings = gate.get("findings") if isinstance(gate.get("findings"), Sequence) else ()
    gate_codes = tuple(str(item.get("code") or "") for item in findings if isinstance(item, Mapping))
    model_p0 = metadata.get("direct_path_model_p0") if isinstance(metadata.get("direct_path_model_p0"), Mapping) else {}
    if bool(model_p0.get("is_p0")):
        kind = str(model_p0.get("p0_kind") or "model_p0")
        return True, f"direct_path_model_p0:{kind}"
    raw_hard_codes = tuple(code for code in codes_from_text(client_message) if code in HARD_P0_CODES)
    safety = classify_answer_safety(
        client_message=client_message,
        context=context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
    )
    if raw_hard_codes:
        return True, "p0_recall_spec:" + ",".join(dict.fromkeys(raw_hard_codes))
    if safety.p0_required and not safety.semantic_non_p0:
        return True, f"answer_safety:{safety.primary_risk or 'p0_required'}"
    if result.route == "manager_only" and any(
        code in {"hard_p0", "zero_collect_required", "p0_promise", "p0_semantic_risk"} for code in gate_codes
    ):
        return True, "authoritative_output_gate:" + ",".join(dict.fromkeys(gate_codes))
    if _humanity_p0_required(result):
        return True, "result_p0_flags"
    return False, ""


def _deal_action_amounts(text: str) -> tuple[str, ...]:
    result: list[str] = []
    for match in PRICE_AMOUNT_RE.finditer(str(text or "")):
        digits = re.sub(r"\D+", "", match.group(0))
        if digits:
            result.append(digits)
    return tuple(dict.fromkeys(result))


def _deal_action_price_backed_by_facts(result: SubscriptionDraftResult, facts: Mapping[str, str]) -> tuple[bool, tuple[str, ...]]:
    amounts = _deal_action_amounts(result.draft_text)
    if not amounts:
        return False, ()
    fact_text = "\n".join(str(value or "") for value in facts.values())
    fact_amounts = set(_deal_action_amounts(fact_text))
    return all(amount in fact_amounts for amount in amounts), amounts


def _deal_action_product_unambiguous(context: Optional[Mapping[str, Any]], facts: Mapping[str, str]) -> bool:
    slots = _deal_action_known_slots(context)
    grade = str(slots.get("grade") or slots.get("class") or slots.get("student_grade") or "").strip()
    subject = str(slots.get("subject") or slots.get("course_subject") or slots.get("interest_subject") or "").strip()
    product = str(slots.get("product") or slots.get("course") or slots.get("product_family") or "").strip()
    if grade and subject:
        return True
    if product and product.casefold() not in {"unknown", "непонятно", "общий"}:
        return True
    return False


def _deal_action_has_objection_or_exit(client_message: str, context: Optional[Mapping[str, Any]]) -> bool:
    plan = _deal_action_intent_plan(context)
    selling = plan.get("selling") if isinstance(plan.get("selling"), Mapping) else {}
    objection = str(selling.get("objection") or "none").strip().casefold()
    if objection and objection != "none":
        return True
    if _truthy_value(selling.get("exit_signal")):
        return True
    value = str(client_message or "").casefold().replace("ё", "е")
    return bool(
        re.search(r"\b(?:дорого|подумаю|подумаем|не\s+сейчас|пока\s+не\s+готов|посмотрю|сравню|не\s+подходит)\b", value)
    )


def _deal_action_payment_confirmed(client_message: str) -> bool:
    text = str(client_message or "")
    if not _DEAL_ACTION_PAYMENT_RE.search(text):
        return False
    if _DEAL_ACTION_PAYMENT_QUESTION_RE.search(text) and not re.search(
        r"\b(?:оформляйте|беру|готов[аы]?\s+(?:оплатить|платить|оформить)|давайте\s+(?:оплат|оформ))\b",
        text,
        re.I,
    ):
        return False
    return True


def _deal_action_schedule_available(
    *,
    context: Optional[Mapping[str, Any]],
    facts: Mapping[str, str],
    exact_keys: Sequence[str],
) -> bool:
    slots = _deal_action_known_slots(context)
    grade = str(slots.get("grade") or slots.get("class") or slots.get("student_grade") or "").strip()
    subject = str(slots.get("subject") or slots.get("course_subject") or slots.get("interest_subject") or "").strip()
    fmt = str(slots.get("format") or slots.get("course_format") or slots.get("preferred_format") or "").strip()
    if not (grade and subject and fmt):
        return False
    exact = set(str(key) for key in exact_keys)
    for key, value in facts.items():
        if exact and str(key) not in exact:
            continue
        combined = f"{key} {value}".casefold()
        if ("tallanto" in combined or "group" in combined or "групп" in combined) and re.search(
            r"распис|вс\b|сб\b|пн\b|вт\b|ср\b|чт\b|пт\b|суббот|воскрес|старт|начал",
            combined,
            re.I,
        ):
            return True
    return False


def _deal_action_materials_available(facts: Mapping[str, str], exact_keys: Sequence[str]) -> bool:
    exact = set(str(key) for key in exact_keys)
    for key, value in facts.items():
        if exact and str(key) not in exact:
            continue
        combined = f"{key} {value}".casefold()
        if re.search(r"онлайн", combined, re.I) and re.search(r"фрагмент|материал|пробн", combined, re.I):
            return True
    return False


def _deal_action_crm_identity_ok(context: Optional[Mapping[str, Any]]) -> tuple[bool, str]:
    if not isinstance(context, Mapping):
        return False, "no_context"
    quality = context.get("context_quality") if isinstance(context.get("context_quality"), Mapping) else {}
    identity = context.get("client_identity") if isinstance(context.get("client_identity"), Mapping) else {}
    active_brand = _active_brand(context)
    brand_values = []
    for key in ("amo_context", "tallanto_context", "read_only_customer_context"):
        mapping = context.get(key) if isinstance(context.get(key), Mapping) else {}
        for brand_key in ("brand", "active_brand", "expected_brand"):
            if str(mapping.get(brand_key) or "").strip():
                brand_values.append(str(mapping.get(brand_key)).strip().casefold())
    brand_ok = not brand_values or active_brand in {"", "unknown"} or all(value in {"", active_brand} for value in brand_values)
    identity_ok = bool(
        _truthy_value(quality.get("customer_identity_found"))
        or _truthy_value(identity.get("verified"))
        or str(identity.get("match_class") or "").strip().casefold() in {"strong", "exact", "verified"}
        or (identity.get("phone") and _truthy_value(identity.get("phone_verified")))
    )
    if not identity_ok:
        return False, "identity_not_strict"
    if not brand_ok:
        return False, "brand_mismatch"
    return True, "strict_identity"


def _deal_action_candidate(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
    facts: Mapping[str, str],
    exact_keys: Sequence[str],
) -> tuple[str, str]:
    text = str(client_message or "")
    low = text.casefold()
    if result.route == "manager_only":
        return "handoff_manager", "manager_only_route"
    if _DEAL_ACTION_CRM_DATA_RE.search(text):
        ok, reason = _deal_action_crm_identity_ok(context)
        return ("send_crm_data", "crm_data_strict_identity") if ok else ("handoff_manager", f"crm_data_requires_manager:{reason}")
    if _DEAL_ACTION_PAYMENT_RE.search(text) or _DEAL_ACTION_PAYMENT_QUERY_RE.search(text):
        backed, amounts = _deal_action_price_backed_by_facts(result, facts)
        if (
            backed
            and amounts
            and _deal_action_product_unambiguous(context, facts)
            and _deal_action_payment_confirmed(text)
            and not _deal_action_has_objection_or_exit(text, context)
        ):
            return "send_payment_link", "payment_preconditions_met"
        if _deal_action_payment_confirmed(text):
            return DEAL_ACTION_UNKNOWN, "payment_preconditions_missing"
    if _DEAL_ACTION_SCHEDULE_RE.search(text):
        if _deal_action_schedule_available(context=context, facts=facts, exact_keys=exact_keys):
            return "send_schedule", "schedule_exact_tallanto_fact"
        return "answer_only", "schedule_missing_precise_group"
    if _DEAL_ACTION_MATERIALS_RE.search(text):
        if _deal_action_materials_available(facts, exact_keys):
            return "send_materials", "online_fragment_fact"
        return DEAL_ACTION_UNKNOWN, "materials_fact_missing"
    if _DEAL_ACTION_FOLLOWUP_RE.search(text):
        return "schedule_followup", "explicit_followup_request"
    if _DEAL_ACTION_LEAD_RE.search(text):
        return "capture_lead", "lead_capture_signal"
    if _DEAL_ACTION_DOCUMENT_RE.search(text):
        return "send_document", "document_request_observed"
    plan = _deal_action_intent_plan(context)
    primary_intent = str(plan.get("primary_intent") or "").strip()
    if primary_intent in {"pricing", "installment", "discount", "address", "matkap", "tax", "platform_access", "identity"}:
        return "answer_only", f"fact_question:{primary_intent}"
    if _DEAL_ACTION_FACT_QUESTION_RE.search(text):
        return "answer_only", "fact_question"
    return DEAL_ACTION_UNKNOWN, "no_explicit_action_signal"


def _deal_action_model_lowers(candidate: str, proposal: Mapping[str, Any]) -> tuple[str, str]:
    proposal_action = _deal_action_normalize(proposal.get("action"))
    if candidate == "send_payment_link" and proposal_action in {"answer_only", "handoff_manager", DEAL_ACTION_UNKNOWN}:
        return proposal_action, "model_lowered_payment"
    if candidate in {"send_materials", "send_schedule"} and proposal_action == "handoff_manager":
        return "handoff_manager", "model_lowered_to_manager"
    return candidate, ""


def _deal_action_text_sync(action: str, result: SubscriptionDraftResult) -> tuple[str, str]:
    text = str(result.draft_text or "")
    if action == "send_payment_link" and not re.search(
        r"\b(?:оплат\w*|оформ\w*|ссылк\w*|плат[её]ж\w*|реквизит\w*)\b",
        text,
        re.I,
    ):
        return "answer_only", "action_not_in_text"
    if action == "send_schedule" and not _DEAL_ACTION_SCHEDULE_RE.search(text):
        return "answer_only", "action_not_in_text"
    if action == "send_materials" and not _DEAL_ACTION_MATERIALS_RE.search(text):
        return "answer_only", "action_not_in_text"
    if action == "schedule_followup" and not _DEAL_ACTION_FOLLOWUP_RE.search(text):
        return "answer_only", "action_not_in_text"
    if action == "capture_lead" and not re.search(r"\b(?:запис|заявк|передам|менеджер|контакт|телефон)\b", text, re.I):
        return "answer_only", "action_not_in_text"
    return action, ""


def _deal_action_manager_note(action: str) -> str:
    notes = {
        "send_schedule": "Рекомендуемый следующий шаг: если данные группы подтверждены, отправить клиенту расписание по выбранному классу, предмету и формату.",
        "send_materials": "Рекомендуемый следующий шаг: если формат онлайн подтверждён, отправить клиенту фрагмент занятия или материалы.",
        "send_crm_data": "Рекомендуемый следующий шаг: менеджеру проверить карточку и ответить по балансу/остатку занятий только этому клиенту.",
        "capture_lead": "Рекомендуемый следующий шаг: зафиксировать заявку и уточнить недостающие данные для записи.",
        "schedule_followup": "Рекомендуемый следующий шаг: поставить безопасный follow-up для менеджера.",
        "send_payment_link": "Рекомендуемый следующий шаг: клиент назвал готовность; если менеджер подтвердит продукт и цену, сформировать безопасную ссылку на оплату.",
        "send_document": "Рекомендуемый следующий шаг: менеджеру подготовить нужный документ/счёт после проверки карточки.",
        "advance_stage": "Рекомендуемый следующий шаг: проверить, нужно ли продвинуть этап сделки в AMO.",
        "handoff_manager": "Рекомендуемый следующий шаг: передать менеджеру, без автоматических действий.",
    }
    return notes.get(action, "")


def _deal_action_requires_manager_approval(
    action: str,
    result: SubscriptionDraftResult,
    *,
    p0_required: bool = False,
) -> bool:
    if p0_required:
        return True
    if result.route in {"manager_only", "draft_for_manager"}:
        return True
    return action in _DEAL_ACTION_MANAGER_APPROVAL_ACTIONS


def apply_deal_action_decision_layer(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _deal_action_decision_enabled(context):
        return result
    metadata = dict(result.metadata)
    if isinstance(metadata.get("action_decision"), Mapping):
        return result
    proposal = _deal_action_proposal(result)
    p0_required, p0_reason = _deal_action_final_p0(result, client_message=client_message, context=context)
    facts, exact_keys = _deal_action_texts(result, context)
    sync_flag = ""
    if p0_required:
        action = "handoff_manager"
        reason = p0_reason or "p0_final_latch"
    else:
        action, reason = _deal_action_candidate(
            result,
            client_message=client_message,
            context=context,
            facts=facts,
            exact_keys=exact_keys,
        )
        action, lower_reason = _deal_action_model_lowers(action, proposal)
        reason = lower_reason or reason
        action, sync_flag = _deal_action_text_sync(action, result)
        if sync_flag:
            reason = sync_flag
    if action not in DEAL_ACTIONS:
        action = DEAL_ACTION_UNKNOWN
        reason = "invalid_action"
    confidence = 1.0 if action in {"handoff_manager", "answer_only"} else 0.75 if action != DEAL_ACTION_UNKNOWN else 0.0
    decision = {
        "schema_version": DEAL_ACTION_DECISION_SCHEMA_VERSION,
        "enabled": True,
        "action": action,
        "confidence": confidence,
        "reason": reason,
        "source": "deterministic",
        "proposal_action": proposal.get("action"),
        "requires_manager_approval": _deal_action_requires_manager_approval(action, result, p0_required=p0_required),
        "no_live_execution": True,
        "p0_latched": bool(p0_required),
        "active_brand": _active_brand(context),
        "exact_fact_keys": list(exact_keys)[:20],
        "threshold_configured": False,
    }
    if sync_flag:
        decision["sync_flag"] = sync_flag
    if action == "send_payment_link":
        decision["preconditions"] = {
            "price_backed_by_facts": True,
            "product_unambiguous": True,
            "explicit_last_reply_confirmation": True,
            "no_objection_or_exit_signal": True,
        }
    metadata["action_proposal"] = proposal
    metadata["action_decision"] = decision
    checklist = list(result.manager_checklist)
    note = _deal_action_manager_note(action)
    if note:
        checklist.append(note)
    return replace(
        result,
        manager_checklist=tuple(dict.fromkeys(item for item in checklist if str(item or "").strip())),
        metadata=metadata,
    )


def _direct_path_finalize_metadata(
    result: SubscriptionDraftResult,
    *,
    before_gate_route: str,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
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
            "route_before_gate": before_gate_route,
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
    template_trace.extend(dict(item) for item in _template_from_kb_context_trace(context))
    if template_trace:
        direct["template_from_kb_trace"] = template_trace
        metadata["template_from_kb_trace"] = template_trace
    metadata["direct_path"] = direct
    metadata["text_composition_source"] = direct.get("text_composition_source") or metadata.get("text_composition_source")
    metadata["is_manager_deferral"] = bool(direct["is_manager_deferral"])
    metadata["reason_class"] = reason_class
    return replace(result, metadata=metadata)


_A2_TIME_RE = re.compile(
    r"\b(?:сегодня|завтра|послезавтра|утром|дн[её]м|вечером|после\s+обеда|до\s+\d{1,2}|"
    r"после\s+\d{1,2}|в\s+\d{1,2}(?::\d{2})?|с\s+\d{1,2}\s+до\s+\d{1,2})\b",
    re.I,
)


_A2_FAKE_DONE_RE = re.compile(
    r"я\s+(?:вас\s+)?записал|вы\s+записаны|запись\s+оформлена|оформил\s+запись|записал\s+на\s+курс",
    re.I,
)


_A2_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]")


_A2_SERIOUS_TAGS = {"p0", "refund", "complaint", "manager_only", "legal", "guarantee"}


def apply_a2_proactive_layer(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    """A2.1 callback/contact capture plus deterministic rich-format guard."""

    updated = result
    if _a2_proactive_enabled(context) or sell_prompt_enabled(context):
        updated = _a2_contact_capture_handoff(updated, client_message=client_message, context=context)
    if _a2_rich_format_enabled(context):
        updated = _a2_apply_rich_format_guard(updated, client_message=client_message, context=context)
    return updated


_TONE_SELL_PROMPT_STEP_RE = re.compile(
    r"\b(?:подскаж(?:у|ите)|помогу|сориентирую|подбер[уеё]м?|подбер[её]т|подобрать|расскажу|обращайтесь|давайте|можно\s+(?:начать|записаться|посмотреть)|"
    r"оставьте\s+(?:телефон|номер|контакт)|позвоним|свяжемся|когда\s+удобн|как\s+записаться|запис[а-яё]*|"
    r"передам\s+менеджеру|менеджер\s+(?:подбер[её]т|поможет|сверит|свяжется))\b",
    re.I,
)


def _tone_sell_prompt_step_observation(text: str, close_meta: Mapping[str, Any]) -> Mapping[str, Any]:
    match = _TONE_SELL_PROMPT_STEP_RE.search(str(text or ""))
    if not match:
        return {
            "has_step": bool(close_meta),
            "step_kind": "close_meta" if close_meta else "",
            "step_match": "",
        }
    fragment = match.group(0).strip()
    low = fragment.casefold().replace("ё", "е")
    if re.search(r"телефон|номер|контакт|позвоним|свяжемся", low, re.I):
        kind = "contact_cta"
    elif re.search(r"пробн|как\s+записаться|запис", low, re.I):
        kind = "enrollment_or_trial_step"
    elif re.search(r"менеджер|передам", low, re.I):
        kind = "manager_handoff"
    elif re.search(r"подбер|давайте|можно\s+(?:начать|посмотреть)", low, re.I):
        kind = "selection_step"
    else:
        kind = "generic_help"
    return {
        "has_step": True,
        "step_kind": kind,
        "step_match": fragment[:120],
    }


def apply_tone_sell_prompt_observer(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not sell_prompt_enabled(context):
        return result
    metadata = dict(result.metadata)
    existing = dict(metadata.get("tone_sell_prompt") or {}) if isinstance(metadata.get("tone_sell_prompt"), Mapping) else {}
    active_self_route = result.route in {"bot_answer_self", "bot_answer_self_for_pilot"}
    serious = _a2_context_tag(result, client_message=client_message, context=context) in _A2_SERIOUS_TAGS
    close_meta = metadata.get("close_detect") if isinstance(metadata.get("close_detect"), Mapping) else {}
    step_observation = _tone_sell_prompt_step_observation(str(result.draft_text or ""), close_meta)
    has_step = bool(step_observation.get("has_step"))
    step_missing = bool(active_self_route and not serious and not has_step)
    metadata["tone_sell_prompt"] = {
        **existing,
        "enabled": True,
        "step_missing": step_missing,
        "has_visible_step": has_step,
        "step_kind": str(step_observation.get("step_kind") or ""),
        "step_match": str(step_observation.get("step_match") or ""),
        "route": result.route,
    }
    if step_missing:
        metadata["sell_prompt_step_missing"] = True
    return replace(result, metadata=metadata)


_TONE_CLOSE_GRATITUDE_RE = re.compile(
    r"\b(?:спасибо|благодарю|понял[аи]?|ок(?:ей)?|хорошо|до\s+свидания|всего\s+доброго|всего\s+хорошего)\b",
    re.I,
)


_TONE_CLOSE_EXIT_SIGNAL_RE = re.compile(
    r"\b(?:подумаю|подумаем|посмотрю|посмотрим|вернусь|вернемся|вернёмся|пока\s+посмотр|посоветуюсь|обсужу|обсудим|решу|решим|сравню|сравним)\b",
    re.I,
)


_TONE_CLOSE_QUESTION_RE = re.compile(
    r"\?|"
    r"\b(?:подскажите|скажите|сколько|когда|как|где|куда|можно|есть\s+ли|будет\s+ли|"
    r"получится\s+ли|подойдет\s+ли|подойд[её]т\s+ли|какой|какая|какие|почему|зачем|что\s+нужно)\b",
    re.I,
)


_TONE_CLOSE_REFUSAL_RE = re.compile(
    r"\b(?:нет|не\s+нужно|не\s+надо|не\s+требуется|не\s+стоит|не\s+хочу)\b",
    re.I,
)


_TONE_CLOSE_CONTACT_CTA_RE = re.compile(
    r"(?:оставьте|подскажите|пришлите|напишите)[^.?!\n]{0,100}\b(?:телефон|номер|контакт)\b"
    r"|\b(?:позвоним|свяжемся)\b"
    r"|\bменеджер\s+подбер[её]т\b",
    re.I,
)


_TONE_CLOSE_TRIAL_CTA_RE = re.compile(
    r"\b(?:бесплатн\w*\s+пробн\w*|подсказать,\s*как\s+записаться)\b",
    re.I,
)


_TONE_CLOSE_STEP_CTA_RE = re.compile(
    rf"(?:{_TONE_CLOSE_CONTACT_CTA_RE.pattern})|(?:{_TONE_CLOSE_TRIAL_CTA_RE.pattern})",
    re.I,
)


_TONE_CLOSE_ADVERSATIVE_RE = re.compile(r"\b(?:но|однако|только)\b", re.I)


_TONE_CLOSE_UNANSWERED_RE = re.compile(
    r"\b(?:не\s+ответил|не\s+ответили|не\s+отвеч|по\s+сути|без\s+ответа|не\s*понятн\w*|не\s+понял[аи]?)\b",
    re.I,
)


_TONE_CLOSE_PROBLEM_MARKER_RE = re.compile(
    r"\b(?:деньг\w*|списал\w*|плат[её]ж\w*|оплат\w*|срочн\w*|заняти[еяй]\w*\s+нет|доступ\w*\s+нет)\b",
    re.I,
)


_TONE_CLOSE_CONTACT_TEXTS = (
    "Рада была помочь! Хотите, менеджер подберёт группу под ваше расписание? Оставьте телефон — позвоним, когда удобно.",
    "Рада была помочь! Если захотите, менеджер подберёт группу под ваше расписание. Оставьте телефон — позвоним в удобное время.",
)


_TONE_CLOSE_TRIAL_TEXTS = (
    "Обращайтесь в любое время! Кстати, можно прийти на бесплатное пробное занятие — посмотрите, как всё устроено. Подсказать, как записаться?",
    "Обращайтесь! У Фотона можно прийти на бесплатное пробное занятие и спокойно посмотреть формат. Подсказать, как записаться?",
)


_TONE_CLOSE_RETURN_TEXTS = (
    "Спасибо вам! Будем рады видеть вас на занятиях — возвращайтесь, если появятся вопросы.",
    "Спасибо вам! Буду рада помочь, если появятся вопросы по занятиям.",
)


_TONE_CLOSE_P0_FLAGS = {
    "p0",
    "refund",
    "refund_claim",
    "payment_dispute",
    "complaint",
    "legal",
    "legal_threat",
    "high_risk",
    "p0_deferral",
}


def apply_tone_close_detect_layer(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not close_detect_enabled(context) or not _tone_close_detect_is_close_message(client_message, context=context):
        return result
    if _tone_close_detect_is_p0(result, context=context):
        return _tone_close_metadata(result, status="suppressed_p0", step="", context=context)
    if _tone_close_pending_manager(context, client_message=client_message):
        return _tone_close_metadata(
            replace(
                result,
                route="bot_answer_self_for_pilot",
                draft_text=_tone_close_pending_text(),
                safety_flags=tuple(dict.fromkeys([*result.safety_flags, "tone_close_detect_pending"])),
            ),
            status="suppressed_pending",
            step="pending",
            context=context,
        )
    status = "suppressed_handoff" if result.route in {"manager_only", "draft_for_manager"} else "fired"
    previous_bot_texts = _humanity_previous_bot_texts(context)
    refused_previous_step = _tone_close_refused_previous_step(client_message, previous_bot_texts)
    old_p0_without_active_latch = _tone_close_old_p0_history(context)
    step, text = _tone_close_next_step_text(
        context,
        previous_bot_texts=previous_bot_texts,
        no_cta=refused_previous_step or old_p0_without_active_latch,
    )
    close_flags = tuple(
        flag
        for flag in result.safety_flags
        if str(flag or "").strip() not in {"manager_approval_required", "no_auto_send", "llm_fallback"}
    )
    flags = tuple(
        dict.fromkeys(
            [
                *close_flags,
                "tone_close_detect",
            ]
        )
    )
    return _tone_close_metadata(
        replace(
            result,
            route="bot_answer_self_for_pilot",
            message_type="other",
            draft_text=text,
            safety_flags=flags,
            manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, "Tone close-detect: проверить тёплое закрытие без новых фактов."])),
            error=None if status == "fired" else result.error,
        ),
        status=status,
        step=step,
        contact_requested=_tone_close_contact_requested_after_step(context, step=step, previous_bot_texts=previous_bot_texts),
        context=context,
    )


def _tone_close_metadata(
    result: SubscriptionDraftResult,
    *,
    status: str,
    step: str,
    context: Optional[Mapping[str, Any]],
    contact_requested: Optional[bool] = None,
) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    if result.route == "bot_answer_self_for_pilot":
        metadata = _metadata_with_self_route_deferral_cleared(metadata)
    memory = context.get("dialogue_memory_view") if isinstance(context, Mapping) else {}
    if contact_requested is None:
        contact_requested = _tone_close_contact_requested_from_memory(memory)
    payload = {
        "enabled": True,
        "status": status,
        "step": step,
        "contact_requested": bool(contact_requested),
    }
    metadata["close_detect"] = payload
    return replace(result, metadata=metadata)


def _tone_close_detect_is_close_message(client_message: str, *, context: Optional[Mapping[str, Any]]) -> bool:
    text = str(client_message or "").strip()
    if not text:
        return False
    if _TONE_CLOSE_EXIT_SIGNAL_RE.search(text):
        return False
    if _TONE_CLOSE_QUESTION_RE.search(text):
        return False
    if _tone_close_has_unanswered_or_problem_continuation(text):
        return False
    contract = context.get("answer_contract") if isinstance(context, Mapping) else None
    if isinstance(contract, Mapping) and str(contract.get("message_type") or "").strip() == "question":
        return False
    if isinstance(context, Mapping) and str(context.get("message_type") or "").strip() == "question":
        return False
    return bool(_TONE_CLOSE_GRATITUDE_RE.search(text))


def _tone_close_has_unanswered_or_problem_continuation(client_message: str) -> bool:
    text = str(client_message or "")
    if _TONE_CLOSE_ADVERSATIVE_RE.search(text) and (
        _TONE_CLOSE_UNANSWERED_RE.search(text) or _TONE_CLOSE_PROBLEM_MARKER_RE.search(text)
    ):
        return True
    return bool(re.search(r"\b(?:деньг\w*\s+списал\w*|списал\w*[^.!?\n]{0,40}деньг\w*|плат[её]ж\w*[^.!?\n]{0,25}нет)\b", text, re.I))


def _tone_close_detect_is_p0(result: SubscriptionDraftResult, *, context: Optional[Mapping[str, Any]]) -> bool:
    flags = {str(flag).strip() for flag in result.safety_flags if str(flag).strip()}
    if flags.intersection(_TONE_CLOSE_P0_FLAGS):
        return True
    if result.route == "manager_only" and any(flag in flags for flag in ("refund_policy_manager_only", "zero_collect_required")):
        return True
    memory = context.get("dialogue_memory_view") if isinstance(context, Mapping) else None
    if isinstance(memory, Mapping):
        latch = memory.get("p0_latch") if isinstance(memory.get("p0_latch"), Mapping) else {}
        if bool(latch.get("active")):
            return True
    return False


def _tone_close_pending_manager(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> bool:
    memory = context.get("dialogue_memory_view") if isinstance(context, Mapping) else None
    if not isinstance(memory, Mapping):
        return False
    pending = False
    if str(memory.get("handoff_state") or "").strip() in {"required", "suggested"}:
        pending = True
    pending = pending or bool(memory.get("pending_manager_actions"))
    if not pending:
        return False
    return _tone_close_message_references_pending(client_message)


def _tone_close_old_p0_history(context: Optional[Mapping[str, Any]]) -> bool:
    memory = context.get("dialogue_memory_view") if isinstance(context, Mapping) else None
    if not isinstance(memory, Mapping):
        return False
    latch = memory.get("p0_latch") if isinstance(memory.get("p0_latch"), Mapping) else {}
    return bool(latch.get("had_hard_p0_claim")) and not bool(latch.get("active"))


_TONE_CLOSE_PENDING_REFERENCE_RE = re.compile(
    r"\b(?:жд\w*|ожида\w*)[^.!?\n]{0,40}\b(?:ответ|менеджер|связ|звон|верн|уточн|провер)"
    r"|\b(?:ответ|менеджер|связ|звон|верн|уточн|провер)[^.!?\n]{0,40}\b(?:жд\w*|ожида\w*)"
    r"|\b(?:пусть|пускай|давайте)?[^.!?\n]{0,25}\bменеджер\b[^.!?\n]{0,50}\b(?:уточн|провер|свер|ответ|связ|верн)",
    re.I,
)


def _tone_close_message_references_pending(client_message: str) -> bool:
    return bool(_TONE_CLOSE_PENDING_REFERENCE_RE.search(str(client_message or "")))


def _tone_close_next_step_text(
    context: Optional[Mapping[str, Any]],
    *,
    previous_bot_texts: Sequence[str] = (),
    no_cta: bool = False,
) -> tuple[str, str]:
    memory = context.get("dialogue_memory_view") if isinstance(context, Mapping) else {}
    if no_cta:
        return (
            "return",
            _select_nonrepeating_text(_TONE_CLOSE_RETURN_TEXTS, previous_bot_texts, fallback=_TONE_CLOSE_RETURN_TEXTS[0]),
        )
    contact_requested = _tone_close_contact_requested_from_memory(
        memory if isinstance(memory, Mapping) else {}
    ) or _tone_close_previous_contact_requested(previous_bot_texts)
    if not contact_requested:
        return (
            "contact",
            _select_nonrepeating_text(_TONE_CLOSE_CONTACT_TEXTS, previous_bot_texts, fallback=_TONE_CLOSE_CONTACT_TEXTS[0]),
        )
    if _active_brand(context) == "foton" and not _tone_close_previous_trial_requested(previous_bot_texts):
        return (
            "trial",
            _select_nonrepeating_text(_TONE_CLOSE_TRIAL_TEXTS, previous_bot_texts, fallback=_TONE_CLOSE_TRIAL_TEXTS[0]),
        )
    return (
        "return",
        _select_nonrepeating_text(_TONE_CLOSE_RETURN_TEXTS, previous_bot_texts, fallback=_TONE_CLOSE_RETURN_TEXTS[0]),
    )


def _tone_close_contact_requested_from_memory(memory: Any) -> bool:
    if not isinstance(memory, Mapping):
        return False
    state = memory.get("proactive_state") if isinstance(memory.get("proactive_state"), Mapping) else {}
    return bool(state.get("contact_requested"))


def _tone_close_contact_requested_after_step(
    context: Optional[Mapping[str, Any]],
    *,
    step: str,
    previous_bot_texts: Sequence[str],
) -> bool:
    memory = context.get("dialogue_memory_view") if isinstance(context, Mapping) else {}
    return bool(
        step == "contact"
        or _tone_close_contact_requested_from_memory(memory)
        or _tone_close_previous_contact_requested(previous_bot_texts)
    )


def _tone_close_previous_contact_requested(previous_bot_texts: Sequence[str]) -> bool:
    return any(_TONE_CLOSE_CONTACT_CTA_RE.search(str(item or "")) for item in previous_bot_texts)


def _tone_close_previous_trial_requested(previous_bot_texts: Sequence[str]) -> bool:
    return any(_TONE_CLOSE_TRIAL_CTA_RE.search(str(item or "")) for item in previous_bot_texts)


def _tone_close_refused_previous_step(client_message: str, previous_bot_texts: Sequence[str]) -> bool:
    text = str(client_message or "")
    if not (_TONE_CLOSE_REFUSAL_RE.search(text) and _TONE_CLOSE_GRATITUDE_RE.search(text)):
        return False
    return any(_TONE_CLOSE_STEP_CTA_RE.search(str(item or "")) for item in previous_bot_texts[-3:])


def _tone_close_pending_text() -> str:
    return "Спасибо! Менеджер уже занимается вашим вопросом и скоро вернётся с ответом."


def _a2_contact_capture_handoff(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    if result.route == "manager_only" or _a2_p0_or_high_risk(result, client_message=client_message, context=context):
        return result
    phone = _a2_extract_phone(client_message)
    phone_known = _a2_context_phone_known(context)
    has_time = _a2_has_time(client_message)
    if not phone and not (phone_known and has_time):
        return result
    metadata = dict(result.metadata)
    metadata["a2_proactive"] = {
        **(dict(metadata.get("a2_proactive") or {}) if isinstance(metadata.get("a2_proactive"), Mapping) else {}),
        "enabled": True,
        "step": "offer_callback",
        "contact_captured": True,
        "phone_masked": _a2_mask_phone(phone) if phone else "[known_phone]",
        "preferred_time": "[provided]" if has_time else "",
        "crm_write": False,
        "policy_source": "deterministic",
    }
    text = (
        "Спасибо, передам менеджеру — он свяжется с вами в удобное время."
        if has_time
        else "Спасибо, передам менеджеру — он свяжется с вами и уточнит удобное время."
    )
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "A2.1: клиент оставил контакт/время; связаться вручную, без CRM-записи из бота.",
            ]
        )
    )
    return replace(
        result,
        route="draft_for_manager" if result.route != "manager_only" else result.route,
        draft_text=text,
        safety_flags=tuple(
            dict.fromkeys(
                [
                    *result.safety_flags,
                    "a2_proactive_contact_captured",
                    "manager_approval_required",
                    "no_auto_send",
                ]
            )
        ),
        manager_checklist=checklist,
        manager_followup_required=True,
        metadata=metadata,
    )


def _a2_apply_rich_format_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    text = str(result.draft_text or "")
    context_tag = _a2_context_tag(result, client_message=client_message, context=context)
    cleaned = _a2_enforce_emoji_limit(text, context_tag=context_tag)
    if cleaned == text:
        return result
    metadata = dict(result.metadata)
    metadata["a2_rich_format"] = {
        **(dict(metadata.get("a2_rich_format") or {}) if isinstance(metadata.get("a2_rich_format"), Mapping) else {}),
        "enabled": True,
        "emoji_guard_applied": True,
        "context_tag": context_tag,
    }
    return replace(
        result,
        draft_text=cleaned,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "a2_rich_format_emoji_guarded"])),
        metadata=metadata,
    )


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


def _a2_p0_or_high_risk(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> bool:
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    if any(marker in flags for marker in ("high_risk", "zero_collect", "legal", "complaint", "payment_dispute")):
        return True
    safety = classify_answer_safety(
        client_message=client_message,
        context=context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
    )
    return bool(safety.p0_required and not safety.semantic_non_p0)


def _a2_has_time(text: str) -> bool:
    return bool(_A2_TIME_RE.search(str(text or "")))


def _a2_mask_phone(phone: str) -> str:
    digits = re.sub(r"\D+", "", str(phone or ""))
    if not digits:
        return ""
    return f"[phone:***{digits[-2:]}]"


def _a2_context_phone_known(context: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(context, Mapping):
        return False
    containers: list[Mapping[str, Any]] = []
    for key in ("known_slots", "known_dialog_fields", "known_client_fields", "client_identity"):
        value = context.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        for key in ("known_slots", "client_confirmed_slots", "crm_known_slots"):
            value = memory.get(key)
            if isinstance(value, Mapping):
                containers.append(value)
    for container in containers:
        for key in ("phone_known", "phone", "normalized_phone", "client_phone"):
            raw = container.get(key)
            if isinstance(raw, Mapping):
                raw = raw.get("value")
            if str(raw or "").strip().casefold() not in {"", "false", "none", "0"}:
                return True
    return False


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
    safety = classify_answer_safety(client_message=client_message, context=context, topic_id=result.topic_id, route=result.route)
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
    findings = _authoritative_gate_findings(result, client_message=client_message, context=context)
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

    route = "draft_for_manager" if direct_path_keep_text else _authoritative_gate_downgraded_route(result.route, actions)
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
    r"(?:записыва(?:й(?:те)?|ю|ем)|запиш(?:и(?:те)?|у|ем)(?:\s+нас)?|реб[её]н(?:ок|ка|ку)?|сын(?:а)?|доч(?:ь|ка|ку|ери)?|"
    r"ученик(?:а)?|ученица|фио|зовут|имя)\s*[:—-]?\s*"
    r"(?P<name>[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){0,2})",
    re.I,
)


_CLIENT_SELF_NAME_MARKER_RE = re.compile(
    r"(?:\bя\b|меня|мама|папа|родител[ья])\s*[:—-]?\s*"
    r"(?P<name>[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){0,1})",
    re.I,
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
            replacement = (
                "[данные у менеджера]"
                if " " in name or _client_name_allowed(name, parent_names)
                else "данные ребёнка"
                if not _client_name_allowed(name, child_first_names)
                else name
            )
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
    client = " ".join(str(client_message or "").split())
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
    for match in _CLIENT_CHILD_IDENTITY_PROMPT_RE.finditer(" ".join(str(client_message or "").split())):
        first = _presale_prompt_child_name_value(match.group("name"))
        if first and first not in result:
            result.append(first)
    for match in _CLIENT_NAME_MARKER_RE.finditer(" ".join(str(client_message or "").split())):
        first = _presale_prompt_child_name_value(match.group("name"))
        if first and first not in result:
            result.append(first)
    return tuple(result)


def _client_dialogue_parent_names(client_message: str) -> tuple[str, ...]:
    result: list[str] = []
    client = " ".join(str(client_message or "").split())
    for pattern in (_CLIENT_PARENT_IDENTITY_PROMPT_RE, _CLIENT_SELF_NAME_MARKER_RE):
        for match in pattern.finditer(client):
            name = " ".join(str(match.group("name") or "").split()).strip(" ,.;:!?")
            for part in ([name] + [item for item in name.split() if item]):
                if part and part not in result:
                    result.append(part)
    return tuple(result)


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
    contract = _pipeline_contract(result, active_brand=_active_brand(gate_context), fact_keys=tuple(facts.keys()))
    previous_bot_texts = _humanity_previous_bot_texts(gate_context)
    p0_already_guarded = _authoritative_gate_p0_already_guarded(result)
    has_pipeline = _authoritative_gate_has_pipeline(result)
    for finding in verify_dialogue_contract_output(
        result.draft_text,
        facts=facts,
        active_brand=_active_brand(gate_context),
        contract=contract,
        client_message=client_message,
        context=gate_context,
        previous_bot_texts=previous_bot_texts,
    ):
        if not has_pipeline and finding.code not in {"brand_leak", "meta_leak", "ai_disclosure", "p0_promise", "p0_semantic_risk"}:
            continue
        if finding.code == "p0_promise" and _authoritative_gate_verified_content_flag(result):
            continue
        if p0_already_guarded and finding.code in {"p0_semantic_risk", "p0_promise"}:
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
    )
    raw_hard_codes = tuple(code for code in codes_from_text(client_message) if code in HARD_P0_CODES)
    hard_codes = raw_hard_codes if safety.p0_required else tuple(code for code in safety.risk_codes if code in HARD_P0_CODES)
    if not p0_already_guarded and (hard_codes or (safety.p0_required and not safety.semantic_non_p0)):
        detail = ",".join(dict.fromkeys([*hard_codes, *[code for code in safety.risk_codes if code in HARD_P0_CODES]]))
        findings.append(_authoritative_gate_finding("hard_p0", detail=detail or safety.primary_risk, source="answer_safety"))
    if safety.zero_collect_required and not p0_already_guarded and (safety.p0_required or hard_codes):
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


def draft_has_identity_disclosure(text: str) -> bool:
    return bool(find_identity_disclosure_phrases(text))


def find_identity_disclosure_phrases(text: str) -> tuple[str, ...]:
    lowered = str(text or "").casefold()
    return tuple(phrase for phrase in IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES if _identity_phrase_present(lowered, phrase))


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


def _safe_template_yield_before_fallback(
    before: SubscriptionDraftResult,
    after: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult | None:
    applied = _safe_template_applied_name(after)
    if applied not in _INFORMATIONAL_SAFE_TEMPLATE_NAMES:
        return None
    if applied == "terminal" and not _is_informational_terminal_template(after.draft_text):
        return None
    if applied == "terminal" and (
        (_asks_non_tax_document_or_contract(client_message, context=context) and _answers_tax_deduction_scope(before.draft_text))
        or (_asks_non_matkap_document_or_contract(client_message, context=context) and _answers_matkap_scope(before.draft_text))
    ):
        return None
    if not _verified_informational_answer(before, client_message=client_message, context=context, template_name=applied):
        return None
    metadata = {
        **dict(before.metadata),
        "safe_template_yielded_to_verified_answer": True,
        "safe_template_yielded_spec": applied,
    }
    flags = tuple(dict.fromkeys([*before.safety_flags, "safe_template_yielded_to_verified_answer"]))
    return replace(before, safety_flags=flags, metadata=metadata)


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


def apply_humanity_guards(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    """Final conversational guard: remove meta leaks and avoid useless handoff/repeats.

    This layer is deliberately conservative. It never weakens real P0/brand/fact
    gates and only promotes an answer from manager-only to draft when a verified
    answer fact is already present.
    """

    raw_p0_required = _humanity_p0_required(result)
    previous_bot_texts = _humanity_previous_bot_texts(context)
    block_a_enabled = _humanity_block_a_route_fix_enabled(context)
    block_generic_fact_answer = _humanity_generic_fact_answer_blocked(result, client_message=client_message)
    has_answer_fact = (not block_generic_fact_answer) and _has_humanity_answer_fact(context)
    preserve_existing_answer = _humanity_preserve_existing_answer(result)
    metadata = dict(result.metadata)
    benign_p0_context = (
        is_benign_hypothetical_refund(client_message)
        or _conversation_plan_semantic_non_p0(context, client_message=client_message)
    )
    hard_p0_text_locked = bool(
        metadata.get("final_p0_text_override")
        or metadata.get("zero_collect_legal_guarded")
        or metadata.get("zero_collect_refund_guarded")
        or metadata.get("complaint_apology_guarded")
        or metadata.get("payment_dispute_manager_only")
    )
    p0_required = raw_p0_required and not (benign_p0_context and not hard_p0_text_locked)
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    route = result.route
    draft_text = result.draft_text
    changed = False

    if has_meta_leak(draft_text) and not _humanity_allows_dry_p0_text(result, p0_required=p0_required):
        cleaned = _sanitize_humanity_meta_text(draft_text)
        markers = meta_markers_present(draft_text)
        if cleaned and not has_meta_leak(cleaned):
            draft_text = cleaned
        else:
            fact_answer = "" if block_generic_fact_answer else _humanity_fact_answer(context, client_message=client_message)
            draft_text = fact_answer or (
                "Передам вопрос менеджеру, он ответит по сути."
            )
            route = "draft_for_manager" if route != "manager_only" else route
        flags.append("humanity_meta_leak_removed")
        checklist.append("Проверить, что клиентский текст не содержит служебных пометок и manager-facing фраз.")
        metadata["humanity_meta_leak_removed"] = True
        metadata["humanity_meta_markers"] = markers
        changed = True

    client_roles = tag_message_roles(client_message)
    draft_roles = tag_message_roles(draft_text)
    direct_question_unanswered = unanswered_direct_question(
        client_message,
        draft_text,
        client_topics=client_roles.topics,
        draft_topics=draft_roles.topics,
    )
    block_a_direct_answer = ""
    if (
        block_a_enabled
        and not p0_required
        and result.message_type not in {"non_question", "wait_for_more", "manager_only"}
    ):
        block_a_direct_answer = _humanity_block_a_direct_answer(
            context,
            client_message=client_message,
            current_draft=draft_text,
            previous_bot_texts=previous_bot_texts,
        )
    if block_a_direct_answer:
        draft_text = block_a_direct_answer
        if "правила можно посмотреть до оплаты" in block_a_direct_answer.casefold():
            route = "draft_for_manager"
        else:
            route = "bot_answer_self_for_pilot" if route != "manager_only" else "draft_for_manager"
        flags.append("humanity_block_a_direct_answer_applied")
        checklist.append("Слой человечности A: ответ перестроен на текущий вопрос без повторения предыдущего шаблона.")
        metadata["humanity_block_a_direct_answer_applied"] = True
        direct_question_unanswered = False
        changed = True

    if not p0_required and has_answer_fact and _humanity_can_trim_cosmetic_opening(result):
        trimmed = _trim_repeated_cosmetic_opening(draft_text, previous_bot_texts)
        if trimmed != draft_text:
            draft_text = trimmed
            flags.append("humanity_cosmetic_opening_trimmed")
            checklist.append("Косметический повторный зачин убран: ответ должен начинаться ближе к факту.")
            metadata["humanity_cosmetic_opening_trimmed"] = True
            changed = True

    if (
        not p0_required
        and has_answer_fact
        and result.message_type not in {"non_question", "context_update", "wait_for_more"}
    ):
        precise_fact_answer = _humanity_context_correction_answer(
            context, client_message=client_message, current_draft=draft_text
        ) or _humanity_precise_fact_answer(
            context, client_message=client_message, current_draft=draft_text
        )
        if precise_fact_answer:
            draft_text = precise_fact_answer
            route = "bot_answer_self_for_pilot" if route != "manager_only" else "draft_for_manager"
            flags.append("humanity_precise_fact_answer_applied")
            checklist.append("Клиент просит точное число/процент: ответ перестроен на точный извлечённый факт.")
            metadata["humanity_precise_fact_answer_applied"] = True
            changed = True

    if (
        not p0_required
        and has_answer_fact
        and not preserve_existing_answer
        and result.route not in {"bot_answer_self", "bot_answer_self_for_pilot"}
        and not _humanity_guarded_handoff_reason(result)
        and direct_question_unanswered
    ):
        fact_answer = "" if block_generic_fact_answer else _humanity_fact_answer(context, client_message=client_message)
        if fact_answer:
            draft_text = fact_answer
            route = "draft_for_manager" if route == "manager_only" else route
            flags.append("humanity_unanswered_question_repaired")
            checklist.append("Ответ был перестроен на прямой вопрос клиента по извлеченному факту.")
            metadata["humanity_unanswered_question_repaired"] = True
            direct_question_unanswered = False
            changed = True

    installment_amount_answer = ""
    if (
        not p0_required
        and not _humanity_guarded_handoff_reason(result)
        and result.route not in {"bot_answer_self", "bot_answer_self_for_pilot"}
        and not metadata.get("humanity_block_a_direct_answer_applied")
    ):
        installment_amount_answer = _humanity_installment_amount_answer(
            context, client_message=client_message
        )
    if installment_amount_answer:
        draft_text = installment_amount_answer
        route = "bot_answer_self_for_pilot"
        flags.append("humanity_installment_amount_repaired")
        checklist.append(
            "Клиент спросил про платёж в месяц: ответить из цены и условий оплаты, не подменяя годовую цену семестром."
        )
        metadata["humanity_installment_amount_repaired"] = True
        changed = True

    if p0_required and route != "manager_only":
        route = "manager_only"
        flags.append("humanity_p0_route_locked")
        metadata["humanity_p0_route_locked"] = True
        changed = True

    strict_antirepeat = _antirepeat_strict_enabled(context)
    repeat_threshold = 0.85 if strict_antirepeat else 0.8
    core_handoff_repeat = (not p0_required) and _is_core_handoff_fallback_repeat(
        draft_text,
        previous_bot_texts,
        threshold=repeat_threshold,
    )
    if not p0_required and is_near_repeat(draft_text, previous_bot_texts, threshold=repeat_threshold):
        fact_answer = (
            block_a_direct_answer
            or ("" if block_generic_fact_answer else _humanity_fact_answer(context, client_message=client_message))
        )
        if fact_answer and not is_near_repeat(fact_answer, previous_bot_texts, threshold=repeat_threshold):
            draft_text = fact_answer
            route = "bot_answer_self_for_pilot" if block_a_direct_answer and route != "manager_only" else "draft_for_manager" if route == "manager_only" else route
            flags.append("humanity_repeat_repaired")
            checklist.append("Ответ почти повторял предыдущую реплику; перестроен на текущий вопрос.")
            metadata["humanity_repeat_repaired"] = True
            changed = True
        elif strict_antirepeat or core_handoff_repeat:
            draft_text = _strict_antirepeat_fallback_text(
                context,
                result=replace(result, route=route, draft_text=draft_text, safety_flags=tuple(flags), metadata=metadata),
                client_message=client_message,
            )
            if strict_antirepeat:
                route = "draft_for_manager" if route == "manager_only" else route
            flags.append("humanity_strict_antirepeat_fallback_applied")
            checklist.append("Строгий анти-повтор: ответ заменён на короткий честный ответ/узкий хендофф по текущему уточнению.")
            metadata["humanity_strict_antirepeat_fallback_applied"] = True
            changed = True
        else:
            flags.append("humanity_repeat_detected")
            checklist.append("Ответ похож на предыдущую реплику: перед отправкой переписать под текущий вопрос.")
            metadata["humanity_repeat_detected"] = True
            changed = True

    if p0_required and route != "manager_only" and not is_benign_hypothetical_refund(client_message):
        route = "manager_only"
        flags.append("humanity_p0_route_preserved")
        metadata["humanity_p0_route_preserved"] = True
        changed = True

    if not _humanity_guarded_handoff_reason(result) and not preserve_existing_answer:
        route_action = humanity_route_action(
            p0_required=p0_required,
            has_retrieved_answer_fact=has_answer_fact,
            route=route,
            message_type=result.message_type,
            direct_question_answered=not direct_question_unanswered,
        )
        if route_action.get("regenerate"):
            fact_answer = "" if block_generic_fact_answer else _humanity_fact_answer(context, client_message=client_message)
            if fact_answer:
                draft_text = fact_answer
                action_route = str(route_action.get("route") or route)
                route = "bot_answer_self_for_pilot" if action_route == "bot_answer_self" else action_route
            flags.append("humanity_route_action_applied")
            checklist.append("Факт-ответ уже извлечён: ответить из него напрямую, не ограничиваться передачей менеджеру без P0.")
            metadata["humanity_route_action_applied"] = True
            metadata["humanity_route_action_reason"] = route_action.get("reason")
            metadata["humanity_route_action_route"] = route_action.get("route")
            changed = True

    if not changed:
        return result
    return replace(
        result,
        route=route,
        draft_text=draft_text,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )


def apply_humanity_x2_rewriter(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    rewrite_runner: Optional[Callable[[str], str]] = None,
) -> SubscriptionDraftResult:
    """Optional X2 form rewrite after all deterministic draft guards.

    X2 is disabled by default and never touches P0/manager_only routes. It can
    only replace the customer-facing text after both framework checks and repo
    gates accept the candidate.
    """

    if not _humanity_x2_rewrite_enabled(context):
        return result
    previous_bot_texts = _humanity_previous_bot_texts(context)
    prev_bot = previous_bot_texts[-1] if previous_bot_texts else ""
    prior_openers = tuple(" ".join(str(text or "").casefold().split()[:4]) for text in previous_bot_texts if str(text or "").strip())
    safety_flags_text = " ".join(result.safety_flags)
    turn = {
        "bot_text": result.draft_text,
        "bot_route": result.route,
        "bot_safety_flags": safety_flags_text,
    }
    linter_flags = lint_turn(turn, prev_bot_text=prev_bot, prior_openers=prior_openers)
    metadata = dict(result.metadata)
    metadata["humanity_x2"] = {
        "enabled": True,
        "mode": _humanity_x2_rewrite_mode(context),
        "linter_flags": linter_flags,
    }

    if result.route == "manager_only" or _humanity_p0_required(result):
        metadata["humanity_x2"]["fallback_reason"] = "locked_p0_or_manager_only"
        return replace(result, metadata=metadata)
    if _humanity_x2_identity_policy_locked(result):
        metadata["humanity_x2"]["fallback_reason"] = "locked_identity_policy"
        return replace(result, metadata=metadata)

    confirmed_facts = _humanity_x2_confirmed_facts(context)
    rules_engine_applied = _rules_engine_result_applied(metadata)

    def validate_candidate(candidate: str) -> str | None:
        return _humanity_x2_repo_gate(candidate, result=result, client_message=client_message, context=context)

    def sanitize_candidate(candidate: str) -> str:
        if not rules_engine_applied:
            return candidate
        stripped = strip_internal_service_markers(candidate)
        return stripped or candidate

    rewrite = apply_humanity_form_rewrite(
        turn,
        rewrite_fn=rewrite_runner,
        confirmed_facts=confirmed_facts,
        active_brand=_active_brand(context),
        client_message=client_message,
        linter_flags=linter_flags,
        sanitize_fn=sanitize_candidate,
        validate_fn=validate_candidate,
        mode=_humanity_x2_rewrite_mode(context),
    )
    metadata["humanity_x2"] = {
        **dict(metadata.get("humanity_x2") or {}),
        "rewritten": bool(rewrite.get("rewritten")),
        "fallback_reason": rewrite.get("fallback_reason"),
    }
    if not rewrite.get("rewritten"):
        return replace(result, metadata=metadata)
    draft_text = str(rewrite.get("draft_text") or "").strip()
    if not draft_text:
        metadata["humanity_x2"]["fallback_reason"] = "empty_candidate_after_rewrite"
        return replace(result, metadata=metadata)
    return replace(
        result,
        draft_text=draft_text,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "humanity_x2_rewritten"])),
        metadata=metadata,
    )


def apply_phase2_tone_layer(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _phase2_tone_enabled(context):
        return result
    before = score_tone(result.draft_text)
    metadata = dict(result.metadata)
    metadata["phase2_tone"] = {
        "enabled": True,
        "tone_before": before.as_dict(),
    }
    if result.route == "manager_only" or _humanity_p0_required(result):
        metadata["phase2_tone"]["fallback_reason"] = "locked_p0_or_manager_only"
        return replace(result, metadata=metadata)
    if before.tone_canc <= 0:
        metadata["phase2_tone"]["fallback_reason"] = "tone_ok"
        return replace(result, metadata=metadata)
    rewrite_fn = _phase2_tone_rewrite_override(context)
    candidate = rewrite_fn(result.draft_text) if rewrite_fn is not None else _phase2_tone_rewrite(result.draft_text)
    candidate = str(candidate or "").strip()
    if not candidate or candidate == str(result.draft_text or "").strip():
        metadata["phase2_tone"]["fallback_reason"] = "no_change"
        return replace(result, metadata=metadata)
    violation = _phase2_text_change_violation(result, candidate, client_message=client_message, context=context)
    if violation:
        metadata["phase2_tone"]["fallback_reason"] = violation
        metadata["phase2_tone"]["candidate_rejected"] = True
        return replace(result, metadata=metadata)
    after = score_tone(candidate)
    metadata["phase2_tone"].update(
        {
            "rewritten": True,
            "tone_after": after.as_dict(),
        }
    )
    return replace(
        result,
        draft_text=candidate,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "phase2_tone_rewritten"])),
        metadata=metadata,
    )


_SEMANTIC_OUTPUT_VERIFIER_CODES = frozenset({"derived_product_claim", "invented_generalization", "individual_diagnosis"})


def build_semantic_output_verifier_prompt(
    *,
    bot_text: str,
    client_message: str = "",
    facts: Mapping[str, str] | None = None,
    active_brand: str = "",
    route: str = "",
) -> str:
    facts_block = "\n".join(f"- {key}: {value}" for key, value in (facts or {}).items()) or "(фактов нет)"
    return (
        "Ты — смысловой верификатор финального текста бота учебного центра. "
        "Проверяй только смысловые производные, которые плохо ловятся регулярными правилами. "
        "Не проверяй цены/проценты/бренд/P0/мета: это делает отдельный детерминированный gate.\n\n"
        "Верни СТРОГО JSON:\n"
        '{"findings":[{"code":"derived_product_claim|invented_generalization|individual_diagnosis",'
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
        "менеджеру/преподавателю.\n\n"
        "НЕ ФЛАГАЙ:\n"
        "- дословный или смысловой пересказ факта;\n"
        "- склейку двух реальных фактов без новой приписки;\n"
        "- каноничную фразу разделения брендов;\n"
        "- общий житейский совет с хеджем, если он не делает продуктовый вывод;\n"
        "- хеджированный ответ по ребёнку с передачей преподавателю/менеджеру;\n"
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
        "- Факт: «Фотон: очные цены 49 000 ₽ и 82 000 ₽; онлайн-цена не указана». "
        "Вопрос: «а онлайн?». Ответ: «Стоимость курса — 49 000 ₽ или 82 000 ₽». "
        "Вердикт: derived_product_claim, relation_to_base=adjacent — цена очного формата не подтверждает онлайн-контекст.\n"
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
    if pure_handoff and not handoff_claim_text and (
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
    )
    if unavailable_reason:
        verifier_meta["retry_attempted"] = True
        findings, unavailable_reason = _run_semantic_output_verifier_once(
            verifier,
            result.draft_text,
            client_message=client_message,
            facts=facts,
            active_brand=_active_brand(gate_context),
            route=result.route,
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
) -> tuple[tuple[Mapping[str, Any], ...], str]:
    prompt = build_semantic_output_verifier_prompt(
        bot_text=bot_text,
        client_message=client_message,
        facts=facts,
        active_brand=active_brand,
        route=route,
    )
    try:
        raw_payload = verifier(prompt)
        payload = extract_json_object(raw_payload) if isinstance(raw_payload, str) else raw_payload
    except subprocess.TimeoutExpired:
        return (), "timeout"
    except Exception as exc:  # noqa: BLE001
        return (), str(exc)[:200] or "verifier_error"
    return _semantic_output_findings_from_payload(payload), ""


def _semantic_output_findings_from_payload(payload: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(payload, Mapping):
        return ()
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
        return ()
    findings: list[Mapping[str, Any]] = []
    for raw in raw_findings:
        if not isinstance(raw, Mapping):
            continue
        code = str(raw.get("code") or "").strip()
        if code == "ok" or code not in _SEMANTIC_OUTPUT_VERIFIER_CODES:
            continue
        action = _authoritative_gate_action(code)
        if action not in {"annotate", "downgrade_keep_text"}:
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


def _llm_retrieve_timeout_sec() -> int:
    try:
        return max(1, int(float(os.getenv(LLM_RETRIEVE_TIMEOUT_ENV) or "30")))
    except Exception:
        return 30


def build_semantic_diagnosis_prompt(
    *,
    bot_text: str,
    client_message: str = "",
) -> str:
    return (
        "Ты — строгий классификатор ОДНОГО ответа бота учебного центра. Определи, содержит ли ответ\n"
        "ИНДИВИДУАЛЬНЫЙ ДИАГНОЗ/ГАРАНТИЮ по КОНКРЕТНОМУ ученику: собственную оценку бота, справится ли /\n"
        "подойдёт ли / потянет ли именно этот ребёнок — БЕЗ хеджа неуверенности и БЕЗ передачи менеджеру/преподавателю.\n\n"
        "СЧИТАЕТСЯ диагнозом (true):\n"
        "- утверждение про конкретного ученика: «да, справится», «с тройками можно идти», «потянет», «ему подойдёт»,\n"
        "  «догонять заранее не нужно», «сможет влиться», «слишком тяжело быть не должно», «посильный ритм»,\n"
        "  «подберут под ребёнка» — как оценка бота;\n"
        "- обещание результата/балла конкретному ученику.\n\n"
        "НЕ считается (false):\n"
        "- общая справка о программе/форматах/уровнях: «есть базовый и продвинутый уровень», «программа идёт от азов»,\n"
        "  «формат семинара, мини-группа»;\n"
        "- хеджированный ответ С ПЕРЕДАЧЕЙ: «уровень лучше подобрать на пробном / уточнит преподаватель / сориентирует менеджер»;\n"
        "- ответ про расписание, цены, документы, логистику.\n\n"
        "Верни СТРОГО JSON, без текста вне него:\n"
        '{"individual_diagnosis": true|false, "span": "<цитата ответа, если true; иначе пусто>", "reason": "<кратко>"}\n\n'
        f"Вопрос клиента для контекста:\n{str(client_message or '').strip()}\n\n"
        f"Ответ бота:\n{str(bot_text or '').strip()}\n"
    )


def apply_semantic_diagnosis_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    classifier_fn: Optional[Callable[[str], object]] = None,
) -> SubscriptionDraftResult:
    if not _semantic_diagnosis_guard_enabled(context):
        return result
    metadata = dict(result.metadata)
    guard_meta: dict[str, Any] = {
        "enabled": True,
        "checked": False,
        "rewritten": False,
    }
    metadata["semantic_diagnosis_guard"] = guard_meta
    if _semantic_diagnosis_locked_deferral(result, client_message=client_message):
        guard_meta["fallback_reason"] = "locked_p0_or_high_risk_deferral"
        return replace(result, metadata=metadata)
    if result.route not in {"bot_answer_self", "bot_answer_self_for_pilot", "draft_for_manager", "manager_only"}:
        guard_meta["fallback_reason"] = "unsupported_route"
        return replace(result, metadata=metadata)
    override = _semantic_diagnosis_classifier_override(context)
    classifier = override or classifier_fn
    if classifier is None:
        guard_meta["fallback_reason"] = "classifier_unavailable"
        return replace(result, metadata=metadata)
    prompt = build_semantic_diagnosis_prompt(bot_text=result.draft_text, client_message=client_message)
    try:
        raw_payload = classifier(prompt)
        payload = extract_json_object(raw_payload) if isinstance(raw_payload, str) else raw_payload
    except Exception as exc:  # noqa: BLE001
        guard_meta["fallback_reason"] = "classifier_error"
        guard_meta["error"] = str(exc)[:200]
        return replace(result, metadata=metadata)
    guard_meta["checked"] = True
    if not isinstance(payload, Mapping):
        guard_meta["fallback_reason"] = "classifier_invalid_payload"
        return replace(result, metadata=metadata)
    diagnosis = _truthy_value(payload.get("individual_diagnosis"))
    guard_meta["individual_diagnosis"] = diagnosis
    guard_meta["span"] = str(payload.get("span") or "")[:220]
    guard_meta["reason"] = str(payload.get("reason") or "")[:220]
    if not diagnosis:
        guard_meta["fallback_reason"] = "not_individual_diagnosis"
        return replace(result, metadata=metadata)
    if _has_diagnosis_hedge_and_transfer(result.draft_text):
        guard_meta["fallback_reason"] = "already_hedged_and_transferred"
        return replace(result, metadata=metadata)
    candidate = SEMANTIC_DIAGNOSIS_SAFE_TEXT
    guard_meta["rewritten"] = True
    guard_meta["fallback_reason"] = None
    return replace(
        result,
        draft_text=candidate,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "semantic_diagnosis_guard_rewritten"])),
        manager_checklist=tuple(
            dict.fromkeys(
                [
                    *result.manager_checklist,
                    "Semantic diagnosis guard: не оценивать конкретного ребёнка заочно; сверить уровень с преподавателем/менеджером.",
                ]
            )
        ),
        metadata=metadata,
    )


def _semantic_diagnosis_classifier_override(context: Optional[Mapping[str, Any]]) -> Optional[Callable[[str], object]]:
    if not isinstance(context, Mapping):
        return None
    value = context.get("semantic_diagnosis_classifier_fn")
    return value if callable(value) else None


def _semantic_diagnosis_locked_deferral(result: SubscriptionDraftResult, *, client_message: str = "") -> bool:
    if result.route != "manager_only":
        return False
    if not (
        _humanity_p0_required(result)
        or _hard_p0_in_client_text(client_message)
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


def _hard_p0_in_client_text(text: str) -> bool:
    return bool(set(codes_from_text(text)).intersection(HARD_P0_CODES))


def _phase2_tone_rewrite(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    replacements = (
        (r"\bСориентирую по проверенным данным[:：]?\s*", ""),
        (r"\bсориентирую по проверенным данным[:：]?\s*", ""),
        (r"\bв рамках текущего учебного центра\b", "по этому центру"),
        (r"\bВ рамках текущего учебного центра\b", "По этому центру"),
        (r"\bосуществляется\b", "проходит"),
        (r"\bОсуществляется\b", "Проходит"),
        (r"\bпредоставляется\b", "есть"),
        (r"\bПредоставляется\b", "Есть"),
        (r"\bближайший шаг уточнит менеджер\b", "дальше подскажет менеджер"),
        (r"\bМенеджер уточнит ближайший шаг\b", "Дальше подскажет менеджер"),
    )
    for pattern, repl in replacements:
        value = re.sub(pattern, repl, value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    return value


def _phase2_text_change_violation(
    result: SubscriptionDraftResult,
    candidate: str,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    if draft_has_identity_disclosure(candidate):
        return "identity_disclosure"
    if _humanity_x2_repo_gate(candidate, result=result, client_message=client_message, context=context):
        return "repo_gate"
    facts = _rules_engine_facts(result, context)
    contract = _pipeline_contract(result, active_brand=_active_brand(context), fact_keys=tuple(facts.keys()))
    findings = verify_dialogue_contract_output(
        candidate,
        facts=facts,
        active_brand=_active_brand(context),
        contract=contract,
        client_message=client_message,
        context=context,
        previous_bot_texts=_humanity_previous_bot_texts(context),
    )
    if findings:
        return "verify_output:" + ",".join(dict.fromkeys(finding.code for finding in findings))
    added_anchors = dialogue_contract_new_concrete_anchors(candidate, original=result.draft_text, facts=facts)
    if added_anchors:
        return "new_concrete_anchor"
    return ""


def _phase2_tone_rewrite_override(context: Optional[Mapping[str, Any]]) -> Optional[Callable[[str], str]]:
    if isinstance(context, Mapping):
        value = context.get("phase2_tone_rewrite_fn")
        if callable(value):
            return value
    return None


def _humanity_x2_identity_policy_locked(result: SubscriptionDraftResult) -> bool:
    if str(result.draft_text or "").strip() in {IDENTITY_PROMPT_SAFE_TEXT, IDENTITY_FOTON_SAFE_TEXT, IDENTITY_UNPK_SAFE_TEXT}:
        return True
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    shadow = pipeline.get("rules_engine_intent_shadow") if isinstance(pipeline.get("rules_engine_intent_shadow"), Mapping) else {}
    return str(shadow.get("selected_source") or "") == "identity_policy"


def _asks_installment(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    if _asks_invoice_monthly_payment(value):
        return False
    return has_any_marker(
        value,
        (
            "рассроч",
            "долями",
            "частями",
            "по частям",
            "помесяч",
            "банк",
            "одобр",
            "без процент",
            "без переплат",
        ),
    )


def _asks_invoice_monthly_payment(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    monthly = has_any_marker(value, ("помесяч", "каждый месяц", "ежемесяч", "по месяцам"))
    invoice_or_transfer = has_any_marker(value, ("по счету", "по счёту", "счет", "счёт", "банковск", "перевод", "реквизит"))
    negates_installment = has_any_marker(value, ("не рассроч", "не долями", "не частями", "не через банк", "не про рассроч"))
    return bool(monthly and (invoice_or_transfer or negates_installment))


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


def _is_core_handoff_fallback_repeat(
    text: str,
    previous_bot_texts: Sequence[str],
    *,
    threshold: float,
) -> bool:
    normalized = " ".join(str(text or "").split())
    known_templates = {SAFE_FALLBACK_DRAFT_TEXT, *_HUMANE_GENERIC_HANDOFF_TEXTS}
    if normalized not in {" ".join(item.split()) for item in known_templates}:
        return False
    return is_near_repeat(text, previous_bot_texts, threshold=threshold)


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


def _humanity_allows_dry_p0_text(result: SubscriptionDraftResult, *, p0_required: bool) -> bool:
    if not p0_required:
        return False
    normalized = " ".join(str(result.draft_text or "").split())
    dry_templates = {
        LEGAL_THREAT_SAFE_TEXT,
        LEGAL_THREAT_PII_SAFE_TEXT,
        *_REFUND_ZERO_COLLECT_VARIANTS,
        *_COMPLAINT_SAFE_VARIANTS,
        *_PAYMENT_DISPUTE_VARIANTS,
        *_LEGAL_SAFE_VARIANTS,
    }
    return normalized in {" ".join(template.split()) for template in dry_templates}


def _has_humanity_answer_fact(context: Optional[Mapping[str, Any]]) -> bool:
    return bool(_first_humanity_fact_text(context))


def _humanity_block_a_route_fix_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        value = context.get("humanity_block_a_route_fix_enabled")
        if value is not None:
            return _truthy_value(value)
    env_value = os.getenv(HUMANITY_BLOCK_A_ROUTE_FIX_ENV)
    if env_value is not None:
        return _truthy_value(env_value)
    return True


def _antirepeat_strict_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        value = context.get("antirepeat_strict_enabled")
        if value is not None:
            return _truthy_value(value)
    env_value = os.getenv(ANTIREPEAT_STRICT_ENV)
    if env_value is not None:
        return _truthy_value(env_value)
    return True


def _humanity_can_trim_cosmetic_opening(result: SubscriptionDraftResult) -> bool:
    if result.topic_id in HIGH_RISK_THEME_IDS:
        return False
    money_or_protective_topics = {
        "theme:001_pricing",
        "theme:002_payment_method",
        "theme:003_payment_status",
        "theme:005_discounts",
        "theme:006_installment",
        "theme:007_matkap_payment",
        "theme:008_tax_deduction",
        "theme:009_refund",
        "theme:011_contract",
    }
    if result.topic_id in money_or_protective_topics:
        return False
    if result.message_type in {"non_question", "context_update", "wait_for_more", "manager_only"}:
        return False
    return True


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


def _humanity_block_a_direct_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
    previous_bot_texts: Sequence[str] = (),
) -> str:
    for candidate in (
        _humanity_unpk_address_confirmation_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
        _humanity_presale_refund_rules_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
        _humanity_unpk_tax_certificate_followup_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
        _humanity_foton_bank_transfer_monthly_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
        _humanity_unpk_weekend_address_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
    ):
        if candidate and not is_near_repeat(candidate, previous_bot_texts):
            return candidate
    return ""


def _humanity_presale_refund_rules_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    dialog = _dialog_context_haystack(context)
    asks_where_to_read = has_any_marker(
        text,
        ("где", "почитать", "посмотреть", "договор", "оферт", "правил", "до оплаты", "заранее"),
    )
    refund_context = has_any_marker(text, ("возврат", "вернут", "вернете", "вернёте", "передума", "отказ")) or has_any_marker(
        dialog,
        ("возврат", "вернут", "вернете", "вернёте", "передума", "отказ"),
    )
    if not (asks_where_to_read and refund_context):
        return ""
    known = known_context_fields(context)
    details = []
    for key in ("grade", "subject", "format"):
        value = str(known.get(key) or "").strip()
        if value:
            details.append(value)
    detail_text = f" по {', '.join(details)}" if details else ""
    return (
        f"Да, правила можно посмотреть до оплаты: менеджер пришлёт актуальный договор или оферту{detail_text}, "
        "и там будут условия отказа/возврата. Передам менеджеру запрос именно по условиям возврата до оплаты. "
        "Точную сумму без документа я не буду обещать, но сформулирую запрос именно так: "
        "прислать правила до оплаты, чтобы вы спокойно посмотрели их заранее."
    )


def _humanity_unpk_address_confirmation_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "unpk":
        return ""
    asks_address_confirmation = (
        "сретен" in text
        and (
            "20" in text
            or has_any_marker(text, ("адрес", "подтверд", "да?", "верно", "правильно"))
        )
    )
    if not asks_address_confirmation:
        return ""
    facts = " ".join(_confirmed_fact_texts(context, limit=16)).casefold().replace("ё", "е")
    if "сретенка, 20" not in facts and "сретенка 20" not in facts:
        return ""
    return (
        "Да, верно: регулярные курсы УНПК в Москве проходят на Сретенке, 20, метро Чистые Пруды. "
        "Класс, предмет и очный формат уже вижу; если захотите записываться, останется только сверить конкретную группу и слот."
    )


def _humanity_unpk_tax_certificate_followup_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "unpk":
        return ""
    facts = " ".join([*_confirmed_fact_texts(context, limit=16), current_draft]).casefold().replace("ё", "е")
    dialog = _dialog_context_haystack(context)
    has_tax_fact = "кнд 1151158" in facts or ("налог" in facts and "вычет" in facts)
    if not has_tax_fact:
        return ""
    mentions_certificate = has_any_marker(text, ("справк", "вычет", "налог", "кнд"))
    follows_tax_context = has_any_marker(dialog, ("налог", "вычет", "кнд 1151158"))
    if not mentions_certificate and not (follows_tax_context and has_any_marker(text, ("менеджер", "напишу", "попрошу"))):
        return ""
    return (
        "Да, для налогового вычета нужна справка по форме КНД 1151158. "
        "Менеджер пришлёт шаблон заявления на email, после заявления справку подготовят и отправят в течение 10 рабочих дней. "
        "Лучше так и написать менеджеру: нужна справка для налогового вычета."
    )


def _humanity_foton_bank_transfer_monthly_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "foton":
        return ""
    asks_transfer = has_any_marker(text, ("перевод", "счет", "счёт", "безнал"))
    asks_monthly = has_any_marker(text, ("помесяч", "каждый месяц", "по месяц", "не все сразу", "не всё сразу"))
    if not (asks_transfer and asks_monthly):
        return ""
    known = known_context_fields(context)
    details: list[str] = []
    grade = str(known.get("grade") or "").strip()
    subject = str(known.get("subject") or "").strip()
    course_format = str(known.get("format") or "").strip()
    if grade:
        details.append(f"{grade} класс")
    if subject:
        details.append(subject)
    if course_format:
        details.append(course_format)
    detail_text = f" для {', '.join(details)}" if details else ""
    return (
        f"Понял: вы спрашиваете не про рассрочку и не про Долями, а про то, можно ли помесячно оплачивать переводом на счёт{detail_text}. "
        "Я не буду подставлять сюда условия рассрочки: это другой способ оплаты. "
        "Менеджер проверит, можно ли оформить именно счёт каждый месяц, и даст корректные реквизиты/порядок оплаты."
    )


def _humanity_unpk_weekend_address_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "unpk":
        return ""
    asks_weekend = has_any_marker(text, ("суббот", "воскрес", "выходн", "по каким дням", "дням", "сб", "вс"))
    asks_direct_yes_no = has_any_marker(text, ("да/нет", "да или нет", "просто да", "просто понять", "заранее"))
    mentions_address = "сретен" in text or "там" in text or "москв" in text
    if not (asks_weekend and (asks_direct_yes_no or mentions_address)):
        return ""
    facts = tuple(_confirmed_fact_texts(context, limit=16))
    facts_low = " ".join(facts).casefold().replace("ё", "е")
    if "разные слоты по выходным" not in facts_low:
        return ""
    known = known_context_fields(context)
    grade = str(known.get("grade") or "").strip()
    subject = str(known.get("subject") or "").strip()
    group_text = ""
    if grade and subject:
        group_text = f" для {grade} класса, {subject}"
    elif grade:
        group_text = f" для {grade} класса"
    asks_specific_weekend_days = has_any_marker(text, ("суббот", "воскрес", "сб", "вс"))
    if asks_specific_weekend_days:
        if has_any_marker(text, ("или только", "просто бывают", "просто по выходным", "вообще там", "да или нет", "просто сказать")):
            return (
                f"Если совсем коротко по Сретенке{group_text}: подтверждено, что есть слоты по выходным. "
                "А вот обещать, что нужная группа идёт именно и в субботу, и в воскресенье, я не буду: такого точного факта по конкретной группе нет. "
                "Значит честный ответ такой: выходные — да; конкретный день или оба дня — только после сверки группы."
            )
        return (
            f"Коротко по Сретенке{group_text}: подтверждённый факт — есть разные слоты по выходным. "
            "То есть смотреть нужно выходные дни; но я не буду обещать, что именно ваша группа будет и в субботу, и в воскресенье одновременно без сетки конкретной группы. "
            "Если нужен точный слот, проверяем уже по группе."
        )
    return (
        f"Да: по УНПК на Сретенке ориентир — выходные, есть разные слоты по выходным. "
        f"Точный день и время{group_text} зависят от конкретной группы, поэтому их нужно сверить отдельно; но сам ответ на вопрос «выходные бывают?» — да."
    )


def _humanity_generic_fact_answer_blocked(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
) -> bool:
    """Do not replace an unresolved operational question with a neighboring fact."""
    text = str(client_message or "").casefold().replace("ё", "е")
    missing = " ".join(str(item or "") for item in result.missing_facts).casefold().replace("ё", "е")
    asks_bank_transfer = (
        has_any_marker(text, ("перевод", "счет", "счёт", "безнал"))
        and has_any_marker(text, ("оплат", "платить", "помесяч"))
    )
    if asks_bank_transfer and (
        "payment_methods.current" in missing
        or "способ" in missing
        or "порядок оплаты" in missing
        or "реквизит" in missing
        or "перевод" in missing
    ):
        return True
    asks_matkap_installment_combo = (
        has_any_marker(text, ("маткап", "материнск"))
        and has_any_marker(text, ("рассроч", "долями", "совмещ", "вместе"))
    )
    if asks_matkap_installment_combo and (
        "совмещ" in missing
        or "сочетан" in missing
        or "рассроч" in missing
        or "installment_terms.current" in missing
    ):
        return True
    return False


def _humanity_preserve_existing_answer(result: SubscriptionDraftResult) -> bool:
    if result.route in {"bot_answer_self", "bot_answer_self_for_pilot"}:
        return True
    flags = set(result.safety_flags)
    return any(
        flag.endswith("_safe_template_applied")
        or flag
        in {
            "autonomy_verified_fact_answer_template_applied",
            "pricing_safe_template_applied",
            "camp_safe_template_applied",
            "installment_safe_template_applied",
            "tax_safe_template_applied",
            "trial_safe_template_applied",
            "offline_free_trial_promise_guarded",
            "presale_refund_policy_manager_check",
            "presale_refund_policy_non_p0",
        }
        for flag in flags
    ) or bool(result.metadata.get("presale_refund_policy_manager_check"))


def _humanity_guarded_handoff_reason(result: SubscriptionDraftResult) -> bool:
    flags = set(result.safety_flags)
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if result.message_type in {"non_question", "context_update", "wait_for_more"}:
        return True
    guarded_flags = {
        "autonomy_default_cautious_live_status_missing",
        "future_price_handoff_applied",
        "price_future_manager_only",
        "unsupported_promise_guarded",
        "unconfirmed_operational_specificity_guarded",
        "message_type_non_question",
        "message_type_context_update",
        "message_type_wait_for_more",
    }
    if flags.intersection(guarded_flags):
        return True
    if metadata.get("future_price_handoff_applied") or metadata.get("autonomy_default_cautious_live_status_missing"):
        return True
    return False


def _first_humanity_fact_text(context: Optional[Mapping[str, Any]]) -> str:
    facts = _confirmed_fact_texts(context, limit=8)
    for fact in facts:
        text = _client_clean_fact_text(fact)
        low = text.casefold().replace("ё", "е")
        if not text or "client_blocked:" in low or "internal_only" in low or "клиенту суммы не называть" in low:
            continue
        return text
    return ""


def _humanity_fact_answer(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> str:
    precise_fact_answer = _humanity_precise_fact_answer(context, client_message=client_message)
    if precise_fact_answer:
        return precise_fact_answer
    installment_amount_answer = _humanity_installment_amount_answer(context, client_message=client_message)
    if installment_amount_answer:
        return installment_amount_answer
    fact = _first_humanity_fact_text(context)
    if not fact:
        return ""
    client_low = client_message.casefold().replace("ё", "е")
    fact_low = fact.casefold().replace("ё", "е")
    if "питан" in client_low and "5-разовым питанием" in fact_low and "5-разовое питание" not in fact_low:
        fact = re.sub(r"с\s+проживанием\s+и\s+5-разовым\s+питанием", "с проживанием; 5-разовое питание включено", fact, flags=re.I)
    fact_sentence = _ensure_sentence(fact)
    next_step = _humanity_next_step(client_message=client_message, context=context)
    return " ".join(part for part in (fact_sentence, next_step) if part).strip()


def _humanity_precise_fact_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    discount_percent_answer = _humanity_discount_percent_answer(
        context, client_message=client_message, current_draft=current_draft
    )
    if discount_percent_answer:
        return discount_percent_answer
    return ""


def _humanity_context_correction_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    weekend_schedule_answer = _humanity_weekend_schedule_no_format_lock_answer(
        context, client_message=client_message, current_draft=current_draft
    )
    if weekend_schedule_answer:
        return weekend_schedule_answer
    return ""


def _humanity_weekend_schedule_no_format_lock_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "unpk":
        return ""
    asks_weekend = has_any_marker(text, ("выходн", "суббот", "воскрес", " сб", " вс", "дням", "по каким дням"))
    rejects_format_lock = (
        has_any_marker(text, ("формат не принцип", "не принципиален", "главное выходн", "почему онлайн", "не про формат"))
        or ("формат" in text and "главное" in text)
    )
    if not (asks_weekend and rejects_format_lock):
        return ""
    draft_low = str(current_draft or "").casefold().replace("ё", "е")
    locks_online = "формат уже вижу как онлайн" in draft_low or "если скажете, какой формат" in draft_low
    mentions_online_instead = "онлайн с записью" in draft_low and "разные слоты по выходным" not in draft_low
    if current_draft and not (locks_online or mentions_online_instead):
        return ""
    facts = _confirmed_fact_texts(context, limit=16)
    has_weekend_fact = any("разные слоты по выходным" in str(fact).casefold().replace("ё", "е") for fact in facts)
    if not has_weekend_fact:
        return ""
    return (
        "Формат не фиксирую: вы написали, что главное — выходные. "
        "По УНПК есть разные слоты по выходным, но точные суббота/воскресенье и время зависят от конкретной группы. "
        "Для 9 класса по математике менеджер сверит ближайшие варианты и наличие мест."
    )


def _humanity_discount_percent_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if "скид" not in text:
        return ""
    asks_percent = "%" in text or has_any_marker(text, ("процент", "сколько", "такая же", "так же"))
    if not asks_percent:
        return ""
    if re.search(r"\b\d{1,2}\s*%", str(current_draft or "")):
        return ""
    format_key = ""
    if has_any_marker(text, ("очн", "офлайн")):
        format_key = "offline"
    elif has_any_marker(text, ("онлайн", "дистанц")):
        format_key = "online"
    facts = _confirmed_fact_texts(context, limit=16)
    if not facts:
        return ""

    def matches_format(value: str) -> bool:
        low = value.casefold().replace("ё", "е")
        if format_key == "offline":
            return "очн" in low or "офлайн" in low
        if format_key == "online":
            return "онлайн" in low or "дистанц" in low
        return True

    selected = ""
    for fact in facts:
        low = str(fact or "").casefold().replace("ё", "е")
        if "скид" not in low or "%" not in low or not matches_format(low):
            continue
        if "втор" in text and "втор" not in low:
            continue
        selected = str(fact)
        if "составляет" in low or "действует" in low:
            break
    if not selected:
        return ""
    match = re.search(r"\b\d{1,2}\s*%", selected)
    if not match:
        return ""
    pct = match.group(0).replace(" ", "")
    brand = _active_brand(context)
    if brand == "foton":
        format_label = "Очно" if format_key == "offline" else "Онлайн" if format_key == "online" else "По этому формату"
        base = f"{format_label} на второй предмет в Фотоне скидка {pct}."
    elif brand == "unpk":
        format_label = "очно" if format_key == "offline" else "онлайн" if format_key == "online" else "по этому формату"
        base = f"В УНПК {format_label} скидка по этому вопросу — {pct}."
    else:
        base = f"Скидка по этому вопросу — {pct}."
    stacking = ""
    for fact in facts:
        low = str(fact or "").casefold().replace("ё", "е")
        if "не сумм" in low or "наибольш" in low:
            stacking = " Скидки не суммируются: применяется наибольшая доступная."
            break
    next_step = " Если хотите, дальше менеджер проверит подходящую группу и оформит скидку к заявке."
    return base + stacking + next_step


def _humanity_installment_amount_answer(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "foton":
        return ""
    asks_monthly_payment = (
        has_any_marker(text, ("помесяч", "каждый месяц", "по месяц", "сумм"))
        or bool(re.search(r"\bсколько\b[^.?!\n]{0,80}\b(?:месяц|выходит|платеж|платёж)", text, flags=re.I))
    )
    plan = context.get("conversation_intent_plan") if isinstance(context, Mapping) and isinstance(context.get("conversation_intent_plan"), Mapping) else {}
    plan_intent = str(plan.get("primary_intent") or plan.get("topic_id") or "").casefold()
    asks_installment = (
        _asks_installment(text)
        or has_any_marker(text, ("рассроч", "частями", "долями"))
        or "installment" in plan_intent
        or "theme:006_installment" in plan_intent
    )
    if not (asks_monthly_payment and asks_installment):
        return ""
    price_text = _foton_online_price_text_from_facts(context)
    if not price_text:
        return ""
    return (
        f"{price_text} По ежемесячному платежу не буду делить сумму на глаз: в Фотоне доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями, "
        "а точный платёж зависит от выбранного срока и условий оформления. Менеджер посчитает платеж именно под выбранный вариант."
    )


def _humanity_next_step(*, client_message: str = "", context: Optional[Mapping[str, Any]] = None) -> str:
    brand = _active_brand(context)
    if has_any_marker(client_message, ("мест", "налич", "брон", "запис")):
        return "Если хотите, менеджер проверит наличие и поможет с оформлением."
    if has_any_marker(client_message, ("цен", "стоим", "сколько", "оплат", "рассроч", "долями")):
        return "Если подходит, менеджер поможет подобрать удобный вариант оплаты и оформить запись."
    if brand == "unpk":
        return "Если хотите, менеджер УНПК поможет подобрать следующий шаг."
    if brand == "foton":
        return "Если хотите, менеджер Фотона поможет подобрать следующий шаг."
    return "Если хотите, менеджер поможет подобрать следующий шаг."


def _sanitize_humanity_meta_text(text: str) -> str:
    value = strip_internal_service_markers(text)
    replacements = (
        "Сориентирую по проверенным данным:",
        "По проверенным данным:",
        "по проверенным данным",
        "Такой вопрос до оплаты не оформляю как жалобу или заявление на возврат.",
        "Не оформляю как жалобу или заявление.",
        "не оформляю как жалобу",
        "не оформляю как заявление",
        "Передам ему контекст диалога.",
    )
    for marker in replacements:
        value = value.replace(marker, "")
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip()


def _asks_money_price_question(text: str) -> bool:
    normalized = str(text or "").casefold().replace("ё", "е")
    if has_marker(normalized, "процент") and not has_any_marker(normalized, ("стоим", "цена", "цену", "прайс", "руб", "почем", "почём")):
        return False
    return bool(
        re.search(r"\b(?:стоим\w*|цена|цену|цены|ценой|прайс|почем|почём|руб(?:\.|лей|ля|ль)?)\b", normalized)
        or re.search(r"\bсколько\b[^.!?\n]{0,80}\b(?:стоит|стоим|руб|₽)", normalized)
        or re.search(r"\bсколько\b[^.!?\n]{0,80}\b(?:выходит|плат[её]ж|в\s+месяц|за\s+месяц)", normalized)
    )


def _foton_online_price_text_from_facts(context: Optional[Mapping[str, Any]]) -> str:
    semester = _price_amount_from_facts(context, required_markers=("онлайн",), period_markers=("семестр",))
    year = _price_amount_from_facts(
        context,
        required_markers=("онлайн",),
        period_markers=("год —", "год -", "годовая", "за год"),
        excluded_markers=("семестр",),
    )
    if not semester and not year:
        return ""
    parts = []
    if semester:
        parts.append(f"за семестр — {semester}")
    if year:
        parts.append(f"за год — {year}")
    return (
        f"Для онлайн-обучения в Фотоне сейчас: {', '.join(parts)}. "
        "Цена скоро подрастёт, поэтому если формат подходит, лучше закрепить текущие условия. "
        "Дальше подберём группу под класс, предмет и уровень ребёнка."
    )


def _price_amount_from_facts(
    context: Optional[Mapping[str, Any]],
    *,
    required_markers: Sequence[str],
    period_markers: Sequence[str],
    excluded_markers: Sequence[str] = (),
) -> str:
    facts = _fresh_fact_texts(context) or _confirmed_fact_texts(context, limit=12)
    for fact in facts:
        normalized = str(fact or "").casefold().replace("ё", "е")
        if not all(marker in normalized for marker in required_markers):
            continue
        if any(marker in normalized for marker in excluded_markers):
            continue
        if not any(marker in normalized for marker in period_markers):
            continue
        match = re.search(r"\b\d{1,3}(?:[ \u00a0]\d{3})+(?:\s*(?:₽|руб(?:\.|лей|ля|ль)?))?", str(fact or ""))
        if match:
            amount = " ".join(match.group(0).replace("\u00a0", " ").split())
            return amount if "₽" in amount or "руб" in amount.casefold() else f"{amount} ₽"
    return ""


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


def _phase2_tone_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (PH2_TONE_ENV, "phase2_tone_enabled"):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(PH2_TONE_ENV))


def _semantic_diagnosis_guard_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (SEMANTIC_DIAGNOSIS_GUARD_ENV, "semantic_diagnosis_guard_enabled"):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(SEMANTIC_DIAGNOSIS_GUARD_ENV))


def _semantic_output_verifier_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    # In a future autonomous send mode Дмитрий may choose fail-closed when this
    # verifier is unavailable; today it is advisory in draft-only mode.
    return _pilot_profile_flag_enabled(context, SEMANTIC_OUTPUT_VERIFIER_ENV, aliases=("semantic_output_verifier_enabled",))


def _answer_quality_llm_rewrite_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        value = context.get("answer_quality_llm_rewrite_enabled")
        if value is not None:
            return _truthy_value(value)
    return _truthy_value(os.getenv(ANSWER_QUALITY_LLM_REWRITE_ENV)) or _truthy_value(os.getenv(ANSWER_QUALITY_LLM_REWRITER_ENV))


def _answer_quality_llm_rewrite_mode(context: Optional[Mapping[str, Any]] = None) -> str:
    if isinstance(context, Mapping):
        value = context.get("answer_quality_llm_rewrite_mode")
        if value is not None:
            return str(value or "").strip().casefold()
    return str(os.getenv(ANSWER_QUALITY_LLM_REWRITE_MODE_ENV) or "").strip().casefold()


def _answer_quality_llm_polish_sales_enabled(
    context: Optional[Mapping[str, Any]],
    result: SubscriptionDraftResult,
) -> bool:
    if not _answer_quality_llm_rewrite_enabled(context):
        return False
    mode = _answer_quality_llm_rewrite_mode(context)
    if mode not in {"polish_sales", "always_sales", "all"}:
        return False
    if result.route == "manager_only" or result.topic_id in HIGH_RISK_THEME_IDS:
        return False
    if any(marker in " ".join(result.safety_flags).casefold() for marker in ("high_risk", "zero_collect", "legal", "complaint")):
        return False
    return result.topic_id in {
        "theme:001_pricing",
        "theme:005_discounts",
        "theme:006_installment",
        "theme:013_schedule",
        "theme:014_format",
        "theme:016_program",
        "theme:020_enrollment",
        "theme:023_trial_class",
        "theme:026_camp_general",
        "service:S5_general_consultation",
    }


def _humanity_x2_rewrite_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        value = context.get("humanity_x2_rewrite_enabled")
        if value is not None:
            return _truthy_value(value)
    return _truthy_value(os.getenv(HUMANITY_X2_REWRITE_ENV))


def _humanity_x2_rewrite_mode(context: Optional[Mapping[str, Any]] = None) -> str:
    if isinstance(context, Mapping):
        value = context.get("humanity_x2_rewrite_mode")
        if value is not None:
            mode = str(value or "").strip().casefold()
            return mode if mode in {"linter", "all_eligible"} else "all_eligible"
    mode = str(os.getenv(HUMANITY_X2_REWRITE_MODE_ENV) or "all_eligible").strip().casefold()
    return mode if mode in {"linter", "all_eligible"} else "all_eligible"


def _humanity_x2_confirmed_facts(context: Optional[Mapping[str, Any]]) -> Any:
    if not isinstance(context, Mapping):
        return ()
    for key in ("confirmed_facts", "selected_facts", "facts_context", "gold_answer_context"):
        value = context.get(key)
        if value:
            return value
    return ()


def _extract_humanity_x2_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = extract_json_object(text)
    except Exception:
        return text.strip().strip('"').strip()
    for key in ("draft_text", "answer", "text", "message"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


_HUMANITY_X2_BLOCKING_SANITIZER_FLAGS: tuple[str, ...] = (
    "raw_json_redacted",
    "internal_metadata_redacted",
    "email_redacted",
    "phone_redacted",
    "person_name_redacted",
    "role_name_redacted",
    "document_reference_redacted",
    "brand_normalized",
    "refund_policy_redacted",
    "service_promise_redacted",
)


_HUMANITY_X2_PRESSURE_RE = re.compile(
    r"только\s+сегодня|последн(?:ий|яя)\s+шанс|успейт|решайт[е]?\s+сейчас|"
    r"срочно\s+(?:оформ|запис|реш)|иначе\s+(?:мест|скид|цен)|мест\s+почти\s+нет|"
    r"надо\s+успеть|не\s+тяните|лучше\s+не\s+тянуть",
    re.I,
)


def _humanity_x2_repo_gate(
    candidate: str,
    *,
    result: SubscriptionDraftResult,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str | None:
    if draft_has_identity_disclosure(candidate):
        return "identity_disclosure"
    stripped = strip_internal_service_markers(candidate)
    if stripped != str(candidate or "").strip():
        return "internal_service_marker"
    safety = classify_answer_safety(
        client_message=client_message,
        context=context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
    )
    if safety.blocks_rewriter or safety.p0_required or safety.manager_only:
        return f"answer_safety:{safety.primary_risk or 'manager_only'}"
    if _HUMANITY_X2_PRESSURE_RE.search(candidate):
        return "pressure"
    sanitized = sanitize_answer(candidate, mode="bot")
    if not sanitized.fixpoint_reached or sanitized.status == "fixpoint_not_reached":
        return "sanitize_answer:fixpoint_not_reached"
    for flag in sanitized.flags:
        if flag in _HUMANITY_X2_BLOCKING_SANITIZER_FLAGS:
            return f"sanitize_answer:{flag}"
    if has_meta_leak(candidate):
        return "repo_meta_leak"
    return None
