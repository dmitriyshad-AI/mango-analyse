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
    _deal_action_decision_enabled,
    _direct_path_model_p0_enabled,
    _direct_default_manager_enabled,
    _p0_model_led_complaint_backstop,
    _p0_model_led_enabled,
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
    apply_assumed_scope_guard,
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

from mango_mvp.channels.subscription_llm_parts.post_layers import (
    ANSWER_QUALITY_LLM_REWRITER_ENV,
    ANSWER_QUALITY_LLM_REWRITE_ENV,
    ANSWER_QUALITY_LLM_REWRITE_MODE_ENV,
    ANSWER_QUALITY_LLM_REWRITE_REASONING_ENV,
    ANTIREPEAT_STRICT_ENV,
    AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION,
    A_PROACTIVE_ENV,
    A_RICH_FORMAT_ENV,
    COMPLAINT_APOLOGY_RE,
    COMPLAINT_DETAIL_COLLECT_RE,
    CONTENT_DELIVERY_ACTION_RE,
    COSMETIC_OPENING_RE,
    DERIVED_PRODUCT_NUMBER_RE,
    DIALOGUE_CONTRACT_SEMANTIC_MATCH_MODEL_ENV,
    DIALOGUE_CONTRACT_SEMANTIC_MATCH_REASONING_ENV,
    DIRECT_PATH_REPLACE_TEXT_GATE_CODES,
    DRAFT_PLACEHOLDER_RE,
    FOLLOWUP_DEADLINE_RE,
    GATE_BLOCKING_CODES,
    HIGH_RISK_INPUT_PATTERNS,
    HUMANITY_BLOCK_A_ROUTE_FIX_ENV,
    HUMANITY_X2_REWRITE_ENV,
    HUMANITY_X2_REWRITE_MODEL_ENV,
    HUMANITY_X2_REWRITE_MODE_ENV,
    HUMANITY_X2_REWRITE_REASONING_ENV,
    LEGAL_CONTEXT_INPUT_RE,
    LLM_RETRIEVE_MODEL_ENV,
    LLM_RETRIEVE_REASONING_ENV,
    LLM_RETRIEVE_TIMEOUT_ENV,
    MANAGER_ACTION_PROMISE_ACTION_RE,
    MANAGER_ACTION_PROMISE_ACTOR_RE,
    MANAGER_ACTION_PROMISE_DEADLINE_RE,
    MANAGER_HANDOFF_REQUEST_SAFE_TEXT,
    NIGHT_HOURS_NOTE_ENV,
    NIGHT_HOURS_NOTE_TEXT,
    OFFLINE_VISIT_INVITATION_RE,
    OUTPUT_SANITIZER_BAD_TONE_PHRASE_RE,
    OUTPUT_SANITIZER_CLIENT_TEXT_RE,
    OUTPUT_SANITIZER_MANAGER_TAG_INSTRUCTION_RE,
    OUTPUT_SANITIZER_MANAGER_TAG_RE,
    OUTPUT_SANITIZER_META_LINE_RE,
    OUTPUT_SANITIZER_OPTION_LINE_RE,
    OUTPUT_SANITIZER_PLACEHOLDER_RE,
    OUTPUT_SANITIZER_RAW_DETAIL_HANDOFF_RE,
    OUTPUT_SANITIZER_SEPARATOR_LINE_RE,
    PH2_TONE_ENV,
    PRESALE_PII_CHILD_NAME_KEY_RE,
    PRESALE_PII_NAME_KEY_RE,
    PRESALE_PII_PHONE_KEY_RE,
    PRESALE_RU_META_LINE_RE,
    PRESALE_SOURCE_ID_PHRASE_RE,
    PRESALE_SOURCE_ID_TOKEN_PATTERN,
    PRESALE_SOURCE_ID_TOKEN_RE,
    PRICE_FIX_PROCESS_SAFE_TEXT,
    REFUND_FORBIDDEN_DETAIL_RE,
    SCHEDULE_ASSUMPTION_RE,
    SEMANTIC_DIAGNOSIS_GUARD_ENV,
    SEMANTIC_DIAGNOSIS_MODEL_ENV,
    SEMANTIC_DIAGNOSIS_REASONING_ENV,
    SEMANTIC_DIAGNOSIS_SAFE_TEXT,
    SEMANTIC_OUTPUT_VERIFIER_MODEL_ENV,
    SEMANTIC_OUTPUT_VERIFIER_REASONING_ENV,
    SEMANTIC_OUTPUT_VERIFIER_SCHEMA_VERSION,
    SEMANTIC_OUTPUT_VERIFIER_TIMEOUT_ENV,
    SEMANTIC_VERIFIER_DOWNGRADE_REASON,
    SEMANTIC_VERIFIER_UNAVAILABLE_REASON,
    UNSUPPORTED_FOLLOWUP_DEADLINE_SAFE_TEXT,
    UNSUPPORTED_OFFLINE_VISIT_INVITATION_SAFE_TEXT,
    UNSUPPORTED_SCHEDULE_ASSUMPTION_SAFE_TEXT,
    ZERO_COLLECT_DRAFT_RE,
    _A2_EMOJI_RE,
    _A2_FAKE_DONE_RE,
    _A2_SERIOUS_TAGS,
    _A2_TIME_RE,
    _CLIENT_NAME_MARKER_RE,
    _CLIENT_NAME_PAIR_RE,
    _CLIENT_NAME_STOPWORDS,
    _CLIENT_PII_CONFIRMATION_RE,
    _CLIENT_RELATION_NAME_STOPWORDS,
    _CLIENT_SELF_NAME_MARKER_RE,
    _DRAFT_PERSON_NAME_CONTEXT_RE,
    _HUMANE_DETAIL_HANDOFF_TEXTS,
    _HUMANE_GENERIC_HANDOFF_TEXTS,
    _HUMANITY_X2_BLOCKING_SANITIZER_FLAGS,
    _HUMANITY_X2_PRESSURE_RE,
    _MANAGER_CONTACT_PROMISE_PATTERNS,
    _SEMANTIC_OUTPUT_VERIFIER_CODES,
    _TONE_CLOSE_ADVERSATIVE_RE,
    _TONE_CLOSE_CONTACT_CTA_RE,
    _TONE_CLOSE_CONTACT_TEXTS,
    _TONE_CLOSE_EXIT_SIGNAL_RE,
    _TONE_CLOSE_GRATITUDE_RE,
    _TONE_CLOSE_P0_FLAGS,
    _TONE_CLOSE_PENDING_REFERENCE_RE,
    _TONE_CLOSE_PROBLEM_MARKER_RE,
    _TONE_CLOSE_QUESTION_RE,
    _TONE_CLOSE_REFUSAL_RE,
    _TONE_CLOSE_RETURN_TEXTS,
    _TONE_CLOSE_STEP_CTA_RE,
    _TONE_CLOSE_TRIAL_CTA_RE,
    _TONE_CLOSE_TRIAL_TEXTS,
    _TONE_CLOSE_UNANSWERED_RE,
    _TONE_SELL_PROMPT_STEP_RE,
    _a2_apply_rich_format_guard,
    _a2_contact_capture_handoff,
    _a2_context_phone_known,
    _a2_context_tag,
    _a2_enforce_emoji_limit,
    _a2_has_time,
    _a2_is_proactive_result,
    _a2_mask_phone,
    _a2_p0_or_high_risk,
    _a2_phone_echoed,
    _a2_proactive_enabled,
    _a2_rich_format_enabled,
    _answer_quality_llm_polish_sales_enabled,
    _answer_quality_llm_rewrite_enabled,
    _answer_quality_llm_rewrite_mode,
    _antirepeat_strict_enabled,
    _asks_installment,
    _asks_invoice_monthly_payment,
    _asks_money_price_question,
    _authoritative_gate_a2_findings,
    _authoritative_gate_action,
    _authoritative_gate_derived_product_number_findings,
    _authoritative_gate_direct_path_keep_text,
    _authoritative_gate_downgraded_route,
    _authoritative_gate_existing_guard_findings,
    _authoritative_gate_fact_texts,
    _authoritative_gate_finding,
    _authoritative_gate_findings,
    _authoritative_gate_has_pipeline,
    _authoritative_gate_p0_already_guarded,
    _authoritative_gate_semantic_output_findings,
    _authoritative_gate_skip_backed_finding,
    _authoritative_gate_slot_text,
    _authoritative_gate_text_guard_findings,
    _authoritative_gate_verified_content_flag,
    _authoritative_guard_changed,
    _client_dialogue_allowed_names,
    _client_dialogue_child_first_names,
    _client_dialogue_parent_names,
    _client_name_allowed,
    _client_name_echoed,
    _client_name_echoes,
    _client_name_stopwords,
    _client_pii_echo_context,
    _client_pii_manager_items,
    _client_pii_slot_context_lines,
    _client_pii_slot_context_lines_as_containers,
    _core_handoff_detail,
    _current_moscow_hour,
    _dedupe_gate_findings,
    _default_autonomy_flip_enabled,
    _derived_product_number_claims,
    _derived_product_number_manager_notes,
    _dialogue_contract_safety_flags,
    _dialogue_contract_style_examples,
    _dialogue_contract_tone_guide,
    _direct_path_finalize_metadata,
    _direct_path_generic_replacement_text,
    _direct_path_p0_text,
    _direct_path_preblocked_result,
    _direct_path_prepare_model_result,
    _extract_humanity_x2_text,
    _first_humanity_fact_text,
    _flexible_name_pattern,
    _format_choice_is_disjunctive_question,
    _foton_online_price_text_from_facts,
    _hard_p0_in_client_text,
    _has_diagnosis_hedge_and_transfer,
    _has_humanity_answer_fact,
    _has_manager_contact_promise,
    _humanity_allows_dry_p0_text,
    _humanity_block_a_direct_answer,
    _humanity_block_a_route_fix_enabled,
    _humanity_can_trim_cosmetic_opening,
    _humanity_context_correction_answer,
    _humanity_discount_percent_answer,
    _humanity_fact_answer,
    _humanity_foton_bank_transfer_monthly_answer,
    _humanity_generic_fact_answer_blocked,
    _humanity_guarded_handoff_reason,
    _humanity_installment_amount_answer,
    _humanity_next_step,
    _humanity_p0_required,
    _humanity_precise_fact_answer,
    _humanity_presale_refund_rules_answer,
    _humanity_preserve_existing_answer,
    _humanity_unpk_address_confirmation_answer,
    _humanity_unpk_tax_certificate_followup_answer,
    _humanity_unpk_weekend_address_answer,
    _humanity_weekend_schedule_no_format_lock_answer,
    _humanity_x2_confirmed_facts,
    _humanity_x2_identity_policy_locked,
    _humanity_x2_repo_gate,
    _humanity_x2_rewrite_enabled,
    _humanity_x2_rewrite_mode,
    _identity_phrase_present,
    _is_core_handoff_fallback_repeat,
    _llm_retrieve_timeout_sec,
    _manager_deadline_promise_detail,
    _name_word_pattern,
    _night_hours_note_enabled,
    _normalize_derived_number_surface,
    _normalize_semantic_relation,
    _normalized_handoff_template_text,
    _operational_specificity_guarded_result,
    _output_sanitizer_degenerate,
    _output_sanitizer_enabled,
    _outside_moscow_work_hours,
    _phase2_text_change_violation,
    _phase2_tone_enabled,
    _phase2_tone_rewrite,
    _phase2_tone_rewrite_override,
    _price_amount_from_facts,
    _raw_detail_handoff_looks_like_question,
    _rules_engine_result_applied,
    _run_semantic_output_verifier_once,
    _safe_template_yield_before_fallback,
    _sanitize_client_pii_echo,
    _sanitize_dialogue_contract_client_text,
    _sanitize_humanity_meta_text,
    _sanitize_output_client_text,
    _sanitize_presale_ru_meta_lines,
    _sanitize_presale_source_id_text,
    _sanitize_raw_detail_handoff_match,
    _sanitize_raw_detail_handoff_text,
    _semantic_diagnosis_classifier_override,
    _semantic_diagnosis_guard_enabled,
    _semantic_diagnosis_high_risk_flagged,
    _semantic_diagnosis_locked_deferral,
    _semantic_diagnosis_plain_deferral_text,
    _semantic_output_filter_findings,
    _semantic_output_finding_detail,
    _semantic_output_findings_from_payload,
    _semantic_output_manager_note,
    _semantic_output_verifier_enabled,
    _semantic_output_verifier_highest_action,
    _semantic_output_verifier_override,
    _semantic_output_verifier_timeout_sec,
    _semantic_verifier_is_whitelisted_pure_handoff,
    _strict_antirepeat_fallback_text,
    _tone_close_contact_requested_after_step,
    _tone_close_contact_requested_from_memory,
    _tone_close_detect_is_close_message,
    _tone_close_detect_is_p0,
    _tone_close_has_unanswered_or_problem_continuation,
    _tone_close_message_references_pending,
    _tone_close_metadata,
    _tone_close_next_step_text,
    _tone_close_old_p0_history,
    _tone_close_pending_manager,
    _tone_close_pending_text,
    _tone_close_previous_contact_requested,
    _tone_close_previous_trial_requested,
    _tone_close_refused_previous_step,
    _tone_sell_prompt_step_observation,
    _topic_id_from_context,
    _trim_repeated_cosmetic_opening,
    _unexpected_client_name_echoes,
    _unsupported_claims_by_pattern,
    _verifier_handoff_claims_enabled,
    apply_a2_proactive_layer,
    apply_deal_action_decision_layer,
    apply_authoritative_output_gate,
    apply_bot_safe_memory_step_guard,
    apply_humanity_guards,
    apply_humanity_x2_rewriter,
    apply_night_hours_note,
    apply_output_sanitizer,
    apply_phase2_tone_layer,
    apply_semantic_diagnosis_guard,
    apply_semantic_output_verifier,
    apply_tone_close_detect_layer,
    apply_tone_sell_prompt_observer,
    apply_unconfirmed_operational_specificity_guard,
    apply_unsupported_promise_guard,
    build_semantic_diagnosis_prompt,
    build_semantic_output_regen_prompt,
    build_semantic_output_verifier_prompt,
    draft_has_identity_disclosure,
    draft_has_internal_service_markers,
    find_identity_disclosure_phrases,
    find_unsupported_content_delivery_action_claims,
    find_unsupported_followup_deadline_claims,
    find_unsupported_offline_visit_invitation_claims,
    find_unsupported_schedule_assumption_claims,
    guard_draft_placeholder,
    guard_identity_disclosure,
    guard_promocode_leak,
)

_Runner = Callable[..., subprocess.CompletedProcess[str]]


def _direct_path_autonomy_matrix_topic_result(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not isinstance(context, Mapping):
        return result
    plan = context.get("conversation_intent_plan")
    if not isinstance(plan, Mapping):
        return result
    topic = str(plan.get("topic_id") or "").strip()
    if not topic or topic == result.topic_id:
        return result
    metadata = dict(result.metadata)
    metadata["direct_path_autonomy_topic_from"] = result.topic_id
    metadata["direct_path_autonomy_topic"] = topic
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping):
        direct_meta = dict(direct)
        direct_meta["autonomy_topic_from"] = result.topic_id
        direct_meta["autonomy_topic"] = topic
        metadata["direct_path"] = direct_meta
    flags = tuple(dict.fromkeys((*result.safety_flags, "direct_path_autonomy_topic_from_plan")))
    return replace(result, topic_id=topic, safety_flags=flags, metadata=metadata)


class SubscriptionLlmDraftProvider:
    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
        timeout_sec: int = 90,
        max_attempts: int = 2,
        cache_dir: Optional[Path | str] = None,
        dialogue_contract_semantic_match_fn: Optional[Callable[[str], object]] = None,
        dialogue_contract_semantic_match_enabled: bool = True,
        runner: Optional[_Runner] = None,
        sleep: Callable[[float], None] = time.sleep,
        base_env: Optional[Mapping[str, str]] = None,
        codex_isolated: bool = False,
    ) -> None:
        self.codex_bin = str(codex_bin or "codex").strip() or "codex"
        self.model = str(model or DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL
        self.reasoning_effort = str(reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT).strip() or DEFAULT_CODEX_REASONING_EFFORT
        self.timeout_sec = max(1, int(timeout_sec))
        self.max_attempts = max(1, int(max_attempts))
        self.runner = runner or subprocess.run
        self.sleep = sleep
        self.base_env = dict(base_env) if base_env is not None else None
        self.codex_isolated = bool(codex_isolated)
        self.cache_dir = _guard_cache_dir(cache_dir) if cache_dir is not None else None
        self._dialogue_contract_semantic_match_override = dialogue_contract_semantic_match_fn
        self._dialogue_contract_semantic_match_enabled = bool(dialogue_contract_semantic_match_enabled)

    def _build_codex_command(
        self,
        *,
        output_path: Path,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        isolated_cwd: Optional[Path] = None,
    ) -> list[str]:
        return build_codex_exec_command(
            output_path=output_path,
            codex_bin=self.codex_bin,
            model=model or self.model,
            reasoning_effort=reasoning_effort or self.reasoning_effort,
            isolated=self.codex_isolated,
            cwd=isolated_cwd,
        )

    def build_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        if _direct_path_enabled(context):
            direct_result = self._build_direct_path_draft(client_message, context=context)
            if _deal_action_decision_enabled(context):
                direct_result = _direct_path_autonomy_matrix_topic_result(direct_result, context=context)
                direct_result = apply_autonomy_matrix_guard(direct_result, client_message=client_message, context=context)
            return apply_deal_action_decision_layer(
                direct_result,
                client_message=client_message,
                context=context,
            )
        if dialogue_contract_pipeline_enabled(context):
            result = self._build_dialogue_contract_pipeline_draft(client_message, context=context)
            guarded = self._apply_dialogue_contract_v2_guard_chain(result, client_message=client_message, context=context)
            rewritten = apply_humanity_x2_rewriter(
                guarded,
                client_message=client_message,
                context=context,
                rewrite_runner=self._humanity_x2_rewrite_runner
                if _humanity_x2_rewrite_enabled(context)
                else None,
            )
            toned = apply_phase2_tone_layer(rewritten, client_message=client_message, context=context)
            proactive = apply_a2_proactive_layer(toned, client_message=client_message, context=context)
            closed = apply_tone_close_detect_layer(proactive, client_message=client_message, context=context)
            observed = apply_tone_sell_prompt_observer(closed, client_message=client_message, context=context)
            semantic_checked = apply_semantic_output_verifier(
                observed,
                client_message=client_message,
                context=context,
                verifier_fn=self._semantic_output_verifier_runner_for_context(context),
                regen_fn=self._semantic_output_regen_runner,
            )
            if not _semantic_output_verifier_enabled(context):
                semantic_checked = apply_semantic_diagnosis_guard(
                    semantic_checked,
                    client_message=client_message,
                    context=context,
                    classifier_fn=self._semantic_diagnosis_guard_runner
                    if _semantic_diagnosis_guard_enabled(context)
                    else None,
                )
            if _deal_action_decision_enabled(context):
                semantic_checked = apply_autonomy_matrix_guard(semantic_checked, client_message=client_message, context=context)
            return apply_deal_action_decision_layer(
                apply_authoritative_output_gate(semantic_checked, client_message=client_message, context=context),
                client_message=client_message,
                context=context,
            )
        else:
            prompt = build_draft_prompt(client_message, context=context)
            result = self.generate_from_prompt(prompt, force_manager_only=should_force_manager_only(context))
        result = apply_payment_confirmation_guard(result, client_message=client_message, context=context)
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_conversation_intent_plan_guard(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        result = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_unconfirmed_operational_specificity_guard(result, context=context)
        result = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        result = apply_funnel_policy_guard(result, context=context)
        result = apply_answer_quality_rewriter(
            result,
            client_message=client_message,
            context=context,
            rewrite_runner=self._answer_quality_llm_rewrite_runner
            if _answer_quality_llm_rewrite_enabled(context)
            else None,
            force_llm_polish=_answer_quality_llm_polish_sales_enabled(context, result),
        )
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_conversation_intent_plan_guard(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        result = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_unconfirmed_operational_specificity_guard(result, context=context)
        result = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        result = apply_funnel_policy_guard(result, context=context)
        result = apply_autonomy_matrix_guard(result, client_message=client_message, context=context)
        result = apply_humanity_guards(result, client_message=client_message, context=context)
        result = apply_humanity_x2_rewriter(
            result,
            client_message=client_message,
            context=context,
            rewrite_runner=self._humanity_x2_rewrite_runner
            if _humanity_x2_rewrite_enabled(context)
            else None,
        )
        result = apply_phase2_tone_layer(result, client_message=client_message, context=context)
        result = apply_a2_proactive_layer(result, client_message=client_message, context=context)
        result = apply_tone_close_detect_layer(result, client_message=client_message, context=context)
        result = apply_tone_sell_prompt_observer(result, client_message=client_message, context=context)
        result = apply_semantic_output_verifier(
            result,
            client_message=client_message,
            context=context,
            verifier_fn=self._semantic_output_verifier_runner_for_context(context),
            regen_fn=self._semantic_output_regen_runner,
        )
        if not _semantic_output_verifier_enabled(context):
            result = apply_semantic_diagnosis_guard(
                result,
                client_message=client_message,
                context=context,
                classifier_fn=self._semantic_diagnosis_guard_runner
                if _semantic_diagnosis_guard_enabled(context)
                else None,
            )
        return apply_deal_action_decision_layer(
            apply_authoritative_output_gate(result, client_message=client_message, context=context),
            client_message=client_message,
            context=context,
        )

    def _build_direct_path_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        llm_retrieve = _llm_retrieve_enabled(context)
        if llm_retrieve:
            empty_pack = _direct_path_empty_fact_pack(_active_brand(context), selected_category="preblocked_before_llm_retrieve")
            preblocked = _direct_path_preblocked_result(client_message, context=context, facts={}, fact_pack=empty_pack)
            if preblocked is not None:
                before_gate_route = preblocked.route
                gated = apply_authoritative_output_gate(preblocked, client_message=client_message, context=context)
                return _direct_path_finalize_metadata(
                    gated,
                    before_gate_route=before_gate_route,
                    client_message=client_message,
                    context=context,
                )
        fact_pack = _direct_path_context_fact_pack(
            context,
            client_message=client_message,
            retriever_fn=self._direct_path_llm_retrieve_runner if llm_retrieve else None,
        )
        facts = dict(fact_pack.get("facts") or {})
        if not llm_retrieve:
            preblocked = _direct_path_preblocked_result(client_message, context=context, facts=facts, fact_pack=fact_pack)
            if preblocked is not None:
                before_gate_route = preblocked.route
                gated = apply_authoritative_output_gate(preblocked, client_message=client_message, context=context)
                return _direct_path_finalize_metadata(
                    gated,
                    before_gate_route=before_gate_route,
                    client_message=client_message,
                    context=context,
                )

        active_brand = _active_brand(context)
        pilot_config = _direct_path_pilot_config(context)
        gold_examples = _direct_path_select_gold_real_examples(client_message, context=context, active_brand=active_brand)
        prompt = _build_direct_path_prompt(client_message, context=context, facts=facts, fact_pack=fact_pack, gold_examples=gold_examples)
        direct_meta = _direct_path_metadata(
            attempted=True,
            model_called=True,
            facts=facts,
            fact_pack=fact_pack,
            gold_examples=gold_examples,
            pilot_config=pilot_config,
            context=context,
        )
        try:
            result = self._direct_path_draft_runner(prompt)
        except subprocess.TimeoutExpired:
            direct_meta.update(
                {
                    "text_composition_source": "provider_runtime_fallback",
                    "reason_class": "provider_runtime",
                    "reason_evidence": {"provider_error": "timeout"},
                    "is_manager_deferral": True,
                }
            )
            result = safe_fallback_draft(reason="timeout", metadata={"direct_path": direct_meta})
        except FileNotFoundError:
            direct_meta.update(
                {
                    "text_composition_source": "provider_runtime_fallback",
                    "reason_class": "provider_runtime",
                    "reason_evidence": {"provider_error": "codex_binary_not_found"},
                    "is_manager_deferral": True,
                }
            )
            result = safe_fallback_draft(reason="codex_binary_not_found", metadata={"direct_path": direct_meta, "codex_bin": self.codex_bin})
        except Exception as exc:  # noqa: BLE001
            direct_meta.update(
                {
                    "text_composition_source": "provider_runtime_fallback",
                    "reason_class": "provider_runtime",
                    "reason_evidence": {"provider_error": str(exc)[:300]},
                    "is_manager_deferral": True,
                }
            )
            result = safe_fallback_draft(reason="direct_path_error", metadata={"direct_path": direct_meta, "last_error": str(exc)[:400]})
        else:
            result = _direct_path_prepare_model_result(result)
            if _direct_path_route_rubric_should_regenerate(result, context=context, facts=facts, model_called=True):
                direct_meta["rubric_reason"] = "missing_justification"
                regen_prompt = _build_direct_path_route_rubric_regen_prompt(prompt, result)
                try:
                    result = _direct_path_prepare_model_result(self._direct_path_draft_runner(regen_prompt))
                    direct_meta["rubric_regenerated"] = True
                except Exception as exc:  # noqa: BLE001
                    direct_meta["rubric_reason"] = f"regen_failed:{str(exc)[:160]}"
            result = _direct_path_merge_metadata(result, direct_meta)
            result = _apply_direct_path_model_p0_route(
                result,
                client_message=client_message,
                context=context,
            )
            result = apply_assumed_scope_guard(result, context=context)

        semantic_checked = apply_semantic_output_verifier(
            result,
            client_message=client_message,
            context=context,
            verifier_fn=self._semantic_output_verifier_runner_for_context(context),
            regen_fn=self._semantic_output_regen_runner,
        )
        semantic_checked = apply_bot_safe_memory_step_guard(semantic_checked, context=context)
        before_gate_route = semantic_checked.route
        gated = apply_authoritative_output_gate(semantic_checked, client_message=client_message, context=context)
        return _direct_path_finalize_metadata(
            gated,
            before_gate_route=before_gate_route,
            client_message=client_message,
            context=context,
        )

    def _build_dialogue_contract_pipeline_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        active_brand = _active_brand(context)
        conversation = build_dialogue_contract_conversation(client_message, context=context)
        fact_store = build_dialogue_contract_fact_store(active_brand=active_brand, context=context)
        semantic_match_fn = (
            self._dialogue_contract_semantic_match_override
            if self._dialogue_contract_semantic_match_override is not None
            else self._dialogue_contract_semantic_match_runner
            if self._dialogue_contract_semantic_match_enabled
            else None
        )
        pipeline_result = run_dialogue_contract_pipeline(
            conversation=conversation,
            active_brand=active_brand,
            fact_store=fact_store,
            understand_fn=self._dialogue_contract_understanding_runner,
            draft_fn=self._dialogue_contract_draft_runner,
            repair_fn=self._dialogue_contract_repair_runner,
            faithfulness_fn=self._dialogue_contract_faithfulness_runner,
            semantic_match_fn=semantic_match_fn,
            warmth_fn=None,
            context=context,
            tone_guide=_dialogue_contract_tone_guide(context),
            style_examples=_dialogue_contract_style_examples(context),
            toggles=DialogueContractToggles(form_warmth=False, warmth_mode=_humanity_x2_rewrite_mode(context)),
        )
        route = "bot_answer_self_for_pilot" if pipeline_result.route == "bot_answer_self" else pipeline_result.route
        payload = {
            "message_type": "manager_only" if pipeline_result.manager_only else "question",
            "broad_group": "dialogue_contract_pipeline",
            "topic_id": _topic_id_from_context(context),
            "confidence_theme": pipeline_result.contract.confidence,
            "confidence_group": pipeline_result.contract.confidence,
            "risk_level": "high" if pipeline_result.contract.is_p0 else "low",
            "route": route,
            "draft_text": pipeline_result.draft_text,
            "manager_checklist": [
                "Параллельный dialogue-contract pipeline: проверить смысл до включения в проде.",
                *(
                    [f"Выходной верификатор: {finding.code} — {finding.detail}" for finding in pipeline_result.findings]
                    if pipeline_result.findings
                    else []
                ),
            ],
            "missing_facts": list(pipeline_result.missing),
            "forbidden_promises_detected": [
                *[finding.code for finding in pipeline_result.findings],
                *[f"unsupported_claim:{item}" for item in pipeline_result.unsupported_claims],
            ],
            "safety_flags": _dialogue_contract_safety_flags(pipeline_result),
            "context_used": ["dialogue_contract", "client_safe_fact_store", "output_verifier"],
            "context_warnings": [pipeline_result.fallback_reason] if pipeline_result.fallback_reason else [],
            "metadata": {
                "dialogue_contract_pipeline": {
                    "contract": pipeline_result.contract.to_json_dict(),
                    "retrieved_fact_keys": list(pipeline_result.facts.keys()),
                    "retrieved_facts": dict(pipeline_result.facts),
                    "missing_fact_keys": list(pipeline_result.missing),
                    "findings": [{"code": f.code, "detail": f.detail} for f in pipeline_result.findings],
                    "unsupported_claims": list(pipeline_result.unsupported_claims),
                    "form_findings": [{"code": f.code, "detail": f.detail} for f in pipeline_result.form_findings],
                    "faithfulness_shadow": list(dialogue_contract_faithfulness_shadow_events(context)),
                    "warmth_attempted": pipeline_result.warmth_attempted,
                    "warmth_mode": pipeline_result.warmth_mode,
                    "warmth_rejected_reason": pipeline_result.warmth_rejected_reason,
                    "warmth_rejected_findings": [
                        {"code": f.code, "detail": f.detail} for f in pipeline_result.warmth_rejected_findings
                    ],
                    "warmth_rejected_unsupported": list(pipeline_result.warmth_rejected_unsupported),
                    "warmth_semantic_available": pipeline_result.warmth_semantic_available,
                    "semantic_match_attempted": pipeline_result.semantic_match_attempted,
                    "semantic_match_replaced": pipeline_result.semantic_match_replaced,
                    "semantic_match_reason": pipeline_result.semantic_match_reason,
                    "fallback_reason": pipeline_result.fallback_reason,
                    "is_manager_deferral": bool(getattr(pipeline_result, "is_manager_deferral", False)),
                    "reason_class": str(getattr(pipeline_result, "reason_class", "") or ""),
                    "reason_evidence": dict(getattr(pipeline_result, "reason_evidence", {}) or {}),
                    "recovery_candidate": pipeline_result.recovery_candidate,
                    "recovery_candidate_validated": bool(pipeline_result.recovery_candidate),
                    "partial_yield_applied": bool(getattr(pipeline_result, "partial_yield_applied", False)),
                    "partial_yield_fact_keys": list(getattr(pipeline_result, "partial_yield_fact_keys", ())),
                    "partial_yield_missing": list(getattr(pipeline_result, "partial_yield_missing", ())),
                    "composite_applied": bool(getattr(pipeline_result, "composite_applied", False)),
                    "composite_fact_keys": list(getattr(pipeline_result, "composite_fact_keys", ())),
                    "composite_missing": list(getattr(pipeline_result, "composite_missing", ())),
                    "next_step_applied": bool(getattr(pipeline_result, "next_step_applied", False)),
                    "next_step_text": str(getattr(pipeline_result, "next_step_text", "") or ""),
                    "text_composition_source": str(getattr(pipeline_result, "text_composition_source", "") or ""),
                    "estimate": {
                        "is_estimate": bool(pipeline_result.is_estimate),
                        "estimate_applied": bool(getattr(pipeline_result, "estimate_applied", False) or pipeline_result.is_estimate),
                        "answer_mode": pipeline_result.estimate_answer_mode,
                        "estimate_domain": pipeline_result.estimate_domain,
                    },
                    "warmed": pipeline_result.warmed,
                    "repaired": pipeline_result.repaired,
                }
            },
        }
        if should_force_manager_only(context) and route != "manager_only":
            payload["route"] = "manager_only"
            payload["safety_flags"].append("forced_manager_only_by_rop_policy")
        return normalize_subscription_draft_payload(payload)

    def _dialogue_contract_understanding_runner(self, prompt: str) -> Mapping[str, Any]:
        try:
            raw = self._run_prompt_text(
                prompt,
                prefix="mango_dialogue_contract_understanding_",
                suffix=".json",
                reasoning_effort=self.reasoning_effort,
            )
        except subprocess.TimeoutExpired:
            return {
                "answerability": "manager_only",
                "confidence": 0.0,
                "runtime_error": "understanding_timeout",
            }
        try:
            return extract_json_object(raw)
        except Exception:
            return {}

    def _dialogue_contract_draft_runner(self, prompt: str) -> str:
        return self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_draft_",
            suffix=".txt",
            reasoning_effort=self.reasoning_effort,
        )

    def _dialogue_contract_faithfulness_runner(self, prompt: str) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_faithfulness_",
            suffix=".json",
            reasoning_effort=os.getenv("TELEGRAM_DIALOGUE_CONTRACT_FAITHFULNESS_REASONING") or "medium",
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw

    def _dialogue_contract_semantic_match_runner(self, prompt: str) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_semantic_match_",
            suffix=".json",
            model=os.getenv(DIALOGUE_CONTRACT_SEMANTIC_MATCH_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(DIALOGUE_CONTRACT_SEMANTIC_MATCH_REASONING_ENV) or "medium",
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw

    def _semantic_diagnosis_guard_runner(self, prompt: str) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_semantic_diagnosis_guard_",
            suffix=".json",
            model=os.getenv(SEMANTIC_DIAGNOSIS_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(SEMANTIC_DIAGNOSIS_REASONING_ENV) or "low",
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw

    def _semantic_output_verifier_runner_for_context(self, context: Optional[Mapping[str, Any]]) -> Callable[[str], Mapping[str, Any] | str]:
        raise_on_provider_error = _presale_safety_enabled(context, subflag=PRESALE_VERIFIER_FAILSOFT_ENV)
        return lambda prompt: self._semantic_output_verifier_runner(prompt, raise_on_provider_error=raise_on_provider_error)

    def _semantic_output_verifier_runner(self, prompt: str, *, raise_on_provider_error: bool = False) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_semantic_output_verifier_",
            suffix=".json",
            model=os.getenv(SEMANTIC_OUTPUT_VERIFIER_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(SEMANTIC_OUTPUT_VERIFIER_REASONING_ENV) or "medium",
            timeout_sec=_semantic_output_verifier_timeout_sec(),
            raise_on_error=raise_on_provider_error,
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw

    def _semantic_output_regen_runner(self, prompt: str) -> str:
        return self._run_prompt_text(
            prompt,
            prefix="mango_semantic_output_regen_",
            suffix=".txt",
            model=os.getenv(SEMANTIC_OUTPUT_VERIFIER_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(SEMANTIC_OUTPUT_VERIFIER_REASONING_ENV) or "medium",
            timeout_sec=_semantic_output_verifier_timeout_sec(),
        )

    def _direct_path_llm_retrieve_runner(self, prompt: str) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_direct_path_retriever_",
            suffix=".json",
            model=os.getenv(LLM_RETRIEVE_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(LLM_RETRIEVE_REASONING_ENV) or "low",
            timeout_sec=_llm_retrieve_timeout_sec(),
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw

    def _dialogue_contract_repair_runner(self, prompt: str) -> str:
        return self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_repair_",
            suffix=".txt",
            reasoning_effort=os.getenv("TELEGRAM_DIALOGUE_CONTRACT_REPAIR_REASONING") or self.reasoning_effort,
        )

    def _dialogue_contract_warmth_runner(self, prompt: str) -> str:
        return self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_warmth_",
            suffix=".txt",
            model=os.getenv(HUMANITY_X2_REWRITE_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(HUMANITY_X2_REWRITE_REASONING_ENV) or "xhigh",
        )

    def _apply_dialogue_contract_v2_guard_chain(
        self,
        result: SubscriptionDraftResult,
        *,
        client_message: str,
        context: Optional[Mapping[str, Any]],
    ) -> SubscriptionDraftResult:
        """v2 post-chain: safety verifiers only; no old intent/template rewrites."""
        guard_steps: list[dict[str, Any]] = []

        def record_step(name: str, before: SubscriptionDraftResult, after: SubscriptionDraftResult) -> None:
            before_flags = set(before.safety_flags)
            after_flags = set(after.safety_flags)
            guard_steps.append(
                {
                    "name": name,
                    "route_before": before.route,
                    "route_after": after.route,
                    "text_changed": before.draft_text != after.draft_text,
                    "added_flags": sorted(after_flags - before_flags),
                }
            )

        guarded = result
        guarded = apply_payment_confirmation_guard(guarded, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("payment_confirmation", result, guarded)
        result = guarded

        guarded = apply_brand_separation_guard(result, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("brand_separation", result, guarded)
        result = guarded

        guarded = apply_input_policy_guards(result, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("input_policy", result, guarded)
        result = guarded

        guarded = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("unstated_subject", result, guarded)
        result = guarded

        guarded = apply_unsupported_promise_guard(result, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("unsupported_promise", result, guarded)
        result = guarded

        guarded = apply_unconfirmed_operational_specificity_guard(result, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("unconfirmed_operational_specificity", result, guarded)
        result = guarded

        guarded = apply_dialogue_contract_v2_template_dispatcher(result, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("safe_template_dispatcher", result, guarded)
        result = guarded

        guarded = apply_funnel_policy_guard(result, context=context)
        record_step("funnel_policy", result, guarded)
        result = guarded

        guarded = self._dialogue_contract_v2_route_permission_guard(result, client_message=client_message, context=context)
        record_step("route_permission", result, guarded)
        result = guarded

        guarded = guard_identity_disclosure(result)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("identity_disclosure", result, guarded)

        sanitized = _sanitize_dialogue_contract_client_text(guarded)
        record_step("sanitize", guarded, sanitized)
        trace_event(
            context,
            "_apply_dialogue_contract_v2_guard_chain",
            {
                "applied_guards": [step["name"] for step in guard_steps],
                "steps": guard_steps,
                "route": sanitized.route,
                "safety_flags": sanitized.safety_flags,
            },
        )
        return sanitized

    def _reverify_dialogue_contract_text_change(
        self,
        before: SubscriptionDraftResult,
        after: SubscriptionDraftResult,
        *,
        client_message: str,
        context: Optional[Mapping[str, Any]],
    ) -> SubscriptionDraftResult:
        if before.draft_text == after.draft_text:
            return after
        metadata = dict(after.metadata)
        pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
        facts = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
        fact_texts = {str(k): str(v) for k, v in facts.items()}
        contract = parse_dialogue_contract(
            pipeline.get("contract"),
            active_brand=_active_brand(context),
            fact_key_catalog=tuple(fact_texts.keys()),
        )
        previous_bot_texts = _humanity_previous_bot_texts(context)
        verified_safe_template = _is_verified_safe_numeric_template(after.draft_text)
        if verified_safe_template:
            fact_texts["_verified_safe_numeric_template"] = after.draft_text
        findings = verify_dialogue_contract_output(
            after.draft_text,
            facts=fact_texts,
            active_brand=_active_brand(context),
            contract=contract,
            client_message=client_message,
            context=context,
            previous_bot_texts=previous_bot_texts,
        )
        if (
            _is_policy_c_identity_question(after, context=context)
            and _is_approved_policy_c_identity_text(after.draft_text, active_brand=_active_brand(context))
            and not contract.is_p0
            and not detect_high_risk_input_markers(client_message, context=context)
        ):
            flags = tuple(
                dict.fromkeys(
                    [
                        *after.safety_flags,
                        "dialogue_contract_text_change_reverified",
                        "identity_policy_c_reverified",
                    ]
                )
            )
            return replace(after, safety_flags=flags)
        if _rules_engine_result_applied(metadata) and fact_texts and not findings:
            flags = tuple(
                dict.fromkeys(
                    [
                        *after.safety_flags,
                        "dialogue_contract_text_change_reverified",
                        "rules_engine_text_change_reverified",
                    ]
                )
            )
            return replace(after, safety_flags=flags)
        semantic_available = True
        unsupported_claims: tuple[str, ...] = ()
        shadow_enabled = dialogue_contract_faithfulness_shadow_enabled(context)
        if facts:
            semantic_result = check_dialogue_contract_faithfulness(
                after.draft_text,
                facts={str(k): str(v) for k, v in facts.items()},
                client_words=client_message,
                faithfulness_fn=self._dialogue_contract_faithfulness_runner,
                established_topic=dialogue_contract_established_topic_from_context(context),
            )
            if shadow_enabled:
                record = dialogue_contract_faithfulness_shadow_record("text_change", semantic_result)
                pipeline = dict(pipeline)
                events = pipeline.get("faithfulness_shadow")
                if not isinstance(events, list):
                    events = []
                events.append(record)
                pipeline["faithfulness_shadow"] = events
                metadata["dialogue_contract_pipeline"] = pipeline
                semantic_available = True
                unsupported_claims = ()
            else:
                semantic_available = semantic_result.available
                unsupported_claims = semantic_result.unsupported
        if verified_safe_template:
            findings = [finding for finding in findings if finding.code not in {"fact_grounding", "p0_promise"}]
        if not findings and not unsupported_claims and semantic_available:
            flags = tuple(dict.fromkeys([*after.safety_flags, "dialogue_contract_text_change_reverified"]))
            return replace(after, safety_flags=flags, metadata=metadata)
        flags = tuple(
            dict.fromkeys(
                [
                    *after.safety_flags,
                    "dialogue_contract_text_change_blocked",
                    "manager_approval_required",
                    "no_auto_send",
                ]
            )
        )
        checklist = tuple(
            dict.fromkeys(
                [
                    *after.manager_checklist,
                    "v2 safety-fallback не прошёл повторную проверку: использовать только после ручной правки.",
                ]
            )
        )
        metadata["dialogue_contract_reverification_findings"] = [
            {"code": finding.code, "detail": finding.detail} for finding in findings
        ]
        if unsupported_claims:
            metadata["dialogue_contract_reverification_unsupported"] = list(unsupported_claims)
        metadata["dialogue_contract_reverification_semantic_available"] = semantic_available
        recovery_candidate = _validated_guardchain_recovery_candidate(
            replace(after, metadata=metadata),
            client_message=client_message,
            context=context,
        )
        if recovery_candidate:
            recovered_flags = tuple(
                dict.fromkeys([*after.safety_flags, "cite_only_recover_at_guardchain"])
            )
            recovered_metadata = {
                **metadata,
                "cite_only_recover_at_guardchain": True,
                "cite_only_recover_at_guardchain_source": "text_change_reverify",
            }
            if _step4_keep_answer_enabled(context):
                recovered_metadata = _metadata_with_self_route_deferral_cleared(recovered_metadata)
            return replace(
                after,
                route="bot_answer_self_for_pilot",
                draft_text=recovery_candidate,
                safety_flags=recovered_flags,
                metadata=recovered_metadata,
            )
        yielded_before = _safe_template_yield_before_fallback(
            before,
            after,
            client_message=client_message,
            context=context,
        )
        if yielded_before is not None:
            return yielded_before
        return replace(
            after,
            route="draft_for_manager" if after.route != "manager_only" else after.route,
            draft_text=SAFE_FALLBACK_DRAFT_TEXT,
            safety_flags=flags,
            manager_checklist=checklist,
            metadata=metadata,
        )

    def _dialogue_contract_v2_route_permission_guard(
        self,
        result: SubscriptionDraftResult,
        *,
        client_message: str,
        context: Optional[Mapping[str, Any]],
    ) -> SubscriptionDraftResult:
        if result.route not in (*AUTONOMOUS_ROUTES, "draft_for_manager"):
            return result
        flags = list(result.safety_flags)
        checklist = list(result.manager_checklist)
        metadata = dict(result.metadata)

        decision = decide_route(
            result,
            client_message=client_message,
            context=context,
            allow_default_autonomy=_default_autonomy_flip_enabled(context),
        )
        if decision.veto_category:
            flags.extend(decision.safety_flags)
            checklist.extend(decision.manager_checklist)
            metadata.update(decision.metadata)
            if decision.veto_category == "high_risk" and _is_combined_high_risk_case(
                result,
                markers=set(detect_high_risk_input_markers(client_message, context=context)),
                client_message=client_message,
                context=context,
            ):
                flags.append("combined_high_risk_manager_only")
                metadata["combined_high_risk_manager_only"] = True
            return replace(
                result,
                route=decision.route,
                veto_category=decision.veto_category,
                safety_flags=tuple(dict.fromkeys(flags)),
                manager_checklist=tuple(dict.fromkeys(checklist)),
                metadata=metadata,
            )

        if decision.autonomous_candidate:
            flags.append("dialogue_contract_route_permission_autonomous_candidate")
            recovery_candidate = _validated_guardchain_recovery_candidate(
                replace(result, metadata=metadata, safety_flags=tuple(dict.fromkeys(flags))),
                client_message=client_message,
                context=context,
            )
            if recovery_candidate:
                flags.append("cite_only_recover_at_guardchain")
                metadata["cite_only_recover_at_guardchain"] = True
                metadata["cite_only_recover_at_guardchain_source"] = "route_permission"
                return replace(
                    result,
                    route="bot_answer_self_for_pilot",
                    draft_text=recovery_candidate,
                    veto_category=decision.veto_category,
                    safety_flags=tuple(dict.fromkeys(flags)),
                    manager_checklist=tuple(dict.fromkeys(checklist)),
                    metadata=metadata,
                )
        return replace(
            result,
            route=decision.route,
            veto_category=decision.veto_category,
            safety_flags=tuple(dict.fromkeys(flags)),
            manager_checklist=tuple(dict.fromkeys(checklist)),
            metadata=metadata,
        )

    def _run_prompt_text(
        self,
        prompt: str,
        *,
        prefix: str,
        suffix: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        timeout_sec: int | None = None,
        raise_on_error: bool = False,
    ) -> str:
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix) as out_file:
            output_path = Path(out_file.name)
            with codex_isolation_cwd(self.codex_isolated) as isolated_cwd:
                cmd = self._build_codex_command(
                    output_path=output_path,
                    model=model or self.model,
                    reasoning_effort=reasoning_effort or self.reasoning_effort,
                    isolated_cwd=isolated_cwd,
                )
                proc = self.runner(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=max(1, int(timeout_sec or self.timeout_sec)),
                    env=build_codex_exec_env(self.base_env),
                )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            if raise_on_error:
                detail = " ".join(str(proc.stderr or proc.stdout or raw or "").split())[:500]
                if detail:
                    raise _PromptProviderError(f"provider_error rc={proc.returncode}: {detail}")
                raise _PromptProviderError(f"provider_error rc={proc.returncode}")
            return ""
        return raw or proc.stdout or proc.stderr or ""

    def _answer_quality_llm_rewrite_runner(
        self,
        *,
        result: SubscriptionDraftResult,
        client_message: str,
        context: Mapping[str, Any] | None,
        assessment: AnswerQualityAssessment,
    ) -> Mapping[str, Any]:
        prompt = build_answer_quality_llm_rewrite_prompt(
            result=result,
            client_message=client_message,
            context=context,
            assessment=assessment,
        )
        reasoning = str(os.getenv(ANSWER_QUALITY_LLM_REWRITE_REASONING_ENV) or "xhigh").strip() or "xhigh"
        with tempfile.NamedTemporaryFile(prefix="mango_answer_quality_rewrite_", suffix=".json") as out_file:
            output_path = Path(out_file.name)
            with codex_isolation_cwd(self.codex_isolated) as isolated_cwd:
                cmd = self._build_codex_command(output_path=output_path, reasoning_effort=reasoning, isolated_cwd=isolated_cwd)
                proc = self.runner(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.timeout_sec,
                    env=build_codex_exec_env(self.base_env),
                )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            return {}
        try:
            payload = extract_json_object(raw or proc.stdout or proc.stderr or "")
        except Exception:
            return {}
        draft_text = str(payload.get("draft_text") or "").strip()
        if not draft_text:
            return {}
        return {
            "draft_text": draft_text,
            "reason": str(payload.get("reason") or "")[:300],
        }

    def _humanity_x2_rewrite_runner(self, prompt: str) -> str:
        model = str(os.getenv(HUMANITY_X2_REWRITE_MODEL_ENV) or "gpt-5.5").strip() or "gpt-5.5"
        reasoning = str(os.getenv(HUMANITY_X2_REWRITE_REASONING_ENV) or "xhigh").strip() or "xhigh"
        with tempfile.NamedTemporaryFile(prefix="mango_humanity_x2_rewrite_", suffix=".txt") as out_file:
            output_path = Path(out_file.name)
            with codex_isolation_cwd(self.codex_isolated) as isolated_cwd:
                cmd = self._build_codex_command(output_path=output_path, model=model, reasoning_effort=reasoning, isolated_cwd=isolated_cwd)
                proc = self.runner(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.timeout_sec,
                    env=build_codex_exec_env(self.base_env),
                )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            return ""
        return _extract_humanity_x2_text(raw or proc.stdout or proc.stderr or "")

    def generate(self, prompt: str) -> SubscriptionDraftResult:
        return self.generate_from_prompt(prompt)

    def generate_from_prompt(self, prompt: str, *, force_manager_only: bool = False) -> SubscriptionDraftResult:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return apply_authoritative_output_gate(safe_fallback_draft(reason="empty_prompt"))

        cache_key = _cache_key(
            {
                "schema_version": SUBSCRIPTION_LLM_SCHEMA_VERSION,
                "provider": "codex_exec",
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "codex_isolated": self.codex_isolated,
                "prompt": prompt_text,
                "force_manager_only": force_manager_only,
            }
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return apply_authoritative_output_gate(_with_metadata(cached, {"cache_hit": True}))

        last_error = "codex_exec_failed"
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self._run_once(prompt_text, force_manager_only=force_manager_only)
            except subprocess.TimeoutExpired:
                return apply_authoritative_output_gate(safe_fallback_draft(reason="timeout", metadata={"attempt": attempt, "timeout_sec": self.timeout_sec}))
            except FileNotFoundError:
                return apply_authoritative_output_gate(safe_fallback_draft(reason="codex_binary_not_found", metadata={"codex_bin": self.codex_bin}))
            except _CodexRetryableError as exc:
                last_error = str(exc) or "retryable_codex_error"
                if attempt < self.max_attempts:
                    self.sleep(min(3.0, float(attempt)))
                    continue
                return apply_authoritative_output_gate(safe_fallback_draft(reason="codex_retryable_error", metadata={"last_error": last_error}))
            except Exception as exc:  # noqa: BLE001
                return apply_authoritative_output_gate(safe_fallback_draft(reason="invalid_json_or_codex_error", metadata={"last_error": str(exc)[:400]}))
            self._cache_put(cache_key, result)
            return apply_authoritative_output_gate(result)
        return apply_authoritative_output_gate(safe_fallback_draft(reason=last_error))

    def _run_once(self, prompt: str, *, force_manager_only: bool) -> SubscriptionDraftResult:
        with tempfile.NamedTemporaryFile(prefix="mango_draft_codex_", suffix=".json") as out_file:
            output_path = Path(out_file.name)
            with codex_isolation_cwd(self.codex_isolated) as isolated_cwd:
                cmd = self._build_codex_command(output_path=output_path, isolated_cwd=isolated_cwd)
                proc = self.runner(
                    cmd,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.timeout_sec,
                    env=build_codex_exec_env(self.base_env),
                )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            message = f"codex exec failed rc={proc.returncode}: {' '.join(stderr.splitlines()[-2:])[:400]}"
            if _is_retryable(stderr):
                raise _CodexRetryableError(message)
            raise RuntimeError(message)

        payload = extract_json_object(raw or proc.stdout or proc.stderr or "")
        result = normalize_subscription_draft_payload(payload, raw_response=raw)
        result = replace(result, metadata=_with_codex_exec_metadata(result.metadata, isolated=self.codex_isolated))
        if force_manager_only and result.route != "manager_only":
            result = replace(
                result,
                route="manager_only",
                safety_flags=tuple(dict.fromkeys([*result.safety_flags, "forced_manager_only_by_rop_policy"])),
                metadata={**dict(result.metadata), "forced_route": "manager_only"},
            )
        return apply_authoritative_output_gate(guard_identity_disclosure(result))

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            raise RuntimeError("empty direct path prompt")
        with tempfile.NamedTemporaryFile(prefix="mango_direct_path_codex_", suffix=".json") as out_file:
            output_path = Path(out_file.name)
            with codex_isolation_cwd(self.codex_isolated) as isolated_cwd:
                cmd = self._build_codex_command(output_path=output_path, isolated_cwd=isolated_cwd)
                proc = self.runner(
                    cmd,
                    input=prompt_text,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.timeout_sec,
                    env=build_codex_exec_env(self.base_env),
                )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            message = f"codex exec failed rc={proc.returncode}: {' '.join(stderr.splitlines()[-2:])[:400]}"
            if _is_retryable(stderr):
                raise _CodexRetryableError(message)
            raise RuntimeError(message)

        payload = extract_json_object(raw or proc.stdout or proc.stderr or "")
        result = _normalize_direct_path_payload(payload, raw_response=raw)
        return replace(result, metadata=_with_codex_exec_metadata(result.metadata, isolated=self.codex_isolated))

    def _cache_get(self, cache_key: str) -> Optional[SubscriptionDraftResult]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return normalize_subscription_draft_payload(payload)
        except Exception:
            return None

    def _cache_put(self, cache_key: str, result: SubscriptionDraftResult) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{cache_key}.json"
        path.write_text(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class FakeSubscriptionLlmDraftProvider:
    def __init__(self, result: Optional[SubscriptionDraftResult | Mapping[str, Any]] = None) -> None:
        self.result = normalize_subscription_draft_payload(result) if result is not None else safe_fallback_draft(
            reason="fake_provider_default"
        )
        self.prompts: list[str] = []

    def build_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        prompt = build_draft_prompt(client_message, context=context)
        result = self.generate_from_prompt(prompt, force_manager_only=should_force_manager_only(context))
        result = apply_payment_confirmation_guard(result, client_message=client_message, context=context)
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_conversation_intent_plan_guard(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        result = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_unconfirmed_operational_specificity_guard(result, context=context)
        result = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        result = apply_funnel_policy_guard(result, context=context)
        result = apply_answer_quality_rewriter(result, client_message=client_message, context=context)
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_conversation_intent_plan_guard(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        result = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_unconfirmed_operational_specificity_guard(result, context=context)
        result = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        result = apply_funnel_policy_guard(result, context=context)
        result = apply_autonomy_matrix_guard(result, client_message=client_message, context=context)
        result = apply_humanity_guards(result, client_message=client_message, context=context)
        result = apply_humanity_x2_rewriter(result, client_message=client_message, context=context)
        result = apply_phase2_tone_layer(result, client_message=client_message, context=context)
        result = apply_a2_proactive_layer(result, client_message=client_message, context=context)
        result = apply_tone_sell_prompt_observer(result, client_message=client_message, context=context)
        result = apply_semantic_diagnosis_guard(result, client_message=client_message, context=context)
        return apply_authoritative_output_gate(result, client_message=client_message, context=context)

    def generate(self, prompt: str) -> SubscriptionDraftResult:
        return self.generate_from_prompt(prompt)

    def generate_from_prompt(self, prompt: str, *, force_manager_only: bool = False) -> SubscriptionDraftResult:
        self.prompts.append(prompt)
        result = self.result
        if force_manager_only:
            result = replace(
                result,
                route="manager_only",
                safety_flags=tuple(dict.fromkeys([*result.safety_flags, "forced_manager_only_by_rop_policy"])),
            )
        return guard_identity_disclosure(result)


def normalize_subscription_draft_payload(payload: Mapping[str, Any] | SubscriptionDraftResult, *, raw_response: Optional[str] = None) -> SubscriptionDraftResult:
    if isinstance(payload, SubscriptionDraftResult):
        return payload
    if not isinstance(payload, Mapping):
        raise RuntimeError("subscription draft response JSON root must be an object")
    schedule = payload.get("safe_schedule_template")
    manager_followup_required = bool(payload.get("manager_followup_required"))
    manager_followup_deadline = _optional_text(payload.get("manager_followup_deadline"))
    if isinstance(schedule, Mapping) and schedule.get("manager_followup_required") is True:
        manager_followup_required = True
        manager_followup_deadline = manager_followup_deadline or _optional_text(
            schedule.get("manager_followup_deadline") or schedule.get("deadline_at")
        )
    result = SubscriptionDraftResult(
        message_type=str(payload.get("message_type") or "question"),
        broad_group=str(payload.get("broad_group") or ""),
        topic_id=str(payload.get("topic_id") or "service:S2_unclear"),
        topic_confidence=_clamp_float(payload.get("confidence_theme", payload.get("topic_confidence"))),
        confidence_group=_clamp_float(payload.get("confidence_group")),
        alternative_themes=tuple(_clean_list(payload.get("alternative_themes"), max_items=5, max_chars=120)),
        risk_level=str(payload.get("risk_level") or "unknown"),
        route=str(payload.get("route") or "manager_only"),
        draft_text=str(payload.get("draft_text") or SAFE_FALLBACK_DRAFT_TEXT),
        manager_checklist=tuple(_clean_list(payload.get("manager_checklist"), max_items=12, max_chars=240)),
        missing_facts=tuple(_clean_list(payload.get("missing_facts"), max_items=12, max_chars=160)),
        forbidden_promises_detected=tuple(_clean_list(payload.get("forbidden_promises_detected"), max_items=12, max_chars=160)),
        crm_recommendations=tuple(_clean_crm_recommendations(payload.get("crm_recommendations"))),
        safety_flags=tuple(_clean_list(payload.get("safety_flags"), max_items=16, max_chars=80)),
        context_used=tuple(_clean_list(payload.get("context_used"), max_items=12, max_chars=100)),
        context_warnings=tuple(_clean_list(payload.get("context_warnings"), max_items=12, max_chars=120)),
        manager_followup_required=manager_followup_required,
        manager_followup_deadline=manager_followup_deadline,
        raw_response=raw_response,
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {},
    )
    return guard_promocode_leak(
        guard_draft_placeholder(guard_identity_disclosure(apply_taxonomy_topic_guard(apply_subscription_policy_guards(result))))
    )


def safe_fallback_draft(*, reason: str, metadata: Optional[Mapping[str, Any]] = None) -> SubscriptionDraftResult:
    extra_flags = ("codex_exec_timeout",) if reason == "timeout" else ()
    return SubscriptionDraftResult(
        message_type="manager_only",
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        manager_checklist=("Проверить вопрос вручную.",),
        missing_facts=("llm_response",),
        safety_flags=(*BASE_SAFETY_FLAGS, "llm_fallback", "draft_only", *extra_flags),
        error=reason,
        metadata=dict(metadata or {}),
    )


_DIRECT_PATH_MODEL_P0_KINDS = frozenset({"payment_dispute", "refund", "complaint", "legal_threat"})


def _direct_path_payload_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "да", "p0", "high"}


def _direct_path_model_p0_kind(value: Any) -> str:
    kind = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if kind == "legal":
        kind = "legal_threat"
    if kind in {"payment", "payment_issue", "payment_problem", "payment_claim"}:
        kind = "payment_dispute"
    return kind if kind in _DIRECT_PATH_MODEL_P0_KINDS else ""


def _direct_path_answerability_value(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"yes", "да", "true", "1", "can_answer", "answer_self"}:
        return "yes"
    if text in {"no", "нет", "false", "0", "manager", "manager_only", "handoff"}:
        return "no"
    if text in {"uncertain", "unknown", "не_уверен", "не уверен", "непонятно"}:
        return "uncertain"
    return text[:40] if text else ""


def _direct_path_answerability_self_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    can_answer_self = _direct_path_answerability_value(payload.get("can_answer_self"))
    missing_facts = _clean_list(payload.get("self_missing_facts"), max_items=12, max_chars=120)
    supporting_facts = _clean_list(payload.get("supporting_facts"), max_items=12, max_chars=160)
    why_manager = " ".join(str(payload.get("why_manager") or "").split())[:300]
    if not any((can_answer_self, missing_facts, supporting_facts, why_manager)):
        return {}
    return {
        "schema_version": "answerability_self_v1_2026_06_15",
        "can_answer_self": can_answer_self,
        "self_missing_facts": list(missing_facts),
        "supporting_facts": list(supporting_facts),
        "why_manager": why_manager,
    }


def _direct_path_model_p0_meta(result: SubscriptionDraftResult) -> Mapping[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    meta = metadata.get("direct_path_model_p0")
    return meta if isinstance(meta, Mapping) else {}


def _direct_path_model_p0_signal(result: SubscriptionDraftResult, *, client_message: str, context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not _direct_path_model_p0_enabled(context):
        return {}
    meta = _direct_path_model_p0_meta(result)
    kind = _direct_path_model_p0_kind(meta.get("p0_kind"))
    risk_level = str(meta.get("risk_level") or result.risk_level or "").strip().casefold()
    model_is_p0 = bool(meta.get("is_p0")) or (risk_level in {"high", "p0", "critical", "high_risk"} and bool(kind))
    floor_reason = str(dialogue_contract_p0_pre_gate(client_message, context=context) or "")
    if floor_reason and _p0_model_led_enabled(context):
        _, floor_kind = _direct_path_p0_text(floor_reason, context)
        if floor_kind == "complaint" and not _p0_model_led_complaint_backstop(client_message):
            floor_reason = ""
    if not model_is_p0 and not floor_reason:
        return {}
    if not kind:
        kind = "complaint"
    return {
        "is_p0": True,
        "p0_kind": kind,
        "risk_level": "high",
        "model_reason": str(meta.get("model_reason") or "").strip()[:240],
        "floor_reason": floor_reason,
        "source": "model_p0" if model_is_p0 else "p0_pre_gate",
    }


def _apply_direct_path_model_p0_route(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    signal = _direct_path_model_p0_signal(result, client_message=client_message, context=context)
    if not signal:
        return result
    kind = str(signal.get("p0_kind") or "complaint")
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    direct["model_p0"] = {
        "is_p0": True,
        "p0_kind": kind,
        "risk_level": "high",
        "model_reason": str(signal.get("model_reason") or ""),
        "floor_reason": str(signal.get("floor_reason") or ""),
        "source": str(signal.get("source") or "model_p0"),
    }
    metadata["direct_path"] = direct
    metadata["direct_path_model_p0"] = dict(direct["model_p0"])
    metadata["reason_class"] = "p0_deferral"
    metadata["is_manager_deferral"] = True
    flags = tuple(
        dict.fromkeys(
            [
                *result.safety_flags,
                f"direct_path_model_p0_{kind}",
                kind,
                "manager_approval_required",
                "no_auto_send",
            ]
        )
    )
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "P0/high-risk: модель прямого пути классифицировала срочное обращение; отвечает менеджер.",
            ]
        )
    )
    return replace(
        result,
        message_type="manager_only",
        route="manager_only",
        risk_level="high",
        safety_flags=flags,
        manager_checklist=checklist,
        metadata=metadata,
    )


def _normalize_direct_path_payload(
    payload: Mapping[str, Any],
    *,
    raw_response: Optional[str] = None,
    include_answerability_self: bool = False,
) -> SubscriptionDraftResult:
    if not isinstance(payload, Mapping):
        raise RuntimeError("direct path response JSON root must be an object")
    route = str(payload.get("route") or "").strip()
    if not route:
        route = "draft_for_manager" if _direct_default_manager_enabled() else "bot_answer_self_for_pilot"
    if route == "bot_answer_self":
        route = "bot_answer_self_for_pilot"
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {}
    risk_level = str(payload.get("risk_level") or "low").strip()
    if any(key in payload for key in ("is_p0", "risk_level", "p0_kind", "p0_code", "model_reason")):
        metadata["direct_path_model_p0"] = {
            "is_p0": _direct_path_payload_bool(payload.get("is_p0")),
            "risk_level": risk_level,
            "p0_kind": _direct_path_model_p0_kind(payload.get("p0_kind") or payload.get("p0_code") or payload.get("risk_code")),
            "model_reason": " ".join(str(payload.get("model_reason") or payload.get("p0_reason") or "").split())[:240],
        }
    proposal = payload.get("action_proposal")
    if isinstance(proposal, Mapping):
        metadata["action_proposal"] = dict(proposal)
    elif isinstance(proposal, str) and proposal.strip():
        metadata["action_proposal"] = {"action": proposal.strip(), "source": "direct_model"}
    if include_answerability_self:
        metadata["answerability_self"] = _direct_path_answerability_self_from_payload(payload)
    return SubscriptionDraftResult(
        message_type=str(payload.get("message_type") or "question"),
        broad_group=str(payload.get("broad_group") or "direct_path"),
        topic_id=str(payload.get("topic_id") or UNKNOWN_TOPIC_FALLBACK_ID),
        topic_confidence=_clamp_float(payload.get("confidence_theme", payload.get("topic_confidence", 0.8))),
        confidence_group=_clamp_float(payload.get("confidence_group", 0.8)),
        alternative_themes=tuple(_clean_list(payload.get("alternative_themes"), max_items=5, max_chars=120)),
        risk_level=risk_level,
        route=route,
        draft_text=str(payload.get("draft_text") or SAFE_FALLBACK_DRAFT_TEXT),
        manager_checklist=tuple(_clean_list(payload.get("manager_checklist"), max_items=12, max_chars=240)),
        missing_facts=tuple(_clean_list(payload.get("missing_facts"), max_items=12, max_chars=160)),
        forbidden_promises_detected=tuple(_clean_list(payload.get("forbidden_promises_detected"), max_items=12, max_chars=160)),
        crm_recommendations=tuple(_clean_crm_recommendations(payload.get("crm_recommendations"))),
        safety_flags=tuple(_clean_list(payload.get("safety_flags"), max_items=16, max_chars=80)),
        context_used=tuple(_clean_list(payload.get("context_used"), max_items=12, max_chars=100)),
        context_warnings=tuple(_clean_list(payload.get("context_warnings"), max_items=12, max_chars=120)),
        manager_followup_required=bool(payload.get("manager_followup_required")),
        manager_followup_deadline=_optional_text(payload.get("manager_followup_deadline")),
        raw_response=raw_response,
        metadata=metadata,
    )


def parse_llm_json(text: str) -> SubscriptionDraftResult:
    try:
        return normalize_subscription_draft_payload(extract_json_object(text), raw_response=text)
    except Exception as exc:  # noqa: BLE001
        return safe_fallback_draft(reason="invalid_json", metadata={"parse_error": str(exc)[:300]})


DraftGenerationResult = SubscriptionDraftResult


CodexExecDraftProvider = SubscriptionLlmDraftProvider


FakeDraftProvider = FakeSubscriptionLlmDraftProvider


contains_bot_identity_disclosure = draft_has_identity_disclosure


def subscription_llm_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": SUBSCRIPTION_LLM_SCHEMA_VERSION,
        "provider": "codex_exec",
        "uses_openai_api_key": False,
        "client_auto_send_allowed": False,
        "crm_write_allowed": False,
        "tallanto_write_allowed": False,
        "stable_runtime_write_allowed": False,
        "fallback_text": SAFE_FALLBACK_DRAFT_TEXT,
        "identity_disclosure_forbidden_phrases": list(IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES),
        "safe_schedule_template": safe_schedule_template(),
    }


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _with_metadata(result: SubscriptionDraftResult, extra: Mapping[str, Any]) -> SubscriptionDraftResult:
    return replace(result, metadata={**dict(result.metadata), **dict(extra)})
