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
    apply_authoritative_output_gate,
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




from mango_mvp.channels.subscription_llm_parts.provider import (
    CodexExecDraftProvider,
    DraftGenerationResult,
    FakeDraftProvider,
    FakeSubscriptionLlmDraftProvider,
    SubscriptionLlmDraftProvider,
    _Runner,
    _normalize_direct_path_payload,
    _optional_text,
    _with_metadata,
    contains_bot_identity_disclosure,
    normalize_subscription_draft_payload,
    parse_llm_json,
    safe_fallback_draft,
    subscription_llm_safety_contract,
)







































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































