from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.tone_block import (
    TONE_RICH_FORMAT_ENV,
    TONE_SELL_PROMPT_ENV,
)
from mango_mvp.channels.draft_prompt_builder import (
    IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES,
    build_draft_prompt,
    safe_schedule_template,
    should_force_manager_only,
)
from mango_mvp.knowledge_base.product_existence_axes_catalog import (
    build_product_existence_axes_catalog,
    verify_product_format_exists,
)
from mango_mvp.knowledge_base.fact_registry import fact_runtime_time_ok


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
    _direct_path_model_p0_enabled,
    _direct_default_manager_enabled,
    _intent_model_led_enabled,
    _p0_model_led_enabled,
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
    PAYMENT_SUBJECT_GUARDS_ENV,
    _template_from_kb_enabled,
    _template_from_kb_trace_event,
    _seats_default_open_allowlisted_result,
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
    semantic_frame_bool as _semantic_frame_bool,
)

from mango_mvp.channels.subscription_llm_parts.reliable_answerer import apply_reliable_answerer_output_guard
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    finalize_reading_trace_metadata,
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
    build_direct_path_slot_topic_shadow_metadata,
    _direct_path_retriever_ids,
    _direct_path_llm_retrieve_fact_pack,
    _retriever_model_driven_enabled,
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
    _direct_path_select_gold_real_examples,
    _direct_path_gold_prompt_block,
    _build_direct_path_prompt,
    _direct_path_metadata,
    _direct_path_merge_metadata,
    apply_direct_path_scope_overclaim_guard,
    _direct_slot_topic_shadow_enabled,
    _semantic_frame_shadow_enabled,
    _semantic_frame_posthoc_shadow_enabled,
    _semantic_frame_decision_shadow_enabled,
    _semantic_frame_manager_action_gate_enabled,
    _semantic_frame_self_answer_shadow_enabled,
    _semantic_frame_existence_proof_shadow_enabled,
    _semantic_frame_proof_reconciliation_shadow_enabled,
    _p0_model_classes_v2_enabled,
    _a2_extract_phone,
    _replace_echoed_phone,
)

from mango_mvp.channels.subscription_llm_parts.text_hygiene import scrub_direct_path_p0_text


from mango_mvp.channels.subscription_llm_parts.policy_routing import (
    ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
    ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
    ADDRESS_UNPK_SAFE_TEXT,
    ADMISSION_GUARANTEE_INPUT_RE,
    ADMISSION_GUARANTEE_SAFE_TEXT,
    ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION_ENV,
    AUTONOMOUS_ROUTES,
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
    FOTON_DOLYAMI_SAFE_TEXT,
    FOTON_INSTALLMENT_SAFE_TEXT,
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
    PRICE_AMOUNT_RE,
    PROGRAM_HANDOFF_SAFE_TEXT,
    PROMOCODE_DRAFT_RE,
    PROMOCODE_SAFE_TEXT,
    QUITTANCE_SAFE_TEXT,
    REFUND_ZERO_COLLECT_SAFE_TEXT,
    RESULT_GUARANTEE_INPUT_RE,
    RESULT_GUARANTEE_SAFE_TEXT,
    SCOPE_FACT_GUARD_ENV,
    SOFT_NEGATIVE_HANDOFF_SAFE_TEXT,
    SUBJECT_GUARD_MARKERS,
    TAX_AMOUNT_SAFE_TEXT,
    TAX_FNS_REVIEW_SAFE_TEXT,
    TAX_LICENSE_SAFE_TEXT,
    TAX_ONLINE_FORM_SAFE_TEXT,
    THIRD_PARTY_PRIVACY_SAFE_TEXT,
    UNKNOWN_TOPIC_FALLBACK_ID,
    UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT,
    UNPK_MONTHLY_SEMESTER_DISCOUNT_TEXT,
    UNPK_SECOND_SUBJECT_DISCOUNT_TEXT,
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
    _context_has_missing_fact_signal,
    _context_with_dialogue_contract_retrieved_facts,
    _conversation_intent_plan,
    _dedupe_sentence,
    _dialog_context_haystack,
    _draft_confirms_payment,
    _extract_numeric_promise_claims,
    _fact_key_root,
    _has_missing_fact_signal,
    _humanity_previous_bot_texts,
    _is_combined_high_risk_case,
    _is_verified_safe_numeric_template,
    _known_fields_from_text,
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
    detect_high_risk_input_markers,
    find_unsupported_numeric_promises,
    is_high_risk_result,
    known_context_fields,
)

from mango_mvp.channels.subscription_llm_parts.post_layers import (
    AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION,
    A_PROACTIVE_ENV,
    A_RICH_FORMAT_ENV,
    CONTENT_DELIVERY_ACTION_RE,
    COSMETIC_OPENING_RE,
    DERIVED_PRODUCT_NUMBER_RE,
    DIRECT_PATH_REPLACE_TEXT_GATE_CODES,
    DRAFT_PLACEHOLDER_RE,
    FOLLOWUP_DEADLINE_RE,
    GATE_BLOCKING_CODES,
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
    PRESALE_PII_CHILD_NAME_KEY_RE,
    PRESALE_PII_NAME_KEY_RE,
    PRESALE_PII_PHONE_KEY_RE,
    PRESALE_RU_META_LINE_RE,
    PRESALE_SOURCE_ID_PHRASE_RE,
    PRESALE_SOURCE_ID_TOKEN_PATTERN,
    PRESALE_SOURCE_ID_TOKEN_RE,
    PRICE_FIX_PROCESS_SAFE_TEXT,
    SCHEDULE_ASSUMPTION_RE,
    SEMANTIC_OUTPUT_VERIFIER_MODEL_ENV,
    SEMANTIC_OUTPUT_VERIFIER_REASONING_ENV,
    SEMANTIC_OUTPUT_VERIFIER_SCHEMA_VERSION,
    SEMANTIC_OUTPUT_VERIFIER_TIMEOUT_ENV,
    SEMANTIC_VERIFIER_DOWNGRADE_REASON,
    SEMANTIC_VERIFIER_UNAVAILABLE_REASON,
    UNSUPPORTED_FOLLOWUP_DEADLINE_SAFE_TEXT,
    UNSUPPORTED_OFFLINE_VISIT_INVITATION_SAFE_TEXT,
    UNSUPPORTED_SCHEDULE_ASSUMPTION_SAFE_TEXT,
    _A2_EMOJI_RE,
    _A2_FAKE_DONE_RE,
    _A2_SERIOUS_TAGS,
    _CLIENT_NAME_MARKER_RE,
    _CLIENT_NAME_PAIR_RE,
    _CLIENT_NAME_STOPWORDS,
    _CLIENT_PII_CONFIRMATION_RE,
    _CLIENT_RELATION_NAME_STOPWORDS,
    _CLIENT_SELF_NAME_MARKER_RE,
    _DRAFT_PERSON_NAME_CONTEXT_RE,
    _HUMANE_DETAIL_HANDOFF_TEXTS,
    _HUMANE_GENERIC_HANDOFF_TEXTS,
    _SEMANTIC_OUTPUT_VERIFIER_CODES,
    _a2_context_tag,
    _a2_enforce_emoji_limit,
    _a2_is_proactive_result,
    _a2_phone_echoed,
    _a2_proactive_enabled,
    _a2_rich_format_enabled,
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
    _flexible_name_pattern,
    _format_choice_is_disjunctive_question,
    _has_diagnosis_hedge_and_transfer,
    _humanity_p0_required,
    _identity_phrase_present,
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
    _raw_detail_handoff_looks_like_question,
    _run_semantic_output_verifier_once,
    _sanitize_client_pii_echo,
    _sanitize_dialogue_contract_client_text,
    _sanitize_output_client_text,
    _sanitize_presale_ru_meta_lines,
    _sanitize_presale_source_id_text,
    _sanitize_raw_detail_handoff_match,
    _sanitize_raw_detail_handoff_text,
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
    _topic_id_from_context,
    _trim_repeated_cosmetic_opening,
    _unexpected_client_name_echoes,
    _unsupported_claims_by_pattern,
    _verifier_handoff_claims_enabled,
    apply_authoritative_output_gate,
    apply_bot_safe_memory_step_guard,
    apply_no_memory_step_frame_guard,
    apply_night_hours_note,
    apply_output_sanitizer,
    apply_semantic_output_verifier,
    apply_unconfirmed_contact_data_claim_guard,
    apply_unconfirmed_operational_specificity_guard,
    apply_unsupported_promise_guard,
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


def _model_owned_direct_path_context(
    context: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if not (
        isinstance(context, Mapping)
        and _intent_model_led_enabled(context)
        and _llm_retrieve_enabled(context)
        and _retriever_model_driven_enabled(context)
    ):
        return context

    cleaned = dict(context)
    for key in (
        "answer_contract",
        "confirmed_facts",
        "conversation_intent_plan",
        "conversation_intent_plan_internal",
        "dialogue_contract_pipeline",
        "facts_context",
        "gold_answer_context",
        "gold_answers_v3",
        "knowledge_snippets",
        "missing_facts",
        "planner_intent",
        "required_fact_keys",
    ):
        cleaned.pop(key, None)

    memory = cleaned.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        clean_memory = dict(memory)
        for key in (
            "current_message_roles",
            "handoff_state",
            "held_state",
            "message_type",
            "open_question",
            "primary_intent",
            "risk_flags",
            "sales_stage",
            "topic",
            "topic_focus",
            "topic_id",
        ):
            clean_memory.pop(key, None)
        cleaned["dialogue_memory_view"] = clean_memory

    rop_policy = cleaned.get("rop_policy")
    if isinstance(rop_policy, Mapping):
        clean_policy = dict(rop_policy)
        for key in ("active_topics", "autonomy_policy", "fact_scope", "required_fact_keys", "topic_id"):
            clean_policy.pop(key, None)
        if clean_policy:
            cleaned["rop_policy"] = clean_policy
        else:
            cleaned.pop("rop_policy", None)
    return cleaned


def _activate_inline_semantic_frame(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    frame = metadata.get("semantic_frame")
    if not (isinstance(frame, Mapping) and frame.get("source") == "inline"):
        return result
    active = {**frame, "mode": "active"}
    metadata["semantic_frame"] = active
    metadata["semantic_frame_shadow"] = active
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping):
        metadata["direct_path"] = {
            **direct,
            "semantic_frame": active,
            "semantic_frame_shadow": active,
        }
    return replace(result, metadata=metadata)


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
        model_context = _model_owned_direct_path_context(context)
        model_owned_semantics = model_context is not context
        context = model_context
        direct_result = self._build_direct_path_draft(client_message, context=context)
        if model_owned_semantics:
            direct_result = _activate_inline_semantic_frame(direct_result)
        scrubbed = scrub_direct_path_p0_text(
            direct_result,
            context=context,
            client_message=client_message,
        )
        guarded = apply_bot_safe_memory_step_guard(scrubbed, context=context)
        guarded = apply_unconfirmed_contact_data_claim_guard(guarded, client_message=client_message, context=context)
        framed = self._apply_direct_path_semantic_frame_posthoc_shadow(
            apply_no_memory_step_frame_guard(guarded, context=context),
            client_message=client_message,
            context=context,
        )
        proof_shadowed = apply_semantic_frame_existence_proof_shadow(framed, context=context)
        reconciled_shadowed = apply_semantic_frame_proof_reconciliation_shadow(proof_shadowed, context=context)
        manager_gated = apply_semantic_frame_manager_action_gate(reconciled_shadowed, context=context)
        self_answer_shadowed = apply_semantic_frame_self_answer_shadow(manager_gated, context=context)
        decision_shadowed = apply_semantic_frame_decision_shadow(self_answer_shadowed, context=context)
        before_final_gate_route = decision_shadowed.route
        protected = decision_shadowed
        if _pilot_profile_default_on_flag_enabled(context, PAYMENT_SUBJECT_GUARDS_ENV):
            protected = apply_payment_confirmation_guard(protected, client_message=client_message, context=context)
            protected = apply_unstated_subject_guard(protected, client_message=client_message, context=context)
        final_gated = apply_authoritative_output_gate(protected, client_message=client_message, context=context)
        finalized = _direct_path_finalize_metadata(
            final_gated,
            before_gate_route=before_final_gate_route,
            client_message=client_message,
            context=context,
        )
        return apply_semantic_reading_trace_finalize(finalized, context=context)

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
        shadow_meta = build_direct_path_slot_topic_shadow_metadata(
            client_message,
            context=context,
            shadow_fn=self._direct_path_slot_topic_shadow_runner if _direct_slot_topic_shadow_enabled(context) else None,
        )
        if shadow_meta is not None:
            fact_pack = {**dict(fact_pack), "slot_topic_shadow": dict(shadow_meta)}
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
            client_message=client_message,
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
            result = safe_fallback_draft(
                reason="codex_binary_not_found",
                metadata={"direct_path": direct_meta, "codex_bin": self.codex_bin},
            )
        except Exception as exc:  # noqa: BLE001
            direct_meta.update(
                {
                    "text_composition_source": "provider_runtime_fallback",
                    "reason_class": "provider_runtime",
                    "reason_evidence": {"provider_error": str(exc)[:300]},
                    "is_manager_deferral": True,
                }
            )
            result = safe_fallback_draft(
                reason="direct_path_error",
                metadata={"direct_path": direct_meta, "last_error": str(exc)[:400]},
            )
        else:
            result = _direct_path_prepare_model_result(result)
            result = _direct_path_merge_metadata(result, direct_meta)
            result = _apply_direct_path_p0_shadow(
                result,
                client_message=client_message,
                context=context,
            )
            result = _apply_direct_path_model_p0_route(
                result,
                client_message=client_message,
                context=context,
            )
            result = apply_reliable_answerer_output_guard(
                result,
                client_message=client_message,
                context=context,
            )
            result = apply_direct_path_scope_overclaim_guard(
                result,
                context=context,
                fact_pack=fact_pack,
            )

        semantic_checked = apply_semantic_output_verifier(
            result,
            client_message=client_message,
            context=context,
            verifier_fn=self._semantic_output_verifier_runner_for_context(context),
            regen_fn=self._semantic_output_regen_runner,
        )
        return semantic_checked







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

    def _direct_path_slot_topic_shadow_runner(self, prompt: str) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_direct_slot_topic_shadow_",
            suffix=".json",
            model=os.getenv(LLM_RETRIEVE_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(LLM_RETRIEVE_REASONING_ENV) or "low",
            timeout_sec=_llm_retrieve_timeout_sec(),
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw






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



    def _direct_path_semantic_frame_shadow_runner(self, prompt: str) -> str:
        model = str(os.getenv("TELEGRAM_SEMANTIC_FRAME_POSTHOC_MODEL") or self.model).strip() or self.model
        reasoning = str(os.getenv("TELEGRAM_SEMANTIC_FRAME_POSTHOC_REASONING") or "medium").strip() or "medium"
        return self._run_prompt_text(
            prompt,
            prefix="mango_semantic_frame_posthoc_",
            suffix=".json",
            model=model,
            reasoning_effort=reasoning,
            timeout_sec=min(self.timeout_sec, 45),
        )

    def _apply_direct_path_semantic_frame_posthoc_shadow(
        self,
        result: SubscriptionDraftResult,
        *,
        client_message: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        if not _semantic_frame_posthoc_shadow_enabled(context):
            return result
        metadata = dict(result.metadata)
        if isinstance(metadata.get("semantic_frame"), Mapping) or isinstance(metadata.get("semantic_frame_shadow"), Mapping):
            return result

        prompt = build_direct_path_semantic_frame_posthoc_prompt(result, client_message=client_message, context=context)
        status = {"attempted": True, "status": "attempted", "mode": "posthoc"}
        try:
            raw = self._direct_path_semantic_frame_shadow_runner(prompt)
            payload = extract_json_object(raw)
            if "semantic_frame" not in payload and "semanticFrame" not in payload and "semantic_frame_shadow" not in payload:
                payload = {"semantic_frame": payload}
            frame = _direct_path_semantic_frame_from_payload(payload, source="posthoc")
        except Exception as exc:  # noqa: BLE001 - shadow telemetry must be fail-soft.
            frame = {}
            status.update({"status": "provider_error", "error": str(exc)[:240]})
        if not frame:
            metadata["semantic_frame_posthoc_shadow"] = status if status.get("status") == "provider_error" else {**status, "status": "empty_frame"}
            return replace(result, metadata=metadata)

        status["status"] = "ok"
        metadata["semantic_frame"] = frame
        metadata["semantic_frame_shadow"] = frame
        metadata["semantic_frame_posthoc_shadow"] = status
        direct = dict(metadata.get("direct_path") or {})
        direct["semantic_frame"] = dict(frame)
        direct["semantic_frame_shadow"] = dict(frame)
        direct["semantic_frame_posthoc_shadow"] = dict(status)
        metadata["direct_path"] = direct
        return replace(result, metadata=metadata)

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
        result = _normalize_direct_path_payload(
            payload,
            raw_response=raw,
            include_semantic_frame_shadow='"semantic_frame"' in prompt_text,
            include_dialog_summary='"dialog_summary"' in prompt_text and "ПРЕДЫДУЩАЯ СВОДКА" in prompt_text,
        )
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
        return self.generate_from_prompt(prompt, force_manager_only=should_force_manager_only(context))

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


_DIRECT_PATH_MODEL_P0_BASE_KINDS = frozenset({"payment_dispute", "refund", "complaint", "legal_threat"})

_DIRECT_PATH_MODEL_P0_V2_KINDS = frozenset(
    {"child_safety", "cancellation_service_request", "contract_dispute", "paid_operation_context"}
)

_DIRECT_PATH_MODEL_P0_KINDS = _DIRECT_PATH_MODEL_P0_BASE_KINDS | _DIRECT_PATH_MODEL_P0_V2_KINDS

_DIRECT_PATH_MODEL_P0_LEGACY_KIND = {
    "child_safety": "complaint",
    "cancellation_service_request": "refund",
    "paid_operation_context": "refund",
    "contract_dispute": "legal_threat",
}


_DIRECT_PATH_MODEL_INTENTS = frozenset({"live_availability", "schedule", "address", "camp", "price_fix", "off_topic", "other"})


def _direct_path_model_intent_value(value: Any) -> str:
    intent = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if intent in {"availability", "live_status", "seats", "seat_availability", "booking"}:
        intent = "live_availability"
    if intent in {"location", "venue", "place"}:
        intent = "address"
    if intent in {"price_lock", "current_terms", "fix_price"}:
        intent = "price_fix"
    if intent in {"out_of_scope", "offtopic", "not_related", "irrelevant"}:
        intent = "off_topic"
    if intent in {"general", "none", "unknown", "not_target"}:
        intent = "other"
    return intent if intent in _DIRECT_PATH_MODEL_INTENTS else ""


def _direct_path_model_intent_meta_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("model_intent")
    if not isinstance(raw, Mapping):
        raw = payload.get("intent_model")
    if not isinstance(raw, Mapping):
        raw = payload.get("direct_path_model_intent")
    if not isinstance(raw, Mapping):
        raw = {}
    primary_intent = _direct_path_model_intent_value(
        raw.get("primary_intent")
        or raw.get("intent")
        or payload.get("model_primary_intent")
        or payload.get("primary_intent")
    )
    if not primary_intent:
        return {}
    return {
        "schema_version": "direct_path_model_intent_v1_2026_06_25",
        "primary_intent": primary_intent,
        "scope": " ".join(str(raw.get("scope") or payload.get("model_intent_scope") or "").split())[:120],
        "sense": " ".join(str(raw.get("sense") or payload.get("model_intent_sense") or "").split())[:120],
        "confidence": _clamp_float(raw.get("confidence", payload.get("model_intent_confidence", 0.0))),
        "reason": " ".join(str(raw.get("reason") or payload.get("model_intent_reason") or "").split())[:240],
    }


def _direct_path_payload_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "да", "p0", "high"}


def _direct_path_model_p0_kind(value: Any, *, include_v2: bool = False) -> str:
    kind = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if kind == "legal":
        kind = "legal_threat"
    if kind in {"contract", "contract_issue", "document_dispute", "contract_claim"}:
        kind = "contract_dispute"
    if kind in {"cancellation", "cancel_service", "service_cancellation", "enrollment_cancel"}:
        kind = "cancellation_service_request"
    if kind in {"paid_context", "paid_operation", "paid_refund_context", "paid_transfer_context"}:
        kind = "paid_operation_context"
    if kind in {"payment", "payment_issue", "payment_problem", "payment_claim"}:
        kind = "payment_dispute"
    allowed = _DIRECT_PATH_MODEL_P0_KINDS if include_v2 else _DIRECT_PATH_MODEL_P0_BASE_KINDS
    return kind if kind in allowed else ""


def _direct_path_model_p0_legacy_kind(kind: str) -> str:
    normalized = _direct_path_model_p0_kind(kind, include_v2=True)
    return _DIRECT_PATH_MODEL_P0_LEGACY_KIND.get(normalized, normalized)


def _direct_path_answerability_value(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"yes", "да", "true", "1", "can_answer", "answer_self"}:
        return "yes"
    if text in {"no", "нет", "false", "0", "manager", "manager_only", "handoff"}:
        return "no"
    if text in {"uncertain", "unknown", "не_уверен", "не уверен", "непонятно"}:
        return "uncertain"
    return text[:40] if text else ""


def _direct_path_semantic_frame_answerability_value(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"answer_self", "self", "can_answer", "yes", "да", "true", "1"}:
        return "answer_self"
    if text in {"manager_only", "manager", "handoff", "no", "нет", "false", "0"}:
        return "manager_only"
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


SEMANTIC_FRAME_SCHEMA_VERSION = "semantic_frame_v1_2026_07_01"
SEMANTIC_FRAME_LEGACY_SHADOW_SCHEMA_VERSION = "semantic_frame_shadow_v1_2026_06_30"

_SEMANTIC_FRAME_PHONE_RE = _A2_PHONE_RE
_SEMANTIC_FRAME_EMAIL_RE = _CLIENT_EMAIL_RE
_SEMANTIC_FRAME_LONG_ID_RE = re.compile(r"(?<!\d)\d{5,}(?!\d)")


def _direct_path_semantic_frame_safe_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    text = _SEMANTIC_FRAME_PHONE_RE.sub("[phone]", text)
    text = _SEMANTIC_FRAME_EMAIL_RE.sub("[email]", text)
    text = _SEMANTIC_FRAME_LONG_ID_RE.sub("[id]", text)
    return text[:limit]


def _direct_path_semantic_frame_from_payload(payload: Mapping[str, Any], *, source: str = "") -> dict[str, Any]:
    raw = payload.get("semantic_frame")
    if not isinstance(raw, Mapping):
        raw = payload.get("semanticFrame")
    if not isinstance(raw, Mapping):
        raw = payload.get("semantic_frame_shadow")
    if not isinstance(raw, Mapping):
        return {}
    requested_product = raw.get("requested_product")
    if isinstance(requested_product, Mapping):
        product = {
            "brand": _direct_path_semantic_frame_safe_text(requested_product.get("brand"), limit=40),
            "subject": _direct_path_semantic_frame_safe_text(requested_product.get("subject"), limit=80),
            "grade": _direct_path_semantic_frame_safe_text(requested_product.get("grade"), limit=40),
            "format": _direct_path_semantic_frame_safe_text(requested_product.get("format"), limit=80),
            "venue": _direct_path_semantic_frame_safe_text(requested_product.get("venue"), limit=80),
            "program_kind": _direct_path_semantic_frame_safe_text(requested_product.get("program_kind"), limit=80),
            "raw_text": _direct_path_semantic_frame_safe_text(requested_product.get("raw_text"), limit=160),
        }
    else:
        product = {"raw_text": _direct_path_semantic_frame_safe_text(requested_product, limit=160)}
    frame_source = str(source or raw.get("source") or "").strip().casefold()
    if frame_source not in {"inline", "posthoc"}:
        frame_source = ""
    mode = str(raw.get("mode") or "shadow").strip().casefold()
    if mode not in {"shadow", "active"}:
        mode = "shadow"
    frame = {
        "schema_version": SEMANTIC_FRAME_SCHEMA_VERSION,
        "legacy_schema_version": SEMANTIC_FRAME_LEGACY_SHADOW_SCHEMA_VERSION,
        "mode": mode,
        "source": frame_source,
        "intent": _direct_path_semantic_frame_safe_text(raw.get("intent"), limit=120),
        "risk_class": _direct_path_semantic_frame_safe_text(raw.get("risk_class"), limit=80),
        "deal_stage": _direct_path_semantic_frame_safe_text(raw.get("deal_stage"), limit=80),
        "payment_readiness": _direct_path_semantic_frame_safe_text(raw.get("payment_readiness"), limit=80),
        "requested_product": product,
        "requested_action": _direct_path_semantic_frame_safe_text(raw.get("requested_action"), limit=120),
        "answerability": _direct_path_semantic_frame_answerability_value(raw.get("answerability")),
        "must_handoff": _direct_path_payload_bool(raw.get("must_handoff")),
        "open_question_unanswered": raw.get("open_question_unanswered")
        if isinstance(raw.get("open_question_unanswered"), bool)
        else None,
        "evidence": [
            safe_item
            for item in _clean_list(raw.get("evidence"), max_items=8, max_chars=300)
            if (safe_item := _direct_path_semantic_frame_safe_text(item, limit=160))
        ],
        "confidence": _clamp_float(raw.get("confidence", 0.0)),
    }
    if not any(
        (
            frame["intent"],
            frame["risk_class"],
            frame["deal_stage"],
            frame["payment_readiness"],
            any(str(value or "").strip() for value in frame["requested_product"].values()),
            frame["requested_action"],
            frame["answerability"],
            frame["must_handoff"],
            frame["open_question_unanswered"],
            frame["evidence"],
            frame["confidence"],
        )
    ):
        return {}
    return frame


def build_direct_path_semantic_frame_posthoc_prompt(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    planner = metadata.get("conversation_intent_plan") if isinstance(metadata.get("conversation_intent_plan"), Mapping) else {}
    recent = list(_direct_path_recent_messages(context, limit=8))
    payload = {
        "active_brand": _active_brand(context),
        "client_message": _direct_path_semantic_frame_safe_text(client_message, limit=900),
        "final_route": result.route,
        "final_draft_text": _direct_path_semantic_frame_safe_text(result.draft_text, limit=1400),
        "safety_flags": list(result.safety_flags)[:30],
        "manager_checklist": list(result.manager_checklist)[:12],
        "recent_messages": [_direct_path_semantic_frame_safe_text(item, limit=360) for item in recent[-8:]],
        "direct_path": {
            "model_p0": direct.get("model_p0") or metadata.get("direct_path_model_p0") or {},
            "model_intent": direct.get("model_intent") or metadata.get("direct_path_model_intent") or {},
            "reason_class": metadata.get("reason_class") or direct.get("reason_class") or "",
        },
        "conversation_intent_plan": planner,
    }
    return (
        "Ты заполняешь только телеметрию SemanticFrame для уже готового черновика Telegram-бота.\n"
        "Нельзя переписывать ответ, менять route, менять safety_flags или предлагать клиентский текст.\n"
        "Оцени смысл ситуации по текущей реплике, истории и уже готовому результату.\n"
        "Верни только JSON с одним ключом semantic_frame.\n\n"
        "Ключевая граница: справочная информация != действие менеджера.\n"
        "Сначала классифицируй смысл запроса клиента отдельно от текущего final_route/final_draft_text. final_route=draft_for_manager/manager_only и текст «менеджер проверит» не являются доказательством, что сам запрос требует менеджера.\n"
        "Ставь must_handoff=false и answerability=answer_self только когда ВЕСЬ запрос является безопасной справкой и есть проверенная client-safe опора в final_draft_text или переданных direct_path/conversation metadata: публичная цена без индивидуальных условий, адрес, формат, платформа, программа, возраст/класс, общий порядок записи, общий порядок тестирования, пауза клиента «подумаем/вернёмся позже» без слов про оплату/место, благодарность/подтверждение без просьбы что-то оформить.\n"
        "Ставь must_handoff=true и answerability=manager_only, если хотя бы часть запроса требует человека: P0/жалоба/юридическое/возврат-претензия, подтверждение оплаты или чек, ссылка/реквизиты/альтернативная оплата, сроки или порядок оплаты («оплачу позже/сегодня/завтра»), рассрочка, предоплата, частичный платёж, фиксация цены, удержание/вычет/списание/отработка/возвратные условия, договорные документы, фактическая запись/бронь/лист ожидания/закрепление места, живое наличие мест или подходящей группы, конкретное расписание/доступ «завтра/после оплаты не видно», просьба администратора связаться, персональный подбор преподавателя/группы, индивидуальная ситуация ребёнка после урока/по болезни/по документам, или отсутствует проверенный факт и безопасно ответить нельзя.\n"
        "Не путай стабильную справку о существовании продукта с живым наличием мест. Вопросы «есть ли курс/лагерь/формат для 5 класса», «подходит ли ребёнку после N класса», «есть онлайн/очно?» без просьбы записать, забронировать или проверить места — это requested_action=answer_question, risk_class=safe, answerability=answer_self при наличии проверенного факта. Вопросы «есть места», «можно попасть/забронировать/записаться сейчас», «есть подходящая группа» — это check_availability/manager_only.\n"
        "Не копируй осторожность из final_route: draft_for_manager может быть просто режимом черновика, а не доказательством, что must_handoff=true.\n"
        "Не называй known factual answer missing_facts только потому, что есть manager_approval_required/no_auto_send или осторожный final_route: если в final_draft_text или переданных metadata есть конкретная проверенная опора на справочный вопрос, risk_class=safe. risk_class=missing_facts ставь только когда безопасная справка не имеет проверенной опоры ни в final_draft_text, ни в metadata.\n"
        "В evidence кратко укажи, где видишь опору или нехватку опоры: final_draft_text/direct_path/conversation_intent_plan/recent_messages. Не придумывай факты и не вставляй персональные данные.\n"
        "Поле answerability верни строго одним из: answer_self, manager_only, uncertain. Не используй yes/no.\n"
        "Поле requested_action верни строго одним из перечисленных enum; для благодарности/паузы/получили ссылку используй answer_question, если отдельного действия нет.\n\n"
        "Схема semantic_frame:\n"
        "{\n"
        '  "intent": "главный смысл запроса",\n'
        '  "risk_class": "safe|p0|manager_action|missing_facts|unknown",\n'
        '  "deal_stage": "cold|interest|qualification|offer|closing|post_payment|support|unknown",\n'
        '  "payment_readiness": "none|asking_price|considering|ready_to_pay|paid|dispute|unknown",\n'
        '  "requested_product": {"brand": "", "subject": "", "grade": "", "format": "", "venue": "", "program_kind": "", "raw_text": ""},\n'
        '  "requested_action": "answer_question|check_availability|enroll|send_materials|send_payment_link|send_document|refund_or_cancel|handoff_manager|unknown",\n'
        '  "answerability": "answer_self|manager_only|uncertain",\n'
        '  "must_handoff": false,\n'
        '  "evidence": ["короткие неперсональные причины без телефонов, email и ФИО"],\n'
        '  "confidence": 0.0\n'
        "}\n\n"
        "Данные для классификации:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _direct_path_model_p0_meta(result: SubscriptionDraftResult) -> Mapping[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    meta = metadata.get("direct_path_model_p0")
    return meta if isinstance(meta, Mapping) else {}


def _direct_path_p0_shadow_metadata(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    if not _direct_path_model_p0_enabled(context):
        return {}
    meta = _direct_path_model_p0_meta(result)
    include_v2 = _p0_model_classes_v2_enabled(context)
    kind = _direct_path_model_p0_kind(meta.get("p0_kind_raw"), include_v2=include_v2)
    if not kind:
        kind = _direct_path_model_p0_kind(meta.get("p0_kind"), include_v2=include_v2)
    model_field_present = (
        meta.get("is_p0_present") is True
        if "is_p0_present" in meta
        else "is_p0" in meta
    )
    validity_marker = meta.get("is_p0_valid", True)
    model_field_valid = model_field_present and isinstance(meta.get("is_p0"), bool) and validity_marker is True
    model_contract_status = "missing" if not model_field_present else "valid" if model_field_valid else "invalid"
    model_is_p0 = bool(meta.get("is_p0")) if model_field_valid else False
    model_effective_is_p0 = model_is_p0
    return {
        "schema_version": "p0_model_shadow_v1_2026_07_29",
        "model_field_present": model_field_present,
        "model_field_valid": model_field_valid,
        "model_contract_status": model_contract_status,
        "model_is_p0": model_is_p0,
        "model_effective_is_p0": model_effective_is_p0,
        "model_p0_kind": kind,
    }


def _apply_direct_path_p0_shadow(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    shadow = _direct_path_p0_shadow_metadata(result, client_message=client_message, context=context)
    if not shadow:
        return result
    metadata = dict(result.metadata)
    metadata["p0_model_shadow"] = shadow
    return replace(result, metadata=metadata)


def _direct_path_model_p0_signal(result: SubscriptionDraftResult, *, client_message: str, context: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not _direct_path_model_p0_enabled(context):
        return {}
    meta = _direct_path_model_p0_meta(result)
    include_v2 = _p0_model_classes_v2_enabled(context)
    kind = _direct_path_model_p0_kind(meta.get("p0_kind_raw"), include_v2=include_v2)
    if not kind:
        kind = _direct_path_model_p0_kind(meta.get("p0_kind"), include_v2=include_v2)
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    shadow = metadata.get("p0_model_shadow")
    if not isinstance(shadow, Mapping):
        shadow = _direct_path_p0_shadow_metadata(result, client_message=client_message, context=context)
    contract_status = str(shadow.get("model_contract_status") or "")
    if contract_status == "missing" or contract_status == "invalid":
        return {"contract_error": contract_status}
    model_is_p0 = bool(shadow.get("model_effective_is_p0"))
    if not model_is_p0:
        return {}
    if not kind:
        kind = "complaint"
    return {
        "is_p0": True,
        "p0_kind": kind,
        "risk_level": "high",
        "model_reason": str(meta.get("model_reason") or "").strip()[:240],
        "source": "model_p0",
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
    contract_error = str(signal.get("contract_error") or "")
    if contract_error:
        metadata = dict(result.metadata)
        direct = dict(metadata.get("direct_path") or {})
        direct["reason_class"] = "provider_runtime"
        direct["reason_evidence"] = {"provider_error": f"model_p0_contract_{contract_error}"}
        direct["text_composition_source"] = "provider_runtime_fallback"
        metadata["direct_path"] = direct
        metadata["direct_path_model_p0_contract"] = {
            "schema_version": "direct_path_model_p0_contract_v1_2026_08_04",
            "status": contract_error,
        }
        metadata["is_manager_deferral"] = True
        fallback = safe_fallback_draft(reason="model_p0_contract_invalid", metadata=metadata)
        return replace(
            fallback,
            safety_flags=tuple(dict.fromkeys((*fallback.safety_flags, "direct_path_model_contract_invalid"))),
        )
    kind = str(signal.get("p0_kind") or "complaint")
    legacy_kind = _direct_path_model_p0_legacy_kind(kind)
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    direct["model_p0"] = {
        "is_p0": True,
        "route_applied": True,
        "p0_kind": kind,
        "legacy_p0_kind": legacy_kind if legacy_kind != kind else "",
        "risk_level": "high",
        "model_reason": str(signal.get("model_reason") or ""),
        "source": str(signal.get("source") or "model_p0"),
    }
    metadata["direct_path_model_p0"] = dict(direct["model_p0"])
    metadata["direct_path"] = direct
    metadata["reason_class"] = "p0_deferral"
    metadata["is_manager_deferral"] = True
    mapped_flags: list[str] = [f"direct_path_model_p0_{kind}", kind]
    if legacy_kind and legacy_kind != kind:
        mapped_flags.extend([f"direct_path_model_p0_{legacy_kind}", legacy_kind])
        if legacy_kind == "legal_threat":
            mapped_flags.append("legal")
    elif kind == "legal_threat":
        mapped_flags.append("legal")
    flags = tuple(
        dict.fromkeys(
            [
                *result.safety_flags,
                *mapped_flags,
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


SEMANTIC_FRAME_DECISION_SHADOW_SCHEMA_VERSION = "semantic_frame_decision_shadow_v1_2026_07_01"
SEMANTIC_FRAME_MANAGER_ACTION_GATE_SCHEMA_VERSION = "semantic_frame_manager_action_gate_v1_2026_07_01"
SEMANTIC_FRAME_SELF_ANSWER_SHADOW_SCHEMA_VERSION = "semantic_frame_self_answer_shadow_v1_2026_07_02"
SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW_SCHEMA_VERSION = "semantic_frame_existence_proof_shadow_v1_2026_07_02"
SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW_SCHEMA_VERSION = "semantic_frame_proof_reconciliation_shadow_v1_2026_07_02"

_SEMANTIC_FRAME_P0_FLAGS = {
    "p0",
    "refund",
    "refund_claim",
    "payment_dispute",
    "complaint",
    "legal",
    "legal_threat",
    "high_risk",
    "p0_deferral",
    "zero_collect_required",
    "direct_path_model_p0_refund",
    "direct_path_model_p0_payment_dispute",
    "direct_path_model_p0_complaint",
    "direct_path_model_p0_legal_threat",
    "direct_path_model_p0_contract_dispute",
    "direct_path_model_p0_cancellation_service_request",
    "direct_path_model_p0_paid_operation_context",
}

_SEMANTIC_FRAME_MANAGER_ACTION_GATE_CONFIDENCE = 0.8
_SEMANTIC_FRAME_MANAGER_ACTION_GATE_STAGES = {"closing", "post_payment", "support"}
_SEMANTIC_FRAME_MANAGER_ACTION_GATE_PAID_STATES = {"paid", "dispute"}

_SEMANTIC_FRAME_SELF_ANSWER_CONFIDENCE = 0.9
_SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_FLAGS = {
    "authoritative_output_gate_blocked",
    "autonomy_default_cautious_unverified_fact",
    "autonomy_default_cautious_live_status_missing",
    "future_price_handoff_applied",
    "price_future_manager_only",
    "presale_refund_policy_manager_check",
    "direct_path_preblocked",
}
_SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_SUBSTRINGS = (
    "manager_only",
    "payment_dispute",
    "refund_claim",
    "complaint",
    "legal",
    "zero_collect",
    "output_sanitizer:",
    "client_name_echo",
    "internal_client_placeholder",
)
_SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_PAYMENT = {"ready_to_pay", "paid", "dispute"}
_SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_STAGES = {"post_payment", "support"}


def _semantic_frame_value(frame: Mapping[str, Any], key: str) -> str:
    return str(frame.get(key) or "").strip().casefold()


def _semantic_frame_from_result(result: SubscriptionDraftResult) -> Mapping[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    for key in ("semantic_frame", "semantic_frame_shadow"):
        frame = metadata.get(key)
        if isinstance(frame, Mapping):
            return frame
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping):
        for key in ("semantic_frame", "semantic_frame_shadow"):
            frame = direct.get(key)
            if isinstance(frame, Mapping):
                return frame
    return {}


def _semantic_frame_posthoc_ok(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    status = metadata.get("semantic_frame_posthoc_shadow")
    if isinstance(status, Mapping) and str(status.get("status") or "").strip() == "ok":
        return True
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping):
        status = direct.get("semantic_frame_posthoc_shadow")
        return isinstance(status, Mapping) and str(status.get("status") or "").strip() == "ok"
    return False


def _semantic_frame_actual_p0(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct_model_p0 = metadata.get("direct_path_model_p0")
    if isinstance(direct_model_p0, Mapping) and bool(direct_model_p0.get("is_p0")):
        return True
    direct = metadata.get("direct_path")
    if isinstance(direct, Mapping) and isinstance(direct.get("model_p0"), Mapping):
        return True
    flags = {str(flag or "").strip() for flag in result.safety_flags if str(flag or "").strip()}
    if flags.intersection(_SEMANTIC_FRAME_P0_FLAGS):
        return True
    return str(result.risk_level or "").strip().casefold() in {"high", "p0", "critical", "high_risk"} and result.route == "manager_only"


def _semantic_frame_alignment(expected: Optional[bool], actual: bool) -> str:
    if expected is None:
        return "unknown"
    return "match" if bool(expected) == bool(actual) else "mismatch"


def _semantic_frame_expected_handoff(frame: Mapping[str, Any]) -> Optional[bool]:
    answerability = _semantic_frame_value(frame, "answerability")
    risk_class = _semantic_frame_value(frame, "risk_class")
    must_handoff = _semantic_frame_bool(frame.get("must_handoff"))
    if must_handoff is True or answerability == "manager_only" or risk_class in {"p0", "manager_action"}:
        return True
    if answerability == "answer_self" or risk_class == "safe":
        return False
    return None


def _semantic_frame_requested_product_brand(frame: Mapping[str, Any]) -> str:
    product = frame.get("requested_product")
    if isinstance(product, Mapping):
        raw = str(product.get("brand") or "").strip().casefold()
        if raw in {"фотон", "foton"}:
            return "foton"
        if raw in {"унпк", "унпк мфти", "unpk"}:
            return "unpk"
        return raw
    return ""


def _semantic_frame_self_class(frame: Mapping[str, Any], direct: Mapping[str, Any]) -> str:
    selected = str(direct.get("selected_category") or "").strip().casefold()
    intent = _direct_path_semantic_frame_safe_text(frame.get("intent"), limit=160).casefold()
    haystack = f"{selected} {intent}"
    if "price" in haystack or "pricing" in haystack or "стоим" in haystack or "цен" in haystack:
        return "price"
    if "schedule" in haystack or "распис" in haystack or "дат" in haystack:
        return "schedule"
    if "address" in haystack or "адрес" in haystack or "площад" in haystack:
        return "address"
    if "format" in haystack or "online" in haystack or "онлайн" in haystack or "очно" in haystack:
        return "format"
    if "platform" in haystack or "платформ" in haystack:
        return "platform"
    if "course" in haystack or "program" in haystack or "курс" in haystack or "программ" in haystack:
        return "program"
    if "thanks" in haystack or "gratitude" in haystack or "спасибо" in haystack:
        return "safe_close"
    return selected or "safe_reference"


def _semantic_frame_truthy_text(value: Any) -> bool:
    normalized = str(value or "").strip().casefold()
    return normalized in {"1", "true", "yes", "y", "да", "on"}


def _semantic_frame_existence_proof_shadow_trace(
    frame: Mapping[str, Any],
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    active_brand = _active_brand(context)
    base: dict[str, Any] = {
        "schema_version": SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW_SCHEMA_VERSION,
        "enabled": True,
        "route_text_shadow_only": True,
        "status": "blocked",
        "reason": "",
        "exact_fact_keys": [],
        "fact_metadata": {},
    }
    if not frame:
        return {**base, "reason": "no_frame"}
    if _semantic_frame_value(frame, "requested_action") != "answer_question":
        return {**base, "reason": "requested_action_not_answer_question"}
    if _semantic_frame_bool(frame.get("must_handoff")) is True and _semantic_frame_value(frame, "risk_class") in {
        "p0",
        "manager_action",
    }:
        return {**base, "reason": "protected_handoff_frame"}
    if _semantic_frame_value(frame, "payment_readiness") in _SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_PAYMENT:
        return {**base, "reason": "payment_readiness_blocked"}
    if _semantic_frame_value(frame, "deal_stage") in _SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_STAGES:
        return {**base, "reason": "deal_stage_blocked"}

    requested = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    requested_brand = _semantic_frame_requested_product_brand(frame)
    brand = requested_brand if requested_brand in {"foton", "unpk"} else active_brand
    if brand not in {"foton", "unpk"}:
        return {**base, "reason": "unknown_brand"}
    if requested_brand and requested_brand != brand:
        return {**base, "reason": "brand_mismatch", "brand": brand, "requested_brand": requested_brand}

    snapshot = _direct_path_load_snapshot(_direct_path_snapshot_path_from_context(context))
    records = [
        fact
        for fact in _direct_path_snapshot_facts(snapshot)
        if _direct_path_client_safe_snapshot_fact(fact, active_brand=brand)
    ]
    if not records:
        return {**base, "reason": "no_client_safe_snapshot_facts", "brand": brand}

    proof = verify_product_format_exists(
        build_product_existence_axes_catalog(records),
        brand=brand,
        grade=str(requested.get("grade") or ""),
        subject=str(requested.get("subject") or ""),
        format=str(requested.get("format") or ""),
        program_kind=str(requested.get("program_kind") or ""),
        product_family=str(requested.get("raw_text") or ""),
    )
    status = str(proof.get("status") or "").strip()
    entry = proof.get("entry") if isinstance(proof.get("entry"), Mapping) else {}
    if status not in {"exists", "not_offered"} or not entry:
        return {
            **base,
            "reason": str(proof.get("reason") or status or "no_exact_product_existence_fact"),
            "brand": brand,
            "proof_status": status,
            "query_axes": proof.get("query_axes") if isinstance(proof.get("query_axes"), Mapping) else {},
        }

    fact_key = str(entry.get("source_fact_key") or "").strip()
    valid_from = str(entry.get("valid_from") or "").strip()
    valid_until = str(entry.get("valid_until") or "").strip()
    freshness_check_date = str(entry.get("freshness_check_date") or "").strip()
    if not fact_key:
        return {**base, "reason": "empty_source_fact_key", "brand": brand, "proof_status": status}
    fact_metadata = {
        fact_key: {
            "brand": brand,
            "client_safe": "true",
            "valid_until": valid_until,
            "valid_from": valid_from,
            "freshness_check_date": freshness_check_date,
            "source": "semantic_frame_existence_proof_shadow",
            "proof_status": status,
            "fact_type": str(entry.get("source_fact_type") or ""),
            "product_family": str(entry.get("product_family") or ""),
            "program_kind": str(entry.get("program_kind") or ""),
            "format": str(entry.get("format") or ""),
        }
    }
    return {
        **base,
        "status": status,
        "reason": str(proof.get("reason") or "exact_product_existence_fact"),
        "brand": brand,
        "query_axes": proof.get("query_axes") if isinstance(proof.get("query_axes"), Mapping) else {},
        "exact_fact_keys": [fact_key],
        "fact_metadata": fact_metadata,
        "source_fact_key": fact_key,
        "valid_until": valid_until,
    }


def _semantic_frame_fresh_client_safe_fact_trace(
    direct: Mapping[str, Any],
    *,
    active_brand: str,
) -> dict[str, Any]:
    exact_keys = [str(key or "").strip() for key in (direct.get("wide_fact_exact_keys") or ()) if str(key or "").strip()]
    fact_meta = dict(direct.get("wide_fact_metadata") if isinstance(direct.get("wide_fact_metadata"), Mapping) else {})
    proof_shadow = direct.get("semantic_frame_existence_proof_shadow")
    proof_keys: list[str] = []
    if isinstance(proof_shadow, Mapping) and str(proof_shadow.get("status") or "") in {"exists", "not_offered"}:
        proof_keys = [
            str(key or "").strip()
            for key in (proof_shadow.get("exact_fact_keys") or ())
            if str(key or "").strip()
        ]
        proof_meta = proof_shadow.get("fact_metadata") if isinstance(proof_shadow.get("fact_metadata"), Mapping) else {}
        for key, value in proof_meta.items():
            if str(key or "").strip() and isinstance(value, Mapping):
                fact_meta[str(key)] = dict(value)
        exact_keys = list(dict.fromkeys([*exact_keys, *proof_keys]))
    checked: list[dict[str, str]] = []
    fresh_checked: list[dict[str, str]] = []
    base = {
        "exact_fact_count": len(exact_keys),
        "existence_proof_shadow_count": len(proof_keys),
        "checked_count": 0,
        "fresh_client_safe_count": 0,
        "all_exact_facts_fresh_client_safe": False,
    }
    if not exact_keys:
        return {"ok": False, "reason": "no_exact_fact_keys", "checked": checked, **base}
    for key in exact_keys:
        raw = fact_meta.get(key) if isinstance(fact_meta, Mapping) else None
        meta = raw if isinstance(raw, Mapping) else {}
        brand = str(meta.get("brand") or "").strip().casefold()
        valid_from = str(meta.get("valid_from") or "").strip()
        valid_until = str(meta.get("valid_until") or "").strip()
        client_safe = _semantic_frame_truthy_text(meta.get("client_safe"))
        validity_window_ok = bool(valid_until) and fact_runtime_time_ok(meta)
        checked.append(
            {
                "fact_key": key,
                "brand": brand,
                "client_safe": "true" if client_safe else "false",
                "valid_until": valid_until,
                "valid_from": valid_from,
                "valid_until_ok": "true" if validity_window_ok else "false",
                "source": str(meta.get("source") or "wide_fact_pack"),
            }
        )
        if brand == active_brand and client_safe and validity_window_ok:
            fresh_checked.append(checked[-1])
    base = {
        **base,
        "checked_count": len(checked),
        "fresh_client_safe_count": len(fresh_checked),
        "all_exact_facts_fresh_client_safe": bool(exact_keys) and len(fresh_checked) == len(exact_keys),
        "checked_truncated": len(checked) > 16,
    }
    checked_trace = checked[:16]
    if fresh_checked:
        first = fresh_checked[0]
        return {
            "ok": True,
            "reason": "fresh_client_safe_exact_fact",
            "fact_key": first.get("fact_key", ""),
            "valid_until": first.get("valid_until", ""),
            "checked": checked_trace,
            **base,
        }
    return {"ok": False, "reason": "no_fresh_client_safe_exact_fact", "checked": checked_trace, **base}


def _semantic_frame_self_answer_blocking_flags(result: SubscriptionDraftResult) -> tuple[str, ...]:
    flags = [str(flag or "").strip() for flag in result.safety_flags if str(flag or "").strip()]
    blocked: list[str] = []
    for flag in flags:
        folded = flag.casefold()
        if flag in _SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_FLAGS:
            blocked.append(flag)
            continue
        if any(marker in folded for marker in _SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_SUBSTRINGS):
            blocked.append(flag)
    return tuple(dict.fromkeys(blocked))


def _semantic_frame_self_answer_shadow_trace(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    frame = _semantic_frame_from_result(result)
    route_before = result.route
    base = {
        "schema_version": SEMANTIC_FRAME_SELF_ANSWER_SHADOW_SCHEMA_VERSION,
        "enabled": True,
        "threshold": _SEMANTIC_FRAME_SELF_ANSWER_CONFIDENCE,
        "route_before": route_before,
        "route_after_if_active": route_before,
    }
    if not frame:
        return {**base, "status": "blocked", "reason": "no_frame"}
    frame_schema = str(frame.get("schema_version") or "").strip()
    confidence = _clamp_float(frame.get("confidence", 0.0))
    active_brand = _active_brand(context)
    product_brand = _semantic_frame_requested_product_brand(frame)
    blocking_flags = _semantic_frame_self_answer_blocking_flags(result)
    freshness = _semantic_frame_fresh_client_safe_fact_trace(direct, active_brand=active_brand)
    frame_trace = {
        "schema_version": frame_schema,
        "confidence": confidence,
        "intent": _direct_path_semantic_frame_safe_text(frame.get("intent"), limit=120),
        "risk_class": _semantic_frame_value(frame, "risk_class"),
        "deal_stage": _semantic_frame_value(frame, "deal_stage"),
        "payment_readiness": _semantic_frame_value(frame, "payment_readiness"),
        "requested_action": _semantic_frame_value(frame, "requested_action"),
        "answerability": _semantic_frame_value(frame, "answerability"),
        "must_handoff": _semantic_frame_bool(frame.get("must_handoff")),
    }
    trace = {
        **base,
        "self_class": _semantic_frame_self_class(frame, direct),
        "active_brand": active_brand,
        "frame": frame_trace,
        "guards": {
            "posthoc_ok": _semantic_frame_posthoc_ok(result),
            "actual_p0": _semantic_frame_actual_p0(result),
            "blocking_flags": list(blocking_flags),
            "has_missing_facts": bool(result.missing_facts),
            "has_forbidden_promises": bool(result.forbidden_promises_detected),
            "freshness": freshness,
        },
    }

    reason = ""
    if frame_schema != SEMANTIC_FRAME_SCHEMA_VERSION:
        reason = "unsupported_frame_schema"
    elif not _semantic_frame_posthoc_ok(result):
        reason = "frame_not_posthoc"
    elif _semantic_frame_actual_p0(result):
        reason = "protected_p0"
    elif route_before != "draft_for_manager":
        reason = "route_not_draft_for_manager"
    elif active_brand not in {"foton", "unpk"}:
        reason = "unknown_active_brand"
    elif product_brand and product_brand not in {active_brand, "unknown"}:
        reason = "frame_brand_mismatch"
    elif confidence < _SEMANTIC_FRAME_SELF_ANSWER_CONFIDENCE:
        reason = "low_confidence"
    elif _semantic_frame_value(frame, "risk_class") != "safe":
        reason = "risk_class_not_safe"
    elif _semantic_frame_value(frame, "answerability") != "answer_self":
        reason = "answerability_not_self"
    elif _semantic_frame_bool(frame.get("must_handoff")) is not False:
        reason = "must_handoff_not_false"
    elif _semantic_frame_value(frame, "requested_action") != "answer_question":
        reason = "requested_action_not_answer_question"
    elif _semantic_frame_value(frame, "payment_readiness") in _SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_PAYMENT:
        reason = "payment_readiness_blocked"
    elif _semantic_frame_value(frame, "deal_stage") in _SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_STAGES:
        reason = "deal_stage_blocked"
    elif blocking_flags:
        reason = "blocking_safety_flags"
    elif bool(direct.get("deferral_text_in_self")):
        reason = "deferral_text_in_self"
    elif result.missing_facts:
        reason = "missing_facts"
    elif result.forbidden_promises_detected:
        reason = "forbidden_promises"
    elif not bool(freshness.get("ok")):
        reason = str(freshness.get("reason") or "freshness_unknown")

    if reason:
        return {**trace, "status": "blocked", "reason": reason}
    return {
        **trace,
        "status": "would_demote_to_self",
        "reason": "safe_answer_self_fresh_fact",
        "route_after_if_active": "bot_answer_self_for_pilot",
    }


def _semantic_frame_manager_action_gate_reason(frame: Mapping[str, Any]) -> tuple[bool, str]:
    confidence = _clamp_float(frame.get("confidence", 0.0))
    if confidence < _SEMANTIC_FRAME_MANAGER_ACTION_GATE_CONFIDENCE:
        return False, "low_confidence"
    if _semantic_frame_value(frame, "risk_class") != "manager_action":
        return False, "risk_class_not_manager_action"

    must_handoff = _semantic_frame_bool(frame.get("must_handoff"))
    answerability = _semantic_frame_value(frame, "answerability")
    if must_handoff is not True and answerability != "manager_only":
        return False, "no_strong_handoff_signal"

    requested_action = _semantic_frame_value(frame, "requested_action")
    deal_stage = _semantic_frame_value(frame, "deal_stage")
    payment_readiness = _semantic_frame_value(frame, "payment_readiness")

    if requested_action == "check_availability" and deal_stage in _SEMANTIC_FRAME_MANAGER_ACTION_GATE_STAGES:
        return True, "manager_action:check_availability"
    if requested_action in {"handoff_manager", "send_document"}:
        return True, f"manager_action:{requested_action}"
    if requested_action == "enroll" and (
        deal_stage in _SEMANTIC_FRAME_MANAGER_ACTION_GATE_STAGES
        or payment_readiness in {"ready_to_pay", *_SEMANTIC_FRAME_MANAGER_ACTION_GATE_PAID_STATES}
    ):
        return True, "manager_action:enroll_closing"
    if payment_readiness in _SEMANTIC_FRAME_MANAGER_ACTION_GATE_PAID_STATES and requested_action in {
        "unknown",
        "handoff_manager",
        "send_document",
    }:
        return True, f"manager_action:payment_{payment_readiness}"
    return False, "unsupported_manager_action"


def apply_semantic_frame_existence_proof_shadow(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _semantic_frame_existence_proof_shadow_enabled(context):
        return result
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    trace = _semantic_frame_existence_proof_shadow_trace(_semantic_frame_from_result(result), context=context)
    metadata["semantic_frame_existence_proof_shadow"] = trace
    direct["semantic_frame_existence_proof_shadow"] = trace
    metadata["direct_path"] = direct
    return replace(result, metadata=metadata)


def _semantic_frame_proof_reconciliation_shadow_trace(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    frame = _semantic_frame_from_result(result)
    active_brand = _active_brand(context)
    proof = direct.get("semantic_frame_existence_proof_shadow")
    if not isinstance(proof, Mapping):
        proof = metadata.get("semantic_frame_existence_proof_shadow")
    if not isinstance(proof, Mapping):
        proof = {}
    freshness = _semantic_frame_fresh_client_safe_fact_trace(direct, active_brand=active_brand)
    base = {
        "schema_version": SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW_SCHEMA_VERSION,
        "enabled": True,
        "route_text_shadow_only": True,
        "active_behavior_allowed": False,
        "status": "blocked",
        "reason": "",
        "route_before": result.route,
        "route_after_if_active": result.route,
        "active_blockers": [],
        "result_missing_facts": [str(item) for item in result.missing_facts],
        "proof_status": str(proof.get("status") or ""),
        "proof_reason": str(proof.get("reason") or ""),
        "source_fact_key": str(proof.get("source_fact_key") or ""),
        "valid_until": str(proof.get("valid_until") or ""),
        "query_axes": proof.get("query_axes") if isinstance(proof.get("query_axes"), Mapping) else {},
        "exact_fact_keys": [
            str(key or "").strip()
            for key in (proof.get("exact_fact_keys") or ())
            if str(key or "").strip()
        ],
        "freshness": freshness,
    }
    if not frame:
        return {**base, "reason": "no_frame", "active_blockers": ["no_frame"]}
    frame_schema = str(frame.get("schema_version") or "").strip()
    frame_trace = {
        "schema_version": frame_schema,
        "confidence": _clamp_float(frame.get("confidence", 0.0)),
        "intent": _direct_path_semantic_frame_safe_text(frame.get("intent"), limit=120),
        "risk_class": _semantic_frame_value(frame, "risk_class"),
        "deal_stage": _semantic_frame_value(frame, "deal_stage"),
        "payment_readiness": _semantic_frame_value(frame, "payment_readiness"),
        "requested_action": _semantic_frame_value(frame, "requested_action"),
        "answerability": _semantic_frame_value(frame, "answerability"),
        "must_handoff": _semantic_frame_bool(frame.get("must_handoff")),
    }
    trace = {
        **base,
        "frame_before": frame_trace,
    }
    if frame_schema != SEMANTIC_FRAME_SCHEMA_VERSION:
        return {**trace, "reason": "unsupported_frame_schema", "active_blockers": ["unsupported_frame_schema"]}
    if not _semantic_frame_posthoc_ok(result):
        return {**trace, "reason": "frame_not_posthoc", "active_blockers": ["frame_not_posthoc"]}
    if _semantic_frame_actual_p0(result):
        return {**trace, "reason": "protected_p0", "active_blockers": ["protected_p0"]}
    requested_action = _semantic_frame_value(frame, "requested_action")
    risk_class = _semantic_frame_value(frame, "risk_class")
    if requested_action not in {"answer_question", "check_availability"}:
        return {**trace, "reason": "requested_action_not_reconcilable", "active_blockers": ["requested_action_not_reconcilable"]}
    if risk_class in {"p0"}:
        return {**trace, "reason": "protected_handoff_frame", "active_blockers": ["protected_handoff_frame"]}
    if risk_class not in {"safe", "missing_facts", "manager_action"}:
        return {**trace, "reason": "risk_class_not_reconcilable", "active_blockers": ["risk_class_not_reconcilable"]}
    if _semantic_frame_value(frame, "payment_readiness") in _SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_PAYMENT:
        return {**trace, "reason": "payment_readiness_blocked", "active_blockers": ["payment_readiness_blocked"]}
    if _semantic_frame_value(frame, "deal_stage") in _SEMANTIC_FRAME_SELF_ANSWER_BLOCKING_STAGES:
        return {**trace, "reason": "deal_stage_blocked", "active_blockers": ["deal_stage_blocked"]}
    if not bool(freshness.get("ok")):
        reason = str(freshness.get("reason") or "freshness_unknown")
        return {**trace, "reason": reason, "active_blockers": [reason]}

    answerability = _semantic_frame_value(frame, "answerability")
    must_handoff = _semantic_frame_bool(frame.get("must_handoff"))
    if risk_class == "safe" and answerability == "answer_self" and must_handoff is False:
        return {**trace, "status": "already_aligned", "reason": "frame_already_safe_answer_self"}
    if risk_class == "missing_facts" or answerability == "manager_only" or must_handoff is True:
        active_blockers = [
            "shadow_only_reconciliation",
            "requires_text_readiness_policy",
            "requires_existence_vs_live_availability_semantic_review",
        ]
        if requested_action == "check_availability":
            active_blockers.append("current_frame_requested_action_check_availability")
        if risk_class == "manager_action":
            active_blockers.append("current_frame_risk_class_manager_action")
        if result.missing_facts:
            active_blockers.append("result_missing_facts_present")
        return {
            **trace,
            "status": "would_reconcile_to_safe_reference",
            "reason": "fresh_proof_contradicts_missing_facts_frame",
            "active_blockers": active_blockers,
            "reconciled_frame_if_applied": {
                "risk_class": "safe",
                "answerability": "answer_self",
                "must_handoff": False,
                "requested_action": "answer_question",
            },
        }
    return {**trace, "status": "pass", "reason": "frame_not_missing_facts_or_manager_only"}


def apply_semantic_frame_proof_reconciliation_shadow(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _semantic_frame_proof_reconciliation_shadow_enabled(context):
        return result
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    trace = _semantic_frame_proof_reconciliation_shadow_trace(result, context=context)
    metadata["semantic_frame_proof_reconciliation_shadow"] = trace
    direct["semantic_frame_proof_reconciliation_shadow"] = trace
    metadata["direct_path"] = direct
    return replace(result, metadata=metadata)


def apply_semantic_frame_manager_action_gate(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _semantic_frame_manager_action_gate_enabled(context):
        return result

    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    frame = _semantic_frame_from_result(result)
    route_before = result.route

    if not frame:
        trace = {
            "schema_version": SEMANTIC_FRAME_MANAGER_ACTION_GATE_SCHEMA_VERSION,
            "enabled": True,
            "status": "no_frame",
            "route_before": route_before,
            "route_after": route_before,
        }
        metadata["semantic_frame_manager_action_gate"] = trace
        direct["semantic_frame_manager_action_gate"] = trace
        metadata["direct_path"] = direct
        return replace(result, metadata=metadata)

    if not _semantic_frame_posthoc_ok(result):
        trace = {
            "schema_version": SEMANTIC_FRAME_MANAGER_ACTION_GATE_SCHEMA_VERSION,
            "enabled": True,
            "status": "frame_not_posthoc",
            "route_before": route_before,
            "route_after": route_before,
        }
        metadata["semantic_frame_manager_action_gate"] = trace
        direct["semantic_frame_manager_action_gate"] = trace
        metadata["direct_path"] = direct
        return replace(result, metadata=metadata)

    if _seats_default_open_allowlisted_result(result):
        trace = {
            "schema_version": SEMANTIC_FRAME_MANAGER_ACTION_GATE_SCHEMA_VERSION,
            "enabled": True,
            "status": "pass",
            "reason": "seats_default_open_regular_groups_allowlist",
            "route_before": route_before,
            "route_after": route_before,
        }
        metadata["semantic_frame_manager_action_gate"] = trace
        direct["semantic_frame_manager_action_gate"] = trace
        metadata["direct_path"] = direct
        return replace(result, metadata=metadata)

    should_gate, reason = _semantic_frame_manager_action_gate_reason(frame)
    route_is_autonomous = route_before in AUTONOMOUS_ROUTES
    status = "promoted_to_draft_for_manager" if should_gate and route_is_autonomous else "pass"
    route_after = "draft_for_manager" if status == "promoted_to_draft_for_manager" else route_before
    trace = {
        "schema_version": SEMANTIC_FRAME_MANAGER_ACTION_GATE_SCHEMA_VERSION,
        "enabled": True,
        "status": status,
        "reason": reason,
        "route_before": route_before,
        "route_after": route_after,
        "frame": {
            "confidence": _clamp_float(frame.get("confidence", 0.0)),
            "intent": _direct_path_semantic_frame_safe_text(frame.get("intent"), limit=120),
            "risk_class": _semantic_frame_value(frame, "risk_class"),
            "deal_stage": _semantic_frame_value(frame, "deal_stage"),
            "payment_readiness": _semantic_frame_value(frame, "payment_readiness"),
            "requested_action": _semantic_frame_value(frame, "requested_action"),
            "answerability": _semantic_frame_value(frame, "answerability"),
            "must_handoff": _semantic_frame_bool(frame.get("must_handoff")),
        },
    }
    metadata["semantic_frame_manager_action_gate"] = trace
    direct["semantic_frame_manager_action_gate"] = trace
    metadata["direct_path"] = direct
    if status != "promoted_to_draft_for_manager":
        return replace(result, metadata=metadata)

    flags = tuple(
        dict.fromkeys(
            [
                *result.safety_flags,
                "semantic_frame_manager_action_gate",
                "manager_approval_required",
                "no_auto_send",
            ]
        )
    )
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "SemanticFrame: проверить действие менеджера перед ответом клиенту.",
            ]
        )
    )
    return replace(
        result,
        route="draft_for_manager",
        safety_flags=flags,
        manager_checklist=checklist,
        metadata=metadata,
    )


def apply_semantic_frame_self_answer_shadow(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _semantic_frame_self_answer_shadow_enabled(context):
        return result
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    trace = _semantic_frame_self_answer_shadow_trace(result, context=context)
    metadata["semantic_frame_self_answer_shadow"] = trace
    direct["semantic_frame_self_answer_shadow"] = trace
    metadata["direct_path"] = direct
    return replace(result, metadata=metadata)


def apply_semantic_frame_decision_shadow(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _semantic_frame_decision_shadow_enabled(context):
        return result
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    frame = _semantic_frame_from_result(result)
    if not frame:
        shadow = {
            "schema_version": SEMANTIC_FRAME_DECISION_SHADOW_SCHEMA_VERSION,
            "enabled": True,
            "status": "no_frame",
            "route_after": result.route,
        }
    else:
        actual_handoff = result.route not in AUTONOMOUS_ROUTES
        actual_p0 = _semantic_frame_actual_p0(result)
        frame_p0 = _semantic_frame_value(frame, "risk_class") == "p0"
        expected_handoff = _semantic_frame_expected_handoff(frame)
        shadow = {
            "schema_version": SEMANTIC_FRAME_DECISION_SHADOW_SCHEMA_VERSION,
            "enabled": True,
            "status": "observed",
            "frame": {
                "mode": str(frame.get("mode") or "").strip(),
                "confidence": _clamp_float(frame.get("confidence", 0.0)),
                "intent": _direct_path_semantic_frame_safe_text(frame.get("intent"), limit=120),
                "risk_class": _semantic_frame_value(frame, "risk_class"),
                "deal_stage": _semantic_frame_value(frame, "deal_stage"),
                "payment_readiness": _semantic_frame_value(frame, "payment_readiness"),
                "requested_action": _semantic_frame_value(frame, "requested_action"),
                "answerability": _semantic_frame_value(frame, "answerability"),
                "must_handoff": _semantic_frame_bool(frame.get("must_handoff")) is True,
                "evidence_count": (
                    len(frame.get("evidence") or ())
                    if isinstance(frame.get("evidence"), Sequence)
                    and not isinstance(frame.get("evidence"), (str, bytes))
                    else 0
                ),
            },
            "actual": {
                "route_after": result.route,
                "handoff": actual_handoff,
                "manager_only": result.route == "manager_only",
                "p0": actual_p0,
                "direct_selected_category": str(direct.get("selected_category") or "").strip(),
            },
            "comparisons": {
                "must_handoff_vs_route": _semantic_frame_alignment(expected_handoff, actual_handoff),
                "p0_vs_actual": _semantic_frame_alignment(frame_p0, actual_p0),
                "answerability_vs_route": _semantic_frame_alignment(expected_handoff, actual_handoff),
            },
        }
    metadata["frame_decision_shadow"] = shadow
    direct["frame_decision_shadow"] = shadow
    metadata["direct_path"] = direct
    return replace(result, metadata=metadata)


def apply_semantic_reading_trace_finalize(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    del context
    metadata = finalize_reading_trace_metadata(result.metadata)
    if metadata == result.metadata:
        return result
    return replace(result, metadata=metadata)


def _normalize_direct_path_payload(
    payload: Mapping[str, Any],
    *,
    raw_response: Optional[str] = None,
    include_answerability_self: bool = False,
    include_semantic_frame_shadow: bool = False,
    include_dialog_summary: bool = False,
) -> SubscriptionDraftResult:
    if not isinstance(payload, Mapping):
        raise RuntimeError("direct path response JSON root must be an object")
    route = str(payload.get("route") or "").strip()
    if not route:
        route = "draft_for_manager" if _direct_default_manager_enabled() else "bot_answer_self_for_pilot"
    if route == "bot_answer_self":
        route = "bot_answer_self_for_pilot"
    # Internal metadata is assembled locally below. The model cannot provide
    # service traces or gates through its free-form metadata object.
    metadata: dict[str, Any] = {}
    risk_level = str(payload.get("risk_level") or "low").strip()
    raw_is_p0 = payload.get("is_p0")
    is_p0_present = "is_p0" in payload
    is_p0_valid = is_p0_present and isinstance(raw_is_p0, bool)
    raw_p0_kind = payload.get("p0_kind") or payload.get("p0_code") or payload.get("risk_code")
    # Model-provided nested metadata is untrusted; only the physical top-level field defines this contract.
    metadata["direct_path_model_p0"] = {
        "is_p0": raw_is_p0 if is_p0_valid else False,
        "is_p0_present": is_p0_present,
        "is_p0_valid": is_p0_valid,
        "risk_level": risk_level,
        "p0_kind": _direct_path_model_p0_kind(raw_p0_kind),
        "p0_kind_raw": " ".join(str(raw_p0_kind or "").split())[:120],
        "model_reason": " ".join(str(payload.get("model_reason") or payload.get("p0_reason") or "").split())[:240],
    }
    model_intent_meta = _direct_path_model_intent_meta_from_payload(payload)
    if model_intent_meta:
        metadata["direct_path_model_intent"] = model_intent_meta
    if include_answerability_self:
        metadata["answerability_self"] = _direct_path_answerability_self_from_payload(payload)
    semantic_frame = _direct_path_semantic_frame_from_payload(payload, source="inline") if include_semantic_frame_shadow else {}
    if semantic_frame:
        metadata["semantic_frame"] = semantic_frame
        # Backward-compatible alias for one release: TZ154/text hygiene and
        # older simulators may still read the historical shadow key.
        metadata["semantic_frame_shadow"] = semantic_frame
    if include_dialog_summary:
        dialog_summary = " ".join(str(payload.get("dialog_summary") or "").split())[:500]
        if dialog_summary:
            metadata["dialog_summary_candidate"] = dialog_summary
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
        # Verified flags are produced only by local guards after normalization.
        safety_flags=(),
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
