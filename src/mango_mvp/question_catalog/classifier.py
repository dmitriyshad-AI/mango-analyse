from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from mango_mvp.question_catalog.parameters_registry import extract_parameters
from mango_mvp.question_catalog.theme_assigner_llm import (
    ThemeAssignmentResult,
    ThemeAssignerConfig,
    assign_theme_llm,
)


TAXONOMY_PATH = Path(__file__).with_name("themes_taxonomy.yaml")


@dataclass(frozen=True)
class ClassifiedQuestion:
    theme_id: str
    extracted_params: dict[str, str]
    confidence: float
    classification_method: str
    required_facts: tuple[str, ...]
    default_bot_permission: str
    theme_name: str
    business_block: str
    reasoning: str = ""
    llm_model: str = ""


@dataclass(frozen=True)
class QuestionClassifierConfig:
    llm_enabled: bool = False
    llm_confidence_threshold: float = 0.7
    llm_config: ThemeAssignerConfig = ThemeAssignerConfig()

    @classmethod
    def from_env(cls) -> "QuestionClassifierConfig":
        llm_config = ThemeAssignerConfig.from_env()
        return cls(
            llm_enabled=_bool_env("QUESTION_CATALOG_LLM_ENABLED", False),
            llm_confidence_threshold=llm_config.llm_confidence_threshold,
            llm_config=llm_config,
        )


@dataclass(frozen=True)
class ThemeDecision:
    theme_id: str
    classification_method: str
    confidence: float | None = None
    reasoning: str = ""
    llm_model: str = ""


@lru_cache(maxsize=1)
def load_taxonomy(path: str | Path | None = None) -> Mapping[str, Any]:
    taxonomy_path = Path(path) if path else TAXONOMY_PATH
    return yaml.safe_load(taxonomy_path.read_text(encoding="utf-8"))


def load_valid_theme_and_service_ids() -> set[str]:
    taxonomy = load_taxonomy()
    return {
        *(item["theme_id"] for item in taxonomy["themes"]),
        *(item["service_id"] for item in taxonomy["service_categories"]),
    }


def classify_question(
    raw_text: Any,
    *,
    source: str = "unknown",
    metadata: Mapping[str, Any] | None = None,
    fallback_signal: str | None = None,
    config: QuestionClassifierConfig | None = None,
    llm_assigner: Callable[..., ThemeAssignmentResult] | None = None,
) -> ClassifiedQuestion:
    text = str(raw_text or "")
    params = extract_parameters(text)
    resolved_config = config or QuestionClassifierConfig.from_env()
    decision = _assign_theme_decision(
        text,
        params,
        metadata={**dict(metadata or {}), "fallback_signal": fallback_signal or ""},
        config=resolved_config,
        llm_assigner=llm_assigner,
    )
    validate_against_taxonomy(decision.theme_id)
    confidence = decision.confidence
    if confidence is None:
        confidence = compute_confidence(
            text,
            params,
            theme_id=decision.theme_id,
            classification_method=decision.classification_method,
        )
    theme_meta = get_theme_metadata(decision.theme_id)
    return ClassifiedQuestion(
        theme_id=decision.theme_id,
        extracted_params=params,
        confidence=confidence,
        classification_method=decision.classification_method,
        required_facts=tuple(theme_meta.get("required_facts") or ()),
        default_bot_permission=str(theme_meta.get("default_bot_permission") or "draft_for_manager"),
        theme_name=str(theme_meta.get("theme_name") or theme_meta.get("service_name") or decision.theme_id),
        business_block=str(theme_meta.get("business_block") or "Служебные категории"),
        reasoning=decision.reasoning,
        llm_model=decision.llm_model,
    )


def assign_theme(
    raw_text: str,
    params: Mapping[str, str],
    *,
    config: QuestionClassifierConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
    llm_assigner: Callable[..., ThemeAssignmentResult] | None = None,
) -> tuple[str, str]:
    decision = _assign_theme_decision(
        raw_text,
        params,
        metadata=dict(metadata or {}),
        config=config or QuestionClassifierConfig.from_env(),
        llm_assigner=llm_assigner,
    )
    return decision.theme_id, decision.classification_method


def _assign_theme_decision(
    raw_text: str,
    params: Mapping[str, str],
    *,
    metadata: Mapping[str, Any],
    config: QuestionClassifierConfig,
    llm_assigner: Callable[..., ThemeAssignmentResult] | None,
) -> ThemeDecision:
    if config.llm_enabled and not _llm_bypassed(metadata):
        try:
            assigner = llm_assigner or assign_theme_llm
            llm_result = assigner(
                raw_text,
                params,
                config=config.llm_config,
            )
            validate_against_taxonomy(llm_result.theme_id)
            if llm_result.confidence >= config.llm_confidence_threshold:
                return ThemeDecision(
                    theme_id=llm_result.theme_id,
                    classification_method="llm_primary",
                    confidence=llm_result.confidence,
                    reasoning=llm_result.reasoning,
                    llm_model=llm_result.model,
                )
            rule_theme_id, _rule_method = _assign_theme_stub(raw_text, params, metadata=metadata)
            return ThemeDecision(
                theme_id=rule_theme_id,
                classification_method="llm_low_confidence_rule_fallback",
                reasoning=f"LLM confidence {llm_result.confidence:.3f} below threshold; {llm_result.reasoning}",
                llm_model=llm_result.model,
            )
        except Exception as exc:  # noqa: BLE001 - the rule fallback is the resilience boundary.
            rule_theme_id, _rule_method = _assign_theme_stub(raw_text, params, metadata=metadata)
            return ThemeDecision(
                theme_id=rule_theme_id,
                classification_method="llm_error_rule_fallback",
                reasoning=f"LLM unavailable or invalid: {type(exc).__name__}: {exc}",
            )

    theme_id, method = _assign_theme_stub(raw_text, params, metadata=metadata)
    return ThemeDecision(theme_id=theme_id, classification_method=method)


def _assign_theme_stub(raw_text: str, params: Mapping[str, str], *, metadata: Mapping[str, Any]) -> tuple[str, str]:
    from mango_mvp.question_catalog import normalization

    text = normalization.clean_text(raw_text)
    fallback_signal = str(metadata.get("fallback_signal") or "")
    if not text:
        return "service:S2_unclear", "rule_stub_unclear"
    lowered = text.casefold()
    if normalization.detect_noise_reason(text):
        return "service:S1_non_question", "rule_stub_service_noise"
    if _has_any(lowered, ("неактуально", "не актуально", "не интерес", "ошиблись", "ошибочный номер", "сами набер")):
        return "service:S3_out_of_scope", "rule_stub_service_out_of_scope"
    if not fallback_signal and not normalization.is_question_like(text) and not _has_any(
        lowered,
        ("сколько", "стоит", "стоим", "цен", "когда", "распис", "ссылк", "оплат", "где", "как", "можно", "пришл"),
    ):
        return "service:S2_unclear", "rule_stub_unclear"

    intent_key, intent_label, _fact_types = normalization._infer_intent(  # noqa: SLF001 - stage C fallback reuses legacy regexes.
        text,
        fallback_signal=fallback_signal,
    )
    if intent_key == "other" and not normalization.is_question_like(text):
        return "service:S2_unclear", "rule_stub_unclear"
    if intent_key in _INTENT_THEME_OVERRIDES:
        return _theme_for_intent(intent_key, text, params), "rule_stub_intent_override"

    by_legacy_parent = _legacy_parent_theme_index()
    theme_id = by_legacy_parent.get(intent_label)
    if theme_id:
        return theme_id, "rule_stub_legacy_parent"
    return "service:S2_unclear", "rule_stub_unclear"


def compute_confidence(
    raw_text: str,
    params: Mapping[str, str],
    *,
    theme_id: str,
    classification_method: str,
) -> float:
    if theme_id.startswith("service:S2"):
        return 0.35
    if classification_method == "rule_stub_intent_override":
        base = 0.78
    elif classification_method == "rule_stub_legacy_parent":
        base = 0.68
    else:
        base = 0.45
    filled_params = sum(1 for value in params.values() if value not in {"не_указано", "нейтральный", "низкая"})
    return min(0.95, base + min(filled_params, 3) * 0.04)


def validate_against_taxonomy(theme_id: str) -> None:
    if theme_id not in load_valid_theme_and_service_ids():
        raise ValueError(f"unknown question catalog theme_id: {theme_id}")


def get_theme_metadata(theme_id: str) -> Mapping[str, Any]:
    taxonomy = load_taxonomy()
    for item in taxonomy["themes"]:
        if item["theme_id"] == theme_id:
            return item
    for item in taxonomy["service_categories"]:
        if item["service_id"] == theme_id:
            return item
    raise ValueError(f"unknown question catalog theme_id: {theme_id}")


@lru_cache(maxsize=1)
def _legacy_parent_theme_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for item in load_taxonomy()["themes"]:
        for parent in item.get("legacy_parent_mapping", []):
            index.setdefault(str(parent), str(item["theme_id"]))
    for item in load_taxonomy()["service_categories"]:
        for parent in item.get("legacy_parent_mapping", []):
            index.setdefault(str(parent), str(item["service_id"]))
    return index


_INTENT_THEME_OVERRIDES = {
    "price",
    "payment_service",
    "discount",
    "installment",
    "tax_deduction",
    "cancellation_change",
    "documents_letter",
    "documents",
    "schedule",
    "lesson_occurrence",
    "format",
    "location",
    "program",
    "teacher",
    "lesson_materials",
    "quality_feedback",
    "service_feedback",
    "enrollment",
    "continuation_decision",
    "age_or_level_fit",
    "level_fit",
    "trial",
    "technical_access",
    "message_not_received",
    "camp_trip",
    "camp_living_conditions",
    "transport_logistics",
    "legal_partner",
    "no_interest",
    "not_customer_question",
    "not_enough_context",
    "incomplete_context",
    "status_followup",
    "general_next_step",
    "general_consultation",
}


def _theme_for_intent(intent_key: str, text: str, params: Mapping[str, str]) -> str:
    lowered = text.casefold()
    if intent_key == "price":
        return "theme:001_pricing"
    if intent_key == "payment_service":
        if _has_any(lowered, ("маткап", "материнск", "пенсионн")):
            return "theme:007_matkap_payment"
        if _has_any(lowered, ("возврат", "вернуть", "деньг", "средств")):
            return "theme:009_refund"
        if _has_any(lowered, ("рассроч", "частями", "долями", "кредит")):
            return "theme:006_installment"
        if _has_any(lowered, ("итого", "сумм", "руб", "чек", "квитанц", "счёт", "счет", "прошла", "подтвержд")):
            return "theme:003_payment_status"
        if _has_any(lowered, ("срок", "когда", "до какого числа")):
            return "theme:004_payment_schedule"
        return "theme:002_payment_method"
    if intent_key == "discount":
        return "theme:005_discounts"
    if intent_key == "installment":
        return "theme:006_installment"
    if intent_key == "tax_deduction":
        return "theme:008_tax_deduction"
    if intent_key == "cancellation_change":
        if _has_any(lowered, ("возврат", "вернуть", "деньг", "средств", "отказ")):
            return "theme:009_refund"
        return "theme:010_change_terms"
    if intent_key in {"documents_letter", "documents"}:
        if _has_any(lowered, ("налог", "вычет", "лиценз")):
            return "theme:008_tax_deduction"
        if _has_any(lowered, ("договор", "оферт")):
            return "theme:011_contract"
        if _has_any(lowered, ("ссылк", "письм", "доступ")):
            return "theme:025_missing_links_access"
        return "theme:012_certificates"
    if intent_key in {"schedule", "lesson_occurrence"}:
        return "theme:013_schedule"
    if intent_key == "format":
        return "theme:014_format"
    if intent_key == "location":
        return "theme:015_address"
    if intent_key == "program":
        return "theme:016_program"
    if intent_key == "teacher":
        return "theme:017_teacher_method"
    if intent_key == "lesson_materials":
        return "theme:018_materials_homework"
    if intent_key in {"quality_feedback", "service_feedback"}:
        if params.get("sentiment") == "позитивный":
            return "theme:019a_positive_feedback"
        if _has_any(lowered, ("прогресс", "успеваем", "посещ", "отметк")):
            return "theme:032_student_progress_inquiry"
        return "theme:019b_negative_feedback"
    if intent_key == "enrollment":
        return "theme:020_enrollment"
    if intent_key == "continuation_decision":
        return "theme:021_continuation"
    if intent_key in {"age_or_level_fit", "level_fit"}:
        return "theme:022_age_level_testing"
    if intent_key == "trial":
        return "theme:023_trial_class"
    if intent_key == "technical_access":
        if _has_any(lowered, ("ссылк", "письм", "запись")):
            return "theme:025_missing_links_access"
        return "theme:024_account_access"
    if intent_key == "message_not_received":
        return "theme:025_missing_links_access"
    if intent_key == "camp_trip":
        return "theme:026_camp_general"
    if intent_key == "camp_living_conditions":
        return "theme:027_camp_living_conditions"
    if intent_key == "transport_logistics":
        return "theme:028_transport_logistics"
    if intent_key == "legal_partner":
        if _has_any(lowered, ("партнер", "партнёр", "сотруднич", "предлож")):
            return "theme:030_partnership_b2b"
        return "theme:029_legal_question"
    if intent_key in {"not_customer_question"}:
        return "service:S1_non_question"
    if intent_key in {"not_enough_context", "incomplete_context"}:
        return "service:S2_unclear"
    if intent_key == "no_interest":
        return "service:S3_out_of_scope"
    if intent_key in {"status_followup", "general_next_step"}:
        return "service:S4_status_request"
    if intent_key == "general_consultation":
        return "service:S5_general_consultation"
    return "service:S2_unclear"


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _llm_bypassed(metadata: Mapping[str, Any]) -> bool:
    return str(metadata.get("llm_bypass") or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _bool_env(name: str, default: bool) -> bool:
    raw = getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
