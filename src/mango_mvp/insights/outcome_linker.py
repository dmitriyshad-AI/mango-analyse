from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.insights.phone_identity import normalize_phone, phones_from_text


PAID_RE = re.compile(
    r"оплатил[аи]?|оплачено|оплачен[аоы]?|опл\.|плат[её]жк[аи]|чек|сч[её]т оплат|оплат[ауы] внес",
    re.I,
)
ENROLLED_RE = re.compile(
    r"записал[аои]?с[ья]|записан[аы]?|записали\b|в группе|занимает[сц]я|учит[сц]я|"
    r"обучает[сц]я|ходит|посещает|прид[её]т на занят",
    re.I,
)
ACTIVE_RE = re.compile(r"продолж(ит|ат|ает|ают)|все устраивает|всё устраивает|довол[ье]н|следующ[ийе].*год", re.I)
REFUSAL_RE = re.compile(
    r"отказ|не актуаль|неактуаль|не интерес|неинтерес|не подходит|не будут|не пойд[её]т|"
    r"не продолж|передумал|не получается|не сможет",
    re.I,
)
PENDING_RE = re.compile(r"дума|подума|реша|совет|обсуд|перезвон|созвон|пока не зна|жд[её]м|жду", re.I)
PAYMENT_PENDING_RE = re.compile(r"оплатит|оплатят|жд[её]м оплат|жду оплат|ссылк[ау] на оплат|сч[её]т|долг|задолж", re.I)
EXISTING_CLIENT_RE = re.compile(r"действующ|ежемесячн|уже обуч|занимает[сц]я|учит[сц]я|оплат[ауы] обуч", re.I)

AMO_OPPORTUNITY_VERDICTS = {
    "reopen_recommended",
    "closed_too_early",
    "follow_up_needed",
    "alternative_offer_needed",
}


@dataclass
class SignalSummary:
    label: str
    confidence_tier: str
    confidence_score: float
    reasons: list[str] = field(default_factory=list)
    latest_signal: str = ""
    latest_signal_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeLinkerConfig:
    project_root: Path
    readiness_root: Path
    out_root: Path
    tallanto_contacts: Path | None
    amo_deal_analysis_root: Path | None
    pilot_limit: int = 500
    outcome_model_mode: str = "off"


def build_outcome_linkage_report(config: OutcomeLinkerConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    readiness_root = config.readiness_root.resolve()
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    chain_rows = _read_csv(readiness_root / "client_chains.csv")
    call_rows = _read_csv(readiness_root / "calls_terminal_analyzed.csv")
    calls_by_phone = _group_calls(call_rows)
    outcome_model_mode = normalize_outcome_model_mode(config.outcome_model_mode)
    tallanto_index = (
        load_tallanto_outcome_index(config.tallanto_contacts, outcome_model_mode=outcome_model_mode)
        if config.tallanto_contacts
        else {}
    )
    amo_index = load_amo_outcome_index(config.amo_deal_analysis_root) if config.amo_deal_analysis_root else {}

    linked_rows = [
        link_chain_outcome(row, calls_by_phone.get(str(row.get("phone") or ""), []), tallanto_index, amo_index)
        for row in chain_rows
    ]
    if outcome_model_mode != "off":
        attach_outcome_model_fields(linked_rows)
    linked_rows.sort(key=lambda row: (-int(row["extraction_priority_score"]), str(row["phone"])))
    pilot_rows = build_outcome_pilot_sample(linked_rows, config.pilot_limit)
    summary = _build_summary(config, linked_rows, pilot_rows, tallanto_index, amo_index)

    outputs = _write_outputs(out_root, summary, linked_rows, pilot_rows)
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def normalize_outcome_model_mode(value: Any) -> str:
    mode = _clean(value).casefold()
    if mode in {"off", "shadow", "primary"}:
        return mode
    return "off"


def load_tallanto_outcome_index(path: Path | None, *, outcome_model_mode: str = "off") -> dict[str, SignalSummary]:
    if path is None or not path.exists():
        return {}
    mode = normalize_outcome_model_mode(outcome_model_mode)
    rows_by_phone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_csv(path):
        phones: list[str] = []
        for key in ("phone_parent", "phone_extra", "phones_joined"):
            phones.extend(phones_from_text(row.get(key)))
        for phone in _unique(phones):
            rows_by_phone[phone].append(row)
    return {phone: classify_tallanto_rows(rows, outcome_model_mode=mode) for phone, rows in rows_by_phone.items()}


def load_amo_outcome_index(root: Path | None) -> dict[str, SignalSummary]:
    if root is None or not root.exists():
        return {}
    rows_by_phone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.glob("*/all_results.csv")):
        for row in _read_csv(path):
            phone = normalize_phone(row.get("Телефон"))
            if phone:
                row = dict(row)
                row["_source_file"] = str(path)
                rows_by_phone[phone].append(row)
    return {phone: classify_amo_rows(rows) for phone, rows in rows_by_phone.items()}


def classify_tallanto_rows(rows: list[dict[str, Any]], *, outcome_model_mode: str = "off") -> SignalSummary:
    legacy = _classify_tallanto_rows_legacy(rows)
    mode = normalize_outcome_model_mode(outcome_model_mode)
    if mode == "off":
        return legacy

    semantic = _classify_tallanto_rows_negation_aware(rows)
    comparison = _outcome_model_comparison(legacy, semantic, mode=mode)
    if mode == "shadow":
        legacy.metadata["outcome_model_shadow"] = comparison
        return legacy

    if _primary_outcome_flip_allowed(legacy.label, semantic.label):
        semantic.metadata["outcome_model_primary"] = {
            **comparison,
            "primary_applied": True,
            "primary_policy": "only_won_paid_or_active_to_known_student_or_lead",
        }
        semantic.metadata["legacy_outcome"] = {
            "label": legacy.label,
            "confidence_tier": legacy.confidence_tier,
            "confidence_score": legacy.confidence_score,
            "reasons": list(legacy.reasons),
        }
        return semantic

    legacy.metadata["outcome_model_primary"] = {
        **comparison,
        "primary_applied": False,
        "primary_blocked_reason": "flip_not_allowlisted",
        "primary_policy": "only_won_paid_or_active_to_known_student_or_lead",
    }
    return legacy


def _classify_tallanto_rows_legacy(rows: list[dict[str, Any]]) -> SignalSummary:
    histories = [_clean(row.get("history_raw")) for row in rows if _clean(row.get("history_raw"))]
    full_history = "\n".join(histories)
    term_counts = {
        "paid": _bool_count(histories, PAID_RE),
        "enrolled": _bool_count(histories, ENROLLED_RE),
        "active": _bool_count(histories, ACTIVE_RE),
        "refusal": _bool_count(histories, REFUSAL_RE),
        "pending": _bool_count(histories, PENDING_RE),
        "payment_pending": _bool_count(histories, PAYMENT_PENDING_RE),
    }
    latest_label, latest_text = _latest_history_signal(full_history)
    has_positive = term_counts["paid"] or term_counts["enrolled"] or term_counts["active"]
    has_refusal = term_counts["refusal"] > 0
    has_pending = term_counts["pending"] or term_counts["payment_pending"]
    student_types = {_clean(row.get("student_type")) for row in rows if _clean(row.get("student_type"))}
    non_listener_student = any(value and value != "Слушатель" for value in student_types)

    reasons: list[str] = []
    if term_counts["paid"]:
        reasons.append("tallanto_history_has_paid_terms")
    if term_counts["enrolled"] or term_counts["active"]:
        reasons.append("tallanto_history_has_learning_terms")
    if has_refusal:
        reasons.append("tallanto_history_has_refusal_terms")
    if has_pending:
        reasons.append("tallanto_history_has_pending_terms")
    if non_listener_student:
        reasons.append("tallanto_student_type_is_grade")

    if has_positive and latest_label == "refusal":
        label = "churn_or_refused_after_activity"
        tier = "strong"
        score = 0.82
    elif has_positive:
        label = "won_paid_or_active"
        tier = "strong"
        score = 0.86 if term_counts["paid"] else 0.78
    elif has_refusal:
        label = "lost_or_refused"
        tier = "strong"
        score = 0.76
    elif term_counts["payment_pending"]:
        label = "payment_pending"
        tier = "proxy"
        score = 0.62
    elif has_pending:
        label = "in_progress_or_undecided"
        tier = "proxy"
        score = 0.52
    elif non_listener_student:
        label = "known_student_or_lead"
        tier = "proxy"
        score = 0.45
    else:
        label = "tallanto_match_without_outcome"
        tier = "proxy"
        score = 0.35

    return SignalSummary(
        label=label,
        confidence_tier=tier,
        confidence_score=score,
        reasons=reasons,
        latest_signal=latest_label,
        latest_signal_text=latest_text,
        metadata={
            "contact_count": len(rows),
            "tallanto_ids": _join_sorted(row.get("tallanto_id") for row in rows),
            "student_types": _join_sorted(student_types),
            "branches": _join_sorted(row.get("branch") for row in rows),
            "responsible": _join_sorted(row.get("responsible") for row in rows),
            "term_counts": term_counts,
        },
    )


def _classify_tallanto_rows_negation_aware(rows: list[dict[str, Any]]) -> SignalSummary:
    histories = [_clean(row.get("history_raw")) for row in rows if _clean(row.get("history_raw"))]
    full_history = "\n".join(histories)
    term_counts = {
        "paid": _affirmed_count(histories, PAID_RE),
        "enrolled": _affirmed_count(histories, ENROLLED_RE),
        "active": _affirmed_count(histories, ACTIVE_RE),
        "refusal": _affirmed_count(histories, REFUSAL_RE),
        "pending": _bool_count(histories, PENDING_RE),
        "payment_pending": _bool_count(histories, PAYMENT_PENDING_RE),
    }
    latest_label, latest_text = _latest_history_signal_negation_aware(full_history)
    has_positive = term_counts["paid"] or term_counts["enrolled"] or term_counts["active"]
    has_refusal = term_counts["refusal"] > 0
    has_pending = term_counts["pending"] or term_counts["payment_pending"]
    student_types = {_clean(row.get("student_type")) for row in rows if _clean(row.get("student_type"))}
    non_listener_student = any(value and value != "Слушатель" for value in student_types)

    reasons: list[str] = []
    if term_counts["paid"]:
        reasons.append("tallanto_history_has_affirmed_paid_terms")
    if term_counts["enrolled"] or term_counts["active"]:
        reasons.append("tallanto_history_has_affirmed_learning_terms")
    if has_refusal:
        reasons.append("tallanto_history_has_affirmed_refusal_terms")
    if has_pending:
        reasons.append("tallanto_history_has_pending_terms")
    if non_listener_student:
        reasons.append("tallanto_student_type_is_grade")

    if has_positive and latest_label == "refusal":
        label = "churn_or_refused_after_activity"
        tier = "strong"
        score = 0.82
    elif has_positive:
        label = "won_paid_or_active"
        tier = "strong"
        score = 0.86 if term_counts["paid"] else 0.78
    elif has_refusal:
        label = "lost_or_refused"
        tier = "strong"
        score = 0.76
    elif term_counts["payment_pending"]:
        label = "payment_pending"
        tier = "proxy"
        score = 0.62
    elif has_pending:
        label = "in_progress_or_undecided"
        tier = "proxy"
        score = 0.52
    elif non_listener_student:
        label = "known_student_or_lead"
        tier = "proxy"
        score = 0.45
    else:
        label = "tallanto_match_without_outcome"
        tier = "proxy"
        score = 0.35

    return SignalSummary(
        label=label,
        confidence_tier=tier,
        confidence_score=score,
        reasons=reasons,
        latest_signal=latest_label,
        latest_signal_text=latest_text,
        metadata={
            "contact_count": len(rows),
            "tallanto_ids": _join_sorted(row.get("tallanto_id") for row in rows),
            "student_types": _join_sorted(student_types),
            "branches": _join_sorted(row.get("branch") for row in rows),
            "responsible": _join_sorted(row.get("responsible") for row in rows),
            "term_counts": term_counts,
        },
    )


def _outcome_model_comparison(legacy: SignalSummary, semantic: SignalSummary, *, mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "legacy_label": legacy.label,
        "semantic_label": semantic.label,
        "legacy_confidence_score": legacy.confidence_score,
        "semantic_confidence_score": semantic.confidence_score,
        "label_changed": legacy.label != semantic.label,
        "semantic_reasons": list(semantic.reasons),
        "semantic_term_counts": semantic.metadata.get("term_counts", {}),
        "primary_allowed": _primary_outcome_flip_allowed(legacy.label, semantic.label),
    }


def _primary_outcome_flip_allowed(legacy_label: str, semantic_label: str) -> bool:
    return legacy_label == "won_paid_or_active" and semantic_label == "known_student_or_lead"


def classify_amo_rows(rows: list[dict[str, Any]]) -> SignalSummary:
    verdicts = Counter(_norm(row.get("AI-вердикт")) for row in rows if _norm(row.get("AI-вердикт")))
    risks = Counter(_norm(row.get("AI-risk")) for row in rows if _norm(row.get("AI-risk")))
    statuses = Counter(_clean(row.get("Статус")) for row in rows if _clean(row.get("Статус")))
    combined_text = "\n".join(
        _clean(row.get(key))
        for row in rows
        for key in ("Основание", "Краткая история", "Следующий шаг")
        if _clean(row.get(key))
    )
    lead_ids = {_clean(row.get("ID сделки amoCRM")) for row in rows if _clean(row.get("ID сделки amoCRM"))}
    contact_ids = {_clean(row.get("ID контакта amoCRM")) for row in rows if _clean(row.get("ID контакта amoCRM"))}
    opportunity_verdicts = [value for value in verdicts if value in AMO_OPPORTUNITY_VERDICTS]

    if opportunity_verdicts:
        label = "reopen_or_follow_up_opportunity"
        tier = "strong"
        score = 0.82
        reasons = [f"amo_verdict:{value}" for value in opportunity_verdicts]
    elif verdicts.get("closed_valid") and EXISTING_CLIENT_RE.search(combined_text):
        label = "existing_client_service_not_new_sale"
        tier = "strong"
        score = 0.78
        reasons = ["amo_closed_valid_existing_client_context"]
    elif verdicts.get("closed_valid"):
        label = "closed_lost_valid"
        tier = "strong"
        score = 0.72
        reasons = ["amo_closed_valid"]
    elif verdicts.get("manual_review"):
        label = "manual_review"
        tier = "proxy"
        score = 0.42
        reasons = ["amo_manual_review"]
    else:
        label = "amo_link_without_actionable_outcome"
        tier = "proxy"
        score = 0.35
        reasons = ["amo_link"]

    return SignalSummary(
        label=label,
        confidence_tier=tier,
        confidence_score=score,
        reasons=reasons,
        metadata={
            "row_count": len(rows),
            "lead_ids": _join_sorted(lead_ids),
            "contact_ids": _join_sorted(contact_ids),
            "statuses": _counter_join(statuses, 8),
            "verdicts": _counter_join(verdicts, 8),
            "risks": _counter_join(risks, 8),
        },
    )


def link_chain_outcome(
    chain_row: dict[str, Any],
    call_rows: list[dict[str, Any]],
    tallanto_index: dict[str, SignalSummary],
    amo_index: dict[str, SignalSummary],
) -> dict[str, Any]:
    phone = str(chain_row.get("phone") or "")
    tallanto = tallanto_index.get(phone)
    amo = amo_index.get(phone)
    call_signal = classify_call_only_signal(chain_row, call_rows)
    final = choose_final_outcome(tallanto, amo, call_signal)
    action = choose_sales_action(final, tallanto, amo, call_signal)
    use_case = choose_extraction_use_case(final, action)
    priority = extraction_priority_score(chain_row, final, action, use_case)

    row = dict(chain_row)
    row.update(
        {
            "final_outcome_label": final.label,
            "outcome_confidence_tier": final.confidence_tier,
            "outcome_confidence_score": final.confidence_score,
            "outcome_sources": " | ".join(final.metadata.get("sources", [])),
            "outcome_reasons": " | ".join(final.reasons),
            "sales_action_label": action.label,
            "sales_action_confidence": action.confidence_score,
            "sales_action_reasons": " | ".join(action.reasons),
            "extraction_use_case": use_case,
            "extraction_priority_score": priority,
            "tallanto_outcome_label": tallanto.label if tallanto else "",
            "tallanto_confidence_tier": tallanto.confidence_tier if tallanto else "",
            "tallanto_confidence_score": tallanto.confidence_score if tallanto else "",
            "tallanto_latest_signal": tallanto.latest_signal if tallanto else "",
            "tallanto_latest_signal_text": tallanto.latest_signal_text if tallanto else "",
            "tallanto_outcome_reasons": " | ".join(tallanto.reasons) if tallanto else "",
            "amo_outcome_label": amo.label if amo else "",
            "amo_confidence_tier": amo.confidence_tier if amo else "",
            "amo_confidence_score": amo.confidence_score if amo else "",
            "amo_outcome_reasons": " | ".join(amo.reasons) if amo else "",
            "call_only_outcome_label": call_signal.label,
            "call_only_reasons": " | ".join(call_signal.reasons),
            "tallanto_ids_linked": tallanto.metadata.get("tallanto_ids", "") if tallanto else "",
            "tallanto_student_types_linked": tallanto.metadata.get("student_types", "") if tallanto else "",
            "tallanto_term_counts_json": json.dumps(tallanto.metadata.get("term_counts", {}), ensure_ascii=False) if tallanto else "{}",
            "amo_lead_ids_linked": amo.metadata.get("lead_ids", "") if amo else "",
            "amo_contact_ids_linked": amo.metadata.get("contact_ids", "") if amo else "",
            "amo_verdicts_linked": amo.metadata.get("verdicts", "") if amo else "",
            "amo_risks_linked": amo.metadata.get("risks", "") if amo else "",
        }
    )
    if tallanto and tallanto.metadata.get("outcome_model_shadow"):
        row["_tallanto_shadow_json"] = json.dumps(tallanto.metadata["outcome_model_shadow"], ensure_ascii=False, sort_keys=True)
    if tallanto and tallanto.metadata.get("outcome_model_primary"):
        row["_tallanto_shadow_json"] = json.dumps(tallanto.metadata["outcome_model_primary"], ensure_ascii=False, sort_keys=True)
    return row


def classify_call_only_signal(chain_row: dict[str, Any], call_rows: list[dict[str, Any]]) -> SignalSummary:
    contentful_count = _as_int(chain_row.get("contentful_call_count"))
    sales_count = _as_int(chain_row.get("sales_call_count"))
    existing_count = _as_int(chain_row.get("existing_client_progress_count"))
    service_count = _as_int(chain_row.get("service_call_count"))
    next_steps = _as_int(chain_row.get("next_step_count"))
    objections = _clean(chain_row.get("objections_top"))
    priorities = Counter(_norm(row.get("lead_priority")) for row in call_rows if _norm(row.get("lead_priority")))
    warm_or_hot = priorities.get("hot", 0) + priorities.get("warm", 0)
    reasons: list[str] = []
    if sales_count:
        reasons.append("has_sales_calls")
    if warm_or_hot:
        reasons.append("has_warm_or_hot_call_priority")
    if next_steps:
        reasons.append("has_next_steps")
    if objections:
        reasons.append("has_objections")

    if existing_count or service_count:
        return SignalSummary("service_or_existing_context", "proxy", 0.45, reasons or ["service_or_existing_calls"])
    if sales_count and (warm_or_hot or next_steps):
        return SignalSummary("open_sales_potential", "proxy", 0.48, reasons)
    if contentful_count:
        return SignalSummary("contentful_unknown_outcome", "proxy", 0.3, reasons or ["contentful_calls"])
    return SignalSummary("no_contentful_outcome_signal", "unknown", 0.0, ["no_contentful_calls"])


def choose_final_outcome(tallanto: SignalSummary | None, amo: SignalSummary | None, call_signal: SignalSummary) -> SignalSummary:
    sources: list[str] = []
    reasons: list[str] = []
    candidates: list[tuple[int, SignalSummary, str]] = []
    if tallanto:
        sources.append("Tallanto")
        candidates.append((_outcome_rank(tallanto.label), tallanto, "Tallanto"))
    if amo:
        sources.append("AMO")
        candidates.append((_outcome_rank(amo.label), amo, "AMO"))
    candidates.append((_outcome_rank(call_signal.label), call_signal, "Calls"))

    candidates.sort(key=lambda item: (item[0], item[1].confidence_score), reverse=True)
    _, selected, selected_source = candidates[0]
    reasons.extend(f"{selected_source}:{reason}" for reason in selected.reasons)

    if tallanto and amo and _conflicting_positive_negative(tallanto.label, amo.label):
        return SignalSummary(
            "mixed_outcome_manual_review",
            "strong",
            max(tallanto.confidence_score, amo.confidence_score),
            [f"Tallanto:{tallanto.label}", f"AMO:{amo.label}", "source_conflict_requires_review"],
            metadata={"sources": sources},
        )

    return SignalSummary(
        selected.label,
        selected.confidence_tier,
        selected.confidence_score,
        reasons,
        latest_signal=selected.latest_signal,
        latest_signal_text=selected.latest_signal_text,
        metadata={"sources": sources or ["Calls"]},
    )


def choose_sales_action(
    final: SignalSummary,
    tallanto: SignalSummary | None,
    amo: SignalSummary | None,
    call_signal: SignalSummary,
) -> SignalSummary:
    if amo and amo.label == "reopen_or_follow_up_opportunity":
        return SignalSummary("sales_reactivation_candidate", "strong", amo.confidence_score, [f"AMO:{reason}" for reason in amo.reasons])
    if final.label == "payment_pending":
        return SignalSummary("payment_follow_up_candidate", "proxy", final.confidence_score, ["Tallanto payment pending signal"])
    if final.label in {"in_progress_or_undecided", "open_sales_potential"}:
        return SignalSummary("soft_follow_up_candidate", "proxy", max(final.confidence_score, call_signal.confidence_score), final.reasons)
    if final.label in {"won_paid_or_active", "existing_client_service_not_new_sale"}:
        return SignalSummary("do_not_reopen_use_for_success_patterns", "strong", final.confidence_score, ["learn_success_pattern"])
    if final.label in {"lost_or_refused", "closed_lost_valid"}:
        return SignalSummary("do_not_reopen_use_for_loss_patterns", "strong", final.confidence_score, ["learn_loss_pattern"])
    if final.label in {"churn_or_refused_after_activity", "mixed_outcome_manual_review"}:
        return SignalSummary("manual_review_or_retention_learning", "strong", final.confidence_score, ["mixed_or_churn_context"])
    return SignalSummary("no_sales_action_without_more_context", "unknown", 0.0, ["unknown_outcome"])


def choose_extraction_use_case(final: SignalSummary, action: SignalSummary) -> str:
    if action.label == "sales_reactivation_candidate":
        return "reactivation_revenue"
    if final.label in {"won_paid_or_active", "existing_client_service_not_new_sale"}:
        return "winner_pattern_for_playbook"
    if final.label in {"lost_or_refused", "closed_lost_valid"}:
        return "loss_pattern_for_objection_playbook"
    if final.label in {"churn_or_refused_after_activity", "mixed_outcome_manual_review"}:
        return "retention_or_manual_review_learning"
    if final.label in {"payment_pending", "in_progress_or_undecided", "open_sales_potential"}:
        return "open_pipeline_learning"
    if final.label == "service_or_existing_context":
        return "support_qa_bot_context"
    return "low_confidence_backlog"


def extraction_priority_score(chain_row: dict[str, Any], final: SignalSummary, action: SignalSummary, use_case: str) -> int:
    score = _as_int(chain_row.get("utility_score"))
    score += {
        "reactivation_revenue": 90,
        "winner_pattern_for_playbook": 75,
        "loss_pattern_for_objection_playbook": 65,
        "retention_or_manual_review_learning": 55,
        "open_pipeline_learning": 50,
        "support_qa_bot_context": 30,
        "low_confidence_backlog": 0,
    }.get(use_case, 0)
    score += int(final.confidence_score * 30)
    score += int(action.confidence_score * 20)
    score += min(_as_int(chain_row.get("contentful_call_count")), 8) * 4
    score += min(_as_int(chain_row.get("sales_call_count")), 5) * 8
    return score


def build_outcome_pilot_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    quotas = {
        "reactivation_revenue": 120,
        "winner_pattern_for_playbook": 120,
        "loss_pattern_for_objection_playbook": 100,
        "retention_or_manual_review_learning": 60,
        "open_pipeline_learning": 60,
        "support_qa_bot_context": 30,
        "low_confidence_backlog": 10,
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("extraction_use_case") or "")].append(row)
    for use_case, quota in quotas.items():
        for row in sorted(by_case.get(use_case, []), key=lambda item: (-int(item["extraction_priority_score"]), str(item["phone"])))[:quota]:
            key = str(row.get("client_key") or row.get("phone"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(dict(row, sample_reason=use_case))
            if len(selected) >= limit:
                return selected
    for row in rows:
        key = str(row.get("client_key") or row.get("phone"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(dict(row, sample_reason="top_priority_fill"))
        if len(selected) >= limit:
            break
    return selected


def _outcome_rank(label: str) -> int:
    return {
        "reopen_or_follow_up_opportunity": 100,
        "won_paid_or_active": 95,
        "churn_or_refused_after_activity": 90,
        "existing_client_service_not_new_sale": 85,
        "lost_or_refused": 80,
        "closed_lost_valid": 78,
        "payment_pending": 70,
        "in_progress_or_undecided": 60,
        "open_sales_potential": 58,
        "known_student_or_lead": 45,
        "service_or_existing_context": 40,
        "manual_review": 35,
        "contentful_unknown_outcome": 20,
    }.get(label, 0)


def _conflicting_positive_negative(left: str, right: str) -> bool:
    positive = {"won_paid_or_active", "existing_client_service_not_new_sale"}
    negative = {"lost_or_refused", "closed_lost_valid"}
    return (left in positive and right in negative) or (right in positive and left in negative)


def _build_summary(
    config: OutcomeLinkerConfig,
    linked_rows: list[dict[str, Any]],
    pilot_rows: list[dict[str, Any]],
    tallanto_index: dict[str, SignalSummary],
    amo_index: dict[str, SignalSummary],
) -> dict[str, Any]:
    final_counts = Counter(str(row["final_outcome_label"]) for row in linked_rows)
    tier_counts = Counter(str(row["outcome_confidence_tier"]) for row in linked_rows)
    action_counts = Counter(str(row["sales_action_label"]) for row in linked_rows)
    use_case_counts = Counter(str(row["extraction_use_case"]) for row in linked_rows)
    pilot_use_cases = Counter(str(row["extraction_use_case"]) for row in pilot_rows)
    year_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in linked_rows:
        for year in re.split(r"\s*\|\s*", str(row.get("years") or "unknown")):
            if not year:
                continue
            year_counts[year]["chains"] += 1
            year_counts[year][f"outcome:{row['final_outcome_label']}"] += 1
            year_counts[year][f"confidence:{row['outcome_confidence_tier']}"] += 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness_root": str(config.readiness_root.resolve()),
        "totals": {
            "client_chains": len(linked_rows),
            "chains_with_tallanto_outcome_signal": sum(1 for row in linked_rows if row["tallanto_outcome_label"]),
            "chains_with_amo_outcome_signal": sum(1 for row in linked_rows if row["amo_outcome_label"]),
            "chains_with_strong_outcome": tier_counts.get("strong", 0),
            "chains_with_proxy_outcome": tier_counts.get("proxy", 0),
            "chains_with_unknown_outcome": tier_counts.get("unknown", 0),
            "reactivation_revenue_candidates": use_case_counts.get("reactivation_revenue", 0),
            "winner_pattern_candidates": use_case_counts.get("winner_pattern_for_playbook", 0),
            "loss_pattern_candidates": use_case_counts.get("loss_pattern_for_objection_playbook", 0),
            "pilot_sample_rows": len(pilot_rows),
            "tallanto_index_phones": len(tallanto_index),
            "amo_index_phones": len(amo_index),
        },
        "final_outcome_counts": dict(final_counts.most_common()),
        "confidence_tier_counts": dict(tier_counts.most_common()),
        "sales_action_counts": dict(action_counts.most_common()),
        "extraction_use_case_counts": dict(use_case_counts.most_common()),
        "pilot_use_case_counts": dict(pilot_use_cases.most_common()),
        "by_year": {year: dict(counter) for year, counter in sorted(year_counts.items())},
        "notes": [
            "Outcome labels are conservative. Tallanto snapshot has context/history but no canonical paid/lost table.",
            "AMO outcome is based on available deal-analysis exports; AMO was operationally meaningful mainly in 2026.",
            "Use strong/proxy/unknown confidence tiers when selecting data for LLM extraction and future sales bot training.",
        ],
    }
    outcome_model_mode = normalize_outcome_model_mode(config.outcome_model_mode)
    if outcome_model_mode != "off":
        shadow_counts = Counter(str(row.get("tallanto_model_shadow_changed") or "") for row in linked_rows)
        flips = Counter(
            f"{row.get('tallanto_model_legacy_label', '')}->{row.get('tallanto_model_semantic_label', '')}"
            for row in linked_rows
            if str(row.get("tallanto_model_shadow_changed") or "") == "Да"
        )
        summary["outcome_model_mode"] = outcome_model_mode
        summary["outcome_model_shadow_counts"] = dict(shadow_counts.most_common())
        summary["outcome_model_label_flips"] = dict(flips.most_common())
    return summary


def _write_outputs(out_root: Path, summary: dict[str, Any], linked_rows: list[dict[str, Any]], pilot_rows: list[dict[str, Any]]) -> dict[str, Path]:
    paths = {
        "client_outcomes_csv": out_root / "client_outcomes.csv",
        "pilot_outcome_sample_csv": out_root / "pilot_outcome_sample.csv",
        "summary_json": out_root / "summary.json",
    }
    _write_csv(paths["client_outcomes_csv"], linked_rows)
    _write_csv(paths["pilot_outcome_sample_csv"], pilot_rows)
    xlsx_path = out_root / "outcome_linkage_report.xlsx"
    try:
        _write_xlsx(xlsx_path, summary, linked_rows, pilot_rows)
        paths["xlsx"] = xlsx_path
    except Exception as exc:  # noqa: BLE001
        (out_root / "xlsx_error.txt").write_text(str(exc), encoding="utf-8")
    return paths


def _write_xlsx(path: Path, summary: dict[str, Any], linked_rows: list[dict[str, Any]], pilot_rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    summary_rows: list[dict[str, Any]] = []
    for key, value in summary.get("totals", {}).items():
        summary_rows.append({"metric": key, "value": value})
    for note in summary.get("notes", []):
        summary_rows.append({"metric": "note", "value": note})
    count_rows = [
        {"section": section, "label": label, "count": count}
        for section, counts in (
            ("final_outcome", summary.get("final_outcome_counts", {})),
            ("confidence_tier", summary.get("confidence_tier_counts", {})),
            ("sales_action", summary.get("sales_action_counts", {})),
            ("extraction_use_case", summary.get("extraction_use_case_counts", {})),
            ("pilot_use_case", summary.get("pilot_use_case_counts", {})),
        )
        for label, count in counts.items()
    ]
    by_year_rows = []
    for year, values in summary.get("by_year", {}).items():
        row = {"year": year}
        row.update(values)
        by_year_rows.append(row)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(count_rows).to_excel(writer, sheet_name="Counts", index=False)
        pd.DataFrame(by_year_rows).to_excel(writer, sheet_name="By Year", index=False)
        pd.DataFrame(pilot_rows).to_excel(writer, sheet_name="Pilot Outcome Sample", index=False)
        pd.DataFrame(linked_rows[:5000]).to_excel(writer, sheet_name="Client Outcomes", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column_cells in sheet.columns:
                max_len = 0
                col = column_cells[0].column_letter
                for cell in column_cells[:200]:
                    max_len = max(max_len, len(str(cell.value or "")))
                sheet.column_dimensions[col].width = min(max(max_len + 2, 10), 60)


def _group_calls(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        phone = str(row.get("phone") or "")
        if phone:
            result[phone].append(row)
    return result


def _latest_history_signal(history: str) -> tuple[str, str]:
    latest_label = ""
    latest_text = ""
    chunks = [line.strip() for line in re.split(r"[\n\r]+", history) if line.strip()]
    if not chunks and history.strip():
        chunks = [history.strip()]
    for chunk in chunks:
        label = _signal_label_for_text(chunk)
        if label:
            latest_label = label
            latest_text = chunk[:500]
    return latest_label, latest_text


def _signal_label_for_text(text: str) -> str:
    if REFUSAL_RE.search(text):
        return "refusal"
    if PAID_RE.search(text):
        return "paid"
    if ENROLLED_RE.search(text) or ACTIVE_RE.search(text):
        return "active_learning"
    if PAYMENT_PENDING_RE.search(text):
        return "payment_pending"
    if PENDING_RE.search(text):
        return "pending"
    return ""


def attach_outcome_model_fields(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        shadow = _parse_json_dict(row.get("_tallanto_shadow_json"))
        row["tallanto_model_legacy_label"] = _clean(shadow.get("legacy_label"))
        row["tallanto_model_semantic_label"] = _clean(shadow.get("semantic_label"))
        row["tallanto_model_shadow_changed"] = "Да" if shadow.get("label_changed") else ("Нет" if shadow else "")
        row["tallanto_model_primary_allowed"] = "Да" if shadow.get("primary_allowed") else ("Нет" if shadow else "")
        row["tallanto_model_semantic_reasons"] = " | ".join(shadow.get("semantic_reasons") or [])
        row.pop("_tallanto_shadow_json", None)


def _latest_history_signal_negation_aware(history: str) -> tuple[str, str]:
    latest_label = ""
    latest_text = ""
    chunks = [line.strip() for line in re.split(r"[\n\r]+", history) if line.strip()]
    if not chunks and history.strip():
        chunks = [history.strip()]
    for chunk in chunks:
        label = _signal_label_for_text_negation_aware(chunk)
        if label:
            latest_label = label
            latest_text = chunk[:500]
    return latest_label, latest_text


def _signal_label_for_text_negation_aware(text: str) -> str:
    if _has_affirmed_match(text, REFUSAL_RE):
        return "refusal"
    if _has_affirmed_match(text, PAID_RE):
        return "paid"
    if _has_affirmed_match(text, ENROLLED_RE) or _has_affirmed_match(text, ACTIVE_RE):
        return "active_learning"
    if PAYMENT_PENDING_RE.search(text):
        return "payment_pending"
    if PENDING_RE.search(text):
        return "pending"
    return ""


def _bool_count(values: Iterable[str], pattern: re.Pattern[str]) -> int:
    return sum(1 for value in values if pattern.search(value))


def _affirmed_count(values: Iterable[str], pattern: re.Pattern[str]) -> int:
    return sum(1 for value in values if _has_affirmed_match(value, pattern))


def _has_affirmed_match(value: str, pattern: re.Pattern[str]) -> bool:
    for match in pattern.finditer(value):
        if not _match_is_negated(value, match):
            return True
    return False


def _match_is_negated(value: str, match: re.Match[str]) -> bool:
    matched = match.group(0).casefold().replace("ё", "е").strip()
    # Some refusal patterns intentionally start with "не": "не актуально", "не подходит".
    if matched.startswith("не ") or matched.startswith("неакту"):
        return False
    prefix = value[max(0, match.start() - 48) : match.start()].casefold().replace("ё", "е")
    prefix = re.sub(r"\s+", " ", prefix)
    prefix = re.split(r"[,.;:!?\-–—]", prefix)[-1]
    return bool(re.search(r"(?:^|[\s,.;:!?])(?:не|нет|без|ни|никак|еще не|ещё не)\s+(?:\S+\s+){0,3}$", prefix))


def _parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = _clean(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _norm(value: Any) -> str:
    return _clean(value).lower()


def _as_int(value: Any) -> int:
    try:
        return int(float(str(value or 0).strip()))
    except (TypeError, ValueError):
        return 0


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _join_sorted(values: Iterable[Any]) -> str:
    return " | ".join(sorted(_clean(value) for value in values if _clean(value)))


def _counter_join(counter: Counter[str], limit: int) -> str:
    return " | ".join(f"{key}: {value}" for key, value in counter.most_common(limit))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link client chains to conservative sales outcomes for insight extraction.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--readiness-root", default="stable_runtime/insight_readiness_report_20260507")
    parser.add_argument("--out-root", default="stable_runtime/outcome_linkage_report_20260507")
    parser.add_argument("--tallanto-contacts", default="stable_runtime/tallanto_snapshot_20260331/tallanto_contacts_normalized.csv")
    parser.add_argument("--amo-deal-analysis-root", default="stable_runtime/amocrm_runtime/deal_analysis")
    parser.add_argument("--pilot-limit", type=int, default=500)
    parser.add_argument(
        "--outcome-model-mode",
        choices=("off", "shadow", "primary"),
        default="off",
        help="Offline-only outcome model mode. off preserves the legacy report.",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> OutcomeLinkerConfig:
    project_root = Path(args.project_root).expanduser().resolve()
    return OutcomeLinkerConfig(
        project_root=project_root,
        readiness_root=(project_root / args.readiness_root).resolve(),
        out_root=(project_root / args.out_root).resolve(),
        tallanto_contacts=(project_root / args.tallanto_contacts).resolve() if args.tallanto_contacts else None,
        amo_deal_analysis_root=(project_root / args.amo_deal_analysis_root).resolve() if args.amo_deal_analysis_root else None,
        pilot_limit=int(args.pilot_limit),
        outcome_model_mode=normalize_outcome_model_mode(args.outcome_model_mode),
    )


__all__ = [
    "OutcomeLinkerConfig",
    "SignalSummary",
    "build_outcome_linkage_report",
    "build_outcome_pilot_sample",
    "choose_extraction_use_case",
    "choose_final_outcome",
    "choose_sales_action",
    "classify_amo_rows",
    "classify_call_only_signal",
    "classify_tallanto_rows",
    "config_from_args",
    "extraction_priority_score",
    "link_chain_outcome",
    "load_amo_outcome_index",
    "load_tallanto_outcome_index",
    "normalize_outcome_model_mode",
    "parse_args",
]
