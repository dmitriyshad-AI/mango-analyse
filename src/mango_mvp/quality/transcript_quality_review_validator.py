from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.transcript_quality_llm_review import DECISIONS, CALL_TYPES, read_jsonl, write_csv, write_jsonl


CONTENTFUL_CALL_TYPES = {"sales_call", "service_call", "technical_call", "existing_client_progress"}


@dataclass(frozen=True)
class ReviewValidatorConfig:
    tasks_jsonl: Path
    reviews_jsonl: Path
    out_root: Path
    review_tier: str = "mini"
    auto_confidence_threshold: float = 0.9
    keep_confidence_threshold: float = 0.85
    advanced_confidence_threshold: float = 0.88


def validate_transcript_quality_reviews(config: ReviewValidatorConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(config.tasks_jsonl)
    reviews = read_jsonl(config.reviews_jsonl) if config.reviews_jsonl.exists() else []
    tasks_by_id = {_task_id(task): task for task in tasks if _task_id(task)}
    reviews_by_id = {_clean(row.get("task_id")): row for row in reviews if _clean(row.get("task_id"))}

    validated: list[dict[str, Any]] = []
    escalation_tasks: list[dict[str, Any]] = []
    escalation_rows: list[dict[str, Any]] = []
    auto_apply_rows: list[dict[str, Any]] = []
    keep_rows: list[dict[str, Any]] = []
    reanalyze_rows: list[dict[str, Any]] = []
    human_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []

    for task_id, task in tasks_by_id.items():
        review = reviews_by_id.get(task_id)
        row = validate_one(task, review, config)
        validated.append(row)
        route = row["validator_route"]
        if route == "escalate_to_advanced_model":
            escalation_tasks.append(task)
            escalation_rows.append(row)
        elif route == "auto_apply_force_non_conversation_candidate":
            auto_apply_rows.append(row)
        elif route == "keep_current_analysis_candidate":
            keep_rows.append(row)
        elif route == "reanalyze_required":
            reanalyze_rows.append(row)
        elif route == "human_or_claude_required":
            human_rows.append(row)
        elif route == "invalid_review":
            invalid_rows.append(row)

    extra_review_ids = sorted(set(reviews_by_id) - set(tasks_by_id))
    for task_id in extra_review_ids:
        invalid_rows.append({"task_id": task_id, "validator_route": "invalid_review", "validator_reason": "review_without_matching_task"})

    outputs = {
        "validated_reviews_jsonl": out_root / "validated_reviews.jsonl",
        "validated_reviews_csv": out_root / "validated_reviews.csv",
        "escalation_tasks_jsonl": out_root / "escalation_tasks.jsonl",
        "escalation_queue_csv": out_root / "escalation_queue.csv",
        "auto_apply_candidates_csv": out_root / "auto_apply_candidates.csv",
        "keep_current_analysis_csv": out_root / "keep_current_analysis.csv",
        "reanalyze_required_csv": out_root / "reanalyze_required.csv",
        "human_or_claude_required_csv": out_root / "human_or_claude_required.csv",
        "invalid_reviews_csv": out_root / "invalid_reviews.csv",
        "summary_json": out_root / "summary.json",
    }
    write_jsonl(outputs["validated_reviews_jsonl"], validated)
    write_csv(outputs["validated_reviews_csv"], validated)
    write_jsonl(outputs["escalation_tasks_jsonl"], escalation_tasks)
    write_csv(outputs["escalation_queue_csv"], escalation_rows)
    write_csv(outputs["auto_apply_candidates_csv"], auto_apply_rows)
    write_csv(outputs["keep_current_analysis_csv"], keep_rows)
    write_csv(outputs["reanalyze_required_csv"], reanalyze_rows)
    write_csv(outputs["human_or_claude_required_csv"], human_rows)
    write_csv(outputs["invalid_reviews_csv"], invalid_rows)
    xlsx_path, xlsx_error = _write_xlsx_if_available(out_root / "transcript_quality_review_validation.xlsx", validated, escalation_rows, auto_apply_rows, keep_rows, reanalyze_rows, human_rows, invalid_rows)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks_jsonl": str(config.tasks_jsonl.resolve()),
        "reviews_jsonl": str(config.reviews_jsonl.resolve()),
        "review_tier": config.review_tier,
        "totals": {
            "tasks": len(tasks_by_id),
            "reviews": len(reviews_by_id),
            "validated": len(validated),
            "extra_review_ids": len(extra_review_ids),
            "escalation_tasks": len(escalation_tasks),
            "auto_apply_candidates": len(auto_apply_rows),
            "keep_current_analysis": len(keep_rows),
            "reanalyze_required": len(reanalyze_rows),
            "human_or_claude_required": len(human_rows),
            "invalid_reviews": len(invalid_rows),
        },
        "counts": {
            "by_route": dict(Counter(row["validator_route"] for row in validated).most_common()),
            "by_decision": dict(Counter(row["decision"] for row in validated).most_common()),
            "by_review_bucket": dict(Counter(row["review_bucket"] for row in validated).most_common()),
            "by_current_call_type": dict(Counter(row["current_call_type"] for row in validated).most_common()),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    if xlsx_path is not None:
        summary["outputs"]["xlsx"] = str(xlsx_path)
    if xlsx_error:
        summary["xlsx_error"] = xlsx_error
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_one(task: dict[str, Any], review: dict[str, Any] | None, config: ReviewValidatorConfig) -> dict[str, Any]:
    guardrail = _dict(task.get("guardrail"))
    call = _dict(task.get("call"))
    task_id = _task_id(task)
    review = review or {}
    decision = _clean(review.get("decision"))
    confidence = _clamp_float(review.get("confidence"), 0.0, 1.0, 0.0)
    evidence = _split_pipe_or_list(review.get("evidence"))
    recommended_call_type = _clean(review.get("recommended_call_type")) or "unknown"
    safe_to_auto_apply = _is_true(review.get("safe_to_auto_apply"))
    current_call_type = _clean(guardrail.get("current_call_type")) or "unknown"
    current_contentful = _is_true(guardrail.get("current_contentful")) or current_call_type in CONTENTFUL_CALL_TYPES
    review_bucket = _clean(guardrail.get("review_bucket"))
    high_risk = is_high_risk_task(task)
    invalid_reasons: list[str] = []

    if not review:
        invalid_reasons.append("missing_review")
    if decision not in DECISIONS:
        invalid_reasons.append("invalid_decision")
    if recommended_call_type not in CALL_TYPES:
        invalid_reasons.append("invalid_recommended_call_type")
    if not evidence:
        invalid_reasons.append("missing_evidence")
    if not _clean(review.get("reason")):
        invalid_reasons.append("missing_reason")

    if invalid_reasons:
        route = "invalid_review"
        route_reason = "|".join(invalid_reasons)
    else:
        route, route_reason = route_valid_review(
            decision=decision,
            confidence=confidence,
            safe_to_auto_apply=safe_to_auto_apply,
            high_risk=high_risk,
            current_contentful=current_contentful,
            review_bucket=review_bucket,
            review_tier=config.review_tier,
            auto_confidence_threshold=config.auto_confidence_threshold,
            keep_confidence_threshold=config.keep_confidence_threshold,
            advanced_confidence_threshold=config.advanced_confidence_threshold,
        )

    return {
        "task_id": task_id,
        "validator_route": route,
        "validator_reason": route_reason,
        "review_tier": config.review_tier,
        "high_risk": high_risk,
        "current_contentful": current_contentful,
        "review_bucket": review_bucket,
        "current_call_type": current_call_type,
        "decision": decision or "missing",
        "confidence": confidence,
        "safe_to_auto_apply": safe_to_auto_apply,
        "recommended_call_type": recommended_call_type,
        "reason": _clean(review.get("reason")),
        "evidence": " | ".join(evidence),
        "call_id": call.get("id", ""),
        "source_filename": call.get("source_filename", ""),
        "started_at": call.get("started_at", ""),
        "duration_sec": call.get("duration_sec", ""),
        "manager_name": call.get("manager_name", ""),
        "phone": call.get("phone", ""),
        "model": _clean(review.get("model")),
        "provider": _clean(review.get("provider")),
    }


def route_valid_review(
    *,
    decision: str,
    confidence: float,
    safe_to_auto_apply: bool,
    high_risk: bool,
    current_contentful: bool,
    review_bucket: str,
    review_tier: str,
    auto_confidence_threshold: float,
    keep_confidence_threshold: float,
    advanced_confidence_threshold: float,
) -> tuple[str, str]:
    tier = _clean(review_tier) or "mini"
    if tier == "mini":
        if (
            high_risk
            or current_contentful
            or review_bucket.startswith("llm_review_contentful")
            or "sales" in review_bucket
            or "borderline" in review_bucket
        ):
            return "escalate_to_advanced_model", "mini_high_risk_or_contentful"
        if decision == "force_non_conversation" and safe_to_auto_apply and confidence >= auto_confidence_threshold:
            return "auto_apply_force_non_conversation_candidate", "mini_low_risk_high_confidence_force"
        if decision == "keep_current_analysis" and confidence >= keep_confidence_threshold:
            return "keep_current_analysis_candidate", "mini_low_risk_high_confidence_keep"
        if decision == "reanalyze_required":
            return "escalate_to_advanced_model", "mini_reanalyze_requires_advanced_confirmation"
        return "escalate_to_advanced_model", "mini_low_confidence_or_not_auto_safe"

    if tier == "advanced":
        if decision == "force_non_conversation" and safe_to_auto_apply and confidence >= advanced_confidence_threshold and not high_risk:
            return "auto_apply_force_non_conversation_candidate", "advanced_low_risk_high_confidence_force"
        if decision == "keep_current_analysis" and confidence >= advanced_confidence_threshold:
            return "keep_current_analysis_candidate", "advanced_high_confidence_keep"
        if decision == "reanalyze_required" and confidence >= 0.75:
            return "reanalyze_required", "advanced_reanalyze_required"
        return "human_or_claude_required", "advanced_low_confidence_or_high_risk"

    return "human_or_claude_required", "unknown_review_tier"


def is_high_risk_task(task: dict[str, Any]) -> bool:
    guardrail = _dict(task.get("guardrail"))
    bucket = _clean(guardrail.get("review_bucket"))
    call_type = _clean(guardrail.get("current_call_type"))
    if bucket.startswith("human_review") or "sales" in bucket or "borderline" in bucket:
        return True
    if call_type in {"sales_call", "service_call", "technical_call", "existing_client_progress"}:
        return True
    if _is_true(guardrail.get("current_contentful")):
        return True
    return False


def _write_xlsx_if_available(
    path: Path,
    validated: list[dict[str, Any]],
    escalation: list[dict[str, Any]],
    auto_apply: list[dict[str, Any]],
    keep: list[dict[str, Any]],
    reanalyze: list[dict[str, Any]],
    human: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
) -> tuple[Path | None, str | None]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        return None, str(exc)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(validated).to_excel(writer, index=False, sheet_name="Validated")
        pd.DataFrame(escalation).to_excel(writer, index=False, sheet_name="Escalation")
        pd.DataFrame(auto_apply).to_excel(writer, index=False, sheet_name="Auto apply")
        pd.DataFrame(keep).to_excel(writer, index=False, sheet_name="Keep")
        pd.DataFrame(reanalyze).to_excel(writer, index=False, sheet_name="Reanalyze")
        pd.DataFrame(human).to_excel(writer, index=False, sheet_name="Human Claude")
        pd.DataFrame(invalid).to_excel(writer, index=False, sheet_name="Invalid")
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column_cells in sheet.columns:
                max_len = max(len(str(cell.value or "")) for cell in column_cells[:200])
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_len + 2, 10), 64)
    return path, None


def _task_id(task: dict[str, Any]) -> str:
    return _clean(task.get("task_id") or task.get("id"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _split_pipe_or_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    return [_clean(part) for part in str(value or "").split("|") if _clean(part)]


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "да"}


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate transcript-quality LLM reviews and build safety queues.")
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--reviews-jsonl", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--review-tier", default="mini", choices=["mini", "advanced"])
    parser.add_argument("--auto-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--keep-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--advanced-confidence-threshold", type=float, default=0.88)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ReviewValidatorConfig:
    return ReviewValidatorConfig(
        tasks_jsonl=args.tasks_jsonl,
        reviews_jsonl=args.reviews_jsonl,
        out_root=args.out_root,
        review_tier=args.review_tier,
        auto_confidence_threshold=args.auto_confidence_threshold,
        keep_confidence_threshold=args.keep_confidence_threshold,
        advanced_confidence_threshold=args.advanced_confidence_threshold,
    )
