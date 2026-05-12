from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.transcript_quality_llm_review import DECISIONS, read_jsonl, write_csv, write_jsonl
from mango_mvp.quality.transcript_quality_review_validator import is_high_risk_task


@dataclass(frozen=True)
class ConsensusConfig:
    tasks_jsonl: Path
    mini_reviews_jsonl: Path
    mini_validation_root: Path
    out_root: Path
    advanced_reviews_jsonl: Path | None = None
    claude_reviews_jsonl: Path | None = None
    auto_confidence_threshold: float = 0.9


def build_transcript_quality_consensus(config: ConsensusConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = {_task_id(task): task for task in read_jsonl(config.tasks_jsonl) if _task_id(task)}
    mini = {_clean(row.get("task_id")): row for row in read_jsonl(config.mini_reviews_jsonl) if _clean(row.get("task_id"))}
    advanced = {}
    if config.advanced_reviews_jsonl and config.advanced_reviews_jsonl.exists():
        advanced = {_clean(row.get("task_id")): row for row in read_jsonl(config.advanced_reviews_jsonl) if _clean(row.get("task_id"))}
    claude = {}
    if config.claude_reviews_jsonl and config.claude_reviews_jsonl.exists():
        claude = {_clean(row.get("task_id")): row for row in read_jsonl(config.claude_reviews_jsonl) if _clean(row.get("task_id"))}
    validation = _read_validation(config.mini_validation_root / "validated_reviews.csv")

    consensus_rows: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        row = consensus_one(
            task,
            mini.get(task_id, {}),
            advanced.get(task_id, {}),
            claude.get(task_id, {}),
            validation.get(task_id, {}),
            auto_confidence_threshold=config.auto_confidence_threshold,
        )
        consensus_rows.append(row)

    auto_apply = [row for row in consensus_rows if row["consensus_route"] == "auto_apply_force_non_conversation"]
    keep = [row for row in consensus_rows if row["consensus_route"] == "keep_current_analysis"]
    reanalyze = [row for row in consensus_rows if row["consensus_route"] == "reanalyze_required"]
    claude_required = [row for row in consensus_rows if row["consensus_route"] == "claude_audit_required"]
    human = [row for row in consensus_rows if row["consensus_route"] == "human_review_required"]
    blocked = [row for row in consensus_rows if row["consensus_route"] == "blocked_or_invalid"]

    outputs = {
        "consensus_jsonl": out_root / "consensus.jsonl",
        "consensus_csv": out_root / "consensus.csv",
        "auto_apply_csv": out_root / "auto_apply_force_non_conversation.csv",
        "keep_current_analysis_csv": out_root / "keep_current_analysis.csv",
        "reanalyze_required_csv": out_root / "reanalyze_required.csv",
        "claude_audit_required_csv": out_root / "claude_audit_required.csv",
        "human_review_required_csv": out_root / "human_review_required.csv",
        "blocked_or_invalid_csv": out_root / "blocked_or_invalid.csv",
        "summary_json": out_root / "summary.json",
    }
    write_jsonl(outputs["consensus_jsonl"], consensus_rows)
    write_csv(outputs["consensus_csv"], consensus_rows)
    write_csv(outputs["auto_apply_csv"], auto_apply)
    write_csv(outputs["keep_current_analysis_csv"], keep)
    write_csv(outputs["reanalyze_required_csv"], reanalyze)
    write_csv(outputs["claude_audit_required_csv"], claude_required)
    write_csv(outputs["human_review_required_csv"], human)
    write_csv(outputs["blocked_or_invalid_csv"], blocked)
    xlsx_path, xlsx_error = _write_xlsx_if_available(out_root / "transcript_quality_consensus.xlsx", consensus_rows, auto_apply, keep, reanalyze, claude_required, human, blocked)
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": len(tasks),
        "mini_reviews": len(mini),
        "advanced_reviews": len(advanced),
        "claude_reviews": len(claude),
        "counts": {
            "by_route": dict(Counter(row["consensus_route"] for row in consensus_rows).most_common()),
            "by_final_decision": dict(Counter(row["final_decision"] for row in consensus_rows).most_common()),
            "by_review_bucket": dict(Counter(row["review_bucket"] for row in consensus_rows).most_common()),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    if xlsx_path is not None:
        summary["outputs"]["xlsx"] = str(xlsx_path)
    if xlsx_error:
        summary["xlsx_error"] = xlsx_error
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def consensus_one(
    task: dict[str, Any],
    mini: dict[str, Any],
    advanced: dict[str, Any],
    claude: dict[str, Any],
    validation: dict[str, Any],
    *,
    auto_confidence_threshold: float,
) -> dict[str, Any]:
    task_id = _task_id(task)
    guardrail = _dict(task.get("guardrail"))
    call = _dict(task.get("call"))
    high_risk = is_high_risk_task(task)
    mini_decision = _clean(mini.get("decision"))
    advanced_decision = _clean(advanced.get("decision"))
    claude_decision = _clean(claude.get("claude_decision") or claude.get("decision"))
    final_source = ""
    final_decision = ""
    route = "blocked_or_invalid"
    reason = "no_valid_review"

    if claude_decision:
        final_source = "claude"
        final_decision = claude_decision
        confidence = _clamp_float(claude.get("claude_confidence") or claude.get("confidence"), 0.0, 1.0, 0.0)
        safe = _is_true(claude.get("safe_to_auto_apply"))
        if final_decision == "force_non_conversation" and safe and confidence >= auto_confidence_threshold:
            route = "auto_apply_force_non_conversation"
            reason = "claude_high_confidence_force"
        elif final_decision == "keep_current_analysis" and confidence >= 0.85:
            route = "keep_current_analysis"
            reason = "claude_high_confidence_keep"
        elif final_decision == "reanalyze_required":
            route = "reanalyze_required"
            reason = "claude_reanalyze_required"
        else:
            route = "human_review_required"
            reason = "claude_low_confidence_or_high_risk"
    elif advanced_decision:
        final_source = "advanced"
        final_decision = advanced_decision
        confidence = _clamp_float(advanced.get("confidence"), 0.0, 1.0, 0.0)
        safe = _is_true(advanced.get("safe_to_auto_apply"))
        if high_risk:
            route = "claude_audit_required"
            reason = "advanced_high_risk_needs_claude"
        elif final_decision == "force_non_conversation" and safe and confidence >= auto_confidence_threshold:
            route = "auto_apply_force_non_conversation"
            reason = "advanced_high_confidence_force"
        elif final_decision == "keep_current_analysis" and confidence >= 0.85:
            route = "keep_current_analysis"
            reason = "advanced_high_confidence_keep"
        elif final_decision == "reanalyze_required":
            route = "reanalyze_required"
            reason = "advanced_reanalyze_required"
        else:
            route = "claude_audit_required"
            reason = "advanced_low_confidence_or_not_safe"
    elif mini_decision:
        final_source = "mini"
        final_decision = mini_decision
        validator_route = _clean(validation.get("validator_route"))
        if validator_route == "auto_apply_force_non_conversation_candidate":
            route = "auto_apply_force_non_conversation"
            reason = "mini_validator_low_risk_auto_apply_candidate"
        elif validator_route == "keep_current_analysis_candidate":
            route = "keep_current_analysis"
            reason = "mini_validator_keep_candidate"
        elif validator_route == "reanalyze_required":
            route = "reanalyze_required"
            reason = "mini_validator_reanalyze"
        elif validator_route == "escalate_to_advanced_model":
            route = "claude_audit_required"
            reason = "advanced_review_missing"
        elif validator_route == "human_or_claude_required":
            route = "claude_audit_required"
            reason = "validator_requires_claude"
        else:
            route = "blocked_or_invalid"
            reason = "validator_blocked_or_invalid"

    return {
        "task_id": task_id,
        "consensus_route": route,
        "consensus_reason": reason,
        "final_source": final_source,
        "final_decision": final_decision,
        "high_risk": high_risk,
        "review_bucket": guardrail.get("review_bucket", ""),
        "current_call_type": guardrail.get("current_call_type", ""),
        "current_contentful": guardrail.get("current_contentful", ""),
        "mini_decision": mini_decision,
        "mini_confidence": mini.get("confidence", ""),
        "advanced_decision": advanced_decision,
        "advanced_confidence": advanced.get("confidence", ""),
        "claude_decision": claude_decision,
        "claude_confidence": claude.get("claude_confidence") or claude.get("confidence", ""),
        "call_id": call.get("id", ""),
        "source_filename": call.get("source_filename", ""),
        "started_at": call.get("started_at", ""),
        "manager_name": call.get("manager_name", ""),
        "phone": call.get("phone", ""),
    }


def _read_validation(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {_clean(row.get("task_id")): row for row in csv.DictReader(fh) if _clean(row.get("task_id"))}


def _write_xlsx_if_available(
    path: Path,
    consensus: list[dict[str, Any]],
    auto_apply: list[dict[str, Any]],
    keep: list[dict[str, Any]],
    reanalyze: list[dict[str, Any]],
    claude_required: list[dict[str, Any]],
    human: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
) -> tuple[Path | None, str | None]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        return None, str(exc)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(consensus).to_excel(writer, index=False, sheet_name="Consensus")
        pd.DataFrame(auto_apply).to_excel(writer, index=False, sheet_name="Auto apply")
        pd.DataFrame(keep).to_excel(writer, index=False, sheet_name="Keep")
        pd.DataFrame(reanalyze).to_excel(writer, index=False, sheet_name="Reanalyze")
        pd.DataFrame(claude_required).to_excel(writer, index=False, sheet_name="Claude required")
        pd.DataFrame(human).to_excel(writer, index=False, sheet_name="Human")
        pd.DataFrame(blocked).to_excel(writer, index=False, sheet_name="Blocked")
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
    parser = argparse.ArgumentParser(description="Build consensus queues from transcript-quality review stages.")
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--mini-reviews-jsonl", type=Path, required=True)
    parser.add_argument("--mini-validation-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--advanced-reviews-jsonl", type=Path, default=None)
    parser.add_argument("--claude-reviews-jsonl", type=Path, default=None)
    parser.add_argument("--auto-confidence-threshold", type=float, default=0.9)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ConsensusConfig:
    return ConsensusConfig(
        tasks_jsonl=args.tasks_jsonl,
        mini_reviews_jsonl=args.mini_reviews_jsonl,
        mini_validation_root=args.mini_validation_root,
        out_root=args.out_root,
        advanced_reviews_jsonl=args.advanced_reviews_jsonl,
        claude_reviews_jsonl=args.claude_reviews_jsonl,
        auto_confidence_threshold=args.auto_confidence_threshold,
    )
