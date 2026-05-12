#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.non_conversation import detect_non_conversation_signals


DEFAULT_PACKAGE_DIR = Path(
    "stable_runtime/transcript_quality_pipeline_v2_risky_3298_m4_20260509_0445/claude_audit_package_full_2548"
)


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    items = read_items(package_dir / "claude_audit_items.jsonl")
    decisions = read_jsonl(package_dir / "claude_decisions.jsonl")
    modes = ["transcript_only", "with_current_analysis"] if args.mode == "both" else [args.mode]

    summaries: dict[str, Any] = {}
    for mode in modes:
        summary, rows = validate_mode(items, decisions, mode=mode)
        summaries[mode] = summary
        write_csv(out_root / f"{mode}_rows.csv", rows)
        write_csv(out_root / f"{mode}_missed_force.csv", [row for row in rows if row["error_type"] == "missed_force"])
        write_csv(out_root / f"{mode}_false_force.csv", [row for row in rows if row["error_type"] == "false_force"])

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_dir": str(package_dir),
        "items": len(items),
        "decisions": len(decisions),
        "mode": args.mode,
        "summaries": summaries,
        "outputs": {
            "summary_json": str(out_root / "summary.json"),
            "report_markdown": str(out_root / "NON_CONVERSATION_CLAUDE_FIXTURE_VALIDATION.md"),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "NON_CONVERSATION_CLAUDE_FIXTURE_VALIDATION.md").write_text(
        markdown_report(payload),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def validate_mode(
    items: dict[str, dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if mode not in {"transcript_only", "with_current_analysis"}:
        raise ValueError("mode must be transcript_only, with_current_analysis, or both")

    label_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    by_expected_decision: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    expected_force = 0
    predicted_force = 0
    true_force = 0
    false_force = 0
    missed_force = 0

    for decision in decisions:
        task_id = str(decision.get("task_id") or "")
        task = items.get(task_id)
        if not task:
            continue
        task_payload = task.get("task") or {}
        call = task_payload.get("call") or {}
        current_analysis = task_payload.get("current_analysis") or {}
        quality_flags = current_analysis.get("quality_flags") or {}
        guardrail = task_payload.get("guardrail") or {}
        call_type = str(quality_flags.get("call_type") or guardrail.get("current_call_type") or "")

        kwargs: dict[str, Any] = {
            "transcript_text": str(task_payload.get("transcript_text") or ""),
            "call_type": call_type,
            "duration_sec": call.get("duration_sec"),
        }
        if mode == "with_current_analysis":
            kwargs.update(
                {
                    "history_summary": str(current_analysis.get("history_summary") or ""),
                    "next_step": str(current_analysis.get("next_step") or ""),
                    "products": current_analysis.get("products") or [],
                    "subjects": current_analysis.get("subjects") or [],
                    "objections": current_analysis.get("objections") or [],
                }
            )

        result = detect_non_conversation_signals(**kwargs)
        expected_decision = str(decision.get("claude_decision") or "")
        expected_is_force = expected_decision == "force_non_conversation"
        predicted_is_force = bool(result.should_force_non_conversation)
        error_type = ""
        if expected_is_force:
            expected_force += 1
            if predicted_is_force:
                true_force += 1
            else:
                missed_force += 1
                error_type = "missed_force"
        elif predicted_is_force:
            false_force += 1
            error_type = "false_force"

        if predicted_is_force:
            predicted_force += 1
        label_counts[result.label] += 1
        decision_counts[expected_decision] += 1
        by_expected_decision[expected_decision][result.label] += 1

        rows.append(
            {
                "task_id": task_id,
                "source_filename": str(call.get("source_filename") or ""),
                "duration_sec": call.get("duration_sec"),
                "expected_decision": expected_decision,
                "expected_safe_to_auto_apply": decision.get("safe_to_auto_apply"),
                "expected_recommended_call_type": decision.get("recommended_call_type"),
                "predicted_label": result.label,
                "predicted_should_force": result.should_force_non_conversation,
                "predicted_requires_manual_review": result.requires_manual_review,
                "predicted_score": result.score,
                "predicted_reason_codes": "|".join(result.reason_codes),
                "error_type": error_type,
                "transcript_excerpt": str(task_payload.get("transcript_text") or "")[:1200].replace("\n", " "),
                "claude_reason": str(decision.get("claude_reason") or ""),
            }
        )

    precision = true_force / predicted_force if predicted_force else 0.0
    recall = true_force / expected_force if expected_force else 0.0
    summary = {
        "mode": mode,
        "rows": len(rows),
        "expected_force": expected_force,
        "predicted_force": predicted_force,
        "true_force": true_force,
        "missed_force": missed_force,
        "false_force": false_force,
        "force_precision": round(precision, 6),
        "force_recall": round(recall, 6),
        "label_counts": dict(label_counts),
        "expected_decision_counts": dict(decision_counts),
        "by_expected_decision": {key: dict(value) for key, value in by_expected_decision.items()},
    }
    return summary, rows


def read_items(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row.get("task_id") or ""): row for row in rows}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(payload)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Non-Conversation Claude Fixture Validation",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Package: `{payload['package_dir']}`",
        f"- Items: `{payload['items']}`",
        f"- Decisions: `{payload['decisions']}`",
        "",
    ]
    for mode, summary in payload["summaries"].items():
        lines.extend(
            [
                f"## {mode}",
                "",
                f"- Expected force: `{summary['expected_force']}`",
                f"- Predicted force: `{summary['predicted_force']}`",
                f"- True force: `{summary['true_force']}`",
                f"- Missed force: `{summary['missed_force']}`",
                f"- False force: `{summary['false_force']}`",
                f"- Force precision: `{summary['force_precision']}`",
                f"- Force recall: `{summary['force_recall']}`",
                "",
                "### Label Counts",
                "",
            ]
        )
        for key, value in summary["label_counts"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate deterministic non-conversation detector on Claude fixtures.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["transcript_only", "with_current_analysis", "both"], default="both")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
