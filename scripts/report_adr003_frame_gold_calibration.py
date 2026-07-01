#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_adr003_semantic_frame_eval import (
    _actual_p0_signal,
    _actual_route_handoff,
    _load_transcripts,
    _strict_bool,
    _turns,
)


SCHEMA_VERSION = "adr003_frame_gold_calibration_v1_2026_07_01"
GOLD_SCHEMA_VERSION = "adr003_frame_gold_labels_v1_2026_07_01"
FIELD_NAMES = ("must_handoff", "risk_class", "answerability", "requested_action")
CONFIDENCE_BUCKETS = (
    (0.0, 0.6, "0.00-0.59"),
    (0.6, 0.8, "0.60-0.79"),
    (0.8, 0.9, "0.80-0.89"),
    (0.9, 1.01, "0.90-1.00"),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ADR-003 SemanticFrame telemetry with manual gold labels.")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(transcripts=args.transcripts, gold=args.gold)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_frame_gold_calibration_report.json"
    md_path = args.out_dir / "adr003_frame_gold_calibration_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, transcripts: Path, gold: Path) -> dict[str, Any]:
    dialogs = _load_transcripts(transcripts)
    gold_rows = _load_gold(gold)
    turn_map = _build_turn_map(dialogs)
    rows: list[dict[str, Any]] = []
    missing_turns: list[dict[str, Any]] = []
    duplicate_gold = 0
    seen_keys: set[tuple[str, int]] = set()
    for item in gold_rows:
        key = (str(item.get("dialog_id") or ""), int(item.get("turn") or 0))
        if key in seen_keys:
            duplicate_gold += 1
            continue
        seen_keys.add(key)
        turn = turn_map.get(key)
        if not turn:
            missing_turns.append({"dialog_id": key[0], "turn": key[1]})
            continue
        rows.append(_compare_row(item, turn))

    summary = _summary(rows, missing_turns=missing_turns, duplicate_gold=duplicate_gold)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"transcripts": str(transcripts), "gold": str(gold)},
        "summary": summary,
        "rows": rows,
        "acceptance": _acceptance(summary),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    lines = [
        "# ADR-003 Frame Gold Calibration",
        "",
        f"- Acceptance: `{acceptance.get('status', 'unknown')}`",
        f"- Labeled rows: `{summary.get('labeled_rows', 0)}`",
        f"- Compared rows: `{summary.get('compared_rows', 0)}`",
        f"- Skipped rows: `{summary.get('skipped_rows', 0)}`",
        f"- Missing transcript rows: `{summary.get('missing_transcript_rows', 0)}`",
        f"- Must-handoff accuracy: `{summary.get('must_handoff_accuracy', 'n/a')}`",
        f"- Too cautious: `{summary.get('too_cautious', 0)}`",
        f"- Too confident: `{summary.get('too_confident', 0)}`",
        f"- Current over-handoff candidates: `{summary.get('current_over_handoff_candidates', 0)}`",
        f"- Safe self candidates: `{summary.get('safe_self_candidates', 0)}`",
        "",
        "## Per-field Accuracy",
        "",
    ]
    for field, metrics in sorted((summary.get("field_accuracy") or {}).items()):
        lines.append(f"- `{field}`: `{metrics.get('accuracy', 'n/a')}` ({metrics.get('correct', 0)}/{metrics.get('total', 0)})")
    lines.extend(["", "## Confidence Buckets", ""])
    for bucket, metrics in sorted((summary.get("confidence_buckets") or {}).items()):
        lines.append(
            f"- `{bucket}`: rows={metrics.get('rows', 0)}, "
            f"must_handoff_accuracy={metrics.get('must_handoff_accuracy', 'n/a')}, "
            f"too_cautious={metrics.get('too_cautious', 0)}, too_confident={metrics.get('too_confident', 0)}"
        )
    lines.extend(["", "## Blocking Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _load_gold(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, Mapping):
                rows.append(item)
    return rows


def _build_turn_map(dialogs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        for index, turn in enumerate(_turns(dialog), 1):
            turn_no = int(turn.get("turn") or index)
            result[(dialog_id, turn_no)] = turn
    return result


def _compare_row(gold: Mapping[str, Any], turn: Mapping[str, Any]) -> dict[str, Any]:
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    expected = gold.get("expected") if isinstance(gold.get("expected"), Mapping) else gold
    expected_must = _strict_bool(expected.get("must_handoff"))
    frame_must = _strict_bool(frame.get("must_handoff"))
    current_route_handoff = _actual_route_handoff(turn)
    current_p0_signal = _actual_p0_signal(turn)
    confidence = _float_or_none(frame.get("confidence"))
    field_results = {
        "must_handoff": _field_result(frame_must, expected_must),
        "risk_class": _field_result(_clean(frame.get("risk_class")), _clean(expected.get("risk_class"))),
        "answerability": _field_result(_clean(frame.get("answerability")), _clean(expected.get("answerability"))),
        "requested_action": _field_result(_clean(frame.get("requested_action")), _clean(expected.get("requested_action"))),
    }
    review_label = _clean(gold.get("review_label")) or _derived_review_label(
        frame_must=frame_must,
        expected_must=expected_must,
        field_results=field_results,
    )
    return {
        "dialog_id": str(gold.get("dialog_id") or ""),
        "turn": int(gold.get("turn") or 0),
        "brand": str(turn.get("brand") or gold.get("brand") or ""),
        "current_route": str(turn.get("bot_route") or ""),
        "current_route_handoff": current_route_handoff,
        "current_p0_signal": current_p0_signal,
        "frame": {
            "must_handoff": frame_must,
            "risk_class": _clean(frame.get("risk_class")),
            "answerability": _clean(frame.get("answerability")),
            "requested_action": _clean(frame.get("requested_action")),
            "confidence": confidence,
        },
        "expected": {
            "must_handoff": expected_must,
            "risk_class": _clean(expected.get("risk_class")),
            "answerability": _clean(expected.get("answerability")),
            "requested_action": _clean(expected.get("requested_action")),
        },
        "field_results": field_results,
        "review_label": review_label,
        "confidence_bucket": _confidence_bucket(confidence),
        "notes": _clean(gold.get("notes")),
    }


def _field_result(actual: Any, expected: Any) -> str:
    if expected in (None, ""):
        return "not_labeled"
    return "correct" if actual == expected else "wrong"


def _derived_review_label(*, frame_must: bool | None, expected_must: bool | None, field_results: Mapping[str, str]) -> str:
    if expected_must is None or frame_must is None:
        return "unclear"
    if frame_must and not expected_must:
        return "frame_too_cautious"
    if not frame_must and expected_must:
        return "frame_too_confident"
    if all(value in {"correct", "not_labeled"} for value in field_results.values()):
        return "frame_correct"
    return "frame_partial"


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    missing_turns: Sequence[Mapping[str, Any]],
    duplicate_gold: int,
) -> dict[str, Any]:
    compared_rows = [row for row in rows if row.get("expected", {}).get("must_handoff") is not None]
    skipped_rows = len(rows) - len(compared_rows)
    labels = Counter(str(row.get("review_label") or "") for row in rows)
    field_accuracy: dict[str, dict[str, Any]] = {}
    for field in FIELD_NAMES:
        total = 0
        correct = 0
        for row in rows:
            result = row.get("field_results", {}).get(field)
            if result == "not_labeled":
                continue
            total += 1
            if result == "correct":
                correct += 1
        field_accuracy[field] = {"correct": correct, "total": total, "accuracy": _ratio(correct, total)}
    too_cautious = sum(1 for row in compared_rows if row["frame"]["must_handoff"] is True and row["expected"]["must_handoff"] is False)
    too_confident = sum(1 for row in compared_rows if row["frame"]["must_handoff"] is False and row["expected"]["must_handoff"] is True)
    current_over_handoff = sum(
        1
        for row in compared_rows
        if row["current_route_handoff"] is True and row["expected"]["must_handoff"] is False
    )
    safe_self_candidates = sum(
        1
        for row in compared_rows
        if row["expected"]["must_handoff"] is False
        and row["expected"]["risk_class"] == "safe"
        and row["expected"]["answerability"] == "answer_self"
    )
    p0_misses = sum(
        1
        for row in compared_rows
        if row["current_p0_signal"] is True and row["expected"]["must_handoff"] is True and row["frame"]["must_handoff"] is False
    )
    return {
        "labeled_rows": len(rows) + len(missing_turns) + duplicate_gold,
        "compared_rows": len(compared_rows),
        "skipped_rows": skipped_rows,
        "duplicate_gold_rows": duplicate_gold,
        "missing_transcript_rows": len(missing_turns),
        "missing_transcript_examples": list(missing_turns)[:10],
        "review_labels": dict(labels),
        "field_accuracy": field_accuracy,
        "must_handoff_accuracy": field_accuracy["must_handoff"]["accuracy"],
        "too_cautious": too_cautious,
        "too_confident": too_confident,
        "p0_misses": p0_misses,
        "current_over_handoff_candidates": current_over_handoff,
        "safe_self_candidates": safe_self_candidates,
        "confidence_buckets": _confidence_buckets(compared_rows),
    }


def _confidence_buckets(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {label: [] for _, _, label in CONFIDENCE_BUCKETS}
    buckets["missing"] = []
    for row in rows:
        buckets[_confidence_bucket(row.get("frame", {}).get("confidence"))].append(row)
    result: dict[str, dict[str, Any]] = {}
    for label, bucket_rows in buckets.items():
        total = len(bucket_rows)
        correct = sum(1 for row in bucket_rows if row.get("field_results", {}).get("must_handoff") == "correct")
        result[label] = {
            "rows": total,
            "must_handoff_correct": correct,
            "must_handoff_accuracy": _ratio(correct, total),
            "too_cautious": sum(
                1 for row in bucket_rows if row["frame"]["must_handoff"] is True and row["expected"]["must_handoff"] is False
            ),
            "too_confident": sum(
                1 for row in bucket_rows if row["frame"]["must_handoff"] is False and row["expected"]["must_handoff"] is True
            ),
        }
    return result


def _acceptance(summary: Mapping[str, Any]) -> dict[str, Any]:
    notes: list[str] = []
    status = "pass"
    if summary.get("missing_transcript_rows"):
        status = "needs_review"
        notes.append("Gold contains rows that are absent from transcripts.")
    if summary.get("duplicate_gold_rows"):
        status = "needs_review"
        notes.append("Gold contains duplicate dialog_id/turn rows.")
    if summary.get("too_confident"):
        status = "blocked_for_active"
        notes.append("Frame has too_confident rows: it would keep self-answer where gold expects manager.")
    if summary.get("too_cautious"):
        notes.append("Frame remains too cautious on safe/self rows; active autonomy needs calibration before Ф3.")
    if summary.get("skipped_rows"):
        status = "needs_review" if status == "pass" else status
        notes.append("Some gold rows are unclear/not comparable.")
    if status == "pass" and summary.get("compared_rows", 0) == 0:
        status = "needs_review"
        notes.append("No comparable gold rows.")
    return {"status": status, "notes": notes}


def _confidence_bucket(value: Any) -> str:
    score = _float_or_none(value)
    if score is None:
        return "missing"
    for low, high, label in CONFIDENCE_BUCKETS:
        if low <= score < high:
            return label
    return "missing"


def _float_or_none(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0 or score > 1.0:
        return None
    return score


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _clean(value: Any) -> str:
    return str(value or "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
