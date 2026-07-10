#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
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


SCHEMA_VERSION = "adr003_frame_gold_queue_v1_2026_07_01"
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?\d[\s\-()]{0,3}){7,}\d")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ADR-003 SemanticFrame mismatch queue for manual gold labelling.")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-matches", action="store_true")
    args = parser.parse_args(argv)

    result = build_queue(transcripts=args.transcripts, include_matches=args.include_matches)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "adr003_frame_gold_queue.jsonl"
    csv_path = args.out_dir / "adr003_frame_gold_queue.csv"
    summary_path = args.out_dir / "adr003_frame_gold_queue_summary.json"
    summary_md_path = args.out_dir / "adr003_frame_gold_queue_summary.md"
    _write_jsonl(jsonl_path, result["rows"])
    _write_csv(csv_path, result["rows"])
    summary_path.write_text(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_md_path.write_text(render_markdown(result["summary"]), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": result["summary"].get("input_status") == "all_framed",
                "input_status": result["summary"].get("input_status"),
                "rows": len(result["rows"]),
                "jsonl": str(jsonl_path),
                "csv": str(csv_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["summary"].get("input_status") == "all_framed" else 2


def build_queue(*, transcripts: Path, include_matches: bool = False) -> dict[str, Any]:
    dialogs = _load_transcripts(transcripts)
    rows: list[dict[str, Any]] = []
    mismatch_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    p0_counts: Counter[str] = Counter()
    frame_risks: Counter[str] = Counter()
    total_turns = 0
    framed_turns = 0
    invalid_frame_turns = 0
    pii_rows = 0
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        brand = str(dialog.get("brand") or "")
        for index, turn in enumerate(_turns(dialog), 1):
            total_turns += 1
            frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
            if not frame:
                continue
            framed_turns += 1
            route_handoff = _actual_route_handoff(turn)
            p0_signal = _actual_p0_signal(turn)
            frame_must = _strict_bool(frame.get("must_handoff"))
            if frame_must is None:
                invalid_frame_turns += 1
            route_match = frame_must == route_handoff if frame_must is not None else False
            p0_match = frame_must == p0_signal if frame_must is not None else False
            mismatch_type = _mismatch_type(frame_must=frame_must, route_handoff=route_handoff, p0_signal=p0_signal)
            route_counts["match" if route_match else "mismatch"] += 1
            p0_counts["match" if p0_match else "mismatch"] += 1
            frame_risks[str(frame.get("risk_class") or "unknown")] += 1
            if mismatch_type:
                mismatch_counts[mismatch_type] += 1
            if not include_matches and not mismatch_type:
                continue
            client_message = _compact_text(turn.get("client_message"), limit=600)
            bot_text = _compact_text(turn.get("bot_text"), limit=900)
            row_has_pii = _has_pii_like(client_message) or _has_pii_like(bot_text)
            if row_has_pii:
                pii_rows += 1
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dialog_id": dialog_id,
                    "brand": brand,
                    "turn": int(turn.get("turn") or index),
                    "mismatch_type": mismatch_type or "match",
                    "needs_gold_label": bool(mismatch_type),
                    "current_route": str(turn.get("bot_route") or ""),
                    "current_route_handoff": route_handoff,
                    "current_p0_signal": p0_signal,
                    "frame_must_handoff": frame_must,
                    "route_alignment": "match" if route_match else "mismatch",
                    "p0_alignment": "match" if p0_match else "mismatch",
                    "frame_intent": str(frame.get("intent") or ""),
                    "frame_risk_class": str(frame.get("risk_class") or ""),
                    "frame_answerability": str(frame.get("answerability") or ""),
                    "frame_requested_action": str(frame.get("requested_action") or ""),
                    "frame_confidence": frame.get("confidence"),
                    "client_message": client_message,
                    "bot_text": bot_text,
                    "pii_risk": row_has_pii,
                    "safety_flags": list(turn.get("bot_safety_flags") or []),
                    "manager_checklist": list(turn.get("bot_manager_checklist") or []),
                    "review_question": _review_question(mismatch_type),
                    "gold_fields_to_fill": {
                        "expected_must_handoff": None,
                        "expected_risk_class": "",
                        "expected_answerability": "",
                        "expected_requested_action": "",
                        "notes": "",
                    },
                }
            )
    if framed_turns == 0:
        input_status = "no_frame"
    elif framed_turns != total_turns:
        input_status = "partial_frame"
    elif invalid_frame_turns:
        input_status = "invalid_frame"
    else:
        input_status = "all_framed"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "transcripts": str(transcripts),
        "input_status": input_status,
        "dialogs": len(dialogs),
        "turns_total": total_turns,
        "framed_turns": framed_turns,
        "missing_frame_turns": total_turns - framed_turns,
        "invalid_frame_turns": invalid_frame_turns,
        "queue_rows": len(rows),
        "include_matches": include_matches,
        "pii_risk": pii_rows > 0,
        "pii_risk_rows": pii_rows,
        "must_handoff_vs_route": dict(route_counts),
        "must_handoff_vs_p0_signal": dict(p0_counts),
        "mismatch_types": dict(mismatch_counts),
        "frame_risk_classes": dict(frame_risks),
    }
    return {"summary": summary, "rows": rows}


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# ADR-003 Frame Gold Queue",
        "",
        f"- Dialogs: `{summary.get('dialogs')}`",
        f"- Turns total: `{summary.get('turns_total')}`",
        f"- Framed turns: `{summary.get('framed_turns')}`",
        f"- Input status: `{summary.get('input_status')}`",
        f"- Queue rows: `{summary.get('queue_rows')}`",
        f"- PII risk rows: `{summary.get('pii_risk_rows')}`",
        f"- Must-handoff vs route: `{summary.get('must_handoff_vs_route')}`",
        f"- Must-handoff vs P0 signal: `{summary.get('must_handoff_vs_p0_signal')}`",
        "",
        "## Mismatch Types",
        "",
    ]
    for key, value in sorted((summary.get("mismatch_types") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _mismatch_type(*, frame_must: bool | None, route_handoff: bool, p0_signal: bool) -> str:
    if frame_must is None:
        return "invalid_frame_must_handoff"
    parts: list[str] = []
    if frame_must and not route_handoff:
        parts.append("frame_handoff_current_self")
    elif not frame_must and route_handoff:
        parts.append("frame_self_current_handoff")
    if frame_must and not p0_signal:
        parts.append("frame_handoff_no_p0_signal")
    elif not frame_must and p0_signal:
        parts.append("frame_self_current_p0_signal")
    return "+".join(parts)


def _review_question(mismatch_type: str) -> str:
    if not mismatch_type:
        return "Confirm frame fields against gold."
    if "frame_handoff_no_p0_signal" in mismatch_type:
        return "Should SemanticFrame add manager review here, or is it too cautious without P0 signal?"
    if "frame_self_current_p0_signal" in mismatch_type:
        return "Is current P0/manager guard correct, or is SemanticFrame right to keep this safe?"
    if "frame_handoff_current_self" in mismatch_type:
        return "Should this self-answer have been manager review, or is SemanticFrame too cautious?"
    if "frame_self_current_handoff" in mismatch_type:
        return "Is this over-handoff that SemanticFrame should fix, or a needed manager review?"
    return "Label expected frame and route policy."


def _compact_text(value: Any, *, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) > limit:
        return text[: limit - 1].rstrip() + "…"
    return text


def _has_pii_like(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "dialog_id",
        "brand",
        "turn",
        "mismatch_type",
        "current_route",
        "current_route_handoff",
        "current_p0_signal",
        "frame_must_handoff",
        "frame_intent",
        "frame_risk_class",
        "frame_answerability",
        "frame_requested_action",
        "frame_confidence",
        "client_message",
        "bot_text",
        "review_question",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
