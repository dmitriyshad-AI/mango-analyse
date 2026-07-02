#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_adr003_fact_gated_self_answer_readiness import build_report as build_readiness_report


SCHEMA_VERSION = "adr003_exact_proof_injection_shadow_v1_2026_07_02"
SAFE_RISK_CLASSES = {"safe", "benign"}
SAFE_ANSWERABILITY = {"answer_self", "self"}
SAFE_ACTIONS = {"answer_question", "acknowledge", "acknowledge_status"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Shadow-check residual blockers after exact proof injection.")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    args = parser.parse_args(argv)

    report = build_report(
        transcripts=args.transcripts,
        gold=args.gold,
        kb_snapshot=args.kb_snapshot,
        confidence_threshold=args.confidence_threshold,
        as_of_date=_parse_date(args.as_of_date),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_exact_proof_injection_shadow_report.json"
    md_path = args.out_dir / "adr003_exact_proof_injection_shadow_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(
    *,
    transcripts: Path,
    gold: Path,
    kb_snapshot: Path,
    confidence_threshold: float = 0.90,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    as_of = as_of_date or date.today()
    readiness = build_readiness_report(
        transcripts=transcripts,
        gold=gold,
        kb_snapshot=kb_snapshot,
        confidence_threshold=confidence_threshold,
    )
    turns = _load_turn_index(transcripts)
    rows = _manager_only_exact_proof_rows(readiness)
    cases = [
        _case_from_row(
            row,
            turns.get((str(row.get("dialog_id") or ""), _int_or_zero(row.get("turn")))),
            confidence_threshold=confidence_threshold,
            as_of_date=as_of,
        )
        for row in rows
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "transcripts": str(transcripts),
            "gold": str(gold),
            "kb_snapshot": str(kb_snapshot),
            "confidence_threshold": confidence_threshold,
            "as_of_date": as_of.isoformat(),
        },
        "totals": _totals(cases, readiness),
        "breakdowns": _breakdowns(cases),
        "cases": cases,
        "acceptance": _acceptance(cases),
        "notes": [
            "Report-only scorer: route/text/runtime behavior is unchanged.",
            "Exact-proof injection is hypothetical telemetry only; no fact is injected into the bot.",
            "manager_only remains non-demotable by this phase.",
        ],
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    breakdowns = report.get("breakdowns") if isinstance(report.get("breakdowns"), Mapping) else {}
    lines = [
        "# ADR-003 F2h Exact-Proof Injection Shadow",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Manager-only exact-proof rows: `{totals.get('manager_only_exact_proof_rows', 0)}`",
        f"- Fresh client-safe proof after hypothetical injection: `{totals.get('fresh_client_safe_exact_proof', 0)}`",
        f"- Evidence-only sufficient rows: `{totals.get('evidence_only_sufficient_rows', 0)}`",
        f"- Rows still blocked after injection: `{totals.get('still_blocked_after_injection', 0)}`",
        "",
        "## Residual Blockers",
        "",
    ]
    for code, count in sorted((breakdowns.get("by_residual_blocker") or {}).items()):
        lines.append(f"- `{code}`: `{count}`")
    lines.extend(["", "## Cases", ""])
    for item in report.get("cases") or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"- `{item.get('dialog_id')}#{item.get('turn')}` route=`{item.get('route')}` "
            f"fresh_proof=`{item.get('fresh_client_safe_exact_proof')}` "
            f"evidence_only_sufficient=`{item.get('evidence_only_sufficient')}`"
        )
        lines.append(f"  - fact: `{item.get('source_fact_key')}` valid_until=`{item.get('source_fact_valid_until')}`")
        lines.append(f"  - residual blockers: `{', '.join(item.get('residual_blockers') or [])}`")
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _load_turn_index(transcripts: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for line in transcripts.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        dialog = json.loads(line)
        if not isinstance(dialog, Mapping):
            continue
        dialog_id = str(dialog.get("dialog_id") or "")
        for turn in dialog.get("turns") or []:
            if not isinstance(turn, Mapping):
                continue
            result[(dialog_id, _int_or_zero(turn.get("turn")))] = turn
    return result


def _manager_only_exact_proof_rows(readiness: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    groups = readiness.get("groups") if isinstance(readiness.get("groups"), Mapping) else {}
    group = groups.get("manager_only_exact_proof_needs_policy")
    examples = group.get("examples") if isinstance(group, Mapping) else []
    return [row for row in examples or [] if isinstance(row, Mapping)]


def _case_from_row(
    row: Mapping[str, Any],
    turn: Mapping[str, Any] | None,
    *,
    confidence_threshold: float,
    as_of_date: date,
) -> dict[str, Any]:
    turn = turn if isinstance(turn, Mapping) else {}
    product_check = row.get("product_existence_check") if isinstance(row.get("product_existence_check"), Mapping) else {}
    entry = product_check.get("entry") if isinstance(product_check.get("entry"), Mapping) else {}
    best = row.get("best_kb_match") if isinstance(row.get("best_kb_match"), Mapping) else {}
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    message_type = str(turn.get("bot_message_type") or "")
    missing_facts = _strings(turn.get("bot_missing_facts"))
    valid_until = str(entry.get("valid_until") or best.get("valid_until") or "")
    source_fact_key = str(entry.get("source_fact_key") or best.get("fact_key") or "")
    fresh_client_safe = _fresh_client_safe_exact_proof(entry, best=best, as_of_date=as_of_date)
    residual = _residual_blockers(
        row=row,
        turn=turn,
        frame=frame,
        confidence_threshold=confidence_threshold,
        fresh_client_safe=fresh_client_safe,
        message_type=message_type,
        missing_facts=missing_facts,
    )
    non_policy_residual = [
        code
        for code in residual
        if code
        not in {
            "route_is_manager_only",
            "message_type_context_update",
        }
    ]
    return {
        "dialog_id": str(row.get("dialog_id") or ""),
        "turn": _int_or_zero(row.get("turn")),
        "route": str(row.get("route") or turn.get("bot_route") or ""),
        "message_type": message_type,
        "source_fact_key": source_fact_key,
        "source_fact_valid_until": valid_until,
        "fresh_client_safe_exact_proof": fresh_client_safe,
        "requested_action": str(row.get("requested_action") or frame.get("requested_action") or ""),
        "frame_risk_class": str(row.get("frame_risk_class") or frame.get("risk_class") or ""),
        "frame_answerability": str(row.get("frame_answerability") or frame.get("answerability") or ""),
        "frame_must_handoff": bool(row.get("frame_must_handoff") or frame.get("must_handoff") is True),
        "frame_confidence": _float_or_none(row.get("frame_confidence") if row.get("frame_confidence") is not None else frame.get("confidence")),
        "missing_fact_count": len(missing_facts),
        "runtime_exact_fact_was_missing": True,
        "residual_blockers": residual,
        "non_policy_residual_blockers": non_policy_residual,
        "evidence_only_sufficient": fresh_client_safe and not non_policy_residual,
        "active_readiness": "no_go",
    }


def _fresh_client_safe_exact_proof(entry: Mapping[str, Any], *, best: Mapping[str, Any], as_of_date: date) -> bool:
    if not entry and not best:
        return False
    status = str(entry.get("existence_status") or best.get("existence_status") or "").strip()
    if status not in {"exists", "not_offered"}:
        return False
    if not str(entry.get("client_safe_text") or best.get("client_safe_text_excerpt") or "").strip():
        return False
    valid_until = str(entry.get("valid_until") or best.get("valid_until") or "").strip()
    if not valid_until:
        return False
    parsed = _parse_date(valid_until)
    return parsed >= as_of_date


def _residual_blockers(
    *,
    row: Mapping[str, Any],
    turn: Mapping[str, Any],
    frame: Mapping[str, Any],
    confidence_threshold: float,
    fresh_client_safe: bool,
    message_type: str,
    missing_facts: Sequence[str],
) -> list[str]:
    blockers: list[str] = ["route_is_manager_only"]
    if not fresh_client_safe:
        blockers.append("exact_proof_not_fresh_client_safe")
    if message_type and message_type != "question":
        blockers.append(f"message_type_{message_type}")
    risk = str(row.get("frame_risk_class") or frame.get("risk_class") or "").strip().casefold()
    answerability = str(row.get("frame_answerability") or frame.get("answerability") or "").strip().casefold()
    action = str(row.get("requested_action") or frame.get("requested_action") or "").strip().casefold()
    if risk not in SAFE_RISK_CLASSES:
        blockers.append("frame_risk_not_safe")
    if answerability not in SAFE_ANSWERABILITY:
        blockers.append("frame_answerability_not_self")
    if action not in SAFE_ACTIONS:
        blockers.append("frame_action_not_safe_reference")
    if row.get("frame_must_handoff") is True or frame.get("must_handoff") is True:
        blockers.append("frame_must_handoff")
    confidence = _float_or_none(row.get("frame_confidence") if row.get("frame_confidence") is not None else frame.get("confidence"))
    if confidence is None or confidence < confidence_threshold:
        blockers.append("frame_confidence_below_threshold")
    if _missing_live_or_operational_facts(missing_facts):
        blockers.append("runtime_missing_live_or_operational_facts")
    elif missing_facts:
        blockers.append("runtime_missing_facts_present")
    return list(dict.fromkeys(blockers))


def _missing_live_or_operational_facts(items: Sequence[str]) -> bool:
    blob = " ".join(str(item or "") for item in items).casefold().replace("ё", "е")
    return any(
        marker in blob
        for marker in (
            "налич",
            "мест",
            "групп",
            "смен",
            "подходит",
            "подходящ",
            "провер",
            "услов",
        )
    )


def _totals(cases: Sequence[Mapping[str, Any]], readiness: Mapping[str, Any]) -> dict[str, Any]:
    readiness_totals = readiness.get("totals") if isinstance(readiness.get("totals"), Mapping) else {}
    return {
        "readiness_strict_f3_draft_candidates": readiness_totals.get("strict_f3_draft_candidates", 0),
        "manager_only_exact_proof_rows": len(cases),
        "fresh_client_safe_exact_proof": sum(1 for case in cases if case.get("fresh_client_safe_exact_proof")),
        "evidence_only_sufficient_rows": sum(1 for case in cases if case.get("evidence_only_sufficient")),
        "still_blocked_after_injection": sum(1 for case in cases if case.get("residual_blockers")),
    }


def _breakdowns(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_residual_blocker": dict(Counter(code for case in cases for code in case.get("residual_blockers") or [])),
        "by_non_policy_residual_blocker": dict(
            Counter(code for case in cases for code in case.get("non_policy_residual_blockers") or [])
        ),
        "by_requested_action": dict(Counter(str(case.get("requested_action") or "") for case in cases)),
        "by_message_type": dict(Counter(str(case.get("message_type") or "") for case in cases)),
    }


def _acceptance(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        status = "pass_no_manager_only_exact_proof_rows"
        notes = ["No manager_only exact-proof rows remain in this input."]
    else:
        status = "pass_shadow_diagnosed"
        notes = [
            "Active remains NO-GO: this phase only simulates telemetry evidence injection.",
            "Fresh exact proof alone is not enough if route, frame, message_type or missing-fact blockers remain.",
            "Any active work needs a separate shadow phase and Claude #1 reggrade.",
        ]
    notes.append("Report-only: no route/text/runtime changes.")
    return {"status": status, "active_readiness": "no_go", "notes": notes}


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return date.min


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item or "").strip()]
    return []


def _int_or_zero(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _source_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
