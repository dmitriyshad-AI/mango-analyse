#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_adr003_existence_fact_verification import build_report as build_existence_report


SCHEMA_VERSION = "adr003_fact_gated_self_answer_readiness_v1_2026_07_02"
STRICT_ROUTE = "draft_for_manager"
SELF_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot", "self_answer"}
MANAGER_ONLY_ROUTE = "manager_only"
SAFE_RISK_CLASSES = {"safe", "benign"}
SAFE_ANSWERABILITY = {"answer_self", "self"}
SAFE_ACTIONS = {"answer_question", "acknowledge", "acknowledge_status"}
MONEY_MARKERS = {
    "payment",
    "paid",
    "refund",
    "money",
    "legal",
    "complaint",
    "dispute",
    "оплат",
    "возврат",
    "жалоб",
    "договор",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report fact-gated SemanticFrame self-answer readiness.")
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    args = parser.parse_args(argv)

    report = build_report(
        transcripts=args.transcripts,
        gold=args.gold,
        kb_snapshot=args.kb_snapshot,
        confidence_threshold=args.confidence_threshold,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_fact_gated_self_answer_readiness_report.json"
    md_path = args.out_dir / "adr003_fact_gated_self_answer_readiness_report.md"
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
) -> dict[str, Any]:
    existence = build_existence_report(transcripts=transcripts, gold=gold, kb_snapshot=kb_snapshot)
    rows = _existence_rows(existence)
    classified = [_classify_row(row, confidence_threshold=confidence_threshold) for row in rows]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "transcripts": str(transcripts),
            "gold": str(gold),
            "kb_snapshot": str(kb_snapshot),
            "confidence_threshold": confidence_threshold,
        },
        "totals": _totals(classified),
        "breakdowns": _breakdowns(classified),
        "groups": _groups(classified),
        "acceptance": _acceptance(classified),
        "notes": [
            "Report-only scorer: route/text/runtime behavior is unchanged.",
            "Strict F3 candidates require route_before=draft_for_manager; manager_only is never demoted here.",
            "Exact proof comes from product_existence_axes_catalog via the F2e existence report.",
        ],
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    groups = report.get("groups") if isinstance(report.get("groups"), Mapping) else {}
    lines = [
        "# ADR-003 F2f Fact-Gated Self-Answer Readiness",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Existence/format rows: `{totals.get('existence_format_rows', 0)}`",
        f"- Current handoff rows: `{totals.get('current_handoff_rows', 0)}`",
        f"- Strict F3 draft candidates: `{totals.get('strict_f3_draft_candidates', 0)}`",
        f"- Manager-only exact-proof needs policy: `{totals.get('manager_only_exact_proof_needs_policy', 0)}`",
        f"- Already self exact proof: `{totals.get('already_self_exact_proof', 0)}`",
        f"- Blocked no exact proof: `{totals.get('blocked_no_exact_proof', 0)}`",
        f"- Excluded danger/money/P0: `{totals.get('excluded_danger_money_p0', 0)}`",
        "",
        "## Groups",
        "",
    ]
    for group in (
        "strict_f3_draft_candidate",
        "manager_only_exact_proof_needs_policy",
        "already_self_exact_proof",
        "blocked_no_exact_proof",
        "excluded_danger_money_p0",
        "blocked_frame_not_self",
        "other",
    ):
        value = groups.get(group) if isinstance(groups.get(group), Mapping) else {}
        lines.append(f"- `{group}`: `{value.get('count', 0)}`")
    lines.extend(["", "## Strict F3 Candidates", ""])
    for item in (groups.get("strict_f3_draft_candidate") or {}).get("examples", [])[:20]:
        lines.extend(_example_lines(item))
    lines.extend(["", "## Manager-Only Exact-Proof Rows", ""])
    for item in (groups.get("manager_only_exact_proof_needs_policy") or {}).get("examples", [])[:20]:
        lines.extend(_example_lines(item))
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _existence_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("rows")
    if isinstance(rows, list):
        return [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("group")
            in {
                "handoff_with_exact_kb_evidence",
                "handoff_without_exact_kb_evidence",
                "already_self_with_exact_kb_evidence",
                "already_self_without_exact_kb_evidence",
                "excluded_danger_money_p0",
            }
        ]
    groups = report.get("groups") if isinstance(report.get("groups"), Mapping) else {}
    rows: list[Mapping[str, Any]] = []
    for group in (
        "handoff_with_exact_kb_evidence",
        "handoff_without_exact_kb_evidence",
        "already_self_with_exact_kb_evidence",
        "already_self_without_exact_kb_evidence",
        "excluded_danger_money_p0",
    ):
        payload = groups.get(group) if isinstance(groups.get(group), Mapping) else {}
        for row in payload.get("examples") or []:
            if isinstance(row, Mapping):
                rows.append(row)
    return rows


def _classify_row(row: Mapping[str, Any], *, confidence_threshold: float) -> dict[str, Any]:
    route = str(row.get("route") or "")
    frame = _frame_from_row(row)
    status = str((row.get("product_existence_check") or {}).get("status") or "")
    exact_proof = status in {"exists", "not_offered"} and row.get("evidence_level") == "kb_exact"
    danger = bool(row.get("excluded_danger_money_p0")) or _money_or_danger(row, frame)
    frame_self_ok, frame_reasons = _frame_self_ok(frame, confidence_threshold=confidence_threshold)
    group = "other"
    blocked_reasons: list[str] = []
    if danger or row.get("group") == "excluded_danger_money_p0":
        group = "excluded_danger_money_p0"
        blocked_reasons.append("danger_money_or_p0")
    elif not exact_proof:
        group = "blocked_no_exact_proof"
        blocked_reasons.append(status or "no_exact_proof")
    elif route in SELF_ROUTES:
        group = "already_self_exact_proof"
    elif route == MANAGER_ONLY_ROUTE:
        group = "manager_only_exact_proof_needs_policy"
        blocked_reasons.append("route_is_manager_only")
        blocked_reasons.extend(frame_reasons)
    elif not frame_self_ok:
        group = "blocked_frame_not_self"
        blocked_reasons.extend(frame_reasons)
    elif route == STRICT_ROUTE:
        group = "strict_f3_draft_candidate"
    else:
        blocked_reasons.append(f"unsupported_route:{route}")
    result = dict(row)
    result["readiness_group"] = group
    result["strict_f3_candidate"] = group == "strict_f3_draft_candidate"
    result["blocked_reasons"] = blocked_reasons
    result["frame_self_ok"] = frame_self_ok
    result["frame_self_reasons"] = frame_reasons
    return result


def _frame_from_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    requested = row.get("requested_product") if isinstance(row.get("requested_product"), Mapping) else {}
    return {
        "risk_class": str(row.get("frame_risk_class") or ""),
        "answerability": str(row.get("frame_answerability") or ""),
        "requested_action": str(row.get("requested_action") or ""),
        "must_handoff": row.get("frame_must_handoff"),
        "confidence": row.get("frame_confidence"),
        "requested_product": requested,
    }


def _frame_self_ok(frame: Mapping[str, Any], *, confidence_threshold: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if str(frame.get("risk_class") or "").casefold() not in SAFE_RISK_CLASSES:
        reasons.append("frame_risk_not_safe")
    if str(frame.get("answerability") or "").casefold() not in SAFE_ANSWERABILITY:
        reasons.append("frame_answerability_not_self")
    if str(frame.get("requested_action") or "").casefold() not in SAFE_ACTIONS:
        reasons.append("frame_action_not_safe_self")
    if frame.get("must_handoff") is True:
        reasons.append("frame_must_handoff")
    confidence = _float_or_none(frame.get("confidence"))
    if confidence is None or confidence < confidence_threshold:
        reasons.append("low_confidence")
    return not reasons, reasons


def _money_or_danger(row: Mapping[str, Any], frame: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            row.get("dialog_id"),
            row.get("client_excerpt"),
            row.get("bot_excerpt"),
            row.get("frame_risk_class"),
            row.get("requested_action"),
            (frame.get("requested_product") or {}).get("raw_text")
            if isinstance(frame.get("requested_product"), Mapping)
            else "",
        )
    ).casefold()
    return any(marker in text for marker in MONEY_MARKERS)


def _totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "existence_format_rows": len(rows),
        "current_handoff_rows": sum(1 for row in rows if row.get("route") in {STRICT_ROUTE, MANAGER_ONLY_ROUTE}),
        "strict_f3_draft_candidates": sum(1 for row in rows if row.get("readiness_group") == "strict_f3_draft_candidate"),
        "manager_only_exact_proof_needs_policy": sum(
            1 for row in rows if row.get("readiness_group") == "manager_only_exact_proof_needs_policy"
        ),
        "already_self_exact_proof": sum(1 for row in rows if row.get("readiness_group") == "already_self_exact_proof"),
        "blocked_no_exact_proof": sum(1 for row in rows if row.get("readiness_group") == "blocked_no_exact_proof"),
        "excluded_danger_money_p0": sum(1 for row in rows if row.get("readiness_group") == "excluded_danger_money_p0"),
        "blocked_frame_not_self": sum(1 for row in rows if row.get("readiness_group") == "blocked_frame_not_self"),
    }


def _breakdowns(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_readiness_group": _counter(rows, "readiness_group"),
        "by_route": _counter(rows, "route"),
        "by_requested_action": _counter(rows, "requested_action"),
        "by_evidence_level": _counter(rows, "evidence_level"),
    }


def _groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in (
        "strict_f3_draft_candidate",
        "manager_only_exact_proof_needs_policy",
        "already_self_exact_proof",
        "blocked_no_exact_proof",
        "excluded_danger_money_p0",
        "blocked_frame_not_self",
        "other",
    ):
        grouped = [row for row in rows if row.get("readiness_group") == group]
        result[group] = {"count": len(grouped), "examples": grouped[:50]}
    return result


def _acceptance(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    strict = [row for row in rows if row.get("readiness_group") == "strict_f3_draft_candidate"]
    manager_only = [row for row in rows if row.get("readiness_group") == "manager_only_exact_proof_needs_policy"]
    notes: list[str] = []
    if strict:
        status = "pass_shadow_candidates_found"
        active = "needs_claude_reggrade_before_active"
        notes.append("Strict draft_for_manager candidates exist, but active still requires Claude #1 reggrade and Dmitry approval.")
    else:
        status = "pass_no_active_candidate"
        active = "no_go"
        notes.append("No strict draft_for_manager candidates; active F3 remains NO-GO.")
    if manager_only:
        notes.append("Exact-proof manager_only rows exist; they need separate policy/upstream work and cannot be demoted by F3 route gate.")
    notes.append("Report-only: no route/text/runtime changes.")
    return {"status": status, "active_readiness": active, "notes": notes}


def _counter(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def _example_lines(item: Mapping[str, Any]) -> list[str]:
    best = item.get("best_kb_match") if isinstance(item.get("best_kb_match"), Mapping) else {}
    product = item.get("requested_product") if isinstance(item.get("requested_product"), Mapping) else {}
    lines = [
        "- "
        f"`{item.get('dialog_id')}#{item.get('turn')}` "
        f"route=`{item.get('route')}` action=`{item.get('requested_action')}` "
        f"confidence=`{item.get('frame_confidence')}` proof=`{item.get('evidence_level')}`",
        f"  - product: brand={product.get('brand','')} grade={product.get('grade','')} "
        f"program={product.get('program_kind','')} subject={product.get('subject','')}",
        f"  - fact: `{best.get('fact_key','')}`",
    ]
    if item.get("blocked_reasons"):
        lines.append(f"  - blocked: `{', '.join(item.get('blocked_reasons') or [])}`")
    return lines


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_rev() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
