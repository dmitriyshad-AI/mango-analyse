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

from mango_mvp.knowledge_base.product_existence_axes_catalog import (
    build_product_existence_axes_catalog,
    verify_product_format_exists,
)

from scripts.report_adr003_current_handoff_fact_gap import (
    _case_missing_categories,
    _listish,
    _load_kb_facts,
    _redacted_excerpt,
    _sha256,
    _source_rev,
)


SCHEMA_VERSION = "adr003_action_gate_counterfactual_proof_v1_2026_07_02"
HARD_RESIDUAL_CATEGORIES = {
    "live_availability",
    "payment_access",
    "boarding_food",
    "dates_schedule",
    "location_address",
    "price_cost",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report ADR-003 action-gate counterfactual proof diagnostics.")
    parser.add_argument("--fact-gap-report", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(
        fact_gap_report=args.fact_gap_report,
        transcripts=args.transcripts,
        kb_snapshot=args.kb_snapshot,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_action_gate_counterfactual_proof_report.json"
    md_path = args.out_dir / "adr003_action_gate_counterfactual_proof_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, fact_gap_report: Path, transcripts: Path, kb_snapshot: Path) -> dict[str, Any]:
    fact_gap = json.loads(fact_gap_report.read_text(encoding="utf-8"))
    facts = _load_kb_facts(kb_snapshot)
    catalog = build_product_existence_axes_catalog(facts)
    turn_index = _load_turn_index(transcripts)
    cases = [
        _case_from_gap_case(case, turn=turn_index.get((str(case.get("dialog_id") or ""), int(case.get("turn") or 0))), catalog=catalog)
        for case in fact_gap.get("cases") or []
        if isinstance(case, Mapping)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "fact_gap_report": str(fact_gap_report),
            "fact_gap_report_sha256": _sha256(fact_gap_report),
            "transcripts": str(transcripts),
            "transcripts_sha256": _sha256(transcripts),
            "kb_snapshot": str(kb_snapshot),
            "kb_snapshot_sha256": _sha256(kb_snapshot),
        },
        "totals": _totals(cases),
        "breakdowns": _breakdowns(cases),
        "cases": cases,
        "acceptance": _acceptance(cases),
        "notes": [
            "Report-only counterfactual: no route/text/runtime behavior changes.",
            "Action-only counterfactual changes only requested_action for diagnostics.",
            "Safe-reference counterfactual changes frame fields only inside this offline report.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    breakdowns = report.get("breakdowns") if isinstance(report.get("breakdowns"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    lines = [
        "# ADR-003 F2aa Action Gate Counterfactual Proof",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Cases: `{totals.get('cases', 0)}`",
        f"- Scope confusion cases: `{totals.get('scope_confusion_total', 0)}`",
        f"- Action-only still blocked: `{totals.get('action_only_still_blocked_total', 0)}`",
        f"- Safe-reference exact proof: `{totals.get('safe_reference_counterfactual_exact_proof_total', 0)}`",
        f"- Residual hard missing after proof: `{totals.get('counterfactual_residual_hard_missing_total', 0)}`",
        f"- New active candidates: `{totals.get('new_active_candidates', 0)}`",
        "",
        "## Statuses",
        "",
    ]
    for code, count in sorted((breakdowns.get("by_counterfactual_status") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{code}`: `{count}`")
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        lines.append(
            f"- `{case.get('dialog_id')}#{case.get('turn')}` route=`{case.get('route')}` "
            f"root=`{case.get('root_cause')}` status=`{case.get('counterfactual_status')}`"
        )
        lines.append(
            f"  - current: action=`{case.get('current_frame', {}).get('requested_action')}` "
            f"risk=`{case.get('current_frame', {}).get('risk_class')}` "
            f"answerability=`{case.get('current_frame', {}).get('answerability')}` "
            f"must_handoff=`{case.get('current_frame', {}).get('must_handoff')}`"
        )
        lines.append(
            f"  - action-only: `{case.get('action_only_counterfactual', {}).get('status')}`/"
            f"`{case.get('action_only_counterfactual', {}).get('reason')}`"
        )
        lines.append(
            f"  - safe-reference: `{case.get('safe_reference_counterfactual', {}).get('status')}`/"
            f"`{case.get('safe_reference_counterfactual', {}).get('reason')}`"
        )
        lines.append(
            f"  - residual: `{', '.join(case.get('residual_missing_categories_after_existence') or [])}`; "
            f"why not active: `{', '.join(case.get('why_not_active') or [])}`"
        )
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _case_from_gap_case(
    gap_case: Mapping[str, Any],
    *,
    turn: Mapping[str, Any] | None,
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    turn = turn if isinstance(turn, Mapping) else {}
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    proof = (
        turn.get("bot_semantic_frame_existence_proof_shadow")
        if isinstance(turn.get("bot_semantic_frame_existence_proof_shadow"), Mapping)
        else {}
    )
    missing_text = " ".join(str(item or "") for item in _listish(gap_case.get("missing_facts") or turn.get("bot_missing_facts")))
    missing_categories = _case_missing_categories(missing_text)
    current_frame = _frame_summary(frame)
    action_only_frame = dict(frame)
    action_only_frame["requested_action"] = "answer_question"
    safe_reference_frame = dict(action_only_frame)
    safe_reference_frame.update({"risk_class": "safe", "answerability": "answer_self", "must_handoff": False})
    action_only = _existence_counterfactual(action_only_frame, catalog=catalog, active_brand=_active_brand(turn, frame))
    safe_reference = _existence_counterfactual(safe_reference_frame, catalog=catalog, active_brand=_active_brand(turn, frame))
    residual = _residual_categories(missing_categories, safe_reference)
    status = _counterfactual_status(gap_case, action_only=action_only, safe_reference=safe_reference, residual=residual)
    why_not_active = _why_not_active(gap_case, status=status, residual=residual)
    return {
        "dialog_id": str(gap_case.get("dialog_id") or ""),
        "turn": int(gap_case.get("turn") or 0),
        "route": str(gap_case.get("route") or ""),
        "root_cause": str(gap_case.get("root_cause") or ""),
        "client_excerpt": _redacted_excerpt(turn.get("client_message"), limit=180),
        "current_frame": current_frame,
        "current_existence_proof": {
            "status": str(proof.get("status") or ""),
            "reason": str(proof.get("reason") or ""),
            "exact_fact_key_count": len(_listish(proof.get("exact_fact_keys"))),
        },
        "action_only_counterfactual": action_only,
        "safe_reference_counterfactual": safe_reference,
        "missing_fact_categories": list(missing_categories),
        "residual_missing_categories_after_existence": residual,
        "counterfactual_status": status,
        "why_not_active": why_not_active,
        "active_behavior_allowed": False,
    }


def _frame_summary(frame: Mapping[str, Any]) -> dict[str, Any]:
    requested = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    return {
        "requested_action": str(frame.get("requested_action") or ""),
        "risk_class": str(frame.get("risk_class") or ""),
        "answerability": str(frame.get("answerability") or ""),
        "must_handoff": frame.get("must_handoff"),
        "confidence": frame.get("confidence"),
        "payment_readiness": str(frame.get("payment_readiness") or ""),
        "deal_stage": str(frame.get("deal_stage") or ""),
        "requested_product": {
            "brand": str(requested.get("brand") or ""),
            "subject": str(requested.get("subject") or ""),
            "grade": str(requested.get("grade") or ""),
            "format": str(requested.get("format") or ""),
            "program_kind": str(requested.get("program_kind") or ""),
        },
    }


def _existence_counterfactual(
    frame: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    active_brand: str,
) -> dict[str, Any]:
    action = str(frame.get("requested_action") or "").strip().casefold()
    risk = str(frame.get("risk_class") or "").strip().casefold()
    must_handoff = bool(frame.get("must_handoff")) if isinstance(frame.get("must_handoff"), bool) else frame.get("must_handoff")
    if action != "answer_question":
        return {"status": "blocked", "reason": "requested_action_not_answer_question", "exact_fact_keys": []}
    if must_handoff is True and risk in {"p0", "manager_action"}:
        return {"status": "blocked", "reason": "protected_handoff_frame", "exact_fact_keys": []}
    if str(frame.get("payment_readiness") or "").strip().casefold() in {"ready_to_pay", "paid", "dispute"}:
        return {"status": "blocked", "reason": "payment_readiness_blocked", "exact_fact_keys": []}
    if str(frame.get("deal_stage") or "").strip().casefold() in {"post_payment", "support"}:
        return {"status": "blocked", "reason": "deal_stage_blocked", "exact_fact_keys": []}
    requested = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    brand = _requested_brand(requested) or active_brand
    if brand not in {"foton", "unpk"}:
        return {"status": "blocked", "reason": "unknown_brand", "exact_fact_keys": []}
    proof = verify_product_format_exists(
        catalog,
        brand=brand,
        grade=str(requested.get("grade") or ""),
        subject=str(requested.get("subject") or ""),
        format=str(requested.get("format") or ""),
        program_kind=str(requested.get("program_kind") or ""),
        product_family=str(requested.get("raw_text") or ""),
    )
    status = str(proof.get("status") or "")
    keys = [
        str(item.get("source_fact_key") or "")
        for item in proof.get("matches") or []
        if isinstance(item, Mapping) and str(item.get("source_fact_key") or "")
    ]
    return {
        "status": status if status in {"exists", "not_offered"} and keys else "blocked",
        "reason": str(proof.get("reason") or status or "no_exact_product_existence_fact"),
        "exact_fact_keys": keys[:8],
        "query_axes": proof.get("query_axes") if isinstance(proof.get("query_axes"), Mapping) else {},
    }


def _counterfactual_status(
    gap_case: Mapping[str, Any],
    *,
    action_only: Mapping[str, Any],
    safe_reference: Mapping[str, Any],
    residual: Sequence[str],
) -> str:
    root = str(gap_case.get("root_cause") or "")
    if root == "danger_adjacent_do_not_lower":
        return "negative_control_preserved"
    if str(action_only.get("status") or "") != "blocked":
        if set(residual).intersection(HARD_RESIDUAL_CATEGORIES):
            return "action_only_exact_proof_but_residual_hard_missing"
        return "action_only_exact_proof_report_only"
    if str(safe_reference.get("status") or "") in {"exists", "not_offered"}:
        if set(residual).intersection(HARD_RESIDUAL_CATEGORIES):
            return "safe_reference_exact_proof_but_residual_hard_missing"
        return "safe_reference_exact_proof_report_only"
    return "safe_reference_no_exact_proof"


def _why_not_active(gap_case: Mapping[str, Any], *, status: str, residual: Sequence[str]) -> list[str]:
    reasons = ["report_only", "active_behavior_allowed_false"]
    if str(gap_case.get("route") or "") == "manager_only":
        reasons.append("route_manager_only")
    if status == "safe_reference_exact_proof_but_residual_hard_missing":
        reasons.append("residual_hard_missing_axes")
    if status == "negative_control_preserved":
        reasons.append("negative_control")
    if residual:
        reasons.append("residual_missing_categories_present")
    return list(dict.fromkeys(reasons))


def _residual_categories(missing_categories: Sequence[str], safe_reference: Mapping[str, Any]) -> list[str]:
    residual = set(str(item) for item in missing_categories)
    if str(safe_reference.get("status") or "") in {"exists", "not_offered"}:
        residual.difference_update({"class_grade", "program_direction"})
    return sorted(residual)


def _active_brand(turn: Mapping[str, Any], frame: Mapping[str, Any]) -> str:
    requested = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    for value in (turn.get("brand"), turn.get("__dialog_brand"), turn.get("active_brand"), requested.get("brand")):
        normalized = str(value or "").strip().casefold()
        if normalized in {"foton", "фотон"}:
            return "foton"
        if normalized in {"unpk", "унпк", "унпк мфти"}:
            return "unpk"
    return ""


def _requested_brand(requested: Mapping[str, Any]) -> str:
    raw = str(requested.get("brand") or "").strip().casefold()
    if raw in {"foton", "фотон"}:
        return "foton"
    if raw in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return ""


def _load_turn_index(transcripts: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    index: dict[tuple[str, int], Mapping[str, Any]] = {}
    with transcripts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            dialog = json.loads(line)
            if not isinstance(dialog, Mapping):
                continue
            dialog_id = str(dialog.get("dialog_id") or "")
            dialog_brand = str(dialog.get("brand") or "")
            for turn_no, turn in enumerate(dialog.get("turns") or [], 1):
                if not isinstance(turn, Mapping):
                    continue
                row = dict(turn)
                row["__dialog_brand"] = dialog_brand or str(turn.get("brand") or "")
                index[(dialog_id, int(turn.get("turn") or turn_no))] = row
    return index


def _totals(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(cases),
        "scope_confusion_total": sum(
            1
            for case in cases
            if case.get("root_cause") == "frame_action_blocks_existence_proof"
            or case.get("current_frame", {}).get("requested_action") == "check_availability"
        ),
        "action_gate_blocked_proof_total": sum(
            1
            for case in cases
            if case.get("current_existence_proof", {}).get("reason") == "requested_action_not_answer_question"
        ),
        "action_only_still_blocked_total": sum(
            1 for case in cases if case.get("action_only_counterfactual", {}).get("status") == "blocked"
        ),
        "safe_reference_counterfactual_exact_proof_total": sum(
            1 for case in cases if case.get("safe_reference_counterfactual", {}).get("status") in {"exists", "not_offered"}
        ),
        "counterfactual_residual_hard_missing_total": sum(
            1
            for case in cases
            if set(case.get("residual_missing_categories_after_existence") or []).intersection(HARD_RESIDUAL_CATEGORIES)
        ),
        "negative_controls_preserved_total": sum(
            1 for case in cases if case.get("counterfactual_status") == "negative_control_preserved"
        ),
        "new_active_candidates": 0,
    }


def _breakdowns(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_root_cause": dict(Counter(str(case.get("root_cause") or "") for case in cases)),
        "by_counterfactual_status": dict(Counter(str(case.get("counterfactual_status") or "") for case in cases)),
        "by_route": dict(Counter(str(case.get("route") or "") for case in cases)),
    }


def _acceptance(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pass_report_only",
        "active_readiness": "no_go",
        "notes": [
            "Active autonomy remains NO-GO: counterfactuals do not change runtime behavior.",
            "Action-only calibration is insufficient when must_handoff/risk_class remain manager_action.",
            "Safe-reference exact proof still requires residual-missing-axis review and owner policy.",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
