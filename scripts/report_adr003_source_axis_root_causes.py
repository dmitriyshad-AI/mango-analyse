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

from scripts.report_adr003_current_handoff_fact_gap import _sha256, _source_rev


SCHEMA_VERSION = "adr003_source_axis_root_causes_v1_2026_07_02"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify ADR-003 source-axis blockers without exporting text.")
    parser.add_argument("--fact-gap-report", type=Path, required=True)
    parser.add_argument("--source-axis-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(fact_gap_report=args.fact_gap_report, source_axis_report=args.source_axis_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_source_axis_root_causes_report.json"
    md_path = args.out_dir / "adr003_source_axis_root_causes_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, fact_gap_report: Path, source_axis_report: Path) -> dict[str, Any]:
    fact_gap = _read_json(fact_gap_report)
    source_axis = _read_json(source_axis_report)
    source_index = _source_axis_index(source_axis)
    cases = [
        _case(case, source=source_index.get(_key(case), {}))
        for case in (fact_gap.get("cases") or [])
        if isinstance(case, Mapping)
    ]
    totals = _totals(cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "fact_gap_report": str(fact_gap_report),
            "fact_gap_report_sha256": _sha256(fact_gap_report),
            "source_axis_report": str(source_axis_report),
            "source_axis_report_sha256": _sha256(source_axis_report),
        },
        "totals": totals,
        "breakdowns": _breakdowns(cases),
        "cases": cases,
        "acceptance": _acceptance(totals),
        "notes": [
            "Report-only diagnostic: no route/text/runtime behavior changes.",
            "No client message or bot answer text is exported.",
            "This report classifies blockers; it must not be used as active policy by itself.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    breakdowns = report.get("breakdowns") if isinstance(report.get("breakdowns"), Mapping) else {}
    lines = [
        "# ADR-003 Source-Axis Root Causes",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Cases: `{totals.get('cases', 0)}`",
        f"- Route-only active candidates: `{totals.get('route_only_active_candidates', 0)}`",
        f"- Missing required slot: `{totals.get('missing_required_slot', 0)}`",
        f"- Platform-axis taxonomy gap: `{totals.get('platform_axis_taxonomy_gap', 0)}`",
        f"- Danger-adjacent: `{totals.get('danger_adjacent', 0)}`",
        f"- Manager-only policy: `{totals.get('manager_only_policy', 0)}`",
        "",
        "## Root causes",
        "",
    ]
    for code, count in sorted((breakdowns.get("by_root_cause") or {}).items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{code}`: `{count}`")
    lines.extend(["", "## Cases", ""])
    for case in report.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        lines.append(
            f"- `{case.get('dialog_id')}#{case.get('turn')}` route=`{case.get('route')}` "
            f"root=`{case.get('root_cause')}` active=`{case.get('active_readiness')}`"
        )
        lines.append(
            f"  - missing slots: `{', '.join(case.get('missing_slots') or [])}`; "
            f"missing categories: `{', '.join(case.get('missing_categories') or [])}`"
        )
        lines.append(
            f"  - support: product=`{case.get('product_check_status')}` "
            f"platform_facts=`{case.get('platform_fact_count')}` price_facts=`{case.get('price_fact_count')}`"
        )
        lines.append(f"  - next: {case.get('recommended_next_step')}")
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _case(case: Mapping[str, Any], *, source: Mapping[str, Any]) -> dict[str, Any]:
    kb = case.get("kb_support") if isinstance(case.get("kb_support"), Mapping) else {}
    root = _root_cause(case, source=source, kb=kb)
    return {
        "dialog_id": str(case.get("dialog_id") or ""),
        "turn": _int(case.get("turn")),
        "route": str(case.get("route") or ""),
        "requested_action": str(case.get("requested_action") or ""),
        "source_axis_policy_status": str(source.get("policy_status") or ""),
        "next_autonomy_workstream": str(case.get("next_autonomy_workstream") or source.get("next_autonomy_workstream") or ""),
        "root_cause": root,
        "active_readiness": "no_go",
        "active_behavior_allowed": False,
        "client_text_exported": False,
        "bot_text_exported": False,
        "missing_slots": [str(item) for item in (kb.get("missing_slots") or []) if str(item)],
        "missing_categories": [str(item) for item in (case.get("missing_fact_categories") or []) if str(item)],
        "uncovered_categories": [str(item) for item in (kb.get("uncovered_categories") or []) if str(item)],
        "source_alignment_uncovered_categories": [
            str(item) for item in (case.get("source_alignment_uncovered_categories") or []) if str(item)
        ],
        "proven_parts": [str(item) for item in (kb.get("proven_parts") or []) if str(item)],
        "product_check_status": str(kb.get("product_check_status") or ""),
        "product_check_reason": str(kb.get("product_check_reason") or ""),
        "product_check_exact_fact_key_count": len(kb.get("product_check_exact_fact_keys") or []),
        "platform_fact_count": _int(kb.get("platform_fact_count")),
        "price_fact_count": _int(kb.get("price_fact_count")),
        "proof_status": str(case.get("proof_status") or ""),
        "proof_reason": str(case.get("proof_reason") or ""),
        "proof_source_fact_key_present": bool(str(case.get("proof_source_fact_key") or "").strip()),
        "recommended_next_step": _recommendation(root),
    }


def _root_cause(case: Mapping[str, Any], *, source: Mapping[str, Any], kb: Mapping[str, Any]) -> str:
    next_workstream = str(case.get("next_autonomy_workstream") or source.get("next_autonomy_workstream") or "")
    route = str(case.get("route") or "")
    missing_slots = [str(item) for item in (kb.get("missing_slots") or []) if str(item)]
    missing_categories = {str(item) for item in (case.get("missing_fact_categories") or []) if str(item)}
    platform_fact_count = _int(kb.get("platform_fact_count"))
    if next_workstream == "danger_adjacent_do_not_lower":
        return "danger_adjacent_do_not_lower"
    if route == "manager_only":
        if "platform_current" in missing_categories and platform_fact_count > 0:
            return "manager_only_with_platform_axis_taxonomy_gap"
        return "manager_only_policy"
    if missing_slots:
        return "missing_required_slot_partial_policy_needed"
    if "platform_current" in missing_categories and platform_fact_count > 0:
        return "platform_axis_taxonomy_gap"
    if case.get("root_cause") == "proof_axis_mismatch" or source.get("policy_status") == "blocked_source_axis_mismatch":
        return "proof_axis_mismatch_unresolved"
    return str(case.get("root_cause") or "measurement_review_required")


def _recommendation(root: str) -> str:
    mapping = {
        "danger_adjacent_do_not_lower": "Keep excluded from autonomy; do not solve with route/text demotion.",
        "manager_only_with_platform_axis_taxonomy_gap": (
            "Do not demote manager_only. Separately fix fact-axis taxonomy for platform_current, then re-measure."
        ),
        "manager_only_policy": "Owner policy is required before any manager_only demotion; current goal does not authorize it.",
        "missing_required_slot_partial_policy_needed": (
            "This is not route-only. A future partial-answer policy may answer proven parts and ask the missing slot, "
            "but only after semantic review and a text policy."
        ),
        "platform_axis_taxonomy_gap": (
            "Fix the fact-axis taxonomy/reporting so platform_current facts are recognized structurally, not by adding live regex."
        ),
        "proof_axis_mismatch_unresolved": "Inspect exact proof selection and required axes before any active work.",
    }
    return mapping.get(root, "Manual review before any behavior change.")


def _source_axis_index(report: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    cases = report.get("cases") if isinstance(report.get("cases"), list) else []
    return {_key(case): case for case in cases if isinstance(case, Mapping)}


def _key(case: Mapping[str, Any]) -> tuple[str, int]:
    return (str(case.get("dialog_id") or ""), _int(case.get("turn")))


def _totals(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "cases": len(cases),
        "route_only_active_candidates": 0,
        "missing_required_slot": sum(
            1 for case in cases if case.get("root_cause") == "missing_required_slot_partial_policy_needed"
        ),
        "platform_axis_taxonomy_gap": sum(
            1
            for case in cases
            if case.get("root_cause") in {"platform_axis_taxonomy_gap", "manager_only_with_platform_axis_taxonomy_gap"}
        ),
        "danger_adjacent": sum(1 for case in cases if case.get("root_cause") == "danger_adjacent_do_not_lower"),
        "manager_only_policy": sum(
            1
            for case in cases
            if case.get("root_cause") in {"manager_only_policy", "manager_only_with_platform_axis_taxonomy_gap"}
        ),
    }


def _breakdowns(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_root_cause": dict(Counter(str(case.get("root_cause") or "") for case in cases)),
        "by_route": dict(Counter(str(case.get("route") or "") for case in cases)),
        "by_next_workstream": dict(Counter(str(case.get("next_autonomy_workstream") or "") for case in cases)),
    }


def _acceptance(totals: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "pass_report_only",
        "active_readiness": "no_go",
        "notes": [
            "Active autonomy remains NO-GO: this report emits no route or text changes.",
            "No route-only active candidates were identified.",
            "Next work should address fact/proof axis taxonomy and partial-answer policy separately.",
        ],
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
