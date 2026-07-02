#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
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

from scripts.report_adr003_frame_calibration_queue import (
    _fact_categories,
    _missing_fact_categories,
)


SCHEMA_VERSION = "adr003_current_handoff_fact_gap_v1_2026_07_02"
HANDOFF_ROUTES = {"manager_only", "draft_for_manager"}
SAFE_SKIP_WORKSTREAMS = {"danger_adjacent_do_not_lower", "no_current_route_leverage"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report root causes for current handoff fact gaps in ADR-003 F2.")
    parser.add_argument("--queue-report", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(
        queue_report=args.queue_report,
        transcripts=args.transcripts,
        kb_snapshot=args.kb_snapshot,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_current_handoff_fact_gap_report.json"
    md_path = args.out_dir / "adr003_current_handoff_fact_gap_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, queue_report: Path, transcripts: Path, kb_snapshot: Path) -> dict[str, Any]:
    queue = json.loads(queue_report.read_text(encoding="utf-8"))
    turns = _load_turn_index(transcripts)
    facts = _load_kb_facts(kb_snapshot)
    catalog = build_product_existence_axes_catalog(facts)
    current_rows = _current_handoff_rows(queue)
    cases = [
        _case_from_row(row, turn=turns.get((str(row.get("dialog_id") or ""), int(row.get("turn") or 0))), facts=facts, catalog=catalog)
        for row in current_rows
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "queue_report": str(queue_report),
            "queue_report_sha256": _sha256(queue_report),
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
            "Report-only diagnostic: route/text/runtime behavior is unchanged.",
            "This report explains why current handoff rows are not active self-answer candidates.",
            "A row is actionable only after exact fact coverage, safe text policy, and owner semantic review.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    breakdowns = report.get("breakdowns") if isinstance(report.get("breakdowns"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    lines = [
        "# ADR-003 F2y Current Handoff Fact Gap",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Active readiness: `{acceptance.get('active_readiness', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Current handoff rows: `{totals.get('current_handoff_rows', 0)}`",
        f"- Non-danger rows: `{totals.get('non_danger_rows', 0)}`",
        f"- Route-only candidates: `{totals.get('route_only_candidates', 0)}`",
        f"- Proof axis mismatch: `{totals.get('proof_axis_mismatch', 0)}`",
        f"- Frame action blocks proof: `{totals.get('frame_action_blocks_proof', 0)}`",
        f"- Partial facts but slot needed: `{totals.get('partial_facts_slot_needed', 0)}`",
        f"- Danger excluded: `{totals.get('danger_excluded', 0)}`",
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
            f"action=`{case.get('requested_action')}` next=`{case.get('next_autonomy_workstream')}`"
        )
        lines.append(f"  - root: `{case.get('root_cause')}`")
        lines.append(f"  - proof: status=`{case.get('proof_status')}` reason=`{case.get('proof_reason')}`")
        lines.append(f"  - missing categories: `{', '.join(case.get('missing_fact_categories') or [])}`")
        lines.append(f"  - fact categories: `{', '.join(case.get('source_fact_categories') or [])}`")
        if case.get("kb_support"):
            support = case["kb_support"]
            lines.append(
                f"  - kb support: price=`{support.get('price_fact_count')}` "
                f"platform=`{support.get('platform_fact_count')}` product_check=`{support.get('product_check_status')}`"
            )
            lines.append(
                f"  - proven parts: `{', '.join(support.get('proven_parts') or [])}`; "
                f"missing slots: `{', '.join(support.get('missing_slots') or [])}`"
            )
        lines.append(f"  - next: {case.get('recommended_next_step')}")
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _current_handoff_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    real = report.get("real_lever_analysis") if isinstance(report.get("real_lever_analysis"), Mapping) else {}
    queue = real.get("current_handoff_queue") if isinstance(real.get("current_handoff_queue"), Mapping) else {}
    examples = queue.get("examples") if isinstance(queue.get("examples"), list) else []
    return [row for row in examples if isinstance(row, Mapping) and str(row.get("route") or "") in HANDOFF_ROUTES]


def _case_from_row(
    row: Mapping[str, Any],
    *,
    turn: Mapping[str, Any] | None,
    facts: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    turn = turn if isinstance(turn, Mapping) else {}
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    requested = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    proof = (
        turn.get("bot_semantic_frame_proof_reconciliation_shadow")
        if isinstance(turn.get("bot_semantic_frame_proof_reconciliation_shadow"), Mapping)
        else {}
    )
    existence = (
        turn.get("bot_semantic_frame_existence_proof_shadow")
        if isinstance(turn.get("bot_semantic_frame_existence_proof_shadow"), Mapping)
        else {}
    )
    retrieval = turn.get("bot_fact_retrieval_trace") if isinstance(turn.get("bot_fact_retrieval_trace"), Mapping) else {}
    missing_text = " ".join(str(item or "") for item in _listish(proof.get("result_missing_facts") or turn.get("bot_missing_facts")))
    missing_categories = _case_missing_categories(missing_text)
    source_fact_key = str(row.get("source_fact_key") or proof.get("source_fact_key") or "").strip()
    source_fact = _find_fact(facts, source_fact_key, active_brand=_active_brand(turn, requested))
    source_categories = _fact_categories(source_fact) if source_fact else []
    kb_support = _kb_support(facts, catalog, turn=turn, requested=requested, missing_categories=missing_categories)
    root_cause = _root_cause(row, proof=proof, existence=existence, kb_support=kb_support)
    case = {
        "dialog_id": str(row.get("dialog_id") or ""),
        "turn": int(row.get("turn") or 0),
        "route": str(row.get("route") or ""),
        "requested_action": str(row.get("requested_action") or frame.get("requested_action") or ""),
        "next_autonomy_workstream": str(row.get("next_autonomy_workstream") or ""),
        "root_cause": root_cause,
        "recommended_next_step": _recommended_next_step(root_cause),
        "active_behavior_allowed": False,
        "client_excerpt": _redacted_excerpt(turn.get("client_message")),
        "bot_excerpt": _redacted_excerpt(turn.get("bot_text")),
        "frame": {
            "risk_class": str(row.get("frame_risk_class") or frame.get("risk_class") or ""),
            "answerability": str(row.get("frame_answerability") or frame.get("answerability") or ""),
            "must_handoff": row.get("frame_must_handoff") if row.get("frame_must_handoff") is not None else frame.get("must_handoff"),
            "confidence": row.get("frame_confidence") if row.get("frame_confidence") is not None else frame.get("confidence"),
        },
        "proof_status": str(proof.get("status") or row.get("proof_reconciliation_status") or ""),
        "proof_reason": str(proof.get("reason") or row.get("proof_reconciliation_reason") or ""),
        "proof_source_fact_key": source_fact_key,
        "existence_proof_status": str(existence.get("status") or ""),
        "existence_proof_reason": str(existence.get("reason") or proof.get("proof_reason") or ""),
        "missing_facts": _listish(proof.get("result_missing_facts") or turn.get("bot_missing_facts")),
        "missing_fact_categories": missing_categories,
        "source_fact_categories": source_categories,
        "source_alignment_uncovered_categories": list(row.get("source_alignment_uncovered_categories") or []),
        "source_fact_brand": str(source_fact.get("brand") or "") if source_fact else "",
        "source_fact_type": str(source_fact.get("fact_type") or "") if source_fact else "",
        "source_fact_structured_keys": sorted(str(key) for key in (source_fact.get("structured_value") or {}).keys())
        if source_fact and isinstance(source_fact.get("structured_value"), Mapping)
        else [],
        "retrieval": {
            "candidate_count": retrieval.get("candidate_count", 0),
            "required_fact_keys": _listish(retrieval.get("required_fact_keys")),
            "selected_exact_ids": _listish(retrieval.get("selected_exact_ids")),
            "selected_adjacent_ids": _listish(retrieval.get("selected_adjacent_ids")),
            "mode": str(retrieval.get("mode") or ""),
        },
        "knowledge_snippet_count": len(_listish(turn.get("bot_knowledge_snippets"))),
        "kb_support": kb_support,
    }
    case["why_not_active"] = _why_not_active(case)
    return case


def _root_cause(
    row: Mapping[str, Any],
    *,
    proof: Mapping[str, Any],
    existence: Mapping[str, Any],
    kb_support: Mapping[str, Any],
) -> str:
    next_workstream = str(row.get("next_autonomy_workstream") or "")
    proof_reason = str(proof.get("reason") or row.get("proof_reconciliation_reason") or "")
    existence_reason = str(existence.get("reason") or proof.get("proof_reason") or "")
    if next_workstream in SAFE_SKIP_WORKSTREAMS:
        return next_workstream
    if next_workstream == "fix_proof_axis_alignment" or row.get("source_alignment_status") == "blocked_source_axis_mismatch":
        return "proof_axis_mismatch"
    if proof_reason == "requested_action_not_answer_question" or existence_reason == "requested_action_not_answer_question":
        return "frame_action_blocks_existence_proof"
    if existence_reason == "required_axis_missing":
        if kb_support.get("price_fact_count") or kb_support.get("platform_fact_count"):
            return "partial_facts_available_but_slot_needed"
        return "required_axis_missing_no_exact_fact"
    if proof_reason == "no_exact_fact_keys":
        return "no_exact_fact_keys"
    return "measurement_review_required"


def _recommended_next_step(root_cause: str) -> str:
    recommendations = {
        "proof_axis_mismatch": "Tighten proof/source alignment: a fact must cover every requested missing-fact axis before text readiness.",
        "frame_action_blocks_existence_proof": "Calibrate SemanticFrame so stable existence/age suitability is answer_question, not check_availability.",
        "partial_facts_available_but_slot_needed": "Add partial-answer shadow: answer proven platform/format facts while asking only for the missing slot; report-only first.",
        "required_axis_missing_no_exact_fact": "Improve fact retrieval/query axes or KB coverage before any self-answer route.",
        "no_exact_fact_keys": "Trace why existence proof produced no exact fact key; do not demote route.",
        "danger_adjacent_do_not_lower": "Keep excluded from autonomy.",
        "no_current_route_leverage": "No autonomy work: runtime already answers self.",
        "measurement_review_required": "Manual review of telemetry/gold needed before implementation.",
    }
    return recommendations.get(root_cause, "Review in shadow before any behavior change.")


def _why_not_active(case: Mapping[str, Any]) -> list[str]:
    reasons = ["report_only"]
    if case.get("route") == "manager_only":
        reasons.append("route_manager_only")
    if case.get("root_cause") != "route_only_candidate":
        reasons.append(str(case.get("root_cause") or "not_ready"))
    if case.get("missing_facts"):
        reasons.append("missing_facts_present")
    return list(dict.fromkeys(reasons))


def _kb_support(
    facts: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    *,
    turn: Mapping[str, Any],
    requested: Mapping[str, Any],
    missing_categories: Sequence[str],
) -> dict[str, Any]:
    brand = _active_brand(turn, requested)
    product_check = verify_product_format_exists(
        catalog,
        brand=brand,
        grade=requested.get("grade"),
        subject=str(requested.get("subject") or requested.get("raw_text") or ""),
        format=str(requested.get("format") or requested.get("raw_text") or ""),
        program_kind=str(requested.get("program_kind") or requested.get("raw_text") or ""),
        product_family=str(requested.get("product_family") or ""),
    )
    price_facts = (
        _supporting_facts(facts, brand=brand, categories={"price_cost"}, requested=requested)
        if "price_cost" in missing_categories
        else []
    )
    platform_facts = (
        _supporting_facts(facts, brand=brand, categories={"platform_current"}, requested=requested, platform=True)
        if "platform_current" in missing_categories
        else []
    )
    proven_parts: list[str] = []
    if product_check.get("status") in {"exists", "not_offered"}:
        proven_parts.append("product_existence")
    if price_facts:
        proven_parts.append("price_cost")
    if platform_facts:
        proven_parts.append("platform_current")
    missing_slots: list[str] = []
    if "class_grade" in missing_categories and not _grade_present(requested.get("grade")):
        missing_slots.append("grade")
    covered_categories = set()
    if price_facts:
        covered_categories.add("price_cost")
    if platform_facts:
        covered_categories.add("platform_current")
    if product_check.get("status") in {"exists", "not_offered"}:
        covered_categories.update({"class_grade", "program_direction"})
    return {
        "product_check_status": str(product_check.get("status") or ""),
        "product_check_reason": str(product_check.get("reason") or ""),
        "product_check_exact_fact_keys": [
            str(item.get("source_fact_key") or "")
            for item in product_check.get("matches") or []
            if isinstance(item, Mapping)
        ][:8],
        "price_fact_count": len(price_facts),
        "price_fact_keys": [item["fact_key"] for item in price_facts[:8]],
        "platform_fact_count": len(platform_facts),
        "platform_fact_keys": [item["fact_key"] for item in platform_facts[:8]],
        "missing_categories": list(missing_categories),
        "proven_parts": proven_parts,
        "missing_slots": missing_slots,
        "uncovered_categories": [category for category in missing_categories if category not in covered_categories],
    }


def _supporting_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    brand: str,
    categories: set[str],
    requested: Mapping[str, Any],
    platform: bool = False,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fact in facts:
        if str(fact.get("brand") or "").strip().casefold() != brand:
            continue
        if fact.get("allowed_for_client_answer") is not True or fact.get("forbidden_for_client") is True or fact.get("internal_only") is True:
            continue
        text = " ".join(
            str(fact.get(key) or "")
            for key in ("fact_key", "fact_type", "product", "program_kind", "client_safe_text")
        ).casefold()
        fact_categories = set(_fact_categories(fact))
        fact_type = str(fact.get("fact_type") or "").strip().casefold()
        if platform:
            matches = "soholms" in text or "online_access_platform" in text or "online_platform_transition" in text
        else:
            matches = fact_type == "price" and bool(categories & fact_categories)
        if not matches or not _fact_matches_requested_hint(fact, requested=requested):
            continue
        result.append(
            {
                "fact_key": str(fact.get("fact_key") or fact.get("fact_id") or ""),
                "fact_type": str(fact.get("fact_type") or ""),
                "valid_until": str(fact.get("valid_until") or ""),
                "text_hash": _hash_text(fact.get("client_safe_text")),
                "text_length": len(str(fact.get("client_safe_text") or "")),
            }
        )
    return result


def _fact_matches_requested_hint(fact: Mapping[str, Any], *, requested: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(fact.get(key) or "")
        for key in ("fact_key", "fact_type", "product", "program_kind", "client_safe_text")
    ).casefold()
    requested_text = " ".join(str(requested.get(key) or "") for key in ("format", "program_kind", "raw_text")).casefold()
    if any(marker in requested_text for marker in ("online", "онлайн", "soholms", "soho")):
        if not any(marker in text for marker in ("online", "онлайн", "soholms", "soho")):
            return False
    if any(marker in requested_text for marker in ("regular_course", "регуляр", "годов")):
        if any(marker in text for marker in ("camp", "лвш", "лагер", "городск", "летн", "интенсив", "ege", "егэ")):
            return False
    return True


def _case_missing_categories(value: str) -> list[str]:
    text = value.casefold()
    categories = set(_missing_fact_categories(text))
    if "platform.current" in text or "платформ" in text or "soholms" in text:
        categories.add("platform_current")
    if "price" in text or "стоим" in text or "цен" in text:
        categories.add("price_cost")
    return sorted(categories)


def _grade_present(value: Any) -> bool:
    return bool(re.search(r"\b([1-9]|1[01])\b", str(value or "")))


def _load_turn_index(transcripts: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    with transcripts.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            dialog = json.loads(line)
            if not isinstance(dialog, Mapping):
                continue
            dialog_id = str(dialog.get("dialog_id") or "")
            for index, turn in enumerate(dialog.get("turns") or [], 1):
                if not isinstance(turn, Mapping):
                    continue
                turn_no = int(turn.get("turn") or index)
                result[(dialog_id, turn_no)] = turn
    return result


def _load_kb_facts(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    facts = payload.get("facts") if isinstance(payload, Mapping) else payload
    return [fact for fact in facts if isinstance(fact, Mapping)] if isinstance(facts, list) else []


def _find_fact(facts: Sequence[Mapping[str, Any]], fact_key: str, *, active_brand: str) -> Mapping[str, Any]:
    if not fact_key:
        return {}
    candidates = [
        fact
        for fact in facts
        if fact_key in {str(fact.get("fact_key") or ""), str(fact.get("fact_id") or ""), str(fact.get("id") or "")}
    ]
    for fact in candidates:
        if str(fact.get("brand") or "").strip().casefold() == active_brand:
            return fact
    return candidates[0] if candidates else {}


def _active_brand(turn: Mapping[str, Any], requested: Mapping[str, Any]) -> str:
    for value in (turn.get("brand"), turn.get("__dialog_brand"), turn.get("active_brand"), requested.get("brand")):
        clean = str(value or "").strip().casefold()
        if clean in {"foton", "unpk"}:
            return clean
    return ""


def _listish(value: object) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if value in (None, "", False):
        return []
    return [value]


def _totals(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "current_handoff_rows": len(cases),
        "non_danger_rows": sum(1 for case in cases if case.get("root_cause") != "danger_adjacent_do_not_lower"),
        "route_only_candidates": sum(1 for case in cases if case.get("root_cause") == "route_only_candidate"),
        "proof_axis_mismatch": sum(1 for case in cases if case.get("root_cause") == "proof_axis_mismatch"),
        "frame_action_blocks_proof": sum(1 for case in cases if case.get("root_cause") == "frame_action_blocks_existence_proof"),
        "partial_facts_slot_needed": sum(1 for case in cases if case.get("root_cause") == "partial_facts_available_but_slot_needed"),
        "danger_excluded": sum(1 for case in cases if case.get("root_cause") == "danger_adjacent_do_not_lower"),
    }


def _breakdowns(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "by_root_cause": dict(Counter(str(case.get("root_cause") or "") for case in cases)),
        "by_route": dict(Counter(str(case.get("route") or "") for case in cases)),
        "by_requested_action": dict(Counter(str(case.get("requested_action") or "") for case in cases)),
    }


def _acceptance(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    notes = [
        "Active autonomy remains NO-GO: this report found no route-only candidate.",
        "Do not build a renderer before proof/source alignment covers requested missing-fact axes.",
        "manager_only rows remain manager_only until an owner-approved policy says otherwise.",
    ]
    if any(case.get("root_cause") == "partial_facts_available_but_slot_needed" for case in cases):
        notes.append("Partial-answer shadow may be worth measuring: answer proven facts while asking only for the missing slot.")
    return {"status": "pass_report_only", "active_readiness": "no_go", "notes": notes}


def _redacted_excerpt(value: Any, *, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", "[email]", text)
    text = re.sub(r"(?<!\d)(?:\+?7|8)?[\s(.-]*\d{3}[\s).-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)", "[phone]", text)
    text = re.sub(r"(?<!\d)\d{7,}(?!\d)", "[id]", text)
    return text[:limit]


def _hash_text(value: Any) -> str:
    text = str(value or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
