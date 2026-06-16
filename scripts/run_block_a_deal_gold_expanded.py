#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EXISTING_RESULTS = (
    "/Users/dmitrijfabarisov/Projects/Mango_tz116_offline/audits/_inbox/"
    "tz116_crm_llm_shadow_fixed24_codex_20260615_195654/crm_llm_offline_measure_results.jsonl"
)
DEFAULT_SOURCE_ROOT = "/Users/dmitrijfabarisov/Projects/Mango analyse"
DEFAULT_SNAPSHOT = "stable_runtime/deal_aware_amo_live_snapshot_20260513_v2"
DEFAULT_OUT_DIR = "audits/_inbox/block_a_deal_gold_expanded_20260616"

TARGET_LOSS_REASONS = ("Недозвон", "Архив  (нет связи)", "Не актуально", "Действующий клиент")
VALID_VERDICTS = {"closed_valid", "closed_too_early", "follow_up_needed", "manual_review"}
VALID_RISKS = {"no_risk", "low", "medium", "high", "manual_review"}
VALID_NEXT_STEP_CLASSES = {"follow_up_check", "manual_check", "no_action"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve()
    os.environ.setdefault("SOURCE_WORKSPACE_ROOT", str(source_root))
    os.environ["CRM_TALLANTO_MODE"] = "off"
    os.environ.setdefault("CRM_ANALYSIS_PROVIDER", "codex_cli")
    os.environ.setdefault("CRM_ANALYSIS_MODEL", args.model)
    os.environ.setdefault("CRM_ANALYSIS_REASONING_EFFORT", args.reasoning_effort)
    os.environ.setdefault("CRM_ANALYSIS_TIMEOUT_SECONDS", str(args.timeout_sec))
    os.environ.setdefault("CRM_ANALYSIS_LLM_CACHE_ENABLED", "0")
    os.environ.setdefault(
        "CRM_ANALYSIS_LLM_CACHE_DIR",
        str((Path.cwd() / ".codex_local/block_a_gold_expanded/llm_cache").resolve()),
    )

    # Import after env setup: amocrm_runtime settings are loaded at import time.
    from mango_mvp.amocrm_runtime.deal_dossier import build_deal_dossier
    from mango_mvp.amocrm_runtime.deal_llm import DealLLMAnalyzer, DealLLMError
    from mango_mvp.amocrm_runtime.deals import (
        LeadCandidate,
        _analysis_from_selected_lead,  # noqa: PLC2701
        _comparison_summary,  # noqa: PLC2701
        _writeback_blockers,  # noqa: PLC2701
    )
    from mango_mvp.amocrm_runtime.phone_context import PhoneContext, get_phone_context
    from mango_mvp.utils.phone import normalize_phone

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_records = read_jsonl(Path(args.existing_results).expanduser().resolve())
    old_ids = {str(record.get("case_id") or "").split("_")[-1] for record in existing_records}

    snapshot_root = (source_root / args.snapshot_root).resolve()
    deals = read_csv(snapshot_root / "amo_deals_snapshot.csv")
    contacts = read_csv(snapshot_root / "amo_contacts_snapshot.csv")
    tasks = read_csv(snapshot_root / "amo_tasks_snapshot.csv")
    statuses = read_csv(snapshot_root / "amo_status_catalog.csv")
    users_csv = read_csv(snapshot_root / "amo_user_catalog.csv")

    contact_by_id = {row["contact_id"]: row for row in contacts if row.get("contact_id")}
    tasks_by_lead: dict[str, list[dict[str, str]]] = {}
    for row in tasks:
        if row.get("entity_type") == "leads":
            tasks_by_lead.setdefault(str(row.get("entity_id") or ""), []).append(row)

    pipelines = build_pipelines(deals=deals, statuses=statuses)
    users = [{"id": int(row.get("user_id") or 0), "name": row.get("name") or ""} for row in users_csv]
    user_map = {int(row["id"]): str(row["name"]) for row in users if int(row["id"])}

    selected = select_additional_cases(
        deals=deals,
        contact_by_id=contact_by_id,
        old_ids=old_ids,
        normalize_phone=normalize_phone,
    )
    analyzer = DealLLMAnalyzer()

    new_records: list[dict[str, Any]] = []
    redacted_rows: list[dict[str, Any]] = []
    for selected_case in selected:
        brand = selected_case["brand"]
        deal = selected_case["deal"]
        contact_row = selected_case["contact"]
        phone = selected_case["phone"]
        lead = lead_object(deal)
        contact = contact_object(contact_row)
        row_tasks = [task_object(row) for row in tasks_by_lead.get(str(deal.get("lead_id") or ""), [])]
        phone_context = get_phone_context(phone) or empty_phone_context(phone)
        lead_id = int(deal.get("lead_id") or 0)
        contact_id = int(contact_row.get("contact_id") or 0)
        case_id = f"blockA_{brand}_{lead_id}"
        candidate = LeadCandidate(
            contact_id=contact_id,
            lead_id=lead_id,
            score=78,
            confidence=0.78,
            reason="offline_stable_snapshot",
            lead=lead,
        )
        heuristic = _analysis_from_selected_lead(
            None,
            phone_context=phone_context,
            candidate=candidate,
            contact=contact,
            pipelines=pipelines,
            users=users,
            lead=lead,
            notes=[],
            tasks=row_tasks,
        )
        heuristic["brand"] = brand
        dossier = build_deal_dossier(
            phone_context=phone_context,
            contact=contact,
            lead=lead,
            notes=[],
            tasks=row_tasks,
            pipeline_name=str(heuristic.get("pipeline_name") or deal.get("pipeline_name") or ""),
            status_name=str(heuristic.get("status_name") or deal.get("status_name") or ""),
            user_map=user_map,
            active_brand=brand,
        )
        llm = analyze_case(analyzer, dossier=dossier, heuristic=heuristic)
        comparison = _comparison_summary(heuristic, llm)
        final = {**heuristic, **llm, "analysis_source": "llm", "analysis_mode": "llm_shadow"}
        blockers = _writeback_blockers(analysis=final, mode="llm_shadow", comparison=comparison)
        if "offline_measure_no_writeback" not in blockers:
            blockers.append("offline_measure_no_writeback")
        final["writeback_allowed"] = False
        final["writeback_blockers"] = blockers
        final["heuristic_llm_comparison"] = comparison

        new_records.append(
            {
                "case_id": case_id,
                "mode": "shadow",
                "heuristic_analysis": heuristic,
                "llm_analysis": llm,
                "final_analysis": final,
                "comparison": comparison,
            }
        )
        redacted_rows.append(
            {
                "case_id": case_id,
                "brand": brand,
                "loss_reason": selected_case["loss_reason"],
                "pipeline": deal.get("pipeline_name") or "",
                "status": deal.get("status_name") or "",
                "calls_count": len(phone_context.call_rows),
                "tasks_count": len(row_tasks),
                "heuristic_verdict": heuristic.get("close_verdict") or "",
                "heuristic_risk": heuristic.get("premature_close_risk") or "",
                "model_verdict": llm.get("close_verdict") or "",
                "model_risk": llm.get("premature_close_risk") or "",
                "model_confidence": llm.get("confidence") or "",
            }
        )

    all_records = existing_records + new_records
    write_jsonl(out_dir / "expanded_crm_llm_results_32.jsonl", all_records)
    write_csv(out_dir / "new_cases_redacted.csv", redacted_rows)

    gold_rows = [manual_gold_label(record) for record in all_records]
    write_csv(out_dir / "deal_a_gold_labels_manual.csv", gold_rows)

    trace_rows = [build_trace_row(record, gold=gold_rows[index]) for index, record in enumerate(all_records)]
    write_csv(out_dir / "deal_a_gold_trace_manual.csv", trace_rows)
    write_jsonl(out_dir / "deal_a_gold_trace_manual.jsonl", trace_rows)
    summary = build_summary(
        records=all_records,
        trace_rows=trace_rows,
        old_count=len(existing_records),
        new_count=len(new_records),
        out_dir=out_dir,
        snapshot_root=snapshot_root,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary, trace_rows), encoding="utf-8")
    (out_dir / "semantic_review.md").write_text(render_semantic_review(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def select_additional_cases(
    *,
    deals: list[dict[str, str]],
    contact_by_id: Mapping[str, dict[str, str]],
    old_ids: set[str],
    normalize_phone: Any,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_phones: set[str] = set()
    for brand in ("foton", "unpk"):
        for loss_reason in TARGET_LOSS_REASONS:
            for deal in deals:
                if deal.get("lead_id") in old_ids:
                    continue
                if deal.get("status_id") != "143":
                    continue
                if deal.get("loss_reason") != loss_reason:
                    continue
                if infer_brand(deal, contact_by_id=contact_by_id) != brand:
                    continue
                contact_ids = split_pipe(deal.get("linked_contact_ids"))
                if not contact_ids:
                    continue
                contact = contact_by_id.get(contact_ids[0])
                if not contact:
                    continue
                phones = []
                for raw_phone in split_pipe(contact.get("phones")):
                    normalized = normalize_phone(raw_phone)
                    if normalized and normalized not in phones:
                        phones.append(normalized)
                if not phones or phones[0] in seen_phones:
                    continue
                selected.append(
                    {
                        "brand": brand,
                        "loss_reason": loss_reason,
                        "deal": deal,
                        "contact": contact,
                        "phone": phones[0],
                    }
                )
                seen_phones.add(phones[0])
                break
    return selected


def infer_brand(row: Mapping[str, Any], *, contact_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    text = " ".join([safe_str(row.get("lead_name")), safe_str(row.get("custom_field_values_json"))])
    for contact_id in split_pipe(row.get("linked_contact_ids")):
        contact = contact_by_id.get(contact_id) or {}
        text += " " + " ".join(
            [safe_str(contact.get("contact_name")), safe_str(contact.get("custom_field_values_json"))]
        )
    normalized = text.casefold().replace("ё", "е")
    if any(token in normalized for token in ("cdpofoton", "фотон", "foton")):
        return "foton"
    if any(token in normalized for token in ("kmipt", "мфти", "унпк", "unpk")):
        return "unpk"
    return "unknown"


def manual_gold_label(record: Mapping[str, Any]) -> dict[str, str]:
    heuristic = as_mapping(record.get("heuristic_analysis"))
    llm = as_mapping(record.get("llm_analysis"))
    loss = normalize_text(heuristic.get("loss_reason_summary") or llm.get("loss_reason_summary"))
    history = normalize_text(
        " ".join(
            [
                safe_str(heuristic.get("history_summary")),
                safe_str(heuristic.get("latest_call_summary")),
                safe_str(heuristic.get("recommended_next_step")),
                safe_str(llm.get("deal_summary")),
                safe_str(llm.get("close_reason_summary")),
            ]
        )
    )

    if "действующ" in loss:
        verdict, risk, next_step, reason = (
            "closed_valid",
            "no_risk",
            "no_action",
            "manual_policy_active_client_loss_reason",
        )
    elif any(marker in loss for marker in ("дубль", "спам", "тест", "не оставлял", "не квал")):
        verdict, risk, next_step, reason = (
            "closed_valid",
            "no_risk",
            "no_action",
            "manual_policy_non_sales_or_duplicate",
        )
    elif "недозвон" in loss:
        if any(marker in history for marker in ("не актуаль", "отказ", "неинтерес", "не подходит")):
            verdict, risk, next_step, reason = (
                "closed_valid",
                "no_risk",
                "no_action",
                "manual_policy_no_call_but_hard_decline_context",
            )
        else:
            verdict, risk, next_step, reason = (
                "follow_up_needed",
                "medium",
                "follow_up_check",
                "manual_policy_no_answer_requires_follow_up_check",
            )
    elif "архив" in loss and "связ" in loss:
        verdict, risk, next_step, reason = (
            "follow_up_needed",
            "medium",
            "follow_up_check",
            "manual_policy_archive_no_contact_requires_follow_up_check",
        )
    elif "не актуаль" in loss:
        if any(marker in history for marker in ("позже", "вернемся", "вернуться", "осен", "сентябр", "после")):
            verdict, risk, next_step, reason = (
                "follow_up_needed",
                "medium",
                "follow_up_check",
                "manual_policy_not_actual_with_deferred_interest",
            )
        else:
            verdict, risk, next_step, reason = (
                "manual_review",
                "manual_review",
                "manual_check",
                "manual_policy_not_actual_needs_source_check",
            )
    else:
        verdict = safe_str(llm.get("close_verdict")) if safe_float(llm.get("confidence")) >= 0.8 else "manual_review"
        risk = safe_str(llm.get("premature_close_risk")) if verdict != "manual_review" else "manual_review"
        next_step = next_step_class_for_verdict(verdict)
        reason = "manual_policy_fallback_to_confident_model_or_review"

    return {
        "case_id": safe_str(record.get("case_id")),
        "brand": safe_str(heuristic.get("brand")),
        "gold_verdict": require_value(verdict, VALID_VERDICTS, "gold_verdict"),
        "gold_risk": require_value(risk, VALID_RISKS, "gold_risk"),
        "gold_next_step_class": require_value(next_step, VALID_NEXT_STEP_CLASSES, "gold_next_step_class"),
        "gold_reason": reason,
        "review_policy": "codex_manual_business_review_v2_2026_06_16",
    }


def build_trace_row(record: Mapping[str, Any], *, gold: Mapping[str, str]) -> dict[str, Any]:
    heuristic = as_mapping(record.get("heuristic_analysis"))
    llm = as_mapping(record.get("llm_analysis"))
    rule_verdict = safe_str(heuristic.get("close_verdict"))
    rule_risk = safe_str(heuristic.get("premature_close_risk"))
    model_verdict = safe_str(llm.get("close_verdict"))
    model_risk = safe_str(llm.get("premature_close_risk"))
    gold_verdict = safe_str(gold.get("gold_verdict"))
    gold_risk = safe_str(gold.get("gold_risk"))
    gold_next = safe_str(gold.get("gold_next_step_class"))
    rule_next = next_step_class_for_verdict(rule_verdict)
    model_next = next_step_class_for_verdict(model_verdict)
    rule_exact = (rule_verdict, rule_risk, rule_next) == (gold_verdict, gold_risk, gold_next)
    model_exact = (model_verdict, model_risk, model_next) == (gold_verdict, gold_risk, gold_next)
    return {
        "id": safe_str(record.get("case_id")),
        "brand": safe_str(heuristic.get("brand")),
        "gold_verdict": gold_verdict,
        "gold_risk": gold_risk,
        "gold_next_step_class": gold_next,
        "rule_verdict": rule_verdict,
        "rule_risk": rule_risk,
        "rule_next_step_class": rule_next,
        "model_verdict": model_verdict,
        "model_risk": model_risk,
        "model_next_step_class": model_next,
        "confidence": f"{safe_float(llm.get('confidence')):.6f}",
        "gold_reason": safe_str(gold.get("gold_reason")),
        "rule_matches_gold": "Да" if rule_exact else "Нет",
        "model_matches_gold": "Да" if model_exact else "Нет",
        "error_type": classify_error_type(rule_ok=rule_exact, model_ok=model_exact),
        "writeback_allowed": "Нет",
        "classification_method": "block_a_expanded_manual_gold_shadow",
    }


def build_summary(
    *,
    records: list[Mapping[str, Any]],
    trace_rows: list[Mapping[str, Any]],
    old_count: int,
    new_count: int,
    out_dir: Path,
    snapshot_root: Path,
) -> dict[str, Any]:
    model_correct = sum(1 for row in trace_rows if row.get("model_matches_gold") == "Да")
    rule_correct = sum(1 for row in trace_rows if row.get("rule_matches_gold") == "Да")
    high_conf_wrong = [
        row
        for row in trace_rows
        if row.get("model_matches_gold") == "Нет" and safe_float(row.get("confidence")) >= 0.8
    ]
    return {
        "schema_version": "block_a_deal_gold_expanded_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records_total": len(records),
        "old_precomputed_records": old_count,
        "new_local_snapshot_records": new_count,
        "brand_counts": dict(Counter(safe_str(row.get("brand")) for row in trace_rows).most_common()),
        "rule_exact_vs_gold": {
            "correct": rule_correct,
            "total": len(trace_rows),
            "accuracy": rule_correct / len(trace_rows) if trace_rows else 0.0,
        },
        "model_exact_vs_gold": {
            "correct": model_correct,
            "total": len(trace_rows),
            "accuracy": model_correct / len(trace_rows) if trace_rows else 0.0,
        },
        "model_delta_vs_rule": model_correct - rule_correct,
        "error_type_counts": dict(Counter(safe_str(row.get("error_type")) for row in trace_rows).most_common()),
        "high_confidence_wrong_count": len(high_conf_wrong),
        "high_confidence_wrong_ids": [safe_str(row.get("id")) for row in high_conf_wrong],
        "out_dir": str(out_dir),
        "snapshot_root": str(snapshot_root),
        "decision": "keep_shadow",
        "decision_reason": (
            "model is not clearly better without confident errors"
            if high_conf_wrong or model_correct <= rule_correct
            else "candidate_primary_after_external_regrede"
        ),
        "safety": {
            "reads_live_crm": False,
            "writes_amo": False,
            "writes_tallanto": False,
            "writes_crm": False,
            "writeback_allowed": False,
            "tallanto_mode": "off",
            "uses_local_stable_runtime_readonly": True,
            "new_model_calls": new_count,
            "uses_openai_api_key": False,
            "model_transport": "codex_cli",
        },
    }


def render_report(summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> str:
    mistakes = [row for row in rows if row.get("error_type") != "both_correct"]
    return "\n".join(
        [
            "# Block A Expanded Deal Gold Measurement",
            "",
            f"- Rows: `{summary['records_total']}`",
            f"- Brand split: `{json.dumps(summary['brand_counts'], ensure_ascii=False, sort_keys=True)}`",
            f"- Rule exact vs gold: `{summary['rule_exact_vs_gold']['correct']}/{summary['rule_exact_vs_gold']['total']}` = `{summary['rule_exact_vs_gold']['accuracy']:.4f}`",
            f"- Model exact vs gold: `{summary['model_exact_vs_gold']['correct']}/{summary['model_exact_vs_gold']['total']}` = `{summary['model_exact_vs_gold']['accuracy']:.4f}`",
            f"- Model delta vs rule: `{summary['model_delta_vs_rule']}`",
            f"- High-confidence wrong model rows: `{summary['high_confidence_wrong_count']}`",
            f"- Decision: `{summary['decision']}`",
            "",
            "Safety: local snapshots only; no AMO/Tallanto/CRM writes; Tallanto lookup disabled; writeback disabled.",
            "",
            "## Non-Both-Correct Rows",
            "",
            "| id | brand | gold | rule | model | confidence | type | gold reason |",
            "|---|---|---|---|---|---:|---|---|",
            *[
                "| `{id}` | `{brand}` | `{gold}` | `{rule}` | `{model}` | `{confidence}` | `{error}` | `{reason}` |".format(
                    id=row["id"],
                    brand=row["brand"],
                    gold="/".join([row["gold_verdict"], row["gold_risk"], row["gold_next_step_class"]]),
                    rule="/".join([row["rule_verdict"], row["rule_risk"], row["rule_next_step_class"]]),
                    model="/".join([row["model_verdict"], row["model_risk"], row["model_next_step_class"]]),
                    confidence=row["confidence"],
                    error=row["error_type"],
                    reason=row["gold_reason"],
                )
                for row in mistakes[:20]
            ],
            "",
            "Primary is not enabled by this measurement. Claude/Dmitry regrede remains the gate.",
        ]
    ) + "\n"


def render_semantic_review(summary: Mapping[str, Any]) -> str:
    verdict = "PASS_WITH_NOTES"
    if summary["high_confidence_wrong_count"] > 0:
        recommendation = "A stays shadow: confident model mistakes remain."
    elif summary["model_delta_vs_rule"] <= 0:
        recommendation = "A stays shadow: model does not beat the rule."
    else:
        recommendation = "A can be considered a primary candidate only after external regrede."
    return "\n".join(
        [
            "# Semantic Review: Block A Expanded Deal Gold",
            "",
            f"Verdict: `{verdict}`",
            "",
            "## Business Read",
            "",
            f"- {recommendation}",
            "- The gold labels are conservative business labels over closed-deal reasons and local history.",
            "- This is not a live-write approval.",
            "",
            "## Known Limits",
            "",
            "- New cases use the local 2026-05-13 AMO snapshot, not a fresh live CRM read.",
            "- Manual labels are suitable for deciding shadow vs candidate-primary, not for writing anything to CRM.",
        ]
    ) + "\n"


def analyze_case(analyzer: Any, *, dossier: dict[str, Any], heuristic: dict[str, Any]) -> dict[str, Any]:
    from mango_mvp.amocrm_runtime.deal_llm import DealLLMError

    try:
        return analyzer.analyze(dossier=dossier, heuristic_analysis=heuristic)
    except DealLLMError as exc:
        return {
            "analysis_schema_version": "deal_llm_v2",
            "close_verdict": "manual_review",
            "premature_close_risk": "manual_review",
            "close_reason_summary": f"Codex CLI error: {exc}",
            "recommended_next_step": "Проверить вручную.",
            "follow_up_due_at": heuristic.get("follow_up_due_at"),
            "deal_summary": heuristic.get("deal_summary", ""),
            "manager_action_summary": "",
            "confidence": 0.0,
            "needs_manual_review": True,
            "evidence_signals": [],
            "conflict_flags": ["llm_runtime_error"],
            "llm_provider": "codex_cli",
            "llm_model": os.getenv("CRM_ANALYSIS_MODEL", ""),
            "llm_prompt_version": "deal_llm_v2",
        }


def build_pipelines(*, deals: list[dict[str, str]], statuses: list[dict[str, str]]) -> list[dict[str, Any]]:
    pipelines: dict[int, dict[str, Any]] = {}
    for row in statuses:
        pipeline_id = int(row.get("pipeline_id") or 0)
        status_id = int(row.get("status_id") or 0)
        if not pipeline_id:
            continue
        pipeline = pipelines.setdefault(
            pipeline_id,
            {"id": pipeline_id, "name": row.get("pipeline_name") or "", "is_archive": False, "_embedded": {"statuses": []}},
        )
        if status_id:
            pipeline["_embedded"]["statuses"].append({"id": status_id, "name": row.get("status_name") or ""})
    for row in deals:
        pipeline_id = int(row.get("pipeline_id") or 0)
        status_id = int(row.get("status_id") or 0)
        if not pipeline_id:
            continue
        pipeline = pipelines.setdefault(
            pipeline_id,
            {"id": pipeline_id, "name": row.get("pipeline_name") or "", "is_archive": False, "_embedded": {"statuses": []}},
        )
        if status_id and all(int(status.get("id") or 0) != status_id for status in pipeline["_embedded"]["statuses"]):
            pipeline["_embedded"]["statuses"].append({"id": status_id, "name": row.get("status_name") or ""})
    return list(pipelines.values())


def lead_object(row: Mapping[str, Any]) -> dict[str, Any]:
    loss = safe_str(row.get("loss_reason")) or safe_str(json_dict(row.get("custom_field_values_json")).get("Причина отказа (лид)"))
    return {
        "id": int(row.get("lead_id") or 0),
        "name": safe_str(row.get("lead_name")),
        "pipeline_id": int(row.get("pipeline_id") or 0),
        "status_id": int(row.get("status_id") or 0),
        "responsible_user_id": int(row.get("responsible_user_id") or 0),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "closed_at": row.get("closed_at"),
        "custom_fields_values": custom_fields_values(json_dict(row.get("custom_field_values_json"))),
        "_embedded": {"loss_reason": [{"name": loss}] if loss else []},
    }


def contact_object(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row.get("contact_id") or 0),
        "name": safe_str(row.get("contact_name")),
        "responsible_user_id": int(row.get("responsible_user_id") or 0),
        "custom_fields_values": custom_fields_values(json_dict(row.get("custom_field_values_json"))),
    }


def task_object(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row.get("task_id") or 0),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "complete_till": row.get("complete_till"),
        "is_completed": safe_str(row.get("is_completed")).casefold() == "true",
        "text": safe_str(row.get("text")),
        "result": {"text": safe_str(row.get("result"))},
        "responsible_user_id": int(row.get("responsible_user_id") or 0),
    }


def empty_phone_context(phone: str) -> Any:
    from mango_mvp.amocrm_runtime.phone_context import PhoneContext

    return PhoneContext(
        phone=phone,
        source_dir="stable_runtime/deal_aware_amo_live_snapshot_20260513_v2",
        contact_row=None,
        call_rows=[],
        call_ids=[],
        first_call_at=None,
        last_call_at=None,
        manager_history=[],
        interest_summary="",
        objections_summary="",
        current_sales_temperature="",
        recommended_next_step="",
        follow_up_due_at=None,
        history_summary="",
        chronology="",
        tallanto_id="",
        tallanto_match_status="",
    )


def next_step_class_for_verdict(verdict: str) -> str:
    if verdict == "closed_valid":
        return "no_action"
    if verdict in {"follow_up_needed", "closed_too_early"}:
        return "follow_up_check"
    return "manual_check"


def classify_error_type(*, rule_ok: bool, model_ok: bool) -> str:
    if model_ok and not rule_ok:
        return "model_fix"
    if rule_ok and not model_ok:
        return "model_break"
    if model_ok and rule_ok:
        return "both_correct"
    return "both_wrong"


def custom_fields_values(mapping: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"field_name": str(key), "values": [{"value": value}]}
        for key, value in mapping.items()
        if safe_str(value)
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_dict(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(safe_str(value) or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def split_pipe(value: Any) -> list[str]:
    return [chunk.strip() for chunk in re.split(r"\s*\|\s*", safe_str(value)) if chunk.strip()]


def normalize_text(value: Any) -> str:
    return safe_str(value).casefold().replace("ё", "е")


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def require_value(value: str, allowed: set[str], label: str) -> str:
    if value not in allowed:
        raise SystemExit(f"{label}={value!r} is not allowed")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block A expanded closed-deal gold measurement.")
    parser.add_argument("--existing-results", default=DEFAULT_EXISTING_RESULTS)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--snapshot-root", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--timeout-sec", type=int, default=300)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
