from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.canonical_readonly_import import (
    AMO_SOURCE,
    DEFAULT_OUT_ROOT,
    MAIL_SOURCE,
    MANGO_SOURCE,
    CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION,
    CanonicalReadonlyTimelineConfig,
    build_customer_index,
    build_source_manifest,
    duplicate_amo_ids_across_sources,
    infer_brand,
    manual_review_reasons,
    read_amo_contacts_by_phone,
    read_amo_deals_by_contact_id,
    read_calls_by_phone,
    read_csv_rows,
    read_mail_aggregates_by_phone,
    resolve_config,
    shared_amo_reasons_by_phone_from_sources,
    write_json,
)
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.utils.phone import normalize_phone


CANONICAL_READONLY_TRIAGE_SCHEMA_VERSION = "canonical_readonly_customer_timeline_triage_v1"

IDENTITY_RISK_REASONS = {
    "email_bridge_blocked_or_ambiguous",
    "multiple_amo_contacts",
    "multiple_amo_deals",
    "multiple_tallanto_candidates",
    "shared_amo_contact_across_customers",
    "shared_amo_lead_across_customers",
}
SOURCE_LABEL_REASONS = {"source_manual_review_required"}
DATA_GAP_REASONS = {"missing_tallanto_context", "no_mango_calls"}


@dataclass(frozen=True)
class CanonicalReadonlyTriageConfig:
    project_root: Path
    timeline_root: Path = DEFAULT_OUT_ROOT
    out_dir: Path | None = None
    current_runtime_json: Path | None = None
    master_contacts_csv: Path | None = None
    master_calls_csv: Path | None = None
    amo_contacts_csv: Path | None = None
    amo_deals_csv: Path | None = None
    mail_handoff_db: Path | None = None
    mail_bridge_db: Path | None = None
    generated_at: datetime | None = None


def build_canonical_readonly_timeline_triage(config: CanonicalReadonlyTriageConfig) -> Mapping[str, Any]:
    resolved = resolve_triage_config(config)
    generated_at = resolved.generated_at or datetime.now(timezone.utc)
    out_dir = resolved.out_dir or resolved.timeline_root / "triage"
    out_dir.mkdir(parents=True, exist_ok=True)

    import_config = resolve_config(
        CanonicalReadonlyTimelineConfig(
            project_root=resolved.project_root,
            out_root=resolved.timeline_root,
            current_runtime_json=resolved.current_runtime_json,
            master_contacts_csv=resolved.master_contacts_csv,
            master_calls_csv=resolved.master_calls_csv,
            amo_contacts_csv=resolved.amo_contacts_csv,
            amo_deals_csv=resolved.amo_deals_csv,
            mail_handoff_db=resolved.mail_handoff_db,
            mail_bridge_db=resolved.mail_bridge_db,
        )
    )
    contacts = read_csv_rows(import_config.master_contacts_csv)
    customers_by_phone = build_customer_index(contacts, tenant_id=import_config.tenant_id, generated_at=generated_at)
    known_phones = set(customers_by_phone)
    phones = sorted(known_phones)
    calls_by_phone = read_calls_by_phone(
        import_config.master_calls_csv,
        known_phones=known_phones,
        max_per_phone=import_config.max_call_events_per_contact,
    )
    amo_contacts_by_phone = read_amo_contacts_by_phone(import_config.amo_contacts_csv, known_phones=known_phones)
    amo_deals_by_contact_id = read_amo_deals_by_contact_id(import_config.amo_deals_csv)
    duplicate_contact_ids, duplicate_lead_ids = duplicate_amo_ids_across_sources(
        contacts,
        amo_contacts_by_phone=amo_contacts_by_phone,
        amo_deals_by_contact_id=amo_deals_by_contact_id,
    )
    shared_reasons_by_phone = shared_amo_reasons_by_phone_from_sources(
        contacts,
        amo_contacts_by_phone=amo_contacts_by_phone,
        amo_deals_by_contact_id=amo_deals_by_contact_id,
        duplicate_amo_contact_ids=duplicate_contact_ids,
        duplicate_amo_lead_ids=duplicate_lead_ids,
    )
    mail_by_phone = read_mail_aggregates_by_phone(import_config.mail_bridge_db, known_phones=known_phones)
    source_manifest = build_source_manifest(
        {
            "current_runtime_json": import_config.current_runtime_json,
            "master_contacts_csv": import_config.master_contacts_csv,
            "master_calls_csv": import_config.master_calls_csv,
            "amo_contacts_csv": import_config.amo_contacts_csv,
            "amo_deals_csv": import_config.amo_deals_csv,
            "mail_handoff_db": import_config.mail_handoff_db,
            "mail_bridge_db": import_config.mail_bridge_db,
        }
    )

    reason_counts: Counter[str] = Counter()
    reason_class_counts: Counter[str] = Counter()
    customer_class_counts: Counter[str] = Counter()
    reason_combo_counts: Counter[str] = Counter()
    brand_counts: Counter[str] = Counter()
    unknown_brand_reason_counts: Counter[str] = Counter()
    unknown_brand_amo_hint_counts: Counter[str] = Counter()
    preview_candidates_by_brand: Counter[str] = Counter()
    identity_risk_by_brand: Counter[str] = Counter()
    source_label_only_by_brand: Counter[str] = Counter()
    mail_context_by_class: Counter[str] = Counter()
    amo_context_by_class: Counter[str] = Counter()

    for phone in phones:
        row = customers_by_phone[phone]["row"]
        brand = infer_brand(row.values())
        brand_counts[brand] += 1
        reasons = manual_review_reasons(
            row=row,
            calls=calls_by_phone.get(phone, ()),
            mail=mail_by_phone.get(phone),
            duplicate_amo_contact_ids=duplicate_contact_ids,
            duplicate_amo_lead_ids=duplicate_lead_ids,
            extra_reasons=shared_reasons_by_phone.get(phone, ()),
        )
        reason_counts.update(reasons)
        reason_combo_counts[reason_combo_key(reasons)] += 1
        class_name = classify_customer_for_triage(brand=brand, reasons=reasons)
        customer_class_counts[class_name] += 1
        for reason in reasons:
            reason_class_counts[classify_reason(reason)] += 1
        if brand == "unknown":
            unknown_brand_reason_counts[unknown_brand_reason(row=row, phone=phone, amo_contacts_by_phone=amo_contacts_by_phone, amo_deals_by_contact_id=amo_deals_by_contact_id)] += 1
            unknown_brand_amo_hint_counts[brand_hint_from_amo(phone, amo_contacts_by_phone, amo_deals_by_contact_id)] += 1
        if class_name == "manager_preview_candidate":
            preview_candidates_by_brand[brand] += 1
        elif class_name == "identity_risk":
            identity_risk_by_brand[brand] += 1
        elif class_name == "source_label_only":
            source_label_only_by_brand[brand] += 1
        if phone in mail_by_phone:
            mail_context_by_class[class_name] += 1
        if phone in amo_contacts_by_phone:
            amo_context_by_class[class_name] += 1

    store_summary = read_store_summary(import_config.timeline_db, import_config.out_root)
    total = len(phones)
    report = {
        "schema_version": CANONICAL_READONLY_TRIAGE_SCHEMA_VERSION,
        "timeline_schema_version": CANONICAL_READONLY_TIMELINE_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "tenant_id": import_config.tenant_id,
        "tenant_note": "tenant_id is a technical export container here; customer brand is tracked separately in brand_counts.",
        "audience": "internal_analyst_and_head_of_sales",
        "source_timeline_root": str(import_config.out_root),
        "paths": {
            "out_dir": str(out_dir),
            "manual_review_triage_report_json": str(out_dir / "manual_review_triage_report.json"),
            "manual_review_triage_report_md": str(out_dir / "manual_review_triage_report.md"),
            "manager_preview_plan_md": str(out_dir / "manager_preview_plan.md"),
        },
        "summary": {
            "total_customers": total,
            "manual_review_customers_estimated": total - int(customer_class_counts.get("manager_preview_candidate", 0)) - int(customer_class_counts.get("unknown_brand_audit_only", 0)),
            "manager_preview_candidate_count": int(customer_class_counts.get("manager_preview_candidate", 0)),
            "unknown_brand_audit_only_count": int(customer_class_counts.get("unknown_brand_audit_only", 0)),
            "no_manual_review_reason_customers": int(reason_combo_counts.get("no_manual_review_reason", 0)),
            "identity_risk_customers": int(customer_class_counts.get("identity_risk", 0)),
            "source_label_only_customers": int(customer_class_counts.get("source_label_only", 0)),
            "unknown_brand_customers": int(brand_counts.get("unknown", 0)),
            "shared_amo_contact_customers": int(reason_counts.get("shared_amo_contact_across_customers", 0)),
            "shared_amo_lead_customers": int(reason_counts.get("shared_amo_lead_across_customers", 0)),
            "duplicate_amo_contact_ids": len(duplicate_contact_ids),
            "duplicate_amo_lead_ids": len(duplicate_lead_ids),
            "timeline_preview_enabled_allowed": False,
            "timeline_primary_read_enabled_allowed": False,
        },
        "customer_class_counts": dict(customer_class_counts),
        "reason_counts": dict(reason_counts),
        "reason_class_counts": dict(reason_class_counts),
        "top_reason_combinations": top_counter_items(reason_combo_counts, limit=15),
        "brand_counts": dict(brand_counts),
        "unknown_brand_analysis": {
            "reason_counts": dict(unknown_brand_reason_counts),
            "amo_hint_counts": dict(unknown_brand_amo_hint_counts),
            "interpretation": (
                "Most unknown-brand rows do not carry a reliable Foton/UNPK token in the master contact row. "
                "AMO hints can help triage, but must not promote brand automatically without review."
            ),
        },
        "shared_amo_analysis": {
            "duplicate_contact_ids": len(duplicate_contact_ids),
            "affected_contact_customers": int(reason_counts.get("shared_amo_contact_across_customers", 0)),
            "duplicate_lead_ids": len(duplicate_lead_ids),
            "affected_lead_customers": int(reason_counts.get("shared_amo_lead_across_customers", 0)),
            "decision": "identity_risk_manual_review_only",
            "safe_behavior": "do_not_merge_customers_by_shared_amo_id",
        },
        "context_by_class": {
            "mail_context": dict(mail_context_by_class),
            "amo_context": dict(amo_context_by_class),
        },
        "preview_plan": build_preview_plan(
            total=total,
            preview_candidates_by_brand=preview_candidates_by_brand,
            identity_risk_by_brand=identity_risk_by_brand,
            source_label_only_by_brand=source_label_only_by_brand,
            unknown_brand_count=int(brand_counts.get("unknown", 0)),
        ),
        "recommended_actions": recommended_actions(),
        "store_counts": dict(store_summary.get("counts", {})),
        "source_manifest_summary": compact_source_manifest(source_manifest),
        "safety": triage_safety_contract(),
        "semantic_status": {
            "formal_pass_required": True,
            "semantic_pass_required_before_preview": True,
            "pilot_ready": False,
            "production_ready": False,
        },
    }
    write_json(out_dir / "manual_review_triage_report.json", report)
    (out_dir / "manual_review_triage_report.md").write_text(render_triage_markdown(report), encoding="utf-8")
    (out_dir / "manager_preview_plan.md").write_text(render_manager_preview_plan(report), encoding="utf-8")
    return report


def resolve_triage_config(config: CanonicalReadonlyTriageConfig) -> CanonicalReadonlyTriageConfig:
    project_root = Path(config.project_root).expanduser().resolve(strict=False)
    timeline_root = guard_customer_timeline_output_path(
        project_root / config.timeline_root if not config.timeline_root.is_absolute() else config.timeline_root,
        project_root,
    )
    timeline_db = timeline_root / "customer_timeline.sqlite"
    if not timeline_db.exists():
        raise FileNotFoundError(timeline_db)
    out_dir = config.out_dir
    if out_dir is not None:
        out_dir = guard_customer_timeline_output_path(
            project_root / out_dir if not out_dir.is_absolute() else out_dir,
            timeline_root,
        )
    return CanonicalReadonlyTriageConfig(
        project_root=project_root,
        timeline_root=timeline_root,
        out_dir=out_dir,
        current_runtime_json=config.current_runtime_json,
        master_contacts_csv=config.master_contacts_csv,
        master_calls_csv=config.master_calls_csv,
        amo_contacts_csv=config.amo_contacts_csv,
        amo_deals_csv=config.amo_deals_csv,
        mail_handoff_db=config.mail_handoff_db,
        mail_bridge_db=config.mail_bridge_db,
        generated_at=config.generated_at,
    )


def classify_reason(reason: str) -> str:
    if reason in IDENTITY_RISK_REASONS:
        return "identity_risk"
    if reason in SOURCE_LABEL_REASONS:
        return "source_label"
    if reason in DATA_GAP_REASONS:
        return "data_gap"
    return "other"


def classify_customer_for_triage(*, brand: str, reasons: Sequence[str]) -> str:
    reason_set = set(reasons)
    if reason_set & IDENTITY_RISK_REASONS:
        return "identity_risk"
    if reason_set == SOURCE_LABEL_REASONS:
        return "source_label_only"
    if reason_set:
        return "data_gap_or_other"
    if brand == "unknown":
        return "unknown_brand_audit_only"
    return "manager_preview_candidate"


def unknown_brand_reason(
    *,
    row: Mapping[str, str],
    phone: str,
    amo_contacts_by_phone: Mapping[str, Sequence[Mapping[str, str]]],
    amo_deals_by_contact_id: Mapping[str, Sequence[Mapping[str, str]]],
) -> str:
    if infer_brand(row.values()) != "unknown":
        return "known_brand"
    amo_hint = brand_hint_from_amo(phone, amo_contacts_by_phone, amo_deals_by_contact_id)
    if amo_hint != "unknown":
        return "amo_snapshot_has_brand_hint_but_master_contact_is_unknown"
    if phone not in amo_contacts_by_phone:
        return "no_brand_token_and_no_amo_context"
    return "no_brand_token_in_master_contact_or_amo_snapshot"


def brand_hint_from_amo(
    phone: str,
    amo_contacts_by_phone: Mapping[str, Sequence[Mapping[str, str]]],
    amo_deals_by_contact_id: Mapping[str, Sequence[Mapping[str, str]]],
) -> str:
    hints: set[str] = set()
    for contact in amo_contacts_by_phone.get(phone, ()):
        contact_id = str(contact.get("contact_id") or "")
        contact_brand = infer_brand(contact.values())
        if contact_brand != "unknown":
            hints.add(contact_brand)
        for deal in amo_deals_by_contact_id.get(contact_id, ()):
            deal_brand = infer_brand(deal.values())
            if deal_brand != "unknown":
                hints.add(deal_brand)
    if len(hints) == 1:
        return next(iter(hints))
    if len(hints) > 1:
        return "mixed"
    return "unknown"


def build_preview_plan(
    *,
    total: int,
    preview_candidates_by_brand: Mapping[str, int],
    identity_risk_by_brand: Mapping[str, int],
    source_label_only_by_brand: Mapping[str, int],
    unknown_brand_count: int,
) -> Mapping[str, Any]:
    preview_total = sum(preview_candidates_by_brand.values())
    clean_smoke_ready = preview_total >= 20
    return {
        "decision": "manager_preview_only_not_bot",
        "timeline_preview_enabled_allowed": False,
        "timeline_primary_read_enabled_allowed": False,
        "eligible_manager_preview_candidates": preview_total,
        "eligible_manager_preview_ratio": ratio(preview_total, total),
        "minimum_clean_smoke_size": 20,
        "clean_smoke_batch_status": "ready" if clean_smoke_ready else "blocked_insufficient_clean_candidates",
        "candidate_counts_by_brand": dict(preview_candidates_by_brand),
        "recommended_batches": [
            {
                "name": "known_brand_clean_smoke",
                "target_size": min(50, preview_total) if clean_smoke_ready else 0,
                "purpose": "Проверить, что менеджер видит цельную историю без спорных склеек.",
                "selection_rule": "brand in foton/unpk, no manual review reasons, no shared AMO risk.",
                "allowed_use": "internal_manager_read_only",
                "status": "ready" if clean_smoke_ready else "blocked_until_enough_clean_known_brand_cases",
            },
            {
                "name": "shared_amo_identity_review",
                "target_size": min(40, sum(identity_risk_by_brand.values())),
                "purpose": "Разобрать общие AMO contact/lead и понять, где семья, дубль или ошибка CRM.",
                "selection_rule": "has shared_amo_contact_across_customers or shared_amo_lead_across_customers.",
                "allowed_use": "internal_identity_review_only",
            },
            {
                "name": "unknown_brand_calibration",
                "target_size": min(50, unknown_brand_count),
                "purpose": "Понять, какие поля надежно восстанавливают бренд.",
                "selection_rule": "brand=unknown, no automatic client-facing use.",
                "allowed_use": "internal_brand_calibration_only",
            },
            {
                "name": "source_manual_label_calibration",
                "target_size": min(50, sum(source_label_only_by_brand.values())),
                "purpose": "Проверить, почему старый источник массово ставит ручную проверку.",
                "selection_rule": "only source_manual_review_required and no identity risk.",
                "allowed_use": "internal_rule_calibration_only",
            },
        ],
        "hard_rules": [
            "Не включать auto-answer.",
            "Не включать timeline_primary_read_enabled.",
            "Не писать в AMO/Tallanto/CRM.",
            "Не показывать клиентам raw history.",
            "Любой shared AMO case остается manual review.",
            "Строки без manual-review причины, но с brand=unknown, не считаются clean preview.",
        ],
    }


def recommended_actions() -> list[Mapping[str, str]]:
    return [
        {
            "priority": "P0",
            "action": "Разобрать source_manual_review_required",
            "why": "Это главный объем ручной проверки, но часть может быть унаследованным слишком строгим флагом.",
        },
        {
            "priority": "P0",
            "action": "Сделать read-only brand enrichment",
            "why": "83% клиентов имеют brand=unknown, поэтому слой нельзя безопасно давать брендовому боту.",
        },
        {
            "priority": "P0",
            "action": "Разобрать shared AMO clusters",
            "why": "Общие contact/lead ID могут склеить разных клиентов или семейные связи.",
        },
        {
            "priority": "P1",
            "action": "Обновить AMO/mail snapshots перед preview",
            "why": "Текущие snapshot устаревают относительно runtime.",
        },
        {
            "priority": "P1",
            "action": "Сформировать локальные private sample files для менеджера",
            "why": "Агрегатов достаточно для решения, но проверка качества требует ручного просмотра конкретных кейсов.",
        },
    ]


def read_store_summary(timeline_db: Path, allowed_root: Path) -> Mapping[str, Any]:
    store = CustomerTimelineSQLiteStore.open_read_only(timeline_db, allowed_root=allowed_root)
    try:
        return store.summary()
    finally:
        store.close()


def compact_source_manifest(source_manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: {
            "exists": value.get("exists"),
            "size_bytes": value.get("size_bytes"),
            "row_count": value.get("row_count"),
        }
        for key, value in source_manifest.items()
        if isinstance(value, Mapping)
    }


def reason_combo_key(reasons: Sequence[str]) -> str:
    return "|".join(sorted(set(reasons))) if reasons else "no_manual_review_reason"


def top_counter_items(counter: Counter[str], *, limit: int) -> list[Mapping[str, Any]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def triage_safety_contract() -> Mapping[str, Any]:
    return {
        "read_only_source_systems": True,
        "write_crm": False,
        "write_tallanto": False,
        "send_email": False,
        "send_messenger": False,
        "run_asr": False,
        "run_ra": False,
        "stable_runtime_writes": False,
        "telegram_import_enabled": False,
        "timeline_preview_enabled_allowed": False,
        "timeline_primary_read_enabled_allowed": False,
        "raw_personal_values_in_reports": False,
    }


def render_triage_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Manual Review Triage",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- total_customers: `{summary['total_customers']}`",
        f"- manager_preview_candidate_count: `{summary['manager_preview_candidate_count']}`",
        f"- unknown_brand_audit_only_count: `{summary['unknown_brand_audit_only_count']}`",
        f"- no_manual_review_reason_customers: `{summary['no_manual_review_reason_customers']}`",
        f"- identity_risk_customers: `{summary['identity_risk_customers']}`",
        f"- source_label_only_customers: `{summary['source_label_only_customers']}`",
        f"- unknown_brand_customers: `{summary['unknown_brand_customers']}`",
        f"- shared_amo_contact_customers: `{summary['shared_amo_contact_customers']}`",
        f"- shared_amo_lead_customers: `{summary['shared_amo_lead_customers']}`",
        "",
        "## Customer Classes",
    ]
    for key, value in sorted(report.get("customer_class_counts", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Manual Review Reasons"])
    for key, value in sorted(report.get("reason_counts", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- `tenant_id` is a technical export container; customer brand is tracked in `brand_counts`.",
            "- `no_manual_review_reason` is not equal to preview-ready: unknown-brand rows stay audit-only.",
            "- Clean manager smoke is blocked until there are at least 20 known-brand clean cases.",
        ]
    )
    lines.extend(["", "## Unknown Brand"])
    for key, value in sorted(report.get("unknown_brand_analysis", {}).get("reason_counts", {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Recommended Actions"])
    for item in report.get("recommended_actions", ()):
        lines.append(f"- `{item['priority']}` `{item['action']}`: {item['why']}")
    lines.extend(["", "## Decision", "- Manager preview only after private sample creation and human approval.", "- Bot preview and primary read remain disabled."])
    return "\n".join(lines) + "\n"


def render_manager_preview_plan(report: Mapping[str, Any]) -> str:
    plan = report["preview_plan"]
    lines = [
        "# Manager Preview Plan",
        "",
        f"- decision: `{plan['decision']}`",
        f"- eligible_manager_preview_candidates: `{plan['eligible_manager_preview_candidates']}`",
        f"- eligible_manager_preview_ratio: `{plan['eligible_manager_preview_ratio']}`",
        f"- clean_smoke_batch_status: `{plan['clean_smoke_batch_status']}`",
        f"- minimum_clean_smoke_size: `{plan['minimum_clean_smoke_size']}`",
        "",
        "## Batches",
    ]
    for batch in plan.get("recommended_batches", ()):
        lines.extend(
            [
                f"- `{batch['name']}`",
                f"  target_size: `{batch['target_size']}`",
                f"  purpose: {batch['purpose']}",
                f"  selection_rule: {batch['selection_rule']}",
                f"  allowed_use: `{batch['allowed_use']}`",
                f"  status: `{batch.get('status', 'ready')}`",
            ]
        )
    lines.extend(["", "## Hard Rules"])
    for rule in plan.get("hard_rules", ()):
        lines.append(f"- {rule}")
    return "\n".join(lines) + "\n"


__all__ = [
    "CANONICAL_READONLY_TRIAGE_SCHEMA_VERSION",
    "CanonicalReadonlyTriageConfig",
    "build_canonical_readonly_timeline_triage",
    "build_preview_plan",
    "classify_customer_for_triage",
    "classify_reason",
    "triage_safety_contract",
]
