from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mango_mvp.amocrm_runtime.amo_integration import (
    AmoAccessContext,
    _amo_http_request,
    _contact_entity_endpoint,
    _contacts_custom_fields_endpoint,
    _flatten_contact_field_item,
    _flatten_lead_field_item,
    _follow_next_link,
    _lead_custom_fields_endpoint,
    _lead_entity_endpoint,
    _normalize_base_url,
    _token_is_stale,
    get_active_connection,
    settings,
)
from mango_mvp.crm_card_aggregator import build_crm_card_projection
from mango_mvp.crm_card_history_summary import CrmHistorySummarizer
from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi
from mango_mvp.deal_aware.amo_rollback import (
    append_snapshot_rows,
    build_pre_write_snapshot_rows,
    extract_custom_field_values,
    sha256_text,
    write_rollback_manifest,
)
from mango_mvp.deal_aware.amo_write_safety import (
    allowed_payload_after_pre_patch,
    append_write_journal_rows,
    blocking_pre_patch_reasons,
    journal_rows_from_decisions,
    pre_patch_write_decisions,
)
from mango_mvp.deal_aware.deal_writeback import brand_writeback_guard_reason, multiple_open_deals_guard_reason
from mango_mvp.quality.crm_text_quality_detector import detect_crm_text_quality_risks


CRM_CARD_AMO_DRY_RUN_SCHEMA_VERSION = "crm_card_amo_writeback_dry_run_v1"
LIVE_CONFIRMATION = "WRITE_AMO_LIVE"
CONTACT_TARGET_FIELDS = (
    "AI-рекомендованный следующий шаг",
    "Последняя AI-сводка",
    "Авто история общения",
)
DEAL_TARGET_FIELDS = (
    "AI-рекомендованный следующий шаг",
    "AI-сводка по сделке",
    "AI-история по сделке",
)
TEXTAREA_TARGET_FIELDS = set(CONTACT_TARGET_FIELDS) | set(DEAL_TARGET_FIELDS)
CLOSED_STATUS_MARKERS = ("закрыто", "успешно")
UNKNOWN_BRANDS = {"", "unknown", "неизвестно", "нет", "none", "null"}


@dataclass(frozen=True)
class CrmCardAmoDryRunConfig:
    timeline_db: Path
    allowed_root: Path
    out_dir: Path
    tenant_id: str = "foton"
    sample_size: int = 5
    customer_ids: tuple[str, ...] = ()
    history_summarizer: CrmHistorySummarizer | None = None


@dataclass(frozen=True)
class AmoReadOnlyClient:
    context: AmoAccessContext
    http_get: Callable[[str], Mapping[str, Any]]

    def fetch_contact(self, contact_id: str | int) -> Mapping[str, Any]:
        return self.http_get(_contact_entity_endpoint(self.context.account_base_url, int(contact_id)))

    def fetch_lead(self, lead_id: str | int) -> Mapping[str, Any]:
        return self.http_get(_lead_entity_endpoint(self.context.account_base_url, int(lead_id)))

    def fetch_contact_field_catalog(self) -> list[dict[str, Any]]:
        return fetch_field_catalog_read_only(
            self.context,
            "contact",
            http_get=self.http_get,
        )

    def fetch_lead_field_catalog(self) -> list[dict[str, Any]]:
        return fetch_field_catalog_read_only(
            self.context,
            "lead",
            http_get=self.http_get,
        )


def resolve_amo_access_context_no_refresh(session: Any) -> AmoAccessContext:
    env_token = str(settings.crm_amo_api_token or "").strip()
    env_base = _normalize_base_url(settings.crm_amo_base_url)
    if env_token and env_base:
        return AmoAccessContext(
            account_base_url=env_base,
            access_token=env_token,
            token_source="env",
            connection=None,
        )
    try:
        connection = get_active_connection(session)
    except Exception as exc:  # noqa: BLE001 - keep dry-run blocker readable.
        raise RuntimeError("AMO read-only dry-run blocked: runtime DB has no usable amo_integration_connections table.") from exc
    if connection is None:
        raise RuntimeError("AMO read-only dry-run blocked: active OAuth connection is missing.")
    if _token_is_stale(connection):
        raise RuntimeError("AMO read-only dry-run blocked: OAuth token is stale; refusing token refresh in dry-run.")
    if not connection.access_token or not connection.account_base_url:
        raise RuntimeError("AMO read-only dry-run blocked: OAuth connection has no access token/base URL.")
    return AmoAccessContext(
        account_base_url=connection.account_base_url,
        access_token=connection.access_token,
        token_source="oauth_no_refresh",
        connection=connection,
    )


def amo_client_from_context(context: AmoAccessContext) -> AmoReadOnlyClient:
    def http_get(url: str) -> Mapping[str, Any]:
        return _amo_http_request(
            method="GET",
            url=url,
            headers={"Authorization": f"Bearer {context.access_token}"},
        )

    return AmoReadOnlyClient(context=context, http_get=http_get)


def fetch_field_catalog_read_only(
    context: AmoAccessContext,
    entity_type: str,
    *,
    http_get: Callable[[str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    getter = http_get or (lambda url: _amo_http_request(method="GET", url=url, headers={"Authorization": f"Bearer {context.access_token}"}))
    if entity_type == "contact":
        next_url = _contacts_custom_fields_endpoint(context.account_base_url)
        embedded_key = "custom_fields"
        flattener = _flatten_contact_field_item
    elif entity_type == "lead":
        next_url = _lead_custom_fields_endpoint(context.account_base_url)
        embedded_key = "custom_fields"
        flattener = _flatten_lead_field_item
    else:
        raise ValueError(f"unsupported AMO catalog entity_type: {entity_type}")
    fields: list[dict[str, Any]] = []
    while next_url:
        payload = getter(next_url)
        embedded = payload.get("_embedded") if isinstance(payload, Mapping) else {}
        items = embedded.get(embedded_key) if isinstance(embedded, Mapping) else []
        if isinstance(items, Sequence):
            fields.extend(flattener(dict(item)) for item in items if isinstance(item, Mapping))
        links = payload.get("_links") if isinstance(payload, Mapping) else {}
        next_link = None
        if isinstance(links, Mapping):
            next_meta = links.get("next")
            if isinstance(next_meta, Mapping):
                next_link = next_meta.get("href")
        next_url = _follow_next_link(context.account_base_url, str(next_link)) if next_link else ""
    return fields


def build_crm_card_amo_payloads(projection: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    contact_card = _mapping(projection.get("contact_card"))
    deal_card = _mapping(projection.get("deal_card"))
    contact_fields = _mapping(contact_card.get("fields"))
    deal_fields = _mapping(deal_card.get("fields"))
    next_step = _safe_text(deal_fields.get("Следующий шаг"))
    latest_summary = _safe_text(contact_fields.get("Последняя сводка"))
    history = _safe_text(contact_fields.get("История общения"))
    contact_payload = _drop_empty(
        {
            "AI-рекомендованный следующий шаг": next_step,
            "Последняя AI-сводка": latest_summary,
            "Авто история общения": history,
        }
    )
    deal_payload = _drop_empty(
        {
            "AI-рекомендованный следующий шаг": next_step,
            "AI-сводка по сделке": latest_summary,
            "AI-история по сделке": history,
        }
    )
    return contact_payload, deal_payload


def select_customer_ids_for_amo_dry_run(
    api: CustomerTimelineReadApi,
    tenant_id: str,
    *,
    sample_size: int,
    timeline_db: Path | None = None,
) -> list[str]:
    if timeline_db is not None:
        db_result = select_customer_ids_for_amo_dry_run_from_db(
            timeline_db,
            tenant_id=tenant_id,
            sample_size=sample_size,
        )
        if db_result:
            return db_result
    result: list[str] = []
    seen: set[str] = set()
    cursor: str | None = None
    scanned = 0
    while len(result) < sample_size and scanned < 200:
        page = api.list_customers(tenant_id, limit=100, cursor=cursor)
        scanned += 1
        for item in page.get("items") or ():
            customer_id = _safe_text(item.get("customer_id"))
            if not customer_id or customer_id in seen:
                continue
            profile = api.customer_profile(tenant_id, customer_id, event_limit=50, bot_context_limit=1)
            contact_id = selected_contact_id(profile)
            open_opportunities = open_amo_opportunities(profile)
            if contact_id.isdigit() and len(open_opportunities) == 1 and normalized_brand(opportunity_brand(open_opportunities[0])):
                seen.add(customer_id)
                result.append(customer_id)
                if len(result) >= sample_size:
                    break
        cursor = _safe_text(page.get("next_cursor")) or None
        if not cursor:
            break
    return result


def build_crm_card_amo_dry_run(
    config: CrmCardAmoDryRunConfig,
    *,
    amo_client: AmoReadOnlyClient,
) -> Mapping[str, Any]:
    out_dir = config.out_dir.expanduser().resolve(strict=False)
    if "stable_runtime" in out_dir.parts:
        raise ValueError("CRM card AMO dry-run output must not be written under stable_runtime")
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_id = out_dir.name
    input_manifest = out_dir / "dry_run_input_manifest.json"
    journal_path = out_dir / "write_journal_dry_run.jsonl"
    contact_catalog = amo_client.fetch_contact_field_catalog()
    lead_catalog = amo_client.fetch_lead_field_catalog()
    contact_catalog_guard = field_catalog_guard(contact_catalog, CONTACT_TARGET_FIELDS)
    lead_catalog_guard = field_catalog_guard(lead_catalog, DEAL_TARGET_FIELDS)
    read_config = {
        "timeline_db": str(config.timeline_db),
        "allowed_root": str(config.allowed_root),
        "tenant_id": config.tenant_id,
        "sample_size": config.sample_size,
        "customer_ids_requested": list(config.customer_ids),
    }
    input_manifest.write_text(json.dumps(read_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    input_sha = sha256_file(input_manifest)
    write_rollback_manifest(
        out_dir,
        batch_id=batch_id,
        input_csv=input_manifest,
        input_sha256=input_sha,
        field_catalog_cache=out_dir / "amo_field_catalogs.json",
        operator_approval_path=None,
    )
    (out_dir / "amo_field_catalogs.json").write_text(
        json.dumps({"contact": contact_catalog, "lead": lead_catalog}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    with _open_api(config) as api:
        customer_ids = list(config.customer_ids) or select_customer_ids_for_amo_dry_run(
            api,
            config.tenant_id,
            sample_size=max(1, min(int(config.sample_size), 5)),
            timeline_db=config.timeline_db,
        )
        for index, customer_id in enumerate(customer_ids, start=1):
            profile = api.customer_profile(config.tenant_id, customer_id, event_limit=50, bot_context_limit=1)
            raw_opportunities = raw_amo_opportunities_for_customer(
                config.timeline_db,
                tenant_id=config.tenant_id,
                customer_id=customer_id,
            )
            projection = build_crm_card_projection(
                profile,
                history_summarizer=config.history_summarizer,
            )
            profile_rows, profile_findings = dry_run_profile(
                profile=profile,
                projection=projection,
                raw_opportunities=raw_opportunities,
                amo_client=amo_client,
                contact_catalog=contact_catalog,
                lead_catalog=lead_catalog,
                contact_catalog_guard=contact_catalog_guard,
                lead_catalog_guard=lead_catalog_guard,
                row_index_base=index,
                batch_id=batch_id,
                input_manifest=input_manifest,
                input_sha256=input_sha,
                out_dir=out_dir,
                journal_path=journal_path,
            )
            rows.extend(profile_rows)
            findings.extend(profile_findings)
    write_csv(out_dir / "crm_card_amo_dry_run_report.csv", rows)
    write_csv(out_dir / "crm_card_amo_dry_run_findings.csv", findings)
    summary = dry_run_summary(
        rows,
        findings,
        config=config,
        out_dir=out_dir,
        contact_catalog_guard=contact_catalog_guard,
        lead_catalog_guard=lead_catalog_guard,
        amo_token_source=amo_client.context.token_source,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _open_api(config: CrmCardAmoDryRunConfig) -> CustomerTimelineReadApi:
    from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApiConfig

    return CustomerTimelineReadApi.open(
        CustomerTimelineReadApiConfig(timeline_db=config.timeline_db, allowed_root=config.allowed_root)
    )


def dry_run_profile(
    *,
    profile: Mapping[str, Any],
    projection: Mapping[str, Any],
    raw_opportunities: Sequence[Mapping[str, Any]] = (),
    amo_client: AmoReadOnlyClient,
    contact_catalog: list[dict[str, Any]],
    lead_catalog: list[dict[str, Any]],
    contact_catalog_guard: list[str],
    lead_catalog_guard: list[str],
    row_index_base: int,
    batch_id: str,
    input_manifest: Path,
    input_sha256: str,
    out_dir: Path,
    journal_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    customer_id = _safe_text(profile.get("customer_id"))
    contact_payload, deal_payload = build_crm_card_amo_payloads(projection)
    contact_id = selected_contact_id(profile)
    if contact_id and contact_id.isdigit():
        contact_row, contact_findings = dry_run_entity(
            entity_type="contact",
            entity_id=contact_id,
            customer_id=customer_id,
            payload=contact_payload,
            current_entity=amo_client.fetch_contact(contact_id),
            fresh_entity=amo_client.fetch_contact(contact_id),
            field_catalog=contact_catalog,
            catalog_guard=contact_catalog_guard,
            row_index=row_index_base * 100,
            batch_id=batch_id,
            input_manifest=input_manifest,
            input_sha256=input_sha256,
            out_dir=out_dir,
            journal_path=journal_path,
            guard_input={},
        )
        rows.append(contact_row)
        findings.extend(contact_findings)
    else:
        row = base_report_row(customer_id=customer_id, entity_type="contact", entity_id=contact_id, payload=contact_payload)
        row.update({"status": "blocked", "reason": "missing_amo_contact_id", "ready_for_write": "no"})
        rows.append(row)
        findings.append(finding(customer_id, "contact", contact_id, "missing_amo_contact_id", "P1"))
    open_opportunities = open_amo_opportunities(profile, raw_opportunities=raw_opportunities)
    open_count = len(open_opportunities)
    if not open_opportunities:
        row = base_report_row(customer_id=customer_id, entity_type="lead", entity_id="", payload=deal_payload)
        row.update({"status": "blocked", "reason": "missing_open_amo_deal", "ready_for_write": "no"})
        rows.append(row)
        findings.append(finding(customer_id, "lead", "", "missing_open_amo_deal", "P1"))
    for deal_index, opportunity in enumerate(open_opportunities, start=1):
        lead_id = _safe_text(opportunity.get("source_id"))
        active_brand = resolve_active_brand(profile, opportunity)
        deal_brand = normalized_brand(opportunity_brand(opportunity))
        guard_input = {
            "active_brand": active_brand,
            "deal_brand": deal_brand,
            "open_deal_count": str(open_count),
            "selected_deal_id": lead_id,
        }
        guard_reasons = deal_guard_reasons(guard_input)
        if guard_reasons or not lead_id.isdigit():
            reason = " | ".join(guard_reasons or ["missing_selected_deal_id"])
            row = base_report_row(customer_id=customer_id, entity_type="lead", entity_id=lead_id, payload=deal_payload)
            row.update(
                {
                    "status": "blocked",
                    "reason": reason,
                    "ready_for_write": "no",
                    "active_brand": active_brand,
                    "deal_brand": deal_brand,
                    "open_deal_count": str(open_count),
                    "guard_input_populated": "yes" if active_brand and str(open_count) else "no",
                }
            )
            rows.append(row)
            findings.extend(finding(customer_id, "lead", lead_id, item, "P1") for item in reason.split(" | "))
            continue
        deal_row, deal_findings = dry_run_entity(
            entity_type="lead",
            entity_id=lead_id,
            customer_id=customer_id,
            payload=deal_payload,
            current_entity=amo_client.fetch_lead(lead_id),
            fresh_entity=amo_client.fetch_lead(lead_id),
            field_catalog=lead_catalog,
            catalog_guard=lead_catalog_guard,
            row_index=row_index_base * 100 + deal_index,
            batch_id=batch_id,
            input_manifest=input_manifest,
            input_sha256=input_sha256,
            out_dir=out_dir,
            journal_path=journal_path,
            guard_input=guard_input,
        )
        rows.append(deal_row)
        findings.extend(deal_findings)
    return rows, findings


def dry_run_entity(
    *,
    entity_type: str,
    entity_id: str,
    customer_id: str,
    payload: Mapping[str, Any],
    current_entity: Mapping[str, Any],
    fresh_entity: Mapping[str, Any],
    field_catalog: list[dict[str, Any]],
    catalog_guard: list[str],
    row_index: int,
    batch_id: str,
    input_manifest: Path,
    input_sha256: str,
    out_dir: Path,
    journal_path: Path,
    guard_input: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    safe_payload = _drop_empty({str(key): _safe_text(value) for key, value in payload.items()})
    row = base_report_row(customer_id=customer_id, entity_type=entity_type, entity_id=entity_id, payload=safe_payload)
    row.update({key: _safe_text(value) for key, value in guard_input.items()})
    row["guard_input_populated"] = "yes" if entity_type == "contact" or (_safe_text(guard_input.get("active_brand")) and _safe_text(guard_input.get("open_deal_count"))) else "no"
    row_findings: list[dict[str, Any]] = []
    if not safe_payload:
        row.update({"status": "blocked", "reason": "empty_payload", "ready_for_write": "no"})
        row_findings.append(finding(customer_id, entity_type, entity_id, "empty_payload", "P1"))
        return row, row_findings
    if catalog_guard:
        row.update({"status": "blocked", "reason": "field_catalog_guard:" + " | ".join(catalog_guard), "ready_for_write": "no"})
        row_findings.extend(finding(customer_id, entity_type, entity_id, item, "P1") for item in catalog_guard)
        return row, row_findings
    text_findings = detect_crm_text_quality_risks(dict(safe_payload), min_severity="P2")
    if text_findings:
        row.update(
            {
                "status": "blocked",
                "reason": "crm_text_quality:" + " | ".join(sorted({item.risk_type for item in text_findings})),
                "ready_for_write": "no",
            }
        )
        row_findings.extend(
            finding(customer_id, entity_type, entity_id, item.risk_type, item.severity, field=item.field)
            for item in text_findings
        )
        return row, row_findings
    snapshot_rows = build_pre_write_snapshot_rows(
        batch_id=batch_id,
        input_csv=input_manifest,
        input_sha256=input_sha256,
        row_index=row_index,
        review_id=f"{customer_id}:{entity_type}:{entity_id}",
        lead_id=entity_id,
        payload=dict(safe_payload),
        current_lead=dict(current_entity),
        field_catalog=field_catalog,
        operator_approval_path=None,
        entity_type=entity_type,
        entity_id=entity_id,
    )
    append_snapshot_rows(out_dir, snapshot_rows)
    decisions = pre_patch_write_decisions(snapshot_rows=snapshot_rows, current_entity=fresh_entity)
    allowed_payload = allowed_payload_after_pre_patch(safe_payload, decisions)
    blocked_reasons = blocking_pre_patch_reasons(decisions)
    append_write_journal_rows(
        journal_path,
        journal_rows_from_decisions(
            decisions,
            action_for_allowed="written-dry",
            reason_for_allowed="dry_run_no_patch",
            snapshot_path=out_dir / "pre_write_snapshot.jsonl",
            contact_id=entity_id if entity_type == "contact" else "",
            deal_id=entity_id if entity_type == "lead" else "",
        ),
    )
    row.update(
        {
            "status": "dry_run" if allowed_payload else "blocked",
            "reason": "live_write_not_confirmed" if allowed_payload else "pre_patch_guard:" + " | ".join(blocked_reasons),
            "ready_for_write": "yes" if allowed_payload else "no",
            "snapshot_status": "saved",
            "snapshot_rows": str(len(snapshot_rows)),
            "pre_patch_status": "allowed" if allowed_payload else "blocked",
            "pre_patch_reasons": " | ".join(blocked_reasons),
            "allowed_fields": " | ".join(allowed_payload),
            "current_values_json": json.dumps(current_field_subset(current_entity, safe_payload), ensure_ascii=False, sort_keys=True),
            "new_values_json": json.dumps(safe_payload, ensure_ascii=False, sort_keys=True),
            "diff_json": json.dumps(diff_payload(current_entity, safe_payload), ensure_ascii=False, sort_keys=True),
            "payload_sha256": payload_sha256(safe_payload),
        }
    )
    if not allowed_payload:
        row_findings.extend(finding(customer_id, entity_type, entity_id, item, "P1") for item in blocked_reasons)
    return row, row_findings


def field_catalog_guard(field_catalog: Sequence[Mapping[str, Any]], target_fields: Sequence[str]) -> list[str]:
    reasons: list[str] = []
    by_name = {str(item.get("name") or "").strip(): item for item in field_catalog if str(item.get("name") or "").strip()}
    for field_name in target_fields:
        meta = by_name.get(field_name)
        if meta is None or meta.get("id") is None:
            reasons.append(f"missing_amo_field:{field_name}")
            continue
        if field_name in TEXTAREA_TARGET_FIELDS and _safe_text(meta.get("type")).casefold() != "textarea":
            reasons.append(f"amo_field_not_textarea:{field_name}:{_safe_text(meta.get('type')) or '<missing>'}")
        if bool(meta.get("is_api_only")):
            reasons.append(f"amo_field_api_only:{field_name}")
    return reasons


def deal_guard_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _safe_text(row.get("active_brand")):
        reasons.append("missing_active_brand")
    if not _safe_text(row.get("open_deal_count")):
        reasons.append("missing_open_deal_count")
    brand_reason = brand_writeback_guard_reason(dict(row))
    if brand_reason:
        reasons.append(brand_reason)
    open_deals_reason = multiple_open_deals_guard_reason(dict(row))
    if open_deals_reason:
        reasons.append(open_deals_reason)
    return reasons


def open_amo_opportunities(
    profile: Mapping[str, Any],
    *,
    raw_opportunities: Sequence[Mapping[str, Any]] = (),
) -> list[Mapping[str, Any]]:
    raw_items = [
        _mapping(item)
        for item in raw_opportunities
        if _safe_text(_mapping(item).get("source_system")) == "amocrm_snapshot"
        and _safe_text(_mapping(item).get("source_id")).isdigit()
        and opportunity_is_open(_mapping(item))
    ]
    if raw_items:
        return sorted(raw_items, key=lambda item: (_safe_text(item.get("opened_at")), _safe_text(item.get("source_id"))), reverse=True)
    manager_projection = _mapping(profile.get("manager_projection"))
    opportunities = [_mapping(item) for item in _sequence(manager_projection.get("opportunities"))]
    if not opportunities:
        opportunities = [_mapping(item) for item in _sequence(profile.get("opportunities"))]
    result = [
        item
        for item in opportunities
        if _safe_text(item.get("source_system")) == "amocrm_snapshot"
        and _safe_text(item.get("source_id")).isdigit()
        and opportunity_is_open(item)
    ]
    return sorted(result, key=lambda item: (_safe_text(item.get("opened_at")), _safe_text(item.get("source_id"))), reverse=True)


def raw_amo_opportunities_for_customer(
    timeline_db: Path,
    *,
    tenant_id: str,
    customer_id: str,
) -> list[Mapping[str, Any]]:
    uri = f"file:{timeline_db.expanduser().resolve(strict=False)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            """
            SELECT record_json
            FROM customer_opportunities
            WHERE tenant_id = ? AND customer_id = ? AND source_system = 'amocrm_snapshot'
            ORDER BY opened_at DESC, opportunity_id
            """,
            (tenant_id, customer_id),
        ).fetchall()
    finally:
        con.close()
    result: list[Mapping[str, Any]] = []
    for (raw_json,) in rows:
        try:
            payload = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            result.append(payload)
    return result


def select_customer_ids_for_amo_dry_run_from_db(
    timeline_db: Path,
    *,
    tenant_id: str,
    sample_size: int,
) -> list[str]:
    uri = f"file:{timeline_db.expanduser().resolve(strict=False)}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            """
            WITH open_deals AS (
              SELECT customer_id, COUNT(*) AS open_count
              FROM customer_opportunities
              WHERE tenant_id = ?
                AND source_system = 'amocrm_snapshot'
                AND COALESCE(closed_at, '') = ''
              GROUP BY customer_id
            ),
            contacts AS (
              SELECT customer_id, MIN(link_value) AS contact_id
              FROM identity_links
              WHERE tenant_id = ?
                AND link_type = 'amo_contact_id'
                AND match_class = 'strong_unique'
              GROUP BY customer_id
            )
            SELECT o.customer_id
            FROM open_deals o
            JOIN contacts c ON c.customer_id = o.customer_id
            JOIN customer_opportunities co
              ON co.tenant_id = ?
             AND co.customer_id = o.customer_id
             AND co.source_system = 'amocrm_snapshot'
             AND COALESCE(co.closed_at, '') = ''
            WHERE o.open_count = 1
              AND lower(COALESCE(json_extract(co.record_json, '$.product_context.brand'), '')) IN ('foton', 'unpk')
            ORDER BY o.customer_id
            LIMIT ?
            """,
            (tenant_id, tenant_id, tenant_id, max(1, min(int(sample_size), 5))),
        ).fetchall()
    finally:
        con.close()
    return [_safe_text(row[0]) for row in rows if _safe_text(row[0])]


def opportunity_is_open(item: Mapping[str, Any]) -> bool:
    if _safe_text(item.get("closed_at")):
        return False
    status = _safe_text(item.get("status")).casefold()
    return not any(marker in status for marker in CLOSED_STATUS_MARKERS)


def selected_contact_id(profile: Mapping[str, Any]) -> str:
    manager_projection = _mapping(profile.get("manager_projection"))
    for value in _sequence(manager_projection.get("amo_contact_ids")):
        text = _safe_text(value)
        if text:
            return text
    for link in _sequence(profile.get("identity_links")):
        item = _mapping(link)
        if _safe_text(item.get("link_type")) == "amo_contact_id" and _safe_text(item.get("link_value")):
            return _safe_text(item.get("link_value"))
    return ""


def resolve_active_brand(profile: Mapping[str, Any], opportunity: Mapping[str, Any]) -> str:
    deal_brand = normalized_brand(opportunity_brand(opportunity))
    if deal_brand:
        return deal_brand
    customer = _mapping(profile.get("customer"))
    summary = _mapping(customer.get("summary"))
    brand = normalized_brand(summary.get("brand"))
    if brand:
        return brand
    brands = [normalized_brand(item) for item in _sequence(summary.get("brands"))]
    brands = [item for item in brands if item]
    return brands[0] if len(set(brands)) == 1 else ""


def opportunity_brand(opportunity: Mapping[str, Any]) -> str:
    product_context = _mapping(opportunity.get("product_context"))
    return _safe_text(product_context.get("brand") or opportunity.get("brand"))


def normalized_brand(value: Any) -> str:
    text = _safe_text(value).casefold()
    if text in UNKNOWN_BRANDS:
        return ""
    if "фотон" in text or "foton" in text:
        return "foton"
    if "унпк" in text or "unpk" in text or "уник" in text:
        return "unpk"
    return text


def current_field_subset(entity: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, str]:
    current = extract_custom_field_values(dict(entity))
    return {field: current.get(field, "") for field in payload}


def diff_payload(entity: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Mapping[str, str]]:
    current = current_field_subset(entity, payload)
    return {
        field: {
            "old": current.get(field, ""),
            "new": _safe_text(new_value),
            "old_sha256": sha256_text(current.get(field, "")),
            "new_sha256": sha256_text(new_value),
            "changed": str(current.get(field, "") != _safe_text(new_value)).lower(),
        }
        for field, new_value in payload.items()
    }


def base_report_row(
    *,
    customer_id: str,
    entity_type: str,
    entity_id: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "status": "",
        "reason": "",
        "ready_for_write": "no",
        "guard_input_populated": "",
        "active_brand": "",
        "deal_brand": "",
        "open_deal_count": "",
        "snapshot_status": "",
        "snapshot_rows": "0",
        "pre_patch_status": "",
        "pre_patch_reasons": "",
        "allowed_fields": "",
        "payload_fields": " | ".join(payload),
        "payload_sha256": payload_sha256(payload),
        "current_values_json": "{}",
        "new_values_json": json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
        "diff_json": "{}",
    }


def dry_run_summary(
    rows: Sequence[Mapping[str, Any]],
    findings: Sequence[Mapping[str, Any]],
    *,
    config: CrmCardAmoDryRunConfig,
    out_dir: Path,
    contact_catalog_guard: Sequence[str],
    lead_catalog_guard: Sequence[str],
    amo_token_source: str,
) -> Mapping[str, Any]:
    status_counts = Counter(_safe_text(row.get("status")) for row in rows)
    entity_counts = Counter(_safe_text(row.get("entity_type")) for row in rows)
    return {
        "schema_version": CRM_CARD_AMO_DRY_RUN_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timeline_db": str(config.timeline_db),
        "allowed_root": str(config.allowed_root),
        "tenant_id": config.tenant_id,
        "rows": len(rows),
        "status_counts": dict(status_counts.most_common()),
        "entity_counts": dict(entity_counts.most_common()),
        "findings": len(findings),
        "guard_input_populated_rows": sum(1 for row in rows if _safe_text(row.get("guard_input_populated")) == "yes"),
        "contact_catalog_guard": list(contact_catalog_guard),
        "lead_catalog_guard": list(lead_catalog_guard),
        "amo_token_source": amo_token_source,
        "outputs": {
            "report_csv": str(out_dir / "crm_card_amo_dry_run_report.csv"),
            "findings_csv": str(out_dir / "crm_card_amo_dry_run_findings.csv"),
            "snapshot_jsonl": str(out_dir / "pre_write_snapshot.jsonl"),
            "snapshot_csv": str(out_dir / "pre_write_snapshot.csv"),
            "rollback_manifest": str(out_dir / "rollback_manifest.json"),
            "journal": str(out_dir / "write_journal_dry_run.jsonl"),
        },
        "safety": {
            "dry_run_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "send_messages": False,
            "refresh_oauth_token": False,
            "patch_function_available": False,
            "write_stable_runtime": False,
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key)) for key in headers})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finding(
    customer_id: str,
    entity_type: str,
    entity_id: str,
    risk_type: str,
    severity: str,
    *,
    field: str = "",
) -> dict[str, str]:
    return {
        "customer_id": customer_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
        "risk_type": risk_type,
        "severity": severity,
        "field": field,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _drop_empty(payload: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): _safe_text(value) for key, value in payload.items() if _safe_text(value)}


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _safe_text(value)


__all__ = [
    "CONTACT_TARGET_FIELDS",
    "DEAL_TARGET_FIELDS",
    "CrmCardAmoDryRunConfig",
    "AmoReadOnlyClient",
    "build_crm_card_amo_dry_run",
    "build_crm_card_amo_payloads",
    "deal_guard_reasons",
    "dry_run_entity",
    "fetch_field_catalog_read_only",
    "open_amo_opportunities",
    "resolve_amo_access_context_no_refresh",
    "select_customer_ids_for_amo_dry_run",
]
