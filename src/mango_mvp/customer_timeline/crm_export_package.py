from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.crm_card_aggregator import build_crm_card_projection, normalize_manager_multiline_text
from mango_mvp.crm_card_amo_writeback import build_crm_card_amo_payloads
from mango_mvp.customer_timeline.read_api import CustomerTimelineReadApi, CustomerTimelineReadApiConfig
from mango_mvp.customer_timeline.safety import guard_customer_timeline_output_path
from mango_mvp.customer_timeline.store import guard_customer_timeline_sqlite_path
from mango_mvp.quality.crm_text_quality_detector import detect_crm_text_quality_risks


CRM_EXPORT_PACKAGE_SCHEMA_VERSION = "customer_timeline_crm_export_package_v1"
CONTACT_WRITER_READY_POLICY = "live_update_ready"
CONTACT_WRITER_REVIEW_POLICY = "manual_review_required"
AMO_BASE_URL = "https://educent.amocrm.ru"
MAX_AMO_TEXTAREA_CHARS = 60000
HISTORY_HEADROOM_CHARS = 52000


@dataclass(frozen=True)
class CrmExportPackageConfig:
    timeline_db_path: Path
    allowed_root: Path
    out_dir: Path
    tenant_id: str = "foton"
    pilot_size: int = 20
    customer_ids: tuple[str, ...] = ()
    batch_limit: int = 0


def build_crm_export_package(config: CrmExportPackageConfig) -> Mapping[str, Any]:
    db_path = _guard_staging_db(config.timeline_db_path)
    allowed_root = Path(config.allowed_root).expanduser().resolve(strict=False)
    out_dir = _guard_staging_output(config.out_dir, allowed_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_at = _stable_generated_at(db_path)
    db_sha256 = _sha256_file(db_path)
    with _connect_ro(db_path) as con:
        candidate_meta = _select_candidate_meta(
            con,
            tenant_id=config.tenant_id,
            customer_ids=config.customer_ids,
            batch_limit=config.batch_limit,
        )
        customers_total = _count(con, "customer_identities", "tenant_id = ?", (config.tenant_id,))
        open_deals_total = _count(
            con,
            "customer_opportunities",
            "tenant_id = ? AND source_system = 'amocrm_snapshot' AND COALESCE(closed_at, '') = ''",
            (config.tenant_id,),
        )
        candidates = _build_candidate_rows(
            con=con,
            config=config,
            db_path=db_path,
            allowed_root=allowed_root,
            candidate_meta=candidate_meta,
        )

    pilot_rows = candidates[: max(0, int(config.pilot_size))]
    ready_rows = [row for row in candidates if row["crm_card_ready"] == "да"]
    package_files = _write_package_files(out_dir, pilot_rows=pilot_rows, batch_rows=ready_rows)
    output_sha256 = {name: _sha256_file(path) for name, path in package_files.items()}
    summary = {
        "schema_version": CRM_EXPORT_PACKAGE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "tenant_id": config.tenant_id,
        "timeline_db_sha256": db_sha256,
        "customers_total": customers_total,
        "open_amo_deals_total": open_deals_total,
        "candidate_rows": len(candidates),
        "pilot_rows": len(pilot_rows),
        "ready_rows": len(ready_rows),
        "blocked_rows": len(candidates) - len(ready_rows),
        "status_counts": _counts(row["crm_card_ready"] for row in candidates),
        "blocker_counts": _blocker_counts(candidates),
        "outputs": {name: path.name for name, path in package_files.items()},
        "output_sha256": output_sha256,
        "safety": {
            "write_amo": False,
            "write_tallanto": False,
            "send_messages": False,
            "prod_db_write": False,
            "source_open_mode": "sqlite_mode_ro",
            "pii_scope": "local_codex_local_only",
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(_json_dumps(summary), encoding="utf-8")
    return {**summary, "manifest_path": str(manifest_path)}


def _build_candidate_rows(
    *,
    con: sqlite3.Connection,
    config: CrmExportPackageConfig,
    db_path: Path,
    allowed_root: Path,
    candidate_meta: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CustomerTimelineReadApi.open(CustomerTimelineReadApiConfig(timeline_db=db_path, allowed_root=allowed_root)) as api:
        for meta in candidate_meta:
            customer_id = str(meta["customer_id"])
            profile = api.customer_profile(config.tenant_id, customer_id, event_limit=100, bot_context_limit=25)
            extras = _load_customer_extras(con, tenant_id=config.tenant_id, customer_id=customer_id)
            manager_facts = {
                "AMO contact IDs": meta["amo_contact_id"],
                "selected_deal_id": meta["amo_lead_id"],
                "Возражения": extras["objections_text"],
            }
            projection = build_crm_card_projection(
                profile,
                manager_facts=manager_facts,
                selected_amo_lead_id=str(meta["amo_lead_id"]),
            )
            _inject_e5_blocks(projection, extras)
            contact_payload, deal_payload = build_crm_card_amo_payloads(projection)
            blockers = _row_blockers(projection, contact_payload=contact_payload, deal_payload=deal_payload)
            ready = not blockers
            row = _candidate_row(
                meta=meta,
                projection=projection,
                contact_payload=contact_payload,
                deal_payload=deal_payload,
                blockers=blockers,
                ready=ready,
                extras=extras,
            )
            rows.append(row)
    return rows


def _candidate_row(
    *,
    meta: Mapping[str, Any],
    projection: Mapping[str, Any],
    contact_payload: Mapping[str, Any],
    deal_payload: Mapping[str, Any],
    blockers: Sequence[str],
    ready: bool,
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    contact_fields = _mapping(_mapping(projection.get("contact_card")).get("fields"))
    deal_fields = _mapping(_mapping(projection.get("deal_card")).get("fields"))
    phone = str(meta.get("phone") or "")
    lead_id = str(meta.get("amo_lead_id") or "")
    contact_id = str(meta.get("amo_contact_id") or "")
    brand = str(meta.get("deal_brand") or "")
    return {
        "customer_id": str(meta["customer_id"]),
        "tenant_id": str(meta.get("tenant_id") or "foton"),
        "crm_card_ready": "да" if ready else "нет",
        "crm_card_blockers": " | ".join(blockers),
        "Телефон клиента": phone,
        "Готово к записи в AMO": "да" if ready else "нет",
        "Тип последнего свежего звонка": str(meta.get("last_call_type") or ""),
        "AMO contact IDs": contact_id,
        "CRM writeback policy": CONTACT_WRITER_READY_POLICY if ready else CONTACT_WRITER_REVIEW_POLICY,
        "CRM writeback blockers": " | ".join(blockers),
        "Следующий шаг": str(deal_fields.get("Следующий шаг") or ""),
        "Краткое резюме последнего свежего звонка": str(contact_fields.get("Последняя сводка") or ""),
        "Краткая история общения": str(contact_fields.get("История общения") or ""),
        "Хронология общения (последние 5 касаний)": str(contact_fields.get("История общения") or ""),
        "Возражения": str(deal_fields.get("Возражения") or ""),
        "Рекомендуемая дата следующего контакта": "",
        "Приоритет лида": _priority_from_extras(extras),
        "Вероятность продажи, %": "",
        "Рекомендуемый продукт": str(contact_fields.get("Запрос") or ""),
        "Продукты интереса": str(contact_fields.get("Запрос") or ""),
        "История общения Tallanto": str(deal_fields.get("Tallanto") or ""),
        "Бренд": brand,
        "Открыть в AMO": f"{AMO_BASE_URL}/leads/detail/{lead_id}" if lead_id else "",
        "Запрос": str(contact_fields.get("Запрос") or ""),
        "Статус сделки": str(deal_fields.get("Статус сделки") or ""),
        "Tallanto": str(deal_fields.get("Tallanto") or ""),
        "Предупреждения": str(deal_fields.get("Предупреждения") or ""),
        "История общения": str(contact_fields.get("История общения") or ""),
        "Готово": "да" if ready else "нет",
        "Блокеры": " | ".join(blockers),
        "Вердикт": "ready_for_owner_review" if ready else "blocked_manual_review",
        "Комментарий": "CRM export package only; AMO write=0",
        "selected_deal_id": lead_id,
        "active_brand": brand,
        "deal_brand": brand,
        "open_deal_count": str(meta.get("open_deal_count") or "1"),
        "purchase_total_in": str(extras.get("purchase_total_in") or "0"),
        "purchase_deals_cnt": str(extras.get("purchase_deals_cnt") or "0"),
        "objections_count": str(extras.get("objections_count") or "0"),
        "active_signals_count": int(extras.get("active_signals_count") or 0),
        "mail_stage2_events_count": str(meta.get("mail_stage2_events_count") or "0"),
        "crm_card_contact_payload_json": _json_dumps(contact_payload).strip(),
        "crm_card_deal_payload_json": _json_dumps(deal_payload).strip(),
        "contact_payload": dict(contact_payload),
        "deal_payload": dict(deal_payload),
        "preview_payload": {
            "contact": dict(contact_payload),
            "deal": dict(deal_payload),
        },
    }


def _select_candidate_meta(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: Sequence[str],
    batch_limit: int,
) -> list[Mapping[str, Any]]:
    params: list[Any] = [tenant_id, tenant_id, tenant_id, tenant_id, tenant_id, tenant_id, tenant_id, tenant_id, tenant_id]
    customer_filter = ""
    if customer_ids:
        placeholders = ",".join("?" for _ in customer_ids)
        customer_filter = f" AND ci.customer_id IN ({placeholders})"
        params.extend(customer_ids)
    limit_clause = ""
    if batch_limit and int(batch_limit) > 0:
        limit_clause = " LIMIT ?"
        params.append(int(batch_limit))
    objection_counts_filter = _crm_objections_sql_filter(con, tenant_id=tenant_id)
    rows = con.execute(
        f"""
        WITH contacts AS (
          SELECT customer_id, MIN(link_value) AS amo_contact_id, COUNT(DISTINCT link_value) AS contact_count
          FROM identity_links
          WHERE tenant_id = ? AND link_type = 'amo_contact_id' AND match_class = 'strong_unique'
          GROUP BY customer_id
        ),
        phones AS (
          SELECT customer_id, MIN(link_value) AS phone
          FROM identity_links
          WHERE tenant_id = ? AND link_type = 'phone' AND match_class = 'strong_unique'
          GROUP BY customer_id
        ),
        open_deals AS (
          SELECT
            customer_id,
            MIN(source_id) AS amo_lead_id,
            COUNT(*) AS open_deal_count,
            lower(COALESCE(MAX(json_extract(record_json, '$.product_context.brand')), '')) AS deal_brand,
            MAX(opened_at) AS opened_at
          FROM customer_opportunities
          WHERE tenant_id = ?
            AND source_system = 'amocrm_snapshot'
            AND COALESCE(closed_at, '') = ''
            AND source_id <> ''
            AND source_id NOT GLOB '*[^0-9]*'
          GROUP BY customer_id
        ),
        mail_counts AS (
          SELECT customer_id, COUNT(*) AS mail_stage2_events_count, MAX(event_at) AS latest_mail_at
          FROM timeline_events
          WHERE tenant_id = ? AND source_system = 'mail_archive_stage2' AND superseded_by IS NULL
          GROUP BY customer_id
        ),
        purchase_counts AS (
          SELECT customer_id, SUM(COALESCE(total_in, 0)) AS purchase_total_in, SUM(deals_cnt) AS purchase_deals_cnt
          FROM customer_purchases_v1
          WHERE tenant_id = ?
          GROUP BY customer_id
        ),
        objection_counts AS (
          SELECT customer_id, COUNT(*) AS objections_count
          FROM customer_objections_v1
          WHERE tenant_id = ?
          {objection_counts_filter}
          GROUP BY customer_id
        ),
        signal_counts AS (
          SELECT customer_id, COUNT(*) AS active_signals_count
          FROM derived_signals
          WHERE tenant_id = ? AND COALESCE(status, 'active') = 'active'
          GROUP BY customer_id
        ),
        call_types AS (
          SELECT customer_id, json_extract(record_json, '$.call_type') AS last_call_type
          FROM timeline_events
          WHERE tenant_id = ? AND event_type = 'mango_call' AND superseded_by IS NULL
          GROUP BY customer_id
          HAVING MAX(event_at)
        )
        SELECT
          ci.tenant_id,
          ci.customer_id,
          contacts.amo_contact_id,
          phones.phone,
          open_deals.amo_lead_id,
          open_deals.open_deal_count,
          open_deals.deal_brand,
          COALESCE(mail_counts.mail_stage2_events_count, 0) AS mail_stage2_events_count,
          COALESCE(purchase_counts.purchase_total_in, 0) AS purchase_total_in,
          COALESCE(purchase_counts.purchase_deals_cnt, 0) AS purchase_deals_cnt,
          COALESCE(objection_counts.objections_count, 0) AS objections_count,
          COALESCE(signal_counts.active_signals_count, 0) AS active_signals_count,
          COALESCE(call_types.last_call_type, '') AS last_call_type,
          (
            COALESCE(mail_counts.mail_stage2_events_count, 0)
            + COALESCE(purchase_counts.purchase_deals_cnt, 0) * 5
            + COALESCE(objection_counts.objections_count, 0) * 3
            + COALESCE(signal_counts.active_signals_count, 0) * 3
          ) AS richness_score
        FROM customer_identities ci
        JOIN contacts ON contacts.customer_id = ci.customer_id AND contacts.contact_count = 1
        JOIN open_deals ON open_deals.customer_id = ci.customer_id AND open_deals.open_deal_count = 1
        LEFT JOIN phones ON phones.customer_id = ci.customer_id
        LEFT JOIN mail_counts ON mail_counts.customer_id = ci.customer_id
        LEFT JOIN purchase_counts ON purchase_counts.customer_id = ci.customer_id
        LEFT JOIN objection_counts ON objection_counts.customer_id = ci.customer_id
        LEFT JOIN signal_counts ON signal_counts.customer_id = ci.customer_id
        LEFT JOIN call_types ON call_types.customer_id = ci.customer_id
        WHERE ci.tenant_id = ?
          AND ci.identity_status = 'strong'
          AND open_deals.deal_brand IN ('foton', 'unpk')
          {customer_filter}
        ORDER BY richness_score DESC, ci.last_seen_at DESC, ci.customer_id
        {limit_clause}
        """,
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def _load_customer_extras(con: sqlite3.Connection, *, tenant_id: str, customer_id: str) -> Mapping[str, Any]:
    purchases = [
        dict(row)
        for row in con.execute(
            """
            SELECT period, total_in, total_out, deals_cnt, last_purchase_at, computability
            FROM customer_purchases_v1
            WHERE tenant_id = ? AND customer_id = ?
            ORDER BY period DESC
            """,
            (tenant_id, customer_id),
        ).fetchall()
    ]
    objections = _load_crm_eligible_objections(con, tenant_id=tenant_id, customer_id=customer_id)
    signals = []
    for row in con.execute(
        """
        SELECT signal_type, severity, expires_at, requires_manager_review, record_json
        FROM derived_signals
        WHERE tenant_id = ? AND customer_id = ? AND COALESCE(status, 'active') = 'active'
        ORDER BY severity DESC, created_at DESC
        LIMIT 8
        """,
        (tenant_id, customer_id),
    ).fetchall():
        item = dict(row)
        record = _json_loads(item.pop("record_json", ""))
        if isinstance(record, Mapping):
            item.update({key: record.get(key) for key in ("evidence_text", "recommended_action") if record.get(key)})
        signals.append(item)
    return {
        "purchases": purchases,
        "objections": objections,
        "signals": signals,
        "purchase_text": _format_purchases(purchases),
        "objections_text": _format_objections(objections),
        "signals_text": _format_signals(signals),
        "purchase_total_in": sum(float(row.get("total_in") or 0) for row in purchases),
        "purchase_deals_cnt": sum(int(row.get("deals_cnt") or 0) for row in purchases),
        "objections_count": len(objections),
        "active_signals_count": len(signals),
    }


def _load_crm_eligible_objections(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_id: str,
) -> list[dict[str, Any]]:
    if not _crm_objections_enabled(con, tenant_id=tenant_id):
        return []
    filter_sql = _crm_objections_sql_filter(con, tenant_id=tenant_id)
    return [
        dict(row)
        for row in con.execute(
            f"""
            SELECT source_channel, objection_type, quote_preview, budget_hint_rub, price_sensitivity
            FROM customer_objections_v1
            WHERE tenant_id = ? AND customer_id = ?
            {filter_sql}
            ORDER BY extracted_at DESC, source_event_id
            LIMIT 8
            """,
            (tenant_id, customer_id),
        ).fetchall()
    ]


def _crm_objections_sql_filter(con: sqlite3.Connection, *, tenant_id: str) -> str:
    if not _crm_objections_enabled(con, tenant_id=tenant_id):
        return "AND 0"
    columns = _table_columns(con, "customer_objections_v1")
    if {"speaker", "confidence"} <= columns:
        return "AND speaker = 'client' AND confidence = 'high'"
    return "AND 0"


def _crm_objections_enabled(con: sqlite3.Connection, *, tenant_id: str) -> bool:
    if not _table_exists(con, "customer_objection_extraction_runs_v1"):
        return True
    row = con.execute(
        """
        SELECT crm_objections_enabled
        FROM customer_objection_extraction_runs_v1
        WHERE tenant_id = ?
        ORDER BY extracted_at DESC
        LIMIT 1
        """,
        (tenant_id,),
    ).fetchone()
    if row is None:
        return True
    return bool(int(row["crm_objections_enabled"] or 0))


def _inject_e5_blocks(projection: Mapping[str, Any], extras: Mapping[str, Any]) -> None:
    contact_card = _mapping(projection.get("contact_card"))
    deal_card = _mapping(projection.get("deal_card"))
    contact_fields = contact_card.get("fields") if isinstance(contact_card.get("fields"), dict) else {}
    deal_fields = deal_card.get("fields") if isinstance(deal_card.get("fields"), dict) else {}
    blocks = []
    if extras.get("purchase_text"):
        blocks.append("Покупки и оплаты:\n" + str(extras["purchase_text"]))
    if extras.get("objections_text"):
        blocks.append("Возражения и бюджет:\n" + str(extras["objections_text"]))
    if extras.get("signals_text"):
        blocks.append("Сигналы:\n" + str(extras["signals_text"]))
    if not blocks:
        return
    current_history = str(contact_fields.get("История общения") or "")
    appended = normalize_manager_multiline_text("\n\n".join([current_history, *blocks]))
    contact_fields["История общения"] = _fit_textarea(appended, HISTORY_HEADROOM_CHARS)
    if extras.get("objections_text") and not str(deal_fields.get("Возражения") or ""):
        deal_fields["Возражения"] = str(extras["objections_text"])
    if extras.get("purchase_text"):
        current_tallanto = str(deal_fields.get("Tallanto") or "")
        deal_fields["Tallanto"] = normalize_manager_multiline_text(
            "\n".join(part for part in (current_tallanto, "Покупки/оплаты: " + str(extras["purchase_text"])) if part)
        )
    if extras.get("signals_text"):
        current_warnings = str(deal_fields.get("Предупреждения") or "")
        deal_fields["Предупреждения"] = normalize_manager_multiline_text(
            "\n".join(part for part in (current_warnings, "Сигналы: " + str(extras["signals_text"])) if part)
        )


def _row_blockers(
    projection: Mapping[str, Any],
    *,
    contact_payload: Mapping[str, Any],
    deal_payload: Mapping[str, Any],
) -> list[str]:
    contact_card = _mapping(projection.get("contact_card"))
    deal_card = _mapping(projection.get("deal_card"))
    blockers = [str(item) for item in (contact_card.get("blockers") or [])]
    blockers.extend(str(item) for item in (deal_card.get("blockers") or []))
    if not bool(contact_card.get("ready_for_amo")):
        blockers.append("contact_card_not_ready")
    if not bool(deal_card.get("ready_for_amo")):
        blockers.append("deal_card_not_ready")
    if not contact_payload:
        blockers.append("empty_contact_payload")
    if not deal_payload:
        blockers.append("empty_deal_payload")
    if any(len(str(value)) > MAX_AMO_TEXTAREA_CHARS for value in (*contact_payload.values(), *deal_payload.values())):
        blockers.append("payload_field_over_amo_textarea_limit")
    # Contact and deal are written as different AMO entities. Cross-entity
    # duplication is expected; quality gate must run per payload, like writeback.
    findings = [
        *detect_crm_text_quality_risks(contact_payload, min_severity="P2"),
        *detect_crm_text_quality_risks(deal_payload, min_severity="P2"),
    ]
    blockers.extend(f"crm_text_quality:{item.risk_type}" for item in findings)
    return _dedupe(blockers)


def _format_purchases(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    parts = []
    for row in rows[:4]:
        total_in = _format_money(row.get("total_in"))
        deals = int(row.get("deals_cnt") or 0)
        period = str(row.get("period") or "all")
        last_at = str(row.get("last_purchase_at") or "")[:10]
        parts.append(f"{period}: оплачено {total_in}; сделок {deals}" + (f"; последняя дата {last_at}" if last_at else ""))
    return "\n".join(f"- {part}" for part in parts)


def _format_objections(rows: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for row in rows[:6]:
        objection = str(row.get("objection_type") or "other")
        sensitivity = str(row.get("price_sensitivity") or "")
        quote = str(row.get("quote_preview") or "").strip()
        budget = _format_money(row.get("budget_hint_rub")) if row.get("budget_hint_rub") is not None else ""
        detail = "; ".join(part for part in (sensitivity, budget, quote) if part)
        parts.append(f"{objection}: {detail}" if detail else objection)
    return "\n".join(f"- {part}" for part in parts)


def _format_signals(rows: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for row in rows[:6]:
        signal = str(row.get("signal_type") or "")
        severity = str(row.get("severity") or "")
        action = str(row.get("recommended_action") or "")
        evidence = str(row.get("evidence_text") or "")
        detail = "; ".join(part for part in (severity, evidence, action) if part)
        parts.append(f"{signal}: {detail}" if detail else signal)
    return "\n".join(f"- {part}" for part in parts if part)


def _write_package_files(
    out_dir: Path,
    *,
    pilot_rows: Sequence[Mapping[str, Any]],
    batch_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    files = {
        "pilot_csv": out_dir / "pilot_20_crm_card_candidates.csv",
        "pilot_jsonl": out_dir / "pilot_20_crm_card_candidates.jsonl",
        "pilot_preview_md": out_dir / "pilot_20_preview.md",
        "batch_csv": out_dir / "batch_ready_crm_card_candidates.csv",
        "batch_jsonl": out_dir / "batch_ready_crm_card_candidates.jsonl",
        "readback_plan_md": out_dir / "readback_plan.md",
    }
    _write_csv(files["pilot_csv"], pilot_rows)
    _write_jsonl(files["pilot_jsonl"], pilot_rows)
    files["pilot_preview_md"].write_text(_render_preview_markdown(pilot_rows), encoding="utf-8")
    _write_csv(files["batch_csv"], batch_rows)
    _write_jsonl(files["batch_jsonl"], batch_rows)
    files["readback_plan_md"].write_text(_render_readback_plan(), encoding="utf-8")
    return files


def _render_preview_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    parts = [
        "# E5 CRM export pilot preview",
        "",
        "Локальный файл с ПДн. Не коммитить и не выносить в Foton.",
        "AMO write=0; Tallanto write=0; client sends=0.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        parts.extend(
            [
                f"## {index}. customer_id={row.get('customer_id')} lead={row.get('selected_deal_id')}",
                "",
                f"- ready: {row.get('crm_card_ready') or row.get('Готово')}",
                f"- blockers: {row.get('crm_card_blockers') or row.get('Блокеры') or '-'}",
                f"- brand: {row.get('active_brand')}",
                "",
                "### Contact payload",
                "```text",
                _payload_text(_mapping(row.get("contact_payload"))),
                "```",
                "",
                "### Deal payload",
                "```text",
                _payload_text(_mapping(row.get("deal_payload"))),
                "```",
                "",
            ]
        )
    return "\n".join(parts).rstrip() + "\n"


def _render_readback_plan() -> str:
    return (
        "# Readback plan for future AMO apply\n\n"
        "Этот пакет сам не пишет в AMO. После отдельного live-write запуска владелец должен:\n\n"
        "1. Для contact fields запустить существующий `scripts/readback_amo_contact_writeback.py` "
        "на фактическом `contact_writeback_report.csv/json`.\n"
        "2. Для deal fields запустить `scripts/readback_deal_aware_amo_fields.py` на фактическом "
        "`deal_stage6_writeback_report.csv/json`.\n"
        "3. Сверить `payload_sha256`, pre-patch `clobber_protected=0`, количество записанных строк "
        "и отсутствие полей вне allowlist.\n"
    )


def _priority_from_extras(extras: Mapping[str, Any]) -> str:
    signals = extras.get("signals") if isinstance(extras.get("signals"), Sequence) else []
    if any(str(_mapping(item).get("severity")) in {"critical", "high"} for item in signals):
        return "hot"
    if int(extras.get("objections_count") or 0) > 0:
        return "review"
    return "warm"


def _stable_generated_at(db_path: Path) -> str:
    with _connect_ro(db_path) as con:
        row = con.execute(
            """
            SELECT MAX(value) AS ts FROM (
              SELECT MAX(created_at) AS value FROM timeline_events
              UNION ALL SELECT MAX(created_at) FROM bot_context_chunks
              UNION ALL SELECT MAX(created_at) FROM derived_signals
            )
            """
        ).fetchone()
    return str(row["ts"] or "1970-01-01T00:00:00+00:00")


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    return con


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return (
        con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )


def _table_columns(con: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(con, table):
        return set()
    return {str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()}


def _guard_staging_db(path: Path) -> Path:
    resolved = guard_customer_timeline_sqlite_path(path)
    if "customer_timeline_prod_20260621" in resolved.parts:
        raise ValueError("CRM export package must not open prod timeline DB")
    if ".codex_local" not in resolved.parts or "staging" not in resolved.parts:
        raise ValueError("CRM export package DB must be under .codex_local/staging")
    return resolved


def _guard_staging_output(path: Path, allowed_root: Path) -> Path:
    resolved = guard_customer_timeline_output_path(path, allowed_root)
    if ".codex_local" not in resolved.parts or "staging" not in resolved.parts:
        raise ValueError("CRM export package output must be under .codex_local/staging")
    return resolved


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = _headers(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in headers})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _headers(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "customer_id",
        "tenant_id",
        "crm_card_ready",
        "crm_card_blockers",
        "Телефон клиента",
        "Готово к записи в AMO",
        "Тип последнего свежего звонка",
        "AMO contact IDs",
        "CRM writeback policy",
        "CRM writeback blockers",
        "Следующий шаг",
        "Краткое резюме последнего свежего звонка",
        "Краткая история общения",
        "Хронология общения (последние 5 касаний)",
        "Возражения",
        "Рекомендуемая дата следующего контакта",
        "Приоритет лида",
        "Вероятность продажи, %",
        "Рекомендуемый продукт",
        "Продукты интереса",
        "История общения Tallanto",
        "Бренд",
        "Открыть в AMO",
        "Запрос",
        "Статус сделки",
        "Tallanto",
        "Предупреждения",
        "История общения",
        "Готово",
        "Блокеры",
        "Вердикт",
        "Комментарий",
        "selected_deal_id",
        "active_brand",
        "deal_brand",
        "open_deal_count",
        "purchase_total_in",
        "purchase_deals_cnt",
        "objections_count",
        "active_signals_count",
        "mail_stage2_events_count",
        "crm_card_contact_payload_json",
        "crm_card_deal_payload_json",
    ]
    seen = set(preferred)
    extra = []
    for row in rows:
        for key in row:
            if key not in seen and not isinstance(row.get(key), (dict, list, tuple)):
                seen.add(key)
                extra.append(key)
    return preferred + extra


def _csv_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def _payload_text(payload: Mapping[str, Any]) -> str:
    return "\n\n".join(f"{key}:\n{value}" for key, value in payload.items())


def _fit_textarea(value: str, limit: int) -> str:
    text = normalize_manager_multiline_text(value)
    if len(text) <= limit:
        return text
    suffix = "\n[текст сжат до лимита CRM-пакета]"
    return text[: max(0, limit - len(suffix))].rstrip() + suffix


def _format_money(value: Any) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "0 руб."
    if amount.is_integer():
        return f"{int(amount):,}".replace(",", " ") + " руб."
    return f"{amount:,.2f}".replace(",", " ") + " руб."


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_loads(value: Any) -> Any:
    try:
        return json.loads(str(value or ""))
    except json.JSONDecodeError:
        return {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count(con: sqlite3.Connection, table: str, where: str, params: Sequence[Any]) -> int:
    row = con.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE {where}", tuple(params)).fetchone()
    return int(row["c"] if row else 0)


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _blocker_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        blockers = str(row.get("CRM writeback blockers") or "").split(" | ")
        for blocker in blockers:
            if blocker:
                result[blocker] = result.get(blocker, 0) + 1
    return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))


def _dedupe(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


__all__ = [
    "CRM_EXPORT_PACKAGE_SCHEMA_VERSION",
    "CrmExportPackageConfig",
    "build_crm_export_package",
]
