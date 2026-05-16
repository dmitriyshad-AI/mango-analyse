from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.deal_aware.stage1_snapshot import quote_ident, read_csv, safe_text, stringify, write_csv
from mango_mvp.utils.phone import normalize_phone


SCHEMA_VERSION = "deal_aware_stage2_attribution_v1"
CONFIDENCE_HIGH_THRESHOLD = 0.76
CONFIDENCE_MEDIUM_THRESHOLD = 0.69
CONFIDENCE_LOW_WARNING_THRESHOLD = CONFIDENCE_MEDIUM_THRESHOLD


@dataclass(frozen=True)
class AttributionPaths:
    stage1_snapshot_root: Path
    out_root: Path
    amo_live_snapshot_root: Path | None = None


def build_deal_attribution_dry_run(paths: AttributionPaths) -> dict[str, Any]:
    paths.out_root.mkdir(parents=True, exist_ok=True)
    stage1 = paths.stage1_snapshot_root
    amo_live = paths.amo_live_snapshot_root or (stage1.parent / "deal_aware_amo_live_snapshot_20260513_v1")

    calls = read_csv(stage1 / "call_snapshot.csv")
    phone_rollup = read_csv(stage1 / "phone_rollup.csv")
    amo_ready = read_csv(stage1 / "amo_ready_snapshot.csv")
    amo_writebacks = read_csv(stage1 / "amo_writeback_snapshot.csv")
    live_deals = read_csv(amo_live / "amo_deals_snapshot.csv") if amo_live.exists() else []
    live_contacts = read_csv(amo_live / "amo_contacts_snapshot.csv") if amo_live.exists() else []
    live_snapshot_summary = load_json(amo_live / "summary.json") if amo_live.exists() else {}
    live_snapshot_available = is_live_snapshot_available(live_snapshot_summary)

    phone_candidates = build_phone_deal_candidates(
        phone_rollup=phone_rollup,
        amo_ready=amo_ready,
        amo_writebacks=amo_writebacks,
        live_deals=live_deals,
        live_contacts=live_contacts,
    )
    links = [
        attribute_call_to_deal(
            call,
            phone_candidates.get(normalize_phone(call.get("phone", "")), []),
            live_snapshot_available=live_snapshot_available,
        )
        for call in calls
    ]
    manual_review = [row for row in links if row["attribution_decision"].startswith("manual_review")]
    skipped = [row for row in links if row["attribution_decision"].startswith("skipped")]
    linked = [row for row in links if row["attribution_decision"].startswith("linked")]
    distribution = build_distribution_rows(links)

    outputs = {
        "deal_call_links_csv": paths.out_root / "deal_call_links.csv",
        "manual_review_csv": paths.out_root / "deal_call_links_manual_review.csv",
        "skipped_csv": paths.out_root / "deal_call_links_skipped.csv",
        "confidence_distribution_csv": paths.out_root / "confidence_distribution.csv",
        "phone_deal_candidates_csv": paths.out_root / "phone_deal_candidates.csv",
        "sqlite": paths.out_root / "deal_aware_stage2_attribution.sqlite",
        "summary_json": paths.out_root / "summary.json",
        "readme": paths.out_root / "README.md",
    }
    write_csv(outputs["deal_call_links_csv"], links)
    write_csv(outputs["manual_review_csv"], manual_review)
    write_csv(outputs["skipped_csv"], skipped)
    write_csv(outputs["confidence_distribution_csv"], distribution)
    write_csv(outputs["phone_deal_candidates_csv"], flatten_phone_candidates(phone_candidates))
    write_sqlite(
        outputs["sqlite"],
        {
            "deal_call_links": links,
            "manual_review": manual_review,
            "skipped": skipped,
            "confidence_distribution": distribution,
            "phone_deal_candidates": flatten_phone_candidates(phone_candidates),
        },
    )
    summary = build_summary(
        paths=paths,
        amo_live_snapshot_root=amo_live,
        calls=calls,
        phone_candidates=phone_candidates,
        live_snapshot_summary=live_snapshot_summary,
        live_snapshot_available=live_snapshot_available,
        links=links,
        manual_review=manual_review,
        skipped=skipped,
        linked=linked,
        outputs=outputs,
    )
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["readme"].write_text(render_readme(summary), encoding="utf-8")
    return summary


def build_phone_deal_candidates(
    *,
    phone_rollup: list[dict[str, str]],
    amo_ready: list[dict[str, str]],
    amo_writebacks: list[dict[str, str]],
    live_deals: list[dict[str, str]],
    live_contacts: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    deal_by_id = {safe_text(deal.get("lead_id")): deal for deal in live_deals if safe_text(deal.get("lead_id"))}

    def add(phone: str, lead_id: str, *, contact_ids: str = "", source: str, deal_meta: dict[str, Any] | None = None) -> None:
        normalized = normalize_phone(phone)
        lead_id = safe_text(lead_id)
        if not normalized or not lead_id:
            return
        item = candidates[normalized].setdefault(
            lead_id,
            {
                "phone": normalized,
                "deal_id": lead_id,
                "contact_ids": set(),
                "candidate_sources": set(),
                "deal_name": "",
                "pipeline_name": "",
                "status_name": "",
                "loss_reason": "",
                "is_terminal_deal": False,
                "is_active_deal": "unknown",
                "is_duplicate_or_existing_client": False,
                "deal_updated_at": "",
                "deal_closed_at": "",
                "duplicate_of_lead_id": "",
            },
        )
        item["candidate_sources"].add(source)
        for contact_id in split_ids(contact_ids):
            item["contact_ids"].add(contact_id)
        if deal_meta:
            enrich_candidate_from_deal_meta(item, deal_meta)

    for row in phone_rollup:
        for lead_id in split_ids(row.get("amo_lead_ids")):
            add(row.get("phone", ""), lead_id, contact_ids=row.get("amo_contact_ids", ""), source="phone_rollup")
    for row in amo_ready:
        for lead_id in split_ids(row.get("amo_lead_ids")):
            add(row.get("phone", ""), lead_id, contact_ids=row.get("amo_contact_ids", ""), source="amo_ready")
    for row in amo_writebacks:
        for lead_id in split_ids(row.get("amo_lead_ids")):
            add(row.get("phone", ""), lead_id, contact_ids=row.get("amo_contact_ids", ""), source="amo_writeback")

    contact_phones: dict[str, set[str]] = defaultdict(set)
    for contact in live_contacts:
        contact_id = safe_text(contact.get("contact_id"))
        if not contact_id:
            continue
        for phone in split_pipe(contact.get("phones")):
            normalized = normalize_phone(phone)
            if normalized:
                contact_phones[contact_id].add(normalized)

    for deal in live_deals:
        lead_id = safe_text(deal.get("lead_id"))
        if not lead_id:
            continue
        linked_contacts = split_pipe(deal.get("linked_contact_ids"))
        meta = deal_meta_from_live_row(deal)
        for contact_id in linked_contacts:
            for phone in contact_phones.get(contact_id, set()):
                add(phone, lead_id, contact_ids=contact_id, source="amo_live_linked_contact", deal_meta=meta)

    result: dict[str, list[dict[str, Any]]] = {}
    for phone, by_deal in candidates.items():
        rows = []
        for item in by_deal.values():
            live_meta = deal_by_id.get(safe_text(item.get("deal_id")))
            if live_meta:
                enrich_candidate_from_deal_meta(item, deal_meta_from_live_row(live_meta))
            row = dict(item)
            row["contact_ids"] = " | ".join(sorted(item["contact_ids"]))
            row["candidate_sources"] = " | ".join(sorted(item["candidate_sources"]))
            rows.append(row)
        rows.sort(key=candidate_sort_key)
        result[phone] = rows
    return result


def attribute_call_to_deal(
    call: dict[str, str],
    candidates: list[dict[str, Any]],
    *,
    live_snapshot_available: bool = True,
) -> dict[str, Any]:
    phone = normalize_phone(call.get("phone", ""))
    contentful = is_yes(call.get("contentful"))
    call_type = safe_text(call.get("call_type"))
    base = {
        "call_id": safe_text(call.get("call_id")),
        "phone": phone,
        "started_at": safe_text(call.get("started_at")),
        "call_type": call_type,
        "contentful": safe_text(call.get("contentful")),
        "manager_name": safe_text(call.get("manager_name")),
        "source_filename": safe_text(call.get("source_filename")),
        "call_summary": safe_text(call.get("call_summary"))[:600],
        "products": safe_text(call.get("products")),
        "subjects": safe_text(call.get("subjects")),
        "call_next_step": safe_text(call.get("next_step")),
        "candidate_deal_ids": " | ".join(row["deal_id"] for row in candidates),
        "candidate_count": len(candidates),
        "candidate_sources": " | ".join(sorted({source for row in candidates for source in split_pipe(row.get("candidate_sources"))})),
        "selected_deal_id": "",
        "selected_contact_ids": "",
        "selected_deal_name": "",
        "selected_pipeline_name": "",
        "selected_status_name": "",
        "selected_loss_reason": "",
        "confidence_score": "0.00",
        "confidence_bucket": "none",
        "attribution_decision": "",
        "manual_review_reason": "",
        "safe_for_deal_writeback": "Нет",
    }

    if not phone:
        return base | {
            "attribution_decision": "manual_review_missing_phone",
            "manual_review_reason": "Телефон звонка не нормализован.",
        }
    if not contentful:
        return base | {
            "attribution_decision": "skipped_non_contentful_call",
            "manual_review_reason": "Звонок несодержательный, не привязываем к сделке.",
        }
    if call_type != "sales_call":
        return base | {
            "attribution_decision": "skipped_non_sales_call",
            "manual_review_reason": f"Тип звонка {call_type or '<empty>'} не является sales_call.",
        }
    if not candidates:
        return base | {
            "attribution_decision": "manual_review_no_deal_candidate",
            "manual_review_reason": "По телефону нет сделки-кандидата в Stage 1 snapshot.",
        }
    if not live_snapshot_available:
        return base | {
            "attribution_decision": "manual_review_live_amo_snapshot_unavailable",
            "manual_review_reason": "Свежий AMO snapshot сделок недоступен; показываем кандидатов, но не считаем привязку надежной.",
            "confidence_bucket": "low",
            "confidence_score": "0.30",
        }
    if len(candidates) > 1:
        active_non_duplicate = [
            candidate
            for candidate in candidates
            if is_active_candidate(candidate) and not is_duplicate_or_existing_client(candidate)
        ]
        if active_non_duplicate:
            candidates = active_non_duplicate
            if len(candidates) > 1:
                return base | {
                    "attribution_decision": "manual_review_multiple_active_deals",
                    "manual_review_reason": f"По телефону найдено {len(candidates)} активных сделок; нужна ручная привязка.",
                    "candidate_deal_ids": " | ".join(candidate["deal_id"] for candidate in candidates),
                    "confidence_bucket": "low",
                    "confidence_score": "0.35",
                }
        elif any(is_active_unknown(candidate) for candidate in candidates):
            return base | {
                "attribution_decision": "manual_review_multiple_deal_candidates",
                "manual_review_reason": "По телефону найдено несколько сделок-кандидатов, но активность части сделок не подтверждена live snapshot.",
                "confidence_bucket": "low",
                "confidence_score": "0.25",
            }
        elif not any(is_active_candidate(candidate) for candidate in candidates):
            return base | {
                "attribution_decision": "manual_review_all_candidates_terminal",
                "manual_review_reason": "Все сделки-кандидаты закрытые/терминальные; нужна ручная привязка.",
                "confidence_bucket": "low",
                "confidence_score": "0.20",
            }

    selected = candidates[0]
    score = single_candidate_confidence(selected)
    if is_truthy(selected.get("is_terminal_deal")):
        return base | selected_fields(selected) | {
            "attribution_decision": "manual_review_single_terminal_deal_candidate",
            "manual_review_reason": "Единственная сделка-кандидат в терминальном статусе; нужна проверка перед записью.",
            "confidence_bucket": confidence_bucket(score),
            "confidence_score": f"{score:.2f}",
            "safe_for_deal_writeback": "Нет",
        }
    return base | selected_fields(selected) | {
        "attribution_decision": "linked_single_deal_candidate",
        "manual_review_reason": "",
        "confidence_bucket": confidence_bucket(score),
        "confidence_score": f"{score:.2f}",
        "safe_for_deal_writeback": "Да" if score >= CONFIDENCE_MEDIUM_THRESHOLD else "Нет",
    }


def selected_fields(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_deal_id": safe_text(candidate.get("deal_id")),
        "selected_contact_ids": safe_text(candidate.get("contact_ids")),
        "selected_deal_name": safe_text(candidate.get("deal_name")),
        "selected_pipeline_name": safe_text(candidate.get("pipeline_name")),
        "selected_status_name": safe_text(candidate.get("status_name")),
        "selected_loss_reason": safe_text(candidate.get("loss_reason")),
    }


def deal_meta_from_live_row(deal: dict[str, Any]) -> dict[str, Any]:
    return {
        "deal_name": deal.get("lead_name") or deal.get("deal_name"),
        "pipeline_name": deal.get("pipeline_name"),
        "status_name": deal.get("status_name"),
        "loss_reason": deal.get("loss_reason"),
        "updated_at": deal.get("updated_at") or deal.get("modified_at"),
        "modified_at": deal.get("modified_at"),
        "closed_at": deal.get("closed_at"),
        "_links": deal.get("_links"),
        "duplicate_of_lead_id": deal.get("duplicate_of_lead_id") or deal.get("duplicate_of"),
    }


def enrich_candidate_from_deal_meta(item: dict[str, Any], deal_meta: dict[str, Any]) -> None:
    for key in ("deal_name", "pipeline_name", "status_name", "loss_reason"):
        if safe_text(deal_meta.get(key)) and not item[key]:
            item[key] = safe_text(deal_meta.get(key))
    updated_at = safe_text(deal_meta.get("updated_at") or deal_meta.get("modified_at"))
    closed_at = safe_text(deal_meta.get("closed_at"))
    duplicate_of = extract_duplicate_of_lead_id(deal_meta)
    if updated_at:
        item["deal_updated_at"] = updated_at
    if closed_at:
        item["deal_closed_at"] = closed_at
    if duplicate_of:
        item["duplicate_of_lead_id"] = duplicate_of
    item["is_terminal_deal"] = is_terminal_status(item | deal_meta)
    item["is_active_deal"] = not item["is_terminal_deal"]
    item["is_duplicate_or_existing_client"] = is_duplicate_or_existing_loss_reason(item.get("loss_reason"))


def extract_duplicate_of_lead_id(deal_meta: dict[str, Any]) -> str:
    direct = safe_text(deal_meta.get("duplicate_of_lead_id") or deal_meta.get("duplicate_of"))
    if direct:
        return direct
    links = deal_meta.get("_links")
    if isinstance(links, str):
        try:
            links = json.loads(links)
        except json.JSONDecodeError:
            links = {}
    if isinstance(links, dict):
        duplicate_of = links.get("duplicate_of")
        if isinstance(duplicate_of, dict):
            return safe_text(duplicate_of.get("id") or duplicate_of.get("lead_id"))
        return safe_text(duplicate_of)
    return ""


def is_duplicate_or_existing_loss_reason(value: Any) -> bool:
    return safe_text(value).casefold() in {"дубль", "действующий клиент"}


def is_duplicate_or_existing_client(candidate: dict[str, Any]) -> bool:
    return is_truthy(candidate.get("is_duplicate_or_existing_client")) or is_duplicate_or_existing_loss_reason(candidate.get("loss_reason"))


def is_active_candidate(candidate: dict[str, Any]) -> bool:
    return is_truthy(candidate.get("is_active_deal"))


def is_active_unknown(candidate: dict[str, Any]) -> bool:
    value = candidate.get("is_active_deal", "unknown")
    return value in {None, "", "unknown"} or safe_text(value).casefold() == "unknown"


def is_truthy(value: Any) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return safe_text(value).casefold() in {"true", "1", "yes", "да"}


def is_falsey(value: Any) -> bool:
    if value is False:
        return True
    if value is True or value is None:
        return False
    return safe_text(value).casefold() in {"false", "0", "no", "нет"}


def single_candidate_confidence(candidate: dict[str, Any]) -> float:
    sources = set(split_pipe(candidate.get("candidate_sources")))
    score = 0.35
    if "amo_live_linked_contact" in sources:
        score += 0.16
    if "phone_rollup" in sources:
        score += 0.08
    if "amo_ready" in sources:
        score += 0.05
    if "amo_writeback" in sources:
        score += 0.03
    if safe_text(candidate.get("deal_name")):
        score += 0.03
    if is_active_candidate(candidate):
        score += 0.10
    if is_duplicate_or_existing_client(candidate):
        score -= 0.08
    if is_truthy(candidate.get("is_terminal_deal")):
        score -= 0.20
    score += status_confidence_bonus(candidate.get("status_name"))
    score += recency_confidence_bonus(candidate.get("deal_updated_at"))
    return round(max(0.0, min(score, 0.98)), 2)


def status_confidence_bonus(value: Any) -> float:
    text = safe_text(value).casefold()
    if any(marker in text for marker in ("ожидание оплаты", "заключение договора", "запись в группу")):
        return 0.08
    if any(marker in text for marker in ("в работе", "принимают решение", "переговоры")):
        return 0.04
    if "перспектива" in text:
        return 0.02
    if "недозвон" in text:
        return -0.04
    return 0.0


def recency_confidence_bonus(value: Any) -> float:
    text = safe_text(value)
    if not text:
        return 0.0
    try:
        updated_at = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    age_days = (datetime.now(timezone.utc) - updated_at).days
    if age_days <= 7:
        return 0.09
    if age_days <= 14:
        return 0.06
    if age_days <= 30:
        return 0.04
    if age_days <= 60:
        return 0.02
    return 0.0


def candidate_source_score(candidate: dict[str, Any]) -> int:
    sources = set(split_pipe(candidate.get("candidate_sources")))
    return (
        (100 if "amo_live_linked_contact" in sources else 0)
        + (40 if "phone_rollup" in sources else 0)
        + (30 if "amo_ready" in sources else 0)
        + (10 if "amo_writeback" in sources else 0)
    )


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, float, int, str]:
    if is_active_candidate(candidate):
        active_rank = 0
    elif is_active_unknown(candidate):
        active_rank = 1
    else:
        active_rank = 2
    duplicate_rank = 1 if is_duplicate_or_existing_client(candidate) else 0
    return (
        active_rank,
        duplicate_rank,
        -timestamp_score(candidate.get("deal_updated_at")),
        -candidate_source_score(candidate),
        safe_text(candidate.get("deal_id")),
    )


def timestamp_score(value: Any) -> float:
    text = safe_text(value)
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def confidence_bucket(score: float) -> str:
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def build_distribution_rows(links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: Counter[tuple[str, str]] = Counter(
        (safe_text(row.get("attribution_decision")), safe_text(row.get("confidence_bucket"))) for row in links
    )
    return [
        {"attribution_decision": decision, "confidence_bucket": bucket, "rows": count}
        for (decision, bucket), count in sorted(grouped.items())
    ]


def build_summary(
    *,
    paths: AttributionPaths,
    amo_live_snapshot_root: Path,
    calls: list[dict[str, str]],
    phone_candidates: dict[str, list[dict[str, Any]]],
    live_snapshot_summary: dict[str, Any],
    live_snapshot_available: bool,
    links: list[dict[str, Any]],
    manual_review: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    linked: list[dict[str, Any]],
    outputs: dict[str, Path],
) -> dict[str, Any]:
    decision_counts = Counter(safe_text(row.get("attribution_decision")) for row in links)
    confidence_counts = Counter(safe_text(row.get("confidence_bucket")) for row in links)
    safe_writeback_rows = [row for row in links if row.get("safe_for_deal_writeback") == "Да"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "stage1_snapshot_root": str(paths.stage1_snapshot_root),
            "amo_live_snapshot_root": str(amo_live_snapshot_root),
        },
        "safety": {
            "read_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "run_asr": False,
            "run_resolve_analyze": False,
        },
        "coverage": {
            "calls_seen": len(calls),
            "phones_with_deal_candidates": len(phone_candidates),
            "deal_candidate_rows": sum(len(rows) for rows in phone_candidates.values()),
            "live_snapshot_available": live_snapshot_available,
            "linked_rows": len(linked),
            "manual_review_rows": len(manual_review),
            "skipped_rows": len(skipped),
            "safe_for_future_deal_writeback_rows": len(safe_writeback_rows),
        },
        "decision_counts": dict(decision_counts.most_common()),
        "confidence_counts": dict(confidence_counts.most_common()),
        "readiness": {
            "deal_attribution_dry_run_built": True,
            "safe_to_write_deal_fields": False,
            "requires_stage3_deal_state_classifier": True,
            "requires_live_amo_snapshot_for_full_coverage": True,
        },
        "amo_live_snapshot": {
            "api_read_succeeded": live_snapshot_summary.get("api_read_succeeded", True) if live_snapshot_summary else False,
            "preflight_error": safe_text(live_snapshot_summary.get("preflight_error")),
            "contacts_seen": safe_text((live_snapshot_summary.get("fetch") or {}).get("contacts_seen") if live_snapshot_summary else ""),
            "leads_seen": safe_text((live_snapshot_summary.get("fetch") or {}).get("leads_seen") if live_snapshot_summary else ""),
            "tasks_seen": safe_text((live_snapshot_summary.get("fetch") or {}).get("tasks_seen") if live_snapshot_summary else ""),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def flatten_phone_candidates(phone_candidates: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for phone, candidates in phone_candidates.items():
        for candidate in candidates:
            rows.append({"phone": phone, **{key: stringify(value) for key, value in candidate.items()}})
    return rows


def split_ids(value: Any) -> list[str]:
    ids = []
    for part in split_pipe(value):
        text = "".join(ch for ch in part if ch.isdigit())
        if text:
            ids.append(text)
    return list(dict.fromkeys(ids))


def split_pipe(value: Any) -> list[str]:
    return [part.strip() for part in safe_text(value).split("|") if part.strip()]


def is_yes(value: Any) -> bool:
    return safe_text(value).casefold() in {"1", "true", "yes", "да"}


def is_terminal_status(candidate: dict[str, Any]) -> bool:
    text = " ".join(
        [
            safe_text(candidate.get("status_name")),
            safe_text(candidate.get("pipeline_name")),
            safe_text(candidate.get("loss_reason")),
        ]
    ).casefold()
    return any(marker in text for marker in ("закрыто", "нереализ", "реализовано", "отказ", "успешно", "оплата получена"))


def is_live_snapshot_available(summary: dict[str, Any]) -> bool:
    if not summary:
        return False
    if summary.get("api_read_succeeded") is False:
        return False
    fetch = summary.get("fetch") if isinstance(summary.get("fetch"), dict) else {}
    return int_or_zero(fetch.get("contacts_seen")) > 0 and int_or_zero(fetch.get("leads_seen")) > 0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def int_or_zero(value: Any) -> int:
    try:
        return int(float(safe_text(value).replace(",", ".")))
    except ValueError:
        return 0


def write_sqlite(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        for table, rows in tables.items():
            if not rows:
                con.execute(f'CREATE TABLE "{table}" (empty TEXT)')
                continue
            columns = sorted({key for row in rows for key in row.keys()})
            con.execute(f'CREATE TABLE "{table}" ({", ".join(f"{quote_ident(col)} TEXT" for col in columns)})')
            placeholders = ", ".join("?" for _ in columns)
            con.executemany(
                f'INSERT INTO "{table}" ({", ".join(quote_ident(col) for col in columns)}) VALUES ({placeholders})',
                [[stringify(row.get(col)) for col in columns] for row in rows],
            )
        con.commit()
    finally:
        con.close()


def render_readme(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    return "\n".join(
        [
            "# Deal-Aware Stage 2 Attribution Dry-Run",
            "",
            "Dry-run only. No AMO/Tallanto writes.",
            "",
            "## Coverage",
            "",
            f"- calls seen: {coverage['calls_seen']}",
            f"- phones with deal candidates: {coverage['phones_with_deal_candidates']}",
            f"- linked rows: {coverage['linked_rows']}",
            f"- manual review rows: {coverage['manual_review_rows']}",
            f"- skipped rows: {coverage['skipped_rows']}",
            f"- future writeback candidates after Stage 3: {coverage['safe_for_future_deal_writeback_rows']}",
            "",
            "## Outputs",
            "",
            *[f"- `{key}`: `{path}`" for key, path in summary["outputs"].items()],
            "",
        ]
    )
