from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to
from mango_mvp.utils.phone import normalize_phone


CRM_ENTITY_RESOLVER_SCHEMA_VERSION = "crm_entity_resolver_v1"
RESOLVE_CRM_ENTITY = "RESOLVE_CRM_ENTITY"
BLOCK_NO_CALL_PHONE = "BLOCK_NO_CALL_PHONE"
BLOCK_NO_CRM_MATCH = "BLOCK_NO_CRM_MATCH"
BLOCK_AMBIGUOUS_CRM_MATCH = "BLOCK_AMBIGUOUS_CRM_MATCH"


@dataclass(frozen=True)
class CrmEntityResolverSummary:
    schema_version: str
    product_db_path: str
    crm_snapshot_path: str
    calls_seen: int
    crm_entities_seen: int
    phones_indexed: int
    resolve_crm_entity: int
    blocked_no_call_phone: int
    blocked_no_crm_match: int
    blocked_ambiguous_crm_match: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_crm_entity_resolution_report(
    product_db_path: Path,
    product_root: Path,
    crm_snapshot_path: Path,
    out_path: Optional[Path] = None,
    *,
    limit: Optional[int] = None,
) -> Mapping[str, Any]:
    product_db_path, product_root, crm_snapshot_path, out_path = resolve_paths(
        product_db_path=product_db_path,
        product_root=product_root,
        crm_snapshot_path=crm_snapshot_path,
        out_path=out_path,
    )
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    calls = read_product_calls(product_db_path, limit=limit)
    entities = load_crm_entities(crm_snapshot_path)
    phone_index = build_phone_index(entities)
    items = [resolve_call_to_crm_entity(call, phone_index=phone_index) for call in calls]
    action_counts = Counter(clean(item.get("action")) for item in items)
    blocked = int(
        action_counts[BLOCK_NO_CALL_PHONE]
        + action_counts[BLOCK_NO_CRM_MATCH]
        + action_counts[BLOCK_AMBIGUOUS_CRM_MATCH]
    )
    warnings = int(action_counts[BLOCK_NO_CRM_MATCH] + action_counts[BLOCK_AMBIGUOUS_CRM_MATCH])
    summary = CrmEntityResolverSummary(
        schema_version=CRM_ENTITY_RESOLVER_SCHEMA_VERSION,
        product_db_path=str(product_db_path),
        crm_snapshot_path=str(crm_snapshot_path),
        calls_seen=len(calls),
        crm_entities_seen=len(entities),
        phones_indexed=len(phone_index),
        resolve_crm_entity=int(action_counts[RESOLVE_CRM_ENTITY]),
        blocked_no_call_phone=int(action_counts[BLOCK_NO_CALL_PHONE]),
        blocked_no_crm_match=int(action_counts[BLOCK_NO_CRM_MATCH]),
        blocked_ambiguous_crm_match=int(action_counts[BLOCK_AMBIGUOUS_CRM_MATCH]),
        validation_ok=True,
        blocked=blocked,
        warnings=warnings,
    )
    report = {
        "summary": summary.to_json_dict(),
        "action_counts": dict(sorted(action_counts.items())),
        "items": items,
        "resolution_index": {
            "by_event_key": resolution_index(items, "event_key"),
            "by_provider_call_id": resolution_index(items, "provider_call_id"),
        },
        "snapshot_contract": snapshot_contract(),
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def resolve_call_to_crm_entity(
    call: Mapping[str, Any],
    *,
    phone_index: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    call_phone = normalize_call_phone(call)
    base = {
        "schema_version": CRM_ENTITY_RESOLVER_SCHEMA_VERSION,
        "tenant_id": clean(call.get("tenant_id")),
        "provider": clean(call.get("telephony_provider")),
        "provider_call_id": clean(call.get("provider_call_id")),
        "event_key": clean(call.get("event_key")),
        "recording_id": clean(call.get("recording_id")) or None,
        "started_at": clean(call.get("started_at")) or None,
        "source_filename": clean(call.get("source_filename")) or None,
        "call_phone": call_phone,
        "crm_provider": "amocrm",
        "write_crm": False,
    }
    if not call_phone:
        return base | {
            "action": BLOCK_NO_CALL_PHONE,
            "reason": "call_phone_not_found",
            "candidate_count": 0,
            "candidates": [],
        }
    candidates = list(phone_index.get(call_phone) or ())
    if not candidates:
        return base | {
            "action": BLOCK_NO_CRM_MATCH,
            "reason": "no_crm_entity_for_phone",
            "candidate_count": 0,
            "candidates": [],
        }
    if len(candidates) > 1:
        return base | {
            "action": BLOCK_AMBIGUOUS_CRM_MATCH,
            "reason": "multiple_crm_entities_for_phone",
            "candidate_count": len(candidates),
            "candidates": [candidate_summary(candidate) for candidate in candidates[:10]],
        }
    candidate = candidates[0]
    return base | {
        "action": RESOLVE_CRM_ENTITY,
        "reason": "exact_phone_single",
        "candidate_count": 1,
        "crm_entity_type": clean(candidate.get("entity_type")),
        "crm_entity_id": clean(candidate.get("entity_id")),
        "crm_entity_name": clean(candidate.get("entity_name")) or None,
        "crm_entity_status": clean(candidate.get("status")) or None,
        "crm_entity_owner_id": clean(candidate.get("owner_id")) or None,
        "crm_entity_owner_name": clean(candidate.get("owner_name")) or None,
        "confidence": 0.95,
        "candidates": [candidate_summary(candidate)],
    }


def load_crm_entities(path: Path) -> list[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            return [normalize_snapshot_row(row, source_ref=f"{path}#row={index}") for index, row in enumerate(csv.DictReader(fh), start=2)]
    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                rows.append(normalize_snapshot_row(value, source_ref=f"{path}#line={line_number}"))
        return rows
    value = json.loads(text)
    rows = extract_json_rows(value)
    return [normalize_snapshot_row(row, source_ref=f"{path}#json[{index}]") for index, row in enumerate(rows)]


def extract_json_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, Mapping)]
    if isinstance(value, Mapping):
        for key in ("entities", "items", "contacts", "leads", "rows"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, Mapping)]
    raise ValueError("CRM snapshot must be a JSON list/object with entities/items/contacts/leads/rows")


def normalize_snapshot_row(row: Mapping[str, Any], *, source_ref: str) -> Mapping[str, Any]:
    entity_id = first_clean(row, "entity_id", "id", "lead_id", "contact_id", "amo_id", "amocrm_id")
    entity_type = first_clean(row, "entity_type", "type", "crm_entity_type") or infer_entity_type(row)
    entity_name = first_clean(row, "entity_name", "name", "title", "contact_name", "lead_name")
    phones = normalize_snapshot_phones(row)
    return {
        "crm_provider": first_clean(row, "crm_provider", "provider") or "amocrm",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "entity_name": entity_name,
        "phones": phones,
        "owner_id": first_clean(row, "owner_id", "responsible_user_id", "crm_owner_id"),
        "owner_name": first_clean(row, "owner_name", "responsible_user_name", "crm_owner_name"),
        "status": first_clean(row, "status", "pipeline_status", "lead_status"),
        "source_ref": source_ref,
        "raw": dict(row),
    }


def normalize_snapshot_phones(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw_values: list[Any] = []
    for key in (
        "phone",
        "phones",
        "phone_numbers",
        "client_phone",
        "contact_phone",
        "Телефон",
        "Телефон клиента",
    ):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            raw_values.extend(value)
        else:
            raw_values.append(value)
    normalized = []
    for value in raw_values:
        phone = normalize_phone(str(value))
        if phone and phone not in normalized:
            normalized.append(phone)
    return tuple(normalized)


def build_phone_index(entities: Sequence[Mapping[str, Any]]) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    for entity in entities:
        if not clean(entity.get("entity_id")) or not clean(entity.get("entity_type")):
            continue
        for phone in entity.get("phones") or ():
            index.setdefault(clean(phone), []).append(entity)
    return {phone: tuple(rows) for phone, rows in sorted(index.items())}


def read_product_calls(product_db_path: Path, limit: Optional[int]) -> list[Mapping[str, Any]]:
    limit_sql = "LIMIT ?" if limit is not None else ""
    params: tuple[Any, ...] = (int(limit),) if limit is not None else ()
    with sqlite3.connect(str(product_db_path)) as con:
        con.row_factory = sqlite3.Row
        ensure_table(con, "product_calls")
        rows = con.execute(
            f"""
            SELECT tenant_id, telephony_provider, provider_call_id, event_key,
                   recording_id, source_filename, started_at, manager_extension,
                   manager_display_name, crm_owner_id, crm_owner_name,
                   raw_payload_ref, source_repository_ref
              FROM product_calls
             ORDER BY started_at DESC, provider_call_id
             {limit_sql}
            """,
            params,
        ).fetchall()
    return [dict(row) for row in rows]


def normalize_call_phone(call: Mapping[str, Any]) -> Optional[str]:
    for key in ("client_phone", "phone", "call_phone"):
        phone = normalize_phone(clean(call.get(key)))
        if phone:
            return phone
    filename = clean(call.get("source_filename"))
    for candidate in phone_candidates_from_text(filename):
        phone = normalize_phone(candidate)
        if phone:
            return phone
    return None


def phone_candidates_from_text(value: str) -> Sequence[str]:
    if not value:
        return ()
    return tuple(match.group(0) for match in re.finditer(r"(?<!\d)(?:7|8)?\d{10}(?!\d)", value))


def resolution_index(items: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Mapping[str, Any]]:
    result = {}
    for item in items:
        if clean(item.get("action")) != RESOLVE_CRM_ENTITY:
            continue
        value = clean(item.get(key))
        if value:
            result[value] = item
    return result


def candidate_summary(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "crm_provider": clean(candidate.get("crm_provider")),
        "entity_type": clean(candidate.get("entity_type")),
        "entity_id": clean(candidate.get("entity_id")),
        "entity_name": clean(candidate.get("entity_name")) or None,
        "phones": list(candidate.get("phones") or ()),
        "status": clean(candidate.get("status")) or None,
        "source_ref": clean(candidate.get("source_ref")) or None,
    }


def snapshot_contract() -> Mapping[str, Any]:
    return {
        "supported_formats": ["json", "jsonl", "csv"],
        "required_fields": ["entity_id or id", "entity_type or type", "phone or phones"],
        "matching_policy": "exact_normalized_phone_single",
        "ambiguous_matches": "blocked",
        "live_crm_reads": False,
        "live_crm_writes": False,
    }


def resolve_paths(
    product_db_path: Path,
    product_root: Path,
    crm_snapshot_path: Path,
    out_path: Optional[Path],
) -> tuple[Path, Path, Path, Optional[Path]]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    crm_snapshot_path = crm_snapshot_path.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    for label, path in (("CRM snapshot", crm_snapshot_path), ("CRM resolver output", out_path)):
        if path is None:
            continue
        if "stable_runtime" in path.parts:
            raise ValueError(f"{label} must not be under stable_runtime")
        if not path_is_relative_to(path, product_root):
            raise ValueError(f"{label} must stay under product root: {product_root}")
    if not crm_snapshot_path.exists() or not crm_snapshot_path.is_file():
        raise FileNotFoundError(f"CRM snapshot not found: {crm_snapshot_path}")
    return product_db_path, product_root, crm_snapshot_path, out_path


def ensure_table(con: sqlite3.Connection, name: str) -> None:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (clean(name),),
    ).fetchone()
    if row is None:
        raise ValueError(f"required table not found: {name}")


def infer_entity_type(row: Mapping[str, Any]) -> str:
    if clean(row.get("lead_id")):
        return "lead"
    if clean(row.get("contact_id")):
        return "contact"
    return "unknown"


def first_clean(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean(row.get(key))
        if value:
            return value
    return ""


def safety_contract() -> Mapping[str, bool]:
    return {
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "live_crm_reads": False,
        "write_crm": False,
        "write_tallanto": False,
        "run_asr": False,
        "run_ra": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
