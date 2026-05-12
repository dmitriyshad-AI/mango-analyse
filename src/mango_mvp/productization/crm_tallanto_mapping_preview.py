from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to
from mango_mvp.utils.phone import normalize_phone


CRM_TALLANTO_MAPPING_PREVIEW_SCHEMA_VERSION = "crm_tallanto_mapping_preview_v1"


@dataclass(frozen=True)
class CrmTallantoMappingPreviewSummary:
    schema_version: str
    product_db_path: str
    capture_rows_seen: int
    amo_snapshot_entities: int
    tallanto_snapshot_entities: int
    amo_resolved: int
    amo_missing: int
    amo_ambiguous: int
    tallanto_resolved: int
    tallanto_missing: int
    tallanto_ambiguous: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_crm_tallanto_mapping_preview(
    product_db_path: Path,
    product_root: Path,
    out_path: Optional[Path] = None,
    *,
    amo_snapshot_path: Optional[Path] = None,
    tallanto_snapshot_path: Optional[Path] = None,
    limit: int = 100,
) -> Mapping[str, Any]:
    product_db_path = product_db_path.resolve(strict=False)
    product_root = product_root.resolve(strict=False)
    out_path = out_path.resolve(strict=False) if out_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    if limit < 1:
        raise ValueError("limit must be positive")
    if out_path:
        guard_under_root(out_path, product_root, "CRM/Tallanto mapping output")
    amo_snapshot_path = resolve_snapshot(product_root, amo_snapshot_path, "amocrm_entities")
    tallanto_snapshot_path = resolve_snapshot(product_root, tallanto_snapshot_path, "tallanto_entities")
    amo_entities = load_snapshot_entities(amo_snapshot_path, provider="amocrm") if amo_snapshot_path else []
    tallanto_entities = load_snapshot_entities(tallanto_snapshot_path, provider="tallanto") if tallanto_snapshot_path else []
    amo_index = build_phone_index(amo_entities)
    tallanto_index = build_phone_index(tallanto_entities)
    rows = read_capture_rows(product_db_path, limit=limit)
    previews = [build_row_preview(row, amo_index=amo_index, tallanto_index=tallanto_index) for row in rows]
    amo_counts = count_status(previews, "amo_status")
    tallanto_counts = count_status(previews, "tallanto_status")
    warnings = (
        int(amo_snapshot_path is None)
        + int(tallanto_snapshot_path is None)
        + amo_counts["missing"]
        + amo_counts["ambiguous"]
        + tallanto_counts["ambiguous"]
    )
    report = {
        "summary": CrmTallantoMappingPreviewSummary(
            schema_version=CRM_TALLANTO_MAPPING_PREVIEW_SCHEMA_VERSION,
            product_db_path=str(product_db_path),
            capture_rows_seen=len(rows),
            amo_snapshot_entities=len(amo_entities),
            tallanto_snapshot_entities=len(tallanto_entities),
            amo_resolved=amo_counts["resolved"],
            amo_missing=amo_counts["missing"],
            amo_ambiguous=amo_counts["ambiguous"],
            tallanto_resolved=tallanto_counts["resolved"],
            tallanto_missing=tallanto_counts["missing"],
            tallanto_ambiguous=tallanto_counts["ambiguous"],
            validation_ok=True,
            blocked=0,
            warnings=warnings,
        ).to_json_dict(),
        "snapshot_paths": {
            "amocrm": str(amo_snapshot_path) if amo_snapshot_path else None,
            "tallanto": str(tallanto_snapshot_path) if tallanto_snapshot_path else None,
        },
        "field_policy": {
            "policy_doc": "docs/AMO_TALLANTO_FIELD_MAPPING_PROD.md",
            "amo_write_mode": "preview_only",
            "tallanto_write_mode": "read_only_context_provider",
            "protected_tallanto_fields": ["Id Tallanto", "Филиал Tallanto"],
        },
        "previews": previews,
        "safety": safety_contract(),
    }
    if out_path:
        write_json(out_path, report)
    return report


def resolve_snapshot(product_root: Path, explicit_path: Optional[Path], stem: str) -> Optional[Path]:
    candidates = [explicit_path.resolve(strict=False)] if explicit_path else [
        product_root / "crm_snapshots" / f"{stem}.json",
        product_root / "crm_snapshots" / f"{stem}.jsonl",
    ]
    for path in candidates:
        if path is None:
            continue
        guard_under_root(path, product_root, f"{stem} snapshot")
        if path.exists() and path.is_file():
            return path
    return None


def load_snapshot_entities(path: Path, *, provider: str) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("entities", []) if isinstance(payload, Mapping) else payload
    entities = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        phones = normalize_entity_phones(row)
        if not phones:
            continue
        entities.append(
            {
                "provider": clean(row.get("crm_provider")) or clean(row.get("provider")) or provider,
                "entity_type": clean(row.get("entity_type")) or clean(row.get("type")) or "contact",
                "entity_id": clean(row.get("entity_id")) or clean(row.get("id")),
                "entity_name": clean(row.get("entity_name")) or clean(row.get("name")) or None,
                "phones": phones,
                "source_ref": clean(row.get("source_ref")) or str(path),
            }
        )
    return entities


def normalize_entity_phones(row: Mapping[str, Any]) -> list[str]:
    raw_values: list[Any] = []
    phones = row.get("phones")
    if isinstance(phones, list):
        raw_values.extend(phones)
    elif phones:
        raw_values.append(phones)
    for key in ("phone", "client_phone", "telephone", "mobile"):
        value = row.get(key)
        if value:
            raw_values.append(value)
    normalized = []
    for value in raw_values:
        phone = normalize_phone(str(value))
        if phone and phone not in normalized:
            normalized.append(phone)
    return normalized


def build_phone_index(entities: Sequence[Mapping[str, Any]]) -> Mapping[str, list[Mapping[str, Any]]]:
    index: dict[str, list[Mapping[str, Any]]] = {}
    for entity in entities:
        for phone in entity.get("phones", []):
            index.setdefault(clean(phone), []).append(entity)
    return index


def read_capture_rows(product_db_path: Path, *, limit: int) -> list[Mapping[str, Any]]:
    uri = f"file:{quote(str(product_db_path), safe='/:')}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT id, tenant_id, provider, event_key, provider_call_id, status,
                   started_at, client_phone, manager_ref, recording_ref
              FROM capture_inbox_items
             WHERE client_phone IS NOT NULL
               AND client_phone != ''
             ORDER BY last_seen_at DESC, id DESC
             LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def build_row_preview(
    row: Mapping[str, Any],
    *,
    amo_index: Mapping[str, list[Mapping[str, Any]]],
    tallanto_index: Mapping[str, list[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    phone = normalize_phone(clean(row.get("client_phone")))
    amo_status, amo_entity = resolve_phone(phone, amo_index)
    tallanto_status, tallanto_entity = resolve_phone(phone, tallanto_index)
    return {
        "capture_id": row.get("id"),
        "event_key": clean(row.get("event_key")),
        "provider_call_id": clean(row.get("provider_call_id")),
        "status": clean(row.get("status")),
        "started_at": clean(row.get("started_at")) or None,
        "manager_ref": clean(row.get("manager_ref")) or None,
        "client_phone": phone or clean(row.get("client_phone")),
        "amo_status": amo_status,
        "amo_entity": compact_entity(amo_entity),
        "tallanto_status": tallanto_status,
        "tallanto_entity": compact_entity(tallanto_entity),
        "write_crm": False,
        "write_tallanto": False,
    }


def resolve_phone(phone: str, index: Mapping[str, list[Mapping[str, Any]]]) -> tuple[str, Optional[Mapping[str, Any]]]:
    if not phone:
        return "missing", None
    matches = index.get(phone, [])
    if len(matches) == 1:
        return "resolved", matches[0]
    if len(matches) > 1:
        return "ambiguous", None
    return "missing", None


def compact_entity(entity: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not entity:
        return None
    return {
        "provider": entity.get("provider"),
        "entity_type": entity.get("entity_type"),
        "entity_id": entity.get("entity_id"),
        "entity_name": entity.get("entity_name"),
    }


def count_status(previews: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, int]:
    return {
        "resolved": sum(1 for row in previews if row.get(key) == "resolved"),
        "missing": sum(1 for row in previews if row.get(key) == "missing"),
        "ambiguous": sum(1 for row in previews if row.get(key) == "ambiguous"),
    }


def guard_under_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


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
