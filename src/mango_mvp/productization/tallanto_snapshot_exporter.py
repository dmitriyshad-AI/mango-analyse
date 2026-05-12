from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol
from urllib.parse import quote

from mango_mvp.productization.product_db import guard_product_db_path
from mango_mvp.productization.test_ingest import clean, path_is_relative_to
from mango_mvp.utils.phone import normalize_phone


TALLANTO_SNAPSHOT_EXPORTER_SCHEMA_VERSION = "tallanto_snapshot_exporter_v1"
TALLANTO_PHONE_FIELDS = ("phone_mobile", "phone_work", "phone_home", "phone_other", "phone", "mobile", "telephone")


class TallantoSearchClient(Protocol):
    def search_contacts_by_phone(self, phone: str, *, max_records: int = 20) -> list[Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class TallantoSnapshotExportSummary:
    schema_version: str
    output_path: str
    product_db_path: str
    phones_seen: int
    phones_queried: int
    entities_exported: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def export_tallanto_snapshot(
    product_root: Path,
    product_db_path: Path,
    output_path: Path,
    *,
    client: Optional[TallantoSearchClient] = None,
    env_path: Optional[Path] = None,
    phone_limit: int = 250,
    max_contacts_per_phone: int = 5,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    output_path = output_path.resolve(strict=False)
    env_path = env_path.resolve(strict=False) if env_path else None
    guard_product_db_path(product_db_path, product_root, must_exist=True)
    guard_under_root(output_path, product_root, "Tallanto snapshot output")
    if env_path and not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    if phone_limit < 1:
        raise ValueError("phone_limit must be positive")
    if max_contacts_per_phone < 1:
        raise ValueError("max_contacts_per_phone must be positive")
    if env_path:
        load_env_file(env_path)
    tallanto = client or build_live_client()
    phones = collect_candidate_phones(product_db_path, limit=phone_limit)
    entities = []
    queried = 0
    for phone in phones:
        queried += 1
        for record in tallanto.search_contacts_by_phone(phone, max_records=max_contacts_per_phone):
            entity = snapshot_entity(record, source_phone=phone)
            if entity:
                entities.append(entity)
    entities = dedupe_entities(entities)
    warnings = 1 if not entities else 0
    payload = {
        "schema_version": TALLANTO_SNAPSHOT_EXPORTER_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "provider": "tallanto",
            "mode": "live_read_only_phone_lookup",
            "phone_limit": phone_limit,
            "max_contacts_per_phone": max_contacts_per_phone,
        },
        "entities": entities,
        "summary": TallantoSnapshotExportSummary(
            schema_version=TALLANTO_SNAPSHOT_EXPORTER_SCHEMA_VERSION,
            output_path=str(output_path),
            product_db_path=str(product_db_path),
            phones_seen=len(phones),
            phones_queried=queried,
            entities_exported=len(entities),
            validation_ok=True,
            blocked=0,
            warnings=warnings,
        ).to_json_dict(),
        "safety": safety_contract(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def collect_candidate_phones(product_db_path: Path, *, limit: int) -> list[str]:
    phones = []
    with sqlite3.connect(readonly_uri(product_db_path), uri=True, timeout=15) as con:
        con.row_factory = sqlite3.Row
        if table_exists(con, "capture_inbox_items"):
            rows = con.execute(
                """
                SELECT DISTINCT client_phone
                  FROM capture_inbox_items
                 WHERE client_phone IS NOT NULL AND TRIM(client_phone) != ''
                 ORDER BY client_phone
                 LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            for row in rows:
                phone = normalize_phone(row["client_phone"])
                if phone and phone not in phones:
                    phones.append(phone)
    return phones[:limit]


def snapshot_entity(record: Mapping[str, Any], *, source_phone: str) -> Mapping[str, Any]:
    entity_id = clean(record.get("id") or record.get("contact_id"))
    phones = []
    for field in TALLANTO_PHONE_FIELDS:
        phone = normalize_phone(clean(record.get(field)))
        if phone and phone not in phones:
            phones.append(phone)
    source = normalize_phone(source_phone)
    if source and source not in phones:
        phones.append(source)
    if not entity_id and not phones:
        return {}
    name = clean(record.get("name")) or " ".join(
        part for part in (clean(record.get("last_name")), clean(record.get("first_name")), clean(record.get("middle_name"))) if part
    )
    return {
        "crm_provider": "tallanto",
        "entity_type": "contact",
        "entity_id": entity_id,
        "entity_name": name or None,
        "phones": phones,
        "owner_id": clean(record.get("assigned_user_id") or record.get("owner_id")) or None,
        "owner_name": clean(record.get("assigned_user_name") or record.get("owner_name")) or None,
        "status": clean(record.get("status") or record.get("status_id")) or None,
        "source_ref": f"tallanto:contact:{entity_id}" if entity_id else "tallanto:contact:phone_lookup",
    }


def dedupe_entities(entities: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    result = []
    seen = set()
    for entity in entities:
        signature = clean(entity.get("entity_id")) or json.dumps(entity, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(entity)
    return result


def build_live_client() -> TallantoSearchClient:
    from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, build_tallanto_api_config

    return TallantoApiClient(build_tallanto_api_config())


def load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        os.environ[key.strip()] = value.strip()


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def readonly_uri(path: Path) -> str:
    return f"file:{quote(str(path), safe='/:')}?mode=ro"


def guard_under_root(path: Path, product_root: Path, label: str) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if not path_is_relative_to(path, product_root):
        raise ValueError(f"{label} must stay under product root: {product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "network_read_only": True,
        "product_db_writes": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
    }
