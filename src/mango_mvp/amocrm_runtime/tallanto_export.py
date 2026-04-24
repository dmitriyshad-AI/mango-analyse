from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient


DEFAULT_DISCOVERY_MODULES = (
    "Contact",
    "Opportunity",
    "Request",
    "most_finances",
    "most_sip_log",
    "most_courses",
    "CoursesContactsRelationship",
    "ClassContactsRelationship",
    "User",
)

DEFAULT_ENUM_OPTIONS = (
    "filial_list",
    "type_client_list",
    "source_contact_list",
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_tallanto_schema(
    client: TallantoApiClient,
    *,
    modules: Optional[Iterable[str]] = None,
    enum_options: Optional[Iterable[str]] = None,
) -> dict:
    selected_modules = [module for module in (modules or DEFAULT_DISCOVERY_MODULES) if str(module).strip()]
    selected_options = [option for option in (enum_options or DEFAULT_ENUM_OPTIONS) if str(option).strip()]
    module_catalog = client.list_possible_modules()
    field_catalog = {
        module: client.list_possible_fields(module)
        for module in selected_modules
    }
    enum_values = client.list_enum_values(selected_options) if selected_options else {}
    return {
        "generated_at": _iso_now(),
        "base_url": client.config.base_url,
        "module_catalog": module_catalog,
        "modules": selected_modules,
        "fields": field_catalog,
        "enum_values": enum_values,
    }


def export_tallanto_schema_bundle(
    client: TallantoApiClient,
    *,
    output_dir: str | Path,
    modules: Optional[Iterable[str]] = None,
    enum_options: Optional[Iterable[str]] = None,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    bundle = discover_tallanto_schema(
        client,
        modules=modules,
        enum_options=enum_options,
    )
    schema_path = destination / "tallanto_schema_bundle.json"
    modules_path = destination / "tallanto_modules.json"
    fields_path = destination / "tallanto_fields.json"
    enums_path = destination / "tallanto_enum_values.json"

    schema_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    modules_path.write_text(json.dumps(bundle["module_catalog"], ensure_ascii=False, indent=2), encoding="utf-8")
    fields_path.write_text(json.dumps(bundle["fields"], ensure_ascii=False, indent=2), encoding="utf-8")
    enums_path.write_text(json.dumps(bundle["enum_values"], ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "schema_bundle": str(schema_path),
        "module_catalog": str(modules_path),
        "fields": str(fields_path),
        "enum_values": str(enums_path),
    }


def export_module_snapshot(
    client: TallantoApiClient,
    *,
    module: str,
    output_path: str | Path,
    select_fields: Optional[Iterable[str]] = None,
    field_values: Optional[dict[str, object]] = None,
    query: Optional[str] = None,
    order_by: Optional[str] = None,
    max_records: Optional[int] = None,
) -> str:
    records = client.iter_entry_list(
        module=module,
        select_fields=select_fields,
        field_values=field_values,
        query=query,
        order_by=order_by,
        max_records=max_records,
    )
    payload = {
        "generated_at": _iso_now(),
        "base_url": client.config.base_url,
        "module": module,
        "record_count": len(records),
        "records": records,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(destination)


__all__ = [
    "DEFAULT_DISCOVERY_MODULES",
    "DEFAULT_ENUM_OPTIONS",
    "discover_tallanto_schema",
    "export_module_snapshot",
    "export_tallanto_schema_bundle",
]
