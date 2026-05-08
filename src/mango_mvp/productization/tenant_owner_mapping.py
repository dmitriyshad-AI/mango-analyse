from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.manager_identity import MANAGER_IDENTITY_TABLE
from mango_mvp.productization.repository import ManagerRollupItem, ProductRepository
from mango_mvp.productization.test_ingest import clean, path_is_relative_to


TENANT_OWNER_MAPPING_SCHEMA_VERSION = "tenant_owner_mapping_v1"


@dataclass(frozen=True)
class TenantOwnerMappingSummary:
    schema_version: str
    db_path: str
    manager_extensions: int
    confirmed_candidates: int
    manual_decisions_required: int
    calls_confirmed: int
    calls_pending: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def build_tenant_owner_mapping_draft(
    db_path: Path,
    out_allowed_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    repo = ProductRepository(db_path=db_path, out_allowed_root=out_allowed_root)
    rows = list(repo.manager_rollup())
    items = [draft_item(row) for row in rows]
    manual_items = [item for item in items if item["decision_status"] == "needs_manual_owner"]
    confirmed = [item for item in items if item["decision_status"] == "confirmed_candidate"]
    summary = TenantOwnerMappingSummary(
        schema_version=TENANT_OWNER_MAPPING_SCHEMA_VERSION,
        db_path=str(repo.db_path),
        manager_extensions=len(items),
        confirmed_candidates=len(confirmed),
        manual_decisions_required=len(manual_items),
        calls_confirmed=sum(int(item["call_count"]) for item in confirmed),
        calls_pending=sum(int(item["call_count"]) for item in manual_items),
        validation_ok=True,
        blocked=0,
        warnings=len(manual_items),
    )
    report = {
        "summary": summary.to_json_dict(),
        "items": items,
        "manual_review_items": manual_items,
        "config_template": build_config_template(items),
    }
    if out_path:
        write_json_under_root(report, out_path.resolve(strict=False), repo.out_allowed_root)
    return report


def apply_tenant_owner_config_dry_run(
    db_path: Path,
    config_path: Path,
    out_allowed_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    repo = ProductRepository(db_path=db_path, out_allowed_root=out_allowed_root)
    config = load_config(config_path)
    rows_by_extension = {row.manager_extension: row for row in repo.manager_rollup()}
    actions = []
    blocked = 0
    for entry in config.get("manager_owner_overrides", []):
        action = validate_override_entry(entry, rows_by_extension)
        if action["action"] == "BLOCK_INVALID_OVERRIDE":
            blocked += 1
        actions.append(action)
    report = {
        "summary": {
            "schema_version": TENANT_OWNER_MAPPING_SCHEMA_VERSION,
            "db_path": str(repo.db_path),
            "config_path": str(config_path.resolve(strict=False)),
            "overrides": len(config.get("manager_owner_overrides", [])),
            "would_update": sum(1 for action in actions if action["action"] == "WOULD_SET_OWNER"),
            "would_confirm_existing": sum(1 for action in actions if action["action"] == "WOULD_CONFIRM_EXISTING"),
            "blocked": blocked,
            "validation_ok": blocked == 0,
        },
        "actions": actions,
    }
    if out_path:
        write_json_under_root(report, out_path.resolve(strict=False), repo.out_allowed_root)
    return report


def apply_tenant_owner_config(
    db_path: Path,
    config_path: Path,
    out_allowed_root: Path,
    out_path: Optional[Path] = None,
) -> Mapping[str, Any]:
    dry_run = apply_tenant_owner_config_dry_run(
        db_path=db_path,
        config_path=config_path,
        out_allowed_root=out_allowed_root,
    )
    if not dry_run["summary"]["validation_ok"]:
        if out_path:
            write_json_under_root(dry_run, out_path.resolve(strict=False), out_allowed_root.resolve(strict=False))
        return dry_run

    db_path = db_path.resolve(strict=False)
    out_allowed_root = out_allowed_root.resolve(strict=False)
    if not path_is_relative_to(db_path, out_allowed_root):
        raise ValueError(f"tenant owner DB must stay under allowed root: {out_allowed_root}")
    applied = 0
    with sqlite3.connect(str(db_path)) as con:
        for action in dry_run["actions"]:
            if action["action"] != "WOULD_SET_OWNER":
                continue
            con.execute(
                f"""
                UPDATE {MANAGER_IDENTITY_TABLE}
                   SET crm_owner_id = ?,
                       crm_owner_name = ?,
                       crm_owner_email = ?,
                       crm_match_status = 'manual_override',
                       notes = ?
                 WHERE tenant_id = ?
                   AND provider = ?
                   AND manager_extension = ?
                """,
                (
                    action["crm_owner_id"],
                    action["crm_owner_name"],
                    action["crm_owner_email"],
                    action["notes"],
                    action["tenant_id"],
                    action["provider"],
                    action["manager_extension"],
                ),
            )
            applied += 1
        con.commit()

    report = dict(dry_run)
    report["summary"] = dict(report["summary"])
    report["summary"]["applied"] = applied
    report["summary"]["dry_run"] = False
    if out_path:
        write_json_under_root(report, out_path.resolve(strict=False), out_allowed_root)
    return report


def draft_item(row: ManagerRollupItem) -> Mapping[str, Any]:
    has_owner = row.crm_owner_id is not None
    return {
        "tenant_id": row.tenant_id,
        "provider": row.provider,
        "manager_extension": row.manager_extension,
        "call_count": row.call_count,
        "mango_name": row.mango_name,
        "mango_email": row.mango_email,
        "crm_owner_id": row.crm_owner_id,
        "crm_owner_name": row.crm_owner_name,
        "crm_owner_email": row.crm_owner_email,
        "crm_match_status": row.crm_match_status,
        "mapping_status": row.mapping_status,
        "decision_status": "confirmed_candidate" if has_owner else "needs_manual_owner",
        "required_action": "confirm_or_override" if has_owner else "set_crm_owner",
    }


def build_config_template(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "schema_version": TENANT_OWNER_MAPPING_SCHEMA_VERSION,
        "tenant_id": items[0]["tenant_id"] if items else "",
        "provider": items[0]["provider"] if items else "",
        "manager_owner_overrides": [
            {
                "manager_extension": item["manager_extension"],
                "mango_name": item["mango_name"],
                "mango_email": item["mango_email"],
                "crm_owner_id": item["crm_owner_id"],
                "crm_owner_name": item["crm_owner_name"],
                "crm_owner_email": item["crm_owner_email"],
                "decision_status": item["decision_status"],
                "confirmed_by": "",
                "notes": "",
            }
            for item in items
        ],
    }


def load_config(config_path: Path) -> Mapping[str, Any]:
    config_path = config_path.resolve(strict=False)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("tenant owner config must be a JSON object")
    if clean(payload.get("schema_version")) != TENANT_OWNER_MAPPING_SCHEMA_VERSION:
        raise ValueError(f"tenant owner config schema_version must be {TENANT_OWNER_MAPPING_SCHEMA_VERSION}")
    overrides = payload.get("manager_owner_overrides")
    if not isinstance(overrides, list):
        raise ValueError("tenant owner config must contain manager_owner_overrides list")
    return payload


def validate_override_entry(
    entry: Mapping[str, Any],
    rows_by_extension: Mapping[str, ManagerRollupItem],
) -> Mapping[str, Any]:
    extension = clean(entry.get("manager_extension"))
    row = rows_by_extension.get(extension)
    if not row:
        return {
            "action": "BLOCK_INVALID_OVERRIDE",
            "reason": "unknown_manager_extension",
            "manager_extension": extension,
        }
    crm_owner_id = optional_int(entry.get("crm_owner_id"))
    crm_owner_name = clean(entry.get("crm_owner_name"))
    crm_owner_email = clean(entry.get("crm_owner_email"))
    if crm_owner_id is None or not crm_owner_name:
        return {
            "action": "BLOCK_INVALID_OVERRIDE",
            "reason": "crm_owner_id_and_name_required",
            "tenant_id": row.tenant_id,
            "provider": row.provider,
            "manager_extension": extension,
        }
    if row.crm_owner_id == crm_owner_id:
        return {
            "action": "WOULD_CONFIRM_EXISTING",
            "tenant_id": row.tenant_id,
            "provider": row.provider,
            "manager_extension": extension,
            "crm_owner_id": crm_owner_id,
            "crm_owner_name": crm_owner_name,
            "crm_owner_email": crm_owner_email or None,
            "notes": clean(entry.get("notes")) or None,
        }
    return {
        "action": "WOULD_SET_OWNER",
        "tenant_id": row.tenant_id,
        "provider": row.provider,
        "manager_extension": extension,
        "crm_owner_id": crm_owner_id,
        "crm_owner_name": crm_owner_name,
        "crm_owner_email": crm_owner_email or None,
        "notes": clean(entry.get("notes")) or "tenant_owner_manual_override",
    }


def write_json_under_root(report: Mapping[str, Any], out_path: Path, out_allowed_root: Path) -> None:
    if not path_is_relative_to(out_path, out_allowed_root):
        raise ValueError(f"tenant owner output must stay under allowed root: {out_allowed_root}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def optional_int(value: Any) -> int | None:
    text = clean(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
