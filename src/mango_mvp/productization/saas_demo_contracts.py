from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.productization.test_ingest import clean, path_is_relative_to
from mango_mvp.utils.phone import normalize_phone


SAAS_DEMO_CONTRACTS_SCHEMA_VERSION = "saas_demo_contracts_v1"
SNAPSHOT_STEMS = ("amocrm_entities", "tallanto_entities")
SNAPSHOT_SUFFIXES = (".json", ".jsonl", ".csv")
REQUIRED_DASHBOARD_PANELS = (
    "capture",
    "processing_queue",
    "scheduler",
    "lifecycle",
    "writeback",
    "crm_mapping",
    "gates",
    "knowledge",
    "settings",
)


def build_snapshot_inventory(product_root: Path) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    guard_product_root(product_root)
    files = []
    total_entities = 0
    phones = set()
    for stem in SNAPSHOT_STEMS:
        path = find_snapshot(product_root, stem)
        if path is None:
            continue
        entities = load_snapshot_entities(path)
        total_entities += len(entities)
        for entity in entities:
            phones.update(entity_phones(entity))
        files.append(
            {
                "stem": stem,
                "provider": stem.split("_", 1)[0],
                "path": str(path),
                "format": path.suffix.lower().lstrip("."),
                "entities": len(entities),
            }
        )
    found_stems = {item["stem"] for item in files}
    missing = [stem for stem in SNAPSHOT_STEMS if stem not in found_stems]
    return {
        "schema_version": SAAS_DEMO_CONTRACTS_SCHEMA_VERSION,
        "summary": {
            "snapshot_files": len(files),
            "entities": total_entities,
            "phones_indexed": len(phones),
            "missing_snapshot_stems": missing,
            "validation_ok": True,
            "blocked": 0,
            "warnings": len(missing),
        },
        "files": files,
        "safety": safety_contract(),
    }


def build_dashboard_demo_readiness(
    product_root: Path,
    product_db_path: Path,
    *,
    panels: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    product_root = product_root.resolve(strict=False)
    product_db_path = product_db_path.resolve(strict=False)
    guard_product_root(product_root)
    if "stable_runtime" in product_db_path.parts:
        raise ValueError("demo dashboard readiness refuses product DB under stable_runtime")
    if not path_is_relative_to(product_db_path, product_root):
        raise ValueError(f"product DB must stay under product root: {product_root}")
    panel_keys = set(panels.keys()) if isinstance(panels, Mapping) else set()
    missing_panels = [panel for panel in REQUIRED_DASHBOARD_PANELS if panels is not None and panel not in panel_keys]
    snapshot_inventory = build_snapshot_inventory(product_root)
    demo_artifacts = find_demo_artifacts(product_root)
    blocked_reasons = []
    warning_reasons = []
    if not product_db_path.exists():
        blocked_reasons.append("product_db_missing")
    if missing_panels:
        blocked_reasons.append("dashboard_panels_missing")
    if int(snapshot_inventory["summary"]["snapshot_files"]) == 0:
        warning_reasons.append("crm_snapshots_missing")
    if not demo_artifacts:
        warning_reasons.append("demo_report_missing")
    return {
        "schema_version": SAAS_DEMO_CONTRACTS_SCHEMA_VERSION,
        "summary": {
            "product_db_present": product_db_path.exists() and product_db_path.is_file(),
            "required_panels": len(REQUIRED_DASHBOARD_PANELS),
            "panels_present": len(REQUIRED_DASHBOARD_PANELS) - len(missing_panels),
            "snapshot_files": int(snapshot_inventory["summary"]["snapshot_files"]),
            "snapshot_entities": int(snapshot_inventory["summary"]["entities"]),
            "demo_artifacts": len(demo_artifacts),
            "validation_ok": not blocked_reasons,
            "blocked": len(blocked_reasons),
            "warnings": len(warning_reasons) + int(snapshot_inventory["summary"]["warnings"]),
        },
        "required_panels": list(REQUIRED_DASHBOARD_PANELS),
        "missing_panels": missing_panels,
        "snapshot_inventory": snapshot_inventory,
        "demo_artifacts": demo_artifacts,
        "blocked_reasons": blocked_reasons,
        "warning_reasons": warning_reasons,
        "safety": safety_contract(),
    }


def find_snapshot(product_root: Path, stem: str) -> Optional[Path]:
    for suffix in SNAPSHOT_SUFFIXES:
        path = product_root / "crm_snapshots" / f"{stem}{suffix}"
        if path.exists() and path.is_file():
            return path
    return None


def load_snapshot_entities(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rows = payload.get("entities", [])
    else:
        rows = payload
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def entity_phones(entity: Mapping[str, Any]) -> Sequence[str]:
    raw = []
    phones = entity.get("phones")
    if isinstance(phones, list):
        raw.extend(phones)
    else:
        raw.append(phones)
    raw.extend(entity.get(key) for key in ("phone", "client_phone", "telephone", "mobile") if entity.get(key))
    normalized = []
    for value in raw:
        phone = normalize_phone(clean(value))
        if phone and phone not in normalized:
            normalized.append(phone)
    return tuple(normalized)


def find_demo_artifacts(product_root: Path) -> list[Mapping[str, str]]:
    candidates = (
        product_root / "sanitized_real_demo_report.json",
        product_root / "demo_tenant_report.json",
        product_root / "product_api_readiness" / "sanitized_real_demo_api_readiness.json",
        product_root / "product_api_readiness" / "demo_api_readiness.json",
    )
    return [{"kind": path.stem, "path": str(path)} for path in candidates if path.exists() and path.is_file()]


def guard_product_root(product_root: Path) -> None:
    if "stable_runtime" in product_root.parts:
        raise ValueError("demo contracts refuse product root under stable_runtime")


def safety_contract() -> Mapping[str, bool]:
    return {
        "read_only": True,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "downloads_audio": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
    }
