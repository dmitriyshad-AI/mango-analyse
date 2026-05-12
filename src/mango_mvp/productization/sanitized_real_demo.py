from __future__ import annotations

import hashlib
import csv
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import quote

from mango_mvp.productization.product_api import build_product_api_readiness_report
from mango_mvp.productization.product_db import audit_product_db, initialize_product_db
from mango_mvp.productization.test_ingest import clean, path_is_relative_to
from mango_mvp.utils.phone import normalize_phone


SANITIZED_REAL_DEMO_SCHEMA_VERSION = "sanitized_real_demo_v1"
SENSITIVE_SNAPSHOT_STEMS = ("amocrm_entities", "tallanto_entities")


@dataclass(frozen=True)
class SanitizedRealDemoSummary:
    schema_version: str
    source_product_db_path: str
    demo_product_root: str
    demo_product_db_path: str
    tenants: int
    product_calls: int
    capture_inbox_items: int
    job_runs: int
    snapshots_written: int
    validation_ok: bool
    blocked: int
    warnings: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return asdict(self)


class Sanitizer:
    def __init__(self, salt: str) -> None:
        self.salt = clean(salt) or "mango-sanitized-real-demo-v1"
        self.maps: dict[str, dict[str, str]] = {}

    def value(self, namespace: str, original: Any, prefix: str, width: int = 4) -> str:
        text = clean(original)
        if not text:
            return ""
        mapping = self.maps.setdefault(namespace, {})
        if text not in mapping:
            mapping[text] = f"{prefix}{len(mapping) + 1:0{width}d}"
        return mapping[text]

    def phone(self, original: Any) -> str:
        phone = normalize_phone(clean(original))
        if not phone:
            return ""
        mapping = self.maps.setdefault("phone", {})
        if phone not in mapping:
            mapping[phone] = f"+7999000{len(mapping) + 1:04d}"
        return mapping[phone]

    def email(self, original: Any, prefix: str = "user") -> str:
        text = clean(original)
        if not text:
            return ""
        return f"{self.value('email', text, prefix.upper() + '-', 4).lower()}@demo.local"

    def short_hash(self, original: Any) -> str:
        text = f"{self.salt}:{clean(original)}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    def id_int(self, namespace: str, original: Any, base: int = 900000) -> Optional[int]:
        text = clean(original)
        if not text:
            return None
        mapping = self.maps.setdefault(namespace, {})
        if text not in mapping:
            mapping[text] = str(base + len(mapping) + 1)
        return int(mapping[text])

    def report(self) -> Mapping[str, Any]:
        return {
            "phones_masked": len(self.maps.get("phone", {})),
            "managers_masked": len(self.maps.get("manager_extension", {})),
            "provider_call_ids_masked": len(self.maps.get("provider_call_id", {})),
            "recording_ids_masked": len(self.maps.get("recording_id", {})),
            "crm_entities_masked": len(self.maps.get("crm_entity_id", {})),
        }


def build_sanitized_real_demo_root(
    source_product_root: Path,
    source_product_db_path: Path,
    demo_product_root: Path,
    out_path: Optional[Path] = None,
    *,
    replace_existing: bool = False,
    salt: str = "mango-sanitized-real-demo-v1",
    row_limit: Optional[int] = None,
) -> Mapping[str, Any]:
    source_product_root = source_product_root.resolve(strict=False)
    source_product_db_path = source_product_db_path.resolve(strict=False)
    demo_product_root = demo_product_root.resolve(strict=False)
    demo_product_db_path = demo_product_root / "mango_product_appliance.sqlite"
    out_path = (out_path or demo_product_root / "sanitized_real_demo_report.json").resolve(strict=False)
    guard_source_paths(source_product_root, source_product_db_path)
    guard_demo_source_separation(source_product_root, source_product_db_path, demo_product_root, demo_product_db_path)
    guard_demo_path(demo_product_root, demo_product_root, "demo product root", allow_root=True)
    guard_demo_path(out_path, demo_product_root, "sanitized demo report")
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be positive")

    sanitizer = Sanitizer(salt)
    init = initialize_product_db(demo_product_db_path, demo_product_root, replace_existing=replace_existing)
    copy_report = copy_sanitized_product_db(
        source_db_path=source_product_db_path,
        demo_db_path=demo_product_db_path,
        sanitizer=sanitizer,
        row_limit=row_limit,
    )
    snapshots = write_sanitized_snapshots(
        source_product_root=source_product_root,
        demo_product_root=demo_product_root,
        sanitizer=sanitizer,
    )
    integrity = audit_product_db(demo_product_db_path, demo_product_root)
    api = build_product_api_readiness_report(
        product_root=demo_product_root,
        product_db_path=demo_product_db_path,
        out_path=demo_product_root / "product_api_readiness" / "sanitized_real_demo_api_readiness.json",
    )
    summary = SanitizedRealDemoSummary(
        schema_version=SANITIZED_REAL_DEMO_SCHEMA_VERSION,
        source_product_db_path=str(source_product_db_path),
        demo_product_root=str(demo_product_root),
        demo_product_db_path=str(demo_product_db_path),
        tenants=int(integrity["summary"]["tenants"]),
        product_calls=int(integrity["summary"]["product_calls"]),
        capture_inbox_items=int(integrity["summary"]["capture_inbox_items"]),
        job_runs=int(integrity["summary"]["job_runs"]),
        snapshots_written=len(snapshots),
        validation_ok=bool(integrity["summary"]["validation_ok"]) and bool(api["summary"]["validation_ok"]),
        blocked=int(integrity["summary"]["blocked"]),
        warnings=int(integrity["summary"]["warnings"]) + int(copy_report["warnings"]),
    )
    report = {
        "summary": summary.to_json_dict(),
        "initialization": init["summary"],
        "copy": copy_report,
        "snapshots": snapshots,
        "integrity": integrity,
        "api_readiness": api["summary"],
        "sanitizer": sanitizer.report(),
        "demo_commands": {
            "serve_dashboard": (
                "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 "
                f"scripts/mango_office_product_api_http.py --product-root {demo_product_root} "
                f"--product-db {demo_product_db_path} serve --host 127.0.0.1 --port 8765"
            ),
            "open_dashboard": "http://127.0.0.1:8765/dashboard",
        },
        "safety": safety_contract(),
    }
    write_json(out_path, report)
    return report


def copy_sanitized_product_db(
    *,
    source_db_path: Path,
    demo_db_path: Path,
    sanitizer: Sanitizer,
    row_limit: Optional[int],
) -> Mapping[str, Any]:
    with sqlite3.connect(readonly_uri(source_db_path), uri=True, timeout=15) as src, sqlite3.connect(str(demo_db_path)) as dst:
        src.row_factory = sqlite3.Row
        dst.row_factory = sqlite3.Row
        dst.execute("PRAGMA foreign_keys = OFF")
        copied = {}
        skipped_columns: dict[str, list[str]] = {}
        for table in (
            "tenants",
            "provider_accounts",
            "crm_accounts",
            "tenant_manager_owner_map",
            "job_types",
            "product_calls",
            "job_runs",
            "capture_inbox_items",
            "retention_policies",
        ):
            if not table_exists(src, table):
                copied[table] = 0
                continue
            rows = read_rows(src, table, row_limit=row_limit if table in {"product_calls", "capture_inbox_items"} else None)
            transformed = [sanitize_row(table, dict(row), sanitizer) for row in rows]
            transformed, skipped = keep_target_columns(dst, table, transformed)
            insert_rows(dst, table, transformed)
            copied[table] = len(transformed)
            if skipped:
                skipped_columns[table] = skipped
        dst.commit()
    return {
        "tables_copied": copied,
        "warnings": len(skipped_columns),
        "skipped_extra_source_columns": skipped_columns,
        "tenant_config_history_copied": False,
        "row_limit": row_limit,
    }


def sanitize_row(table: str, row: dict[str, Any], sanitizer: Sanitizer) -> dict[str, Any]:
    if table == "tenants":
        tenant = sanitizer.value("tenant_id", row.get("tenant_id"), "demo_tenant_", 3)
        row["tenant_id"] = tenant
        row["display_name"] = tenant.replace("_", " ").title()
    elif table in {"provider_accounts", "crm_accounts", "retention_policies"}:
        row["tenant_id"] = sanitizer.value("tenant_id", row.get("tenant_id"), "demo_tenant_", 3)
        if "config_ref" in row:
            row["config_ref"] = "sanitized/config_ref.json" if clean(row.get("config_ref")) else None
    elif table == "tenant_manager_owner_map":
        row["tenant_id"] = sanitizer.value("tenant_id", row.get("tenant_id"), "demo_tenant_", 3)
        row["manager_extension"] = sanitizer.value("manager_extension", row.get("manager_extension"), "M", 3)
        row["mango_name"] = f"Manager {row['manager_extension']}"
        row["mango_email"] = sanitizer.email(row.get("mango_email"), prefix="manager") if clean(row.get("mango_email")) else None
        row["crm_owner_id"] = sanitizer.id_int("crm_owner_id", row.get("crm_owner_id"))
        row["crm_owner_name"] = f"CRM Owner {row['manager_extension']}" if row.get("crm_owner_id") else None
        row["crm_owner_email"] = sanitizer.email(row.get("crm_owner_email"), prefix="owner") if clean(row.get("crm_owner_email")) else None
        row["source_ref"] = "sanitized/source"
        row["config_ref"] = "sanitized/config"
        row["notes"] = "sanitized real demo"
    elif table == "product_calls":
        row["tenant_id"] = sanitizer.value("tenant_id", row.get("tenant_id"), "demo_tenant_", 3)
        original_call_id = row.get("provider_call_id")
        original_recording_id = row.get("recording_id")
        call_id = sanitizer.value("provider_call_id", original_call_id, "CALL-", 6)
        recording_id = sanitizer.value("recording_id", original_recording_id, "REC-", 6) if clean(original_recording_id) else None
        manager = sanitizer.value("manager_extension", row.get("manager_extension"), "M", 3) if clean(row.get("manager_extension")) else None
        row["provider_call_id"] = call_id
        row["event_key"] = f"{row['tenant_id']}:{clean(row.get('telephony_provider')) or 'mango'}:{call_id}"
        row["recording_id"] = recording_id
        row["source_filename"] = sanitized_filename(row.get("started_at"), call_id, recording_id)
        row["manager_extension"] = manager
        row["manager_display_name"] = f"Manager {manager}" if manager else None
        row["crm_owner_id"] = sanitizer.id_int("crm_owner_id", row.get("crm_owner_id"))
        row["crm_owner_name"] = f"CRM Owner {manager}" if row.get("crm_owner_id") and manager else None
        row["raw_payload_ref"] = "sanitized/raw_payload.jsonl" if clean(row.get("raw_payload_ref")) else None
        row["source_repository_ref"] = "sanitized_real_demo"
    elif table == "job_runs":
        row["tenant_id"] = sanitizer.value("tenant_id", row.get("tenant_id"), "demo_tenant_", 3) if clean(row.get("tenant_id")) else None
        row["input_ref"] = sanitized_json_ref(row.get("input_ref"), sanitizer)
        row["output_ref"] = "sanitized/scheduler_output.json" if clean(row.get("output_ref")) else None
        row["error"] = "sanitized_error_sample" if clean(row.get("error")) else None
        if "lock_owner" in row and clean(row.get("lock_owner")):
            row["lock_owner"] = "sanitized-worker"
        if "result_json" in row and clean(row.get("result_json")):
            row["result_json"] = json.dumps({"sanitized": True}, ensure_ascii=False, sort_keys=True)
    elif table == "capture_inbox_items":
        row["tenant_id"] = sanitizer.value("tenant_id", row.get("tenant_id"), "demo_tenant_", 3)
        call_id = sanitizer.value("provider_call_id", row.get("provider_call_id"), "CALL-", 6)
        recording_id = sanitizer.value("recording_id", row.get("recording_ref"), "REC-", 6) if clean(row.get("recording_ref")) else None
        manager = sanitizer.value("manager_extension", row.get("manager_ref"), "M", 3) if clean(row.get("manager_ref")) else None
        row["event_key"] = f"{row['tenant_id']}:{clean(row.get('provider')) or 'mango'}:{call_id}"
        row["provider_call_id"] = call_id
        row["source_report_ref"] = "sanitized/source_report.json" if clean(row.get("source_report_ref")) else None
        row["raw_payload_ref"] = "sanitized/raw_payload.jsonl" if clean(row.get("raw_payload_ref")) else None
        row["client_phone"] = sanitizer.phone(row.get("client_phone")) or None
        row["manager_ref"] = manager
        row["recording_ref"] = recording_id
        row["recording_url"] = None
        row["audio_ref"] = recording_id
        row["candidate_json"] = json.dumps({"sanitized": True}, sort_keys=True)
        row["event_json"] = json.dumps({"sanitized": True}, sort_keys=True)
        row["reserved_by"] = "sanitized-worker" if clean(row.get("reserved_by")) else None
        row["error"] = "sanitized_error_sample" if clean(row.get("error")) else None
    return row


def sanitized_filename(started_at: Any, call_id: str, recording_id: Optional[str]) -> str:
    date = clean(started_at)[:10] or "unknown-date"
    rec = recording_id or "NO-REC"
    return f"{date}__sanitized__{call_id}__{rec}.mp3"


def sanitized_json_ref(value: Any, sanitizer: Sanitizer) -> Optional[str]:
    text = clean(value)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return "sanitized/input_ref"
    if isinstance(parsed, Mapping):
        return json.dumps(sanitize_json_payload(parsed, sanitizer), ensure_ascii=False, sort_keys=True)
    return json.dumps({"sanitized": True}, ensure_ascii=False, sort_keys=True)


def sanitize_json_payload(payload: Mapping[str, Any], sanitizer: Sanitizer) -> Mapping[str, Any]:
    result = {}
    for key, value in payload.items():
        key_text = clean(key)
        if key_text in {"tenant_id"}:
            result[key_text] = sanitizer.value("tenant_id", value, "demo_tenant_", 3)
        elif "path" in key_text or key_text.endswith("_dir") or key_text.endswith("_ref"):
            result[key_text] = f"sanitized/{key_text}"
        elif "credential" in key_text or "token" in key_text or "key" in key_text:
            result[key_text] = "masked"
        elif isinstance(value, Mapping):
            result[key_text] = sanitize_json_payload(value, sanitizer)
        elif isinstance(value, list):
            result[key_text] = ["sanitized" if not isinstance(item, Mapping) else sanitize_json_payload(item, sanitizer) for item in value]
        else:
            result[key_text] = value
    return result


def write_sanitized_snapshots(
    *,
    source_product_root: Path,
    demo_product_root: Path,
    sanitizer: Sanitizer,
) -> list[Mapping[str, Any]]:
    written = []
    for stem in SENSITIVE_SNAPSHOT_STEMS:
        source = find_snapshot(source_product_root, stem)
        if source is None:
            continue
        entities = load_snapshot_entities(source)
        sanitized_entities = [sanitize_snapshot_entity(entity, sanitizer, provider=stem.split("_", 1)[0]) for entity in entities]
        out = demo_product_root / "crm_snapshots" / f"{stem}.json"
        write_json(
            out,
            {
                "schema_version": SANITIZED_REAL_DEMO_SCHEMA_VERSION,
                "source": {"kind": "sanitized_real_snapshot", "source_stem": stem},
                "entities": sanitized_entities,
            },
        )
        written.append({"source": str(source), "out": str(out), "entities": len(sanitized_entities)})
    return written


def find_snapshot(source_product_root: Path, stem: str) -> Optional[Path]:
    for suffix in (".json", ".jsonl", ".csv"):
        path = source_product_root / "crm_snapshots" / f"{stem}{suffix}"
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


def sanitize_snapshot_entity(entity: Mapping[str, Any], sanitizer: Sanitizer, *, provider: str) -> Mapping[str, Any]:
    entity_id = clean(entity.get("entity_id")) or clean(entity.get("id"))
    entity_type = clean(entity.get("entity_type")) or clean(entity.get("type")) or "contact"
    phones = []
    entity_phones = entity.get("phones")
    raw_phones = list(entity_phones) if isinstance(entity_phones, list) else [entity_phones]
    raw_phones.extend(entity.get(key) for key in ("phone", "client_phone", "telephone", "mobile") if entity.get(key))
    for phone in raw_phones:
        masked = sanitizer.phone(phone)
        if masked and masked not in phones:
            phones.append(masked)
    masked_id = sanitizer.value("crm_entity_id", f"{provider}:{entity_type}:{entity_id}", "CRM-", 6) if entity_id else ""
    return {
        "crm_provider": clean(entity.get("crm_provider")) or clean(entity.get("provider")) or provider,
        "entity_type": entity_type,
        "entity_id": masked_id,
        "entity_name": f"{entity_type.title()} {masked_id}" if masked_id else None,
        "phones": phones,
        "owner_id": sanitizer.value("crm_owner_snapshot_id", entity.get("owner_id"), "OWNER-", 5) if clean(entity.get("owner_id")) else None,
        "owner_name": "Sanitized Owner" if clean(entity.get("owner_name")) else None,
        "status": clean(entity.get("status")) or None,
        "source_ref": "sanitized/snapshot",
    }


def read_rows(con: sqlite3.Connection, table: str, row_limit: Optional[int] = None) -> list[sqlite3.Row]:
    sql = f"SELECT * FROM {table}"
    params: tuple[Any, ...] = ()
    if row_limit is not None:
        sql += " LIMIT ?"
        params = (int(row_limit),)
    return list(con.execute(sql, params).fetchall())


def insert_rows(con: sqlite3.Connection, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    sql = f"INSERT OR REPLACE INTO {table} ({column_sql}) VALUES ({placeholders})"
    con.executemany(sql, [tuple(row.get(column) for column in columns) for row in rows])


def keep_target_columns(
    con: sqlite3.Connection,
    table: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[str]]:
    if not rows:
        return [], []
    target_columns = table_columns(con, table)
    kept = [{column: row.get(column) for column in target_columns if column in row} for row in rows]
    skipped = sorted({column for row in rows for column in row.keys() if column not in target_columns})
    return kept, skipped


def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in con.execute(f"PRAGMA table_info({table})").fetchall()]


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def readonly_uri(path: Path) -> str:
    return f"file:{quote(str(path), safe='/:')}?mode=ro"


def guard_source_paths(source_product_root: Path, source_product_db_path: Path) -> None:
    if "stable_runtime" in source_product_root.parts or "stable_runtime" in source_product_db_path.parts:
        raise ValueError("sanitized real demo source must not be stable_runtime")
    if source_product_db_path.name in {"mango_mvp.db", "ai_office.db"}:
        raise ValueError("sanitized real demo source must be product DB, not runtime DB")
    if not path_is_relative_to(source_product_db_path, source_product_root):
        raise ValueError(f"source product DB must stay under source product root: {source_product_root}")
    if not source_product_db_path.exists() or not source_product_db_path.is_file():
        raise FileNotFoundError(f"source product DB not found: {source_product_db_path}")


def guard_demo_source_separation(
    source_product_root: Path,
    source_product_db_path: Path,
    demo_product_root: Path,
    demo_product_db_path: Path,
) -> None:
    if demo_product_db_path == source_product_db_path:
        raise ValueError("demo product DB must be separate from source product DB")
    if demo_product_root == source_product_root:
        raise ValueError("demo product root must be separate from source product root")
    if path_is_relative_to(demo_product_root, source_product_root):
        raise ValueError("demo product root must not be inside source product root")
    if path_is_relative_to(source_product_root, demo_product_root):
        raise ValueError("source product root must not be inside demo product root")


def guard_demo_path(path: Path, demo_product_root: Path, label: str, *, allow_root: bool = False) -> None:
    if "stable_runtime" in path.parts:
        raise ValueError(f"{label} must not be under stable_runtime")
    if allow_root and path == demo_product_root:
        return
    if not path_is_relative_to(path, demo_product_root):
        raise ValueError(f"{label} must stay under demo product root: {demo_product_root}")


def safety_contract() -> Mapping[str, bool]:
    return {
        "reads_source_product_db": True,
        "reads_runtime_db": False,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "copies_audio": False,
        "downloads_audio": False,
        "live_crm_reads": False,
        "write_crm": False,
        "write_tallanto": False,
        "run_asr": False,
        "run_ra": False,
        "contains_real_personal_data": False,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
