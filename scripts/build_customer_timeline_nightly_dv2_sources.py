#!/usr/bin/env python3
"""Prepare staging-local inputs for Customer Timeline nightly D v2.

The script reads local handoff artifacts only and writes normalized JSONL plus
service config under .codex_local/staging. It never opens prod DB, calls APIs,
runs ASR, or invokes LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mail_archive import (  # noqa: E402
    CANONICAL_MAIL_ARCHIVE_DB,
    CANONICAL_MAIL_CURRENT_IDENTITY_DB,
    CANONICAL_MAIL_IDENTITY_DB,
    CANONICAL_MAIL_STAGE2_DELTA_EVENTS,
    DEFAULT_MAIL_DATA_ROOT,
)
from mango_mvp.existing_clients.amo_step1_snapshot import DEFAULT_ENV_PATH as DEFAULT_AMO_MCP_ENV  # noqa: E402
from mango_mvp.customer_timeline.store import customer_timeline_readonly_uri  # noqa: E402
from mango_mvp.customer_timeline.nightly_service import (  # noqa: E402
    NIGHTLY_SERVICE_CONFIG_SCHEMA_VERSION,
    REQUIRED_MANIFEST_SOURCE_STEP_MAP,
)

DEFAULT_SOURCE_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse")
MANGO_READY_PACKAGE_DB = (
    DEFAULT_SOURCE_ROOT / "product_data" / "mango_calls_two_processes" / "drop" / "mango_calls_ready.sqlite"
)
DEFAULT_NIGHTLY_HOME = Path(
    os.getenv("CUSTOMER_TIMELINE_NIGHTLY_HOME", "~/.mango_local/customer_timeline_nightly")
).expanduser()
DEFAULT_STAGING_ROOT = DEFAULT_NIGHTLY_HOME / ".codex_local" / "staging"
DEFAULT_OUT_ROOT = DEFAULT_STAGING_ROOT / "nightly_dv2_sources"
DEFAULT_TIMELINE_DB = DEFAULT_STAGING_ROOT / "customer_timeline_staging.sqlite"
DEFAULT_BASE_SERVICE_CONFIG = DEFAULT_STAGING_ROOT / "nightly_service" / "customer_timeline_nightly_service_config.json"
DEFAULT_CURSOR = "2026-06-19T14:53:27+00:00"
DEFAULT_TALLANTO_ATTENDANCE_SINCE = "2026-06-09T00:00:00+03:00"
DEFAULT_TALLANTO_READONLY_ENV = Path("~/.mango_secrets/tallanto_readonly.env").expanduser()
DEFAULT_WAPPI_ENV = Path.home() / ".mango_secrets" / "amo_wappi.env"
DEFAULT_WAPPI_CONFIG = Path.home() / ".mango_secrets" / "amo_wappi_phase1.json"
DEFAULT_WAPPI_PAIRS = Path.home() / ".mango_secrets" / "draft_loop_pairs.json"
DEFAULT_WAPPI_AUTO_PAIRS = Path.home() / ".mango_local" / "draft_loop" / "empty_auto_pairs.json"
DEFAULT_WAPPI_AMO_ENV = Path.home() / ".mango_secrets" / "foton_crm_readonly_mcp_connector.env"
DEFAULT_WAPPI_STOPLIST = Path.home() / ".mango_secrets" / "shared_phones_stoplist.json"
REQUIRED_CALL_SOURCES = {"mango_processed_summary": "mango_processed_summary"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build D v2 staging-local nightly source files.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--mail-data-root", default=str(DEFAULT_MAIL_DATA_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--timeline-db", default=str(DEFAULT_TIMELINE_DB))
    parser.add_argument("--mail-cursor", default=DEFAULT_CURSOR)
    parser.add_argument("--base-service-config", default=str(DEFAULT_BASE_SERVICE_CONFIG))
    parser.add_argument("--service-config-out")
    parser.add_argument("--text-limit", type=int, default=1200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve(strict=False)
    mail_data_root = Path(args.mail_data_root).expanduser().resolve(strict=False)
    out_root = Path(args.out_root).expanduser().resolve(strict=False)
    timeline_db = Path(args.timeline_db).expanduser().resolve(strict=False)
    base_service_config = Path(args.base_service_config).expanduser().resolve(strict=False)
    out_root.mkdir(parents=True, exist_ok=True)
    mail_cursor = parse_dt(args.mail_cursor)

    mail_jsonl = out_root / "mail_archive_stage2_incremental.jsonl"
    mail_manifest = out_root / "mail_archive_stage2_incremental_manifest.json"
    mail_report = build_mail_increment(
        mail_data_root,
        out_jsonl=mail_jsonl,
        manifest_path=mail_manifest,
        since=mail_cursor,
        text_limit=args.text_limit,
        timeline_db=timeline_db,
    )
    mango_manifest = out_root / "mango_api_freshness_manifest.json"
    mango_report = build_mango_freshness(source_root, mango_manifest)
    tallanto_manifest = out_root / "tallanto_freshness_manifest.json"
    tallanto_report = build_tallanto_freshness(tallanto_manifest)
    service_config = build_service_config(
        timeline_db=timeline_db,
        out_root=out_root,
        mail_jsonl=mail_jsonl,
        mail_manifest=mail_manifest,
        mango_manifest=mango_manifest,
        tallanto_manifest=tallanto_manifest,
        base_service_config=base_service_config,
    )
    config_out = (
        Path(args.service_config_out).expanduser().resolve(strict=False)
        if args.service_config_out
        else out_root / "customer_timeline_nightly_service_dv2_config.json"
    )
    config_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_config = config_out.with_suffix(config_out.suffix + ".tmp")
    temporary_config.write_text(
        json.dumps(service_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_config.replace(config_out)
    report = {
        "schema_version": "customer_timeline_nightly_dv2_source_builder_v1",
        "source_root": str(source_root),
        "out_root": str(out_root),
        "timeline_db": str(timeline_db),
        "service_config": str(config_out),
        "mail": mail_report,
        "mango_api_freshness": mango_report,
        "tallanto_freshness": tallanto_report,
        "safety": {
            "writes_prod_db": False,
            "opens_prod_db": False,
            "network_calls": False,
            "runs_asr": False,
            "runs_llm": False,
            "writes_amo": False,
            "writes_tallanto": False,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_mail_increment(
    mail_data_root: Path,
    *,
    out_jsonl: Path,
    manifest_path: Path,
    since: datetime,
    text_limit: int,
    timeline_db: Path | None = None,
    archive_db_paths: Sequence[Path] | None = None,
    missing_only: bool = False,
    tenant_id: str = "foton",
) -> Mapping[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    existing_state = load_existing_mail_link_state(timeline_db, tenant_id=tenant_id) if timeline_db else {}
    existing_source_ids = (
        load_existing_mail_source_ids(timeline_db, tenant_id=tenant_id) if missing_only else set()
    )
    existing_skipped = 0
    fallback_date_rows = 0
    stage2_paths = [
        mail_data_root / CANONICAL_MAIL_STAGE2_DELTA_EVENTS,
    ]
    archive_dbs = (
        list(archive_db_paths)
        if archive_db_paths is not None
        else [mail_data_root / CANONICAL_MAIL_ARCHIVE_DB]
    )
    missing_archive_dbs = [path for path in archive_dbs if not path.is_file()]
    if not archive_dbs or missing_archive_dbs:
        missing = missing_archive_dbs or [mail_data_root / CANONICAL_MAIL_ARCHIVE_DB]
        raise FileNotFoundError("required mail archive input is missing: " + ", ".join(map(str, missing)))
    inputs: list[Mapping[str, Any]] = []
    for path in stage2_paths:
        count_before = len(rows)
        if path.is_file():
            for row in read_jsonl(path):
                event_at = parse_optional_dt(
                    row.get("date_last")
                    or row.get("date_first")
                    or row.get("event_at")
                    or row.get("updated_at")
                )
                if event_at is None or (not missing_only and event_at < since):
                    continue
                message_sha = str(row.get("message_sha256") or row.get("sha256") or "").strip()
                if not message_sha or message_sha in seen:
                    continue
                if message_sha in existing_source_ids:
                    seen.add(message_sha)
                    existing_skipped += 1
                    continue
                seen.add(message_sha)
                text = str(row.get("thread_summary") or row.get("summary") or row.get("subject") or "").strip()
                rows.append(
                    merge_existing_mail_state(
                        {
                            "source_id": message_sha,
                            "source_ref": f"mail_stage2:{path.name}:{message_sha[:16]}",
                            "message_sha256": message_sha,
                            "event_at": event_at.isoformat(),
                            "updated_at": event_at.isoformat(),
                            "date_first": row.get("date_first"),
                            "date_last": row.get("date_last"),
                            "customer_id": row.get("customer_id") or None,
                            "subject": row.get("subject") or "Email message",
                            "summary": text[:text_limit],
                            "text_preview": text[:240],
                            "brand": row.get("brand") or "unknown",
                            "summary_status": row.get("summary_status") or "stage2_handoff",
                            "needs_summary_later": False,
                            "source_file": str(path),
                        },
                        existing_state.get(message_sha),
                    )
                )
        inputs.append({"path": str(path), "exists": path.is_file(), "rows_selected": len(rows) - count_before})
    for db_path in archive_dbs:
        count_before = len(rows)
        for row in read_archive_messages(
            db_path,
            since=None if missing_only else since,
            text_limit=text_limit,
        ):
            message_sha = str(row.get("message_sha256") or "").strip()
            if not message_sha or message_sha in seen:
                continue
            if message_sha in existing_source_ids:
                seen.add(message_sha)
                existing_skipped += 1
                continue
            seen.add(message_sha)
            if row.pop("_fallback_date", False):
                fallback_date_rows += 1
            rows.append(merge_existing_mail_state(row, existing_state.get(message_sha)))
        inputs.append({"path": str(db_path), "exists": True, "rows_selected": len(rows) - count_before})
    rows.sort(key=lambda item: str(item.get("event_at") or ""))
    write_jsonl(out_jsonl, rows)
    max_event_at = max((str(row.get("event_at") or "") for row in rows), default=None)
    manifest = {
        "schema_version": "mail_archive_stage2_incremental_manifest_v1",
        "tenant_id": tenant_id,
        "cursor_start": since.isoformat(),
        "inputs": inputs,
        "output_jsonl": str(out_jsonl),
        "rows_written": len(rows),
        "missing_only": missing_only,
        "existing_skipped": existing_skipped,
        "fallback_date_rows": fallback_date_rows,
        "linked_rows": sum(1 for row in rows if row.get("customer_id")),
        "pending_rows": sum(1 for row in rows if not row.get("customer_id")),
        "preserved_mail_link_state_rows": sum(1 for row in rows if row.get("mail_link_enrich")),
        "max_event_at": max_event_at,
        "safety": {"network_calls": False, "runs_llm": False, "writes_prod_db": False},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_existing_mail_link_state(
    timeline_db: Path | None,
    *,
    tenant_id: str = "foton",
) -> dict[str, Mapping[str, Any]]:
    if not timeline_db or not timeline_db.exists():
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    with _connect_timeline_ro(timeline_db) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        for row in con.execute(
            """
            SELECT source_id, customer_id, match_status, confidence, record_json
            FROM timeline_events
            WHERE tenant_id = ? AND source_system = 'mail_archive_stage2'
            """,
            (tenant_id,),
        ):
            try:
                payload = json.loads(row["record_json"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else {}
            if not isinstance(metadata, Mapping):
                metadata = {}
            result[str(row["source_id"])] = {
                "customer_id": row["customer_id"],
                "match_status": row["match_status"],
                "confidence": row["confidence"],
                "pending_attribution": metadata.get("pending_attribution"),
                "pending_reason": metadata.get("pending_reason"),
                "fresh_relink": metadata.get("fresh_relink"),
                "mail_link_enrich": metadata.get("mail_link_enrich"),
                "brand": metadata.get("brand"),
            }
    return result


def load_existing_mail_source_ids(
    timeline_db: Path | None,
    *,
    tenant_id: str = "foton",
) -> set[str]:
    if not timeline_db or not timeline_db.exists():
        return set()
    with _connect_timeline_ro(timeline_db) as con:
        con.execute("PRAGMA query_only=ON")
        return {
            str(row[0])
            for row in con.execute(
                "SELECT source_id FROM timeline_events "
                "WHERE tenant_id=? AND source_system='mail_archive_stage2'",
                (tenant_id,),
            )
            if row[0]
        }


def _connect_timeline_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(customer_timeline_readonly_uri(path), uri=True)


def merge_existing_mail_state(row: dict[str, Any], state: Mapping[str, Any] | None) -> dict[str, Any]:
    if not state:
        return row
    for key in (
        "customer_id",
        "match_status",
        "confidence",
        "pending_attribution",
        "pending_reason",
        "fresh_relink",
        "mail_link_enrich",
    ):
        if state.get(key) is not None:
            row[key] = state[key]
    if state.get("brand") and str(state["brand"]) != "unknown":
        row["brand"] = state["brand"]
    return row


def read_archive_messages(db_path: Path, *, since: datetime | None, text_limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        for row in con.execute(
            """
            SELECT sha256, message_date_iso, subject, message_kind, mailbox, extracted_text_path,
                   updated_at, first_ingested_at
            FROM messages
            ORDER BY COALESCE(message_date_iso, updated_at, first_ingested_at), sha256
            """
        ):
            primary_event_at = parse_optional_dt(row["message_date_iso"])
            event_at = primary_event_at or parse_optional_dt(row["updated_at"]) or parse_optional_dt(row["first_ingested_at"])
            if event_at is None or (since is not None and event_at < since):
                continue
            text = read_text_preview(row["extracted_text_path"], text_limit) or str(row["subject"] or "").strip()
            sha = str(row["sha256"] or "").strip()
            result.append(
                {
                    "source_id": sha,
                    "source_ref": f"mail_stage2:{db_path.parent.parent.name}:{sha[:16]}",
                    "message_sha256": sha,
                    "event_at": event_at.isoformat(),
                    "updated_at": parse_optional_dt(row["updated_at"]).isoformat()
                    if parse_optional_dt(row["updated_at"])
                    else event_at.isoformat(),
                    "subject": row["subject"] or "Email message",
                    "summary": text[:text_limit],
                    "text_preview": text[:240],
                    "message_kind": row["message_kind"],
                    "mailbox": row["mailbox"],
                    "brand": "unknown",
                    "summary_status": "needs_summary_later",
                    "needs_summary_later": True,
                    "source_db": str(db_path),
                    "_fallback_date": primary_event_at is None,
                }
            )
    return result


def build_mango_freshness(source_root: Path, manifest_path: Path) -> Mapping[str, Any]:
    product_data = source_root / "product_data"
    roots = sorted(product_data.glob("mango_update_after_202607*")) if product_data.exists() else []
    items = [
        {
            "path": str(path),
            "name": path.name,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "has_plan_summary": (path / "plan_summary.json").exists(),
            "has_ra_final_summary": any(path.rglob("RA_FINAL_SUMMARY.json")),
        }
        for path in roots
    ]
    report = {
        "schema_version": "mango_api_freshness_manifest_v1",
        "roots_count": len(items),
        "latest": items[-1] if items else None,
        "items": items,
        "safety": {"network_calls": False, "runs_asr": False},
    }
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_tallanto_freshness(manifest_path: Path) -> Mapping[str, Any]:
    staging_root = manifest_path.parent.parent
    snapshot = staging_root / "block2_tallanto" / "tallanto_money_snapshot.json"
    report = {
        "schema_version": "tallanto_freshness_manifest_v1",
        "snapshot_path": str(snapshot),
        "snapshot_exists": snapshot.exists(),
        "status": "optional_skip_no_nightly_ready_export",
        "cursor_tallanto_snapshot": "2026-05-21T08:59:36+00:00",
        "cursor_tallanto_crm_call": "2026-06-04T16:54:54+00:00",
        "safety": {"network_calls": False, "writes_tallanto": False},
    }
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_service_config(
    *,
    timeline_db: Path,
    out_root: Path,
    mail_jsonl: Path,
    mail_manifest: Path,
    mango_manifest: Path,
    tallanto_manifest: Path,
    base_service_config: Path | None = None,
) -> Mapping[str, Any]:
    allowed_root = timeline_db.parent.resolve(strict=False)
    steps: list[Mapping[str, Any]] = []
    mango_sweep_jsonl = out_root / "mango_processed_sweep.jsonl"
    steps.append(
        {
            "name": "mango_processed_sweep",
            "kind": "mango_processed_sweep",
            "enabled": True,
            "required": True,
            "config": {
                "producer_script": str(ROOT / "scripts" / "build_mango_call_timeline_increment.py"),
                "scan_roots": [str(Path("/Users/dmitrijfabarisov/Projects/Mango analyse/product_data"))],
                "package_globs": ["mango_update_after_*"],
                "package_dbs": [str(MANGO_READY_PACKAGE_DB)],
                "out_jsonl": str(mango_sweep_jsonl),
                "report_out": str(out_root / "mango_processed_sweep_producer_report.json"),
                "manifest_path": str(out_root / "mango_processed_sweep_manifest.json"),
                "inventory_out": str(out_root / "mango_processed_sweep_inventory.json"),
            },
        }
    )
    existing_config = base_service_config or (
        allowed_root / "nightly_service" / "customer_timeline_nightly_service_config.json"
    )
    payload = json.loads(existing_config.read_text(encoding="utf-8")) if existing_config.is_file() else {}
    calls_step: Mapping[str, Any] | None = None
    for step in payload.get("steps") or ():
        if isinstance(step, Mapping) and step.get("name") == "calls_and_amo_incremental":
            calls_step = step
            break
    normalized = json.loads(json.dumps(calls_step)) if calls_step is not None else {
        "name": "calls_and_amo_incremental",
        "kind": "nightly_incremental",
        "config": {
            "journal_path": str(out_root / "calls_and_amo_incremental_journal.jsonl"),
            "safety_margin_seconds": 0,
            "sources": [
                {
                    "name": "mango_processed_sweep",
                    "source_system": "mango_processed_summary",
                    "path": str(mango_sweep_jsonl),
                    "source_ref": "mango:processed_sweep:latest",
                    "normalizer": "mango_processed_summary",
                    "required": True,
                }
            ],
        },
    }
    normalized["required"] = True
    normalized["enabled"] = True
    call_sources = {
        str(source.get("source_system") or ""): source
        for source in normalized.get("config", {}).get("sources", [])
        if isinstance(source, Mapping)
    }
    missing_call_sources = sorted(REQUIRED_CALL_SOURCES.keys() - call_sources.keys())
    if missing_call_sources:
        raise RuntimeError(
            "calls_and_amo_incremental misses required sources: " + ",".join(missing_call_sources)
        )
    for source_system, normalizer in REQUIRED_CALL_SOURCES.items():
        source = call_sources[source_system]
        if source.get("normalizer") != normalizer or source.get("required") is not True:
            raise RuntimeError(f"calls_and_amo_incremental source contract is invalid: {source_system}")
        source_path = Path(str(source.get("path") or "")).expanduser().resolve(strict=False)
        try:
            source_path.relative_to(allowed_root)
        except ValueError as exc:
            raise RuntimeError(
                f"calls_and_amo_incremental source is outside persistent staging root: {source_system}"
            ) from exc
    journal_path = Path(str(normalized.get("config", {}).get("journal_path") or "")).expanduser().resolve(
        strict=False
    )
    try:
        journal_path.relative_to(allowed_root)
    except ValueError as exc:
        raise RuntimeError("calls_and_amo_incremental journal is outside persistent staging root") from exc
    mango_source_found = False
    for source in normalized.get("config", {}).get("sources", []):
        if source.get("source_system") == "mango_processed_summary":
            mango_source_found = True
            source["name"] = "mango_processed_sweep"
            source["path"] = str(mango_sweep_jsonl)
            source["source_ref"] = "mango:processed_sweep:latest"
            source["ignore_cursor"] = True
            source["preserve_cursor"] = True
    if not mango_source_found:
        raise RuntimeError("calls_and_amo_incremental misses mango_processed_summary source")
    steps.append(normalized)
    steps.append(
        {
            "name": "amo_incremental_shadow",
            "kind": "amo_incremental",
            "enabled": True,
            "required": True,
            "config": {
                "source_db": str(timeline_db),
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "out_root": str(out_root / "amo_incremental_shadow"),
                "mcp_env": str(DEFAULT_AMO_MCP_ENV),
                "safety_overlap_seconds": 300,
                "page_limit": 50,
                "max_pages": 20,
                "sleep_sec": 1.05,
            },
        }
    )
    steps.append(
        {
            "name": "wappi_history_incremental",
            "kind": "wappi_history",
            "enabled": True,
            "required": True,
            "config": {
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "env_file": str(DEFAULT_WAPPI_ENV),
                "phase1_config": str(DEFAULT_WAPPI_CONFIG),
                "pairs_file": str(DEFAULT_WAPPI_PAIRS),
                "auto_pairs_file": str(DEFAULT_WAPPI_AUTO_PAIRS),
                "amo_mcp_env_file": str(DEFAULT_WAPPI_AMO_ENV),
                "shared_phone_stoplist": str(DEFAULT_WAPPI_STOPLIST),
                "amo_auto_resolver_enabled": True,
                "widget_link_db": str(allowed_root / "wappi_amo_links.sqlite"),
                "apply": True,
                "require_nonempty_profiles": True,
                "require_widget_linkage": True,
                "refresh_widget_links": True,
                "chat_limit_per_profile": 5000,
                "messages_per_chat": 100,
                "message_limit_total": 50000,
                "request_limit_total": 50000,
                "page_size": 100,
                "sleep_seconds": 0.2,
                "show_all_chats": True,
                "complete_message_history": True,
            },
        }
    )
    steps.append(
        {
            "name": "mail_archive_incremental",
            "kind": "nightly_incremental",
            "enabled": True,
            "required": True,
            "config": {
                "journal_path": str(out_root / "mail_archive_incremental_journal.jsonl"),
                "safety_margin_seconds": 0,
                "sources": [
                    {
                        "name": "mail_archive_stage2_incremental",
                        "source_system": "mail_archive_stage2",
                        "path": str(mail_jsonl),
                        "source_ref": "nightly_dv2:mail_archive_stage2",
                        "normalizer": "mail_archive_stage2",
                        "required": True,
                    }
                ],
            },
        }
    )
    steps.append(
        {
            "name": "mail_link_enrich",
            "kind": "mail_link_enrich",
            "enabled": True,
            "required": True,
            "config": {
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "out_dir": str(out_root / "mail_link_enrich"),
                "tenant_id": "foton",
                "apply": True,
                "tallanto_identity_dbs": [
                    str(DEFAULT_MAIL_DATA_ROOT / CANONICAL_MAIL_CURRENT_IDENTITY_DB),
                    str(DEFAULT_MAIL_DATA_ROOT / CANONICAL_MAIL_IDENTITY_DB),
                ],
            },
        }
    )
    steps.append(
        {
            "name": "tallanto_money_api_incremental",
            "kind": "tallanto_money_api",
            "enabled": True,
            "required": True,
            "config": {
                "importer_script": str(ROOT / "scripts" / "import_tallanto_payments_to_timeline.py"),
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "tallanto_env_file": str(DEFAULT_TALLANTO_READONLY_ENV),
                "tenant_id": "foton",
                "apply": True,
            },
        }
    )
    steps.append(
        {
            "name": "tallanto_attendance_api_incremental",
            "kind": "tallanto_attendance_api",
            "enabled": True,
            "required": True,
            "config": {
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "tallanto_env_file": str(DEFAULT_TALLANTO_READONLY_ENV),
                "initial_since": DEFAULT_TALLANTO_ATTENDANCE_SINCE,
                "tenant_id": "foton",
                "apply": True,
            },
        }
    )
    steps.insert(
        next(index for index, step in enumerate(steps) if step.get("name") == "tallanto_attendance_api_incremental"),
        {
            "name": "tallanto_cards_sync",
            "kind": "tallanto_cards",
            "enabled": True,
            "required": True,
            "config": {
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "out_root": str(out_root / "tallanto_cards_sync"),
                "tallanto_env_file": str(DEFAULT_TALLANTO_READONLY_ENV),
                "tenant_id": "foton",
                "max_pages": 20,
            },
        },
    )
    steps.append(
        {
            "name": "family_graph_refresh",
            "kind": "family_graph",
            "enabled": True,
            "required": True,
            "config": {
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "out_path": str(out_root / "family_graph_refresh.json"),
                "tenant_id": "foton",
                "apply": True,
            },
        }
    )
    steps.append(
        {
            "name": "bot_safe_rebuild",
            "kind": "bot_safe_rebuild",
            "enabled": True,
            "required": True,
            "config": {
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "tenant_id": "foton",
                "apply": True,
            },
        }
    )
    steps.extend(
        [
            monitor_step(
                "mango_api_freshness",
                mango_manifest,
                cursor_source_system="mango_api_freshness",
                cursor_ts=DEFAULT_CURSOR,
                reason="optional_local_daily_capture_monitor_only",
            ),
        ]
    )
    return {
        # B1: explicit config schema version. validate_nightly_config() in
        # scripts/run_customer_timeline_codex_task.py rejects any on-disk
        # config whose version does not match, so a config written by an
        # older copy of this builder (e.g. before required_manifest_sources
        # existed) is rebuilt by ensure_nightly_config() instead of silently
        # passing preflight.
        "config_schema_version": NIGHTLY_SERVICE_CONFIG_SCHEMA_VERSION,
        "timeline_db": str(timeline_db),
        "allowed_root": str(allowed_root),
        "out_root": str(out_root / "nightly_service_runs"),
        "publish_dir": str(allowed_root / "nightly_service" / "published"),
        "tenant_id": "foton",
        "steps": steps,
        # B2/B1: the 10 mandatory business sources the nightly manifest must
        # attest to -- sourced from nightly_service.REQUIRED_MANIFEST_SOURCE_STEP_MAP
        # itself (not a hand-copied literal) so this list can never silently
        # drift out of sync with the gate that enforces it.
        "required_manifest_sources": list(REQUIRED_MANIFEST_SOURCE_STEP_MAP.keys()),
    }


def monitor_step(
    name: str,
    metrics_path: Path,
    *,
    cursor_source_system: str,
    cursor_ts: str,
    reason: str,
    deprecated_cursor_source_systems: Sequence[str] = (),
) -> Mapping[str, Any]:
    return {
        "name": name,
        "kind": "local_freshness_monitor",
        "enabled": True,
        "required": False,
        "config": {
            "metrics_path": str(metrics_path),
            "paths": [str(metrics_path)],
            "cursor_source_system": cursor_source_system,
            "cursor_ts": cursor_ts,
            "reason": reason,
            "deprecated_cursor_source_systems": list(deprecated_cursor_source_systems),
            "empty_status": "skipped",
        },
    }


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                parsed = json.loads(text)
                if isinstance(parsed, Mapping):
                    result.append(parsed)
    return result


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_text_preview(path: Any, limit: int) -> str:
    if not path:
        return ""
    candidate = Path(str(path))
    try:
        text = candidate.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return " ".join(text.split())[:limit]


def parse_dt(raw: str) -> datetime:
    parsed = parse_optional_dt(raw)
    if parsed is None:
        raise ValueError(f"invalid datetime: {raw!r}")
    return parsed


def parse_optional_dt(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


if __name__ == "__main__":
    raise SystemExit(main())
