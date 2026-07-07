#!/usr/bin/env python3
"""Prepare staging-local inputs for Customer Timeline nightly D v2.

The script reads local handoff artifacts only and writes normalized JSONL plus
service config under .codex_local/staging. It never opens prod DB, calls APIs,
runs ASR, or invokes LLM.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_SOURCE_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse")
DEFAULT_OUT_ROOT = ROOT / ".codex_local" / "staging" / "nightly_dv2_sources"
DEFAULT_TIMELINE_DB = ROOT / ".codex_local" / "staging" / "customer_timeline_staging.sqlite"
DEFAULT_CURSOR = "2026-06-19T14:53:27+00:00"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build D v2 staging-local nightly source files.")
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--timeline-db", default=str(DEFAULT_TIMELINE_DB))
    parser.add_argument("--mail-cursor", default=DEFAULT_CURSOR)
    parser.add_argument("--service-config-out")
    parser.add_argument("--text-limit", type=int, default=1200)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_root = Path(args.source_root).expanduser().resolve(strict=False)
    out_root = Path(args.out_root).expanduser().resolve(strict=False)
    timeline_db = Path(args.timeline_db).expanduser().resolve(strict=False)
    out_root.mkdir(parents=True, exist_ok=True)
    mail_cursor = parse_dt(args.mail_cursor)

    mail_jsonl = out_root / "mail_archive_stage2_incremental.jsonl"
    mail_manifest = out_root / "mail_archive_stage2_incremental_manifest.json"
    mail_report = build_mail_increment(
        source_root,
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
    )
    config_out = (
        Path(args.service_config_out).expanduser().resolve(strict=False)
        if args.service_config_out
        else out_root / "customer_timeline_nightly_service_dv2_config.json"
    )
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(json.dumps(service_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    source_root: Path,
    *,
    out_jsonl: Path,
    manifest_path: Path,
    since: datetime,
    text_limit: int,
    timeline_db: Path | None = None,
) -> Mapping[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    existing_state = load_existing_mail_link_state(timeline_db) if timeline_db else {}
    stage2_paths = [
        source_root
        / "_external_handoffs/mail_archive_2026-06-20/regru_edu/incremental_20260513_to_20260620/"
        / "stage2_delta_ingest_20260621/stage2_delta_full_events.jsonl",
    ]
    archive_dbs = [
        source_root
        / "_external_handoffs/mail_archive_2026-05-12/regru_edu/increment_since_20260629_20260630_manual/archive/mail_archive.sqlite",
        source_root
        / "_external_handoffs/mail_archive_2026-05-12/regru_edu/increment_since_20260630_20260707_manual/archive/mail_archive.sqlite",
    ]
    inputs: list[Mapping[str, Any]] = []
    for path in stage2_paths:
        count_before = len(rows)
        if path.exists():
            for row in read_jsonl(path):
                event_at = parse_optional_dt(row.get("date_last") or row.get("date_first") or row.get("event_at"))
                if event_at is None or event_at < since:
                    continue
                message_sha = str(row.get("message_sha256") or row.get("sha256") or "").strip()
                if not message_sha or message_sha in seen:
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
        inputs.append({"path": str(path), "exists": path.exists(), "rows_selected": len(rows) - count_before})
    for db_path in archive_dbs:
        count_before = len(rows)
        if db_path.exists():
            for row in read_archive_messages(db_path, since=since, text_limit=text_limit):
                message_sha = str(row.get("message_sha256") or "").strip()
                if not message_sha or message_sha in seen:
                    continue
                seen.add(message_sha)
                rows.append(merge_existing_mail_state(row, existing_state.get(message_sha)))
        inputs.append({"path": str(db_path), "exists": db_path.exists(), "rows_selected": len(rows) - count_before})
    rows.sort(key=lambda item: str(item.get("event_at") or ""))
    write_jsonl(out_jsonl, rows)
    max_event_at = max((str(row.get("event_at") or "") for row in rows), default=None)
    manifest = {
        "schema_version": "mail_archive_stage2_incremental_manifest_v1",
        "cursor_start": since.isoformat(),
        "inputs": inputs,
        "output_jsonl": str(out_jsonl),
        "rows_written": len(rows),
        "linked_rows": sum(1 for row in rows if row.get("customer_id")),
        "pending_rows": sum(1 for row in rows if not row.get("customer_id")),
        "preserved_mail_link_state_rows": sum(1 for row in rows if row.get("mail_link_enrich")),
        "max_event_at": max_event_at,
        "safety": {"network_calls": False, "runs_llm": False, "writes_prod_db": False},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_existing_mail_link_state(timeline_db: Path | None) -> dict[str, Mapping[str, Any]]:
    if not timeline_db or not timeline_db.exists():
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    with sqlite3.connect(f"file:{timeline_db}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        for row in con.execute(
            """
            SELECT source_id, customer_id, match_status, confidence, record_json
            FROM timeline_events
            WHERE source_system = 'mail_archive_stage2'
              AND json_extract(record_json, '$.metadata.mail_link_enrich.outcome') IS NOT NULL
            """
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


def read_archive_messages(db_path: Path, *, since: datetime, text_limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        for row in con.execute(
            """
            SELECT sha256, message_date_iso, subject, message_kind, mailbox, extracted_text_path, updated_at
            FROM messages
            WHERE message_date_iso IS NOT NULL AND message_date_iso != ''
            ORDER BY message_date_iso, sha256
            """
        ):
            event_at = parse_optional_dt(row["message_date_iso"])
            if event_at is None or event_at < since:
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
    snapshot = ROOT / ".codex_local" / "staging" / "block2_tallanto" / "tallanto_money_snapshot.json"
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
) -> Mapping[str, Any]:
    allowed_root = ROOT / ".codex_local" / "staging"
    steps: list[Mapping[str, Any]] = []
    existing_config = allowed_root / "nightly_service" / "customer_timeline_nightly_service_config.json"
    if existing_config.exists():
        payload = json.loads(existing_config.read_text(encoding="utf-8"))
        for step in payload.get("steps") or ():
            if step.get("name") == "calls_and_amo_incremental":
                normalized = dict(step)
                normalized["required"] = True
                normalized["enabled"] = True
                steps.append(normalized)
                break
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
            "required": False,
            "config": {
                "timeline_db": str(timeline_db),
                "allowed_root": str(allowed_root),
                "out_dir": str(out_root / "mail_link_enrich"),
                "tenant_id": "foton",
                "apply": True,
            },
        }
    )
    wappi_metrics = allowed_root / "wappi_history_block4" / "block4_wappi_metrics.json"
    steps.extend(
        [
            monitor_step(
                "tallanto_money_incremental",
                tallanto_manifest,
                cursor_source_system="tallanto_snapshot",
                cursor_ts="2026-05-21T08:59:36+00:00",
                reason="optional_no_nightly_ready_export",
            ),
            monitor_step(
                "wappi_history_incremental",
                wappi_metrics,
                cursor_source_system="wappi_history_pending",
                cursor_ts=DEFAULT_CURSOR,
                reason="optional_pending_only_no_timeline_events",
                deprecated_cursor_source_systems=("wappi_history",),
            ),
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
        "timeline_db": str(timeline_db),
        "allowed_root": str(allowed_root),
        "out_root": str(out_root / "nightly_service_runs"),
        "publish_dir": str(allowed_root / "nightly_service" / "published"),
        "tenant_id": "foton",
        "steps": steps,
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
