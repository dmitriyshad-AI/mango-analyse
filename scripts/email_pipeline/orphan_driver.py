#!/usr/bin/env python3
"""Classify mail summary cache rows that are not present in mail_archive_stage2.

The script is deliberately dry-run only: it prepares owner-review tables for a
later approved ingest, but never writes Customer Timeline events.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.email_pipeline.archive_sources import default_archive_specs, resolve_data_path, scan_eml_metadata
from scripts.email_pipeline.classification import (
    ClassificationInput,
    OWN_DOMAINS,
    build_outbound_template_counts,
    classify_message,
    domain_of,
    local_of,
    norm_subject,
    participants_for,
)


SOURCE_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse")
STAGING_DB = ROOT / ".codex_local" / "staging" / "customer_timeline_staging.sqlite"
FOTON_DAILY = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/_daily")


def discover_archive_dbs(source_root: Path) -> list[Path]:
    return [spec.path for spec in default_archive_specs(source_root) if spec.path.is_file()]


def load_stage2_source_ids(db_path: Path) -> set[str]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        return {
            str(row[0])
            for row in con.execute(
                "SELECT source_id FROM timeline_events WHERE source_system = 'mail_archive_stage2'"
            )
            if row[0]
        }


def load_cache_rows(db_path: Path) -> dict[str, Mapping[str, Any]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        return {
            str(sha): {
                "source_kind": str(source_kind or ""),
                "summary_payload": _loads(payload),
            }
            for sha, source_kind, payload in con.execute(
                "SELECT message_sha256, source_kind, summary_payload_json FROM email_summary_cache_v1"
            )
        }


def _loads(value: str | None) -> Mapping[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def owner_bucket(klass: str, *, source_kind: str) -> str:
    if klass == "real_correspondence":
        return "real_manager_review" if source_kind == "llm_review_needed" else "real_candidate"
    if klass == "bounce":
        return "delivery_failure_skip"
    if klass in {"bulk_newsletter", "internal", "outbound_campaign"}:
        return "broadcast_or_internal_skip"
    return "service_auto_skip"


def classify_orphans(
    *,
    timeline_db: Path,
    source_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    cache_rows = load_cache_rows(timeline_db)
    stage2_ids = load_stage2_source_ids(timeline_db)
    orphan_shas = set(cache_rows) - stage2_ids
    archive_dbs = discover_archive_dbs(source_root)
    outbound_templates = {
        subject for subject, count in build_outbound_template_counts(archive_dbs).items() if count >= 10
    }
    found: dict[str, dict[str, Any]] = {}
    archive_hits: Counter[str] = Counter()

    for db_path in archive_dbs:
        if not orphan_shas - set(found):
            break
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
            con.execute("PRAGMA query_only=ON")
            participants = participants_for(con)
            placeholders = ",".join("?" for _ in orphan_shas - set(found))
            if not placeholders:
                continue
            rows = con.execute(
                "SELECT sha256, subject, mailbox, message_kind, message_date_iso, extracted_text_chars, raw_eml_path "
                f"FROM messages WHERE sha256 IN ({placeholders})",
                tuple(orphan_shas - set(found)),
            ).fetchall()
            for sha, subject, mailbox, kind, date_iso, body_chars, raw_eml_path in rows:
                record = participants.get(sha, {"from": None, "to": [], "cc": []})
                from_record = record.get("from") or ("", "", "")
                to_records = tuple(record.get("to") or [])
                from_email = str(from_record[1] or "")
                from_domain = str(from_record[2] or domain_of(from_email)).lower()
                to_domains = tuple(str(item[2] or domain_of(item[1])).lower() for item in to_records)
                is_outbound = (
                    from_domain in OWN_DOMAINS
                    or mailbox in ("Sent", "Sent Messages", "Drafts", "Templates")
                    or "Шаблоны" in str(mailbox or "")
                )
                resolved_eml = resolve_data_path(raw_eml_path, source_root=source_root, repo_root=repo_root)
                eml_meta = scan_eml_metadata(resolved_eml)
                eml_flags = {"list_unsub": False, "bulk": False, "auto": False, "campaign": False}
                if not is_outbound and kind != "internal" and from_domain not in OWN_DOMAINS:
                    eml_flags = {key: bool(eml_meta.get(key)) for key in ("list_unsub", "bulk", "auto", "campaign")}
                klass, reason = classify_message(
                    ClassificationInput(
                        kind=str(kind or ""),
                        mailbox=str(mailbox or ""),
                        from_email=from_email,
                        from_dom=from_domain,
                        from_local=local_of(from_email),
                        to_doms=to_domains,
                        subject=str(subject or ""),
                        body_chars=int(body_chars or 0),
                        eml_flags=eml_flags,
                        is_outbound=is_outbound,
                    ),
                    outbound_templates,
                )
                source_kind = str(cache_rows[str(sha)].get("source_kind") or "")
                payload = cache_rows[str(sha)].get("summary_payload")
                found[str(sha)] = {
                    "message_sha256": str(sha),
                    "archive_db": str(db_path),
                    "date_iso": str(date_iso or ""),
                    "subject": str(subject or ""),
                    "mailbox": str(mailbox or ""),
                    "from_domain": from_domain,
                    "to_domains": list(to_domains),
                    "direction": "outbound" if is_outbound else "inbound",
                    "archive_class": klass,
                    "archive_reason": reason,
                    "owner_bucket": owner_bucket(klass, source_kind=source_kind),
                    "source_kind": source_kind,
                    "event_type": str(payload.get("event_type") or ""),
                    "memory_status": str(payload.get("memory_status") or payload.get("status") or ""),
                    "summary": str(payload.get("summary") or ""),
                }
                archive_hits[str(db_path)] += 1

    rows = list(found.values())
    return {
        "schema_version": "orphan_mail_owner_report_v3",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeline_db": str(timeline_db),
        "m1_cache_rows": len(cache_rows),
        "stage2_source_ids_total": len(stage2_ids),
        "orphan_count": len(orphan_shas),
        "orphan_messages_found_in_archives": len(found),
        "orphan_messages_not_found_in_archives": len(orphan_shas) - len(found),
        "archive_dbs_scanned": len(archive_dbs),
        "archive_class_counts": dict(Counter(row["archive_class"] for row in rows)),
        "archive_reason_top": dict(Counter(row["archive_reason"] for row in rows).most_common(20)),
        "owner_bucket_counts": dict(Counter(row["owner_bucket"] for row in rows)),
        "source_kind_counts": dict(Counter(row["source_kind"] for row in rows)),
        "event_type_counts": dict(Counter(row["event_type"] for row in rows)),
        "raw_header_sources": {
            "archive_dbs_scanned": len(archive_dbs),
            "top_archive_hits": dict(archive_hits.most_common(20)),
        },
        "recommendation": (
            "Owner should approve real_candidate/real_manager_review rows before any event creation. "
            "This driver is dry-run only and creates no Customer Timeline events."
        ),
        "safety": {
            "client_sends": False,
            "writes_crm": False,
            "writes_prod_db": False,
            "writes_staging_db": False,
            "writes_tallanto": False,
            "pii_scope": ".codex_local only",
        },
        "rows_sensitive": rows,
    }


def write_outputs(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sensitive_rows = list(report.get("rows_sensitive") or [])
    public_report = dict(report)
    public_report.pop("rows_sensitive", None)
    report_path = out_dir / "orphan_owner_report_v3.json"
    rows_path = out_dir / "orphan_rows_sensitive.jsonl"
    report_path.write_text(json.dumps(public_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in sensitive_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    daily_path = write_daily(public_report, report_path, rows_path)
    return report_path, rows_path, daily_path


def write_daily(report: Mapping[str, Any], report_path: Path, rows_path: Path) -> Path:
    FOTON_DAILY.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = FOTON_DAILY / f"{stamp}_orphan_mail_owner_review.md"
    lines = [
        "# Orphan mail owner review",
        "",
        f"- m1_cache_rows: `{report.get('m1_cache_rows')}`",
        f"- orphan_count: `{report.get('orphan_count')}`",
        f"- found_in_archives: `{report.get('orphan_messages_found_in_archives')}`",
        f"- archive_class_counts: `{json.dumps(report.get('archive_class_counts'), ensure_ascii=False, sort_keys=True)}`",
        f"- owner_bucket_counts: `{json.dumps(report.get('owner_bucket_counts'), ensure_ascii=False, sort_keys=True)}`",
        f"- writes: prod=0, staging=0, CRM=0, Tallanto=0, client_sends=0",
        f"- local_public_report: `{report_path}`",
        f"- local_sensitive_rows: `{rows_path}`",
        "",
        "Apply не запускался: реальные orphan-события заводить только после просмотра owner-таблицы.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run classify mail orphan rows by raw archive envelope/header.")
    parser.add_argument("--timeline-db", type=Path, default=STAGING_DB)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = classify_orphans(timeline_db=args.timeline_db, source_root=args.source_root, repo_root=ROOT)
    report_path, rows_path, daily_path = write_outputs(report, args.out_dir)
    public = dict(report)
    public.pop("rows_sensitive", None)
    public["report_path"] = str(report_path)
    public["rows_sensitive_path"] = str(rows_path)
    public["daily_summary_path"] = str(daily_path)
    print(json.dumps(public, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
