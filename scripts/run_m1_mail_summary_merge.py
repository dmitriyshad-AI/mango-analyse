#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.customer_timeline.a2_mail_ingest import (
    A2V3MailIngestConfig,
    apply_a2v3_mail_ingest,
    create_test_db_backup,
    ensure_not_prod_apply_path,
    validate_a2v3_mail_ingest,
)
from mango_mvp.customer_timeline.mail_link_enrich import (
    _archive_db_for_source_payload,
    _contact_from_archive_row,
    _source_payload,
)
from mango_mvp.customer_timeline.stage4b_bot_opening import (
    Stage4BBotOpeningConfig,
    run_stage4b_bot_opening,
)
from mango_mvp.customer_timeline.store import json_loads

from scripts.run_marathon2_mail_summary_enrich import (
    PROMPT_VERSION,
    SOURCE_SYSTEM,
    _brand_for_row,
    _cache_counts,
    _compact_apply,
    _compact_validation,
    _ensure_cache_table,
    _is_summary_review_needed_payload,
    _sanitize_summary_payload_for_stage2,
    _write_jsonl,
)
from scripts.email_pipeline.quality import evaluate_quality, quality_to_dict, sanitize_summary_payload_for_quality
from scripts.email_pipeline.summary import split_thread_context


SCHEMA_VERSION = "m1_mail_summary_merge_v1"
DEFAULT_ARCHIVE = Path(
    "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/mail_masspack/return/"
    "m1_full_mail_summary_return_mail_full_20260708_001339.tar.gz"
)
DEFAULT_EXTERNAL_MANIFEST = DEFAULT_ARCHIVE.parent / "manifest.json"
DEFAULT_EXPECTED_ARCHIVE_SHA = "b9f2ad678719907ce921b45496d90351df34eb44578bc56dd0e24e479f0d1033"
DEFAULT_OPENCLAW_BACKUP_DIR = Path("~/Yandex.Disk.localized/OpenClaw/prod_backups").expanduser()
OUTPUT_ALLOWED_CLIENT_UNSAFE_REASONS = ("sensitive_contract", "sensitive_money", "sensitive_tax")
OUTPUT_FORBIDDEN_CLIENT_UNSAFE_REASONS = (
    "has_manager_note",
    "manager_action_required",
    "sensitive_bank_requisites",
    "sensitive_credentials",
    "sensitive_document_data",
    "sensitive_medical",
    "sensitive_payment_details",
    "sensitive_personal_data",
)


@dataclass(frozen=True)
class M1MailMergeConfig:
    archive: Path
    external_manifest: Path
    timeline_db: Path
    prod_timeline_db: Path
    allowed_root: Path
    out_dir: Path
    tallanto_identity_db: Path | None = None
    tenant_id: str = "foton"
    apply: bool = False
    expected_archive_sha256: str = DEFAULT_EXPECTED_ARCHIVE_SHA

    def __post_init__(self) -> None:
        for field in ("archive", "external_manifest", "timeline_db", "prod_timeline_db", "allowed_root", "out_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)).expanduser())
        if self.tallanto_identity_db is not None:
            object.__setattr__(self, "tallanto_identity_db", Path(self.tallanto_identity_db).expanduser())


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = M1MailMergeConfig(
        archive=args.archive,
        external_manifest=args.external_manifest,
        timeline_db=args.timeline_db,
        prod_timeline_db=args.prod_timeline_db,
        allowed_root=args.allowed_root,
        out_dir=args.out_dir,
        tallanto_identity_db=args.tallanto_identity_db,
        tenant_id=args.tenant_id,
        apply=args.apply,
        expected_archive_sha256=args.expected_archive_sha256,
    )
    report = run_m1_mail_summary_merge(config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_m1_mail_summary_merge(config: M1MailMergeConfig) -> Mapping[str, Any]:
    started = time.monotonic()
    ensure_not_prod_apply_path(config.timeline_db, allowed_root=config.allowed_root)
    if "staging" not in config.timeline_db.resolve(strict=False).parts:
        raise ValueError("M1 mail summary merge must target .codex_local/staging")
    config.out_dir.mkdir(parents=True, exist_ok=True)

    external_manifest = _load_external_manifest(config.external_manifest)
    archive_report, cache_rows = _verify_archive_and_load_cache(
        config.archive,
        external_manifest=external_manifest,
        expected_sha256=config.expected_archive_sha256,
    )
    archive_reasoning = str(external_manifest.get("reasoning") or archive_report.get("reasoning") or "medium")
    archive_model = str(external_manifest.get("model") or "gpt-5.5")
    message_shas = {str(row["message_sha256"]).strip().lower() for row in cache_rows}

    db_before = _db_metrics(config.timeline_db)
    prod_before = _readonly_prod_fingerprint(config.prod_timeline_db)
    run_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "apply" if config.apply else "dry_run",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timeline_db": str(config.timeline_db),
        "prod_timeline_db": str(config.prod_timeline_db),
        "tallanto_identity_db": str(config.tallanto_identity_db) if config.tallanto_identity_db else "",
        "archive": archive_report,
        "external_manifest": {
            "path": str(config.external_manifest),
            "run_id": external_manifest.get("run_id"),
            "model": archive_model,
            "reasoning": archive_reasoning,
            "counters": external_manifest.get("counters"),
        },
        "db_before": db_before,
        "prod_before": prod_before,
        "safety": {
            "prod_write": False,
            "crm_write": False,
            "client_sends": False,
            "llm_calls_total": 0,
            "raw_pii_scope": ".codex_local/staging",
        },
    }

    input_jsonl = config.out_dir / "mail_summary_input.jsonl"
    a2_config = A2V3MailIngestConfig(
        input_jsonl=input_jsonl,
        prod_timeline_db=config.prod_timeline_db,
        timeline_db_path=config.timeline_db,
        allowed_root=config.allowed_root,
        out_dir=config.out_dir / "a2v3_apply",
        tallanto_identity_db=config.tallanto_identity_db,
        tenant_id=config.tenant_id,
        source_ref=f"m1_full_mail_summary_{external_manifest.get('run_id') or '20260708'}",
        enrich_existing=True,
        chunk_rich_text=True,
        refresh_purchases=False,
    )
    backup: Mapping[str, Any] | None = None
    backup_copy: Mapping[str, Any] | None = None
    if config.apply:
        backup = create_test_db_backup(a2_config, label="m1_mail_summary_merge_prewrite")
        backup_copy = _copy_backup_to_openclaw(backup, DEFAULT_OPENCLAW_BACKUP_DIR)

    if config.apply:
        con = sqlite3.connect(config.timeline_db)
    else:
        con = sqlite3.connect(f"{config.timeline_db.resolve(strict=False).as_uri()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        if not config.apply:
            con.execute("PRAGMA query_only=ON")
        if config.apply:
            _ensure_cache_table(con)
        cache_before = _cache_counts(con)
        if config.apply:
            cache_import = _import_cache_rows(con, cache_rows)
            cache_after = _cache_counts(con)
        else:
            cache_import = _cache_import_plan(con, cache_rows)
            cache_after = _predicted_cache_counts(
                cache_before,
                cache_rows,
                existing_shas=cache_import["existing_message_sha256"],
            )
        stage2_rows = _load_matching_stage2_rows(con, tenant_id=config.tenant_id, message_shas=message_shas)
        matched_shas = {str(row["message_sha256"]).lower() for row in stage2_rows}
        match_report = {
            "m1_cache_rows": len(cache_rows),
            "m1_unique_message_sha256": len(message_shas),
            "stage2_matched_rows": len(stage2_rows),
            "stage2_matched_unique_message_sha256": len(matched_shas),
            "m1_not_in_stage2": len(message_shas - matched_shas),
        }
        prepared_rows, cache_stats = _prepare_rows_with_m1_summaries(
            stage2_rows,
            payload_by_sha={
                str(row["message_sha256"]).strip().lower(): row["summary_payload_json"]
                for row in cache_rows
            },
        )
        _write_jsonl(input_jsonl, prepared_rows)
        if config.apply:
            con.commit()
    finally:
        con.close()

    validation = validate_a2v3_mail_ingest(a2_config)
    run_report.update(
        {
            "cache_import": cache_import,
            "cache_before": cache_before,
            "cache_after": cache_after,
            "match_report": match_report,
            "summary_cache": cache_stats,
            "input_jsonl": str(input_jsonl),
            "validation": _compact_validation(validation),
        }
    )

    if config.apply:
        apply_report = apply_a2v3_mail_ingest(a2_config, backup_manifest_path=Path(backup["manifest_path"]))
        stage4b = run_stage4b_bot_opening(
            Stage4BBotOpeningConfig(
                timeline_db_path=config.timeline_db,
                allowed_root=config.allowed_root,
                out_dir=config.out_dir / "stage4b_reopen",
                apply=True,
            )
        )
        run_report.update(
            {
                "backup": {
                    "manifest_path": backup["manifest_path"],
                    "backup_sha256": backup["backup_sha256"],
                    "created_before_cache_import": True,
                    "openclaw_copy": backup_copy,
                },
                "apply_report": _compact_apply(apply_report),
                "stage4b_reopen": {
                    "plan": stage4b.get("plan"),
                    "apply": stage4b.get("apply"),
                    "final_checks": stage4b.get("final_checks"),
                    "client_unsafe_mail_chunks_indexed": stage4b.get("client_unsafe_mail_chunks_indexed"),
                },
                "db_after": _db_metrics(config.timeline_db),
                "final_checks": _final_checks(config.timeline_db),
            }
        )

    prod_after = _readonly_prod_fingerprint(config.prod_timeline_db)
    run_report["prod_after"] = prod_after
    run_report["prod_sha256_unchanged"] = prod_before["sha256"] == prod_after["sha256"]
    run_report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report_path = config.out_dir / "m1_mail_summary_merge_report.json"
    report_path.write_text(json.dumps(run_report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    run_report["report_path"] = str(report_path)
    return run_report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge M1 full mail summary cache into Customer Timeline staging")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--external-manifest", type=Path, default=DEFAULT_EXTERNAL_MANIFEST)
    parser.add_argument("--expected-archive-sha256", default=DEFAULT_EXPECTED_ARCHIVE_SHA)
    parser.add_argument("--timeline-db", type=Path, required=True)
    parser.add_argument("--prod-timeline-db", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tallanto-identity-db", type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def _load_external_manifest(path: Path) -> Mapping[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"external manifest is not a JSON object: {path}")
    return data


def _verify_archive_and_load_cache(
    archive: Path,
    *,
    external_manifest: Mapping[str, Any],
    expected_sha256: str,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    archive_sha = _file_sha256(archive)
    manifest_archive = external_manifest.get("archive") if isinstance(external_manifest.get("archive"), Mapping) else {}
    expected = str(expected_sha256 or manifest_archive.get("sha256") or "")
    if expected and archive_sha != expected:
        raise ValueError(f"archive sha256 mismatch: {archive_sha} != {expected}")
    expected_files = {
        str(item.get("path")): item
        for item in external_manifest.get("files_inside_archive", [])
        if isinstance(item, Mapping) and item.get("path")
    }
    cache_rows: list[dict[str, Any]] = []
    actual_files: list[Mapping[str, Any]] = []
    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.isfile()]
        for member in members:
            extracted = tar.extractfile(member)
            if extracted is None:
                raise ValueError(f"failed to read archive member: {member.name}")
            data = extracted.read()
            actual_sha = hashlib.sha256(data).hexdigest()
            actual_files.append({"path": member.name, "sha256": actual_sha, "bytes": len(data)})
            expected_item = expected_files.get(member.name)
            if expected_item is not None and actual_sha != str(expected_item.get("sha256")):
                raise ValueError(f"archive member sha mismatch for {member.name}")
            if member.name == "data/email_summary_cache.jsonl":
                cache_rows = _parse_cache_jsonl(data)
    if expected_files and {item["path"] for item in actual_files} != set(expected_files):
        missing = sorted(set(expected_files) - {item["path"] for item in actual_files})
        extra = sorted({item["path"] for item in actual_files} - set(expected_files))
        raise ValueError(f"archive member set mismatch: missing={missing}, extra={extra}")
    if not cache_rows:
        raise ValueError("data/email_summary_cache.jsonl was not found or empty")
    return (
        {
            "path": str(archive),
            "sha256": archive_sha,
            "files": len(actual_files),
            "cache_rows": len(cache_rows),
            "reasoning": external_manifest.get("reasoning"),
            "files_verified_against_external_manifest": bool(expected_files),
        },
        cache_rows,
    )


def _parse_cache_jsonl(data: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        if not isinstance(row, dict):
            raise ValueError(f"cache row {line_number} is not an object")
        payload = row.get("summary_payload_json")
        if isinstance(payload, str):
            parsed_payload = json.loads(payload)
        else:
            parsed_payload = payload
        if not isinstance(parsed_payload, Mapping):
            raise ValueError(f"cache row {line_number} has invalid summary_payload_json")
        row = dict(row)
        row["summary_payload_json"] = dict(parsed_payload)
        rows.append(row)
    return rows


def _import_cache_rows(con: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    source_counts: dict[str, int] = {}
    quality_flag_counts: dict[str, int] = {}
    values = []
    for row in rows:
        message_sha = str(row.get("message_sha256") or "").strip().lower()
        if not message_sha:
            raise ValueError("M1 cache row missing message_sha256")
        payload = row.get("summary_payload_json")
        if not isinstance(payload, Mapping):
            raise ValueError(f"M1 cache row {message_sha[:16]} has invalid payload")
        source_kind = str(row.get("source_kind") or "m1")
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
        for flag in _quality_flags(payload.get("quality_flags")):
            quality_flag_counts[flag] = quality_flag_counts.get(flag, 0) + 1
        values.append(
            (
                message_sha,
                str(row.get("text_sha256") or ""),
                str(row.get("prompt_version") or PROMPT_VERSION),
                str(row.get("provider") or "codex_cli"),
                str(row.get("model") or "gpt-5.5"),
                str(row.get("reasoning") or "medium"),
                source_kind,
                str(row.get("summary_text") or payload.get("summary") or ""),
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                str(row.get("created_at") or datetime.now(timezone.utc).isoformat()),
            )
        )
    con.executemany(
        """
        INSERT INTO email_summary_cache_v1 (
          message_sha256, text_sha256, prompt_version, provider, model, reasoning,
          source_kind, summary_text, summary_payload_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_sha256) DO UPDATE SET
          text_sha256 = excluded.text_sha256,
          prompt_version = excluded.prompt_version,
          provider = excluded.provider,
          model = excluded.model,
          reasoning = excluded.reasoning,
          source_kind = excluded.source_kind,
          summary_text = excluded.summary_text,
          summary_payload_json = excluded.summary_payload_json,
          created_at = excluded.created_at
        """,
        values,
    )
    return {"rows_upserted": len(values), "source_kind_counts": source_counts, "quality_flag_counts": quality_flag_counts}


def _cache_import_plan(con: sqlite3.Connection, rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    existing: set[str] = set()
    if _table_exists(con, "email_summary_cache_v1"):
        existing = {
            str(row["message_sha256"]).lower()
            for row in con.execute("SELECT message_sha256 FROM email_summary_cache_v1").fetchall()
        }
    package_shas = {str(row.get("message_sha256") or "").strip().lower() for row in rows if row.get("message_sha256")}
    source_counts: dict[str, int] = {}
    quality_flag_counts: dict[str, int] = {}
    for row in rows:
        source_kind = str(row.get("source_kind") or "m1")
        source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
        payload = row.get("summary_payload_json")
        if isinstance(payload, Mapping):
            for flag in _quality_flags(payload.get("quality_flags")):
                quality_flag_counts[flag] = quality_flag_counts.get(flag, 0) + 1
    return {
        "dry_run": True,
        "rows_would_upsert": len(rows),
        "existing_message_sha256": sorted(existing & package_shas),
        "new_message_sha256": len(package_shas - existing),
        "source_kind_counts": source_counts,
        "quality_flag_counts": quality_flag_counts,
    }


def _predicted_cache_counts(
    cache_before: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    existing_shas: Sequence[str],
) -> Mapping[str, Any]:
    source_counts = dict(cache_before.get("source_kind_counts") or {})
    existing = set(existing_shas)
    package_shas = {str(row.get("message_sha256") or "").strip().lower() for row in rows if row.get("message_sha256")}
    for row in rows:
        sha = str(row.get("message_sha256") or "").strip().lower()
        if not sha or sha in existing:
            continue
        source_kind = str(row.get("source_kind") or "m1")
        source_counts[source_kind] = int(source_counts.get(source_kind, 0)) + 1
    return {
        "rows": int(cache_before.get("rows") or 0) + len(package_shas - existing),
        "source_kind_counts": source_counts,
        "dry_run_prediction": True,
    }


def _quality_flags(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(str(key) for key, enabled in value.items() if enabled)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _load_matching_stage2_rows(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    message_shas: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[sqlite3.Row] = []
    sorted_shas = sorted({sha for sha in message_shas if sha})
    for offset in range(0, len(sorted_shas), 900):
        chunk = sorted_shas[offset : offset + 900]
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            con.execute(
                f"""
                WITH phones AS (
                  SELECT customer_id, MIN(link_value) AS contact_phone
                  FROM identity_links
                  WHERE tenant_id = ?
                    AND link_type = 'phone'
                    AND match_class = 'strong_unique'
                  GROUP BY customer_id
                )
                SELECT
                  e.event_id, e.tenant_id, e.customer_id, e.event_at, e.source_id, e.source_ref,
                  e.direction, e.subject, e.summary, e.text_preview, e.record_json, phones.contact_phone
                FROM timeline_events e
                LEFT JOIN phones ON phones.customer_id = e.customer_id
                WHERE e.tenant_id = ?
                  AND e.source_system = ?
                  AND COALESCE(e.superseded_by, '') = ''
                  AND e.source_id IN ({placeholders})
                ORDER BY e.customer_id, e.event_at, e.source_id
                """,
                (tenant_id, tenant_id, SOURCE_SYSTEM, *chunk),
            ).fetchall()
        )
    rows.sort(key=lambda row: (str(row["customer_id"] or ""), str(row["event_at"] or ""), str(row["source_id"] or "")))
    result: list[dict[str, Any]] = []
    archive_cache: dict[Path, sqlite3.Connection] = {}
    try:
        for line_number, row in enumerate(rows, start=1):
            payload = json_loads(str(row["record_json"] or "{}"))
            record = payload.get("record") if isinstance(payload.get("record"), Mapping) else {}
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
            full_text = str(record.get("full_clean_text") or "")
            clean_text, thread_context = split_thread_context(full_text)
            body_missing = not bool(clean_text.strip())
            subject = str(row["subject"] or record.get("subject") or "Email message")
            brand = _brand_for_row(row=row, record=record, metadata=metadata, text=clean_text)
            envelope = _mail_archive_envelope_for_row(
                payload,
                message_sha=str(row["source_id"]).lower(),
                direction=str(row["direction"] or "unknown"),
                archive_cache=archive_cache,
            )
            contact_phone = row["contact_phone"]
            contact_email = envelope.get("contact_email")
            contact_source = (
                envelope.get("contact_source")
                or ("staging_identity_link_phone" if contact_phone else "missing")
            )
            contact_reason = (
                envelope.get("contact_reason")
                or ("strong_phone_identity_link" if contact_phone else "no_strong_phone_or_email_identity_link")
            )
            result.append(
                {
                    "_line_number": line_number,
                    "event_id": str(row["event_id"]),
                    "customer_id": str(row["customer_id"] or ""),
                    "message_sha256": str(row["source_id"]).lower(),
                    "date_iso": str(row["event_at"] or ""),
                    "direction": str(row["direction"] or "unknown"),
                    "brand": brand,
                    "brand_source": str(
                        record.get("brand_source") or record.get("brand_signal") or metadata.get("brand_source") or "staging_event"
                    ),
                    "raw_infer_offline_brand": brand,
                    "classification_reason": "m1_full_mail_summary_existing_stage2_event",
                    "subject_full": subject,
                    "subject": subject,
                    "full_clean_text": clean_text,
                    "body_missing": body_missing,
                    "thread_context": thread_context,
                    "thread_context_source": "raw_body_split_thread_context" if thread_context else "none",
                    "full_clean_text_chars": len(clean_text),
                    "body_chars": len(full_text),
                    "body": clean_text,
                    "contact_phone": contact_phone,
                    "contact_email": contact_email,
                    "contact_name": envelope.get("contact_name"),
                    "contact_source": contact_source,
                    "contact_missing": not bool(contact_phone or contact_email),
                    "contact_ambiguous": bool(envelope.get("contact_ambiguous")),
                    "contact_reason": contact_reason,
                    "from_email": envelope.get("from_email"),
                    "from_domain": envelope.get("from_domain"),
                    "to_domains": envelope.get("to_domains", []),
                    "to_emails": envelope.get("to_emails", []),
                    "cc_emails": envelope.get("cc_emails", []),
                    "external_recipient_count": int(envelope.get("external_recipient_count") or 0),
                    "outbound_template_freq": 0,
                    "is_outbound_template": False,
                    "is_mass_recipient": False,
                    "has_attachment": False,
                }
            )
    finally:
        for archive_con in archive_cache.values():
            archive_con.close()
    return result


def _mail_archive_envelope_for_row(
    event_payload: Mapping[str, Any],
    *,
    message_sha: str,
    direction: str,
    archive_cache: dict[Path, sqlite3.Connection],
) -> Mapping[str, Any]:
    source_payload = _source_payload(event_payload)
    archive_db = _archive_db_for_source_payload(source_payload)
    if archive_db is None:
        return {}
    db = archive_db.resolve(strict=False)
    archive_con = archive_cache.get(db)
    if archive_con is None:
        archive_con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        archive_con.row_factory = sqlite3.Row
        archive_con.execute("PRAGMA query_only=ON")
        archive_cache[db] = archive_con
    participants = [
        dict(row)
        for row in archive_con.execute(
            """
            SELECT header_name, display_name, email_normalized, domain
            FROM message_participants
            WHERE message_sha256 = ?
            """,
            (message_sha,),
        )
    ]
    if not participants:
        return {}
    contact = _contact_from_archive_row(direction, {"participants": participants, "text": ""})
    from_email = _first_participant_email(participants, "from")
    to_emails = _participant_emails(participants, "to")
    cc_emails = _participant_emails(participants, "cc")
    return {
        "contact_email": contact.contact_email,
        "contact_name": contact.contact_name,
        "contact_source": contact.contact_source,
        "contact_reason": contact.contact_reason,
        "contact_ambiguous": contact.contact_ambiguous,
        "external_recipient_count": contact.external_recipient_count,
        "from_email": from_email,
        "from_domain": _email_domain(from_email),
        "to_emails": to_emails,
        "to_domains": [_email_domain(email) for email in to_emails],
        "cc_emails": cc_emails,
    }


def _participant_emails(participants: Sequence[Mapping[str, Any]], header_name: str) -> list[str]:
    result: list[str] = []
    for item in participants:
        if str(item.get("header_name") or "").strip().lower() != header_name:
            continue
        email = str(item.get("email_normalized") or "").strip().lower()
        if email and email not in result:
            result.append(email)
    return result


def _first_participant_email(participants: Sequence[Mapping[str, Any]], header_name: str) -> str | None:
    emails = _participant_emails(participants, header_name)
    return emails[0] if emails else None


def _email_domain(email: str | None) -> str:
    if not email or "@" not in email:
        return ""
    return email.rsplit("@", 1)[-1].strip().lower()


def _prepare_rows_with_m1_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    payload_by_sha: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    prepared: list[dict[str, Any]] = []
    stats = {
        "rows": len(rows),
        "m1_payload_hits": 0,
        "m1_payload_missing": 0,
        "summary_review_needed": 0,
        "sanitized_payloads": 0,
        "quality_flag_counts": {},
    }
    for source_row in rows:
        row = dict(source_row)
        message_sha = str(row.get("message_sha256") or "").strip().lower()
        payload = payload_by_sha.get(message_sha)
        if payload is None:
            stats["m1_payload_missing"] += 1
            continue
        sanitized_payload = _sanitize_summary_payload_for_stage2(payload)
        row["summary_payload"] = sanitized_payload
        quality = evaluate_quality(row)
        quality_dict = quality_to_dict(quality)
        payload_flags = list(_quality_flags(sanitized_payload.get("quality_flags")))
        if payload_flags:
            merged_flags = list(dict.fromkeys([*quality_dict.get("quality_flags", []), *payload_flags]))
            quality_dict["quality_flags"] = merged_flags
            for flag in payload_flags:
                stats["quality_flag_counts"][flag] = stats["quality_flag_counts"].get(flag, 0) + 1
        sanitized_for_quality = sanitize_summary_payload_for_quality(dict(row.get("summary_payload") or {}), quality)
        if sanitized_for_quality != row.get("summary_payload"):
            row["summary_payload"] = sanitized_for_quality
            stats["sanitized_payloads"] += 1
        if _is_summary_review_needed_payload(row.get("summary_payload")):
            flags = list(quality_dict.get("quality_flags") or [])
            flags.append("summary_review_needed")
            quality_dict["quality_flags"] = list(dict.fromkeys(flags))
            quality_dict["memory_status"] = "summary_review_needed"
            stats["summary_review_needed"] += 1
        row["quality"] = quality_dict
        stats["m1_payload_hits"] += 1
        prepared.append(row)
    if stats["m1_payload_missing"]:
        raise RuntimeError(f"missing M1 payloads for {stats['m1_payload_missing']} matched stage2 rows")
    return prepared, stats


def _db_metrics(db_path: Path) -> Mapping[str, Any]:
    uri = f"{db_path.resolve(strict=False).as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        result = {
            "sha256": _file_sha256(db_path),
            "quick_check": con.execute("PRAGMA quick_check").fetchone()[0],
            "timeline_events": _count_table(con, "timeline_events"),
            "bot_context_chunks": _count_table(con, "bot_context_chunks"),
            "a2v3_mail_event_facts": _count_table(con, "a2v3_mail_event_facts"),
            "email_summary_cache_v1": _count_table(con, "email_summary_cache_v1"),
            "mail_stage2_events": int(
                con.execute("SELECT count(*) FROM timeline_events WHERE source_system = ?", (SOURCE_SYSTEM,)).fetchone()[0]
            ),
            "mail_stage2_open_chunks": int(
                con.execute(
                    """
                    SELECT count(*)
                    FROM bot_context_chunks
                    WHERE source_system = ?
                      AND COALESCE(superseded_by, '') = ''
                      AND allowed_for_bot = 1
                      AND requires_manager_review = 0
                    """,
                    (SOURCE_SYSTEM,),
                ).fetchone()[0]
            ),
            "mail_stage2_client_unsafe_open_chunks": _unsafe_open_chunks(con),
            "mail_stage2_output_forbidden_open_chunks": _forbidden_open_chunks(con),
            "mail_stage2_money_draft_open_chunks": _allowed_money_draft_open_chunks(con),
        }
    return result


def _final_checks(db_path: Path) -> Mapping[str, Any]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only=ON")
        return {
            "quick_check": con.execute("PRAGMA quick_check").fetchone()[0],
            "foreign_key_check_rows": len(con.execute("PRAGMA foreign_key_check").fetchall()),
            "mail_stage2_invalid_flag_pairs": int(
                con.execute(
                    """
                    SELECT count(*)
                    FROM bot_context_chunks
                    WHERE source_system = ?
                      AND COALESCE(superseded_by, '') = ''
                      AND (
                        (allowed_for_bot = 1 AND requires_manager_review = 1)
                        OR (allowed_for_bot = 0 AND requires_manager_review = 0)
                      )
                    """,
                    (SOURCE_SYSTEM,),
                ).fetchone()[0]
            ),
            "mail_stage2_client_unsafe_open_chunks": _unsafe_open_chunks(con),
            "mail_stage2_output_forbidden_open_chunks": _forbidden_open_chunks(con),
            "mail_stage2_open_not_allowed_by_output_gate": _open_not_allowed_by_output_gate(con),
            "mail_stage2_money_draft_open_chunks": _allowed_money_draft_open_chunks(con),
        }


def _unsafe_open_chunks(con: sqlite3.Connection) -> int:
    if not _table_exists(con, "a2v3_mail_event_facts"):
        return 0
    return int(
        con.execute(
            """
            SELECT count(*)
            FROM bot_context_chunks c
            JOIN a2v3_mail_event_facts f ON f.event_id = c.event_id
            WHERE c.source_system = ?
              AND COALESCE(c.superseded_by, '') = ''
              AND c.allowed_for_bot = 1
              AND c.requires_manager_review = 0
              AND f.client_safe = 0
            """,
            (SOURCE_SYSTEM,),
        ).fetchone()[0]
    )


def _forbidden_open_chunks(con: sqlite3.Connection) -> int:
    if not _table_exists(con, "a2v3_mail_event_facts"):
        return 0
    placeholders = ",".join("?" for _ in OUTPUT_FORBIDDEN_CLIENT_UNSAFE_REASONS)
    return int(
        con.execute(
            f"""
            SELECT count(*)
            FROM bot_context_chunks c
            JOIN a2v3_mail_event_facts f ON f.event_id = c.event_id
            WHERE c.source_system = ?
              AND COALESCE(c.superseded_by, '') = ''
              AND c.allowed_for_bot = 1
              AND c.requires_manager_review = 0
              AND (
                f.client_safe_reason IN ({placeholders})
                OR f.sensitivity_tags_json LIKE '%sensitive_credentials%'
                OR f.sensitivity_tags_json LIKE '%sensitive_bank_requisites%'
                OR f.sensitivity_tags_json LIKE '%sensitive_payment_details%'
                OR f.sensitivity_tags_json LIKE '%sensitive_personal_data%'
                OR f.sensitivity_tags_json LIKE '%sensitive_document_data%'
              )
            """,
            (SOURCE_SYSTEM, *OUTPUT_FORBIDDEN_CLIENT_UNSAFE_REASONS),
        ).fetchone()[0]
    )


def _open_not_allowed_by_output_gate(con: sqlite3.Connection) -> int:
    if not _table_exists(con, "a2v3_mail_event_facts"):
        return 0
    allowed_placeholders = ",".join("?" for _ in OUTPUT_ALLOWED_CLIENT_UNSAFE_REASONS)
    return int(
        con.execute(
            f"""
            SELECT count(*)
            FROM bot_context_chunks c
            LEFT JOIN a2v3_mail_event_facts f ON f.event_id = c.event_id
            WHERE c.source_system = ?
              AND COALESCE(c.superseded_by, '') = ''
              AND c.allowed_for_bot = 1
              AND c.requires_manager_review = 0
              AND NOT (
                COALESCE(f.client_safe, 0) = 1
                OR f.client_safe_reason IN ({allowed_placeholders})
              )
            """,
            (SOURCE_SYSTEM, *OUTPUT_ALLOWED_CLIENT_UNSAFE_REASONS),
        ).fetchone()[0]
    )


def _allowed_money_draft_open_chunks(con: sqlite3.Connection) -> int:
    if not _table_exists(con, "a2v3_mail_event_facts"):
        return 0
    placeholders = ",".join("?" for _ in OUTPUT_ALLOWED_CLIENT_UNSAFE_REASONS)
    return int(
        con.execute(
            f"""
            SELECT count(*)
            FROM bot_context_chunks c
            JOIN a2v3_mail_event_facts f ON f.event_id = c.event_id
            WHERE c.source_system = ?
              AND COALESCE(c.superseded_by, '') = ''
              AND c.allowed_for_bot = 1
              AND c.requires_manager_review = 0
              AND f.client_safe_reason IN ({placeholders})
            """,
            (SOURCE_SYSTEM, *OUTPUT_ALLOWED_CLIENT_UNSAFE_REASONS),
        ).fetchone()[0]
    )


def _readonly_prod_fingerprint(path: Path) -> Mapping[str, Any]:
    uri = f"{path.resolve(strict=False).as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as con:
        con.execute("PRAGMA query_only=ON")
        return {
            "path": str(path),
            "sha256": _file_sha256(path),
            "quick_check": con.execute("PRAGMA quick_check").fetchone()[0],
            "timeline_events": _count_table(con, "timeline_events"),
            "bot_context_chunks": _count_table(con, "bot_context_chunks"),
        }


def _count_table(con: sqlite3.Connection, table: str) -> int:
    if not _table_exists(con, table):
        return 0
    return int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _copy_backup_to_openclaw(backup: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    source = Path(str(backup["backup_db_path"]))
    target_dir = root / f"m1_mail_summary_merge_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    target_dir.mkdir(parents=True, exist_ok=False)
    target = target_dir / source.name
    shutil.copy2(source, target)
    target_sha = _file_sha256(target)
    expected_sha = str(backup["backup_sha256"])
    if target_sha != expected_sha:
        raise ValueError(f"OpenClaw backup copy sha mismatch: {target_sha} != {expected_sha}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "openclaw_async_backup_copy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_backup_db_path": str(source),
        "target_backup_db_path": str(target),
        "sha256": target_sha,
        "cloud_sync": "yandex_disk_async",
    }
    (target_dir / "backup_copy_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


if __name__ == "__main__":
    sys.exit(main())
