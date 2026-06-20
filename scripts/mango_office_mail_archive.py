#!/usr/bin/env python3
"""Build read-only mail archive and Tallanto identity matching artifacts.

The IMAP ingest command downloads messages with BODY.PEEK[] into a local raw
archive. It never sends, deletes, moves, marks as read, or writes CRM/Tallanto.
Passwords are read only from an environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mango_mvp.productization.mail_archive import (  # noqa: E402
    DEFAULT_ATTACHMENT_PARSE_ALLOW_EXTENSIONS,
    DEFAULT_ATTACHMENT_IMAGE_OCR_EXTENSIONS,
    DEFAULT_ATTACHMENT_PARSE_REVIEW_EXTENSIONS,
    DEFAULT_ATTACHMENT_PDF_EXTRACT_EXTENSIONS,
    DEFAULT_ATTACHMENT_TEXT_EXTRACT_EXTENSIONS,
    MailArchiveIngestConfig,
    MailArchivePreflightConfig,
    MailArchiveVerificationConfig,
    MailAttachmentImageOcrPlanConfig,
    MailAttachmentOcrPreflightConfig,
    MailAttachmentOcrPilotConfig,
    MailAttachmentParsePlanConfig,
    MailAttachmentPdfExtractConfig,
    MailAttachmentStage6PlanConfig,
    MailAttachmentTextExtractConfig,
    MailAttachmentTextIndexConfig,
    MailCustomerHistoryHandoffConfig,
    MailCustomerRelinkPreviewConfig,
    MailStage2CustomerRelinkPreviewConfig,
    MailMangoBridgePreviewConfig,
    MailPhoneLiftPreviewConfig,
    MangoPhoneIndexPreviewConfig,
    MailMatchingReportConfig,
    TallantoIdentityMapConfig,
    TallantoIdentityMapUnionConfig,
    TallantoIdentityMapUnionSourceConfig,
    build_mail_archive_ingest,
    build_mail_archive_preflight,
    build_mail_attachment_image_ocr_plan,
    build_mail_attachment_ocr_preflight,
    build_mail_attachment_ocr_pilot,
    build_mail_attachment_parse_plan,
    build_mail_attachment_pdf_extract,
    build_mail_attachment_stage6_plan,
    build_mail_attachment_text_extract,
    build_mail_attachment_text_index,
    build_mail_customer_history_handoff,
    build_mail_customer_relink_preview,
    build_mail_stage2_customer_relink_preview,
    build_mail_mango_bridge_preview,
    build_mail_phone_lift_preview,
    build_mango_phone_index_preview,
    build_mail_matching_report,
    build_tallanto_identity_map,
    build_tallanto_identity_map_union,
    valid_env_var_name,
    verify_mail_archive_pilot,
)
from mango_mvp.productization.mail_imap_snapshot import MailImapCredentials  # noqa: E402


DEFAULT_HOST = "mail.hosting.reg.ru"
DEFAULT_PORT = 993
DEFAULT_PASSWORD_ENV = "MAIL_IMAP_PASSWORD"
DEFAULT_ACCOUNT_LABEL = "regru_edu"
DEFAULT_TALLANTO_CSV = (
    "_external_handoffs/tallanto_students_export_2026-05-12/Ученики.csv"
)
DEFAULT_MAIL_ARCHIVE_ROOT = "_external_handoffs/mail_archive_2026-05-12"
DEFAULT_MANGO_PRODUCT_DB = (
    "_local_archive_mango_api_downloads_20260507/product_appliance/mango_product_appliance.sqlite"
)
DEFAULT_MANGO_RECORDING_ROOT = "_local_archive_mango_api_downloads_20260507"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "identity-map":
        return run_identity_map(args)
    if args.command == "identity-map-union":
        return run_identity_map_union(args)
    if args.command == "preflight":
        return run_preflight(args)
    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "verify-pilot":
        return run_verify_pilot(args)
    if args.command == "match-report":
        return run_match_report(args)
    if args.command == "history-handoff":
        return run_history_handoff(args)
    if args.command == "mango-bridge-preview":
        return run_mango_bridge_preview(args)
    if args.command == "mango-phone-index-preview":
        return run_mango_phone_index_preview(args)
    if args.command == "phone-lift-preview":
        return run_phone_lift_preview(args)
    if args.command == "customer-relink-preview":
        return run_customer_relink_preview(args)
    if args.command == "stage2-customer-relink-preview":
        return run_stage2_customer_relink_preview(args)
    if args.command == "attachment-parse-plan":
        return run_attachment_parse_plan(args)
    if args.command == "attachment-text-extract":
        return run_attachment_text_extract(args)
    if args.command == "attachment-pdf-extract":
        return run_attachment_pdf_extract(args)
    if args.command == "attachment-image-ocr-plan":
        return run_attachment_image_ocr_plan(args)
    if args.command == "attachment-text-index":
        return run_attachment_text_index(args)
    if args.command == "attachment-stage6-plan":
        return run_attachment_stage6_plan(args)
    if args.command == "attachment-ocr-preflight":
        return run_attachment_ocr_preflight(args)
    if args.command == "attachment-ocr-pilot":
        return run_attachment_ocr_pilot(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def run_identity_map(args: argparse.Namespace) -> int:
    try:
        report = build_tallanto_identity_map(
            TallantoIdentityMapConfig(
                tallanto_csv_path=Path(args.tallanto_csv),
                out_dir=Path(args.out_dir or default_identity_out_dir(args.account_label)),
                encoding=args.encoding,
                delimiter=args.delimiter,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL identity map failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_identity_map_union(args: argparse.Namespace) -> int:
    try:
        report = build_tallanto_identity_map_union(
            TallantoIdentityMapUnionConfig(
                sources=[
                    TallantoIdentityMapUnionSourceConfig(
                        label="old_students_20260512",
                        tallanto_csv_path=Path(args.old_tallanto_csv),
                        encoding=args.old_encoding,
                        delimiter=args.old_delimiter,
                    ),
                    TallantoIdentityMapUnionSourceConfig(
                        label="contacts_20260620",
                        tallanto_csv_path=Path(args.fresh_tallanto_csv),
                        encoding=args.fresh_encoding,
                        delimiter=args.fresh_delimiter,
                    ),
                ],
                out_dir=Path(args.out_dir),
                db_filename=args.db_filename,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL identity map union failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_preflight(args: argparse.Namespace) -> int:
    load_dotenv_file(Path(args.dotenv)) if args.dotenv else None
    email_address = args.email or os.environ.get("MAIL_IMAP_EMAIL", "")
    host = args.host or os.environ.get("MAIL_IMAP_HOST", DEFAULT_HOST)
    port = resolve_port(args.port)
    password_env_valid = valid_env_var_name(args.password_env)
    report = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=Path(args.out_dir or default_ingest_out_dir(args.account_label)),
            mailbox=args.mailbox,
            since_days=args.since_days,
            before_days=args.before_days,
            max_messages=args.max_messages,
            account_label=args.account_label,
            host=host,
            port=port,
            email_address=email_address,
            password_env_name=args.password_env,
            password_env_present=password_env_valid and bool(os.environ.get(args.password_env)),
            identity_db_path=Path(args.identity_db) if args.identity_db else None,
            allow_large_batch=args.allow_large_batch,
        )
    )
    print_summary(report)
    return 0 if report.get("preflight_pass") else 2


def run_ingest(args: argparse.Namespace) -> int:
    load_dotenv_file(Path(args.dotenv)) if args.dotenv else None
    if not valid_env_var_name(args.password_env):
        print("MAIL archive ingest failed: invalid password env variable name.", file=sys.stderr)
        return 2
    email_address = args.email or os.environ.get("MAIL_IMAP_EMAIL", "")
    host = args.host or os.environ.get("MAIL_IMAP_HOST", DEFAULT_HOST)
    port = resolve_port(args.port)
    out_dir = Path(args.out_dir or default_ingest_out_dir(args.account_label))
    excluded_sha256s = load_excluded_message_sha256s(args.exclude_archive_db or [])
    preflight = build_mail_archive_preflight(
        MailArchivePreflightConfig(
            out_dir=out_dir,
            mailbox=args.mailbox,
            since_days=args.since_days,
            before_days=args.before_days,
            max_messages=args.max_messages,
            account_label=args.account_label,
            host=host,
            port=port,
            email_address=email_address,
            password_env_name=args.password_env,
            password_env_present=bool(os.environ.get(args.password_env)),
            allow_large_batch=args.allow_large_batch,
        )
    )
    if not preflight.get("preflight_pass"):
        print(
            "MAIL archive ingest failed: preflight checks did not pass.",
            file=sys.stderr,
        )
        print_summary(preflight)
        return 2
    password = os.environ.get(args.password_env, "")

    try:
        report = build_mail_archive_ingest(
            credentials=MailImapCredentials(
                host=host,
                port=port,
                email_address=email_address,
                password=password,
            ),
            config=MailArchiveIngestConfig(
                out_dir=out_dir,
                mailbox=args.mailbox,
                mailbox_label=args.mailbox_label or args.mailbox.strip('"'),
                since_days=args.since_days,
                before_days=args.before_days,
                max_messages=args.max_messages,
                account_label=args.account_label,
                internal_domains=tuple(args.internal_domain or []),
                extracted_text_max_chars=args.extracted_text_max_chars,
                exclude_message_sha256s=tuple(excluded_sha256s),
            ),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL archive ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0 if not report.get("errors") else 1


def run_verify_pilot(args: argparse.Namespace) -> int:
    try:
        report = verify_mail_archive_pilot(
            MailArchiveVerificationConfig(
                archive_dir=Path(args.archive_dir),
                expected_max_messages=args.expected_max_messages,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL archive verify failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0 if report.get("verification_pass") else 1


def run_match_report(args: argparse.Namespace) -> int:
    try:
        report = build_mail_matching_report(
            MailMatchingReportConfig(
                archive_db_path=Path(args.archive_db),
                identity_db_path=Path(args.identity_db),
                out_dir=Path(args.out_dir or default_match_out_dir(args.account_label)),
                mailbox_email=args.email or os.environ.get("MAIL_IMAP_EMAIL", ""),
                internal_domains=tuple(args.internal_domain or []),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL matching report failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_history_handoff(args: argparse.Namespace) -> int:
    try:
        report = build_mail_customer_history_handoff(
            MailCustomerHistoryHandoffConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                identity_db_path=Path(args.identity_db),
                out_dir=Path(args.out_dir),
                mailbox_email=args.email or os.environ.get("MAIL_IMAP_EMAIL", ""),
                internal_domains=tuple(args.internal_domain or []),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL customer history handoff failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_mango_bridge_preview(args: argparse.Namespace) -> int:
    try:
        report = build_mail_mango_bridge_preview(
            MailMangoBridgePreviewConfig(
                mail_handoff_db_path=Path(args.mail_handoff_db),
                identity_db_path=Path(args.identity_db),
                product_db_path=Path(args.product_db),
                out_dir=Path(args.out_dir),
                max_call_refs_per_candidate=args.max_call_refs_per_candidate,
                mango_phone_index_db_path=(
                    Path(args.mango_phone_index_db) if args.mango_phone_index_db else None
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL Mango bridge preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_mango_phone_index_preview(args: argparse.Namespace) -> int:
    try:
        report = build_mango_phone_index_preview(
            MangoPhoneIndexPreviewConfig(
                product_db_path=Path(args.product_db),
                recording_roots=[
                    Path(path)
                    for path in (args.recording_root or [DEFAULT_MANGO_RECORDING_ROOT])
                ],
                out_dir=Path(args.out_dir),
                include_product_db=not args.skip_product_db,
                include_recording_filenames=not args.skip_recording_filenames,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Mango phone index preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_phone_lift_preview(args: argparse.Namespace) -> int:
    try:
        report = build_mail_phone_lift_preview(
            MailPhoneLiftPreviewConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                identity_db_path=Path(args.identity_db),
                out_dir=Path(args.out_dir),
                max_text_chars_per_message=args.max_text_chars_per_message,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL phone lift preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_customer_relink_preview(args: argparse.Namespace) -> int:
    live_lookup = None
    if args.live_tallanto_lookup:
        load_dotenv_file(Path(args.dotenv)) if args.dotenv else None
        from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, build_tallanto_api_config

        client = TallantoApiClient(build_tallanto_api_config())
        live_lookup = lambda phone: client.search_contacts_by_phone(  # noqa: E731
            phone,
            max_records=args.live_max_records,
        )
    try:
        report = build_mail_customer_relink_preview(
            MailCustomerRelinkPreviewConfig(
                mail_handoff_db_path=Path(args.mail_handoff_db),
                identity_db_path=Path(args.identity_db),
                out_dir=Path(args.out_dir),
                classification_index_path=(
                    Path(args.classification_index) if args.classification_index else None
                ),
                live_phone_lookup=live_lookup,
                require_real_correspondence=not args.allow_non_real_correspondence,
                max_text_chars_per_message=args.max_text_chars_per_message,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL customer relink preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_stage2_customer_relink_preview(args: argparse.Namespace) -> int:
    try:
        report = build_mail_stage2_customer_relink_preview(
            MailStage2CustomerRelinkPreviewConfig(
                event_jsonl_paths=[Path(path) for path in args.event_jsonl],
                identity_db_path=Path(args.identity_db),
                out_dir=Path(args.out_dir),
                max_text_chars_per_message=args.max_text_chars_per_message,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL stage2 customer relink preview failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_attachment_parse_plan(args: argparse.Namespace) -> int:
    try:
        report = build_mail_attachment_parse_plan(
            MailAttachmentParsePlanConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                out_dir=Path(args.out_dir),
                max_size_bytes=args.max_size_bytes,
                allow_extensions=tuple(args.allow_extension or []),
                review_extensions=tuple(args.review_extension or []),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL attachment parse plan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_attachment_text_extract(args: argparse.Namespace) -> int:
    try:
        report = build_mail_attachment_text_extract(
            MailAttachmentTextExtractConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                parse_plan_db_path=Path(args.parse_plan_db),
                out_dir=Path(args.out_dir),
                stage_extensions=tuple(args.stage_extension or DEFAULT_ATTACHMENT_TEXT_EXTRACT_EXTENSIONS),
                max_attachment_bytes=args.max_attachment_bytes,
                max_text_chars_per_attachment=args.max_text_chars_per_attachment,
                max_csv_rows=args.max_csv_rows,
                max_csv_columns=args.max_csv_columns,
                max_xlsx_sheets=args.max_xlsx_sheets,
                max_xlsx_rows_per_sheet=args.max_xlsx_rows_per_sheet,
                max_xlsx_columns_per_sheet=args.max_xlsx_columns_per_sheet,
                max_zip_members=args.max_zip_members,
                max_zip_uncompressed_bytes=args.max_zip_uncompressed_bytes,
                max_attachments=args.max_attachments,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL attachment text extract failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_attachment_pdf_extract(args: argparse.Namespace) -> int:
    try:
        report = build_mail_attachment_pdf_extract(
            MailAttachmentPdfExtractConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                parse_plan_db_path=Path(args.parse_plan_db),
                out_dir=Path(args.out_dir),
                stage_extensions=tuple(args.stage_extension or DEFAULT_ATTACHMENT_PDF_EXTRACT_EXTENSIONS),
                max_attachment_bytes=args.max_attachment_bytes,
                max_pdf_pages=args.max_pdf_pages,
                max_text_chars_per_attachment=args.max_text_chars_per_attachment,
                max_page_text_chars=args.max_page_text_chars,
                max_pdf_objects_to_scan=args.max_pdf_objects_to_scan,
                pdf_timeout_seconds=args.pdf_timeout_seconds,
                max_attachments=args.max_attachments,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL attachment PDF extract failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_attachment_image_ocr_plan(args: argparse.Namespace) -> int:
    try:
        report = build_mail_attachment_image_ocr_plan(
            MailAttachmentImageOcrPlanConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                parse_plan_db_path=Path(args.parse_plan_db),
                out_dir=Path(args.out_dir),
                stage_extensions=tuple(args.stage_extension or DEFAULT_ATTACHMENT_IMAGE_OCR_EXTENSIONS),
                max_attachment_bytes=args.max_attachment_bytes,
                max_image_dimension=args.max_image_dimension,
                max_image_pixels=args.max_image_pixels,
                inspect_headers=args.inspect_headers,
                max_attachments=args.max_attachments,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL attachment image OCR plan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_attachment_text_index(args: argparse.Namespace) -> int:
    try:
        report = build_mail_attachment_text_index(
            MailAttachmentTextIndexConfig(
                out_dir=Path(args.out_dir),
                text_extract_db_paths=[Path(path) for path in (args.text_extract_db or [])],
                pdf_extract_db_paths=[Path(path) for path in (args.pdf_extract_db or [])],
                image_ocr_plan_db_paths=[Path(path) for path in (args.image_ocr_plan_db or [])],
                parse_plan_db_path=Path(args.parse_plan_db) if args.parse_plan_db else None,
                max_rows=args.max_rows,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL attachment text index failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_attachment_stage6_plan(args: argparse.Namespace) -> int:
    try:
        report = build_mail_attachment_stage6_plan(
            MailAttachmentStage6PlanConfig(
                parse_plan_db_path=Path(args.parse_plan_db),
                text_index_db_path=Path(args.text_index_db),
                out_dir=Path(args.out_dir),
                ocr_pilot_limit=args.ocr_pilot_limit,
                min_pilot_attachment_bytes=args.min_pilot_attachment_bytes,
                max_pilot_attachment_bytes=args.max_pilot_attachment_bytes,
                pilot_extensions=tuple(args.pilot_extension or DEFAULT_ATTACHMENT_IMAGE_OCR_EXTENSIONS),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL attachment stage 6 plan failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_attachment_ocr_preflight(args: argparse.Namespace) -> int:
    try:
        report = build_mail_attachment_ocr_preflight(
            MailAttachmentOcrPreflightConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                stage6_plan_db_path=Path(args.stage6_plan_db),
                out_dir=Path(args.out_dir),
                max_candidates=args.max_candidates,
                max_attachment_bytes=args.max_attachment_bytes,
                verify_sha256=not args.skip_sha256,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL attachment OCR preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def run_attachment_ocr_pilot(args: argparse.Namespace) -> int:
    try:
        report = build_mail_attachment_ocr_pilot(
            MailAttachmentOcrPilotConfig(
                archive_db_paths=[Path(path) for path in args.archive_db],
                ocr_preflight_db_path=Path(args.ocr_preflight_db),
                out_dir=Path(args.out_dir),
                max_candidates=args.max_candidates,
                max_attachment_bytes=args.max_attachment_bytes,
                languages=args.languages,
                page_segmentation_mode=args.psm,
                tesseract_timeout_seconds=args.tesseract_timeout_seconds,
                max_text_chars_per_attachment=args.max_text_chars_per_attachment,
                workers=args.workers,
                tesseract_thread_limit=args.tesseract_thread_limit,
                reuse_existing_ocr_text=args.reuse_existing_ocr_text,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"MAIL attachment OCR pilot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print_summary(report)
    return 0


def print_summary(report: Mapping[str, object]) -> None:
    safe_keys = (
        "schema_version",
        "created_at",
        "preflight_pass",
        "verification_pass",
        "blocking_risks",
        "warnings",
        "batch_mode",
        "requested_pilot",
        "checks",
        "recommended_command",
        "archive_dir",
        "expected_max_messages",
        "ingest_summary",
        "db_counts",
        "file_counts",
        "provenance",
        "source_file",
        "row_count",
        "duplicate_tallanto_id_values",
        "source_id_quality",
        "identity_values",
        "row_identity_classes",
        "row_coverage",
        "identity_link_counts",
        "sanity_checks",
        "audit_readiness",
        "message_count",
        "source_archive_count",
        "candidate_count",
        "matched_message_count",
        "distinct_matched_candidates",
        "distinct_candidate_keys",
        "evaluated_message_count",
        "counts",
        "message_kind_counts",
        "match_class_by_message_kind",
        "mail_link_reconciliation",
        "mango_source_counts",
        "product_source_counts",
        "by_original_match_class",
        "lift_class_counts",
        "potential_lift",
        "attachment_count",
        "parse_plan_counts",
        "risk_level_counts",
        "risk_reason_counts",
        "parse_plan_queue_count",
        "stage_supported_queue_count",
        "stage_skipped_queue_count",
        "status_counts",
        "status_reason_counts",
        "parser_counts",
        "warning_counts",
        "image_format_counts",
        "top_extensions",
        "top_declared_content_types",
        "top_stage_extensions",
        "text_access_counts",
        "source_file_counts",
        "source_db_count",
        "source_text_extract_count",
        "source_pdf_extract_count",
        "source_image_ocr_plan_count",
        "source_stage_counts",
        "source_status_counts",
        "text_status_counts",
        "coverage_counts",
        "gap_count",
        "gap_extension_counts",
        "gap_class_counts",
        "gap_action_counts",
        "ocr_pilot",
        "selected_candidate_count",
        "extension_counts",
        "ocr",
        "phone_index_counts",
        "artifact_counts",
        "artifact_integrity",
        "pii_artifacts",
        "privacy",
        "columns_used",
        "account_label",
        "mailbox",
        "since_days",
        "before_days",
        "window_days",
        "search_criteria",
        "max_messages",
        "messages_found_since",
        "messages_attempted",
        "messages_inserted_or_seen",
        "messages_excluded_by_sha256",
        "raw_eml_written",
        "attachments_written",
        "text_files_written",
        "errors",
        "safety",
        "paths",
    )
    print(
        json.dumps(
            {key: report[key] for key in safe_keys if key in report},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def default_identity_out_dir(account_label: str) -> str:
    return f"{DEFAULT_MAIL_ARCHIVE_ROOT}/{safe_label(account_label)}/identity_map"


def default_ingest_out_dir(account_label: str) -> str:
    suffix = datetime.now().strftime("%Y%m%d")
    return f"{DEFAULT_MAIL_ARCHIVE_ROOT}/{safe_label(account_label)}/pilot_{suffix}"


def default_match_out_dir(account_label: str) -> str:
    suffix = datetime.now().strftime("%Y%m%d")
    return f"{DEFAULT_MAIL_ARCHIVE_ROOT}/{safe_label(account_label)}/matching_{suffix}"


def safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def load_excluded_message_sha256s(paths: Sequence[str]) -> set[str]:
    values: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        uri = f"file:{quote(str(path.resolve(strict=False)), safe='/:')}?mode=ro"
        with sqlite3.connect(uri, uri=True) as con:
            for row in con.execute("SELECT sha256 FROM messages"):
                sha256 = str(row[0] or "").strip()
                if is_full_sha256(sha256):
                    values.add(sha256)
    return values


def is_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def resolve_port(arg_port: int | None) -> int:
    if arg_port:
        return int(arg_port)
    raw = os.environ.get("MAIL_IMAP_PORT", "")
    if not raw:
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        return 0


def is_small_pilot(since_days: int, max_messages: int) -> bool:
    return 1 <= int(since_days) <= 7 and 1 <= int(max_messages) <= 5


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build read-only mail archive artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity-map", help="Build Tallanto email/phone map.")
    identity.add_argument("--tallanto-csv", default=DEFAULT_TALLANTO_CSV)
    identity.add_argument("--out-dir")
    identity.add_argument("--encoding", default="cp1251")
    identity.add_argument("--delimiter", default="\t")
    identity.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)

    identity_union = subparsers.add_parser(
        "identity-map-union",
        help="Build one Tallanto identity map from old students TSV and fresh contacts CSV.",
    )
    identity_union.add_argument("--old-tallanto-csv", required=True)
    identity_union.add_argument("--fresh-tallanto-csv", required=True)
    identity_union.add_argument("--out-dir", required=True)
    identity_union.add_argument("--old-encoding", default="cp1251")
    identity_union.add_argument("--old-delimiter", default="\t")
    identity_union.add_argument("--fresh-encoding", default="utf-8-sig")
    identity_union.add_argument("--fresh-delimiter", default=",")
    identity_union.add_argument(
        "--db-filename",
        default="tallanto_identity_map_union_20260620.sqlite",
    )

    preflight = subparsers.add_parser("preflight", help="Check a live IMAP pilot without network calls.")
    preflight.add_argument("--host", help=f"Defaults to MAIL_IMAP_HOST or {DEFAULT_HOST}.")
    preflight.add_argument("--port", type=int, help=f"Defaults to MAIL_IMAP_PORT or {DEFAULT_PORT}.")
    preflight.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    preflight.add_argument("--password-env", default=DEFAULT_PASSWORD_ENV)
    preflight.add_argument("--dotenv", default=".env", help="Optional dotenv file to load into env.")
    preflight.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    preflight.add_argument("--mailbox", default="INBOX")
    preflight.add_argument("--out-dir")
    preflight.add_argument("--since-days", type=int, default=3)
    preflight.add_argument("--before-days", type=int)
    preflight.add_argument("--max-messages", type=int, default=1)
    preflight.add_argument("--identity-db")
    preflight.add_argument(
        "--allow-large-batch",
        action="store_true",
        help="Approve a controlled non-pilot preflight window after pilot review.",
    )

    ingest = subparsers.add_parser("ingest", help="Ingest one read-only IMAP pilot batch.")
    ingest.add_argument("--host", help=f"Defaults to MAIL_IMAP_HOST or {DEFAULT_HOST}.")
    ingest.add_argument("--port", type=int, help=f"Defaults to MAIL_IMAP_PORT or {DEFAULT_PORT}.")
    ingest.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    ingest.add_argument(
        "--password-env",
        default=DEFAULT_PASSWORD_ENV,
        help=f"Environment variable that contains the mailbox password. Default: {DEFAULT_PASSWORD_ENV}.",
    )
    ingest.add_argument("--dotenv", default=".env", help="Optional dotenv file to load into env.")
    ingest.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    ingest.add_argument("--mailbox", default="INBOX")
    ingest.add_argument("--mailbox-label")
    ingest.add_argument("--out-dir")
    ingest.add_argument("--since-days", type=int, default=3)
    ingest.add_argument("--before-days", type=int)
    ingest.add_argument("--max-messages", type=int, default=1)
    ingest.add_argument(
        "--allow-large-batch",
        action="store_true",
        help="Allow non-pilot windows after a successful small pilot review.",
    )
    ingest.add_argument("--internal-domain", action="append", default=["kmipt.ru"])
    ingest.add_argument("--extracted-text-max-chars", type=int, default=250_000)
    ingest.add_argument(
        "--exclude-archive-db",
        action="append",
        help="Read message sha256 values from an existing archive DB and skip those messages.",
    )

    verify = subparsers.add_parser("verify-pilot", help="Verify a completed mail archive pilot.")
    verify.add_argument("--archive-dir", required=True)
    verify.add_argument("--expected-max-messages", type=int, default=5)

    match = subparsers.add_parser("match-report", help="Build mail to Tallanto matching report.")
    match.add_argument("--archive-db", required=True)
    match.add_argument("--identity-db", required=True)
    match.add_argument("--out-dir")
    match.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    match.add_argument("--account-label", default=DEFAULT_ACCOUNT_LABEL)
    match.add_argument("--internal-domain", action="append", default=["kmipt.ru"])

    handoff = subparsers.add_parser(
        "history-handoff",
        help="Build read-only mail-to-customer history handoff from archive DBs.",
    )
    handoff.add_argument("--archive-db", action="append", required=True)
    handoff.add_argument("--identity-db", required=True)
    handoff.add_argument("--out-dir", required=True)
    handoff.add_argument("--email", help="Defaults to MAIL_IMAP_EMAIL.")
    handoff.add_argument("--internal-domain", action="append", default=["kmipt.ru"])

    bridge = subparsers.add_parser(
        "mango-bridge-preview",
        help="Build read-only mail-to-Mango phone bridge preview.",
    )
    bridge.add_argument("--mail-handoff-db", required=True)
    bridge.add_argument("--identity-db", required=True)
    bridge.add_argument("--product-db", default=DEFAULT_MANGO_PRODUCT_DB)
    bridge.add_argument("--out-dir", required=True)
    bridge.add_argument("--max-call-refs-per-candidate", type=int, default=50)
    bridge.add_argument("--mango-phone-index-db")

    phone_index = subparsers.add_parser(
        "mango-phone-index-preview",
        help="Build a read-only Mango phone index preview from product DB and local recording filenames.",
    )
    phone_index.add_argument("--product-db", default=DEFAULT_MANGO_PRODUCT_DB)
    phone_index.add_argument("--recording-root", action="append")
    phone_index.add_argument("--out-dir", required=True)
    phone_index.add_argument("--skip-product-db", action="store_true")
    phone_index.add_argument("--skip-recording-filenames", action="store_true")

    phone_lift = subparsers.add_parser(
        "phone-lift-preview",
        help="Preview phone-based lifts for ambiguous/missing mail matches.",
    )
    phone_lift.add_argument("--archive-db", action="append", required=True)
    phone_lift.add_argument("--identity-db", required=True)
    phone_lift.add_argument("--out-dir", required=True)
    phone_lift.add_argument("--max-text-chars-per-message", type=int, default=250_000)

    relink = subparsers.add_parser(
        "customer-relink-preview",
        help="Preview conservative mail relinking through tallanto_id address book.",
    )
    relink.add_argument("--mail-handoff-db", required=True)
    relink.add_argument("--identity-db", required=True)
    relink.add_argument("--out-dir", required=True)
    relink.add_argument("--classification-index")
    relink.add_argument("--max-text-chars-per-message", type=int, default=250_000)
    relink.add_argument(
        "--allow-non-real-correspondence",
        action="store_true",
        help="Do not block messages classified as newsletters/service. Use only for diagnostics.",
    )
    relink.add_argument(
        "--live-tallanto-lookup",
        action="store_true",
        help="Use configured Tallanto API for read-only phone lookup when local identity map misses.",
    )
    relink.add_argument("--live-max-records", type=int, default=10)
    relink.add_argument("--dotenv", default=".env", help="Optional dotenv file for live Tallanto lookup.")

    stage2_relink = subparsers.add_parser(
        "stage2-customer-relink-preview",
        help="Preview Stage2 mail event relinking through a Tallanto address book.",
    )
    stage2_relink.add_argument("--event-jsonl", action="append", required=True)
    stage2_relink.add_argument("--identity-db", required=True)
    stage2_relink.add_argument("--out-dir", required=True)
    stage2_relink.add_argument("--max-text-chars-per-message", type=int, default=250_000)

    attachment_plan = subparsers.add_parser(
        "attachment-parse-plan",
        help="Build a safe metadata-only attachment parsing queue without opening files.",
    )
    attachment_plan.add_argument("--archive-db", action="append", required=True)
    attachment_plan.add_argument("--out-dir", required=True)
    attachment_plan.add_argument("--max-size-bytes", type=int, default=20_000_000)
    attachment_plan.add_argument(
        "--allow-extension",
        action="append",
        default=list(DEFAULT_ATTACHMENT_PARSE_ALLOW_EXTENSIONS),
    )
    attachment_plan.add_argument(
        "--review-extension",
        action="append",
        default=list(DEFAULT_ATTACHMENT_PARSE_REVIEW_EXTENSIONS),
    )

    attachment_text = subparsers.add_parser(
        "attachment-text-extract",
        help="Extract text from safe allowlisted attachment formats using an existing parse plan.",
    )
    attachment_text.add_argument("--archive-db", action="append", required=True)
    attachment_text.add_argument("--parse-plan-db", required=True)
    attachment_text.add_argument("--out-dir", required=True)
    attachment_text.add_argument(
        "--stage-extension",
        action="append",
    )
    attachment_text.add_argument("--max-attachment-bytes", type=int, default=10_000_000)
    attachment_text.add_argument("--max-text-chars-per-attachment", type=int, default=100_000)
    attachment_text.add_argument("--max-csv-rows", type=int, default=200)
    attachment_text.add_argument("--max-csv-columns", type=int, default=50)
    attachment_text.add_argument("--max-xlsx-sheets", type=int, default=5)
    attachment_text.add_argument("--max-xlsx-rows-per-sheet", type=int, default=100)
    attachment_text.add_argument("--max-xlsx-columns-per-sheet", type=int, default=50)
    attachment_text.add_argument("--max-zip-members", type=int, default=2_000)
    attachment_text.add_argument("--max-zip-uncompressed-bytes", type=int, default=50_000_000)
    attachment_text.add_argument("--max-attachments", type=int, default=0)

    attachment_pdf = subparsers.add_parser(
        "attachment-pdf-extract",
        help="Extract text from safe PDF attachments using an existing parse plan.",
    )
    attachment_pdf.add_argument("--archive-db", action="append", required=True)
    attachment_pdf.add_argument("--parse-plan-db", required=True)
    attachment_pdf.add_argument("--out-dir", required=True)
    attachment_pdf.add_argument(
        "--stage-extension",
        action="append",
    )
    attachment_pdf.add_argument("--max-attachment-bytes", type=int, default=20_000_000)
    attachment_pdf.add_argument("--max-pdf-pages", type=int, default=10)
    attachment_pdf.add_argument("--max-text-chars-per-attachment", type=int, default=100_000)
    attachment_pdf.add_argument("--max-page-text-chars", type=int, default=25_000)
    attachment_pdf.add_argument("--max-pdf-objects-to-scan", type=int, default=10_000)
    attachment_pdf.add_argument("--pdf-timeout-seconds", type=int, default=10)
    attachment_pdf.add_argument("--max-attachments", type=int, default=0)

    attachment_image = subparsers.add_parser(
        "attachment-image-ocr-plan",
        help="Plan image OCR candidates without running OCR, decoding full images, or writing thumbnails.",
    )
    attachment_image.add_argument("--archive-db", action="append", required=True)
    attachment_image.add_argument("--parse-plan-db", required=True)
    attachment_image.add_argument("--out-dir", required=True)
    attachment_image.add_argument(
        "--stage-extension",
        action="append",
    )
    attachment_image.add_argument("--max-attachment-bytes", type=int, default=20_000_000)
    attachment_image.add_argument("--max-image-dimension", type=int, default=20_000)
    attachment_image.add_argument("--max-image-pixels", type=int, default=50_000_000)
    attachment_image.add_argument("--inspect-headers", action="store_true")
    attachment_image.add_argument("--max-attachments", type=int, default=0)

    attachment_text_index = subparsers.add_parser(
        "attachment-text-index",
        help="Build a metadata-only unified index over attachment text extract outputs.",
    )
    attachment_text_index.add_argument("--text-extract-db", action="append")
    attachment_text_index.add_argument("--pdf-extract-db", action="append")
    attachment_text_index.add_argument("--image-ocr-plan-db", action="append")
    attachment_text_index.add_argument("--parse-plan-db")
    attachment_text_index.add_argument("--out-dir", required=True)
    attachment_text_index.add_argument("--max-rows", type=int, default=0)

    attachment_stage6 = subparsers.add_parser(
        "attachment-stage6-plan",
        help="Build a metadata-only Stage 6 gap report and gated OCR pilot plan.",
    )
    attachment_stage6.add_argument("--parse-plan-db", required=True)
    attachment_stage6.add_argument("--text-index-db", required=True)
    attachment_stage6.add_argument("--out-dir", required=True)
    attachment_stage6.add_argument("--ocr-pilot-limit", type=int, default=15)
    attachment_stage6.add_argument("--min-pilot-attachment-bytes", type=int, default=0)
    attachment_stage6.add_argument("--max-pilot-attachment-bytes", type=int, default=5_000_000)
    attachment_stage6.add_argument("--pilot-extension", action="append")

    attachment_ocr_preflight = subparsers.add_parser(
        "attachment-ocr-preflight",
        help="Verify selected OCR pilot attachment bytes and hashes without running OCR.",
    )
    attachment_ocr_preflight.add_argument("--archive-db", action="append", required=True)
    attachment_ocr_preflight.add_argument("--stage6-plan-db", required=True)
    attachment_ocr_preflight.add_argument("--out-dir", required=True)
    attachment_ocr_preflight.add_argument("--max-candidates", type=int, default=15)
    attachment_ocr_preflight.add_argument("--max-attachment-bytes", type=int, default=5_000_000)
    attachment_ocr_preflight.add_argument("--skip-sha256", action="store_true")

    attachment_ocr_pilot = subparsers.add_parser(
        "attachment-ocr-pilot",
        help="Run Tesseract OCR for verified pilot attachments only.",
    )
    attachment_ocr_pilot.add_argument("--archive-db", action="append", required=True)
    attachment_ocr_pilot.add_argument("--ocr-preflight-db", required=True)
    attachment_ocr_pilot.add_argument("--out-dir", required=True)
    attachment_ocr_pilot.add_argument("--max-candidates", type=int, default=15)
    attachment_ocr_pilot.add_argument("--max-attachment-bytes", type=int, default=5_000_000)
    attachment_ocr_pilot.add_argument("--languages", default="rus+eng")
    attachment_ocr_pilot.add_argument("--psm", type=int, default=6)
    attachment_ocr_pilot.add_argument("--tesseract-timeout-seconds", type=int, default=30)
    attachment_ocr_pilot.add_argument("--max-text-chars-per-attachment", type=int, default=100_000)
    attachment_ocr_pilot.add_argument("--workers", type=int, default=1)
    attachment_ocr_pilot.add_argument("--tesseract-thread-limit", type=int, default=1)
    attachment_ocr_pilot.add_argument("--reuse-existing-ocr-text", action="store_true")

    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
