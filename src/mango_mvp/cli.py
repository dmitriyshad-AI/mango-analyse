from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import func, select

from mango_mvp.config import get_settings
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services.analyze import AnalyzeService
from mango_mvp.services.ingest import ingest_from_directory
from mango_mvp.services.resolve import ResolveService
from mango_mvp.services.sync_amocrm import AmoCRMSyncService
from mango_mvp.services.transcribe import TranscribeService
from mango_mvp.services.worker import run_worker


def _json_print(payload):
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _join_list(value: Any) -> str:
    items = []
    for item in _as_list(value):
        text = _clean_str(item)
        if text:
            items.append(text)
    return " | ".join(items)


def cmd_init_db(_args) -> int:
    settings = get_settings()
    init_db(settings)
    _json_print({"ok": True, "database_url": settings.database_url})
    return 0


def cmd_ingest(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        result = ingest_from_directory(
            session,
            recordings_dir=Path(args.recordings_dir),
            metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
            limit=args.limit,
        )
    _json_print(result)
    return 0


def cmd_transcribe(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    service = TranscribeService(settings)
    with session_factory() as session:
        result = service.run(session, limit=args.limit)
    _json_print(result)
    return 0


def cmd_analyze(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    service = AnalyzeService(settings)
    with session_factory() as session:
        result = service.run(session, limit=args.limit)
    _json_print(result)
    return 0


def cmd_resolve(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    service = ResolveService(settings)
    with session_factory() as session:
        result = service.run(session, limit=args.limit)
    _json_print(result)
    return 0


def cmd_export_review_queue(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    service = ResolveService(settings)
    with session_factory() as session:
        result = service.export_manual_review_queue(
            session,
            out_path=Path(args.out),
            limit=args.limit,
        )
    _json_print(result)
    return 0


def cmd_export_failed_resolve_queue(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    service = ResolveService(settings)
    with session_factory() as session:
        result = service.export_failed_resolve_queue(
            session,
            out_path=Path(args.out),
            limit=args.limit,
        )
    _json_print(result)
    return 0


def cmd_export_crm_fields(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        query = select(CallRecord).where(CallRecord.analysis_json.is_not(None))
        if args.only_done:
            query = query.where(CallRecord.analysis_status == "done")
        calls = session.scalars(query.order_by(CallRecord.id.asc()).limit(args.limit)).all()

    rows = []
    for call in calls:
        raw = (call.analysis_json or "").strip()
        if not raw:
            continue
        try:
            analysis = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(analysis, dict):
            continue

        blocks = _as_dict(analysis.get("structured_fields"))
        if not blocks:
            blocks = _as_dict(analysis.get("crm_blocks"))
        people = _as_dict(blocks.get("people"))
        contacts = _as_dict(blocks.get("contacts"))
        student = _as_dict(blocks.get("student"))
        interests = _as_dict(blocks.get("interests"))
        commercial = _as_dict(blocks.get("commercial"))
        next_step = _as_dict(blocks.get("next_step"))

        history_summary = (
            _clean_str(analysis.get("history_summary"))
            or _clean_str(analysis.get("history_short"))
            or _clean_str(analysis.get("summary"))
        )
        target_product = (
            _clean_str(analysis.get("target_product"))
            or (_as_list(interests.get("products"))[:1] or [""])[0]
        )
        row = {
            "id": call.id,
            "source_filename": call.source_filename,
            "source_file": call.source_file,
            "manager_name": call.manager_name or "",
            "phone": call.phone or "",
            "duration_sec": round(float(call.duration_sec or 0.0), 3),
            "history_summary": history_summary,
            "parent_fio": _clean_str(people.get("parent_fio")),
            "child_fio": _clean_str(people.get("child_fio")),
            "email": _clean_str(contacts.get("email")),
            "preferred_channel": _clean_str(contacts.get("preferred_channel")),
            "grade_current": _clean_str(student.get("grade_current")),
            "school": _clean_str(student.get("school")),
            "interests_products": _join_list(interests.get("products")),
            "interests_format": _join_list(interests.get("format")),
            "interests_subjects": _join_list(interests.get("subjects")),
            "exam_targets": _join_list(interests.get("exam_targets")),
            "target_product": _clean_str(target_product),
            "price_sensitivity": _clean_str(commercial.get("price_sensitivity")),
            "budget": _clean_str(commercial.get("budget")) or _clean_str(analysis.get("budget")),
            "discount_interest": _clean_str(commercial.get("discount_interest")),
            "objections": _join_list(blocks.get("objections")) or _join_list(analysis.get("objections")),
            "next_step_action": _clean_str(next_step.get("action")) or _clean_str(analysis.get("next_step")),
            "next_step_due": _clean_str(next_step.get("due")) or _clean_str(analysis.get("timeline")),
            "lead_priority": _clean_str(blocks.get("lead_priority")),
            "personal_offer": _clean_str(analysis.get("personal_offer")),
            "follow_up_score": analysis.get("follow_up_score"),
            "follow_up_reason": _clean_str(analysis.get("follow_up_reason")),
            "tags": _join_list(analysis.get("tags")),
            "analysis_schema_version": _clean_str(analysis.get("analysis_schema_version")),
        }
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix == ".jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif suffix == ".json":
        out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        headers = [
            "id",
            "source_filename",
            "source_file",
            "manager_name",
            "phone",
            "duration_sec",
            "history_summary",
            "parent_fio",
            "child_fio",
            "email",
            "preferred_channel",
            "grade_current",
            "school",
            "interests_products",
            "interests_format",
            "interests_subjects",
            "exam_targets",
            "target_product",
            "price_sensitivity",
            "budget",
            "discount_interest",
            "objections",
            "next_step_action",
            "next_step_due",
            "lead_priority",
            "personal_offer",
            "follow_up_score",
            "follow_up_reason",
            "tags",
            "analysis_schema_version",
        ]
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    _json_print({"exported": len(rows), "out": str(out_path.resolve())})
    return 0


def cmd_sync(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    service = AmoCRMSyncService(settings)
    with session_factory() as session:
        result = service.run(session, limit=args.limit)
    _json_print(result)
    return 0


def cmd_run_all(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)

    with session_factory() as session:
        ingest_result = ingest_from_directory(
            session,
            recordings_dir=Path(args.recordings_dir),
            metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
            limit=args.ingest_limit,
        )
    with session_factory() as session:
        transcribe_result = TranscribeService(settings).run(session, limit=args.stage_limit)
    with session_factory() as session:
        resolve_result = ResolveService(settings).run(session, limit=args.stage_limit)
    with session_factory() as session:
        analyze_result = AnalyzeService(settings).run(session, limit=args.stage_limit)
    with session_factory() as session:
        sync_result = AmoCRMSyncService(settings).run(session, limit=args.stage_limit)

    _json_print(
        {
            "ingest": ingest_result,
            "transcribe": transcribe_result,
            "resolve": resolve_result,
            "analyze": analyze_result,
            "sync": sync_result,
        }
    )
    return 0


def cmd_stats(_args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        total = session.scalar(select(func.count(CallRecord.id))) or 0
        by_transcribe = session.execute(
            select(CallRecord.transcription_status, func.count(CallRecord.id)).group_by(
                CallRecord.transcription_status
            )
        ).all()
        by_analyze = session.execute(
            select(CallRecord.analysis_status, func.count(CallRecord.id)).group_by(
                CallRecord.analysis_status
            )
        ).all()
        by_resolve = session.execute(
            select(CallRecord.resolve_status, func.count(CallRecord.id)).group_by(
                CallRecord.resolve_status
            )
        ).all()
        by_sync = session.execute(
            select(CallRecord.sync_status, func.count(CallRecord.id)).group_by(CallRecord.sync_status)
        ).all()
        by_dead_letter = session.execute(
            select(CallRecord.dead_letter_stage, func.count(CallRecord.id))
            .where(CallRecord.dead_letter_stage.is_not(None))
            .group_by(CallRecord.dead_letter_stage)
        ).all()

    _json_print(
        {
            "total_calls": total,
            "transcription_status": {k: v for k, v in by_transcribe},
            "resolve_status": {k: v for k, v in by_resolve},
            "analysis_status": {k: v for k, v in by_analyze},
            "sync_status": {k: v for k, v in by_sync},
            "dead_letter_stage": {k: v for k, v in by_dead_letter},
        }
    )
    return 0


def cmd_worker(args) -> int:
    settings = get_settings()

    def _progress(payload):
        print(json.dumps({"type": "worker_cycle", **payload}, ensure_ascii=False), flush=True)

    result = run_worker(
        settings,
        stage_limit=args.stage_limit,
        once=args.once,
        poll_sec=args.poll_sec,
        max_idle_cycles=args.max_idle_cycles,
        progress_callback=_progress,
    )
    _json_print(result)
    return 0


def cmd_requeue(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    stage = args.stage
    with session_factory() as session:
        query = select(CallRecord).where(CallRecord.dead_letter_stage.is_not(None))
        if stage != "all":
            query = query.where(CallRecord.dead_letter_stage == stage)
        calls = session.scalars(query.order_by(CallRecord.id.asc()).limit(args.limit)).all()
        updated = 0
        for call in calls:
            if call.dead_letter_stage == "transcribe" or stage == "transcribe" or stage == "all":
                call.transcription_status = "pending"
                call.transcribe_attempts = 0
                call.resolve_status = "pending"
                call.resolve_attempts = 0
                call.resolve_json = None
                call.resolve_quality_score = None
                call.analysis_status = "pending"
                call.sync_status = "pending"
            elif call.dead_letter_stage == "resolve" or stage == "resolve":
                call.resolve_status = "pending"
                call.resolve_attempts = 0
                call.resolve_json = None
                call.resolve_quality_score = None
                call.analysis_status = "pending"
                call.sync_status = "pending"
            elif call.dead_letter_stage == "analyze" or stage == "analyze":
                call.analysis_status = "pending"
                call.analyze_attempts = 0
                call.sync_status = "pending"
            elif call.dead_letter_stage == "sync" or stage == "sync":
                call.sync_status = "pending"
                call.sync_attempts = 0
            call.dead_letter_stage = None
            call.next_retry_at = None
            call.last_error = None
            session.add(call)
            updated += 1
        session.commit()
    _json_print({"stage": stage, "updated": updated})
    return 0


def cmd_reset_transcribe(args) -> int:
    settings = get_settings()
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        query = select(CallRecord)
        if args.only_missing_variants:
            query = query.where(
                (CallRecord.transcript_variants_json.is_(None))
                | (CallRecord.transcript_variants_json == "")
            )
        if args.only_done:
            query = query.where(CallRecord.transcription_status == "done")
        calls = session.scalars(query.order_by(CallRecord.id.asc()).limit(args.limit)).all()
        updated = 0
        for call in calls:
            call.transcription_status = "pending"
            call.resolve_status = "pending"
            call.analysis_status = "pending"
            call.sync_status = "pending"
            call.transcribe_attempts = 0
            call.resolve_attempts = 0
            call.analyze_attempts = 0
            call.sync_attempts = 0
            call.next_retry_at = None
            call.dead_letter_stage = None
            call.resolve_json = None
            call.resolve_quality_score = None
            if args.clear_transcripts:
                call.transcript_manager = None
                call.transcript_client = None
                call.transcript_text = None
                call.transcript_variants_json = None
                call.analysis_json = None
            call.last_error = None
            session.add(call)
            updated += 1
        session.commit()
    _json_print(
        {
            "updated": updated,
            "only_done": args.only_done,
            "only_missing_variants": args.only_missing_variants,
            "clear_transcripts": args.clear_transcripts,
        }
    )
    return 0


def cmd_migrate_analysis_schema(args) -> int:
    settings = get_settings()
    target_version = str(args.target_version or "").strip().lower()
    if target_version != "v2":
        raise RuntimeError(
            f"Unsupported --target-version={target_version!r}. Only 'v2' is available now."
        )

    session_factory = build_session_factory(settings)
    service = AnalyzeService(settings)
    with session_factory() as session:
        query = select(CallRecord).where(CallRecord.analysis_json.is_not(None))
        if args.only_done:
            query = query.where(CallRecord.analysis_status == "done")
        calls = session.scalars(query.order_by(CallRecord.id.asc()).limit(args.limit)).all()

        scanned = 0
        updated = 0
        already_current = 0
        skipped_empty = 0
        errors = 0

        for call in calls:
            scanned += 1
            raw_json = (call.analysis_json or "").strip()
            if not raw_json:
                skipped_empty += 1
                continue
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                errors += 1
                continue
            if not isinstance(payload, dict):
                errors += 1
                continue

            current_version = service.analysis_schema_version(payload)
            if current_version == target_version and not args.force:
                already_current += 1
                continue

            migrated = service.migrate_analysis_payload(call, payload)
            migrated_version = service.analysis_schema_version(migrated)
            if migrated_version != target_version:
                errors += 1
                continue

            if not args.dry_run:
                call.analysis_json = json.dumps(migrated, ensure_ascii=False)
                session.add(call)
            updated += 1

        if not args.dry_run:
            session.commit()

    _json_print(
        {
            "scanned": scanned,
            "updated": updated,
            "already_current": already_current,
            "skipped_empty": skipped_empty,
            "errors": errors,
            "target_version": target_version,
            "dry_run": bool(args.dry_run),
            "only_done": bool(args.only_done),
            "force": bool(args.force),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mango-mvp",
        description="MVP pipeline for Mango call recordings -> transcript -> resolve -> analysis -> amoCRM",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="Create SQLite/Postgres tables")
    p_init.set_defaults(func=cmd_init_db)

    p_ingest = sub.add_parser("ingest", help="Index audio files into DB")
    p_ingest.add_argument("--recordings-dir", required=True)
    p_ingest.add_argument("--metadata-csv")
    p_ingest.add_argument("--limit", type=int, default=None)
    p_ingest.set_defaults(func=cmd_ingest)

    p_transcribe = sub.add_parser("transcribe", help="Transcribe pending calls")
    p_transcribe.add_argument("--limit", type=int, default=100)
    p_transcribe.set_defaults(func=cmd_transcribe)

    p_resolve = sub.add_parser(
        "resolve",
        help="Resolve low-quality transcripts via LLM + rescue ASR fallback",
    )
    p_resolve.add_argument("--limit", type=int, default=100)
    p_resolve.set_defaults(func=cmd_resolve)

    p_analyze = sub.add_parser("analyze", help="Analyze transcribed calls")
    p_analyze.add_argument("--limit", type=int, default=100)
    p_analyze.set_defaults(func=cmd_analyze)

    p_review = sub.add_parser(
        "export-review-queue",
        help="Export manual review queue from resolve stage",
    )
    p_review.add_argument("--out", required=True)
    p_review.add_argument("--limit", type=int, default=10000)
    p_review.set_defaults(func=cmd_export_review_queue)

    p_failed_review = sub.add_parser(
        "export-failed-resolve-queue",
        help="Export failed/dead resolve queue",
    )
    p_failed_review.add_argument("--out", required=True)
    p_failed_review.add_argument("--limit", type=int, default=10000)
    p_failed_review.set_defaults(func=cmd_export_failed_resolve_queue)

    p_crm_export = sub.add_parser(
        "export-crm-fields",
        help="Export analyzed CRM-ready fields from analysis_json",
    )
    p_crm_export.add_argument("--out", required=True)
    p_crm_export.add_argument("--limit", type=int, default=100000)
    p_crm_export.add_argument("--only-done", action="store_true")
    p_crm_export.set_defaults(func=cmd_export_crm_fields)

    p_sync = sub.add_parser("sync", help="Sync analyzed calls to amoCRM")
    p_sync.add_argument("--limit", type=int, default=100)
    p_sync.set_defaults(func=cmd_sync)

    p_run_all = sub.add_parser("run-all", help="Run all stages sequentially")
    p_run_all.add_argument("--recordings-dir", required=True)
    p_run_all.add_argument("--metadata-csv")
    p_run_all.add_argument("--ingest-limit", type=int, default=None)
    p_run_all.add_argument("--stage-limit", type=int, default=100)
    p_run_all.set_defaults(func=cmd_run_all)

    p_stats = sub.add_parser("stats", help="Show pipeline status counters")
    p_stats.set_defaults(func=cmd_stats)

    p_worker = sub.add_parser(
        "worker",
        help="Run resilient background loop for transcribe/resolve/analyze/sync",
    )
    p_worker.add_argument("--stage-limit", type=int, default=100)
    p_worker.add_argument("--once", action="store_true")
    p_worker.add_argument("--poll-sec", type=int, default=None)
    p_worker.add_argument("--max-idle-cycles", type=int, default=None)
    p_worker.set_defaults(func=cmd_worker)

    p_requeue = sub.add_parser("requeue-dead", help="Requeue dead-letter calls back to pending")
    p_requeue.add_argument(
        "--stage",
        choices=["transcribe", "resolve", "analyze", "sync", "all"],
        default="all",
    )
    p_requeue.add_argument("--limit", type=int, default=1000)
    p_requeue.set_defaults(func=cmd_requeue)

    p_reset = sub.add_parser("reset-transcribe", help="Move calls back to pending for re-transcription")
    p_reset.add_argument("--limit", type=int, default=1000)
    p_reset.add_argument("--only-done", action="store_true")
    p_reset.add_argument("--only-missing-variants", action="store_true")
    p_reset.add_argument("--clear-transcripts", action="store_true")
    p_reset.set_defaults(func=cmd_reset_transcribe)

    p_migrate = sub.add_parser(
        "migrate-analysis-schema",
        help="Migrate existing analysis_json payloads to a target schema version",
    )
    p_migrate.add_argument("--limit", type=int, default=10000)
    p_migrate.add_argument("--target-version", default="v2")
    p_migrate.add_argument("--only-done", action="store_true")
    p_migrate.add_argument("--dry-run", action="store_true")
    p_migrate.add_argument("--force", action="store_true")
    p_migrate.set_defaults(func=cmd_migrate_analysis_schema)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
