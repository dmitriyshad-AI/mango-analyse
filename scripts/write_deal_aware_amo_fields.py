#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ENV_FILES = (
    ROOT / "stable_runtime" / "amocrm_runtime" / ".env.private",
    ROOT / "prod_runtime_transfer" / ".env.private",
)


def _load_env_files_early() -> None:
    import os

    for path in ENV_FILES:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
    os.environ.setdefault("DATABASE_URL", f"sqlite:///{(ROOT / 'stable_runtime' / 'amocrm_runtime' / 'amo_runtime.db').resolve()}")


_load_env_files_early()

from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_FIELDS  # noqa: E402
from mango_mvp.deal_aware.amo_rollback import (  # noqa: E402
    RetryPolicy,
    append_snapshot_rows,
    build_pre_write_snapshot_rows,
    call_with_retries,
    write_rollback_manifest,
)
from mango_mvp.deal_aware.deal_writeback import (  # noqa: E402
    DealAwareStage6Paths,
    LIVE_CONFIRMATION,
    build_dry_run_row,
    load_json,
    run_deal_aware_stage6_preflight,
)
from mango_mvp.deal_aware.stage1_snapshot import read_csv, safe_text, write_csv  # noqa: E402


DEFAULT_INPUT = ROOT / "stable_runtime" / "deal_aware_stage5_quality_gate_20260513_v1" / "deal_stage5_stage6_dry_run_candidates.csv"
DEFAULT_STAGE5_SUMMARY = ROOT / "stable_runtime" / "deal_aware_stage5_quality_gate_20260513_v1" / "summary.json"
DEFAULT_FIELD_CATALOG = ROOT / "stable_runtime" / "amocrm_runtime" / "deal_analysis" / "lead_field_catalog_cache.json"
DEFAULT_OUT = ROOT / "stable_runtime" / "deal_aware_stage6_writeback_preflight_20260513_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deal-aware AMO lead fields dry-run/live writer.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--stage5-summary", default=str(DEFAULT_STAGE5_SUMMARY))
    parser.add_argument("--field-catalog-cache", default=str(DEFAULT_FIELD_CATALOG))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT))
    parser.add_argument("--analysis-date", default="2026-05-13")
    parser.add_argument("--stage20-size", type=int, default=20)
    parser.add_argument("--execute-live-write", action="store_true")
    parser.add_argument("--live-confirmation", default="")
    parser.add_argument("--expected-written", type=int, default=None)
    parser.add_argument("--operator-approval", default="")
    parser.add_argument("--max-live-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--delay-ms", type=int, default=750)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume-from-report", default="")
    parser.add_argument("--require-commercial-fields", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_csv = Path(args.input).expanduser().resolve()
    stage5_summary = Path(args.stage5_summary).expanduser().resolve()
    field_catalog_cache = Path(args.field_catalog_cache).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if args.execute_live_write:
        return run_live_write(
            input_csv=input_csv,
            stage5_summary=stage5_summary,
            field_catalog_cache=field_catalog_cache,
            out_root=out_root,
            live_confirmation=args.live_confirmation,
            expected_written=args.expected_written,
            operator_approval=Path(args.operator_approval).expanduser().resolve() if args.operator_approval else None,
            analysis_date=args.analysis_date,
            max_live_rows=args.max_live_rows,
            batch_size=args.batch_size,
            delay_ms=args.delay_ms,
            max_retries=args.max_retries,
            resume_from_report=Path(args.resume_from_report).expanduser().resolve() if args.resume_from_report else None,
            require_commercial_fields=args.require_commercial_fields,
            fail_fast=args.fail_fast,
        )

    summary = run_deal_aware_stage6_preflight(
        DealAwareStage6Paths(
            input_csv=input_csv,
            stage5_summary_json=stage5_summary,
            field_catalog_cache_json=field_catalog_cache,
            out_root=out_root,
            analysis_date=args.analysis_date,
            stage20_size=args.stage20_size,
            require_commercial_fields=args.require_commercial_fields,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["readiness"]["passed_for_stage20_preflight"] else 1


def run_live_write(
    *,
    input_csv: Path,
    stage5_summary: Path,
    field_catalog_cache: Path,
    out_root: Path,
    live_confirmation: str,
    expected_written: int | None,
    operator_approval: Path | None,
    analysis_date: str,
    max_live_rows: int | None = None,
    batch_size: int = 10,
    delay_ms: int = 750,
    max_retries: int = 3,
    resume_from_report: Path | None = None,
    require_commercial_fields: bool = False,
    fail_fast: bool = False,
    session_factory=None,
    preflight_func=None,
    fetch_lead_func=None,
    send_update_func=None,
    snapshot_writer=append_snapshot_rows,
    sleep_func=None,
) -> int:
    if live_confirmation != LIVE_CONFIRMATION:
        print(f"Refusing live AMO deal-aware writeback: --live-confirmation must be {LIVE_CONFIRMATION!r}.", file=sys.stderr)
        return 2
    approval = load_json(operator_approval) if operator_approval else {}
    if not approval or safe_text(approval.get("input")) != str(input_csv):
        print("Refusing live AMO deal-aware writeback: operator approval is missing or points to another input.", file=sys.stderr)
        return 2
    if expected_written is not None and int(approval.get("expected_written") or -1) != expected_written:
        print("Refusing live AMO deal-aware writeback: operator approval expected_written mismatch.", file=sys.stderr)
        return 2

    if session_factory is None or preflight_func is None or fetch_lead_func is None or send_update_func is None:
        from scripts.write_amo_ready_contacts import _load_env_files, _preflight_runtime_db  # noqa: PLC0415

        _load_env_files()
        from mango_mvp.amocrm_runtime.amo_integration import fetch_lead, send_lead_custom_field_update  # noqa: PLC0415
        from mango_mvp.amocrm_runtime.db import SessionLocal  # noqa: PLC0415

        session_factory = session_factory or SessionLocal
        preflight_func = preflight_func or _preflight_runtime_db
        fetch_lead_func = fetch_lead_func or fetch_lead
        send_update_func = send_update_func or send_lead_custom_field_update
    from mango_mvp.deal_aware.deal_writeback import sha256_file, validate_field_catalog  # noqa: PLC0415

    out_root.mkdir(parents=True, exist_ok=True)
    field_catalog_payload = load_json(field_catalog_cache)
    field_catalog = field_catalog_payload.get("fields") if isinstance(field_catalog_payload.get("fields"), list) else []
    field_guard = validate_field_catalog(field_catalog, require_commercial_fields=require_commercial_fields)
    rows = read_csv(input_csv)
    input_sha256 = sha256_file(input_csv)
    batch_id = out_root.name
    retry_policy = RetryPolicy(max_retries=max_retries, delay_ms=delay_ms, sleep_func=sleep_func or (lambda seconds: __import__("time").sleep(seconds)))
    resume_keys = load_written_resume_keys(resume_from_report)
    write_rollback_manifest(
        out_root,
        batch_id=batch_id,
        input_csv=input_csv,
        input_sha256=input_sha256,
        field_catalog_cache=field_catalog_cache,
        operator_approval_path=operator_approval,
    )
    if max_live_rows is not None and len(rows) > max_live_rows:
        print(f"Refusing live AMO deal-aware writeback: input has {len(rows)} rows, max-live-rows is {max_live_rows}.", file=sys.stderr)
        return 2
    if max_live_rows is not None and expected_written is not None and expected_written > max_live_rows:
        print(
            f"Refusing live AMO deal-aware writeback: expected-written {expected_written} exceeds max-live-rows {max_live_rows}.",
            file=sys.stderr,
        )
        return 2
    session = session_factory()
    report_rows: list[dict[str, Any]] = []
    try:
        ok, error = preflight_func(session)
        if not ok:
            summary = live_summary(
                out_root,
                input_csv,
                rows,
                report_rows,
                preflight_failed=True,
                preflight_error=error,
                expected_written=expected_written,
                batch_size=batch_size,
                delay_ms=delay_ms,
                max_retries=max_retries,
            )
            write_live_outputs(out_root, report_rows, summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 2
        for index, row in enumerate(rows, start=1):
            dry_row, findings = build_dry_run_row(
                row,
                row_index=index,
                field_catalog=field_catalog,
                field_guard=field_guard,
                analysis_date=analysis_date,
                require_commercial_fields=require_commercial_fields,
            )
            report_row = {
                "row_index": index,
                "mode": "live_write",
                "lead_id": safe_text(row.get("selected_deal_id") or row.get("lead_id")),
                "review_id": safe_text(row.get("review_id")),
                "status": "",
                "reason": "",
                "updated_fields": "",
                "preview_payload": dry_row.get("preview_payload", "{}"),
                "snapshot_status": "",
                "snapshot_rows": 0,
            }
            resume_key = f"{index}|{report_row['lead_id']}"
            if resume_key in resume_keys:
                report_row["status"] = "skipped"
                report_row["reason"] = "resume_written_already_processed"
                report_rows.append(report_row)
                write_live_outputs(
                    out_root,
                    report_rows,
                    live_summary(
                        out_root,
                        input_csv,
                        rows,
                        report_rows,
                        preflight_failed=False,
                        preflight_error="",
                        expected_written=expected_written,
                        batch_size=batch_size,
                        delay_ms=delay_ms,
                        max_retries=max_retries,
                    ),
                )
                continue
            if findings or dry_row.get("stage6_status") != "dry_run":
                report_row["status"] = "failed"
                report_row["reason"] = safe_text(dry_row.get("stage6_reason")) or "stage6_preflight_failed"
                report_rows.append(report_row)
                write_live_outputs(
                    out_root,
                    report_rows,
                    live_summary(
                        out_root,
                        input_csv,
                        rows,
                        report_rows,
                        preflight_failed=False,
                        preflight_error="",
                        expected_written=expected_written,
                        batch_size=batch_size,
                        delay_ms=delay_ms,
                        max_retries=max_retries,
                    ),
                )
                if fail_fast:
                    break
                continue
            try:
                payload = json.loads(safe_text(dry_row.get("preview_payload")) or "{}")
                lead_id = int(report_row["lead_id"])
                current_lead = call_with_retries(lambda: fetch_lead_func(session, lead_id=lead_id), retry_policy=retry_policy)
                snapshot_rows = build_pre_write_snapshot_rows(
                    batch_id=batch_id,
                    input_csv=input_csv,
                    input_sha256=input_sha256,
                    row_index=index,
                    review_id=report_row["review_id"],
                    lead_id=report_row["lead_id"],
                    payload=payload,
                    current_lead=current_lead,
                    field_catalog=field_catalog,
                    operator_approval_path=operator_approval,
                )
                snapshot_writer(out_root, snapshot_rows)
                report_row["snapshot_status"] = "saved"
                report_row["snapshot_rows"] = len(snapshot_rows)
                result = call_with_retries(
                    lambda: send_update_func(session, lead_id=lead_id, field_payload=payload),
                    retry_policy=retry_policy,
                )
                session.commit()
                report_row["status"] = "written"
                report_row["reason"] = "ok"
                report_row["updated_fields"] = " | ".join(result.get("updated_fields") or list(DEAL_AI_FIELDS))
                if delay_ms > 0:
                    retry_policy.sleep_func(delay_ms / 1000)
            except Exception as exc:  # noqa: BLE001
                session.rollback()
                report_row["status"] = "failed"
                if not report_row.get("snapshot_status"):
                    report_row["snapshot_status"] = "failed"
                    report_row["reason"] = f"snapshot_failed_or_pre_patch_failed: {exc}"
                else:
                    report_row["reason"] = str(exc)
            report_rows.append(report_row)
            write_live_outputs(
                out_root,
                report_rows,
                live_summary(
                    out_root,
                    input_csv,
                    rows,
                    report_rows,
                    preflight_failed=False,
                    preflight_error="",
                    expected_written=expected_written,
                    batch_size=batch_size,
                    delay_ms=delay_ms,
                    max_retries=max_retries,
                ),
            )
            if fail_fast and report_row["status"] == "failed":
                break
            progress_step = max(1, batch_size)
            if index % progress_step == 0 or index == len(rows):
                written = sum(1 for item in report_rows if item["status"] == "written")
                failed = sum(1 for item in report_rows if item["status"] == "failed")
                print(f"[{index}/{len(rows)}] written={written} failed={failed}", flush=True)
    finally:
        session.close()

    summary = live_summary(
        out_root,
        input_csv,
        rows,
        report_rows,
        preflight_failed=False,
        preflight_error="",
        expected_written=expected_written,
        batch_size=batch_size,
        delay_ms=delay_ms,
        max_retries=max_retries,
    )
    write_live_outputs(out_root, report_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 and not summary["expected_count_mismatch"] else 1


def live_summary(
    out_root: Path,
    input_csv: Path,
    rows: list[dict[str, Any]],
    report_rows: list[dict[str, Any]],
    *,
    preflight_failed: bool,
    preflight_error: str,
    expected_written: int | None,
    batch_size: int = 10,
    delay_ms: int = 750,
    max_retries: int = 3,
) -> dict[str, Any]:
    written = sum(1 for row in report_rows if row.get("status") == "written")
    failed = sum(1 for row in report_rows if row.get("status") == "failed")
    expected_mismatch = expected_written is not None and written != expected_written
    return {
        "schema_version": "deal_aware_stage6_live_writeback_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "live_write",
        "live_write": True,
        "input": str(input_csv),
        "total_rows": len(rows),
        "written": written,
        "failed": failed,
        "expected_written": expected_written,
        "expected_count_mismatch": expected_mismatch,
        "preflight_failed": preflight_failed,
        "preflight_error": preflight_error,
        "target_fields": list(DEAL_AI_FIELDS),
        "snapshot": {
            "required_before_patch": True,
            "pre_write_snapshot_jsonl": str(out_root / "pre_write_snapshot.jsonl"),
            "pre_write_snapshot_csv": str(out_root / "pre_write_snapshot.csv"),
            "rollback_manifest": str(out_root / "rollback_manifest.json"),
        },
        "rate_limit_policy": {
            "batch_size": batch_size,
            "delay_ms": delay_ms,
            "max_retries": max_retries,
        },
        "report_dir": str(out_root),
    }


def write_live_outputs(out_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    write_csv(out_root / "deal_stage6_writeback_report.csv", rows)
    write_csv(out_root / "deal_stage6_dry_run_report.csv", rows)
    write_csv(out_root / "live_write_report.csv", rows)
    (out_root / "deal_stage6_writeback_report.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "live_write_report.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def load_written_resume_keys(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else []
    else:
        rows = read_csv(path)
    return {
        f"{safe_text(row.get('row_index'))}|{safe_text(row.get('lead_id'))}"
        for row in rows
        if safe_text(row.get("status")) == "written"
    }


if __name__ == "__main__":
    raise SystemExit(main())
