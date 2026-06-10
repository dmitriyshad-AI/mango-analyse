#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B Analyze test on the same sample of calls.")
    parser.add_argument("--source-db", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cli", required=True, type=Path, help="Path to stable_runtime/run-cli.sh")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.4", "gpt-5.3-codex-spark"],
        help="Codex model ids to compare",
    )
    parser.add_argument(
        "--reasoning",
        default="medium",
        choices=["low", "medium", "high"],
        help="Reasoning effort for both models",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=8,
        help="How many ids from ids-file to use",
    )
    parser.add_argument(
        "--provider",
        default="codex_cli",
        help="Analyze provider to use in the test",
    )
    parser.add_argument(
        "--prompt-profile",
        default="compact",
        choices=["compact", "full"],
        help="Analyze prompt profile to use",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=(
            "Optional explicit arms as name:model:prompt_profile. "
            "When set, --models and --prompt-profile are ignored."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=180,
        help="Codex CLI timeout passed via env",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use ANALYZE_PROVIDER=mock for a fast local smoke test",
    )
    parser.add_argument(
        "--keep-export-files",
        action="store_true",
        help="Keep Analyze history/structured export files. Off by default to avoid raw PII in reports.",
    )
    return parser.parse_args()


def slugify_model(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model).strip("_") or "model"


def read_ids(ids_file: Path, sample_size: int) -> list[int]:
    ids: list[int] = []
    for line in ids_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(int(line))
    if sample_size > 0:
        ids = ids[:sample_size]
    if not ids:
        raise RuntimeError("No sample ids found")
    return ids


def parse_arms(args: argparse.Namespace) -> list[dict[str, str]]:
    arms: list[dict[str, str]] = []
    if args.arms:
        for raw in args.arms:
            parts = raw.split(":")
            if len(parts) != 3:
                raise RuntimeError(f"Invalid arm format: {raw!r}; expected name:model:prompt_profile")
            name, model, prompt_profile = (part.strip() for part in parts)
            if not name or not model or prompt_profile not in {"compact", "full"}:
                raise RuntimeError(f"Invalid arm format: {raw!r}; expected name:model:compact|full")
            arms.append({"name": name, "model": model, "prompt_profile": prompt_profile})
        return arms
    return [
        {"name": slugify_model(model), "model": model, "prompt_profile": args.prompt_profile}
        for model in args.models
    ]


def backup_db(source_db: Path, target_db: Path) -> None:
    target_db.parent.mkdir(parents=True, exist_ok=True)
    if target_db.exists():
        target_db.unlink()
    src = sqlite3.connect(source_db)
    dst = sqlite3.connect(target_db)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def ensure_call_records_table(conn: sqlite3.Connection) -> None:
    if table_exists(conn, "call_records"):
        return
    if not table_exists(conn, "canonical_calls"):
        raise RuntimeError("source db must contain call_records or canonical_calls")
    conn.execute(
        """
        CREATE TABLE call_records (
            id INTEGER PRIMARY KEY,
            source_file VARCHAR(1024),
            source_filename VARCHAR(255),
            source_call_id VARCHAR(128),
            audio_codec VARCHAR(64),
            sample_rate INTEGER,
            channels INTEGER,
            duration_sec FLOAT,
            phone VARCHAR(64),
            manager_name VARCHAR(255),
            direction VARCHAR(32),
            started_at DATETIME,
            transcription_status VARCHAR(16),
            resolve_status VARCHAR(16),
            analysis_status VARCHAR(16),
            sync_status VARCHAR(16),
            transcribe_attempts INTEGER NOT NULL DEFAULT 0,
            resolve_attempts INTEGER NOT NULL DEFAULT 0,
            analyze_attempts INTEGER NOT NULL DEFAULT 0,
            sync_attempts INTEGER NOT NULL DEFAULT 0,
            pipeline_stage VARCHAR(32),
            pipeline_worker_id VARCHAR(64),
            pipeline_claimed_at DATETIME,
            analysis_worker_id VARCHAR(64),
            analysis_claimed_at DATETIME,
            next_retry_at DATETIME,
            dead_letter_stage VARCHAR(16),
            transcript_manager TEXT,
            transcript_client TEXT,
            transcript_text TEXT,
            transcript_variants_json TEXT,
            resolve_json TEXT,
            resolve_quality_score FLOAT,
            analysis_json TEXT,
            amocrm_contact_id INTEGER,
            amocrm_lead_id INTEGER,
            last_error TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO call_records (
            id, source_file, source_filename, source_call_id,
            duration_sec, phone, manager_name, direction, started_at,
            transcription_status, resolve_status, analysis_status, sync_status,
            analyze_attempts, dead_letter_stage, transcript_manager, transcript_client,
            transcript_text, transcript_variants_json, resolve_json, resolve_quality_score,
            analysis_json, amocrm_contact_id, amocrm_lead_id, last_error, created_at, updated_at
        )
        SELECT
            canonical_call_id, source_file, source_filename, source_call_id,
            duration_sec, phone, manager_name, direction, started_at,
            transcription_status, resolve_status, analysis_status, sync_status,
            0, dead_letter_stage, transcript_manager, transcript_client,
            transcript_text, transcript_variants_json, resolve_json, resolve_quality_score,
            analysis_json, amocrm_contact_id, amocrm_lead_id, last_error,
            COALESCE(created_at, started_at, CURRENT_TIMESTAMP),
            COALESCE(selected_updated_at, created_at, started_at, CURRENT_TIMESTAMP)
        FROM canonical_calls
        """
    )
    conn.commit()


def prepare_db_copy(db_path: Path, sample_ids: list[int]) -> None:
    placeholders = ",".join("?" for _ in sample_ids)
    conn = sqlite3.connect(db_path)
    try:
        ensure_call_records_table(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            """
            UPDATE call_records
            SET analysis_status = CASE
                WHEN transcription_status = 'done' AND resolve_status IN ('done', 'skipped') THEN 'done'
                ELSE analysis_status
            END,
                analysis_worker_id = NULL,
                analysis_claimed_at = NULL,
                last_error = NULL
            """
        )
        conn.execute(
            f"""
            UPDATE call_records
            SET analysis_status = 'pending',
                analysis_json = NULL,
                last_error = NULL,
                analyze_attempts = 0,
                analysis_worker_id = NULL,
                analysis_claimed_at = NULL,
                dead_letter_stage = NULL,
                next_retry_at = NULL
            WHERE id IN ({placeholders})
            """,
            sample_ids,
        )
        conn.commit()
    finally:
        conn.close()


def run_analyze(cli_path: Path, db_path: Path, model: str, reasoning: str, provider: str, timeout_sec: int, export_dir: Path, limit: int, prompt_profile: str, keep_export_files: bool = False) -> tuple[int, float, str, str]:
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{db_path}"
    env["ANALYZE_PROVIDER"] = provider
    env["CODEX_MERGE_MODEL"] = model
    env["CODEX_ANALYZE_MODEL"] = model
    env["CODEX_REASONING_EFFORT"] = reasoning
    env["CODEX_CLI_TIMEOUT_SEC"] = str(timeout_sec)
    env["TRANSCRIPT_EXPORT_DIR"] = str(export_dir) if keep_export_files else ""
    env["ANALYZE_PROMPT_PROFILE"] = prompt_profile
    start = time.monotonic()
    cmd = [str(cli_path), "analyze", "--limit", str(limit)]
    if cli_path.name.startswith("python"):
        cmd = [str(cli_path), "-m", "mango_mvp.cli", "analyze", "--limit", str(limit)]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    elapsed = time.monotonic() - start
    return proc.returncode, elapsed, proc.stdout, proc.stderr


def mask_phone(value: Any) -> str:
    digits = re.sub(r"\D+", "", str(value or ""))
    if len(digits) < 7:
        return ""
    return f"{digits[:1]}***{digits[-4:]}"


def redact_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[email]", text, flags=re.I)
    text = re.sub(r"(?<!\d)\+?\d[\d\s().-]{8,}\d(?!\d)", "[phone]", text)
    return text


def filename_fingerprint(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def summarize_call(row: sqlite3.Row) -> dict[str, Any]:
    raw = row["analysis_json"]
    parsed: dict[str, Any] = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"_raw_analysis_json": raw}
    structured = parsed.get("structured_fields") if isinstance(parsed, dict) else {}
    if not isinstance(structured, dict):
        structured = {}
    contacts = structured.get("contacts") if isinstance(structured.get("contacts"), dict) else {}
    student = structured.get("student") if isinstance(structured.get("student"), dict) else {}
    interests = structured.get("interests") if isinstance(structured.get("interests"), dict) else {}
    next_step = structured.get("next_step") if isinstance(structured.get("next_step"), dict) else {}
    meta = parsed.get("analysis_meta") if isinstance(parsed.get("analysis_meta"), dict) else {}
    quality = parsed.get("quality_flags") if isinstance(parsed.get("quality_flags"), dict) else {}
    history_summary_text = redact_text(parsed.get("history_summary"))
    next_step_action_text = redact_text(next_step.get("action"))
    summary = {
        "id": row["id"],
        "source_filename_sha256": filename_fingerprint(row["source_filename"]),
        "duration_sec": row["duration_sec"],
        "manager_present": bool(row["manager_name"]),
        "phone_masked": mask_phone(row["phone"]),
        "analysis_status": row["analysis_status"],
        "analysis_error": redact_text(row["last_error"]),
        "analysis_model": meta.get("analysis_model"),
        "analysis_provider": meta.get("analysis_provider"),
        "analysis_prompt_version": meta.get("analysis_prompt_version") or quality.get("analyze_prompt_version"),
        "history_summary_present": bool(history_summary_text.strip()),
        "history_summary_len": len(history_summary_text),
        "summary_looks_like_dialogue_dump": bool(
            re.search(r"(^|\n)\s*(manager|client|менеджер|клиент)\s*:", history_summary_text, re.I)
        ),
        "summary_contains_english": bool(re.search(r"\b(call back|follow[- ]?up|send material)\b", history_summary_text, re.I)),
        "lead_priority": parsed.get("follow_up_priority") or structured.get("lead_priority") or parsed.get("lead_priority"),
        "follow_up_score": parsed.get("follow_up_score"),
        "follow_up_reason_present": bool(redact_text(parsed.get("follow_up_reason")).strip()),
        "target_product": redact_text(parsed.get("target_product")),
        "preferred_channel": contacts.get("preferred_channel"),
        "email_present": bool(contacts.get("email")),
        "grade_current": student.get("grade_current"),
        "products": interests.get("products"),
        "format": interests.get("format"),
        "subjects": interests.get("subjects"),
        "next_step_action_present": bool(next_step_action_text.strip()),
        "next_step_action_len": len(next_step_action_text),
        "next_step_contains_english": bool(re.search(r"[A-Za-z]{4,}", next_step_action_text)),
        "next_step_due_present": bool(redact_text(next_step.get("due")).strip()),
        "objections": parsed.get("objections") if isinstance(parsed.get("objections"), list) else structured.get("objections"),
        "tags": parsed.get("tags"),
        "quality_flags": parsed.get("quality_flags"),
    }
    return summary


def quality_flags(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "summary_missing": not bool(summary.get("history_summary_present")),
        "summary_looks_like_dialogue_dump": bool(summary.get("summary_looks_like_dialogue_dump")),
        "summary_contains_english": bool(summary.get("summary_contains_english")),
        "next_step_contains_english": bool(summary.get("next_step_contains_english")),
        "marked_non_conversation": "non_conversation" in [str(item).lower() for item in (summary.get("tags") or [])],
        "product_missing": not bool(summary.get("target_product")),
        "subjects_missing": not bool(summary.get("subjects")),
    }


def collect_results(db_path: Path, sample_ids: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in sample_ids)
    try:
        rows = conn.execute(
            f"""
            SELECT id, source_filename, duration_sec, manager_name, phone,
                   analysis_status, last_error, analysis_json
            FROM call_records
            WHERE id IN ({placeholders})
            ORDER BY id
            """,
            sample_ids,
        ).fetchall()
    finally:
        conn.close()
    calls: list[dict[str, Any]] = []
    metrics = {
        "total": len(rows),
        "done": 0,
        "failed": 0,
        "pending": 0,
        "summary_missing": 0,
        "summary_looks_like_dialogue_dump": 0,
        "summary_contains_english": 0,
        "next_step_contains_english": 0,
        "marked_non_conversation": 0,
        "product_missing": 0,
        "subjects_missing": 0,
        "analysis_model_missing": 0,
        "analysis_prompt_version_missing": 0,
    }
    for row in rows:
        summary = summarize_call(row)
        calls.append(summary)
        status = summary["analysis_status"] or "pending"
        if status in metrics:
            metrics[status] += 1
        flags = quality_flags(summary)
        for key, value in flags.items():
            if value:
                metrics[key] += 1
        if not summary.get("analysis_model"):
            metrics["analysis_model_missing"] += 1
        if not summary.get("analysis_prompt_version"):
            metrics["analysis_prompt_version_missing"] += 1
    return calls, metrics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_coverage_row(record: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(calls)
    done = [call for call in calls if call.get("analysis_status") == "done"]
    denominator = max(1, len(done))
    prompt_versions = sorted(
        {str(call.get("analysis_prompt_version")) for call in done if call.get("analysis_prompt_version")}
    )
    history_lengths = [
        int(call.get("history_summary_len") or 0)
        for call in done
        if call.get("history_summary_present")
    ]

    def pct(count: int) -> float:
        return round(count * 100.0 / denominator, 2)

    return {
        "arm": record["arm"],
        "model": record["model"],
        "provider": record["provider"],
        "prompt_profile": record["prompt_profile"],
        "prompt_version": ",".join(prompt_versions),
        "returncode": record["returncode"],
        "total": total,
        "done": len(done),
        "target_product_present_pct": pct(sum(1 for call in done if has_value(call.get("target_product")))),
        "next_step_action_present_pct": pct(sum(1 for call in done if call.get("next_step_action_present"))),
        "objections_present_pct": pct(sum(1 for call in done if has_value(call.get("objections")))),
        "history_summary_present_pct": pct(sum(1 for call in done if call.get("history_summary_present"))),
        "history_summary_avg_len": round(sum(history_lengths) / max(1, len(history_lengths)), 1),
        "analysis_model_missing": record["metrics"].get("analysis_model_missing", 0),
        "analysis_prompt_version_missing": record["metrics"].get("analysis_prompt_version_missing", 0),
        "elapsed_sec": record["elapsed_sec"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_diff_sample(all_calls: dict[str, list[dict[str, Any]]], sample_size: int = 10) -> list[dict[str, Any]]:
    if len(all_calls) < 2:
        return []
    arms = list(all_calls)
    baseline = arms[0]
    baseline_by_id = {call["id"]: call for call in all_calls[baseline]}
    rows: list[dict[str, Any]] = []
    for call_id in sorted(baseline_by_id)[:sample_size]:
        row: dict[str, Any] = {"id": call_id, "baseline_arm": baseline}
        base = baseline_by_id[call_id]
        row[f"{baseline}_target_product"] = base.get("target_product")
        row[f"{baseline}_next_step_action_present"] = bool(base.get("next_step_action_present"))
        row[f"{baseline}_next_step_action_len"] = base.get("next_step_action_len")
        row[f"{baseline}_history_summary_len"] = base.get("history_summary_len")
        for arm in arms[1:]:
            call = next((candidate for candidate in all_calls[arm] if candidate.get("id") == call_id), {})
            row[f"{arm}_target_product"] = call.get("target_product")
            row[f"{arm}_next_step_action_present"] = bool(call.get("next_step_action_present"))
            row[f"{arm}_next_step_action_len"] = call.get("next_step_action_len")
            row[f"{arm}_history_summary_len"] = call.get("history_summary_len")
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    sample_ids = read_ids(args.ids_file, args.sample_size)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {
        "source_db": str(args.source_db.resolve()),
        "ids_file": str(args.ids_file.resolve()),
        "sample_ids": sample_ids,
        "models": [],
        "arms": [],
    }
    provider = "mock" if args.dry_run else args.provider
    coverage_rows: list[dict[str, Any]] = []
    calls_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in parse_arms(args):
        model = arm["model"]
        prompt_profile = arm["prompt_profile"]
        slug = slugify_model(arm["name"])
        model_dir = out_dir / slug
        db_copy = model_dir / "test.db"
        export_dir = model_dir / "exports"
        backup_db(args.source_db.resolve(), db_copy)
        prepare_db_copy(db_copy, sample_ids)
        rc, elapsed, stdout, stderr = run_analyze(
            cli_path=args.cli.resolve(),
            db_path=db_copy,
            model=model,
            reasoning=args.reasoning,
            provider=provider,
            timeout_sec=args.timeout_sec,
            export_dir=export_dir,
            limit=len(sample_ids),
            prompt_profile=prompt_profile,
            keep_export_files=bool(args.keep_export_files),
        )
        calls, metrics = collect_results(db_copy, sample_ids)
        record = {
            "arm": arm["name"],
            "model": model,
            "provider": provider,
            "reasoning": args.reasoning,
            "prompt_profile": prompt_profile,
            "elapsed_sec": round(elapsed, 3),
            "returncode": rc,
            "metrics": metrics,
            "stdout": redact_text(stdout),
            "stderr": redact_text(stderr),
            "db_copy": str(db_copy),
            "export_dir": str(export_dir) if args.keep_export_files else "",
        }
        overall["models"].append(record)
        overall["arms"].append(arm)
        coverage_rows.append(build_coverage_row(record, calls))
        calls_by_arm[arm["name"]] = calls
        write_json(model_dir / "calls.json", calls)
        write_json(model_dir / "summary.json", record)
    write_json(out_dir / "ab_summary.json", overall)
    write_json(out_dir / "coverage_matrix.json", coverage_rows)
    write_csv(out_dir / "coverage_matrix.csv", coverage_rows)
    write_csv(out_dir / "diff_sample.csv", build_diff_sample(calls_by_arm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
