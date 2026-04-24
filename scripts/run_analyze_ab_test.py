#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def prepare_db_copy(db_path: Path, sample_ids: list[int]) -> None:
    placeholders = ",".join("?" for _ in sample_ids)
    conn = sqlite3.connect(db_path)
    try:
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
                analysis_claimed_at = NULL
            WHERE id IN ({placeholders})
            """,
            sample_ids,
        )
        conn.commit()
    finally:
        conn.close()


def run_analyze(cli_path: Path, db_path: Path, model: str, reasoning: str, provider: str, timeout_sec: int, export_dir: Path, limit: int, prompt_profile: str) -> tuple[int, float, str, str]:
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{db_path}"
    env["ANALYZE_PROVIDER"] = provider
    env["CODEX_MERGE_MODEL"] = model
    env["CODEX_REASONING_EFFORT"] = reasoning
    env["CODEX_CLI_TIMEOUT_SEC"] = str(timeout_sec)
    env["TRANSCRIPT_EXPORT_DIR"] = str(export_dir)
    env["ANALYZE_PROMPT_PROFILE"] = prompt_profile
    start = time.monotonic()
    proc = subprocess.run(
        [str(cli_path), "analyze", "--limit", str(limit)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    elapsed = time.monotonic() - start
    return proc.returncode, elapsed, proc.stdout, proc.stderr


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
    summary = {
        "id": row["id"],
        "source_filename": row["source_filename"],
        "duration_sec": row["duration_sec"],
        "manager_name": row["manager_name"],
        "phone": row["phone"],
        "analysis_status": row["analysis_status"],
        "analysis_error": row["last_error"],
        "history_summary": parsed.get("history_summary"),
        "lead_priority": parsed.get("follow_up_priority") or structured.get("lead_priority") or parsed.get("lead_priority"),
        "follow_up_score": parsed.get("follow_up_score"),
        "follow_up_reason": parsed.get("follow_up_reason"),
        "target_product": parsed.get("target_product"),
        "preferred_channel": contacts.get("preferred_channel"),
        "email": contacts.get("email"),
        "grade_current": student.get("grade_current"),
        "products": interests.get("products"),
        "format": interests.get("format"),
        "subjects": interests.get("subjects"),
        "next_step_action": next_step.get("action"),
        "next_step_due": next_step.get("due"),
        "tags": parsed.get("tags"),
        "quality_flags": parsed.get("quality_flags"),
    }
    return summary


def quality_flags(summary: dict[str, Any]) -> dict[str, bool]:
    text = (summary.get("history_summary") or "").strip()
    next_step_action = (summary.get("next_step_action") or "").strip()
    return {
        "summary_missing": not bool(text),
        "summary_looks_like_dialogue_dump": bool(
            re.search(r"(^|\n)\s*(manager|client|менеджер|клиент)\s*:", text, re.I)
        ),
        "summary_contains_english": bool(re.search(r"\b(call back|follow[- ]?up|send material)\b", text, re.I)),
        "next_step_contains_english": bool(re.search(r"[A-Za-z]{4,}", next_step_action)),
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
    return calls, metrics


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    }
    provider = "mock" if args.dry_run else args.provider
    for model in args.models:
        slug = slugify_model(model)
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
            prompt_profile=args.prompt_profile,
        )
        calls, metrics = collect_results(db_copy, sample_ids)
        record = {
            "model": model,
            "provider": provider,
            "reasoning": args.reasoning,
            "prompt_profile": args.prompt_profile,
            "elapsed_sec": round(elapsed, 3),
            "returncode": rc,
            "metrics": metrics,
            "stdout": stdout,
            "stderr": stderr,
            "db_copy": str(db_copy),
            "export_dir": str(export_dir),
        }
        overall["models"].append(record)
        write_json(model_dir / "calls.json", calls)
        write_json(model_dir / "summary.json", record)
    write_json(out_dir / "ab_summary.json", overall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
