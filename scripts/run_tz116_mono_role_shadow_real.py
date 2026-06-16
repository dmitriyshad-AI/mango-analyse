#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from mango_mvp.config import get_settings
from mango_mvp.services.transcribe import TranscribeService


MODES = {"off", "shadow", "primary"}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    mode = normalize_mode(args.mode)

    db_path = Path(args.db).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    calls = load_mono_calls(
        db_path,
        limit=max(1, int(args.limit)),
        min_duration_sec=max(0.0, float(args.min_duration_sec)),
        order=str(args.order),
    )
    service = build_service(mode, low_info_filter_mode=str(args.low_info_filter_mode or ""))
    llm_attempts = {"count": 0}
    if mode in {"shadow", "primary"}:
        wrap_codex_counter(service, llm_attempts)

    rows: list[dict[str, Any]] = []
    jsonl_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for call in calls:
        result = evaluate_call(call, service=service, mode=mode)
        rows.append(result["row"])
        jsonl_rows.append(result["jsonl"])
        gold_rows.append(result["gold_row"])

    counters = Counter(str(row.get("assignment_provider") or "none") for row in rows)
    status_counts = Counter(str(row.get("status") or "") for row in rows)
    summary = {
        "schema_version": "tz116_mono_role_shadow_real_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "db": str(db_path),
        "limit": int(args.limit),
        "min_duration_sec": float(args.min_duration_sec),
        "loaded_calls": len(calls),
        "assignment_provider_counts": dict(counters),
        "status_counts": dict(status_counts),
        "llm_calls_total": llm_attempts["count"],
        "model_transport": "codex_cli" if mode in {"shadow", "primary"} else "none",
        "low_info_filter_mode": "off" if mode == "off" else (str(args.low_info_filter_mode).strip().lower() or "off"),
        "segment_guard_mode": "off",
        "safety": {
            "reads_db_mode": "ro",
            "writes_db": False,
            "reads_audio": False,
            "runs_asr": False,
            "calls_openai_api": False,
            "uses_openai_api_key": False,
            "writes_crm": False,
            "writes_tallanto": False,
            "mode_default_off": True,
            "primary_blocked_without_flag": False,
            "primary_scope": (
                "offline_low_confidence_codex_selective"
                if mode == "primary"
                else "not_enabled"
            ),
            "segment_guard_forced_off_in_primary": mode == "primary",
        },
    }
    write_csv(out_dir / "mono_role_shadow_real_rows.csv", rows)
    write_csv(out_dir / "mono_role_gold_review_sample.csv", gold_rows)
    write_jsonl(out_dir / "mono_role_shadow_real_results.jsonl", jsonl_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def normalize_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    return mode if mode in MODES else "off"


def build_service(mode: str, *, low_info_filter_mode: str) -> TranscribeService:
    settings = get_settings()
    mono_mode = "off" if mode == "off" else "codex_selective"
    return TranscribeService(
        replace(
            settings,
            mono_role_assignment_mode=mono_mode,
            mono_role_low_info_filter_mode=(
                "off" if mode == "off" else (low_info_filter_mode.strip().lower() or "off")
            ),
            mono_role_segment_guard_mode="off",
            llm_cache_enabled=False,
            llm_cache_dir=".codex_local/tz116_mono_role_shadow_real/cache_disabled",
        )
    )


def wrap_codex_counter(service: TranscribeService, counter: dict[str, int]) -> None:
    original = service._assign_roles_with_codex  # noqa: SLF001

    def counted(turns: list[dict[str, Any]], manager_name: str) -> dict[str, Any]:
        counter["count"] += 1
        return original(turns, manager_name)

    service._assign_roles_with_codex = counted  # type: ignore[method-assign] # noqa: SLF001


def load_mono_calls(
    db_path: Path,
    *,
    limit: int,
    min_duration_sec: float,
    order: str,
) -> list[dict[str, Any]]:
    order_sql = {
        "duration_desc": "duration_sec DESC, canonical_call_id ASC",
        "id_asc": "canonical_call_id ASC",
    }.get(order, "duration_sec DESC, canonical_call_id ASC")
    uri = f"file:{db_path}?mode=ro"
    query = f"""
        SELECT
            canonical_call_id,
            source_filename,
            started_at,
            manager_name,
            duration_sec,
            transcript_text,
            transcript_variants_json
        FROM canonical_calls
        WHERE has_transcript_variants_json = 1
          AND transcript_variants_json LIKE '%mono_or_fallback%'
          AND COALESCE(duration_sec, 0) >= ?
        ORDER BY {order_sql}
        LIMIT ?
    """
    with sqlite3.connect(uri, uri=True) as con:
        con.row_factory = sqlite3.Row
        return [dict(row) for row in con.execute(query, (min_duration_sec, limit)).fetchall()]


def evaluate_call(call: dict[str, Any], *, service: TranscribeService, mode: str) -> dict[str, Any]:
    payload = parse_json_dict(call.get("transcript_variants_json"))
    full = payload.get("full") if isinstance(payload.get("full"), dict) else {}
    full_segments = full.get("segments") or full.get("resolved_segments") or []
    fallback_text = str(full.get("final") or call.get("transcript_text") or "").strip()
    duration = safe_float(call.get("duration_sec"))
    turns = service._build_mono_turns(full_segments, fallback_text, duration)  # noqa: SLF001
    warnings: list[str] = []
    assignment: dict[str, Any] | None = None
    if mode != "off" and turns:
        assignment = service._assign_roles_for_mono(turns, str(call.get("manager_name") or ""), warnings)  # noqa: SLF001
    meta = assignment.get("meta", {}) if assignment else {}
    roles = meta.get("roles") if isinstance(meta.get("roles"), list) else []
    provider = str(meta.get("provider") or "none")
    status = "assigned" if assignment else ("off" if mode == "off" else "not_assigned")
    row = {
        "canonical_call_id": call.get("canonical_call_id", ""),
        "source_filename": call.get("source_filename", ""),
        "started_at": call.get("started_at", ""),
        "manager_name": call.get("manager_name", ""),
        "duration_sec": call.get("duration_sec", ""),
        "turn_count": len(turns),
        "mode": mode,
        "status": status,
        "assignment_provider": provider,
        "confidence": meta.get("confidence", ""),
        "has_both_roles": "Да" if meta.get("has_both_roles") else ("Нет" if assignment else ""),
        "llm_notes": meta.get("notes", ""),
        "warnings": " | ".join(warnings),
        "roles_json": json.dumps(roles, ensure_ascii=False),
        "transcript_preview": fallback_text[:500],
    }
    gold_row = {
        "canonical_call_id": call.get("canonical_call_id", ""),
        "source_filename": call.get("source_filename", ""),
        "started_at": call.get("started_at", ""),
        "manager_name": call.get("manager_name", ""),
        "duration_sec": call.get("duration_sec", ""),
        "turn_count": len(turns),
        "gold_roles": "",
        "notes_for_reviewer": "",
        "turns_json": json.dumps(
            [{"i": idx + 1, "start": turn.get("start"), "text": turn.get("text", "")} for idx, turn in enumerate(turns)],
            ensure_ascii=False,
        ),
    }
    return {
        "row": row,
        "gold_row": gold_row,
        "jsonl": {
            "canonical_call_id": call.get("canonical_call_id"),
            "mode": mode,
            "turns": turns,
            "assignment": assignment,
            "warnings": warnings,
        },
    }


def parse_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# TZ-116 D Real Mono Role Shadow",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Loaded calls: `{summary['loaded_calls']}`",
        f"- LLM calls total: `{summary['llm_calls_total']}`",
        f"- Model transport: `{summary['model_transport']}`",
        f"- Low-info filter mode: `{summary.get('low_info_filter_mode', '')}`",
        f"- Segment guard mode: `{summary.get('segment_guard_mode', '')}`",
        f"- Provider counts: `{json.dumps(summary['assignment_provider_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "Safety: read-only SQLite, no audio, no ASR, no OpenAI API key, no CRM/Tallanto writes; primary is offline-only and low-confidence selective.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-116 D: real mono-call role assignment shadow runner via Codex CLI.")
    parser.add_argument("--db", required=True, help="Path to canonical_calls_master.db; opened read-only.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz116_mono_role_shadow_real")
    parser.add_argument("--mode", choices=sorted(MODES), default="off")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--min-duration-sec", type=float, default=120.0)
    parser.add_argument("--order", choices=("duration_desc", "id_asc"), default="duration_desc")
    parser.add_argument("--low-info-filter-mode", choices=["off", "mark", "filter"], default="mark")
    parser.add_argument("--allow-primary-after-gold-regrede", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
