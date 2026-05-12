from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url


BACKFILL_VERSION = "safe_non_contentful_v1"
GUARDRAILS_VERSION = "non_conversation_v2"
SAFE_REVIEW_DECISION = "safe_auto_apply_candidate"
SAFE_CALL_TYPES = {"non_conversation", "unknown"}
SAFE_CONSENSUS_ROUTE = "auto_apply_force_non_conversation"
SAFE_FINAL_DECISION = "force_non_conversation"
SAFE_FINAL_SOURCES = {"mini", "advanced", "claude"}
SAFE_LLM_CONSENSUS_MIN_CONFIDENCE = 0.90
SAFE_HARD_GATE_CONSENSUS_QUEUE = "consensus_auto_apply"
SAFE_HARD_GATE_GPT_QUEUE = "gpt_auto_apply"
SAFE_HARD_GATE_DECISION = "safe_apply"
SAFE_HARD_GATE_REVIEW_DECISIONS = {
    "hard_gate_consensus_auto_apply",
    "hard_gate_gpt_auto_apply",
}


@dataclass(frozen=True)
class TranscriptQualityBackfillConfig:
    database_url: str
    candidates_csv: Path
    out_root: Path
    mode: str = "dry-run"
    limit: int | None = None
    create_backup: bool = True


def run_transcript_quality_backfill(config: TranscriptQualityBackfillConfig) -> dict[str, Any]:
    if config.mode not in {"dry-run", "apply"}:
        raise ValueError("mode must be 'dry-run' or 'apply'")
    db_path = _sqlite_path_from_database_url(config.database_url)
    candidates_path = config.candidates_csv.resolve()
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {db_path}")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_path}")

    raw_candidates = _read_csv(candidates_path)
    if config.limit is not None:
        raw_candidates = raw_candidates[: max(0, int(config.limit))]

    backup_path: Path | None = None
    if config.mode == "apply" and config.create_backup:
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{db_path.stem}_transcript_quality_backfill_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}{db_path.suffix}"
        shutil.copy2(db_path, backup_path)

    planned_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    applied_rows: list[dict[str, Any]] = []
    already_applied_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    now_sql = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    now_iso = datetime.now(timezone.utc).isoformat()

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        for candidate in raw_candidates:
            blockers = _candidate_blockers(candidate)
            if blockers:
                blocked_rows.append(_report_row(candidate, status="blocked", blockers=blockers))
                continue
            row = _fetch_call(con, candidate)
            if row is None:
                missing_rows.append(_report_row(candidate, status="missing_in_db", blockers=["missing_in_db"]))
                continue
            current_analysis = _safe_json_object(row["analysis_json"])
            if _already_applied(current_analysis, candidate):
                already_applied_rows.append(_report_row(candidate, status="already_applied", blockers=[]))
                continue
            try:
                update_payload = _build_update_payload(
                    row,
                    candidate,
                    existing_analysis=current_analysis,
                    candidates_csv=candidates_path,
                    applied_at=now_iso,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(_report_row(candidate, status="error", blockers=[str(exc)]))
                continue

            planned = _report_row(candidate, status="planned", blockers=[])
            planned["new_history_summary"] = update_payload["history_summary"]
            planned_rows.append(planned)

            if config.mode == "apply":
                _apply_update(con, row, update_payload, now_sql)
                applied = dict(planned)
                applied["status"] = "applied"
                applied_rows.append(applied)

        if config.mode == "apply":
            con.commit()
    except Exception:
        if config.mode == "apply":
            con.rollback()
        raise
    finally:
        con.close()

    outputs = {
        "summary_json": out_root / "summary.json",
        "report_markdown": out_root / "TRANSCRIPT_QUALITY_BACKFILL_REPORT.md",
        "planned_updates_csv": out_root / "planned_updates.csv",
        "applied_updates_csv": out_root / "applied_updates.csv",
        "blocked_rows_csv": out_root / "blocked_rows.csv",
        "already_applied_csv": out_root / "already_applied.csv",
        "missing_rows_csv": out_root / "missing_rows.csv",
        "errors_csv": out_root / "errors.csv",
    }
    _write_csv(outputs["planned_updates_csv"], planned_rows)
    _write_csv(outputs["applied_updates_csv"], applied_rows)
    _write_csv(outputs["blocked_rows_csv"], blocked_rows)
    _write_csv(outputs["already_applied_csv"], already_applied_rows)
    _write_csv(outputs["missing_rows_csv"], missing_rows)
    _write_csv(outputs["errors_csv"], errors)

    summary = {
        "generated_at": now_iso,
        "mode": config.mode,
        "database_url": _redact_database_url(config.database_url),
        "database_path": str(db_path),
        "candidates_csv": str(candidates_path),
        "backup_path": str(backup_path) if backup_path else None,
        "input_candidates": len(raw_candidates),
        "planned_updates": len(planned_rows),
        "applied_updates": len(applied_rows),
        "blocked_rows": len(blocked_rows),
        "already_applied": len(already_applied_rows),
        "missing_rows": len(missing_rows),
        "errors": len(errors),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["report_markdown"].write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _candidate_blockers(candidate: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    outbound_voicemail = _is_outbound_voicemail_candidate(candidate)
    llm_consensus_candidate = _is_llm_consensus_force_candidate(candidate)
    hard_gate_consensus_candidate = _has_hard_gate_consensus_columns(candidate)
    if not _clean(candidate.get("id")):
        blockers.append("missing_id")
    if not _clean(candidate.get("source_filename")):
        blockers.append("missing_source_filename")
    if hard_gate_consensus_candidate:
        return blockers + _hard_gate_consensus_blockers(candidate)
    if _clean(candidate.get("review_decision")) != SAFE_REVIEW_DECISION:
        blockers.append("review_decision_not_safe")
    if _is_true(candidate.get("current_contentful")) and not outbound_voicemail:
        blockers.append("current_contentful_true")
    if (_clean(candidate.get("current_call_type")) or "unknown") not in SAFE_CALL_TYPES and not outbound_voicemail:
        blockers.append("current_call_type_not_safe")
    if llm_consensus_candidate:
        confidence = _safe_float(_candidate_confidence(candidate))
        if confidence is None or confidence < SAFE_LLM_CONSENSUS_MIN_CONFIDENCE:
            blockers.append("llm_consensus_confidence_too_low")
        return blockers
    if not _is_true(candidate.get("should_force_non_conversation")):
        blockers.append("should_force_non_conversation_false")
    if _clean(candidate.get("guardrail_label")) != "non_conversation_high_confidence":
        blockers.append("guardrail_label_not_high_confidence")
    return blockers


def _has_hard_gate_consensus_columns(candidate: dict[str, Any]) -> bool:
    return any(
        _clean(candidate.get(key))
        for key in (
            "consensus_queue",
            "gpt_decision",
            "claude_decision",
            "policy_auto_apply_allowed",
        )
    )


def _hard_gate_consensus_blockers(candidate: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    policy_allowed = _clean(candidate.get("policy_auto_apply_allowed"))
    consensus_queue = _clean(candidate.get("consensus_queue"))
    policy_queue = _clean(candidate.get("policy_queue"))
    if policy_allowed and not _is_true(policy_allowed):
        blockers.append("hard_gate_policy_auto_apply_not_allowed")
    if not policy_queue and consensus_queue and consensus_queue not in {SAFE_HARD_GATE_CONSENSUS_QUEUE, SAFE_HARD_GATE_GPT_QUEUE}:
        blockers.append("hard_gate_consensus_queue_not_auto_apply")
    if policy_queue and policy_queue not in {SAFE_HARD_GATE_CONSENSUS_QUEUE, SAFE_HARD_GATE_GPT_QUEUE}:
        blockers.append("hard_gate_policy_queue_not_auto_apply")
    if _clean(candidate.get("gpt_decision")) != SAFE_HARD_GATE_DECISION:
        blockers.append("hard_gate_gpt_decision_not_safe_apply")
    if _clean(candidate.get("review_decision")) not in {"", *SAFE_HARD_GATE_REVIEW_DECISIONS}:
        blockers.append("hard_gate_review_decision_not_safe")
    if _clean(candidate.get("guardrail_label")) != "non_conversation_high_confidence":
        blockers.append("guardrail_label_not_high_confidence")
    if not _is_true(candidate.get("should_force_non_conversation")):
        blockers.append("should_force_non_conversation_false")
    if _is_true(candidate.get("guardrail_requires_manual_review")) or _is_true(candidate.get("requires_manual_review")):
        blockers.append("guardrail_requires_manual_review")
    if _is_true(candidate.get("guardrail_protected_live_dialogue")) or _is_true(candidate.get("protected_live_dialogue")):
        blockers.append("guardrail_protected_live_dialogue")
    return blockers


def _is_llm_consensus_force_candidate(candidate: dict[str, Any]) -> bool:
    final_source = _clean(candidate.get("final_source"))
    return (
        _clean(candidate.get("consensus_route")) == SAFE_CONSENSUS_ROUTE
        and _clean(candidate.get("final_decision")) == SAFE_FINAL_DECISION
        and final_source in SAFE_FINAL_SOURCES
    )


def _candidate_confidence(candidate: dict[str, Any]) -> Any:
    final_source = _clean(candidate.get("final_source"))
    if final_source == "advanced":
        return candidate.get("advanced_confidence")
    if final_source == "mini":
        return candidate.get("mini_confidence")
    if final_source == "claude":
        return candidate.get("claude_confidence") or candidate.get("confidence")
    return None


def _is_outbound_voicemail_candidate(candidate: dict[str, Any]) -> bool:
    reason_codes = _split_codes(candidate.get("guardrail_reason_codes"))
    subtype = _clean(candidate.get("recommended_contact_subtype"))
    return "outbound_voicemail" in reason_codes or subtype == "outbound_voicemail"


def _fetch_call(con: sqlite3.Connection, candidate: dict[str, Any]) -> sqlite3.Row | None:
    row_id = int(float(_clean(candidate.get("id"))))
    source_filename = _clean(candidate.get("source_filename"))
    return con.execute(
        """
        select *
          from call_records
         where id = ?
           and source_filename = ?
        """,
        (row_id, source_filename),
    ).fetchone()


def _build_update_payload(
    row: sqlite3.Row,
    candidate: dict[str, Any],
    *,
    existing_analysis: dict[str, Any],
    candidates_csv: Path,
    applied_at: str,
) -> dict[str, Any]:
    transcript = "\n".join(
        _clean(part)
        for part in [row["transcript_manager"], row["transcript_client"], row["transcript_text"]]
        if _clean(part)
    )
    reason_codes = _split_codes(candidate.get("guardrail_reason_codes"))
    summary, reason = _classify_summary(transcript, reason_codes=reason_codes)
    contact_subtype = _clean(candidate.get("recommended_contact_subtype")) or (
        "outbound_voicemail" if "outbound_voicemail" in reason_codes else "no_live_or_voicemail"
    )
    started_text = str(row["started_at"] or "").replace("T", " ").strip()
    manager = _clean(row["manager_name"]) or "не указан"
    phone = _clean(row["phone"])
    structured_fields = _empty_structured_fields(phone)
    existing_quality = _safe_dict(existing_analysis.get("quality_flags"))
    quality_flags = dict(existing_quality)
    llm_consensus_candidate = _is_llm_consensus_force_candidate(candidate)
    backfill_label = "llm_consensus_force_non_conversation" if llm_consensus_candidate else "non_conversation_high_confidence"
    backfill_meta = {
        "version": BACKFILL_VERSION,
        "applied_at": applied_at,
        "review_hash": _clean(candidate.get("review_hash")),
        "source_candidates_csv": str(candidates_csv),
        "source_review_decision": _clean(candidate.get("review_decision")),
    }
    for source_key, target_key in [
        ("consensus_route", "source_consensus_route"),
        ("consensus_reason", "source_consensus_reason"),
        ("gpt_decision", "source_gpt_decision"),
        ("claude_decision", "source_claude_decision"),
        ("audit_id", "source_audit_id"),
        ("final_source", "source_final_source"),
        ("final_decision", "source_final_decision"),
        ("mini_confidence", "source_mini_confidence"),
        ("advanced_confidence", "source_advanced_confidence"),
        ("claude_confidence", "source_claude_confidence"),
    ]:
        value = _clean(candidate.get(source_key))
        if value:
            backfill_meta[target_key] = value
    quality_flags.update(
        {
            "call_type": "non_conversation",
            "needs_review": False,
            "review_reasons": [],
            "transcript_quality_guardrails_version": GUARDRAILS_VERSION,
            "transcript_quality_guardrails_mode": "backfill_apply",
            "transcript_quality_label": backfill_label,
            "transcript_quality_score": _safe_int(candidate.get("guardrail_score")),
            "transcript_quality_reason_codes": reason_codes,
            "transcript_quality_should_force_non_conversation": True,
            "transcript_quality_requires_manual_review": False,
            "transcript_quality_protected_live_dialogue": False,
            "transcript_quality_recommended_call_type": "non_conversation",
            "transcript_quality_recommended_contact_subtype": contact_subtype,
            "transcript_quality_backfill": backfill_meta,
        }
    )
    quality_flags["transcript_quality_guardrails"] = {
        "version": GUARDRAILS_VERSION,
        "mode": "backfill_apply",
        "label": backfill_label,
        "score": _safe_int(candidate.get("guardrail_score")),
        "reason_codes": reason_codes,
        "should_force_non_conversation": True,
        "requires_manual_review": False,
        "protected_live_dialogue": False,
        "recommended_call_type": "non_conversation",
        "recommended_contentful": False,
        "recommended_contact_subtype": contact_subtype,
        "backfill_version": BACKFILL_VERSION,
    }
    no_live_subtypes = {"outbound_voicemail", "probable_no_live", "no_live_or_voicemail"}
    interaction_text = (
        "пытался связаться с клиентом"
        if contact_subtype in no_live_subtypes or "no_live_marker" in reason_codes
        else "обрабатывал несодержательный звонок"
    )
    history_summary = (
        f"{started_text[:16]} менеджер {manager} {interaction_text}. "
        f"{summary} Итог: Нет содержательного диалога менеджер-клиент для анализа продаж."
    ).strip()
    return {
        "analysis_schema_version": "v2",
        "history_summary": history_summary,
        "structured_fields": structured_fields,
        "history_short": summary,
        "crm_blocks": structured_fields,
        "evidence": [],
        "quality_flags": quality_flags,
        "summary": summary,
        "interests": [],
        "student_grade": None,
        "target_product": None,
        "personal_offer": None,
        "pain_points": [],
        "budget": None,
        "timeline": None,
        "objections": [],
        "next_step": None,
        "follow_up_score": 0,
        "follow_up_reason": reason,
        "tags": ["non_conversation"],
        "needs_review": False,
        "review_reasons": [],
    }


def _apply_update(con: sqlite3.Connection, row: sqlite3.Row, update_payload: dict[str, Any], now_sql: str) -> None:
    resolve_status = _clean(row["resolve_status"])
    next_resolve_status = "skipped" if resolve_status in {"", "pending", "manual", "failed"} else resolve_status
    resolve_json = row["resolve_json"]
    resolve_quality_score = row["resolve_quality_score"]
    if next_resolve_status == "skipped" and not _clean(resolve_json):
        resolve_json = json.dumps(
            {
                "version": "v1",
                "decision": "transcript_quality_backfill_non_conversation",
                "reasons": ["transcript_quality_backfill", "non_conversation"],
                "note": update_payload["follow_up_reason"],
                "ts_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        )
        resolve_quality_score = 100.0
    con.execute(
        """
        update call_records
           set resolve_status = ?,
               analysis_status = 'done',
               sync_status = 'pending',
               resolve_json = ?,
               resolve_quality_score = ?,
               analysis_json = ?,
               analyze_attempts = case when analyze_attempts < 1 then 1 else analyze_attempts end,
               dead_letter_stage = null,
               last_error = null,
               next_retry_at = null,
               updated_at = ?
         where id = ?
        """,
        (
            next_resolve_status,
            resolve_json,
            resolve_quality_score,
            json.dumps(update_payload, ensure_ascii=False),
            now_sql,
            int(row["id"]),
        ),
    )


def _already_applied(analysis: dict[str, Any], candidate: dict[str, Any]) -> bool:
    quality = _safe_dict(analysis.get("quality_flags"))
    meta = _safe_dict(quality.get("transcript_quality_backfill"))
    return meta.get("version") == BACKFILL_VERSION and _clean(meta.get("review_hash")) == _clean(candidate.get("review_hash"))


def _classify_summary(text: str, *, reason_codes: list[str] | None = None) -> tuple[str, str]:
    normalized = _clean(text).lower()
    if "outbound_voicemail" in (reason_codes or []):
        return (
            "Нецелевой звонок: менеджер оставил сообщение на автоответчике/голосовой почте, живого диалога не было.",
            "Менеджер говорил в автоответчик или системную голосовую почту; клиент не вступал в диалог, поэтому звонок нельзя использовать как содержательный разговор.",
        )
    if any(token in normalized for token in ("голосов", "почтовый ящик", "автоответчик", "оставьте сообщение")):
        return (
            "Нецелевой звонок: автоответчик или голосовая почта, содержательного разговора не произошло.",
            "На записи только автоответчик/голосовая почта или телеком-система, клиент не вступал в содержательный диалог.",
        )
    if any(token in normalized for token in ("занят", "не отвечает", "не может ответить", "недоступен", "не берет трубку")):
        return (
            "Нецелевой звонок: технический недозвон без ответа клиента, содержательного разговора не произошло.",
            "На записи только сообщение о недозвоне/занятости/недоступности, клиент не вступал в содержательный диалог.",
        )
    return (
        "Нецелевой звонок: запись не содержит полноценного диалога менеджера с клиентом.",
        "Запись не подходит для анализа продаж, потому что содержательного разговора менеджера с клиентом нет.",
    )


def _empty_structured_fields(phone: str) -> dict[str, Any]:
    return {
        "people": {"parent_fio": None, "child_fio": None},
        "contacts": {"email": None, "phone_from_filename": phone or None, "preferred_channel": None},
        "student": {"grade_current": None, "school": None},
        "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
        "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
        "objections": [],
        "next_step": {"action": None, "due": None},
        "lead_priority": "cold",
    }


def _report_row(candidate: dict[str, Any], *, status: str, blockers: list[str]) -> dict[str, Any]:
    return {
        "id": _clean(candidate.get("id")),
        "source_filename": _clean(candidate.get("source_filename")),
        "current_call_type": _clean(candidate.get("current_call_type")),
        "current_contentful": _clean(candidate.get("current_contentful")),
        "analysis_status": _clean(candidate.get("analysis_status")),
        "resolve_status": _clean(candidate.get("resolve_status")),
        "guardrail_label": _clean(candidate.get("guardrail_label")),
        "guardrail_reason_codes": _clean(candidate.get("guardrail_reason_codes")),
        "review_decision": _clean(candidate.get("review_decision")),
        "review_hash": _clean(candidate.get("review_hash")),
        "consensus_route": _clean(candidate.get("consensus_route")),
        "final_source": _clean(candidate.get("final_source")),
        "final_decision": _clean(candidate.get("final_decision")),
        "status": status,
        "blockers": "|".join(blockers),
    }


def _sqlite_path_from_database_url(database_url: str) -> Path:
    if not database_url.startswith("sqlite"):
        raise ValueError("Only sqlite database_url is supported for this staged backfill")
    url = make_url(database_url)
    database = url.database
    if not database or database == ":memory:":
        raise ValueError("Backfill requires a file-backed SQLite database")
    return Path(database).expanduser().resolve()


def _safe_json_object(raw: Any) -> dict[str, Any]:
    cleaned = _clean(raw)
    if not cleaned:
        return {}
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split_codes(value: Any) -> list[str]:
    return [part for part in _clean(value).split("|") if part]


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(_clean(value)))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(_clean(value))
    except ValueError:
        return None


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _is_true(value: Any) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "да"}


def _redact_database_url(database_url: str) -> str:
    if database_url.startswith("sqlite"):
        return database_url
    return database_url.split("@")[-1] if "@" in database_url else database_url


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Transcript Quality Backfill Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Database: `{summary['database_path']}`",
        f"- Candidates CSV: `{summary['candidates_csv']}`",
        f"- Input candidates: `{summary['input_candidates']}`",
        f"- Planned updates: `{summary['planned_updates']}`",
        f"- Applied updates: `{summary['applied_updates']}`",
        f"- Blocked rows: `{summary['blocked_rows']}`",
        f"- Already applied: `{summary['already_applied']}`",
        f"- Missing rows: `{summary['missing_rows']}`",
        f"- Errors: `{summary['errors']}`",
        f"- Backup path: `{summary['backup_path']}`",
        "",
        "## Outputs",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Staged backfill for safe transcript quality non-conversation candidates.")
    db_group = parser.add_mutually_exclusive_group(required=True)
    db_group.add_argument("--database-url")
    db_group.add_argument("--db", type=Path)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["dry-run", "apply"], default="dry-run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-backup", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TranscriptQualityBackfillConfig:
    database_url = args.database_url or f"sqlite:///{args.db}"
    return TranscriptQualityBackfillConfig(
        database_url=database_url,
        candidates_csv=args.candidates_csv,
        out_root=args.out_root,
        mode=args.mode,
        limit=args.limit,
        create_backup=not bool(args.no_backup),
    )


__all__ = [
    "BACKFILL_VERSION",
    "TranscriptQualityBackfillConfig",
    "config_from_args",
    "parse_args",
    "run_transcript_quality_backfill",
]
