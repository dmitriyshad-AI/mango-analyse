#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_summary(text: str) -> tuple[str, str]:
    normalized = (text or "").strip().lower()
    if any(needle in normalized for needle in ("автоответчик", "голосов", "почтовый ящик")):
        return (
            "Нецелевой звонок: недозвон с автоответчиком/голосовой почтой, содержательного разговора не произошло.",
            "На записи только автоответчик или голосовая почта, клиент не вступал в диалог.",
        )
    if any(needle in normalized for needle in ("занят", "не берет трубку", "не отвечает", "продолжаем дозваниваться", "не может ответить")):
        return (
            "Нецелевой звонок: технический недозвон без ответа клиента, содержательного разговора не произошло.",
            "На записи только реплики телеком-системы о недозвоне/занятости и короткие технические фразы менеджера.",
        )
    return (
        "Нецелевой звонок: технический контакт без содержательного разговора менеджера с клиентом.",
        "Запись не содержит полноценного диалога менеджера с клиентом и не подходит для анализа продаж.",
    )


def _build_resolve_json(*, duration_sec: float, reason: str) -> str:
    payload = {
        "version": "v1",
        "decision": "manual_skip_non_conversation",
        "duration_sec": round(float(duration_sec or 0.0), 3),
        "reasons": ["manual_tail_finalize", "non_conversation"],
        "note": reason,
        "ts_utc": _iso_now(),
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_analysis_json(*, started_at: str, manager_name: str, phone: str, summary: str, reason: str, mode: str) -> str:
    started_text = str(started_at or "").replace("T", " ").strip()
    manager = str(manager_name or "").strip()
    payload = {
        "analysis_schema_version": "v2",
        "history_summary": f"{started_text[:16]} менеджер {manager} общался с клиентом. {summary} Итог: Нет содержательного диалога менеджер-клиент для анализа продаж.",
        "structured_fields": {
            "people": {"parent_fio": None, "child_fio": None},
            "contacts": {"email": None, "phone_from_filename": phone, "preferred_channel": None},
            "student": {"grade_current": None, "school": None},
            "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
            "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
            "objections": [],
            "next_step": {"action": None, "due": None},
            "lead_priority": "cold",
        },
        "history_short": summary,
        "crm_blocks": {
            "people": {"parent_fio": None, "child_fio": None},
            "contacts": {"email": None, "phone_from_filename": phone, "preferred_channel": None},
            "student": {"grade_current": None, "school": None},
            "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
            "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
            "objections": [],
            "next_step": {"action": None, "due": None},
            "lead_priority": "cold",
        },
        "evidence": [],
        "quality_flags": {
            "mode": mode,
            "mono_fallback": mode == "mono_or_fallback",
            "secondary_provider": "gigaam",
            "warnings_count": 0,
            "has_secondary_empty_warning": False,
            "call_type": "non_conversation",
            "needs_review": False,
            "review_reasons": [],
        },
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
    return json.dumps(payload, ensure_ascii=False)


def finalize_tail(
    db_path: Path,
    *,
    create_backup: bool = True,
    source_filenames: set[str] | None = None,
) -> dict[str, Any]:
    if create_backup:
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{db_path.stem}_manual_tail_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{db_path.suffix}"
        shutil.copy2(db_path, backup_path)
    else:
        backup_path = None

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    now_sql = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    updated: list[dict[str, Any]] = []
    try:
        where = "resolve_status='manual' and analysis_status='pending'"
        params: list[Any] = []
        if source_filenames:
            placeholders = ",".join("?" for _ in source_filenames)
            where += f" and source_filename in ({placeholders})"
            params.extend(sorted(source_filenames))
        rows = con.execute(
            f"""
            select id, source_filename, phone, manager_name, started_at, duration_sec,
                   transcript_manager, transcript_client, transcript_text, resolve_json
              from call_records
             where {where}
             order by started_at asc, id asc
            """,
            params,
        ).fetchall()
        for row in rows:
            transcript_text = str(row["transcript_text"] or "").strip()
            manager_text = str(row["transcript_manager"] or "").strip()
            client_text = str(row["transcript_client"] or "").strip()
            combined = "\n".join(part for part in [manager_text, client_text, transcript_text] if part)
            summary, reason = _classify_summary(combined)

            resolve_raw = str(row["resolve_json"] or "").strip()
            mode = "mono_or_fallback"
            if resolve_raw:
                try:
                    payload = json.loads(resolve_raw)
                except json.JSONDecodeError:
                    payload = {}
                meta_mode = (
                    (((payload.get("llm") or {}).get("meta") or {}).get("mode"))
                    or (((payload.get("rescue") or {}).get("meta") or {}).get("mode"))
                )
                if str(meta_mode or "").strip():
                    mode = str(meta_mode).strip()
                elif manager_text and client_text:
                    mode = "stereo"
            elif manager_text and client_text:
                mode = "stereo"

            analysis_json = _build_analysis_json(
                started_at=row["started_at"],
                manager_name=row["manager_name"],
                phone=row["phone"],
                summary=summary,
                reason=reason,
                mode=mode,
            )
            resolve_json = _build_resolve_json(
                duration_sec=float(row["duration_sec"] or 0.0),
                reason=reason,
            )
            con.execute(
                """
                update call_records
                   set resolve_status='skipped',
                       analysis_status='done',
                       resolve_attempts=case when resolve_attempts < 1 then 1 else resolve_attempts end,
                       analyze_attempts=case when analyze_attempts < 1 then 1 else analyze_attempts end,
                       dead_letter_stage=null,
                       last_error=null,
                       resolve_json=?,
                       resolve_quality_score=100.0,
                       analysis_json=?,
                       updated_at=?
                 where id=?
                """,
                (resolve_json, analysis_json, now_sql, int(row["id"])),
            )
            updated.append(
                {
                    "id": int(row["id"]),
                    "source_filename": str(row["source_filename"]),
                    "new_resolve_status": "skipped",
                    "new_analysis_status": "done",
                    "summary": summary,
                }
            )
        con.commit()
    finally:
        con.close()

    return {
        "db_path": str(db_path),
        "backup_path": str(backup_path) if backup_path else None,
        "updated": updated,
        "updated_count": len(updated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize manual non-conversation tail calls into terminal skipped/done state.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--report", required=False)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument(
        "--source-filename",
        action="append",
        default=[],
        help="Limit finalization to this source filename. May be repeated.",
    )
    args = parser.parse_args()
    result = finalize_tail(
        Path(args.db).expanduser().resolve(),
        create_backup=not args.no_backup,
        source_filenames=set(args.source_filename) if args.source_filename else None,
    )
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
