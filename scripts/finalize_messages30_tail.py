#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TAIL_CASES: dict[str, dict[str, Any]] = {
    "2025-09-13__14-12-22__Фадеев Ян__89527902178.mp3": {
        "summary": "Нецелевой звонок: автоответчик/голосовая почта, клиент не выходил на связь.",
        "reason": "На записи только короткая реплика менеджера и автоответчик клиента; содержательного диалога нет.",
        "mode": "stereo",
    },
    "2026-02-05__12-19-39__Леонов Алексей__79885053513.mp3": {
        "summary": "Нецелевой звонок: технический дозвон без ответа, система сообщает, что абонент не берет трубку.",
        "reason": "Запись содержит только голос телеком-системы о недозвоне; клиент не вступал в диалог.",
        "mode": "mono_or_fallback",
    },
    "2026-04-11__13-31-02__Головченко Карина__79262828455.mp3": {
        "summary": "Нецелевой звонок: технический сбой связи/удержание, содержательного разговора не произошло.",
        "reason": "На записи только 'алло', жалоба на отсутствие звука и технические артефакты удержания.",
        "mode": "stereo",
    },
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def finalize_tail(db_path: Path, *, create_backup: bool = True) -> dict[str, Any]:
    if create_backup:
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{db_path.stem}_tail_backup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{db_path.suffix}"
        shutil.copy2(db_path, backup_path)
    else:
        backup_path = None

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    updated: list[dict[str, Any]] = []
    now_sql = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        for filename, case in TAIL_CASES.items():
            row = con.execute("select * from call_records where source_filename=?", (filename,)).fetchone()
            if row is None:
                raise SystemExit(f"call not found: {filename}")
            analysis_json = _build_analysis_json(
                started_at=row["started_at"],
                manager_name=row["manager_name"],
                phone=row["phone"],
                summary=case["summary"],
                reason=case["reason"],
                mode=case["mode"],
            )
            resolve_json = _build_resolve_json(duration_sec=float(row["duration_sec"] or 0.0), reason=case["reason"])
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
                 where source_filename=?
                """,
                (resolve_json, analysis_json, now_sql, filename),
            )
            updated.append(
                {
                    "source_filename": filename,
                    "new_resolve_status": "skipped",
                    "new_analysis_status": "done",
                    "summary": case["summary"],
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
    parser = argparse.ArgumentParser(description="Finalize the remaining 3 messages(30) tail calls as non-conversation.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    result = finalize_tail(Path(args.db).expanduser().resolve(), create_backup=not args.no_backup)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
