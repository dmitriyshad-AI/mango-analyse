from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICY_VERSION = "hard_gate_gpt_policy_v1"
SAFE_DECISION = "safe_apply"
AUTO_REVIEW_DECISION = "hard_gate_gpt_auto_apply"
BLOCKED_REVIEW_DECISION = "blocked_by_hard_gate_gpt_policy"
DEFAULT_TRANSCRIPT_CHAR_LIMIT = 12000


@dataclass(frozen=True)
class HardGateGptApplyPlanConfig:
    candidates_csv: Path
    out_root: Path
    project_root: Path = Path(".")
    gpt_decisions_jsonl: Path | None = None
    transcript_char_limit: int = DEFAULT_TRANSCRIPT_CHAR_LIMIT
    include_review_tasks: bool = True


def build_hard_gate_gpt_apply_plan(config: HardGateGptApplyPlanConfig) -> dict[str, Any]:
    project_root = config.project_root.expanduser().resolve()
    candidates_path = _resolve_path(config.candidates_csv, project_root)
    out_root = _resolve_path(config.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if not candidates_path.exists():
        raise FileNotFoundError(f"candidates_csv not found: {candidates_path}")

    candidates = _read_csv(candidates_path)
    decisions = _read_decisions(config.gpt_decisions_jsonl, project_root=project_root)

    plan_rows: list[dict[str, Any]] = []
    auto_apply_rows: list[dict[str, Any]] = []
    gpt_review_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    keep_current_rows: list[dict[str, Any]] = []
    manual_review_rows: list[dict[str, Any]] = []
    review_tasks: list[dict[str, Any]] = []
    queue_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()

    for index, candidate in enumerate(candidates, start=1):
        audit_id = f"hgate_full_{index:06d}"
        task_id = _task_id(candidate)
        decision = decisions.get(task_id) or decisions.get(_candidate_key(candidate)) or {}
        gpt_decision = _normalize_decision(decision)
        deterministic_blockers = _deterministic_blockers(candidate)
        risk_level = _risk_level(candidate)
        queue = _queue(candidate, gpt_decision=gpt_decision, deterministic_blockers=deterministic_blockers)
        plan_row = _plan_row(
            candidate,
            audit_id=audit_id,
            task_id=task_id,
            queue=queue,
            risk_level=risk_level,
            gpt_decision=gpt_decision,
            decision=decision,
            deterministic_blockers=deterministic_blockers,
        )
        plan_rows.append(plan_row)
        queue_counts[queue] += 1
        risk_counts[risk_level] += 1

        if queue == "auto_apply_ready":
            auto_apply_rows.append(plan_row)
        elif queue == "gpt_review_required":
            gpt_review_rows.append(plan_row)
            if config.include_review_tasks:
                review_tasks.append(
                    _review_task(
                        candidate,
                        audit_id=audit_id,
                        task_id=task_id,
                        project_root=project_root,
                        transcript_char_limit=max(1000, int(config.transcript_char_limit)),
                    )
                )
        elif queue == "keep_current":
            keep_current_rows.append(plan_row)
        elif queue == "manual_review":
            manual_review_rows.append(plan_row)
        else:
            blocked_rows.append(plan_row)

    outputs = {
        "summary_json": out_root / "summary.json",
        "report_markdown": out_root / "PHASE6_HARD_GATE_GPT_APPLY_PLAN.md",
        "full_apply_plan_csv": out_root / "full_apply_plan.csv",
        "auto_apply_ready_csv": out_root / "auto_apply_ready.csv",
        "gpt_review_required_csv": out_root / "gpt_review_required.csv",
        "blocked_candidates_csv": out_root / "blocked_candidates.csv",
        "keep_current_csv": out_root / "keep_current.csv",
        "manual_review_csv": out_root / "manual_review.csv",
        "queue_summary_csv": out_root / "queue_summary.csv",
        "month_summary_csv": out_root / "month_summary.csv",
        "risk_summary_csv": out_root / "risk_summary.csv",
        "gpt_review_prompt_md": out_root / "GPT_REVIEW_PROMPT_RU.md",
    }
    if config.include_review_tasks:
        outputs["gpt_review_tasks_jsonl"] = out_root / "gpt_review_tasks.jsonl"
        outputs["gpt_decisions_template_jsonl"] = out_root / "gpt_decisions_template.jsonl"

    _write_csv(outputs["full_apply_plan_csv"], plan_rows)
    _write_csv(outputs["auto_apply_ready_csv"], auto_apply_rows)
    _write_csv(outputs["gpt_review_required_csv"], gpt_review_rows)
    _write_csv(outputs["blocked_candidates_csv"], blocked_rows)
    _write_csv(outputs["keep_current_csv"], keep_current_rows)
    _write_csv(outputs["manual_review_csv"], manual_review_rows)
    _write_csv(outputs["queue_summary_csv"], _counter_rows(queue_counts, "queue"))
    _write_csv(outputs["month_summary_csv"], _month_summary(plan_rows))
    _write_csv(outputs["risk_summary_csv"], _counter_rows(risk_counts, "risk_level"))
    outputs["gpt_review_prompt_md"].write_text(_gpt_review_prompt(), encoding="utf-8")
    if config.include_review_tasks:
        _write_jsonl(outputs["gpt_review_tasks_jsonl"], review_tasks)
        _write_jsonl(outputs["gpt_decisions_template_jsonl"], [_decision_template(task) for task in review_tasks])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "gpt_only_apply_plan_read_only",
        "policy_version": POLICY_VERSION,
        "candidates_csv": str(candidates_path),
        "gpt_decisions_jsonl": str(_resolve_path(config.gpt_decisions_jsonl, project_root)) if config.gpt_decisions_jsonl else None,
        "project_root": str(project_root),
        "input_candidates": len(candidates),
        "decisions_loaded": len(decisions),
        "queue_counts": dict(queue_counts.most_common()),
        "risk_counts": dict(risk_counts.most_common()),
        "auto_apply_ready": len(auto_apply_rows),
        "gpt_review_required": len(gpt_review_rows),
        "blocked_candidates": len(blocked_rows),
        "keep_current": len(keep_current_rows),
        "manual_review": len(manual_review_rows),
        "review_tasks": len(review_tasks),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["report_markdown"].write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _deterministic_blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    required = ("db", "id", "source_filename")
    for key in required:
        if not _clean(row.get(key)):
            blockers.append(f"missing_{key}")
    if _clean(row.get("status")) != "would_update":
        blockers.append("status_not_would_update")
    if _clean(row.get("normalized_call_type")) != "non_conversation":
        blockers.append("normalized_call_type_not_non_conversation")
    if _clean(row.get("guardrail_label")) != "non_conversation_high_confidence":
        blockers.append("guardrail_label_not_high_confidence")
    if not _is_true(row.get("guardrail_should_force_non_conversation")):
        blockers.append("guardrail_should_force_false")
    if _is_true(row.get("guardrail_requires_manual_review")):
        blockers.append("guardrail_requires_manual_review")
    if _is_true(row.get("guardrail_protected_live_dialogue")):
        blockers.append("guardrail_protected_live_dialogue")
    if not _is_true(row.get("hard_validation_applied")):
        blockers.append("hard_validation_not_applied")
    update_reasons = set(_split_codes(row.get("update_reasons")))
    if "call_type_to_non_conversation" not in update_reasons:
        blockers.append("missing_call_type_to_non_conversation_reason")
    if "clear_sales_fields" not in update_reasons:
        blockers.append("missing_clear_sales_fields_reason")
    return blockers


def _queue(row: dict[str, Any], *, gpt_decision: str, deterministic_blockers: list[str]) -> str:
    if deterministic_blockers:
        return "blocked_deterministic"
    if not gpt_decision:
        return "gpt_review_required"
    if gpt_decision == "safe_apply":
        return "auto_apply_ready"
    if gpt_decision == "keep_current":
        return "keep_current"
    if gpt_decision == "manual_review":
        return "manual_review"
    return "blocked_gpt_decision"


def _plan_row(
    row: dict[str, Any],
    *,
    audit_id: str,
    task_id: str,
    queue: str,
    risk_level: str,
    gpt_decision: str,
    decision: dict[str, Any],
    deterministic_blockers: list[str],
) -> dict[str, Any]:
    allowed = queue == "auto_apply_ready"
    review_hash = _review_hash(row, audit_id=audit_id, gpt_decision=gpt_decision)
    return {
        "audit_id": audit_id,
        "task_id": task_id,
        "db": _clean(row.get("db")),
        "id": _clean(row.get("id")),
        "source_filename": _clean(row.get("source_filename")),
        "started_at": _clean(row.get("started_at")),
        "month": _clean(row.get("month")),
        "phone": _clean(row.get("phone")),
        "manager_name": _clean(row.get("manager_name")),
        "duration_sec": _clean(row.get("duration_sec")),
        "current_call_type": _clean(row.get("current_call_type")),
        "normalized_call_type": _clean(row.get("normalized_call_type")),
        "guardrail_label": _clean(row.get("guardrail_label")),
        "guardrail_score": _clean(row.get("guardrail_score")),
        "guardrail_reason_codes": _clean(row.get("guardrail_reason_codes")),
        "should_force_non_conversation": _clean(row.get("guardrail_should_force_non_conversation")),
        "recommended_contact_subtype": _clean(row.get("guardrail_recommended_contact_subtype")),
        "current_follow_up_score": _clean(row.get("current_follow_up_score")),
        "current_next_step": _clean(row.get("current_next_step")),
        "current_products": _clean(row.get("current_products")),
        "current_subjects": _clean(row.get("current_subjects")),
        "current_objections": _clean(row.get("current_objections")),
        "risk_level": risk_level,
        "queue": queue,
        "policy_mode": "gpt-only",
        "policy_version": POLICY_VERSION,
        "policy_queue": "gpt_auto_apply" if allowed else queue,
        "policy_auto_apply_allowed": allowed,
        "gpt_decision": gpt_decision,
        "gpt_confidence": _clean(decision.get("confidence")),
        "gpt_reason": _clean(decision.get("reason_ru") or decision.get("reason")),
        "review_decision": AUTO_REVIEW_DECISION if allowed else BLOCKED_REVIEW_DECISION,
        "review_hash": review_hash,
        "deterministic_blockers": "|".join(deterministic_blockers),
        "requires_gpt_review": queue == "gpt_review_required",
        "recommended_action": _recommended_action(queue),
    }


def _risk_level(row: dict[str, Any]) -> str:
    current_type = _clean(row.get("current_call_type"))
    subtype = _clean(row.get("guardrail_recommended_contact_subtype"))
    duration = _safe_float(row.get("duration_sec")) or 0.0
    has_sales_fields = any(
        _clean(row.get(key))
        for key in ("current_next_step", "current_products", "current_subjects", "current_objections")
    )
    follow_up_score = _safe_int(row.get("current_follow_up_score")) or 0
    if current_type in {"sales_call", "existing_client_progress"}:
        return "critical"
    if duration >= 75 or has_sales_fields or follow_up_score >= 80:
        return "high"
    if subtype == "outbound_voicemail":
        return "medium"
    return "medium" if current_type == "service_call" else "low"


def _review_task(
    row: dict[str, Any],
    *,
    audit_id: str,
    task_id: str,
    project_root: Path,
    transcript_char_limit: int,
) -> dict[str, Any]:
    transcript = _fetch_transcript(row, project_root=project_root, limit=transcript_char_limit)
    return {
        "audit_id": audit_id,
        "task_id": task_id,
        "policy_version": POLICY_VERSION,
        "audit_goal": "Проверить, можно ли безопасно заменить старый анализ на non_conversation и очистить продажные поля.",
        "call": {
            "db": _clean(row.get("db")),
            "call_record_id": _clean(row.get("id")),
            "source_filename": _clean(row.get("source_filename")),
            "started_at": _clean(row.get("started_at")),
            "month": _clean(row.get("month")),
            "phone": _clean(row.get("phone")),
            "manager_name": _clean(row.get("manager_name")),
            "duration_sec": _clean(row.get("duration_sec")),
        },
        "current_state": {
            "call_type": _clean(row.get("current_call_type")),
            "follow_up_score": _clean(row.get("current_follow_up_score")),
            "next_step": _clean(row.get("current_next_step")),
            "products": _split_codes(row.get("current_products")),
            "subjects": _split_codes(row.get("current_subjects")),
            "objections": _split_codes(row.get("current_objections")),
            "history_summary_excerpt": _clean(row.get("current_history_summary_excerpt")),
        },
        "proposed_change": {
            "new_call_type": "non_conversation",
            "new_follow_up_score": "0",
            "new_next_step": "",
            "new_products": [],
            "new_subjects": [],
            "new_objections": [],
            "new_history_summary_excerpt": _clean(row.get("normalized_history_summary_excerpt")),
        },
        "guardrail": {
            "label": _clean(row.get("guardrail_label")),
            "score": _clean(row.get("guardrail_score")),
            "reason_codes": _split_codes(row.get("guardrail_reason_codes")),
            "recommended_contact_subtype": _clean(row.get("guardrail_recommended_contact_subtype")),
        },
        "transcript": transcript,
        "auditor_required_output": _decision_template({"audit_id": audit_id, "task_id": task_id}),
    }


def _fetch_transcript(row: dict[str, Any], *, project_root: Path, limit: int) -> dict[str, Any]:
    db_value = _clean(row.get("db"))
    call_id = _clean(row.get("id"))
    source_filename = _clean(row.get("source_filename"))
    fallback = _clean(row.get("transcript_excerpt"))
    if not db_value or not call_id:
        return {"full_text": _truncate(fallback, limit), "source": "candidate_excerpt", "fetch_error": "missing_db_or_id"}
    db_path = _resolve_path(Path(db_value), project_root)
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            db_row = con.execute(
                """
                select transcript_manager, transcript_client, transcript_text
                  from call_records
                 where id = ?
                   and source_filename = ?
                """,
                (int(float(call_id)), source_filename),
            ).fetchone()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        return {"full_text": _truncate(fallback, limit), "source": "candidate_excerpt", "fetch_error": str(exc)}
    if db_row is None:
        return {"full_text": _truncate(fallback, limit), "source": "candidate_excerpt", "fetch_error": "not_found_in_db"}
    parts = [
        _clean(db_row["transcript_manager"]),
        _clean(db_row["transcript_client"]),
        _clean(db_row["transcript_text"]),
    ]
    text = "\n\n".join(part for part in parts if part) or fallback
    return {
        "full_text": _truncate(text, limit),
        "source": "call_records",
        "truncated": len(text) > limit,
    }


def _read_decisions(path: Path | None, *, project_root: Path) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = _resolve_path(path, project_root)
    if not resolved.exists():
        raise FileNotFoundError(f"gpt_decisions_jsonl not found: {resolved}")
    decisions: dict[str, dict[str, Any]] = {}
    with resolved.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{resolved}:{line_no}: invalid JSON: {exc}") from exc
            task_id = _clean(payload.get("task_id"))
            if task_id:
                decisions[task_id] = payload
            key = _clean(payload.get("candidate_key"))
            if key:
                decisions[key] = payload
    return decisions


def _normalize_decision(decision: dict[str, Any]) -> str:
    raw = _clean(decision.get("decision") or decision.get("gpt_decision"))
    if raw in {"safe_apply", "force_non_conversation"}:
        return "safe_apply"
    if raw in {"keep_current", "keep_current_analysis"}:
        return "keep_current"
    if raw in {"manual_review", "human_review_required"}:
        return "manual_review"
    if raw == "reanalyze_required":
        return "reanalyze_required"
    return ""


def _task_id(row: dict[str, Any]) -> str:
    return f"hard_gate_gpt::{_clean(row.get('db'))}::{_clean(row.get('id'))}"


def _candidate_key(row: dict[str, Any]) -> str:
    return f"{_clean(row.get('db'))}::{_clean(row.get('id'))}::{_clean(row.get('source_filename'))}"


def _review_hash(row: dict[str, Any], *, audit_id: str, gpt_decision: str) -> str:
    payload = "|".join(
        [
            POLICY_VERSION,
            audit_id,
            _candidate_key(row),
            _clean(row.get("guardrail_reason_codes")),
            gpt_decision,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _decision_template(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_id": _clean(task.get("audit_id")),
        "task_id": _clean(task.get("task_id")),
        "decision": "safe_apply | keep_current | manual_review",
        "confidence": 0.0,
        "reason_ru": "",
        "evidence": [],
    }


def _recommended_action(queue: str) -> str:
    if queue == "auto_apply_ready":
        return "can_be_used_as_backfill_input_after_backup"
    if queue == "gpt_review_required":
        return "run_gpt_review_before_apply"
    if queue == "keep_current":
        return "do_not_apply_keep_existing_analysis"
    if queue == "manual_review":
        return "manual_or_external_audit_before_apply"
    return "blocked_do_not_apply"


def _month_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counters: dict[str, Counter[str]] = {}
    for row in rows:
        month = _clean(row.get("month")) or "unknown"
        counter = counters.setdefault(month, Counter())
        counter["total"] += 1
        counter[f"queue:{row.get('queue')}"] += 1
        counter[f"risk:{row.get('risk_level')}"] += 1
    keys = sorted({key for counter in counters.values() for key in counter})
    result: list[dict[str, Any]] = []
    for month in sorted(counters):
        row: dict[str, Any] = {"month": month}
        for key in keys:
            row[key] = counters[month].get(key, 0)
        result.append(row)
    return result


def _counter_rows(counter: Counter[str], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, "count": value} for key, value in counter.most_common()]


def _gpt_review_prompt() -> str:
    return """# GPT Review Prompt: hard_gate_gpt_policy_v1

Задача: проверить каждый `gpt_review_tasks.jsonl` item и вернуть JSONL с решениями.

Нужно решить, безопасно ли заменить старый анализ звонка на `non_conversation` и очистить продажные поля.

Разрешай `safe_apply`, если:
- в записи нет живого содержательного диалога менеджер-клиент;
- это автоответчик, voicemail, IVR, голосовой ассистент, виртуальный секретарь, недозвон, занято, номер недоступен;
- менеджер оставил даже содержательное сообщение на автоответчике, но клиент не участвовал в диалоге;
- запись содержит ASR-мусор без реального клиентского разговора.

Ставь `keep_current`, если:
- есть живой клиент/родитель/ученик/представитель клиента;
- обсуждаются курсы, обучение, расписание, цена, рассрочка, возражения, следующий шаг;
- звонок начался/закончился IVR или voicemail, но внутри есть содержательный живой разговор;
- менеджер переводит звонок, клиент соглашается, и до voicemail был смысловой контакт.

Ставь `manual_review`, если:
- не хватает данных;
- ASR противоречивый;
- есть признаки живого разговора, но уверенности недостаточно.

Формат ответа на каждую строку:

```json
{"audit_id":"...","task_id":"...","decision":"safe_apply","confidence":0.98,"reason_ru":"...","evidence":["..."]}
```

Допустимые `decision`: `safe_apply`, `keep_current`, `manual_review`.
"""


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 6 Hard Gate GPT Apply Plan",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Policy: `{summary['policy_version']}`",
        f"- Input candidates: `{summary['input_candidates']}`",
        f"- Decisions loaded: `{summary['decisions_loaded']}`",
        f"- Auto-apply ready: `{summary['auto_apply_ready']}`",
        f"- GPT review required: `{summary['gpt_review_required']}`",
        f"- Blocked candidates: `{summary['blocked_candidates']}`",
        f"- Keep current: `{summary['keep_current']}`",
        f"- Manual review: `{summary['manual_review']}`",
        "",
        "## Queue Counts",
    ]
    for key, value in summary["queue_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Risk Counts"])
    for key, value in summary["risk_counts"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Outputs"])
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Run GPT review on `gpt_review_tasks.jsonl`, then rebuild this plan with `--gpt-decisions-jsonl`.",
            "Only `auto_apply_ready.csv` is suitable as input for staged backfill, and only after backup/rollback manifest.",
        ]
    )
    return "\n".join(lines) + "\n"


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _split_codes(value: Any) -> list[str]:
    cleaned = _clean(value)
    if not cleaned:
        return []
    return [part.strip() for part in cleaned.split("|") if part.strip()]


def _truncate(value: str, limit: int) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 18)].rstrip() + " [truncated]"


def _resolve_path(path: Path | None, project_root: Path) -> Path:
    if path is None:
        raise ValueError("path is required")
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _is_true(value: Any) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "да"}


def _safe_float(value: Any) -> float | None:
    try:
        return float(_clean(value))
    except ValueError:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(_clean(value)))
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-corpus GPT-only hard-gate apply plan.")
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--gpt-decisions-jsonl")
    parser.add_argument("--transcript-char-limit", type=int, default=DEFAULT_TRANSCRIPT_CHAR_LIMIT)
    parser.add_argument("--no-review-tasks", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> HardGateGptApplyPlanConfig:
    return HardGateGptApplyPlanConfig(
        candidates_csv=Path(args.candidates_csv),
        out_root=Path(args.out_root),
        project_root=Path(args.project_root),
        gpt_decisions_jsonl=Path(args.gpt_decisions_jsonl) if args.gpt_decisions_jsonl else None,
        transcript_char_limit=args.transcript_char_limit,
        include_review_tasks=not bool(args.no_review_tasks),
    )


__all__ = [
    "HardGateGptApplyPlanConfig",
    "build_hard_gate_gpt_apply_plan",
    "config_from_args",
    "parse_args",
]
