from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.channels.telegram_pilot_metrics import build_daily_metrics
from mango_mvp.channels.telegram_pilot_store import (
    PILOT_FEEDBACK_MANAGER_ONLY,
    PILOT_FEEDBACK_NEEDS_EDIT,
    PILOT_FEEDBACK_TOPIC_WRONG,
    PILOT_FEEDBACK_UNSAFE_FACT_ATTEMPT,
    PILOT_FEEDBACK_USEFUL,
    TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
    TelegramPilotSQLiteStore,
    guard_telegram_pilot_path,
)


TELEGRAM_PILOT_DAILY_REPORT_SCHEMA_VERSION = "telegram_pilot_daily_report_v2_2026_05_23"

PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8)?[\s(.-]*\d{3}[\s).-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
TOKEN_RE = re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{25,}\b")
INTERNAL_MARKER_RE = re.compile(r"\b(?:AMO|Tallanto|CRM|source_id|lead_id|contact_id|token|api[_-]?key)\b", re.I)
PRECISE_FACT_RE = re.compile(
    r"\b\d[\d\s\u00a0]{1,9}\s*(?:₽|руб|р\.)|\b\d{1,3}\s*%|\b(?:до|по|с)\s+\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)",
    re.I,
)
P0_RE = re.compile(r"возврат|верн\w+\s+деньг|жалоб|суд|иск|претензи|прокуратур|роспотребнадзор", re.I)
TEMPLATE_RE = re.compile(
    r"спасибо за обращение|ваш вопрос очень важен|оптимальн\w+\s+образовательн\w+\s+продукт|менеджер свяжется",
    re.I,
)
GENERIC_RE = re.compile(r"уточн(?:ю|им)|передам|свяжется|подбер[её]м|проверит", re.I)

P0_TOPICS = {"theme:009_refund", "theme:019b_negative_feedback", "theme:029_legal_question"}
AUTONOMOUS_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot"}
HIGH_RISK_FLAGS = {"high_risk_manager_only", "combined_high_risk_manager_only", "autonomy_blocked_high_risk"}

REPORT_FIELDS = (
    "review_id",
    "draft_id",
    "message_key",
    "timestamp",
    "brand",
    "chat_id",
    "input_text",
    "answer_text",
    "route",
    "topic_id",
    "message_type",
    "risk_level",
    "safety_flags",
    "post_filter_flags",
    "semantic_flags",
    "active_brand",
    "knowledge_base_version",
    "prompt_version",
    "model",
    "reasoning",
    "latency_seconds",
    "sent_to_client",
    "known_client_fields",
    "known_dialog_fields",
    "known_slots",
    "missing_slots",
    "next_best_question",
    "next_step_type",
    "asked_known_data_again",
    "asked_again_fields",
    "facts_used",
    "facts_missing",
    "manager_summary",
    "employee_feedback",
    "review_reasons",
    "human_verdict",
    "human_comment",
    "corrected_answer",
)


@dataclass(frozen=True)
class FeedbackImportSummary:
    imported: int
    skipped: int
    errors: tuple[str, ...] = ()

    def to_json_dict(self) -> Mapping[str, Any]:
        return {"imported": self.imported, "skipped": self.skipped, "errors": list(self.errors)}


def build_pilot_daily_report(
    store_or_db_path: TelegramPilotSQLiteStore | Path | str,
    day: date | str,
    *,
    out_dir: Path | str,
    p0_register_path: Path | str | None = None,
) -> Mapping[str, Any]:
    target = Path(out_dir)
    guard_telegram_pilot_path(target)
    target.mkdir(parents=True, exist_ok=True)

    close_store = False
    if isinstance(store_or_db_path, TelegramPilotSQLiteStore):
        store = store_or_db_path
    else:
        store = TelegramPilotSQLiteStore.open_read_only(store_or_db_path)
        close_store = True
    try:
        rows = build_pilot_message_rows(store, day)
        metrics = build_daily_metrics(store, day)
    finally:
        if close_store:
            store.close()

    review_rows = [row for row in rows if str(row.get("review_reasons") or "").strip()]
    regression_rows = [row for row in review_rows if is_regression_candidate(row)]
    p0_rows = build_p0_rows(rows, p0_register_path=p0_register_path)
    reask_rows = [row for row in rows if str(row.get("asked_known_data_again") or "") == "true"]
    template_rows = [row for row in rows if "template_or_generic" in str(row.get("review_reasons") or "")]
    facts_summary = build_facts_used_summary(rows)

    write_csv(target / "pilot_messages.csv", rows, REPORT_FIELDS)
    write_jsonl(target / "pilot_messages.jsonl", rows)
    write_csv(target / "semantic_review_queue.csv", review_rows, REPORT_FIELDS)
    write_csv(target / "regression_candidates.csv", regression_rows, REPORT_FIELDS)
    write_csv(target / "employee_review_sheet.csv", rows, REPORT_FIELDS)
    write_csv(target / "p0_incidents.csv", p0_rows, REPORT_FIELDS)
    write_csv(target / "known_data_reask_cases.csv", reask_rows, REPORT_FIELDS)
    write_csv(target / "template_or_generic_cases.csv", template_rows, REPORT_FIELDS)
    write_csv(target / "facts_used_summary.csv", facts_summary, ("fact", "count", "brands", "routes"))

    summary = build_pilot_summary(rows, metrics=metrics.to_json_dict(), review_rows=review_rows, regression_rows=regression_rows)
    (target / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (target / "pilot_summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    (target / "implementation_notes.md").write_text(render_implementation_notes(summary), encoding="utf-8")
    (target / "semantic_review.md").write_text(render_semantic_review(summary), encoding="utf-8")

    return {
        "schema_version": TELEGRAM_PILOT_DAILY_REPORT_SCHEMA_VERSION,
        "out_dir": str(target),
        "summary": summary,
        "files": sorted(path.name for path in target.iterdir() if path.is_file()),
    }


def build_pilot_message_rows(store: TelegramPilotSQLiteStore, day: date | str) -> list[dict[str, Any]]:
    messages = {str(item.get("message_key") or ""): dict(item) for item in store.list_messages(day=day)}
    contexts = {str(item.get("message_key") or ""): dict(item) for item in store.list_contexts(day=day)}
    feedback = feedback_by_draft(store.list_feedback_events(day=day))
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for draft in store.list_drafts(day=day):
        row = build_report_row(draft, messages=messages, contexts=contexts, feedback=feedback)
        counters[row["brand"]] += 1
        row["review_id"] = f"{row['brand'] or 'unknown'}-{counters[row['brand']]:04d}"
        rows.append(row)
    return rows


def build_report_row(
    draft: Mapping[str, Any],
    *,
    messages: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    feedback: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    message_key = str(draft.get("message_key") or "")
    message = dict(messages.get(message_key) or {})
    context_record = dict(contexts.get(message_key) or {})
    context = _mapping(context_record.get("context"))
    metadata = _mapping(draft.get("metadata"))
    llm_result = _mapping(metadata.get("llm_result"))
    funnel = _mapping(metadata.get("funnel_state") or context.get("funnel_state"))
    known_client = _mapping(context.get("known_client_fields"))
    known_dialog = _mapping(context.get("known_dialog_fields"))
    safety_flags = _list(draft.get("safety_flags") or llm_result.get("safety_flags"))
    semantic_flags = _list(metadata.get("semantic_flags") or funnel.get("semantic_flags") or context.get("semantic_flags"))
    asked_again_fields = _list(metadata.get("asked_known_data_again_fields"))
    if not asked_again_fields:
        asked_again_fields = detect_asked_known_data_again(
            str(draft.get("draft_text") or ""),
            known_client=known_client,
            known_dialog=known_dialog,
            known_slots=_mapping(metadata.get("known_slots") or funnel.get("filled_slots")),
        )
    route = str(draft.get("route") or llm_result.get("route") or "")
    topic_id = str(draft.get("topic_id") or llm_result.get("topic_id") or "")
    answer = str(draft.get("draft_text") or "")
    input_text = str(message.get("text") or "")
    message_meta = _mapping(message.get("metadata"))
    brand = str(metadata.get("brand") or context.get("active_brand") or message_meta.get("brand") or "")
    employee_feedback = list(feedback.get(str(draft.get("draft_id") or ""), ()))
    row = {
        "review_id": "",
        "draft_id": str(draft.get("draft_id") or ""),
        "message_key": message_key,
        "timestamp": str(draft.get("created_at") or message.get("received_at") or ""),
        "brand": brand,
        "chat_id": mask_text(str(message.get("channel_thread_id") or "")),
        "input_text": mask_text(input_text),
        "answer_text": mask_text(answer),
        "route": route,
        "topic_id": topic_id,
        "message_type": str(llm_result.get("message_type") or ""),
        "risk_level": str(llm_result.get("risk_level") or ""),
        "safety_flags": json_dump_masked(safety_flags),
        "post_filter_flags": json_dump_masked(llm_result.get("post_filter_flags") or metadata.get("post_filter_flags") or []),
        "semantic_flags": json_dump_masked(semantic_flags),
        "active_brand": str(context.get("active_brand") or brand),
        "knowledge_base_version": str(draft.get("knowledge_base_version") or ""),
        "prompt_version": str(draft.get("prompt_version") or ""),
        "model": model_from_prompt_version(str(draft.get("prompt_version") or "")),
        "reasoning": reasoning_from_prompt_version(str(draft.get("prompt_version") or "")),
        "latency_seconds": str(metadata.get("latency_seconds") or ""),
        "sent_to_client": "true" if bool(metadata.get("client_send_executed")) else "false",
        "known_client_fields": json_dump_masked(known_client),
        "known_dialog_fields": json_dump_masked(known_dialog),
        "known_slots": json_dump_masked(metadata.get("known_slots") or funnel.get("filled_slots") or context.get("known_slots") or {}),
        "missing_slots": json_dump_masked(metadata.get("missing_slots") or funnel.get("missing_slots") or context.get("missing_slots") or []),
        "next_best_question": mask_text(str(funnel.get("next_best_question") or context.get("next_best_question") or "")),
        "next_step_type": str(metadata.get("next_step_type") or funnel.get("next_step_type") or context.get("next_step_type") or ""),
        "asked_known_data_again": "true" if asked_again_fields or bool(metadata.get("asked_known_data_again")) else "false",
        "asked_again_fields": json_dump_masked(asked_again_fields),
        "facts_used": json_dump_masked(facts_used_from_context_and_result(context, llm_result)),
        "facts_missing": json_dump_masked(llm_result.get("missing_facts") or context.get("missing_facts") or []),
        "manager_summary": mask_text(str(metadata.get("manager_summary") or "")),
        "employee_feedback": json_dump_masked(employee_feedback),
        "review_reasons": "",
        "human_verdict": "",
        "human_comment": "",
        "corrected_answer": "",
    }
    row["review_reasons"] = "; ".join(review_reasons_for_row(row, raw_answer=answer, raw_input=input_text))
    return row


def feedback_by_draft(events: Sequence[Mapping[str, Any]]) -> Mapping[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        draft_id = str(event.get("draft_id") or "")
        if not draft_id:
            continue
        result.setdefault(draft_id, []).append(
            {
                "event_type": str(event.get("event_type") or ""),
                "actor": mask_text(str(event.get("actor") or "")),
                "reason": mask_text(str(event.get("reason") or "")),
                "occurred_at": str(event.get("occurred_at") or ""),
            }
        )
    return result


def import_employee_feedback_csv(
    store_or_db_path: TelegramPilotSQLiteStore | Path | str,
    csv_path: Path | str,
    *,
    actor: str = "employee_review_sheet",
) -> FeedbackImportSummary:
    close_store = False
    if isinstance(store_or_db_path, TelegramPilotSQLiteStore):
        store = store_or_db_path
    else:
        store = TelegramPilotSQLiteStore(store_or_db_path)
        close_store = True
    imported = 0
    skipped = 0
    errors: list[str] = []
    try:
        with Path(csv_path).open("r", encoding="utf-8", newline="") as file:
            for line_no, row in enumerate(csv.DictReader(file), start=2):
                draft_id = str(row.get("draft_id") or "").strip()
                verdict = str(row.get("human_verdict") or "").strip()
                comment = str(row.get("human_comment") or "").strip()
                corrected = str(row.get("corrected_answer") or "").strip()
                if not draft_id or not verdict:
                    skipped += 1
                    continue
                event_type = feedback_event_type(verdict)
                try:
                    result = store.record_feedback(
                        draft_id,
                        event_type,
                        actor=actor,
                        reason=comment or verdict,
                        metadata={
                            "human_verdict": verdict,
                            "human_comment": comment,
                            "corrected_answer": corrected,
                            "source_csv": str(csv_path),
                            "line_no": line_no,
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    skipped += 1
                    errors.append(f"line {line_no}: {exc}")
                    continue
                imported += 1 if result.created else 0
                skipped += 0 if result.created else 1
    finally:
        if close_store:
            store.close()
    return FeedbackImportSummary(imported=imported, skipped=skipped, errors=tuple(errors))


def feedback_event_type(verdict: str) -> str:
    normalized = "_".join(str(verdict or "").strip().casefold().split())
    mapping = {
        "useful": PILOT_FEEDBACK_USEFUL,
        "minor_edit": PILOT_FEEDBACK_NEEDS_EDIT,
        "needs_edit": PILOT_FEEDBACK_NEEDS_EDIT,
        "rewrite": PILOT_FEEDBACK_NEEDS_EDIT,
        "wrong_fact": PILOT_FEEDBACK_UNSAFE_FACT_ATTEMPT,
        "unsafe": PILOT_FEEDBACK_UNSAFE_FACT_ATTEMPT,
        "too_robotic": PILOT_FEEDBACK_NEEDS_EDIT,
        "asked_known_data": PILOT_FEEDBACK_NEEDS_EDIT,
        "manager_only": PILOT_FEEDBACK_MANAGER_ONLY,
        "topic_wrong": PILOT_FEEDBACK_TOPIC_WRONG,
    }
    return mapping.get(normalized, f"human_{normalized or 'other'}")


def review_reasons_for_row(row: Mapping[str, Any], *, raw_answer: str, raw_input: str) -> list[str]:
    reasons: list[str] = []
    flags_text = " ".join([str(row.get("safety_flags") or ""), str(row.get("semantic_flags") or "")]).casefold()
    topic = str(row.get("topic_id") or "")
    route = str(row.get("route") or "")
    if topic in P0_TOPICS or P0_RE.search(raw_input) or any(flag in flags_text for flag in HIGH_RISK_FLAGS):
        reasons.append("p0_or_high_risk")
    if route in AUTONOMOUS_ROUTES and (topic in P0_TOPICS or P0_RE.search(raw_input)):
        reasons.append("p0_autonomous_attempt")
    if str(row.get("asked_known_data_again") or "") == "true":
        reasons.append("asked_known_data_again")
    if TEMPLATE_RE.search(raw_answer) or (GENERIC_RE.search(raw_answer) and len(raw_answer) < 220):
        reasons.append("template_or_generic")
    if PRECISE_FACT_RE.search(raw_answer):
        reasons.append("precise_number_date_or_percent")
    if INTERNAL_MARKER_RE.search(raw_answer):
        reasons.append("internal_marker_leak")
    if not raw_answer.strip():
        reasons.append("empty_answer")
    if str(row.get("facts_missing") or "") not in {"", "[]"} and route in AUTONOMOUS_ROUTES:
        reasons.append("autonomous_with_missing_facts")
    return list(dict.fromkeys(reasons))


def build_p0_rows(rows: Sequence[Mapping[str, Any]], *, p0_register_path: Path | str | None = None) -> list[Mapping[str, Any]]:
    p0_rows = [row for row in rows if "p0_or_high_risk" in str(row.get("review_reasons") or "")]
    if p0_register_path:
        path = Path(p0_register_path)
        if path.exists():
            p0_rows = [*p0_rows, *read_p0_register_rows(path)]
    return p0_rows


def read_p0_register_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for source in csv.DictReader(file):
            rows.append(
                {
                    "review_id": str(source.get("incident_id") or source.get("created_at") or "p0-register"),
                    "draft_id": "",
                    "message_key": "",
                    "timestamp": str(source.get("created_at") or ""),
                    "brand": str(source.get("brand") or ""),
                    "chat_id": mask_text(str(source.get("chat_id") or "")),
                    "input_text": mask_text(str(source.get("input_text") or "")),
                    "answer_text": mask_text(str(source.get("answer_text") or "")),
                    "route": str(source.get("route") or ""),
                    "topic_id": str(source.get("topic_id") or ""),
                    "review_reasons": "p0_register",
                }
            )
    return rows


def build_facts_used_summary(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, str]]:
    facts: dict[str, Counter[str]] = {}
    brands: dict[str, set[str]] = {}
    routes: dict[str, set[str]] = {}
    for row in rows:
        for fact in parse_json_list(row.get("facts_used")):
            text = mask_text(str(fact)).strip()
            if not text:
                continue
            facts.setdefault(text, Counter()).update(["count"])
            brands.setdefault(text, set()).add(str(row.get("brand") or ""))
            routes.setdefault(text, set()).add(str(row.get("route") or ""))
    return [
        {
            "fact": fact,
            "count": str(counter["count"]),
            "brands": ", ".join(sorted(brands.get(fact) or [])),
            "routes": ", ".join(sorted(routes.get(fact) or [])),
        }
        for fact, counter in sorted(facts.items(), key=lambda item: (-item[1]["count"], item[0]))
    ]


def build_pilot_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    review_rows: Sequence[Mapping[str, Any]],
    regression_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    route_counts = Counter(str(row.get("route") or "") for row in rows)
    brand_counts = Counter(str(row.get("brand") or "") for row in rows)
    reasons = Counter(
        reason
        for row in rows
        for reason in [part.strip() for part in str(row.get("review_reasons") or "").split(";")]
        if reason
    )
    latencies = [float(row.get("latency_seconds") or 0) for row in rows if str(row.get("latency_seconds") or "").strip()]
    return {
        "schema_version": TELEGRAM_PILOT_DAILY_REPORT_SCHEMA_VERSION,
        "store_schema_version": TELEGRAM_PILOT_STORE_SCHEMA_VERSION,
        "metrics": dict(metrics),
        "messages_total": len(rows),
        "brand_counts": dict(brand_counts),
        "route_counts": dict(route_counts),
        "autonomous_answers": route_counts.get("bot_answer_self_for_pilot", 0) + route_counts.get("bot_answer_self", 0),
        "manager_only": route_counts.get("manager_only", 0),
        "draft_for_manager": route_counts.get("draft_for_manager", 0),
        "review_queue_count": len(review_rows),
        "regression_candidates_count": len(regression_rows),
        "known_data_reask_count": sum(1 for row in rows if str(row.get("asked_known_data_again") or "") == "true"),
        "template_or_generic_count": reasons.get("template_or_generic", 0),
        "p0_or_high_risk_count": reasons.get("p0_or_high_risk", 0),
        "precise_fact_count": reasons.get("precise_number_date_or_percent", 0),
        "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "median_latency_seconds": round(statistics.median(latencies), 3) if latencies else None,
        "review_reason_counts": dict(reasons),
        "formal_passed": True,
        "semantic_passed": len(review_rows) == 0,
        "semantic_status": "PASS" if not review_rows else "PASS_WITH_NOTES",
    }


def is_regression_candidate(row: Mapping[str, Any]) -> bool:
    reasons = str(row.get("review_reasons") or "")
    return any(
        marker in reasons
        for marker in (
            "p0_autonomous_attempt",
            "internal_marker_leak",
            "asked_known_data_again",
            "autonomous_with_missing_facts",
            "template_or_generic",
        )
    )


def facts_used_from_context_and_result(context: Mapping[str, Any], llm_result: Mapping[str, Any]) -> list[str]:
    facts: list[str] = []
    for key in ("context_used", "facts_used", "confirmed_facts_used"):
        facts.extend(str(item) for item in _list(llm_result.get(key)))
    confirmed = context.get("confirmed_facts")
    if isinstance(confirmed, Mapping):
        facts.extend(f"{key}: {value}" for key, value in confirmed.items())
    facts_context = context.get("facts_context")
    if isinstance(facts_context, Mapping):
        for key, value in facts_context.items():
            if isinstance(value, (str, int, float)):
                facts.append(f"{key}: {value}")
    return list(dict.fromkeys(item for item in facts if str(item).strip()))


def detect_asked_known_data_again(
    answer_text: str,
    *,
    known_client: Mapping[str, Any] | None = None,
    known_dialog: Mapping[str, Any] | None = None,
    known_slots: Mapping[str, Any] | None = None,
) -> list[str]:
    known: dict[str, str] = {}
    for source in (known_client or {}, known_dialog or {}, known_slots or {}):
        for key, value in source.items():
            if str(value or "").strip():
                known[str(key)] = str(value)
    text = str(answer_text or "").casefold().replace("ё", "е")
    result: list[str] = []
    if known.get("student_name") and re.search(r"(фио|имя|как\s+зовут)[^.!?\n]{0,80}(ребенк|ученик)", text):
        result.append("student_name")
    if known.get("parent_name") and re.search(r"(ваше\s+имя|как\s+вас\s+зовут|фио\s+родител)", text):
        result.append("parent_name")
    if known.get("phone") and re.search(r"(телефон|номер\s+телефона|контактн\w+\s+номер)", text):
        result.append("phone")
    if known.get("grade") and re.search(r"(какой\s+класс|класс\s+ребенк|напишите[^.!?\n]{0,40}класс|подскажите[^.!?\n]{0,40}класс)", text):
        result.append("grade")
    if known.get("subject") and re.search(r"(какой\s+предмет|предмет[^.!?\n]{0,30}интерес|напишите[^.!?\n]{0,40}предмет|подскажите[^.!?\n]{0,40}предмет)", text):
        result.append("subject")
    if known.get("format") and re.search(r"(онлайн\s+или\s+очн|очный\s+или\s+онлайн|какой\s+формат)", text):
        result.append("format")
    return list(dict.fromkeys(result))


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# Telegram pilot daily report: {summary.get('metrics', {}).get('day')}",
            "",
            f"- formal_passed: `{summary.get('formal_passed')}`",
            f"- semantic_status: `{summary.get('semantic_status')}`",
            f"- messages_total: `{summary.get('messages_total')}`",
            f"- autonomous_answers: `{summary.get('autonomous_answers')}`",
            f"- manager_only: `{summary.get('manager_only')}`",
            f"- draft_for_manager: `{summary.get('draft_for_manager')}`",
            f"- review_queue_count: `{summary.get('review_queue_count')}`",
            f"- regression_candidates_count: `{summary.get('regression_candidates_count')}`",
            f"- known_data_reask_count: `{summary.get('known_data_reask_count')}`",
            f"- template_or_generic_count: `{summary.get('template_or_generic_count')}`",
            f"- avg_latency_seconds: `{summary.get('avg_latency_seconds')}`",
            "",
            "## Что смотреть первым",
            "",
            "1. `p0_incidents.csv`.",
            "2. `known_data_reask_cases.csv`.",
            "3. `template_or_generic_cases.csv`.",
            "4. `semantic_review_queue.csv`.",
            "5. `employee_review_sheet.csv` для разметки сотрудниками.",
            "",
        ]
    )


def render_implementation_notes(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Implementation Notes",
            "",
            "Отчёт построен из локального TelegramPilotSQLiteStore в read-only режиме.",
            "AMO, Tallanto, CRM, stable_runtime и Telegram live-send этим отчётом не меняются.",
            "",
            f"- schema: `{summary.get('schema_version')}`",
            f"- store_schema: `{summary.get('store_schema_version')}`",
            f"- messages_total: `{summary.get('messages_total')}`",
            "",
        ]
    )


def render_semantic_review(summary: Mapping[str, Any]) -> str:
    verdict = summary.get("semantic_status")
    return "\n".join(
        [
            "# Semantic Review",
            "",
            f"Verdict: `{verdict}`",
            "",
            "Смысловая проверка здесь автоматическая и не заменяет ручное ревью спорных ответов.",
            "Все строки из `semantic_review_queue.csv` должны быть просмотрены до расширения пилота.",
            "",
            "## Главные очереди",
            "",
            f"- P0/high-risk: `{summary.get('p0_or_high_risk_count')}`",
            f"- Повтор известных данных: `{summary.get('known_data_reask_count')}`",
            f"- Шаблонные/общие ответы: `{summary.get('template_or_generic_count')}`",
            f"- Кандидаты в регрессии: `{summary.get('regression_candidates_count')}`",
            "",
        ]
    )


def model_from_prompt_version(prompt_version: str) -> str:
    parts = str(prompt_version or "").split(":")
    return parts[1] if len(parts) >= 2 else ""


def reasoning_from_prompt_version(prompt_version: str) -> str:
    parts = str(prompt_version or "").split(":")
    return parts[2] if len(parts) >= 3 else ""


def mask_text(value: str) -> str:
    text = str(value or "")
    text = TOKEN_RE.sub("[TOKEN_MASKED]", text)
    text = EMAIL_RE.sub("[EMAIL_MASKED]", text)
    text = PHONE_RE.sub("[PHONE_MASKED]", text)
    return text


def json_dump_masked(value: Any) -> str:
    return mask_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def parse_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
