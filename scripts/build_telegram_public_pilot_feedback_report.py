#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8)?[\s(.-]*\d{3}[\s).-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
TOKEN_RE = re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{25,}\b")
SECRET_KEY_RE = re.compile(r"(?:token|api[_-]?key|secret|password)", re.I)
MONEY_DATE_PERCENT_RE = re.compile(
    r"\b\d[\d\s\u00a0]{1,9}\s*(?:₽|руб|р\.)|\b\d{1,3}\s*%|\b(?:до|по|с)\s+\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)",
    re.I,
)
HIGH_RISK_TOPIC_IDS = {"theme:009_refund", "theme:019b_negative_feedback", "theme:029_legal_question"}
HIGH_RISK_TEXT_RE = re.compile(r"возврат|верн\w+\s+деньг|жалоб|суд|иск|претензи|прокуратур|роспотребнадзор", re.I)
FALLBACK_RE = re.compile(r"передам вопрос менеджеру|менеджер свяжется|проверит и свяжется", re.I)
TEMPLATE_RE = re.compile(
    r"спасибо за обращение|ваш вопрос очень важен|оптимальн\w+\s+образовательн\w+\s+продукт|индивидуальн\w+\s+подход",
    re.I,
)
GENERIC_RE = re.compile(r"уточн(?:ю|им)|передам|свяжется|подбер[её]м|проверит", re.I)
QUESTION_RE = re.compile(r"\?")

REPORT_FIELDS = (
    "review_id",
    "brand",
    "timestamp",
    "chat_id",
    "input_text",
    "answer_text",
    "route",
    "topic_id",
    "message_type",
    "risk_level",
    "safety_flags",
    "context_flags",
    "known_client_fields",
    "known_dialog_fields",
    "asked_again_fields",
    "latency_seconds",
    "human_tone_score",
    "human_tone_flags",
    "why_review",
    "human_verdict",
    "human_comment",
    "next_action",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build review reports from Telegram public pilot bot logs.")
    parser.add_argument("--log-dir", type=Path, default=Path(".codex_local/telegram_pilot_bots/logs"))
    parser.add_argument("--date", default=date.today().isoformat(), help="UTC log date YYYY-MM-DD.")
    parser.add_argument("--brand", choices=("all", "foton", "unpk"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    records = load_log_records(args.log_dir, date_filter=args.date, brand=args.brand)
    messages = build_message_rows(records)
    write_reports(messages, output_dir=args.output_dir, report_date=args.date, brand=args.brand)
    return 0


def load_log_records(log_dir: Path, *, date_filter: str, brand: str = "all") -> list[dict[str, Any]]:
    brands = ("foton", "unpk") if brand == "all" else (brand,)
    result: list[dict[str, Any]] = []
    for item_brand in brands:
        path = log_dir / f"{date_filter}_{item_brand}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    record.setdefault("brand", item_brand)
                    result.append(record)
    result.sort(key=lambda item: str(item.get("ts") or ""))
    return result


def build_message_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    last_input_by_chat: dict[tuple[str, str], str] = {}
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for record in records:
        brand = str(record.get("brand") or "")
        chat_id = str(record.get("chat_id") or "")
        event = str(record.get("event") or "")
        key = (brand, chat_id)
        if record.get("input_text"):
            last_input_by_chat[key] = str(record.get("input_text") or "")
        if event != "reply_sent":
            continue
        counters[brand] += 1
        input_text = str(record.get("input_text") or last_input_by_chat.get(key, ""))
        answer_text = str(record.get("answer_text") or "")
        known_client = _mapping(record.get("known_client_fields"))
        known_dialog = _mapping(record.get("known_dialog_fields"))
        asked_again = detect_asked_known_data_again(answer_text, known_client=known_client, known_dialog=known_dialog)
        tone = score_human_tone(answer_text, input_text=input_text, asked_again_fields=asked_again)
        review_reasons = review_reasons_for_record(record, input_text=input_text, answer_text=answer_text, asked_again=asked_again, tone=tone)
        row = {
            "review_id": f"{brand}-{counters[brand]:04d}",
            "brand": brand,
            "timestamp": str(record.get("ts") or ""),
            "chat_id": mask_text(chat_id),
            "input_text": mask_text(input_text),
            "answer_text": mask_text(answer_text),
            "route": str(record.get("route") or ""),
            "topic_id": str(record.get("topic_id") or ""),
            "message_type": str(record.get("message_type") or ""),
            "risk_level": str(record.get("risk_level") or ""),
            "safety_flags": _json_masked(record.get("safety_flags") or []),
            "context_flags": _json_masked(record.get("context_flags") or {}),
            "known_client_fields": _json_masked(known_client),
            "known_dialog_fields": _json_masked(known_dialog),
            "asked_again_fields": _json_masked(asked_again),
            "latency_seconds": _number(record.get("latency_seconds")),
            "human_tone_score": tone["score"],
            "human_tone_flags": _json_masked(tone["flags"]),
            "why_review": "; ".join(review_reasons),
            "human_verdict": "",
            "human_comment": "",
            "next_action": suggested_next_action(review_reasons),
        }
        rows.append(row)
    return rows


def write_reports(rows: Sequence[Mapping[str, Any]], *, output_dir: Path, report_date: str, brand: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pilot_messages.csv", rows)
    write_jsonl(output_dir / "pilot_messages.jsonl", rows)
    review_rows = [row for row in rows if str(row.get("why_review") or "").strip()]
    write_csv(output_dir / "semantic_review_queue.csv", review_rows)
    regression_rows = [row for row in review_rows if regression_candidate(row)]
    write_csv(output_dir / "regression_candidates.csv", regression_rows)
    write_csv(output_dir / "employee_review_sheet.csv", review_rows or rows)
    summary = build_summary(rows, review_rows=review_rows, regression_rows=regression_rows, report_date=report_date, brand=brand)
    (output_dir / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary_md = render_summary_markdown(summary)
    (output_dir / "pilot_summary.md").write_text(summary_md, encoding="utf-8")
    (output_dir / f"daily_pilot_report_{report_date}.md").write_text(summary_md, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REPORT_FIELDS})


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    review_rows: Sequence[Mapping[str, Any]],
    regression_rows: Sequence[Mapping[str, Any]],
    report_date: str,
    brand: str,
) -> dict[str, Any]:
    total = len(rows)
    latencies = [float(row.get("latency_seconds") or 0) for row in rows if str(row.get("latency_seconds") or "")]
    routes = Counter(str(row.get("route") or "") for row in rows)
    tone_scores = [int(row.get("human_tone_score") or 0) for row in rows if str(row.get("human_tone_score") or "")]
    asked_again = [row for row in rows if "asked_known_data_again" in str(row.get("human_tone_flags") or "")]
    template_like = [row for row in rows if "template_like_answer" in str(row.get("human_tone_flags") or "")]
    useful_rows = [row for row in rows if int(row.get("human_tone_score") or 0) >= 65]
    context_rows = [row for row in rows if "true" in str(row.get("context_flags") or "").casefold()]
    return {
        "schema_version": "telegram_public_pilot_feedback_report_v1",
        "report_date": report_date,
        "brand": brand,
        "messages_total": total,
        "routes": dict(routes),
        "autonomous_answers": routes.get("bot_answer_self_for_pilot", 0) + routes.get("bot_answer_self", 0),
        "manager_only": routes.get("manager_only", 0),
        "draft_for_manager": routes.get("draft_for_manager", 0),
        "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else None,
        "median_latency_seconds": round(statistics.median(latencies), 3) if latencies else None,
        "review_queue_count": len(review_rows),
        "regression_candidates_count": len(regression_rows),
        "asked_known_data_again_count": len(asked_again),
        "template_like_answer_count": len(template_like),
        "useful_answer_rate": round(len(useful_rows) / total, 3) if total else 0,
        "context_used_rate": round(len(context_rows) / total, 3) if total else 0,
        "avg_human_tone_score": round(statistics.mean(tone_scores), 1) if tone_scores else None,
    }


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"# Telegram pilot daily report: {summary.get('report_date')}",
            "",
            f"- Бренд: `{summary.get('brand')}`",
            f"- Ответов бота: `{summary.get('messages_total')}`",
            f"- Автономных ответов: `{summary.get('autonomous_answers')}`",
            f"- Manager-only: `{summary.get('manager_only')}`",
            f"- Черновиков менеджеру: `{summary.get('draft_for_manager')}`",
            f"- Очередь смысловой проверки: `{summary.get('review_queue_count')}`",
            f"- Кандидаты в регрессии: `{summary.get('regression_candidates_count')}`",
            f"- Повторно спросил известные данные: `{summary.get('asked_known_data_again_count')}`",
            f"- Шаблонных ответов: `{summary.get('template_like_answer_count')}`",
            f"- Средняя оценка живости: `{summary.get('avg_human_tone_score')}`",
            f"- Средняя задержка, сек: `{summary.get('avg_latency_seconds')}`",
            "",
            "## Что смотреть вручную",
            "",
            "Сначала откройте `semantic_review_queue.csv`: там P0/high-risk, ответы с цифрами, ответы с CRM/Tallanto-контекстом, слишком шаблонные ответы и случаи, где бот мог спросить уже известные данные.",
            "",
        ]
    )


def detect_asked_known_data_again(
    answer_text: str,
    *,
    known_client: Mapping[str, Any] | None = None,
    known_dialog: Mapping[str, Any] | None = None,
) -> list[str]:
    known: dict[str, str] = {}
    for source in (known_client or {}, known_dialog or {}):
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
    return list(dict.fromkeys(result))


def score_human_tone(answer_text: str, *, input_text: str = "", asked_again_fields: Sequence[str] = ()) -> dict[str, Any]:
    text = str(answer_text or "").strip()
    lowered = text.casefold()
    score = 70
    flags: list[str] = []
    if not text:
        return {"score": 0, "flags": ["empty_answer", "human_tone_review_required"], "why": "empty_answer"}
    if TEMPLATE_RE.search(text):
        score -= 25
        flags.append("template_like_answer")
    if FALLBACK_RE.search(text) and len(text) < 180:
        score -= 20
        flags.append("too_generic_answer")
    if len(QUESTION_RE.findall(text)) > 3:
        score -= 15
        flags.append("too_many_questions")
    if asked_again_fields:
        score -= 30
        flags.append("asked_known_data_again")
    if "менеджер" in lowered and not any(marker in lowered for marker in ("курс", "класс", "предмет", "формат", "стоим", "распис", "смен", "документ", "скид", "оплат")):
        score -= 10
        flags.append("manager_handoff_without_value")
    if any(marker in lowered for marker in ("да,", "можно", "есть", "подойдет", "сориентирую", "вариант", "курс", "смен")):
        score += 10
    if any(marker in lowered for marker in ("следующий шаг", "подскажите", "напишите", "если")):
        score += 5
    score = max(0, min(100, score))
    if score < 55:
        flags.append("human_tone_review_required")
    return {"score": score, "flags": list(dict.fromkeys(flags)), "why": ", ".join(flags)}


def review_reasons_for_record(
    record: Mapping[str, Any],
    *,
    input_text: str,
    answer_text: str,
    asked_again: Sequence[str],
    tone: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    flags_text = " ".join(str(item) for item in _as_list(record.get("safety_flags"))).casefold()
    topic = str(record.get("topic_id") or "")
    route = str(record.get("route") or "")
    if topic in HIGH_RISK_TOPIC_IDS or HIGH_RISK_TEXT_RE.search(input_text) or "high_risk" in flags_text:
        reasons.append("high_risk_or_p0")
    if MONEY_DATE_PERCENT_RE.search(answer_text):
        reasons.append("precise_number_date_or_percent")
    if "crm" in json.dumps(record.get("context_flags") or {}, ensure_ascii=False).casefold() or record.get("known_client_fields"):
        reasons.append("crm_tallanto_or_known_context_used")
    if asked_again:
        reasons.append("asked_known_data_again")
    if route == "manager_only" and not answer_text:
        reasons.append("empty_manager_only_answer")
    if route == "manager_only" and FALLBACK_RE.search(answer_text):
        reasons.append("fallback_or_handoff")
    for flag in tone.get("flags") or []:
        if flag in {"template_like_answer", "too_generic_answer", "human_tone_review_required"}:
            reasons.append(str(flag))
    return list(dict.fromkeys(reasons))


def suggested_next_action(reasons: Sequence[str]) -> str:
    if not reasons:
        return ""
    if "high_risk_or_p0" in reasons:
        return "Проверить безопасность текста и маршрут."
    if "asked_known_data_again" in reasons:
        return "Проверить контекст и добавить регрессионный тест."
    if "template_like_answer" in reasons or "too_generic_answer" in reasons:
        return "Переписать правило тона или шаблон."
    return "Проверить вручную."


def regression_candidate(row: Mapping[str, Any]) -> bool:
    reasons = str(row.get("why_review") or "")
    return any(marker in reasons for marker in ("high_risk_or_p0", "asked_known_data_again", "template_like_answer", "too_generic_answer"))


def mask_text(text: str) -> str:
    value = str(text or "")
    value = TOKEN_RE.sub("[TG_TOKEN_MASKED]", value)
    value = EMAIL_RE.sub("[EMAIL_MASKED]", value)
    value = PHONE_RE.sub("[PHONE_MASKED]", value)
    return value


def _json_masked(value: Any) -> str:
    return mask_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return []


def _number(value: Any) -> str:
    try:
        return str(round(float(value), 3))
    except (TypeError, ValueError):
        return ""


def validate_no_secret_leak(rows: Sequence[Mapping[str, Any]]) -> None:
    raw = json.dumps(list(rows), ensure_ascii=False)
    if TOKEN_RE.search(raw) or SECRET_KEY_RE.search(raw):
        raise RuntimeError("report contains token-like or secret-like text")


if __name__ == "__main__":
    raise SystemExit(main())
