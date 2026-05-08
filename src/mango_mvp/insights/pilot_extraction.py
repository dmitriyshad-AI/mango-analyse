from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CONTENTFUL_FALSE_VALUES = {"", "false", "0", "no", "нет"}
ROLE_BLOCK_RE = re.compile(r"(?ims)^\s*(MANAGER|CLIENT|МЕНЕДЖЕР|КЛИЕНТ)\s*:\s*(.*?)(?=^\s*(?:MANAGER|CLIENT|МЕНЕДЖЕР|КЛИЕНТ)\s*:|\Z)")
QUESTION_RE = re.compile(r"\?|подскаж|сколько|когда|можно|как |какой|какая|какие|где|есть ли|интерес|хотим|нужн", re.I)
MANAGER_ANSWER_RE = re.compile(
    r"можем|предлага|отправ|пришл|ссылк|оплат|стоимост|цен|расписан|групп|курс|лагер|"
    r"интенсив|заняти|перезвон|уточн|провер|скидк|договор|квитанц",
    re.I,
)
PRICE_RE = re.compile(r"цен|стоимост|дорог|дешев|бюджет|скидк|рассроч|маткапитал|материнск", re.I)
SCHEDULE_RE = re.compile(r"расписан|врем|день|занят|нагрузк|неудоб|ездить|онлайн|очно", re.I)
TRUST_RE = re.compile(r"отзыв|результат|преподавател|качество|довер|рекоменд|обратн", re.I)
PAYMENT_RE = re.compile(r"оплат|счет|счёт|чек|договор|квитанц|qr", re.I)
REFUSAL_RE = re.compile(r"отказ|не актуаль|неинтерес|не интерес|не подходит|не будем|не пойд|не продолж", re.I)
NEXT_YEAR_RE = re.compile(r"следующ.*год|нов.*учебн|продолж|дальше", re.I)
PRODUCT_RE = re.compile(r"курс|лагер|школ|интенсив|заняти|математ|физик|информат|егэ|огэ|олимпиад", re.I)


@dataclass(frozen=True)
class PilotExtractionConfig:
    project_root: Path
    readiness_root: Path
    outcome_root: Path
    out_root: Path
    max_clients: int = 500
    max_calls_per_client: int = 6
    transcript_chars_for_llm: int = 9000


def build_pilot_sales_moments(config: PilotExtractionConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pilot_rows = _read_csv(config.outcome_root / "pilot_outcome_sample.csv")[: config.max_clients]
    call_rows = _read_csv(config.readiness_root / "calls_terminal_analyzed.csv")
    calls_by_phone = _group_by(call_rows, "phone")

    selected_calls: list[dict[str, Any]] = []
    client_sequences: list[dict[str, Any]] = []
    for chain in pilot_rows:
        phone = str(chain.get("phone") or "")
        chain_calls = sorted(calls_by_phone.get(phone, []), key=lambda row: str(row.get("started_at") or ""))
        selected = select_calls_for_client(chain, chain_calls, config.max_calls_per_client)
        selected_names = {row.get("source_filename") for row in selected}
        selected_calls.extend(selected)
        client_sequences.append(build_client_sequence_row(chain, chain_calls, selected_names))

    db_cache: dict[str, sqlite3.Connection] = {}
    moments: list[dict[str, Any]] = []
    llm_inputs: list[dict[str, Any]] = []
    try:
        for idx, call in enumerate(selected_calls, start=1):
            chain = _chain_for_call(call, pilot_rows)
            db_record = load_call_record(project_root, call, db_cache)
            if db_record is None:
                continue
            moment = extract_sales_moment(idx, chain, call, db_record)
            moments.append(moment)
            llm_inputs.append(build_llm_input(moment, chain, call, db_record, config.transcript_chars_for_llm))
    finally:
        for con in db_cache.values():
            con.close()

    summary = build_summary(config, pilot_rows, selected_calls, moments, client_sequences)
    outputs = write_outputs(out_root, summary, moments, client_sequences, llm_inputs)
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def select_calls_for_client(chain: dict[str, Any], calls: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    contentful = [row for row in calls if _is_contentful(row)]
    if not contentful or limit <= 0:
        return []

    scored = sorted(contentful, key=lambda row: (-_call_selection_score(row, chain), str(row.get("started_at") or "")))
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    anchors = [contentful[0], contentful[-1]]
    for row in anchors + scored:
        name = str(row.get("source_filename") or "")
        if not name or name in seen:
            continue
        seen.add(name)
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda row: str(row.get("started_at") or ""))


def build_client_sequence_row(chain: dict[str, Any], calls: list[dict[str, Any]], selected_names: set[Any]) -> dict[str, Any]:
    contentful = [row for row in calls if _is_contentful(row)]
    return {
        "client_key": chain.get("client_key", ""),
        "phone": chain.get("phone", ""),
        "years": chain.get("years", ""),
        "first_seen_at": chain.get("first_seen_at", ""),
        "last_seen_at": chain.get("last_seen_at", ""),
        "touch_count": chain.get("touch_count", ""),
        "contentful_call_count": chain.get("contentful_call_count", ""),
        "selected_call_count": len(selected_names),
        "final_outcome_label": chain.get("final_outcome_label", ""),
        "outcome_confidence_tier": chain.get("outcome_confidence_tier", ""),
        "sales_action_label": chain.get("sales_action_label", ""),
        "extraction_use_case": chain.get("extraction_use_case", ""),
        "dominant_call_type": chain.get("dominant_call_type", ""),
        "managers": chain.get("managers", ""),
        "products_top": chain.get("products_top", ""),
        "subjects_top": chain.get("subjects_top", ""),
        "objections_top": chain.get("objections_top", ""),
        "sequence_preview": " | ".join(
            f"{row.get('started_at', '')[:10]} {row.get('call_type', '')}: {_clip(row.get('history_summary'), 220)}"
            for row in contentful[:12]
        ),
    }


def load_call_record(project_root: Path, call: dict[str, Any], db_cache: dict[str, sqlite3.Connection]) -> dict[str, Any] | None:
    db_raw = str(call.get("source_db") or "")
    source_filename = str(call.get("source_filename") or "")
    if not db_raw or not source_filename:
        return None
    db_path = Path(db_raw)
    if not db_path.is_absolute():
        db_path = project_root / db_path
    key = str(db_path.resolve())
    con = db_cache.get(key)
    if con is None:
        con = sqlite3.connect(key, timeout=15)
        con.row_factory = sqlite3.Row
        db_cache[key] = con
    row = con.execute(
        "select * from call_records where source_filename = ? order by updated_at desc limit 1",
        (source_filename,),
    ).fetchone()
    if row is None:
        return None
    return {name: row[name] for name in row.keys()}


def extract_sales_moment(moment_index: int, chain: dict[str, Any], call: dict[str, Any], db_record: dict[str, Any]) -> dict[str, Any]:
    analysis = _safe_json(db_record.get("analysis_json"))
    structured = _dict(analysis.get("structured_fields")) or _dict(analysis.get("crm_blocks"))
    transcript = str(db_record.get("transcript_text") or "")
    roles = parse_role_blocks(transcript)
    manager_text = roles.get("manager", "")
    client_text = roles.get("client", "")
    history = _clean(call.get("history_summary")) or _clean(analysis.get("history_summary"))
    question = choose_customer_question_or_need(client_text, history, structured, call)
    answer = choose_manager_answer(manager_text, history, structured, call)
    signal, signal_evidence = infer_customer_signal(chain, call, question, client_text, history)
    stage = infer_hidden_sales_stage(chain, call, signal)
    quality_score, quality_band, quality_reasons = score_manager_response(chain, call, signal, question, answer)
    missed, missed_reason = infer_missed_opportunity(chain, call, signal, answer, quality_score)
    ideal_reaction, ideal_template = ideal_reaction_for_signal(signal, chain, call)
    touch_position = _touch_position(call, chain)

    return {
        "moment_id": f"pilot-{moment_index:05d}",
        "client_key": chain.get("client_key", ""),
        "phone": chain.get("phone", ""),
        "source_filename": call.get("source_filename", ""),
        "started_at": call.get("started_at", ""),
        "manager_name": call.get("manager_name", ""),
        "call_type": call.get("call_type", ""),
        "chain_touch_position": touch_position,
        "chain_touch_count": chain.get("touch_count", ""),
        "hidden_sales_stage": stage,
        "final_outcome_label": chain.get("final_outcome_label", ""),
        "outcome_confidence_tier": chain.get("outcome_confidence_tier", ""),
        "sales_action_label": chain.get("sales_action_label", ""),
        "extraction_use_case": chain.get("extraction_use_case", ""),
        "customer_question_or_need": question,
        "manager_answer_or_reaction": answer,
        "customer_signal_label": signal,
        "customer_signal_evidence": signal_evidence,
        "manager_response_quality_score": quality_score,
        "manager_response_quality_band": quality_band,
        "manager_response_quality_reasons": " | ".join(quality_reasons),
        "missed_opportunity_flag": missed,
        "missed_opportunity_reason": missed_reason,
        "ideal_manager_reaction": ideal_reaction,
        "ideal_answer_template": ideal_template,
        "products": call.get("products", ""),
        "subjects": call.get("subjects", ""),
        "formats": call.get("formats", ""),
        "objections": call.get("objections", ""),
        "next_step": call.get("next_step", ""),
        "lead_priority": call.get("lead_priority", ""),
        "follow_up_score": call.get("follow_up_score", ""),
        "history_summary": history,
        "transcript_excerpt": _clip(_clean(transcript), 1200),
    }


def build_llm_input(
    moment: dict[str, Any],
    chain: dict[str, Any],
    call: dict[str, Any],
    db_record: dict[str, Any],
    transcript_chars: int,
) -> dict[str, Any]:
    analysis = _safe_json(db_record.get("analysis_json"))
    transcript = str(db_record.get("transcript_text") or "")
    return {
        "id": moment["moment_id"],
        "task": "sales_moment_quality_extraction_v1",
        "schema": {
            "customer_question": "string",
            "manager_answer": "string",
            "customer_signal": "one of: price, schedule, trust, payment, refusal, next_year, product_interest, service, unknown",
            "hidden_sales_stage": "string",
            "answer_quality_score_0_100": "integer",
            "answer_quality_reason": "string",
            "ideal_manager_reaction": "string",
            "ideal_answer_template": "string",
        },
        "chain_context": {
            "phone": chain.get("phone", ""),
            "years": chain.get("years", ""),
            "touch_count": chain.get("touch_count", ""),
            "final_outcome_label": chain.get("final_outcome_label", ""),
            "outcome_confidence_tier": chain.get("outcome_confidence_tier", ""),
            "extraction_use_case": chain.get("extraction_use_case", ""),
            "products_top": chain.get("products_top", ""),
            "subjects_top": chain.get("subjects_top", ""),
            "objections_top": chain.get("objections_top", ""),
        },
        "call_context": {
            "source_filename": call.get("source_filename", ""),
            "started_at": call.get("started_at", ""),
            "manager_name": call.get("manager_name", ""),
            "call_type": call.get("call_type", ""),
            "history_summary": call.get("history_summary", ""),
            "structured_fields": analysis.get("structured_fields") or analysis.get("crm_blocks") or {},
        },
        "deterministic_seed": {
            key: moment.get(key, "")
            for key in (
                "customer_question_or_need",
                "manager_answer_or_reaction",
                "customer_signal_label",
                "hidden_sales_stage",
                "manager_response_quality_score",
                "ideal_manager_reaction",
                "ideal_answer_template",
            )
        },
        "transcript": _middle_clip(transcript, transcript_chars),
    }


def parse_role_blocks(transcript: str) -> dict[str, str]:
    roles = {"manager": "", "client": ""}
    for match in ROLE_BLOCK_RE.finditer(transcript or ""):
        role = match.group(1).lower()
        key = "manager" if role in {"manager", "менеджер"} else "client"
        roles[key] = (roles[key] + "\n" + match.group(2).strip()).strip()
    if not roles["manager"] and not roles["client"]:
        roles["client"] = transcript or ""
    return roles


def choose_customer_question_or_need(client_text: str, history: str, structured: dict[str, Any], call: dict[str, Any]) -> str:
    client_sentences = split_sentences(client_text)
    best = _best_sentence(client_sentences, QUESTION_RE)
    if best:
        return best
    interests = _dict(structured.get("interests"))
    products = _string_list(interests.get("products")) or _split_joined(call.get("products"))
    subjects = _string_list(interests.get("subjects")) or _split_joined(call.get("subjects"))
    objections = _string_list(structured.get("objections")) or _split_joined(call.get("objections"))
    parts = []
    if products or subjects:
        parts.append("интерес: " + ", ".join(products + subjects))
    if objections:
        parts.append("возражение/ограничение: " + ", ".join(objections[:3]))
    if not parts:
        parts.append(_clip(history, 260))
    return "; ".join(part for part in parts if part)


def choose_manager_answer(manager_text: str, history: str, structured: dict[str, Any], call: dict[str, Any]) -> str:
    manager_sentences = split_sentences(manager_text)
    best = _best_sentence(manager_sentences, MANAGER_ANSWER_RE)
    if best:
        return best
    next_step = _clean(call.get("next_step")) or _clean(_dict(structured.get("next_step")).get("action"))
    if next_step:
        return f"Менеджер согласовал следующий шаг: {next_step}."
    return _clip(history, 320)


def infer_customer_signal(chain: dict[str, Any], call: dict[str, Any], question: str, client_text: str, history: str) -> tuple[str, str]:
    text = " ".join([question, client_text, history, _clean(call.get("objections"))])
    checks = [
        ("refusal_or_cooling", REFUSAL_RE),
        ("next_year_interest", NEXT_YEAR_RE),
        ("payment_service", PAYMENT_RE),
        ("price_or_payment", PRICE_RE),
        ("schedule_or_format_constraint", SCHEDULE_RE),
        ("trust_or_quality_question", TRUST_RE),
        ("product_interest", PRODUCT_RE),
    ]
    for label, pattern in checks:
        match = pattern.search(text)
        if match:
            return label, _clip(_sentence_around(text, match.start()), 300)
    if _clean(call.get("call_type")) in {"service_call", "existing_client_progress"}:
        return "service_or_existing_client", _clip(history, 300)
    return "unknown", _clip(question or history, 300)


def infer_hidden_sales_stage(chain: dict[str, Any], call: dict[str, Any], signal: str) -> str:
    outcome = str(chain.get("final_outcome_label") or "")
    use_case = str(chain.get("extraction_use_case") or "")
    call_type = str(call.get("call_type") or "")
    position = _touch_position(call, chain)
    total = max(_as_int(chain.get("touch_count")), 1)
    if call_type in {"service_call", "existing_client_progress"} or signal in {"payment_service", "service_or_existing_client"}:
        return "service_retention_or_expansion"
    if outcome in {"won_paid_or_active", "existing_client_service_not_new_sale"}:
        return "success_path_validation"
    if outcome in {"lost_or_refused", "closed_lost_valid", "churn_or_refused_after_activity"}:
        return "loss_or_churn_path"
    if use_case == "reactivation_revenue":
        return "reactivation_after_lost_deal"
    if position <= 1:
        return "first_contact_or_need_discovery"
    if position >= total - 1:
        return "decision_or_follow_up"
    if signal in {"price_or_payment", "schedule_or_format_constraint", "trust_or_quality_question"}:
        return "objection_handling"
    return "nurture_and_qualification"


def score_manager_response(
    chain: dict[str, Any],
    call: dict[str, Any],
    signal: str,
    question: str,
    answer: str,
) -> tuple[int, str, list[str]]:
    score = 45
    reasons: list[str] = []
    if len(answer) >= 80:
        score += 10
        reasons.append("substantive_answer")
    if _clean(call.get("next_step")):
        score += 15
        reasons.append("next_step_fixed")
    if signal in {"price_or_payment", "schedule_or_format_constraint", "trust_or_quality_question"}:
        if _signal_addressed(signal, answer):
            score += 15
            reasons.append("addresses_customer_signal")
        else:
            score -= 15
            reasons.append("signal_not_clearly_addressed")
    if _clean(call.get("lead_priority")).lower() in {"hot", "warm"}:
        score += 5
        reasons.append("warm_or_hot_priority")
    if str(chain.get("final_outcome_label") or "") in {"won_paid_or_active", "reopen_or_follow_up_opportunity"}:
        score += 5
        reasons.append("positive_or_reactivation_outcome")
    if not question or len(question) < 20:
        score -= 8
        reasons.append("weak_customer_need_extraction")
    if not answer or len(answer) < 30:
        score -= 15
        reasons.append("weak_manager_answer_extraction")
    score = max(0, min(100, score))
    if score >= 75:
        band = "high"
    elif score >= 55:
        band = "medium"
    else:
        band = "low"
    return score, band, reasons


def infer_missed_opportunity(chain: dict[str, Any], call: dict[str, Any], signal: str, answer: str, quality_score: int) -> tuple[bool, str]:
    use_case = str(chain.get("extraction_use_case") or "")
    if use_case == "reactivation_revenue" and quality_score < 70:
        return True, "Клиент находится в reactivation-сегменте, но реакция менеджера требует проверки/усиления."
    if signal in {"price_or_payment", "schedule_or_format_constraint", "trust_or_quality_question"} and not _signal_addressed(signal, answer):
        return True, f"Сигнал клиента `{signal}` не был явно закрыт в ответе менеджера."
    if _clean(call.get("lead_priority")).lower() in {"hot", "warm"} and not _clean(call.get("next_step")):
        return True, "Теплый/горячий контакт без зафиксированного следующего шага."
    return False, ""


def ideal_reaction_for_signal(signal: str, chain: dict[str, Any], call: dict[str, Any]) -> tuple[str, str]:
    product_hint = _clean(call.get("products")) or _clean(chain.get("products_top")) or "подходящий курс"
    if signal == "price_or_payment":
        return (
            "Сначала подтвердить важность бюджета, затем разложить ценность, варианты оплаты и зафиксировать конкретный следующий шаг.",
            f"Понимаю, что стоимость важна. Давайте подберем формат по бюджету: {product_hint}, варианты оплаты/скидки и срок, до которого можно закрепить место.",
        )
    if signal == "schedule_or_format_constraint":
        return (
            "Уточнить реальные ограничения по дням/формату и предложить 2-3 конкретные альтернативы вместо общего описания.",
            f"Чтобы не перегружать ребенка, давайте зафиксируем удобные дни и формат. Я подберу варианты по {product_hint} и вернусь с конкретным расписанием.",
        )
    if signal == "trust_or_quality_question":
        return (
            "Дать доказательства качества: преподаватель, результаты, отзывы, обратная связь по ученику; затем согласовать следующий контакт.",
            "Понимаю, что важно видеть результат. Я пришлю отзыв преподавателя, программу и примеры результатов, а потом отдельно обсудим, подходит ли это ребенку.",
        )
    if signal == "payment_service":
        return (
            "Быстро закрыть сервисный вопрос и не продавать поверх нерешенной оплаты; после решения мягко спросить про продолжение.",
            "Сначала решу вопрос с оплатой/договором и подтвержу письмом. После этого, если удобно, отдельно обсудим продолжение обучения.",
        )
    if signal == "refusal_or_cooling":
        return (
            "Не давить; уточнить причину отказа, сохранить разрешение на будущий контакт и предложить более релевантный формат позже.",
            "Понял, не буду настаивать. Подскажите, причина больше в расписании, цене или формате? Я зафиксирую и вернусь только если появится подходящий вариант.",
        )
    if signal == "next_year_interest":
        return (
            "Собрать предметы/цели/ограничения на следующий год и поставить точный follow-up на дату появления расписания.",
            "Зафиксирую интерес на следующий учебный год. Какие предметы и дни вам принципиальны? Когда расписание будет готово, вернусь с конкретными вариантами.",
        )
    return (
        "Сформулировать потребность клиента своими словами, предложить следующий конкретный шаг и зафиксировать дату/канал.",
        "Правильно понимаю вашу задачу: нужно подобрать подходящий формат обучения. Я уточню детали и вернусь с конкретным предложением и сроками.",
    )


def split_sentences(text: str) -> list[str]:
    cleaned = _clean(text)
    if not cleaned:
        return []
    rough = re.split(r"(?<=[.!?…])\s+|\n+", cleaned)
    sentences: list[str] = []
    for item in rough:
        item = item.strip()
        if len(item) <= 260:
            if item:
                sentences.append(item)
            continue
        words = item.split()
        chunk: list[str] = []
        for word in words:
            chunk.append(word)
            if len(" ".join(chunk)) >= 220:
                sentences.append(" ".join(chunk))
                chunk = []
        if chunk:
            sentences.append(" ".join(chunk))
    return sentences


def build_summary(
    config: PilotExtractionConfig,
    pilot_rows: list[dict[str, Any]],
    selected_calls: list[dict[str, Any]],
    moments: list[dict[str, Any]],
    client_sequences: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness_root": str(config.readiness_root.resolve()),
        "outcome_root": str(config.outcome_root.resolve()),
        "config": {
            "max_clients": config.max_clients,
            "max_calls_per_client": config.max_calls_per_client,
            "transcript_chars_for_llm": config.transcript_chars_for_llm,
        },
        "totals": {
            "pilot_clients": len(pilot_rows),
            "selected_calls": len(selected_calls),
            "sales_moments": len(moments),
            "client_sequences": len(client_sequences),
            "unique_phones_in_moments": len({row["phone"] for row in moments if row.get("phone")}),
        },
        "moment_counts": {
            "by_use_case": dict(Counter(row["extraction_use_case"] for row in moments).most_common()),
            "by_outcome": dict(Counter(row["final_outcome_label"] for row in moments).most_common()),
            "by_signal": dict(Counter(row["customer_signal_label"] for row in moments).most_common()),
            "by_hidden_stage": dict(Counter(row["hidden_sales_stage"] for row in moments).most_common()),
            "by_quality_band": dict(Counter(row["manager_response_quality_band"] for row in moments).most_common()),
            "top_managers": dict(Counter(row["manager_name"] for row in moments if row.get("manager_name")).most_common(30)),
        },
        "notes": [
            "This is deterministic pilot extraction. It is suitable for triage and LLM input preparation, not final expert scoring.",
            "llm_sales_moment_input.jsonl contains clipped transcripts and deterministic seeds for GPT-based refinement.",
            "One primary sales moment is extracted per selected contentful call.",
        ],
    }


def write_outputs(
    out_root: Path,
    summary: dict[str, Any],
    moments: list[dict[str, Any]],
    client_sequences: list[dict[str, Any]],
    llm_inputs: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "sales_moments_csv": out_root / "sales_moments.csv",
        "client_stage_sequences_csv": out_root / "client_stage_sequences.csv",
        "llm_input_jsonl": out_root / "llm_sales_moment_input.jsonl",
        "summary_json": out_root / "summary.json",
    }
    _write_csv(paths["sales_moments_csv"], moments)
    _write_csv(paths["client_stage_sequences_csv"], client_sequences)
    with paths["llm_input_jsonl"].open("w", encoding="utf-8") as fh:
        for item in llm_inputs:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    xlsx_path = out_root / "pilot_sales_moments.xlsx"
    try:
        _write_xlsx(xlsx_path, summary, moments, client_sequences)
        paths["xlsx"] = xlsx_path
    except Exception as exc:  # noqa: BLE001
        (out_root / "xlsx_error.txt").write_text(str(exc), encoding="utf-8")
    return paths


def _write_xlsx(path: Path, summary: dict[str, Any], moments: list[dict[str, Any]], client_sequences: list[dict[str, Any]]) -> None:
    import pandas as pd

    summary_rows: list[dict[str, Any]] = []
    for key, value in summary.get("totals", {}).items():
        summary_rows.append({"metric": key, "value": value})
    for key, values in summary.get("moment_counts", {}).items():
        for label, count in values.items():
            summary_rows.append({"metric": f"{key}:{label}", "value": count})
    for note in summary.get("notes", []):
        summary_rows.append({"metric": "note", "value": note})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(moments).to_excel(writer, sheet_name="Sales Moments", index=False)
        pd.DataFrame(client_sequences).to_excel(writer, sheet_name="Client Sequences", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column_cells in sheet.columns:
                max_len = 0
                col = column_cells[0].column_letter
                for cell in column_cells[:200]:
                    max_len = max(max_len, len(str(cell.value or "")))
                sheet.column_dimensions[col].width = min(max(max_len + 2, 10), 64)


def _call_selection_score(row: dict[str, Any], chain: dict[str, Any]) -> int:
    score = 0
    call_type = str(row.get("call_type") or "")
    score += {"sales_call": 50, "existing_client_progress": 35, "service_call": 30, "technical_call": 10}.get(call_type, 0)
    score += 20 if _clean(row.get("next_step")) else 0
    score += 15 if _clean(row.get("objections")) else 0
    score += 10 if _clean(row.get("lead_priority")).lower() in {"hot", "warm"} else 0
    score += min(len(_clean(row.get("history_summary"))) // 80, 12)
    if str(chain.get("extraction_use_case") or "") == "reactivation_revenue":
        score += 10
    return score


def _signal_addressed(signal: str, answer: str) -> bool:
    checks = {
        "price_or_payment": PRICE_RE,
        "schedule_or_format_constraint": SCHEDULE_RE,
        "trust_or_quality_question": TRUST_RE,
        "payment_service": PAYMENT_RE,
    }
    pattern = checks.get(signal)
    return bool(pattern and pattern.search(answer or ""))


def _best_sentence(sentences: list[str], pattern: re.Pattern[str]) -> str:
    scored: list[tuple[int, str]] = []
    for sentence in sentences:
        score = 0
        if pattern.search(sentence):
            score += 100
        if PRODUCT_RE.search(sentence):
            score += 20
        if 40 <= len(sentence) <= 280:
            score += 10
        if score:
            scored.append((score, sentence))
    if not scored:
        return ""
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return _clip(scored[0][1], 360)


def _sentence_around(text: str, pos: int) -> str:
    start = max(text.rfind(".", 0, pos), text.rfind("\n", 0, pos), text.rfind(";", 0, pos))
    end_candidates = [idx for idx in (text.find(".", pos), text.find("\n", pos), text.find(";", pos)) if idx != -1]
    end = min(end_candidates) if end_candidates else min(len(text), pos + 240)
    return text[start + 1 : end].strip()


def _touch_position(call: dict[str, Any], chain: dict[str, Any]) -> int:
    raw_first = str(chain.get("first_seen_at") or "")
    raw_started = str(call.get("started_at") or "")
    if raw_first and raw_started[:10] == raw_first[:10]:
        return 1
    return max(1, _as_int(call.get("_sequence_position")) or 1)


def _chain_for_call(call: dict[str, Any], chains: list[dict[str, Any]]) -> dict[str, Any]:
    phone = str(call.get("phone") or "")
    for chain in chains:
        if str(chain.get("phone") or "") == phone:
            return chain
    return {}


def _is_contentful(row: dict[str, Any]) -> bool:
    if str(row.get("contentful") or "").strip().lower() in CONTENTFUL_FALSE_VALUES:
        return False
    return str(row.get("call_type") or "").strip().lower() != "non_conversation"


def _group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row.get(key) or "")].append(row)
    for items in result.values():
        items.sort(key=lambda row: str(row.get("started_at") or ""))
        for idx, row in enumerate(items, start=1):
            row["_sequence_position"] = str(idx)
    return result


def _safe_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    cleaned = _clean(value)
    return [cleaned] if cleaned else []


def _split_joined(value: Any) -> list[str]:
    return [_clean(item) for item in re.split(r"\s*\|\s*", _clean(value)) if _clean(item)]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _clip(value: Any, limit: int) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _middle_clip(value: Any, limit: int) -> str:
    text = _clean(value)
    if len(text) <= limit:
        return text
    head = int(limit * 0.65)
    tail = limit - head - 25
    return text[:head].rstrip() + "\n...[middle clipped]...\n" + text[-tail:].lstrip()


def _as_int(value: Any) -> int:
    try:
        return int(float(str(value or 0).strip()))
    except (TypeError, ValueError):
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic pilot sales moments for knowledge-base extraction.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--readiness-root", default="stable_runtime/insight_readiness_report_20260507")
    parser.add_argument("--outcome-root", default="stable_runtime/outcome_linkage_report_20260507")
    parser.add_argument("--out-root", default="stable_runtime/pilot_sales_moments_20260507")
    parser.add_argument("--max-clients", type=int, default=500)
    parser.add_argument("--max-calls-per-client", type=int, default=6)
    parser.add_argument("--transcript-chars-for-llm", type=int, default=9000)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PilotExtractionConfig:
    project_root = Path(args.project_root).expanduser().resolve()
    return PilotExtractionConfig(
        project_root=project_root,
        readiness_root=(project_root / args.readiness_root).resolve(),
        outcome_root=(project_root / args.outcome_root).resolve(),
        out_root=(project_root / args.out_root).resolve(),
        max_clients=int(args.max_clients),
        max_calls_per_client=int(args.max_calls_per_client),
        transcript_chars_for_llm=int(args.transcript_chars_for_llm),
    )


__all__ = [
    "PilotExtractionConfig",
    "build_client_sequence_row",
    "build_llm_input",
    "build_pilot_sales_moments",
    "choose_customer_question_or_need",
    "choose_manager_answer",
    "config_from_args",
    "extract_sales_moment",
    "ideal_reaction_for_signal",
    "infer_customer_signal",
    "infer_hidden_sales_stage",
    "parse_args",
    "parse_role_blocks",
    "score_manager_response",
    "select_calls_for_client",
    "split_sentences",
]
