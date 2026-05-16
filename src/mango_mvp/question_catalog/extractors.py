from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from mango_mvp.question_catalog.contracts import (
    ANSWER_STATUS_DRAFT_NEEDS_REVIEW,
    ANSWER_STATUS_MANAGER_ONLY,
    ANSWER_STATUS_NOT_CUSTOMER_QUESTION,
    ANSWER_STATUS_NOT_ENOUGH_CONTEXT,
    ANSWER_STATUS_NEEDS_ROP_ANSWER,
    ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
    BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    BOT_PERMISSION_DRAFT_ONLY,
    BOT_PERMISSION_MANAGER_ONLY,
    BOT_PERMISSION_NOT_ALLOWED,
    FACT_TYPE_DISCOUNT,
    FACT_TYPE_DOCUMENTS,
    FACT_TYPE_INSTALLMENT,
    FACT_TYPE_LOCATION,
    FACT_TYPE_PRICE,
    FACT_TYPE_PROGRAM,
    FACT_TYPE_SCHEDULE,
    FACT_TYPE_TRIAL,
    SOURCE_CALL,
    SOURCE_EMAIL,
    SOURCE_TELEGRAM,
    QuestionItem,
    stable_digest,
    stable_question_class_id,
)
from mango_mvp.question_catalog.classifier import classify_question
from mango_mvp.question_catalog.normalization import (
    clean_text,
    detect_noise_reason,
    is_question_like,
    split_candidate_questions,
)
from mango_mvp.question_catalog.safety import redact_public_text, redact_review_text


SINCE_DEFAULT = datetime(2025, 1, 1, tzinfo=timezone.utc)


def parse_datetime(value: Any) -> Optional[datetime]:
    text = clean_text(value)
    if not text:
        return None
    candidates = [text, text.replace(" ", "T")]
    for candidate in candidates:
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def extract_call_questions(
    reviews_csv: Path | str,
    *,
    tenant_id: str,
    since: datetime = SINCE_DEFAULT,
) -> tuple[list[QuestionItem], Mapping[str, Any]]:
    path = Path(reviews_csv)
    if not path.exists():
        return [], _missing_source(path, "call_enriched_reviews")
    rows = read_csv_rows(path)
    items: list[QuestionItem] = []
    skipped_before_since = 0
    skipped_empty = 0
    for row_index, row in enumerate(rows, start=2):
        occurred_at = parse_datetime(row.get("started_at") or row.get("call_started_at") or row.get("created_at"))
        if occurred_at and occurred_at < since:
            skipped_before_since += 1
            continue
        question_raw = clean_text(row.get("customer_question_sanitized") or row.get("customer_question") or row.get("customer_quote_sanitized") or row.get("customer_quote"))
        if not question_raw:
            skipped_empty += 1
            continue
        if detect_noise_reason(question_raw):
            skipped_empty += 1
            continue
        if not is_question_like(question_raw) and clean_text(row.get("llm_customer_signal_type")) not in _SIGNALS_ALLOWED_WITHOUT_MARKER:
            skipped_empty += 1
            continue
        answer_raw = row.get("bot_safe_answer") or row.get("ideal_answer_manager_sanitized") or row.get("manager_answer") or row.get("ideal_answer_example")
        source_id = clean_text(row.get("moment_id") or row.get("call_id") or row.get("recording_id") or f"row-{row_index}")
        source_ref = f"call:enriched_reviews.csv#moment={_short_hash(source_id)}"
        item = build_question_item(
            tenant_id=tenant_id,
            source_channel=SOURCE_CALL,
            source_ref=source_ref,
            question_raw=question_raw,
            manager_raw=answer_raw,
            occurred_at=occurred_at,
            fallback_signal=row.get("llm_customer_signal_type"),
            answer_source="sales_insight_knowledge_base",
            metadata={
                "row_index": row_index,
                "call_id": clean_text(row.get("call_id")),
                "recording_id": clean_text(row.get("recording_id")),
                "moment_id": clean_text(row.get("moment_id")),
                "source_row_index": row_index,
                "source_kind": "call",
                "source_table": "enriched_reviews.csv",
                "source_id_raw": source_id,
                "signal": clean_text(row.get("llm_customer_signal_type")),
                "stage": clean_text(row.get("llm_hidden_sales_stage")),
                "answer_pattern": clean_text(row.get("answer_pattern")),
                "bot_seed_status": clean_text(row.get("bot_seed_status")),
                "outcome_group": clean_text(row.get("outcome_group")),
            },
        )
        items.append(item)
    return items, {
        "source_id": "call_enriched_reviews",
        "path": str(path),
        "status": "processed",
        "rows_total": len(rows),
        "items_extracted": len(items),
        "skipped_before_since": skipped_before_since,
        "skipped_empty_or_not_question": skipped_empty,
        "period_from": since.isoformat(),
    }


def extract_telegram_questions(
    messages_jsonl: Path | str,
    *,
    tenant_id: str,
    since: datetime = SINCE_DEFAULT,
) -> tuple[list[QuestionItem], Mapping[str, Any]]:
    path = Path(messages_jsonl)
    if not path.exists():
        return [], _missing_source(path, "telegram_messages")
    messages = list(_read_jsonl(path))
    messages.sort(key=lambda row: (str(row.get("dialog_id")), str(row.get("date")), int(row.get("message_id") or 0)))
    outbound_by_dialog: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in messages:
        if row.get("out") is True:
            outbound_by_dialog[str(row.get("dialog_id"))].append(row)

    items: list[QuestionItem] = []
    skipped_before_since = 0
    skipped_not_question = 0
    skipped_non_user = 0
    skipped_noise = 0
    for row in messages:
        if row.get("out") is True:
            continue
        if row.get("peer_kind") != "user":
            skipped_non_user += 1
            continue
        occurred_at = parse_datetime(row.get("date"))
        if occurred_at and occurred_at < since:
            skipped_before_since += 1
            continue
        text = clean_text(row.get("text") or row.get("raw_text"))
        if detect_noise_reason(text):
            skipped_noise += 1
            continue
        candidates = split_candidate_questions(text, max_parts=2)
        if not candidates:
            skipped_not_question += 1
            continue
        manager_raw = _next_outbound_text(outbound_by_dialog.get(str(row.get("dialog_id")), []), row)
        for ordinal, candidate in enumerate(candidates, start=1):
            dialog_hash = _short_hash(str(row.get("dialog_id")))
            source_ref = f"telegram:local_vm_2024-04-01#dialog={dialog_hash};message={row.get('message_id')};part={ordinal}"
            item = build_question_item(
                tenant_id=tenant_id,
                source_channel=SOURCE_TELEGRAM,
                source_ref=source_ref,
                question_raw=candidate,
                manager_raw=manager_raw,
                occurred_at=occurred_at,
                answer_source="telegram_local_export" if manager_raw else None,
                metadata={
                    "dialog_hash": dialog_hash,
                    "message_id": row.get("message_id"),
                    "has_media": bool(row.get("has_media")),
                    "reply_to_msg_id": row.get("reply_to_msg_id"),
                },
            )
            items.append(item)
    return items, {
        "source_id": "telegram_messages",
        "path": str(path),
        "status": "processed",
        "rows_total": len(messages),
        "items_extracted": len(items),
        "skipped_before_since": skipped_before_since,
        "skipped_non_user": skipped_non_user,
        "skipped_not_question": skipped_not_question,
        "skipped_noise": skipped_noise,
        "period_from": since.isoformat(),
    }


def extract_mail_questions(
    mail_root: Path | str,
    *,
    tenant_id: str,
    since: datetime = SINCE_DEFAULT,
    max_chars_per_message: int = 5000,
) -> tuple[list[QuestionItem], Mapping[str, Any]]:
    root = Path(mail_root)
    if not root.exists():
        return [], _missing_source(root, "mail_archives")
    archive_paths = sorted(root.glob("**/mail_archive.sqlite"))
    items: list[QuestionItem] = []
    seen: set[tuple[str, str]] = set()
    archive_reports: list[dict[str, Any]] = []
    total_rows = 0
    skipped_before_since = 0
    skipped_sent = 0
    skipped_service_or_not_question = 0
    skipped_noise = 0
    for archive_path in archive_paths:
        archive_count = 0
        try:
            rows = _read_mail_messages(archive_path)
        except sqlite3.DatabaseError as exc:
            archive_reports.append(
                {
                    "path": str(archive_path),
                    "status": "blocked",
                    "reason": f"sqlite_error: {exc}",
                    "items_extracted": 0,
                }
            )
            continue
        total_rows += len(rows)
        for row in rows:
            occurred_at = parse_datetime(row.get("message_date_iso"))
            if occurred_at and occurred_at < since:
                skipped_before_since += 1
                continue
            direction = _mail_direction(archive_path, row)
            if direction != "inbound":
                skipped_sent += 1
                continue
            if row.get("message_kind") == "service":
                skipped_service_or_not_question += 1
                continue
            text_path = Path(str(row.get("extracted_text_path") or ""))
            text = _read_text_if_safe(text_path, max_chars=max_chars_per_message)
            subject = clean_text(row.get("subject"))
            combined = _preprocess_mail_text(f"{subject}. {text}".strip())
            if _outbound_notice_reason(
                subject=subject,
                text=combined,
                from_header=clean_text(row.get("from_header")),
                to_header=clean_text(row.get("to_header")),
            ):
                skipped_service_or_not_question += 1
                continue
            if detect_noise_reason(combined):
                skipped_noise += 1
                continue
            candidates = split_candidate_questions(combined, max_parts=3)
            if not candidates:
                skipped_service_or_not_question += 1
                continue
            for ordinal, candidate in enumerate(candidates, start=1):
                item_key = (str(row.get("sha256")), candidate.lower())
                if item_key in seen:
                    continue
                seen.add(item_key)
                source_ref = f"email:archive#message={_short_hash(str(row.get('sha256')))};part={ordinal}"
                item = build_question_item(
                    tenant_id=tenant_id,
                    source_channel=SOURCE_EMAIL,
                    source_ref=source_ref,
                    question_raw=candidate,
                    manager_raw=None,
                    occurred_at=occurred_at,
                    answer_source=None,
                    metadata={
                        "archive": str(archive_path.parent.name),
                        "subject_hash": _short_hash(subject),
                        "message_hash": _short_hash(str(row.get("sha256"))),
                    },
                )
                items.append(item)
                archive_count += 1
        archive_reports.append(
            {
                "path": str(archive_path),
                "status": "processed",
                "rows_total": len(rows),
                "items_extracted": archive_count,
            }
        )
    return items, {
        "source_id": "mail_archives",
        "path": str(root),
        "status": "processed",
        "archives_total": len(archive_paths),
        "rows_total": total_rows,
        "items_extracted": len(items),
        "skipped_before_since": skipped_before_since,
        "skipped_sent": skipped_sent,
        "skipped_service_or_not_question": skipped_service_or_not_question,
        "skipped_noise": skipped_noise,
        "archives": archive_reports,
        "period_from": since.isoformat(),
    }


def build_question_item(
    *,
    tenant_id: str,
    source_channel: str,
    source_ref: str,
    question_raw: Any,
    manager_raw: Any = None,
    occurred_at: Optional[datetime] = None,
    fallback_signal: str | None = None,
    answer_source: str | None = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> QuestionItem:
    classified = classify_question(
        question_raw,
        source=source_channel,
        metadata={**dict(metadata or {}), "fallback_signal": fallback_signal or ""},
        fallback_signal=fallback_signal,
    )
    params = classified.extracted_params
    required_facts = classified.required_facts
    legacy_fact_types = _legacy_dynamic_fact_types(required_facts)
    bot_permission = _legacy_bot_permission(classified.default_bot_permission)
    answer_status = _answer_status_for_classification(classified.theme_id, required_facts, bot_permission)
    customer_text, customer_flags = redact_public_text(question_raw)
    customer_review_text, customer_review_flags = redact_review_text(question_raw)
    manager_text = None
    manager_flags: tuple[str, ...] = ()
    if clean_text(manager_raw):
        manager_text, manager_flags = redact_public_text(manager_raw, max_chars=700)
    question_class_id = stable_question_class_id(tenant_id=tenant_id, class_key=classified.theme_id)
    safety_flags = tuple(dict.fromkeys((*customer_flags, *customer_review_flags, *manager_flags)))
    return QuestionItem(
        tenant_id=tenant_id,
        source_channel=source_channel,
        source_ref=source_ref,
        occurred_at=occurred_at,
        customer_text_redacted=customer_text,
        manager_text_redacted=manager_text,
        question_class_id=question_class_id,
        intent=classified.theme_id,
        product=_param_or_none(params.get("product")),
        grade=_param_or_none(params.get("grade")),
        subject=_param_or_none(params.get("subject")),
        format=_param_or_none(params.get("format")),
        price_related=classified.theme_id == "theme:001_pricing" or any(fact.startswith("prices.") for fact in required_facts),
        schedule_related=classified.theme_id == "theme:013_schedule" or any(fact.startswith("schedule.") for fact in required_facts),
        documents_related=_has_document_fact(required_facts),
        safety_flags=safety_flags,
        answer_evidence_status=answer_status if answer_source else ANSWER_STATUS_NEEDS_ROP_ANSWER,
        answer_source=answer_source,
        requires_dynamic_facts=bool(required_facts),
        dynamic_fact_types=legacy_fact_types,
        fact_freshness_required=_fact_freshness_policy(required_facts),
        metadata={
            **dict(metadata or {}),
            "theme_id": classified.theme_id,
            "theme_name": classified.theme_name,
            "business_block": classified.business_block,
            "extracted_params": dict(params),
            "confidence_hint": classified.confidence,
            "classification_method": classified.classification_method,
            "parent_question_class": classified.business_block,
            "question_subclass": classified.theme_name,
            "question_subclass_key": classified.theme_id,
            "canonical_question": classified.theme_name,
            "narrow_scope": f"Вопрос клиента отнесен к теме «{classified.theme_name}».",
            "exclusions": "Не смешивать с другими темами и персональным контекстом без проверки.",
            "answer_status": answer_status,
            "bot_permission": bot_permission,
            "manager_handoff_reason": _manager_reason_for_classification(classified.theme_id, bot_permission, required_facts),
            "required_fact_keys": required_facts,
            "fact_freshness_policy": _fact_freshness_policy(required_facts),
            "fallback_when_fact_missing": "Не называть конкретные условия, передать менеджеру." if required_facts else None,
            "customer_text_for_rop": customer_review_text,
        },
    )


def _param_or_none(value: Any) -> str | None:
    text = clean_text(value)
    return None if text in {"", "не_указано", "нейтральный", "низкая"} else text


def _legacy_bot_permission(permission: str) -> str:
    if permission == "answer_after_fact_check":
        return BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK
    if permission == "manager_only":
        return BOT_PERMISSION_MANAGER_ONLY
    if permission == "bot_self":
        return BOT_PERMISSION_DRAFT_ONLY
    return BOT_PERMISSION_DRAFT_ONLY


def _answer_status_for_classification(theme_id: str, required_facts: Sequence[str], bot_permission: str) -> str:
    if theme_id == "service:S1_non_question":
        return ANSWER_STATUS_NOT_CUSTOMER_QUESTION
    if theme_id == "service:S2_unclear":
        return ANSWER_STATUS_NOT_ENOUGH_CONTEXT
    if bot_permission == BOT_PERMISSION_MANAGER_ONLY:
        return ANSWER_STATUS_MANAGER_ONLY
    if required_facts:
        return ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT
    return ANSWER_STATUS_DRAFT_NEEDS_REVIEW


def _manager_reason_for_classification(theme_id: str, bot_permission: str, required_facts: Sequence[str]) -> str | None:
    if theme_id.startswith("service:"):
        return "Служебная категория: не отправлять автономный содержательный ответ."
    if bot_permission == BOT_PERMISSION_MANAGER_ONLY:
        return "Нужен менеджер: вопрос зависит от персонального контекста или ручного решения."
    if required_facts:
        return "Нужна проверка актуальных фактов перед клиентским ответом."
    return None


def _fact_freshness_policy(required_facts: Sequence[str]) -> str | None:
    return "Нужны актуальные факты из v2 fact-source registry." if required_facts else None


def _has_document_fact(required_facts: Sequence[str]) -> bool:
    markers = ("document", "contract", "license", "matkap", "tax", "payment", "receipt")
    return any(any(marker in fact for marker in markers) for fact in required_facts)


def _legacy_dynamic_fact_types(required_facts: Sequence[str]) -> tuple[str, ...]:
    """Keep v1 QuestionItem.dynamic_fact_types compact while v2 metadata stores exact fact keys."""
    result: list[str] = []
    for fact in required_facts:
        lowered = fact.lower()
        if lowered.startswith("prices.") or "price" in lowered:
            result.append(FACT_TYPE_PRICE)
        elif lowered.startswith("schedule.") or "availability" in lowered:
            result.append(FACT_TYPE_SCHEDULE)
        elif lowered.startswith("addresses.") or "address" in lowered or "location" in lowered:
            result.append(FACT_TYPE_LOCATION)
        elif "discount" in lowered:
            result.append(FACT_TYPE_DISCOUNT)
        elif "installment" in lowered:
            result.append(FACT_TYPE_INSTALLMENT)
        elif "trial" in lowered:
            result.append(FACT_TYPE_TRIAL)
        elif any(marker in lowered for marker in ("document", "contract", "license", "matkap", "tax", "payment", "receipt")):
            result.append(FACT_TYPE_DOCUMENTS)
        elif "program" in lowered:
            result.append(FACT_TYPE_PROGRAM)
    return tuple(dict.fromkeys(result))


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _next_outbound_text(outbound_rows: Sequence[dict[str, Any]], inbound_row: Mapping[str, Any]) -> str:
    inbound_time = parse_datetime(inbound_row.get("date"))
    inbound_id = int(inbound_row.get("message_id") or 0)
    if not inbound_time:
        return ""
    for row in outbound_rows:
        row_time = parse_datetime(row.get("date"))
        row_id = int(row.get("message_id") or 0)
        if row_time and row_time >= inbound_time and row_id > inbound_id:
            text = clean_text(row.get("text") or row.get("raw_text"))
            if text:
                return text[:1200]
    return ""


def _read_mail_messages(db_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT sha256, message_date_iso, subject, from_header, to_header, mailbox,
                   mailbox_raw, message_kind, extracted_text_path, extracted_text_chars
            FROM messages
            ORDER BY message_date_iso, sha256
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _mail_direction(archive_path: Path, row: Mapping[str, Any]) -> str:
    blob = " ".join(str(part).casefold() for part in (*archive_path.parts, row.get("mailbox"), row.get("mailbox_raw")))
    if "sent" in blob or "отправ" in blob:
        return "outbound"
    return "inbound"


def _read_text_if_safe(path: Path, *, max_chars: int) -> str:
    if not path or not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def _preprocess_mail_text(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = clean_text(raw_line)
        lower = line.casefold()
        if not line:
            continue
        if line.startswith(">"):
            continue
        if lower.startswith(("от:", "from:", "кому:", "to:", "cc:", "subject:", "тема:", "дата:", "sent:")):
            continue
        if lower.startswith(("от кого:", "кому отправлено:", "-------- пересылаемое", "пересылаемое сообщение")):
            continue
        if any(
            marker in lower
            for marker in (
                "отправлено из mail",
                "отправлено из почты mail",
                "пересылаемое сообщение",
                "unsubscribe",
                "отписаться от рассылки",
                "mail_link_tracker",
                "geteml.com",
                "bitrix/admin",
                "для добавления в стоп-лист",
                "для просмотра сессии",
                "если у вас остались вопросы",
                "вы можете задать их",
                "написав в telegram",
                "написав в whatsapp",
                "с уважением",
                "команда фотон",
                "nbsp",
                "zwnj",
                "compose",
                "mime-version",
                "content-type",
                "vlagere",
                "заявка в обработке",
                "статус не менялся",
                "сутокздравствуйте",
                "недельздравствуйте",
                "пересланное письмо",
                "все пришло, спасибо",
                "спасибо, пришла смс",
                "спасибо большое",
                "useragent",
                "посетитель -",
                "сессия -",
                "поисковик -",
                "будьте в курсе",
                "продуктовую новинку",
                "tel:",
                "mailto:",
                "utm_",
                "http://",
                "https://",
                "http :",
                "https :",
                "ticket?",
                "1-ofd",
                "consumer.",
                "счёт:",
                "счет:",
                "инн:",
                "кпп:",
                "окпо",
                "огрн",
                "swift",
                "почтовый адрес банка",
                "идентификатор участника эдо",
                "======",
            )
        ):
            continue
        lines.append(line)
    text = " ".join(lines)
    if "--- part ---" in text:
        text = text.split("--- part ---", 1)[0]
    return clean_text(text)


def _outbound_notice_reason(*, subject: str, text: str, from_header: str, to_header: str) -> str | None:
    blob = clean_text(f"{subject}. {text}").casefold()
    from_lower = from_header.casefold()
    tenant_sender = any(marker in from_lower for marker in ("@kmipt.ru", "@cdpofoton.ru", "@foton", "фотон", "кмипт"))
    has_direct_customer_question = bool(
        re.search(
            r"\b(?:подскажите|уточните|можно\s+ли|скажите|сколько|как\s+мне|что\s+нужно|хотел[аи]\s+бы|прошу|пожалуйста\s+помогите)\b",
            blob,
            re.I,
        )
    )
    notice_markers = (
        r"ваше\s+расписание\s+занятий",
        r"вы\s+записаны\s+на",
        r"перенос\s+заняти[йя]",
        r"заняти[ея]\s+переносятся",
        r"чек\s+по\s+оплате",
        r"оплата\s+принята",
        r"участие\s+в\s+пробном\s+(?:огэ|егэ)",
        r"ознакомиться\s+с\s+форматом\s+экзамена",
        r"обучающие\s+курсы.+переносятся",
        r"подготовительные\s+курсы.+учебн\w+\s+год",
        r"напоминаем,\s+что\s+заняти",
        r"изменени[ея]\s+в\s+расписании",
        r"уведомление\s+о\s+закрытии\s+группы",
        r"вы\s+записаны\s+онлайн",
    )
    if any(re.search(marker, blob, re.I) for marker in notice_markers) and (tenant_sender or not has_direct_customer_question):
        return "outbound_notice_or_receipt"
    return None


def _short_hash(value: str, *, length: int = 12) -> str:
    return stable_digest({"value": value})[:length]


def _missing_source(path: Path, source_id: str) -> Mapping[str, Any]:
    return {
        "source_id": source_id,
        "path": str(path),
        "status": "missing",
        "items_extracted": 0,
    }


_SIGNALS_ALLOWED_WITHOUT_MARKER = {
    "price_question",
    "price_objection",
    "discount_or_installment_question",
    "schedule_question",
    "format_question_online_offline",
    "location_question",
    "program_question",
    "teacher_question",
    "level_fit_question",
    "payment_or_contract_service",
    "materials_request",
}
