from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mango_mvp.question_catalog.contracts import (
    ANSWER_STATUS_DRAFT_NEEDS_REVIEW,
    ANSWER_STATUS_MANAGER_ONLY,
    ANSWER_STATUS_NEEDS_ROP_ANSWER,
    ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
    BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    BOT_PERMISSION_DRAFT_ONLY,
    BOT_PERMISSION_MANAGER_ONLY,
    FACT_TYPE_DISCOUNT,
    FACT_TYPE_DOCUMENTS,
    FACT_TYPE_INSTALLMENT,
    FACT_TYPE_LOCATION,
    FACT_TYPE_PRICE,
    FACT_TYPE_PROGRAM,
    FACT_TYPE_SCHEDULE,
    FACT_TYPE_TRIAL,
    normalize_key,
)


QUESTION_MARKERS = (
    "?",
    "подскаж",
    "сколько",
    "стоим",
    "цена",
    "как ",
    "можно",
    "какой",
    "какая",
    "какие",
    "когда",
    "где",
    "есть ли",
    "будет ли",
    "интерес",
    "хочу",
    "пришл",
    "нужен",
    "нужна",
    "нужно",
    "запис",
    "распис",
    "адрес",
    "очно",
    "онлайн",
)
SERVICE_NOISE_MARKERS = (
    "отписаться от рассылки",
    "письмо сгенерировано автоматически",
    "privacy policy",
    "unsubscribe",
    "mail delivery",
    "useragent",
    "mail_link_tracker",
    "geteml.com",
    "bitrix/admin",
    "для добавления в стоп-лист",
    "для просмотра сессии",
    "посетитель -",
    "сессия -",
    "поисковик -",
    "как договаривались",
)
SUBJECT_PATTERNS = (
    ("math", "математика", re.compile(r"\bматем|профил|алгебр|геометр", re.I)),
    ("physics", "физика", re.compile(r"\bфизик", re.I)),
    ("informatics", "информатика", re.compile(r"\bинформат|программирован|python|питон|кодинг", re.I)),
    ("chemistry", "химия", re.compile(r"\bхими", re.I)),
    ("russian", "русский язык", re.compile(r"\bрусск", re.I)),
    ("english", "английский язык", re.compile(r"\bангл", re.I)),
    ("social", "обществознание", re.compile(r"\bобществ", re.I)),
    ("biology", "биология", re.compile(r"\bбиолог", re.I)),
    ("literature", "литература", re.compile(r"\bлитератур", re.I)),
)
PRODUCT_PATTERNS = (
    ("ege", "ЕГЭ", re.compile(r"\bегэ\b", re.I)),
    ("oge", "ОГЭ", re.compile(r"\bогэ\b", re.I)),
    ("olympiad", "олимпиады", re.compile(r"\bолимпиад", re.I)),
    ("summer_school", "летняя школа", re.compile(r"\bлвш|летн\w+\s+(?:очная|выездная|школ)", re.I)),
    ("zvsh", "ЗВШ", re.compile(r"\bзвш|заочн", re.I)),
    ("ovsh", "ОВШ", re.compile(r"\bовш|очн\w+\s+вечерн", re.I)),
    ("regular_course", "регулярный курс", re.compile(r"\bкурс|занят", re.I)),
    ("trial", "пробное занятие", re.compile(r"\bпробн", re.I)),
)
INTENT_PATTERNS = (
    ("price", "стоимость", (FACT_TYPE_PRICE,), re.compile(r"\bсколько|стоимост|цена|стоит|оплат|абонем", re.I)),
    ("schedule", "расписание", (FACT_TYPE_SCHEDULE,), re.compile(r"\bраспис|когда|время|дни|час|график", re.I)),
    ("location", "адрес / очная площадка", (FACT_TYPE_LOCATION,), re.compile(r"\bгде|адрес|очно|филиал|метро|площадк", re.I)),
    ("format", "формат обучения", (FACT_TYPE_PROGRAM,), re.compile(r"\bонлайн|очно|формат|дистанц", re.I)),
    ("discount", "скидки", (FACT_TYPE_DISCOUNT,), re.compile(r"\bскид|акци|льгот|многодет|приведи", re.I)),
    ("installment", "рассрочка", (FACT_TYPE_INSTALLMENT,), re.compile(r"\bрассроч|долями|сплит|частями|кредит", re.I)),
    ("trial", "пробное занятие", (FACT_TYPE_TRIAL, FACT_TYPE_SCHEDULE), re.compile(r"\bпробн", re.I)),
    ("technical_access", "доступ / технический вопрос", (), re.compile(r"\bдоступ|ссылк|платформ|личн\w+\s+кабинет|не\s+открыва|не\s+заход|логин|парол", re.I)),
    ("service_feedback", "обратная связь по обучению", (), re.compile(r"\bжалоб|обратн\w+\s+связ|домашн|дз\b|пропуск|отработк|перенос|ошибк|результат|успеваем", re.I)),
    ("tax_deduction", "налоговый вычет / справки", (FACT_TYPE_DOCUMENTS,), re.compile(r"\bналогов\w+\s+вычет|вычет|справк|чек|сертификат", re.I)),
    ("documents", "документы / договор", (FACT_TYPE_DOCUMENTS,), re.compile(r"\bдокумент|договор|оферт|справк|чек|сертификат", re.I)),
    ("program", "программа курса", (FACT_TYPE_PROGRAM,), re.compile(r"\bпрограмм|темы|что проход|курс|предмет", re.I)),
    ("teacher", "преподаватель", (), re.compile(r"\bпреподав|учител|педагог", re.I)),
    ("level_fit", "подходит ли уровень", (), re.compile(r"\bуровень|подойдет|подойд[её]т|сильн|слаб", re.I)),
    ("enrollment", "запись на обучение", (FACT_TYPE_SCHEDULE,), re.compile(r"\bзапис|мест[ао]|набор|попасть", re.I)),
    ("callback", "обратная связь", (), re.compile(r"\bперезвон|связаться|позвон|напишите|ответьте", re.I)),
)


@dataclass(frozen=True)
class QuestionMetadata:
    intent: str
    intent_label: str
    product: str | None
    product_key: str | None
    grade: str | None
    subject: str | None
    subject_key: str | None
    format: str | None
    dynamic_fact_types: tuple[str, ...]
    class_key: str
    canonical_question: str
    narrow_scope: str
    exclusions: str
    answer_status: str
    bot_permission: str
    manager_handoff_reason: str | None
    rop_review_priority: str
    required_fact_keys: tuple[str, ...]
    fact_freshness_policy: str | None
    fallback_when_fact_missing: str | None


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return "" if text.lower() in {"nan", "none", "null"} else text


def is_question_like(value: Any) -> bool:
    text = clean_text(value).lower()
    if len(text) < 5:
        return False
    if any(marker in text for marker in SERVICE_NOISE_MARKERS):
        return False
    words = re.findall(r"[a-zа-яё0-9]+", text, re.I)
    if len(words) < 3:
        return False
    if re.fullmatch(r"да[,.!\s]+можно[.!]?", text):
        return False
    return any(marker in text for marker in QUESTION_MARKERS)


def split_candidate_questions(value: Any, *, max_parts: int = 3) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    chunks = re.split(r"(?<=[?!.])\s+|\n+", text)
    candidates: list[str] = []
    for chunk in chunks:
        part = clean_text(chunk)
        if is_question_like(part):
            candidates.append(part[:700])
        if len(candidates) >= max_parts:
            break
    if not candidates and is_question_like(text):
        candidates.append(text[:700])
    return candidates


def infer_question_metadata(value: Any, *, fallback_signal: str | None = None) -> QuestionMetadata:
    text = clean_text(value)
    intent_key, intent_label, fact_types = _infer_intent(text, fallback_signal=fallback_signal)
    product_key, product = _infer_product(text)
    subject_key, subject = _infer_subject(text)
    grade = _infer_grade(text)
    fmt = _infer_format(text)
    fact_types = tuple(dict.fromkeys(fact_types))
    required_fact_keys = tuple(f"{fact_type}.current" for fact_type in fact_types)
    answer_status = ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT if fact_types else ANSWER_STATUS_DRAFT_NEEDS_REVIEW
    bot_permission = BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK if fact_types else BOT_PERMISSION_DRAFT_ONLY
    manager_reason = None
    priority = "medium"
    if intent_key in {"price", "discount", "installment", "schedule", "location", "enrollment"}:
        priority = "high"
        manager_reason = "Нужна проверка актуальных фактов перед клиентским ответом."
    if intent_key in {"documents", "teacher", "level_fit", "technical_access", "service_feedback", "tax_deduction"}:
        bot_permission = BOT_PERMISSION_MANAGER_ONLY if intent_key in {"documents", "teacher", "technical_access", "service_feedback", "tax_deduction"} else bot_permission
        answer_status = ANSWER_STATUS_MANAGER_ONLY if intent_key in {"documents", "teacher", "technical_access", "service_feedback", "tax_deduction"} else answer_status
        manager_reason = "Нужен менеджер: вопрос зависит от персонального контекста или документов."
    class_parts = [
        f"intent={intent_key}",
        f"product={product_key or 'any'}",
        f"subject={subject_key or 'any'}",
        f"grade={grade or 'any'}",
        f"format={_format_key(fmt) if fmt else 'any'}",
    ]
    class_key = "|".join(class_parts)
    canonical = _canonical_question(intent_label, product, subject, grade, fmt)
    return QuestionMetadata(
        intent=intent_key,
        intent_label=intent_label,
        product=product,
        product_key=product_key,
        grade=grade,
        subject=subject,
        subject_key=subject_key,
        format=fmt,
        dynamic_fact_types=fact_types,
        class_key=class_key,
        canonical_question=canonical,
        narrow_scope=_narrow_scope(intent_label, product, subject, grade, fmt),
        exclusions="Не смешивать с другими предметами, классами, форматами и периодами обучения.",
        answer_status=answer_status,
        bot_permission=bot_permission,
        manager_handoff_reason=manager_reason,
        rop_review_priority=priority,
        required_fact_keys=required_fact_keys,
        fact_freshness_policy="Нужен свежий подтвержденный файл фактов перед ответом." if fact_types else None,
        fallback_when_fact_missing="Не называть конкретные условия, передать менеджеру." if fact_types else None,
    )


def classify_question(value: Any, *, fallback_signal: str | None = None) -> Mapping[str, Any]:
    metadata = infer_question_metadata(value, fallback_signal=fallback_signal)
    return metadata.__dict__


def _infer_intent(text: str, *, fallback_signal: str | None) -> tuple[str, str, tuple[str, ...]]:
    signal = clean_text(fallback_signal).lower()
    signal_map = {
        "price_question": ("price", "стоимость", (FACT_TYPE_PRICE,)),
        "price_objection": ("price", "стоимость", (FACT_TYPE_PRICE,)),
        "discount_or_installment_question": ("installment", "скидки / рассрочка", (FACT_TYPE_DISCOUNT, FACT_TYPE_INSTALLMENT)),
        "schedule_question": ("schedule", "расписание", (FACT_TYPE_SCHEDULE,)),
        "format_question_online_offline": ("format", "формат обучения", (FACT_TYPE_PROGRAM,)),
        "location_question": ("location", "адрес / очная площадка", (FACT_TYPE_LOCATION,)),
        "program_question": ("program", "программа курса", (FACT_TYPE_PROGRAM,)),
        "teacher_question": ("teacher", "преподаватель", ()),
        "level_fit_question": ("level_fit", "подходит ли уровень", ()),
        "payment_or_contract_service": ("documents", "оплата / договор / документы", (FACT_TYPE_DOCUMENTS,)),
        "technical_or_access_issue": ("technical_access", "доступ / технический вопрос", ()),
        "complaint_or_service_risk": ("service_feedback", "обратная связь по обучению", ()),
        "existing_client_progress": ("service_feedback", "обратная связь по обучению", ()),
        "callback_request": ("callback", "обратная связь", ()),
        "materials_request": ("program", "материалы / программа", (FACT_TYPE_PROGRAM,)),
    }
    if signal in signal_map:
        return signal_map[signal]
    for key, label, fact_types, pattern in INTENT_PATTERNS:
        if pattern.search(text):
            return key, label, tuple(fact_types)
    return "other", "общий вопрос", ()


def _infer_subject(text: str) -> tuple[str | None, str | None]:
    for key, label, pattern in SUBJECT_PATTERNS:
        if pattern.search(text):
            return key, label
    return None, None


def _infer_product(text: str) -> tuple[str | None, str | None]:
    for key, label, pattern in PRODUCT_PATTERNS:
        if pattern.search(text):
            return key, label
    return None, None


def _infer_grade(text: str) -> str | None:
    match = re.search(r"\b([1-9]|1[01])\s*(?:класс|кл\.?|класса|классе)\b", text, re.I)
    if match:
        return f"{match.group(1)} класс"
    match = re.search(r"\bдля\s+([1-9]|1[01])[-\s]?(?:го|ого|класса)\b", text, re.I)
    if match:
        return f"{match.group(1)} класс"
    if re.search(r"\b11\b.*\bегэ\b|\bегэ\b.*\b11\b", text, re.I):
        return "11 класс"
    if re.search(r"\b9\b.*\bогэ\b|\bогэ\b.*\b9\b", text, re.I):
        return "9 класс"
    return None


def _infer_format(text: str) -> str | None:
    has_online = bool(re.search(r"\bонлайн|дистанц", text, re.I))
    has_offline = bool(re.search(r"\bочно|очная|очный|офлайн", text, re.I))
    if has_online and has_offline:
        return "онлайн или очно"
    if has_online:
        return "онлайн"
    if has_offline:
        return "очно"
    return None


def _format_key(value: str | None) -> str:
    if value == "онлайн":
        return "online"
    if value == "очно":
        return "offline"
    if value == "онлайн или очно":
        return "online_or_offline"
    return normalize_key(value or "any", "format")


def _canonical_question(intent_label: str, product: str | None, subject: str | None, grade: str | None, fmt: str | None) -> str:
    parts = [intent_label]
    if product:
        parts.append(product)
    if subject:
        parts.append(subject)
    if grade:
        parts.append(grade)
    if fmt:
        parts.append(fmt)
    return " / ".join(parts)


def _narrow_scope(intent_label: str, product: str | None, subject: str | None, grade: str | None, fmt: str | None) -> str:
    details = []
    if product:
        details.append(f"продукт: {product}")
    if subject:
        details.append(f"предмет: {subject}")
    if grade:
        details.append(f"класс: {grade}")
    if fmt:
        details.append(f"формат: {fmt}")
    tail = "; ".join(details) if details else "без уточненного продукта, предмета и класса"
    return f"Вопрос клиента про {intent_label}; {tail}."
