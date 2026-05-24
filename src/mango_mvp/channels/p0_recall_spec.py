from __future__ import annotations

import re
from typing import Sequence


P0_RECALL_SPEC_SCHEMA_VERSION = "p0_recall_spec_v1_2026_05_24"

# This module is the shared P0 recall source for runtime guards, tests and KB
# trigger checks. Keep hard signals conservative: false negatives are more
# dangerous than false positives, but benign phrases below must stay non-P0.
REFUND_RE = re.compile(
    r"\bвозв?рат(?!\w*\s+к\s+(?:тем|урок|материал|заняти))\w*"
    r"|\bвозвращ\w*\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
    r"|\bверн\w*(?:\s+мне|\s+нам|\s+пожалуйста)?\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
    r"|\bверн\w*(?:\s+мне|\s+нам|\s+пожалуйста)?\s+(?:одну|один|лишн\w*|повторн\w*|дублирующ\w*)"
    r"|\bвернуть\s+оплат\w*"
    r"|\bхочу\s+деньги\s+назад\b"
    r"|\bденьги\s+назад\b"
    r"|\bрасторг\w*\s+договор"
    r"|\bотказ\w*\s+от\s+обучен"
    r"|\bзабрать\s+деньги",
    re.I,
)

LEGAL_RE = re.compile(
    r"\bсуд\b|\bиск\b|претензи\w*|досудеб|роспотребнадзор|прокуратур|адвокат|юрист"
    r"|прав[ао][^.!?\n]{0,60}потребител|защит[а-яё]*\s+прав\s+потребител"
    r"|наруш\w*\s+прав|расторжен\w*\s+договор"
    r"|по\s+закону[^.!?\n]{0,80}(?:обязан|должн|наруш)"
    r"|незаконн\w*",
    re.I,
)

COMPLAINT_RE = re.compile(
    r"\bжал(?:об|у|ова)\w*|пожал(?:уюсь|уемся|уетесь|оваться|овались|уется|уются)\b"
    r"|возмущ\w*|недовол\w*|претензи|конфликт"
    r"|обман|мошенн\w*|ужасн|плохо\s+учит|плохо\s+пров[её]л|некомпетентн\w*",
    re.I,
)

REPUTATION_RE = re.compile(
    r"отзыв\w*\s+в\s+интернет|всех\s+предупреж\w*|напиш\w*\s+отзыв|остав\w*\s+отзыв",
    re.I,
)

PAYMENT_DISPUTE_RE = re.compile(
    r"(?:оплатил|оплатила|пров[её]л(?:и)?\s+плат[её]ж|списал[иось]*|деньги\s+списал)"
    r"[^.!?\n]{0,100}(?:не\s+вид|не\s+прош|нет\s+оплат|не\s+зачисл|не\s+получ)"
    r"|(?:оплат[ау]\s+не\s+вид|плат[её]ж\s+не\s+(?:прош[её]л|видно|зачисл))"
    r"|(?:дважды|два\s+раза|двойн\w*|повторно|ошибочно|лишн\w*)[^.!?\n]{0,80}(?:списал|списали|плат[её]ж|оплат|снял)"
    r"|(?:списал|списали|снял[ио]?)[^.!?\n]{0,80}(?:дважды|два\s+раза|двойн\w*|повторно|ошибочно|лишн\w*)",
    re.I,
)

BENIGN_HYPOTHETICAL_REFUND_RE = re.compile(
    r"(?:у\s+знаком\w*|у\s+друз\w*|у\s+подруг\w*|у\s+других|слышал[аи]?)"
    r"[^.!?\n]{0,80}\bвозврат\w*[^.!?\n]{0,80}(?:у\s+вас\s+как|как\s+у\s+вас|какие\s+правила|с\s+такими\s+ситуац)",
    re.I,
)

PRESALE_REFUND_POLICY_RE = re.compile(
    r"(?:если|вдруг)[^.!?\n]{0,80}(?:не\s+понравит|не\s+подойд)[^.!?\n]{0,80}(?:верн\w*\s+деньги|деньги\s+верн\w*|возврат)"
    r"|(?:условия|правила|порядок)[^.!?\n]{0,50}возврат"
    r"|возврат[^.!?\n]{0,120}(?:до\s+оплат|перед\s+оплат|заранее|услов|правил|если\s+не\s+понравит)",
    re.I,
)
PRESALE_REFUND_CONTEXT_RE = re.compile(
    r"(?:заранее|до\s+оплат|перед\s+оплат|понять|понимать)[\s\S]{0,160}возврат"
    r"|возврат[\s\S]{0,160}(?:заранее|до\s+оплат|перед\s+оплат|понять|понимать|услов|правил)",
    re.I,
)

SOFT_NEGATIVE_ONLY_RE = re.compile(
    r"\b(?:подумаю|обсудить|обсудим|с менеджером обсудить|наверное\s+подумаем)\b",
    re.I,
)

P0_HARD_TEXT_MARKERS: tuple[str, ...] = (
    "возврат",
    "вернуть деньги",
    "верните деньги",
    "деньги назад",
    "расторг",
    "жалоб",
    "жаловаться",
    "пожалуюсь",
    "претензи",
    "мошенн",
    "незаконн",
    "суд",
    "прокурат",
    "роспотреб",
)

P0_TRUE_POSITIVE_CASES: tuple[tuple[str, str], ...] = (
    ("Вы мошенники, верните деньги.", "complaint"),
    ("Незаконно списали оплату, буду жаловаться.", "legal"),
    ("Буду жаловаться в Роспотребнадзор.", "legal"),
    ("Пожалуюсь на вас, если не решите вопрос.", "complaint"),
    ("Хочу возврат денег за курс.", "refund"),
    ("Пойду в суд и прокуратуру.", "legal"),
    ("Оплатил, но оплату не видно.", "payment_dispute"),
    ("С меня дважды списали деньги за оплату, верните одну.", "refund"),
    ("Ошибочно списали оплату второй раз.", "payment_dispute"),
)

P0_BENIGN_CASES: tuple[str, ...] = (
    "Хочу обсудить с менеджером расписание.",
    "Подумаю и вернусь позже.",
    "Чтобы записаться, надо приезжать или можно дистанционно?",
    "Возврат к теме: сколько стоит курс?",
    "Где запросить справку для налогового вычета?",
    "У знакомых был возврат, а у вас как с такими ситуациями?",
    "А если ребёнку не понравится, деньги вернёте?",
    "Перед оплатой хочу понять условия возврата.",
)


def has_complaint_signal(text: str) -> bool:
    if re.search(r"\b(?:это\s+)?не\s+(?:как\s+)?(?:жалоб\w*|претензи\w*)\b", str(text or ""), re.I) and not re.search(
        r"мошенн|незаконн|возмущ|недовол|обман|ужасн|плохо\s+уч|некомпетент|суд|прокурат|роспотреб",
        str(text or ""),
        re.I,
    ):
        return False
    if re.search(r"\bжалоба\s+на\s+сайт\b", str(text or ""), re.I) and not re.search(
        r"мошенн|незаконн|претензи|возмущ|недовол|обман|ужасн|плохо\s+уч|некомпетент",
        str(text or ""),
        re.I,
    ):
        return False
    if SOFT_NEGATIVE_ONLY_RE.search(text) and not COMPLAINT_RE.search(text):
        return False
    return bool(COMPLAINT_RE.search(text))


def codes_from_text(text: str) -> tuple[str, ...]:
    value = str(text or "")
    result: list[str] = []
    benign_refund_context = _is_benign_hypothetical_refund(value)
    if REFUND_RE.search(value) and not benign_refund_context:
        result.append("refund")
    if LEGAL_RE.search(value):
        result.append("legal")
    if has_complaint_signal(value):
        result.append("complaint")
    if REPUTATION_RE.search(value):
        result.append("reputation_threat")
    if PAYMENT_DISPUTE_RE.search(value):
        result.append("payment_dispute")
    if "payment_dispute" in result and REFUND_RE.search(value) and not benign_refund_context:
        result.insert(0, "refund")
    return tuple(dict.fromkeys(result))


def memory_risk_flags_from_text(text: str) -> tuple[str, ...]:
    mapping = {
        "refund": "refund",
        "legal": "legal_threat",
        "complaint": "complaint",
        "reputation_threat": "complaint",
        "payment_dispute": "payment_dispute",
    }
    return tuple(dict.fromkeys(mapping.get(code, code) for code in codes_from_text(text)))


def contains_any_p0(codes: Sequence[str]) -> bool:
    return any(str(code or "").strip() for code in codes)


def is_benign_hypothetical_refund(text: str) -> bool:
    return _is_benign_hypothetical_refund(text)


def _is_benign_hypothetical_refund(text: str) -> bool:
    value = str(text or "")
    if not (
        BENIGN_HYPOTHETICAL_REFUND_RE.search(value)
        or PRESALE_REFUND_POLICY_RE.search(value)
        or PRESALE_REFUND_CONTEXT_RE.search(value)
    ):
        return False
    return not re.search(
        r"\b(?:верните|хочу\s+(?:вернуть|возврат|деньги|отказ)|требую|мне\s+верн|нам\s+верн|за\s+наш|с\s+меня|у\s+меня|списал|оплатил|оплатила|договор|уже|после\s+оплат|не\s+понравилось)\b",
        value,
        re.I,
    )
