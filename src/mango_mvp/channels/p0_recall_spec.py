from __future__ import annotations

import re
from typing import Sequence

from mango_mvp.channels.semantic_roles import is_negated_refund_topic, tag_message_roles


P0_RECALL_SPEC_SCHEMA_VERSION = "p0_recall_spec_v1_2026_05_24"
HARD_P0_CODES = frozenset({"refund", "legal", "complaint", "payment_dispute"})
SOFT_P0_CODES = frozenset({"reputation_threat"})

# This module is the shared P0 recall source for runtime guards, tests and KB
# trigger checks. Keep hard signals conservative: false negatives are more
# dangerous than false positives, but benign phrases below must stay non-P0.
REFUND_RE = re.compile(
    r"\bвозв?рат(?!\w*\s+к\s+(?:тем|урок|материал|заняти))\w*"
    r"|\bвозвращ\w*\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
    r"|\bверн\w*(?:\s+мне|\s+нам|\s+пожалуйста)?\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
    r"|\bверн\w*(?:\s+мне|\s+нам|\s+пожалуйста)?\s+(?:одну|один|лишн\w*|повторн\w*|дублирующ\w*)"
    r"|\bвернуть\s+оплат\w*"
    r"|\b(?:отдайте|отдать|отдаю|забрать|заберу)\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)\s+(?:назад|обратно)\b"
    r"|\bхочу\s+деньги\s+назад\b"
    r"|\bденьги\s+назад\b"
    r"|\bрасторг\w*\s+договор"
    r"|\bаннулир\w*\s+договор"
    r"|\bотказ\w*\s+от\s+обучен"
    r"|\bзабрать\s+деньги",
    re.I,
)

LEGAL_RE = re.compile(
    r"\bсуд\b|\bиск\b|претензи\w*|досудеб|роспотребнадзор|прокуратур|адвокат|юрист"
    r"|прав[ао]?[^.!?\n]{0,60}потребител|защит[а-яё]*\s+прав\s+потребител"
    r"|наруш\w*[^.!?\n]{0,60}прав[^.!?\n]{0,60}потребител"
    r"|наруш\w*\s+(?:моих|наших|своих|ваших\s+)?прав|расторжен\w*\s+договор"
    r"|по\s+закону[^.!?\n]{0,80}(?:обязан|должн|наруш)"
    r"|незаконн\w*",
    re.I,
)

COMPLAINT_RE = re.compile(
    r"\bжал(?:об|у|ова)\w*|пожал(?:уюсь|уемся|уетесь|оваться|овались|уется|уются)\b"
    r"|возмущ\w*|недовол\w*|претензи|конфликт"
    r"|обман|мошенн\w*|развод|развел[аи]?\w*|ужасн|плохо\s+учит|плохо\s+пров[её]л|некомпетентн\w*",
    re.I,
)

REPUTATION_RE = re.compile(
    r"отзыв\w*\s+в\s+интернет|всех\s+предупреж\w*|напиш\w*\s+отзыв|остав\w*\s+отзыв",
    re.I,
)

_PAYMENT_MOVED_PATTERN = (
    r"(?:(?<!не\s)\b(?:оплатил[аи]?|оплатили|заплатил[аи]?|заплатили)\b"
    r"|(?<!не\s)\bпров[её]л(?:и)?\s+плат[её]ж\b"
    r"|(?<!не\s)\b(?:списал[аи]?|списали|списалось|снял[аи]?|сняли)\b"
    r"|\b(?:деньги|оплат\w*|плат[её]ж\w*)\s+(?:ушл\w*|списал\w*|прош[её]л\w*|снял\w*)\b)"
)
_PAYMENT_RESULT_TARGET_PATTERN = r"(?:плат[её]ж\w*|оплат\w*|заняти[еяй]\w*|доступ\w*|курс|кабинет)"
_PAYMENT_RESULT_GAP = r"[^,.:;!?—–\-\n]{0,15}"
_PAYMENT_RESULT_MISSING_PATTERN = (
    rf"(?:(?:{_PAYMENT_RESULT_TARGET_PATTERN})"
    rf"{_PAYMENT_RESULT_GAP}(?:нет|не\s+(?:видн|появ|прош[её]л|зачисл|откр|получ)|пуст\w*)"
    r"|(?:нет|не\s+(?:видн|появ|прош[её]л|зачисл|откр|получ)|пуст\w*)"
    rf"{_PAYMENT_RESULT_GAP}(?:{_PAYMENT_RESULT_TARGET_PATTERN}))"
)
_PAYMENT_BLOCK_GAP = r"[^.!?:;—–\-\n]{0,100}"
_PAYMENT_DUPLICATE_CHARGE_PATTERN = (
    r"(?:(?:с\s+карты\s+)?(?:списал[аи]?|списали|списалось|снял[аи]?|сняли)"
    r"[^.!?\n]{0,30}\b(?:дважды|два\s+раза|повторн\w*|двойн\w*)\b"
    r"|\b(?:деньги|оплат\w*|плат[её]ж\w*)[^.!?\n]{0,30}"
    r"(?:списал\w*|снял\w*)[^.!?\n]{0,30}\b(?:дважды|два\s+раза|повторн\w*|двойн\w*)\b"
    r"|\b(?:дважды|два\s+раза|повторн\w*|двойн\w*)\b[^.!?\n]{0,30}"
    r"(?:списал[аи]?|списали|списалось|снял[аи]?|сняли|списан\w*|снят\w*))"
)
PAYMENT_DISPUTE_RE = re.compile(
    rf"(?:{_PAYMENT_MOVED_PATTERN}{_PAYMENT_BLOCK_GAP}{_PAYMENT_RESULT_MISSING_PATTERN}"
    rf"|{_PAYMENT_RESULT_MISSING_PATTERN}{_PAYMENT_BLOCK_GAP}{_PAYMENT_MOVED_PATTERN}"
    rf"|{_PAYMENT_DUPLICATE_CHARGE_PATTERN}"
    r"|чарджб[еэ]к|chargeback"
    r"|оспор\w*\s+(?:операци\w*|плат[её]ж\w*|списан\w*)"
    r"|отмен\w*\s+плат[её]ж\w*\s+через\s+банк"
    r"|не\s+буду\s+платить[^.!?\n]{0,80}(?:развод|обман|мошенн))",
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

PAYMENT_DISPUTE_POSITIVE_CASES: tuple[str, ...] = (
    "Деньги списали, а платежа в системе нет.",
    "Оплатила курс, но занятий в системе нет.",
    "Платёж не появился, хотя деньги ушли.",
    "Доступа не видно, деньги списали.",
    "Платёж прошёл, а кабинет пуст.",
    "Списали дважды!",
    "С карты списали два раза.",
    "Деньги списали повторно.",
)

PAYMENT_DISPUTE_BENIGN_CASES: tuple[str, ...] = (
    "Оплатила вчера, занятия завтра — в системе пока нет расписания.",
    "Я ещё не оплатил, занятия в системе не появились?",
    "Платежа нет, потому что мы пока не платили.",
    "Оплата будет позже, доступ пока не нужен.",
    "Можно оплатить, если группы в системе ещё нет?",
    "Если платёж не прошёл, как попробовать снова?",
    "Где будет видно оплату после платежа?",
    "Занятий в системе нет, это расписание ещё не открыли?",
    "Оплатил два курса.",
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
    ("Напишу в банк чарджбек.", "payment_dispute"),
    ("Оспорю операцию в банке.", "payment_dispute"),
    ("Отменю платёж через банк.", "payment_dispute"),
    ("Деньги ушли, а доступа нет.", "payment_dispute"),
    ("Оплатил, а курс так и не открыли.", "payment_dispute"),
    *((message, "payment_dispute") for message in PAYMENT_DISPUTE_POSITIVE_CASES),
    ("Не буду платить, это развод.", "complaint"),
    ("Отдайте оплату обратно.", "refund"),
    ("Аннулировать договор.", "refund"),
    ("Нарушение моих прав потребителя.", "legal"),
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
    "Если ребёнок надолго заболеет, за пропущенное вернёте?",
    "Вернуться к теме цены.",
    "Верните меня в список рассылки.",
    *PAYMENT_DISPUTE_BENIGN_CASES,
    "Оплатить можно позже, когда появится доступ?",
)


def has_complaint_signal(text: str) -> bool:
    if re.search(r"\b(?:вас|их|родител\w*|клиент\w*)[^.!?\n]{0,30}\bне\s+обманыва\w*", str(text or ""), re.I) and not re.search(
        r"мошенн|незаконн|возмущ|недовол|ужасн|плохо\s+уч|некомпетент|суд|прокурат|роспотреб|верн\w*\s+деньг",
        str(text or ""),
        re.I,
    ):
        return False
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
    refund_frame = tag_message_roles(value).refund_frame
    benign_refund_context = refund_frame == "presale_policy"
    negated_refund_topic = is_negated_refund_topic(value)
    if refund_frame == "dispute" or (REFUND_RE.search(value) and not benign_refund_context and not negated_refund_topic):
        result.append("refund")
    if LEGAL_RE.search(value):
        result.append("legal")
    if has_complaint_signal(value):
        result.append("complaint")
    if REPUTATION_RE.search(value):
        result.append("reputation_threat")
    if PAYMENT_DISPUTE_RE.search(value):
        result.append("payment_dispute")
    if "payment_dispute" in result and REFUND_RE.search(value) and not benign_refund_context and not negated_refund_topic:
        result.insert(0, "refund")
    return tuple(dict.fromkeys(result))


def hard_codes_from_text(text: str) -> tuple[str, ...]:
    return tuple(code for code in codes_from_text(text) if code in HARD_P0_CODES)


def soft_codes_from_text(text: str) -> tuple[str, ...]:
    return tuple(code for code in codes_from_text(text) if code in SOFT_P0_CODES)


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
    return tag_message_roles(text).refund_frame == "presale_policy"
