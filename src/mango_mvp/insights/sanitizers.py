from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SanitizerMode = Literal["manager", "bot", "customer"]
MAX_SANITIZER_PASSES = 5


BRAND_ARTIFACT_RE = re.compile(
    r"\b(?:[ОУ]?НПК|ЛНПК|МПК|УНФК|ЛФТ|НП\s*К|О\s*Н\s*П\s*К)\s*М\s*[ФШ]\s*[ТД]\s*[ИI]?\b|"
    r"черн(?:ый|ой)\s+центр|чеб[её]н?\s*центр|чебноцентр|вечерний\s+центр|"
    r"ч[ао]дов\w*\s+центр|ячерн\w*\s+центр|парковочн\w*\s+центр|"
    r"\b(?:МФТИ|ФТИ|МФШТИ|МФШДИ|МФТДИ|МФТ|МФТЫ|НФК|УНФК)\b|черныйцентр",
    re.I,
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
BROKEN_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]{2,}\s*@(?:\s*\.\s*){1,}[A-ZА-ЯЁ0-9._%+-]*", re.I)
MESSENGER_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{4,}(?!\w)")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s()\-]*){10,15}(?!\d)")
NON_MONEY_UNIT_RE = (
    r"человек(?:а|у|ом|е)?|люд(?:и|ей|ям|ьми|ях)?|"
    r"ученик(?:а|ов|у|ам|ами|ах)?|дет(?:и|ей|ям|ьми|ях)?|"
    r"клиент(?:а|ов|у|ам|ами|ах)?|балл(?:а|ов|у|ам|ами|ах)?|"
    r"мест(?:о|а|у|ам|ами|ах)?|заявк(?:а|и|ок|е|ам|ами|ах)?|"
    r"сообщени(?:е|я|й|ю|ям|ями|ях)?|касани(?:е|я|й|ю|ям|ями|ях)?"
)
NON_DISCOUNT_PERCENT_CONTEXT_RE = (
    r"результат\w*|успех\w*|гарант\w*|охват\w*|"
    r"посещаем\w*|сдач\w*|выполнен\w*|готовност\w*|"
    r"действующ\w*"
)
MONEY_AMOUNT_RE = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[\s\u00a0]\d{3})+|\d+[.,]\d+|\d[\d\s]{2,})\s*"
    r"(?:руб(?:\.|лей|ля|ль)?|₽|р\.?|тыс\.?|тысяч\w*)(?!\w)|"
    rf"(?<!\w)\d{{1,3}}(?:[\s\u00a0]\d{{3}})+(?!\s*(?:{NON_MONEY_UNIT_RE})\b)(?!\w)|"
    r"(?<![\w:])\d{1,3}[кk](?!\w)|"
    r"\b\d{1,4}\s*(?:т\.\s*р\.|тыс\.?\s*руб\.?|тыс\.?\s*р\.?)(?!\w)|"
    r"\b(?:цен[ауыойе]?|стоимост\w*|стоит|оплат[аеуыой]?|плат[её]ж\w*|абонемент\w*|"
    r"сумм[ауыойе]?|при\s+(?:ранн\w+\s+)?оплат[еуыой]?|перв(?:ый|ого)\s+семестр\w*|"
    r"втор(?:ой|ого)\s+семестр\w*|год\s+целиком|за\s+год|в\s+год|за\s+семестр|в\s+семестр)"
    r"\D{0,30}\b\d{4,6}\b|"
    rf"\bза\s+\d{{4,6}}\b(?!\s*(?:год(?:а|у|ом|е)?|г\.?))(?!\s*(?:{NON_MONEY_UNIT_RE})\b)|"
    r"\b\d{4,6}\s+(?:за|в|на)\s+(?:\d{1,3}\s+)?"
    r"(?:семестр\w*|год\w*|занят\w*|урок\w*|курс\w*|смен\w*|месяц\w*)\b|"
    r"\b(?:пятьдесят|сорок|тридцать|двадцать|десять|пятнадцать|шестьдесят|семьдесят|восемьдесят|девяносто|сто)\s+(?:тысяч\w*(?:\s+рубл\w*)?|рубл\w*)\b",
    re.I,
)
PERCENT_RE = re.compile(
    rf"(?<!\w)\d{{1,3}}\s*(?:%|процент(?:а|ов)?)(?!\s*(?:{NON_DISCOUNT_PERCENT_CONTEXT_RE})\b)(?!\w)|"
    r"\b(?:пять|десять|пятнадцать|двадцать|тридцать|сорок|пятьдесят)\s+процент(?:а|ов)?\b",
    re.I,
)
DISCOUNT_RE = re.compile(
    r"\b(?:скидк\w*|акци\w*|промокод\w*|льгот\w*|бонус\w*|ранн\w+\s+бронирован\w*)"
    r"(?:\s+(?:до\s+)?\[PERCENT\]|\s+\d{1,3}\s*%)?",
    re.I,
)
INSTALLMENT_RE = re.compile(
    r"\b(?:рассрочк\w*|рассрочн\w*|сплит|кредит\w*|помесячн\w*|част(?:ями)?|"
    r"предоплат\w*|перв(?:ый|ую)\s+плат[её]ж|втор(?:ой|ую)\s+плат[её]ж)"
    r"(?:\s+(?:на|до)\s+\d{1,2}\s+(?:месяц\w*|мес\.?|год\w*))?",
    re.I,
)
REFUND_RE = re.compile(
    r"\b(?:возврат\w*|верн[её]м\w*\s+деньг\w*|гаранти\w*\s+возврат\w*|можно\s+вернуть|"
    r"договор\w*|оферт\w*|юридическ\w*|гарантир\w*|гарантия|обязуемся|обязательств\w*)\b",
    re.I,
)
PAYMENT_PROVIDER_RE = re.compile(r"\b(?:альфа(?:[-\s]?банк\w*)?|алфа(?:[-\s]?банк\w*)?|сбер(?:банк)?\w*|тинькофф|т-банк\w*|яндекс\s*сплит)\b", re.I)
SURNAMELIKE_WORD = (
    r"[А-ЯЁ][а-яё]{2,}(?:ов|ев|ёв|ин|ын|ский|цкий|ская|цкая|"
    r"ова|ева|ёва|ина|енко|ук|юк|ич|ович|евич|овна|евна|ична|инична)(?:[а-яё]{0,4})?"
)
TEACHER_NAME_RE = re.compile(
    r"\b(?P<role>(?i:преподавател[ьяюеем]*|педагог[а-яё]*|учител[ьяюеем]*|куратор[а-яё]*|"
    r"методист[а-яё]*|администратор[а-яё]*|наставник[а-яё]*))\s*(?:[-–—:]\s*)?"
    r"[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){0,3}\b"
)
ORPHAN_NAME_AFTER_PLACEHOLDER_RE = re.compile(
    rf"\b(?:\[CLIENT_NAME\]|[Уу]ченик(?:а|у|ом|е)?|[Рр]еб[её]н(?:ок|ка|ку|ком|ке)|"
    rf"[Кк]лиент(?:а|у|ом|е)?|[Мм]енеджер(?:а|у|ом|е)?|[Пп]реподавател\w*|[Пп]едагог\w*|"
    rf"[Уу]чител\w*|[Кк]уратор\w*|[Мм]етодист\w*|[Аа]дминистратор\w*|[Нн]аставник\w*)\s+{SURNAMELIKE_WORD}\b",
)
FAMILY_NAME_LABEL_RE = re.compile(rf"\b[Фф]амил[иьею][а-яё]*\s+{SURNAMELIKE_WORD}\b")
ACTION_VERB_SURNAME_RE = re.compile(
    rf"\b(?:[Бб]удет(?:\s+ли)?\s+)?{SURNAMELIKE_WORD}\s+"
    r"(?:вести|вед[её]т|проводит|препода[её]т|преподавать|читает|занимается|курирует)\b|"
    r"\b(?:[Бб]удет\s+)?(?:вести|вед[её]т|проводит|препода[её]т|преподавать|читает|занимается|курирует)\s+"
    rf"{SURNAMELIKE_WORD}\b"
)
CONTEXT_SURNAME_RE = re.compile(
    rf"\b(?:[Пп]о|[Кк]|[Кк]о|[Уу]|[Сс]|[Сс]о|[Дд]ля|[Оо]|[Оо]б|[Пп]ро)\s+{SURNAMELIKE_WORD}\b",
)
ADDRESS_RE = re.compile(
    r"\b(?:ул\.?|улиц[ауыей]?|проспект[а-яё]*|пр-т|переул(?:ок|ка|ке)?|пер\.?|шоссе|"
    r"бульвар[а-яё]*|площад[ьи]|проезд[а-яё]*)\s+[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z0-9\\-]+"
    r"(?:\s+[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z0-9\\-]+){0,3}"
    r"(?:\s*,?\s*(?:дом|д\.|корпус|корп\.?|к\.|строение|стр\.?)\s*\d+[А-Яа-яA-Za-z]?)?|"
    r"\b(?:дом|д\.|корпус|корп\.?|к\.|строение|стр\.?)\s*\d+[А-Яа-яA-Za-z]?\b|"
    r"\b(?:кабинет\w*|каб\.|аудитор\w*)\s*\d+\b|"
    r"\b(?:метро|м\.)\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b|"
    r"\b(?:Сухарев\w*|Долгопрудн\w*|С?корн[яе]жн\w*|С?коряжн\w*|Сретенк\w*|Ховрин\w*|"
    r"Первомайск\w*|Майск\w*|Менделеев\w*|Пацаев\w*|Чист(?:ые|ыми)\s+пруд(?:ы|ами)?|КПМ)\b",
    re.I,
)
DOCUMENT_REFERENCE_RE = re.compile(
    r"\b(?:файл|документ|word|pdf|excel|эксел[ья]?|таблиц[ауы])"
    r"(?:\s+(?:word|pdf|excel|эксел[ья]?))?\s+[«\"'][^»\"']{1,80}[»\"']|"
    r"\b(?:файл|документ|word|pdf|excel|эксел[ья]?|таблиц[ауы])\s+[\wа-яёА-ЯЁ0-9 ._-]{2,40}",
    re.I,
)
DEADLINE_RE = re.compile(
    r"\b(?:до|по)\s+(?:конца\s+)?(?:\d{1,2}\s+)?"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря|"
    r"дня|недели|месяца|года|каникул|учебного\s+года|сезона|акции|обучения|курса|занятия|урока|смены|"
    r"сегодня|завтра|понедельник[а]?|вторник[а]?|сред[уы]|четверг[а]?|"
    r"пятниц[уы]|суббот[уы]|воскресень[яе])\b|"
    r"\b(?:сегодня|завтра|послезавтра|на\s+этой\s+неделе|на\s+следующей\s+неделе)\b|"
    r"\b(?:понедельник|вторник|сред[ауы]|четверг|пятниц[ауы]|суббот[ауы]|воскресень[ея])\b|"
    r"\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b|"
    r"\b(?:начале|середине|конце)\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b|"
    r"\bв\s+течение\s+(?:\d+|одн[аои]|двух|тр[её]х|четыр[её]х|пяти|десяти|пятнадцати|двадцати)\s+"
    r"(?:минут|час(?:ов|а)?|дн(?:ей|я)?)\b|"
    r"\b\d{1,2}[.\/-]\d{1,2}(?:[.\/-]\d{2,4})?\b|"
    r"\b(?:до\s+)?(?:\d{1,2}\s*(?:и|,)\s*)*\d{1,2}\s+числа\b|"
    r"\b\d{1,2}\s*:\s*\d{2}(?:\s*[-–]\s*\d{1,2}\s*:\s*\d{2})?\b|"
    r"\bс\s+\d{1,2}\s+до\s+\d{1,2}\b|"
    r"\b(?:бронь|брониров\w*|заброниров\w*|заброниру\w*|держим)\b",
    re.I,
)
SOFT_PROMISE_RE = re.compile(
    r"\b(?:верн[её]м(?:ся)?|перезвон\w*|свяж(?:емся|ется)|напиш\w*|сообщ\w*|уточн\w*|"
    r"провер\w*|компенсир\w*)\b.{0,70}?"
    r"\b(?:до\s+конца\s+\w+|сегодня|завтра|в\s+течение\s+\w+|как\s+только)\b|"
    r"\b(?:до\s+конца\s+\w+|сегодня|завтра|в\s+течение\s+\w+|как\s+только)\b.{0,70}?"
    r"\b(?:верн[её]м(?:ся)?|перезвон\w*|свяж(?:емся|ется)|напиш\w*|сообщ\w*|уточн\w*|"
    r"провер\w*|компенсир\w*)\b",
    re.I,
)
COMPENSATION_LANG_RE = re.compile(
    r"\b(?:компенсир\w+|возмест\w+|верн[её]м\s+деньг\w*|возврат\s+(?:средств|оплат[ыау]?|денег))\b",
    re.I,
)
PERSON_NAME_RE = re.compile(
    r"\b[А-ЯЁ][а-яё]{2,}\s+(?:[А-ЯЁ][а-яё]{2,}(?:вич|вна|ична|овна|евна|инична|ич)|[А-ЯЁ][а-яё]{2,})\b"
)
COMMON_SINGLE_NAME_RE = re.compile(
    r"\b(?:"
    r"Аня|Анн[ауы]?|Андре[яйю]|Алекс(?:ей|ея|ею)|Арт[её]м[ауы]?|Ван[яеию]|Владимир[ауы]?|"
    r"Дарь[яеию]|Дмитри[йяю]|Егор[ауы]?|Иван(?:а|у|ы|ом)?|Ирин[ауы]?|Кирилл[ауы]?|"
    r"Мар(?:ия|ии|ию|ья|ью)|Маргарит[ауы]?|Марк[ауы]?|Матве[йяю]|Максим[ауы]?|Михаил[ауы]?|"
    r"Назир[ауы]?|Ольг[аеиу]|Полин[ауы]?|Пав(?:ел|ла|лу)|Роман[ауы]?|Серге[йяю]|"
    r"Елен[аые]?|Наталь[яиюи]|Татьян[аые]?|Светлан[аые]?|Карин[аые]?|Дарин[аые]?|"
    r"Екатерин[аыеу]?|Анастаси[яиюи]|Никола[йяю]|Даниил[ауы]?|Алис[ауы]?|Александр[ауы]?|"
    r"Никит[аеуы]?|Анн[аеуы]?|Ев(?:а|у|ы|ой)?|Амир[ауы]?|Георги[йяю]|Глеб[ауы]?|Сав[аы]?|"
    r"Арсени[йяю]|Евгени[йяю]|Паш[аеиу]?|Платон[ауы]?|Таиси[яиюи]|Валери[яиюи]|"
    r"Лиз[аые]?|Вов[ауы]?|Ром[ауы]?|Антон[ауы]?|Ф[её]дор[ауы]?|Демид[ауы]?|Злат[ауы]?|Иль[яеию]|Берев[ауы]?|Лук[аиуы]?|Маш[аеиу]?|Елизавет[аые]?|Владислав[ауы]?|"
    r"Агни[яиюи]|Ярослав[ауы]?|Константин[ауы]?|Тимур[ауы]?|Аслан[ауы]?|Надежд[ауы]?|"
    r"Васили[йяю]|Наст[яеию]|Святослав[ауы]?|Антони[йяю]|Виктори[яиюи]|Иван(?:а|у|ы|ом)?|"
    r"Влад[ауы]?|Амир(?:а|у|ы|ом)?|Миш(?:а|е|у|и|ей)?|Вячеслав[ауы]?|Ибрагимов[ауы]?|"
    r"Софи[яьи]|Софь[яюи]|Юр[аы]|Юри[йяю]"
    r"|Катерин[аыеу]?|Кат[яеию]|Олег[ауы]?|Марин[аыеу]?|Игор[ьяюя]|Ал[её]н[аыеу]?"
    r")\b"
)
BOT_PLACEHOLDER_RE = re.compile(
    r"\[(?:CURRENT_PRICE|CURRENT_DEADLINE|CURRENT_LOCATION|CURRENT_DOCUMENT|PAYMENT_OPTIONS|"
    r"REFUND_POLICY|SERVICE_PROMISE|CLIENT_NAME|PHONE|EMAIL|COMPANY_NAME)\]"
)
BOT_UNSAFE_PLACEHOLDER_RE = re.compile(r"\[(?:АКТУАЛЬН|СТОИМОСТ|СКИДК|РАССРОЧ|ВОЗВРАТ|ТЕЛЕФОН|ПОЧТ|ИМЯ|КОМПАН)\w*\]", re.I)
PERSONAL_PLACEHOLDER_RE = re.compile(r"\[(?:CLIENT_NAME|PHONE|EMAIL)\]")
INTERNAL_METADATA_RE = re.compile(
    r"\b(?:fact_id|source_id|trace_id)\s*[:=]\s*[^\s,;}\]]+|"
    r"\bfact:v3:[A-Za-z0-9_:\-]+|"
    r"\bsource:[A-Za-z0-9_:\-]+",
    re.I,
)
RAW_JSON_LEAK_RE = re.compile(
    r"^\s*[\{\[]\s*\"(?:message_type|route|topic_id|draft_text|safety_flags|manager_checklist|fact_id|source_id|trace_id)\"|"
    r"\"(?:message_type|route|topic_id|draft_text|safety_flags|manager_checklist|fact_id|source_id|trace_id)\"\s*:",
    re.I | re.S,
)

FLAG_GROUPS: dict[str, tuple[str, ...]] = {
    "brand_risk_flag": ("brand_normalized",),
    "money_or_discount_flag": ("price_redacted", "percent_redacted", "discount_terms_redacted", "payment_provider_redacted"),
    "installment_flag": ("installment_terms_redacted",),
    "legal_or_refund_flag": ("refund_policy_redacted",),
    "deadline_or_promise_flag": ("deadline_redacted", "service_promise_redacted"),
    "personal_data_flag": ("email_redacted", "phone_redacted", "person_name_redacted", "role_name_redacted", "location_redacted", "document_reference_redacted"),
}


@dataclass(frozen=True)
class SanitizedText:
    text: str
    flags: tuple[str, ...]
    status: str
    pass_count: int = 0
    fixpoint_reached: bool = True


def sanitize_answer(text: object, *, mode: SanitizerMode = "manager", max_passes: int = MAX_SANITIZER_PASSES) -> SanitizedText:
    """Sanitize until a fixed point; unresolved rule cycles are blocked.

    The extra passes are cheap in batch exports, but future online bot calls should
    keep measuring pass_count/latency before enabling autonomous responses.
    """
    if max_passes < 1:
        raise ValueError("max_passes must be >= 1")
    source = clean_text(text)
    if not source:
        return SanitizedText("", (), "empty")

    current = source
    all_flags: list[str] = []
    for pass_index in range(1, max_passes + 1):
        passed = _sanitize_answer_pass(current, mode=mode)
        all_flags.extend(passed.flags)
        if passed.text == current:
            flags = tuple(dict.fromkeys(all_flags))
            return SanitizedText(
                passed.text,
                flags,
                _status_for_text(passed.text, flags, mode=mode),
                pass_count=max(1, pass_index - 1),
                fixpoint_reached=True,
            )
        current = passed.text

    flags = tuple(dict.fromkeys(all_flags))
    if mode in {"bot", "customer"}:
        return SanitizedText("", flags, "fixpoint_not_reached", pass_count=max_passes, fixpoint_reached=False)
    return SanitizedText(current, flags, "fixpoint_not_reached", pass_count=max_passes, fixpoint_reached=False)


def _sanitize_answer_pass(text: object, *, mode: SanitizerMode = "manager") -> SanitizedText:
    source = clean_text(text)
    flags: list[str] = []
    if not source:
        return SanitizedText("", (), "empty")

    result = source
    if mode == "bot" and RAW_JSON_LEAK_RE.search(result):
        flags.append("raw_json_redacted")
        result = RAW_JSON_LEAK_RE.sub("", result)
    result = _replace(INTERNAL_METADATA_RE, result, "", flags, "internal_metadata_redacted")
    result = _replace(EMAIL_RE, result, "[EMAIL]", flags, "email_redacted")
    result = _replace(BROKEN_EMAIL_RE, result, "[EMAIL]", flags, "email_redacted")
    result = _replace(MESSENGER_HANDLE_RE, result, "[EMAIL]", flags, "email_redacted")
    result = _replace(PHONE_RE, result, "[PHONE]", flags, "phone_redacted")
    if mode == "bot":
        result = _replace_role_name(TEACHER_NAME_RE, result, flags)
    result = _replace(PERSON_NAME_RE, result, "[CLIENT_NAME]", flags, "person_name_redacted")
    if mode == "bot":
        result = _replace(ORPHAN_NAME_AFTER_PLACEHOLDER_RE, result, "[CLIENT_NAME]", flags, "person_name_redacted")
        result = _replace(FAMILY_NAME_LABEL_RE, result, "[CLIENT_NAME]", flags, "person_name_redacted")
        result = _replace(ACTION_VERB_SURNAME_RE, result, "[CLIENT_NAME]", flags, "person_name_redacted")
        result = _replace(CONTEXT_SURNAME_RE, result, "[CLIENT_NAME]", flags, "person_name_redacted")
        result = _replace(COMMON_SINGLE_NAME_RE, result, "[CLIENT_NAME]", flags, "person_name_redacted")
        result = _replace(ADDRESS_RE, result, "[CURRENT_LOCATION]", flags, "location_redacted")
        result = _replace(DOCUMENT_REFERENCE_RE, result, "[CURRENT_DOCUMENT]", flags, "document_reference_redacted")

    brand_replacement = "Фотон" if mode == "manager" else "[COMPANY_NAME]"
    result = _replace(BRAND_ARTIFACT_RE, result, brand_replacement, flags, "brand_normalized")

    result = _replace(MONEY_AMOUNT_RE, result, "[CURRENT_PRICE]", flags, "price_redacted")
    result = _replace(PERCENT_RE, result, "[PERCENT]", flags, "percent_redacted")
    result = _replace(DISCOUNT_RE, result, "[PAYMENT_OPTIONS]", flags, "discount_terms_redacted")
    result = _replace(INSTALLMENT_RE, result, "[PAYMENT_OPTIONS]", flags, "installment_terms_redacted")
    result = _replace(REFUND_RE, result, "[REFUND_POLICY]", flags, "refund_policy_redacted")
    if mode == "bot":
        result = _replace(PAYMENT_PROVIDER_RE, result, "[PAYMENT_OPTIONS]", flags, "payment_provider_redacted")
        result = _replace(SOFT_PROMISE_RE, result, "[SERVICE_PROMISE]", flags, "service_promise_redacted")
        result = _replace(COMPENSATION_LANG_RE, result, "[SERVICE_PROMISE]", flags, "service_promise_redacted")
    result = _replace(DEADLINE_RE, result, "[CURRENT_DEADLINE]", flags, "deadline_redacted")
    result = result.replace("[PERCENT]", "[PAYMENT_OPTIONS]")
    result = normalize_spacing(result)

    if mode == "bot":
        result = harden_bot_answer(result, flags)

    flags_tuple = tuple(dict.fromkeys(flags))
    return SanitizedText(result, flags_tuple, _status_for_text(result, flags_tuple, mode=mode), pass_count=1)


def _status_for_text(text: str, flags: tuple[str, ...] | list[str], *, mode: SanitizerMode) -> str:
    status = "safe_with_placeholders" if flags else "safe_no_changes"
    if mode == "bot" and has_any_safety_risk(text):
        return "blocked_unresolved_safety_risk"
    return status


def sanitize_customer_text(text: object) -> SanitizedText:
    return sanitize_answer(text, mode="customer")


def harden_bot_answer(text: str, flags: list[str]) -> str:
    result = text
    result = re.sub(r"^\[CLIENT_NAME\],?\s*", "", result)
    result = re.sub(r"\bу\s+\[CLIENT_NAME\]", "у ученика", result)
    result = result.replace("[CLIENT_NAME]", "ученик")
    result = result.replace("[PHONE]", "удобный контакт")
    result = result.replace("[EMAIL]", "удобный контакт")
    if "[CURRENT_PRICE]" in result:
        result = result.replace("[CURRENT_PRICE]", "актуальную стоимость")
    if "[PAYMENT_OPTIONS]" in result:
        result = result.replace("[PAYMENT_OPTIONS]", "актуальные варианты")
    if "[REFUND_POLICY]" in result:
        result = result.replace("[REFUND_POLICY]", "действующие правила изменения или отмены услуги")
    if "[CURRENT_DEADLINE]" in result:
        result = result.replace("[CURRENT_DEADLINE]", "актуальное окно записи")
    if "[CURRENT_LOCATION]" in result:
        result = result.replace("[CURRENT_LOCATION]", "адрес, который подтвердит менеджер")
    if "[CURRENT_DOCUMENT]" in result:
        result = result.replace("[CURRENT_DOCUMENT]", "материал, который пришлет менеджер")
    if "[SERVICE_PROMISE]" in result:
        result = result.replace("[SERVICE_PROMISE]", "менеджер свяжется с вами после проверки")
    if "[COMPANY_NAME]" in result:
        result = result.replace("[COMPANY_NAME]", "наш учебный центр")
    if any(flag in flags for flag in ("price_redacted", "discount_terms_redacted", "installment_terms_redacted", "refund_policy_redacted", "deadline_redacted")):
        suffix = "Точные условия менеджер подтвердит по актуальным правилам."
        if suffix.lower() not in result.lower():
            result = f"{result.rstrip()} {suffix}"
    return normalize_spacing(result)


def flags_to_text(flags: tuple[str, ...] | list[str]) -> str:
    return " | ".join(flags)


def flag_booleans(flags: tuple[str, ...] | list[str]) -> dict[str, str]:
    flag_set = set(flags)
    return {
        field: "Да" if any(flag in flag_set for flag in group_flags) else "Нет"
        for field, group_flags in FLAG_GROUPS.items()
    }


def has_brand_risk(text: object) -> bool:
    return bool(BRAND_ARTIFACT_RE.search(clean_text(text)) or BOT_UNSAFE_PLACEHOLDER_RE.search(clean_text(text)))


def has_money_or_terms_risk(text: object) -> bool:
    value = clean_text(text)
    return bool(
        MONEY_AMOUNT_RE.search(value)
        or PERCENT_RE.search(value)
        or DISCOUNT_RE.search(value)
        or INSTALLMENT_RE.search(value)
        or REFUND_RE.search(value)
        or PAYMENT_PROVIDER_RE.search(value)
        or DEADLINE_RE.search(value)
        or SOFT_PROMISE_RE.search(value)
        or COMPENSATION_LANG_RE.search(value)
    )


def has_personal_data_risk(text: object) -> bool:
    value = clean_text(text)
    return bool(
        EMAIL_RE.search(value)
        or BROKEN_EMAIL_RE.search(value)
        or MESSENGER_HANDLE_RE.search(value)
        or PHONE_RE.search(value)
        or TEACHER_NAME_RE.search(value)
        or ORPHAN_NAME_AFTER_PLACEHOLDER_RE.search(value)
        or FAMILY_NAME_LABEL_RE.search(value)
        or ACTION_VERB_SURNAME_RE.search(value)
        or CONTEXT_SURNAME_RE.search(value)
        or ADDRESS_RE.search(value)
        or DOCUMENT_REFERENCE_RE.search(value)
        or PERSON_NAME_RE.search(value)
        or COMMON_SINGLE_NAME_RE.search(value)
        or PERSONAL_PLACEHOLDER_RE.search(value)
    )


def has_any_safety_risk(text: object) -> bool:
    return has_brand_risk(text) or has_money_or_terms_risk(text) or has_personal_data_risk(text) or has_internal_metadata_risk(text)


def has_internal_metadata_risk(text: object) -> bool:
    value = clean_text(text)
    return bool(INTERNAL_METADATA_RE.search(value) or RAW_JSON_LEAK_RE.search(value))


def _replace(pattern: re.Pattern[str], text: str, replacement: str, flags: list[str], flag: str) -> str:
    if pattern.search(text):
        flags.append(flag)
        return pattern.sub(replacement, text)
    return text


def _replace_role_name(pattern: re.Pattern[str], text: str, flags: list[str]) -> str:
    if not pattern.search(text):
        return text
    flags.append("role_name_redacted")
    return pattern.sub(lambda match: match.group("role"), text)


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=\S)", r"\1 ", text)
    return text.strip()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


__all__ = [
    "MAX_SANITIZER_PASSES",
    "SanitizedText",
    "flag_booleans",
    "flags_to_text",
    "has_any_safety_risk",
    "has_brand_risk",
    "has_money_or_terms_risk",
    "has_personal_data_risk",
    "sanitize_answer",
    "sanitize_customer_text",
]
