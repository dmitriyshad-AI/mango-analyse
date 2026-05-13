from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mango_mvp.question_catalog.contracts import (
    ANSWER_STATUS_DRAFT_NEEDS_REVIEW,
    ANSWER_STATUS_MANAGER_ONLY,
    ANSWER_STATUS_NEEDS_ROP_ANSWER,
    ANSWER_STATUS_NOT_CUSTOMER_QUESTION,
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
    "перезвон",
    "позвон",
    "пришло",
    "письмо",
    "ссылка",
    "возврат",
    "чек",
    "сумм",
    "лагер",
    "смен",
    "автобус",
    "транспорт",
    "продолжать",
    "отзыв",
    "почему",
    "правильно",
    "состо",
    "отмен",
    "перенос",
    "пропуск",
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
    "будьте в курсе",
    "продуктовую новинку",
    "счёт:",
    "счет:",
    "инн:",
    "кпп:",
    "окпо",
    "огрн",
    "swift",
    "почтовый адрес банка",
    "идентификатор участника эдо",
    "consumer. 1-ofd",
    "consumer.1-ofd",
    "ticket?",
    "utm_",
    "yclid",
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
    "content-transfer-encoding",
    "this is a multi-part message",
    "vlagere",
    "заявка в обработке",
    "статус не менялся",
    "сутокздравствуйте",
    "недельздравствуйте",
    "пересланное письмо",
    "все пришло, спасибо",
    "спасибо, пришла смс",
    "спасибо большое",
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
    ("summer_school", "летняя школа", re.compile(r"\bлвш|летн\w+\s+(?:очная|выездная|школ)|лагер|смен[ауы]", re.I)),
    ("zvsh", "ЗВШ", re.compile(r"\bзвш|заочн", re.I)),
    ("ovsh", "ОВШ", re.compile(r"\bовш|очн\w+\s+вечерн", re.I)),
    ("regular_course", "регулярный курс", re.compile(r"\bкурс|занят", re.I)),
    ("trial", "пробное занятие", re.compile(r"\bпробн", re.I)),
)
INTENT_PATTERNS = (
    ("not_customer_question", "служебное сообщение / не вопрос клиента", (), re.compile(r"\b(?:с\s+уважением|если\s+у\s+вас\s+остались\s+вопросы|nbsp|zwnj|compose|mime-version)\b", re.I)),
    ("call_reason", "уточнение причины звонка", (), re.compile(r"\b(?:по\s+какому\s+поводу|звонили|кто\s+звонил|что\s+случилось|что\s+за\s+звонок|за\s+девочк)", re.I)),
    ("incomplete_context", "неполный контекст / уточнение предыдущего сообщения", (), re.compile(r"\b(?:павлович|а\s+про\s+это|это\s+вы\s+же|это\s+он\s+и\s+есть|или\s+это\s+другое|правильно\s+ли\s+я\s+понимаю|его\s+не\s+нужно|так\s+какое\s+же|вот\s+таким\s+вариантом|а\s+в\s+принципе\s+они\s+есть|наверное,\s+нашла|здесь\s+верно)", re.I)),
    ("general_consultation", "общая консультация / нужна помощь", (), re.compile(r"\b(?:доброе\s+утро,\s+подскажете|можно\s+еще\s+вопрос|можно\s+ещё\s+вопрос|здесь\s+возможно\s+узнать\s+информацию|можно\s+ли\s+получить\s+подробную\s+информацию|нам\s+нужна\s+помощь|владеете\s+.*информац|с\s+кем\s+можно\s+обсудить|можно\s+обращаться\s+с\s+вопросами|возможно\s+заказать\s+звонок|могли\s+бы.*созвониться|могли\s+бы.*набрать)", re.I)),
    ("callback", "обратная связь", (), re.compile(r"\bперезвон|связаться|позвон|напишите|ответьте|свяжитесь|наберите|когда\s+можно\s+созвон", re.I)),
    ("no_interest", "отказ / неактуально", (), re.compile(r"\b(?:неактуаль|не\s+интерес|неинтерес|не\s+надо|не\s+требуется|пока\s+нет|не\s+планир|уже\s+узнал|сам[аи]?\s+набер|продолжение\s+пока\s+неинтересно)", re.I)),
    ("continuation_decision", "продолжение обучения / решение семьи", (), re.compile(r"\b(?:продолжать\s+ли|продолжать\s+обучение|буду\s+ли\s+я\s+продолжать|следующем\s+году|нужно\s+обсудить|пока\s+не\s+знаю|решение\s+пока\s+не\s+принято|думаю|подумаю|нужно\s+подумать|решаем|определимся|посоветуемся|чуть\s+позднее|попозже\s+напишу|как\s+они\s+соберутся)", re.I)),
    ("message_not_received", "письмо / ссылка / доступ не пришли", (), re.compile(r"\b(?:не\s+пришл|ничего\s+не\s+пришло|ничего\s+на\s+почту|так\s+и\s+ничего\s+не\s+прислали|письмо\s+не\s+пришло|ссылка\s+не\s+пришла|нет\s+ссылк|ссылок\s+нет|жду\s+письмо|не\s+получил[аи]?|не\s+дошло|на\s+какую\s+почту|открыть\s+почту|скинуть\s+на\s+почту)", re.I)),
    ("cancellation_change", "изменение / отмена услуги", (FACT_TYPE_DOCUMENTS,), re.compile(r"\b(?:отмен[ауы]|изменени[ея]|перенос|возврат|заявлени[ея]|правила\s+изменения|правила\s+отмены|отказ\s+от\s+услуги)", re.I)),
    ("payment_service", "оплата / возврат / чек", (FACT_TYPE_DOCUMENTS,), re.compile(r"\b(?:чек|квитанц|возврат|к[эе]шб[эе]к|задолженност|вернуть\s+деньги|можно\s+оплачивать|безналич|сумм[ауеы]|половину\s+суммы|как\s+оплатить|как\s+оплачивать|заплатить|назначени[ея]\s+платеж|счет\s+на\s+оплату|сч[её]т\s+на\s+оплату)", re.I)),
    ("documents_letter", "письма / справки / подтверждающие документы", (FACT_TYPE_DOCUMENTS,), re.compile(r"\b(?:письм[оа]|одно\s+или\s+три\s+письма|оформить\s+письмо|копи[яю]|за\s+прошлый\s+год|налоговая|срок\s+действия|паспорт|подписать|направить|справк)", re.I)),
    ("legal_partner", "юридические / партнерские вопросы", (FACT_TYPE_DOCUMENTS,), re.compile(r"\b(?:партнер|партн[её]р|сколково|грант|эдо|реквизит|договор|инн|кпп|юр(?:идическ)?|организац|резюме|работать\s+с\s+вами)", re.I)),
    ("status_followup", "статус запроса / ожидает ответа", (), re.compile(r"\b(?:получили|можете\s+уточнить,\s+получили|посмотрите.*все\s+ли\s+хорошо|появилась\s+ли\s+информация|есть\s+новости|вы\s+тогда\s+сообщите|передали\s+мою\s+позицию|напоминаю|ждать\s+ответа|сделанн\w+\s+заявк|статус\s+заявк)", re.I)),
    ("general_next_step", "следующий шаг / порядок действий", (), re.compile(r"\b(?:что\s+дальше\s+делать|какие\s+дальнейшие\s+действия|какой\s+порядок\s+действий|что\s+для\s+этого\s+нужно|что\s+необходимо|какая\s+информация\s+от\s+меня\s+нужна|подскажите\s+как\s+это\s+сделать|вы\s+сможете\s+помочь|есть\s+ли\s+такая\s+возможность|но\s+как\s+быть|что\s+написать)", re.I)),
    ("transport_logistics", "транспорт / логистика", (FACT_TYPE_LOCATION,), re.compile(r"\b(?:автобус|транспорт|трансфер|проезд|дорог|как\s+добраться|везут|доехать|вокзал|метро|привезу\s+сама)", re.I)),
    ("camp_living_conditions", "бытовые условия лагеря", (FACT_TYPE_PROGRAM,), re.compile(r"\b(?:режим\s+дня|подъ[её]м|отбой|медпункт|медицинская\s+страховка|своя\s+еда|приносить\s+свою\s+еду|холодильн|микроволнов|бытов)", re.I)),
    ("attendance_absence", "посещение / отсутствие / отметки", (), re.compile(r"\b(?:ребенка\s+нет|реб[её]нок\s+пришел|отсутствовала|не\s+будет|не\s+попал|пропустил|замена|отметите\s+заранее|отмечено\s+посещение)", re.I)),
    ("lesson_occurrence", "состоится ли занятие", (FACT_TYPE_SCHEDULE,), re.compile(r"\b(?:заняти[ея]\s+состо|урок\s+состо|будет\s+ли\s+занятие|состоятся\s+занятия)", re.I)),
    ("lesson_materials", "материалы урока / домашние задания", (FACT_TYPE_PROGRAM,), re.compile(r"\b(?:материал|лекци|задач|домашн|дз\b|урок[ауы]\s+посмотреть|запись\s+урока|что\s+проход)", re.I)),
    ("quality_feedback", "качество обучения / обратная связь", (), re.compile(r"\b(?:жалоб|отзыв|связи\s+нет|не\s+оповещ|оповещали|слишком\s+легк|слишком\s+сложн|легкие\s+задания|ошибк|результат|успеваем|не\s+нрав|оценивают.*прогресс|поведени|шутки|спамил|страдать)", re.I)),
    ("site_confusion", "сайт / источник информации", (FACT_TYPE_PROGRAM,), re.compile(r"\b(?:это\s+ваш\s+сайт|ваш\s+сайт|друго[йе]\s+сайт|сайта?\s+нашла|как\s+оно\s+работает|все\s+относятся\s+к\s+наш)", re.I)),
    ("price", "стоимость", (FACT_TYPE_PRICE,), re.compile(r"\bсколько|стоимост|цен[ауы]|стоит|оплат|абонем|питание\s+и\s+проживание", re.I)),
    ("schedule", "расписание", (FACT_TYPE_SCHEDULE,), re.compile(r"\bраспис|когда|время|дни|час|график|июн|июл|август|сентябр|учебный\s+год|след\s+учебн|вечером|утром|по\s+каким\s+дням|до\s+какого\s+числа|количество\s+дней|дневные|в\s+подряд|перерыв|работаете|начались|начало|через\s+неделю|в\s+один\s+день|с\s+января|заранее|сроки|до\s+.*числа", re.I)),
    ("location", "адрес / очная площадка", (FACT_TYPE_LOCATION,), re.compile(r"\bгде|адрес|очно|филиал|метро|площадк|каком\s+городе", re.I)),
    ("format", "формат обучения", (FACT_TYPE_PROGRAM,), re.compile(r"\bонлайн|очно|формат|дистанц", re.I)),
    ("discount", "скидки", (FACT_TYPE_DISCOUNT,), re.compile(r"\bскид|акци|льгот|многодет|приведи", re.I)),
    ("installment", "рассрочка", (FACT_TYPE_INSTALLMENT,), re.compile(r"\bрассроч|долями|сплит|частями|кредит", re.I)),
    ("trial", "пробное занятие", (FACT_TYPE_TRIAL, FACT_TYPE_SCHEDULE), re.compile(r"\bпробн", re.I)),
    ("camp_trip", "лагерь / смена / поездка", (FACT_TYPE_PROGRAM, FACT_TYPE_SCHEDULE), re.compile(r"\b(?:лагер|лвш|смен[ауы]|поезд|кампус|путевк|вожат)", re.I)),
    ("technical_access", "доступ / технический вопрос", (), re.compile(r"\bдоступ|ссылк|платформ|личн\w+\s+кабинет|лк\b|не\s+открыва|не\s+заход|логин|парол|повтор[ыа]?|вебинар|родительский\s+чат|вк\s+не\s+работает|скачивать|смс|максе|телеграмм|сообщения\s+закрыты|добавить\s+в\s+группу|настроек\s+чата|историю\s+.*пусто|просит\s+зарегистрироваться|ссылочку", re.I)),
    ("service_feedback", "обратная связь по обучению", (), re.compile(r"\bжалоб|обратн\w+\s+связ|домашн|дз\b|пропуск|отработк|перенос|ошибк|результат|успеваем", re.I)),
    ("tax_deduction", "налоговый вычет / справки", (FACT_TYPE_DOCUMENTS,), re.compile(r"\bналогов\w+\s+вычет|вычет|справк|чек|сертификат", re.I)),
    ("documents", "документы / договор", (FACT_TYPE_DOCUMENTS,), re.compile(r"\bдокумент|договор|оферт|справк|чек|сертификат|распечат|поликлиник|печать|079|манту|дискинтест|привив", re.I)),
    ("program", "программа курса", (FACT_TYPE_PROGRAM,), re.compile(r"\bпрограмм|темы|что проход|курс|предмет|впр|подтянуть|что\s+можете\s+предложить|активност|центр\s+репетиторов|уклон|направления|зимн\w+\s+.*школ|летние\s+лаге\s*ря|учиться\s+сама|как\s+.*работает|как\s+происходит|выезды\s+в\s+школ|подготовит[сц]я|учебник|скан|материал|проверять|уроком|домашн|д/з|группы\s+большие", re.I)),
    ("age_or_level_fit", "возраст / класс / уровень", (), re.compile(r"\b(?:какой\s+возраст|возраст|года\s+рождения|класс\s+.*закончил|следующий\s+.*переходит|по\s+уровн|уровн[яюеь]|тест|зачисля|распредел|контроль\s+уровня|подготовк)", re.I)),
    ("teacher", "преподаватель", (), re.compile(r"\bпреподав|учител|педагог", re.I)),
    ("level_fit", "подходит ли уровень", (), re.compile(r"\bуровн|подойдет|подойд[её]т|сильн|слаб|сложновато|будет\s+ли\s+сложно|интересно\s+в\s+этой\s+смен|классникам.*интересно|какой\s+класс\s+не\s+знаю|в\s+какой\s+.*класс.*возьмут", re.I)),
    ("enrollment", "запись на обучение", (FACT_TYPE_SCHEDULE,), re.compile(r"\bзапис|мест[ао]?|набор|попасть|добавить\s+е[щш]е\s+попытку|варианты\s+продления|пришло\s+приглашение", re.I)),
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
    if detect_noise_reason(text):
        return False
    words = re.findall(r"[a-zа-яё0-9]+", text, re.I)
    if len(words) < 3:
        return False
    if re.fullmatch(r"да[,.!\s]+можно[.!]?", text):
        return False
    return any(marker in text for marker in QUESTION_MARKERS)


def detect_noise_reason(value: Any) -> str | None:
    text = clean_text(value).lower()
    if any(marker in text for marker in SERVICE_NOISE_MARKERS):
        return "service_or_marketing_noise"
    if re.search(r"\b(?:с\s+уважением|если\s+у\s+вас\s+остались\s+вопросы|вы\s+можете\s+задать\s+их)\b", text, re.I):
        return "signature_or_footer"
    if re.search(r"\b(?:telegram|whatsapp|whats\s*app|mailto|mime-version|content-type)\b", text, re.I):
        return "technical_or_contact_footer"
    if re.search(r"\b(?:сч[её]т|инн|кпп|окпо|огрн|swift|корреспондентский\s+счет)\b", text, re.I):
        return "bank_or_legal_requisites"
    return None


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
    if intent_key in {"camp_trip", "transport_logistics", "lesson_occurrence"}:
        priority = "high"
        manager_reason = "Нужна проверка актуальных условий перед клиентским ответом."
    manager_only_intents = {
        "documents",
        "teacher",
        "technical_access",
        "service_feedback",
        "tax_deduction",
        "incomplete_context",
        "general_consultation",
        "message_not_received",
        "cancellation_change",
        "payment_service",
        "documents_letter",
        "legal_partner",
        "quality_feedback",
        "status_followup",
        "camp_living_conditions",
        "attendance_absence",
        "no_interest",
        "continuation_decision",
        "age_or_level_fit",
    }
    if intent_key in manager_only_intents:
        bot_permission = BOT_PERMISSION_MANAGER_ONLY
        answer_status = ANSWER_STATUS_MANAGER_ONLY
        manager_reason = "Нужен менеджер: вопрос зависит от персонального контекста или документов."
    if intent_key == "not_customer_question":
        bot_permission = BOT_PERMISSION_NOT_ALLOWED
        answer_status = ANSWER_STATUS_NOT_CUSTOMER_QUESTION
        manager_reason = "Не клиентский вопрос: исключить из рабочего пакета ответов."
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
    has_offline = bool(re.search(r"\bочн|офлайн", text, re.I))
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
