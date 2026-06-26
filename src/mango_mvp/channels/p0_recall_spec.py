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
    r"|\b(?:снят\w*|снять)(?:\s+реб[её]н\w*)?\s+с\s+(?:кружк\w*|курс\w*|заняти\w*|обучени\w*|групп\w*)"
    r"|\b(?:снят\w*|снять)[^.!?\n]{0,40}запис\w*"
    r"|\b(?:хочу|нужно|надо|прошу|можно)\s+снять(?:\s+реб[её]н\w*)?\s+с\s+(?:кружк\w*|курс\w*|заняти\w*|обучени\w*|групп\w*)"
    r"|\b(?:хочу|нужно|надо|прошу|можно)\s+снять[^.!?\n]{0,40}запис\w*"
    r"|\b(?:хочу|нужно|надо|прошу|можно)\s+отписа\w*(?:\s+реб[её]н\w*)?\s+от\s+(?:заняти\w*|обучени\w*|курс\w*)"
    r"|\bвыпис\w*[^.!?\n]{0,80}(?:реб[её]н\w*|с\s+(?:курс\w*|заняти\w*|обучени\w*)|из\s+групп\w*)"
    r"|\bотмен\w*[^.!?\n]{0,80}(?:запис\w*|курс\w*|обучени\w*|посещени\w*)"
    r"|\bотказ\w*\s+от\s+(?:заняти\w*|посещени\w*|курс\w*|обучени\w*|ходить)"
    r"|\b(?:прекратить|перестать)\s+(?:заниматься|ходить|посещать)\b"
    r"|\bбольше\s+не\s+будем\s+(?:ходить|заниматься|посещать)\b"
    r"|\b(?:возв?рат|верн\w*|возвращ\w*)[^.!?\n]{0,80}(?:за|по)\s+(?:смен\w*|лагер\w*|пут[её]вк\w*)"
    r"|\b(?:перенос|перенест\w*)[^.!?\n]{0,80}(?:смен\w*|пут[её]вк\w*|лагер\w*)"
    r"[^.!?\n]{0,80}(?:оплач\w*|уже\s+оплат\w*|деньг\w*|стоимост\w*|возв?рат)"
    r"|\b(?:перенос|перенест\w*)[^.!?\n]{0,80}(?:оплач\w*|уже\s+оплат\w*|деньг\w*|стоимост\w*|возв?рат)"
    r"[^.!?\n]{0,80}(?:смен\w*|пут[её]вк\w*|лагер\w*)"
    r"|\b(?:оплач\w*|уже\s+оплат\w*|деньг\w*|стоимост\w*|возв?рат)"
    r"[^.!?\n]{0,80}(?:смен\w*|пут[её]вк\w*|лагер\w*)[^.!?\n]{0,80}(?:перенос|перенест\w*)"
    r"|\b(?:сгорает|сгор\w*|пропад\w*)[^.!?\n]{0,80}(?:смен\w*|пут[её]вк\w*|оплат\w*\s+за\s+смен\w*)"
    r"|\b(?:смен\w*|пут[её]вк\w*|оплат\w*\s+за\s+смен\w*)[^.!?\n]{0,80}(?:сгорает|сгор\w*|пропад\w*)"
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
    r"|незаконн\w*"
    r"|(?:договор\w*|документ\w*|квитанц\w*|чек\w*|акт\w*)[^.!?\n]{0,100}"
    r"(?:неверн\w*|неправильн\w*|ошибк\w*|не\s+т[ао]\w*|не\s+тот|опечатк\w*|не\s+совпада\w*)"
    r"[^.!?\n]{0,100}(?:дат\w*|фамил\w*|фио|ф\\.?\s*и\\.?\s*о\\.?\s*|паспорт\w*|реквизит\w*|данн\w*)"
    r"|(?:договор\w*|документ\w*|квитанц\w*|чек\w*|акт\w*)[^.!?\n]{0,100}"
    r"(?:дат\w*|фамил\w*|фио|ф\\.?\s*и\\.?\s*о\\.?\s*|паспорт\w*|реквизит\w*|данн\w*)"
    r"[^.!?\n]{0,100}(?:неверн\w*|неправильн\w*|ошибк\w*|не\s+т[ао]\w*|не\s+тот|опечатк\w*|не\s+совпада\w*)"
    r"|(?:дат\w*|фамил\w*|фио|ф\\.?\s*и\\.?\s*о\\.?\s*|паспорт\w*|реквизит\w*|данн\w*)"
    r"[^.!?\n]{0,100}(?:договор\w*|документ\w*|квитанц\w*|чек\w*|акт\w*)"
    r"[^.!?\n]{0,100}(?:неверн\w*|неправильн\w*|ошибк\w*|не\s+т[ао]\w*|не\s+тот|опечатк\w*|не\s+совпада\w*)"
    r"|\b(?:исправ\w*|переделать|передела\w*|переоформ\w*)[^.!?\n]{0,80}"
    r"(?:договор\w*|документ\w*)[^.!?\n]{0,100}"
    r"(?:дат\w*|фамил\w*|фио|ф\\.?\s*и\\.?\s*о\\.?\s*|паспорт\w*|ошибк\w*|неверн\w*|неправильн\w*)",
    re.I,
)

COMPLAINT_RE = re.compile(
    r"\bжал(?:об|у|ова)\w*|пожал(?:уюсь|уемся|уетесь|оваться|овались|уется|уются)\b"
    r"|возмущ\w*|недовол\w*|претензи|конфликт"
    r"|обман|мошенн\w*|развод|развел[аи]?\w*|ужасн|плохо\s+учит|плохо\s+пров[её]л|некомпетентн\w*",
    re.I,
)
QUALITY_COMPLAINT_RE = re.compile(
    r"(?:преподавател\w*|педагог\w*|учител\w*)[^.!?\n]{0,90}"
    r"(?:не\s+объясня\w*|плохо\s+учит|некомпетентн\w*|ничему\s+не\s+учит)"
    r"|(?:не\s+объясня\w*|плохо\s+учит|некомпетентн\w*|ничему\s+не\s+учит)"
    r"[^.!?\n]{0,90}(?:преподавател\w*|педагог\w*|учител\w*)"
    r"|безобрази\w*|возмутительн\w*|отвратительн\w*|хамств\w*"
    r"|реб[её]н\w*[^.!?\n]{0,90}(?:ничего\s+)?не\s+понима\w*"
    r"|(?:ничего\s+)?не\s+понима\w*[^.!?\n]{0,90}реб[её]н\w*",
    re.I,
)

_CHILD_CONTEXT_PATTERN = r"(?:реб[её]н\w*|сын\w*|доч\w*|дочка|ученик\w*|учениц\w*|школьник\w*|школьниц\w*)"
_CHILD_INCIDENT_PATTERN = (
    r"(?:униз\w*|оскорб\w*|накрич\w*|высмея\w*|издева\w*|"
    r"дов[её]л\w*\s+до\s+сл[её]з|довели\s+до\s+сл[её]з)"
)
CHILD_INCIDENT_COMPLAINT_RE = re.compile(
    rf"(?:{_CHILD_CONTEXT_PATTERN}[^.!?\n]{{0,100}}{_CHILD_INCIDENT_PATTERN}"
    rf"|{_CHILD_INCIDENT_PATTERN}[^.!?\n]{{0,100}}{_CHILD_CONTEXT_PATTERN})",
    re.I,
)
CHILD_COMPLAINT_ESCALATION_RE = re.compile(
    rf"{_CHILD_CONTEXT_PATTERN}[^.!?\n]{{0,140}}(?:при\s+всех[^.!?\n]{{0,80}})?(?:этого\s+так\s+не\s+оставл\w*|буду\s+разбират\w*|"
    r"напиш\w*\s+жалоб\w*)",
    re.I,
)
CHILD_SAFETY_COMPLAINT_RE = re.compile(
    rf"(?:{_CHILD_CONTEXT_PATTERN}[^.!?\n]{{0,120}}"
    r"(?:остал\w*\s+один|никто\s+не\s+следил|без\s+(?:присмотр\w*|надзор\w*)|потерял\w*)"
    rf"|(?:остал\w*\s+один|никто\s+не\s+следил|без\s+(?:присмотр\w*|надзор\w*)|потерял\w*)"
    rf"[^.!?\n]{{0,120}}{_CHILD_CONTEXT_PATTERN}"
    r"|педагог[^.!?\n]{0,60}(?:отсутствовал\w*|не\s+приш[её]л\w*)"
    r"|преподавател\w*[^.!?\n]{0,60}(?:отсутствовал\w*|не\s+приш[её]л\w*))",
    re.I,
)
SERVICE_COMPLAINT_RE = re.compile(
    r"(?:менеджер[^.!?\n]{0,80}не\s+(?:отвеч\w*|перезвон\w*)"
    r"|никто[^.!?\n]{0,40}не\s+перезвон\w*"
    r"|сколько\s+можно\s+ждать"
    r"|жд[ау]\w*[^.!?\n]{0,40}(?:второй|третий|\d+)[-\s]*(?:й|ой|ий)?\s+д(?:е|н)\w*)",
    re.I,
)

REPUTATION_RE = re.compile(
    r"отзыв\w*\s+в\s+интернет|всех\s+предупреж\w*|напиш\w*\s+отзыв|остав\w*\s+отзыв",
    re.I,
)

_PAYMENT_MOVED_PATTERN = (
    r"(?:(?<!не\s)\b(?:оплатил[аи]?|оплатили|заплатил[аи]?|заплатили)\b"
    r"|(?<!не\s)\bпров[её]л(?:и)?\s+плат[её]ж\b"
    r"|(?<!не\s)\b(?:внес(?:ли|ла)?|оплачен\w*)\b"
    r"|(?<!не\s)\b(?:списал[аи]?|списали|списалось|снял[аи]?|сняли)\b"
    r"|\b(?:деньги|оплат\w*|плат[её]ж\w*)\s+(?:ушл\w*|списал\w*|прош[её]л\w*|снял\w*)\b)"
)
_PAYMENT_RESULT_ACCESS_TARGET_PATTERN = r"(?:ссылк\w*|приглашени\w*|логин\w*|парол\w*|платформ\w*|доступ\w*)"
_PAYMENT_RESULT_GENERIC_TARGET_PATTERN = r"(?:плат[её]ж\w*|оплат\w*|заняти[еяй]\w*|курс|кабинет)"
_PAYMENT_RESULT_ACCESS_GAP = r"[^.!?\n]{0,60}"
_PAYMENT_RESULT_GENERIC_GAP = r"[^,.:;!?—–\-\n]{0,15}"
_PAYMENT_RESULT_MISS_PATTERN = r"(?:нет|не\s+(?:видн|появ|прош[её]л|зачисл|откр|получ|приш\w*|да(?:ли|ют))|пуст\w*)"
_PAYMENT_RESULT_MISSING_PATTERN = (
    rf"(?:(?:{_PAYMENT_RESULT_ACCESS_TARGET_PATTERN}){_PAYMENT_RESULT_ACCESS_GAP}{_PAYMENT_RESULT_MISS_PATTERN}"
    rf"|{_PAYMENT_RESULT_MISS_PATTERN}{_PAYMENT_RESULT_ACCESS_GAP}(?:{_PAYMENT_RESULT_ACCESS_TARGET_PATTERN})"
    rf"|(?:{_PAYMENT_RESULT_GENERIC_TARGET_PATTERN}){_PAYMENT_RESULT_GENERIC_GAP}{_PAYMENT_RESULT_MISS_PATTERN}"
    rf"|{_PAYMENT_RESULT_MISS_PATTERN}{_PAYMENT_RESULT_GENERIC_GAP}(?:{_PAYMENT_RESULT_GENERIC_TARGET_PATTERN}))"
)
_PAYMENT_BLOCK_GAP = r"[^.!?\n]{0,100}"
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
    ("Оплатил онлайн-математику — доступа так и нет.", "payment_dispute"),
    ("Внесли оплату, приглашение не пришло.", "payment_dispute"),
    ("Оплатили курс, логин и пароль не дали.", "payment_dispute"),
    ("Оплатил, ссылка на платформу не пришла.", "payment_dispute"),
    *((message, "payment_dispute") for message in PAYMENT_DISPUTE_POSITIVE_CASES),
    ("Ребёнка унизили на занятии, я этого так не оставлю.", "complaint"),
    ("На ребёнка накричали при всей группе.", "complaint"),
    ("Преподаватель высмеял сына при всех.", "complaint"),
    ("Дочку довели до слёз на занятии, буду разбираться.", "complaint"),
    ("Учитель оскорбил ребёнка.", "complaint"),
    ("Ребёнок остался один после занятия, никто не следил.", "complaint"),
    ("Педагог не пришёл, дети были одни.", "complaint"),
    ("Сын потерялся, пока никто не следил за группой.", "complaint"),
    ("Менеджер не отвечает третий день.", "complaint"),
    ("Никто не перезвонил, сколько можно ждать?", "complaint"),
    ("Не буду платить, это развод.", "complaint"),
    ("Преподаватель не объясняет, ребёнок ничего не понимает.", "complaint"),
    ("Педагог некомпетентный, ничему не учит.", "complaint"),
    ("Это безобразие, как ведут занятия.", "complaint"),
    ("Отдайте оплату обратно.", "refund"),
    ("Аннулировать договор.", "refund"),
    ("Нарушение моих прав потребителя.", "legal"),
    ("Хочу снять ребёнка с кружка.", "refund"),
    ("Можно отказаться и выписать ребёнка с занятий?", "refund"),
    ("Нужно перенести оплаченную смену, или вернёте деньги за смену?", "refund"),
    ("Нужно перенести оплаченную смену.", "refund"),
    ("В договоре неверная дата и фамилия ребёнка, исправьте.", "legal"),
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
    "Как выбрать преподавателя?",
    "Оплачу позже.",
    "Ребёнок расстроился после занятия, как ему помочь?",
    "Ребёнок стесняется отвечать при всех, что посоветуете?",
    "Педагог вышел на минуту — это нормально?",
    "Можно перенести занятие на другой день?",
    "Перенесите урок со вторника на четверг, пожалуйста.",
    "Подскажите по договору-оферте, где он размещён?",
    "Когда пришлёте договор на подпись?",
    "Хочу перевести ребёнка в группу посильнее.",
    "Можно снять копию договора?",
    "Хочу отписаться от рассылки.",
    "Как снять стресс ребёнку?",
    "Как снять усталость ребёнку?",
    "Можно снять напряжение после занятий?",
)


def has_complaint_signal(text: str) -> bool:
    if (
        CHILD_INCIDENT_COMPLAINT_RE.search(text)
        or CHILD_COMPLAINT_ESCALATION_RE.search(text)
        or CHILD_SAFETY_COMPLAINT_RE.search(text)
        or QUALITY_COMPLAINT_RE.search(text)
        or SERVICE_COMPLAINT_RE.search(text)
    ):
        return True
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
