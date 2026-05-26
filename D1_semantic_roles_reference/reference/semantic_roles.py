from __future__ import annotations

"""Типизированные смысловые роли реплики клиента (референс-реализация).

Зачем этот модуль (корень проблемы):
    В репозитории распознавание интента/формата/темы живёт в виде КАСКАДА
    частных `if _has_any_marker(...)` (conversation_intent_plan._fact_scope_constraints,
    _primary_intent) и НАБОРА узких regex (p0_recall_spec.PRESALE_REFUND_*).
    Каждый новый кейс = новая ветка → «крот»: правим одно, всплывает соседнее.

    Этот модуль ЗАМЕНЯЕТ каскад одной декларативной разметкой по НЕЗАВИСИМЫМ ОСЯМ.
    Реплика клиента описывается набором ОРТОГОНАЛЬНЫХ ролей, а не одним интентом:

        - training_format  — формат ОБУЧЕНИЯ (онлайн / очно / выездной-лагерь)
        - enrollment_vs_recording — смысл слова «запись» (оформиться / запись урока)
        - transfer_sense   — смысл слова «перевод» (группа / деньги / на менеджера)
        - payment_method   — СПОСОБ оплаты (долями / рассрочка / единоразово)
        - payment_source   — ИСТОЧНИК оплаты (маткапитал / вычет / сертификат)
        - place            — вопрос про адрес/город
        - refund_frame     — тип упоминания возврата (предпродажный / спор)
        - topics           — ВСЕ темы в реплике (мультитема, не одна)

    Способ оплаты и источник оплаты — РАЗНЫЕ оси. Их смешение давало баг
    «рассрочка при маткапитале». place/format/оформление — тоже разные оси.

Совместимость для переноса в репо (см. PORTING_FOR_CODEX.md):
    Модуль зависит ТОЛЬКО от stdlib `re`. Функция has_marker ниже повторяет
    поведение mango_mvp.channels.text_signals.has_marker (границы слова, casefold,
    ё→е). При переносе её нужно ИМПОРТИРОВАТЬ из text_signals, а не дублировать.
"""

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence


SEMANTIC_ROLES_SCHEMA_VERSION = "semantic_roles_ref_v1_2026_05_25"

_WORD_CHARS = "0-9a-zа-я"


# --------------------------------------------------------------------------- #
# Низкоуровневое сопоставление по ГРАНИЦАМ СЛОВА (зеркало text_signals.py)      #
# При переносе в репо: from mango_mvp.channels.text_signals import has_marker   #
# --------------------------------------------------------------------------- #
def normalize_signal_text(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").replace(" ", " ").split())


def has_marker(text: object, marker: str) -> bool:
    value = normalize_signal_text(text)
    needle = normalize_signal_text(marker)
    if not needle:
        return False
    if re.search(rf"[^{_WORD_CHARS}]", needle):
        return needle in value
    return bool(re.search(rf"(?<![{_WORD_CHARS}]){re.escape(needle)}[{_WORD_CHARS}]*", value))


def has_any_marker(text: object, markers: Sequence[str]) -> bool:
    return any(has_marker(text, marker) for marker in markers)


# --------------------------------------------------------------------------- #
# Декларативные словари осей. Это НЕ цепочка if — это данные.                   #
# Добавление нового слова = строка в словаре, НЕ новая ветка логики.            #
# --------------------------------------------------------------------------- #
_FORMAT_MARKERS: dict[str, tuple[str, ...]] = {
    "online": ("онлайн", "дистанц", "вебинар", "удаленно", "удалённо", "из дома"),
    "ochno": ("очно", "очный", "очных", "офлайн", "в классе", "в аудитор", "пацаева", "сретен", "красносель"),
    "vyezd_camp": ("лагер", "лвш", "смена", "смену", "выездн", "менделеево", "проживан", "с проживанием"),
}

_PAYMENT_METHOD_MARKERS: dict[str, tuple[str, ...]] = {
    "dolyami": ("долями",),
    "rassrochka": ("рассроч", "частями", "помесяч", "в рассрочку", "по месяцам"),
    "edinorazovo": ("единоразово", "сразу всю", "сразу всю сумму", "целиком", "полностью оплат", "одним платеж"),
}

_PAYMENT_SOURCE_MARKERS: dict[str, tuple[str, ...]] = {
    "matkap": ("маткап", "материнск", "материнским"),
    "tax_deduction": ("налоговый вычет", "вычет", "фнс", "кнд", "3-ндфл", "3 ндфл"),
    "sertifikat": ("сертификат", "сертификатом"),
}

# Темы (мультитема). Каждая — независимый флаг присутствия.
_TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    "price": ("цен", "стои", "сколько стоит", "прайс", "руб", "почем"),
    "discount": ("скид", "акци", "льгот", "процент", "суммир", "дешевле"),
    "trial": ("пробн", "пробное", "фрагмент", "попроб"),
    "camp": ("лагер", "лвш", "смена", "менделеево", "проживан", "питан", "трансфер"),
    "schedule": ("распис", "во сколько", "по каким дням", "когда занят", "дни занят", "время занят"),
    "format": ("формат", "онлайн или очно", "очно или онлайн"),
    "address": ("адрес", "где наход", "наход", "как добрат", "метро", "площадк"),
    "document": ("справк", "договор", "квитанц", "чек", "документ"),
    "identity": ("вы бот", "ты бот", "ты человек", "вы человек", "вы живой", "живой человек", "робот", "с кем я общаюсь", "это бот", "gpt", "нейросет"),
    "off_topic": ("айфон", "iphone", "погода", "биткоин", "сочинение про"),
}

# Слова-ловушки для оси «запись»: оформиться vs запись урока.
_ENROLL_NEIGHBORS = ("курс", "программ", "обучен", "группу", "занятия с", "к вам", "на физик", "на математ", "на информат")
_RECORDING_NEIGHBORS = ("урок", "заняти", "лекци", "вебинар", "пропущ", "пропуст", "пересмотр", "запис уроков")

# Слова-ловушки для оси «перевод».
_TRANSFER_GROUP_NEIGHBORS = ("групп", "класс", "на другой курс", "на курс", "сильнее", "послабее", "посильнее", "уровень")
_TRANSFER_MONEY_NEIGHBORS = ("деньги", "оплат", "платеж", "платёж", "банковск", "на счет", "на счёт", "реквизит")
_TRANSFER_MANAGER_NEIGHBORS = ("менеджер", "человек", "специалист", "оператор", "живой", "сотрудник")

# Сигналы возврата (ось refund_frame).
_REFUND_MENTION = ("возврат", "вернут", "вернете", "вернёте", "верну", "вернуть деньги", "деньги назад", "расторг", "отказ от обучен", "забрать деньги")
# Демонстрация требования (делает возврат спором независимо от оплаты).
_REFUND_DEMAND = ("верните", "требую", "хочу возврат", "хочу вернуть", "хочу деньги назад", "верните деньги", "расторгнуть договор", "немедленно верн")
# Признаки уже состоявшейся сделки/оплаты (делает возврат спором — P0).
_REFUND_POST_PAYMENT = ("оплатил", "оплатила", "уже оплат", "после оплаты", "списали", "списал", "заключили договор", "мы платили", "я платил", "за наш", "с меня сняли")
# Предпродажная рамка: гипотеза/условие/до оплаты (делает возврат БЕЗОПАСНЫМ вопросом политики).
_REFUND_PRESALE_FRAME = (
    "если", "вдруг", "до начала", "до оплаты", "перед оплатой", "заранее", "передумаю", "передумаем",
    "не понравит", "не подойд", "какие условия", "какие правила", "условия возврата", "правила возврата",
    "можно ли вернуть", "вернут ли", "а если",
)


@dataclass(frozen=True)
class MessageRoles:
    """Структурированные роли одной реплики клиента."""

    training_format: str = ""          # online | ochno | vyezd_camp | ""
    enrollment_vs_recording: str = ""  # enroll | recording | ""
    transfer_sense: str = ""           # group | money | manager | ""
    payment_method: str = ""           # dolyami | rassrochka | edinorazovo | ""
    payment_source: str = ""           # matkap | tax_deduction | sertifikat | ""
    asks_place: bool = False
    refund_frame: str = "none"         # none | presale_policy | dispute
    topics: tuple[str, ...] = ()       # все темы в реплике (мультитема)
    evidence: Mapping[str, str] = field(default_factory=dict)

    def to_prompt_view(self) -> Mapping[str, object]:
        return {
            "schema_version": SEMANTIC_ROLES_SCHEMA_VERSION,
            "training_format": self.training_format,
            "enrollment_vs_recording": self.enrollment_vs_recording,
            "transfer_sense": self.transfer_sense,
            "payment_method": self.payment_method,
            "payment_source": self.payment_source,
            "asks_place": self.asks_place,
            "refund_frame": self.refund_frame,
            "topics": list(self.topics),
            "evidence": dict(self.evidence),
        }


def _first_axis_value(text: str, table: Mapping[str, tuple[str, ...]]) -> tuple[str, str]:
    """Вернуть первую ось из таблицы, чьи маркеры найдены, + найденный маркер."""
    for value, markers in table.items():
        for marker in markers:
            if has_marker(text, marker):
                return value, marker
    return "", ""


def _single_axis_value(text: str, table: Mapping[str, tuple[str, ...]]) -> tuple[str, str]:
    """Вернуть ось формата. Если совпали несколько осей И в реплике есть «или»
    (дизъюнкция: «онлайн или очно?») — клиент СПРАШИВАЕТ/СРАВНИВАЕТ, а не выбрал,
    возвращаем пусто, чтобы не латчить ложный выбор. Без «или» (например
    «на онлайн вместо очного») считаем выбранным первый найденный."""
    matched: list[tuple[str, str]] = []
    for value, markers in table.items():
        for marker in markers:
            if has_marker(text, marker):
                matched.append((value, marker))
                break
    if not matched:
        return "", ""
    if len(matched) > 1 and has_marker(text, "или"):
        return "", "ambiguous_question:" + "/".join(v for v, _ in matched)
    return matched[0]


def _enrollment_vs_recording(text: str) -> str:
    """Снять омонимию слова «запись»: оформиться vs запись урока.

    Важно: смысл «запись урока» возникает и БЕЗ слова «запись» — по сигналам
    «пропустил/пересмотреть/вебинар». Поэтому ось включается, если есть либо
    слово «запись/оформить», либо признак пропущенного/записи занятия.
    """
    has_zapis = has_marker(text, "запис")
    # «оформ» как стем покрывает оформить/оформиться/оформление (существительное тоже).
    enroll_verb = has_any_marker(text, ("записаться", "записать", "оформ"))
    near_recording = has_any_marker(text, _RECORDING_NEIGHBORS)
    near_enroll = has_any_marker(text, _ENROLL_NEIGHBORS)
    if not (has_zapis or enroll_verb or near_recording):
        return ""
    # Явный глагол оформления побеждает (записаться/оформиться на урок = оформление).
    if enroll_verb or near_enroll:
        return "enroll"
    if near_recording:
        return "recording"
    # Голое «запись» без контекста урока — считаем оформлением (частотнее в продаже).
    return "enroll"


def _transfer_sense(text: str, context: Mapping[str, object] | None = None) -> str:
    """Снять омонимию слова «перевод/перевести».

    Если слово есть, но соседа в самой реплике нет (голый follow-up
    «переводят?», «перевести потом можно?»), смысл берётся из КОНТЕКСТА диалога:
    последний разрешённый transfer_sense или активный групповой топик. Гадать без
    контекста нельзя — иначе ложный money/manager."""
    if not has_any_marker(text, ("перевод", "перевест", "перевед", "переведите", "переключите")):
        return ""
    if has_any_marker(text, _TRANSFER_MANAGER_NEIGHBORS):
        return "manager"
    if has_any_marker(text, _TRANSFER_MONEY_NEIGHBORS):
        return "money"
    if has_any_marker(text, _TRANSFER_GROUP_NEIGHBORS):
        return "group"
    if context:
        last = str(context.get("last_transfer_sense") or "")
        if last:
            return last
        if context.get("group_topic_active"):
            return "group"
    return ""


def _refund_frame(text: str) -> tuple[str, str]:
    """Типизировать упоминание возврата по осям сделки и речевого акта.

    Это ЗАМЕНА растущего списка узких regex (PRESALE_REFUND_POLICY_RE и т.п.).
    Логика общая, а не по фразам:
      - есть требование ИЛИ признак уже состоявшейся оплаты → спор (P0);
      - возврат назван в гипотетической/условной/до-оплатной рамке без требования
        и без оплаты → предпродажный вопрос политики (безопасно отвечаем);
      - иначе возврат не упомянут.
    Так фраза «передумаю до начала, деньги вернут?» становится presale_policy
    БЕЗ внесения её в список — потому что нет ни оплаты, ни требования, есть рамка.
    """
    demand_hit = next((m for m in _REFUND_DEMAND if has_marker(text, m)), "")
    paid_hit = next((m for m in _REFUND_POST_PAYMENT if has_marker(text, m)), "")
    # Требование вернуть деньги само по себе = упоминание возврата.
    mentions_refund = has_any_marker(text, _REFUND_MENTION) or bool(demand_hit)
    if not mentions_refund:
        return "none", ""
    if demand_hit:
        return "dispute", f"demand:{demand_hit}"
    if paid_hit:
        return "dispute", f"post_payment:{paid_hit}"
    frame_hit = next((m for m in _REFUND_PRESALE_FRAME if has_marker(text, m)), "")
    if frame_hit:
        return "presale_policy", f"presale_frame:{frame_hit}"
    # Возврат назван, но без рамки, без оплаты, без требования — осторожно как спор:
    # ложное P0 безопаснее пропуска реального возврата (CLAUDE.md: P0 консервативно).
    return "dispute", "bare_refund_mention"


def tag_message_roles(text: str, *, context: Mapping[str, object] | None = None) -> MessageRoles:
    """Главная точка входа: разметить реплику клиента по всем осям сразу.

    context (необязательно) — лёгкое held-состояние диалога для разрешения
    контекст-зависимых follow-up (сейчас используется для оси «перевод»). Без
    context распознаватель работает по одной реплике, как раньше."""
    value = str(text or "")
    evidence: dict[str, str] = {}

    training_format, fmt_ev = _single_axis_value(value, _FORMAT_MARKERS)
    if fmt_ev:
        evidence["training_format"] = fmt_ev

    payment_method, pm_ev = _first_axis_value(value, _PAYMENT_METHOD_MARKERS)
    if pm_ev:
        evidence["payment_method"] = pm_ev

    payment_source, ps_ev = _first_axis_value(value, _PAYMENT_SOURCE_MARKERS)
    if ps_ev:
        evidence["payment_source"] = ps_ev

    enrollment_vs_recording = _enrollment_vs_recording(value)
    transfer_sense = _transfer_sense(value, context)
    asks_place = has_any_marker(value, _TOPIC_MARKERS["address"])
    refund_frame, refund_ev = _refund_frame(value)
    if refund_ev:
        evidence["refund_frame"] = refund_ev

    topics: list[str] = []
    for topic, markers in _TOPIC_MARKERS.items():
        if has_any_marker(value, markers):
            topics.append(topic)
    if payment_method:
        topics.append("installment")
    if payment_source == "matkap":
        topics.append("matkap")
    if payment_source == "tax_deduction":
        topics.append("tax")
    if enrollment_vs_recording == "enroll":
        topics.append("enrollment")
    if enrollment_vs_recording == "recording":
        topics.append("recording")
    if refund_frame == "dispute":
        topics.append("refund_dispute")
    if refund_frame == "presale_policy":
        topics.append("refund_presale")
    # уникализировать, сохранив порядок
    topics_unique = tuple(dict.fromkeys(topics))

    return MessageRoles(
        training_format=training_format,
        enrollment_vs_recording=enrollment_vs_recording,
        transfer_sense=transfer_sense,
        payment_method=payment_method,
        payment_source=payment_source,
        asks_place=asks_place,
        refund_frame=refund_frame,
        topics=topics_unique,
        evidence=evidence,
    )
