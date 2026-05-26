from __future__ import annotations

"""Typed semantic roles for a client message.

This module keeps word-boundary signal matching in one place and exposes
orthogonal axes such as training format, payment method/source and refund frame.
It is intentionally structural: generation still answers from facts and context.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from mango_mvp.channels.text_signals import has_any_marker, has_marker


SEMANTIC_ROLES_SCHEMA_VERSION = "semantic_roles_v1_2026_05_25"
INTENT_STATE_REPAIR_ENV = "TELEGRAM_INTENT_STATE_REPAIR"

FORMAT_MARKERS: dict[str, tuple[str, ...]] = {
    "online": ("онлайн", "дистанц", "вебинар", "удаленно", "удалённо", "из дома"),
    "ochno": ("очно", "очный", "очных", "офлайн", "в классе", "в аудитор", "пацаева", "сретен", "красносель"),
    "vyezd_camp": ("лагер", "лвш", "смена", "смену", "выездн", "менделеево", "проживан", "с проживанием"),
}

PAYMENT_METHOD_MARKERS: dict[str, tuple[str, ...]] = {
    "dolyami": ("долями",),
    "rassrochka": ("рассроч", "частями", "помесяч", "в рассрочку", "по месяцам"),
    "edinorazovo": ("единоразово", "сразу всю", "сразу всю сумму", "целиком", "полностью оплат", "одним платеж"),
}

INVOICE_MONTHLY_MARKERS = ("помесяч", "каждый месяц", "ежемесяч", "по месяцам")
INVOICE_TRANSFER_MARKERS = (
    "по счету",
    "по счёту",
    "счет",
    "счёт",
    "банковск",
    "перевод",
    "реквизит",
    "платеж",
    "платёж",
)
NOT_INSTALLMENT_MARKERS = (
    "не рассроч",
    "не долями",
    "не частями",
    "не через банк",
    "без рассроч",
    "не про рассроч",
)

PAYMENT_SOURCE_MARKERS: dict[str, tuple[str, ...]] = {
    "matkap": ("маткап", "материнск", "материнским", "сфр"),
    "tax_deduction": ("налоговый вычет", "вычет", "фнс", "кнд", "3-ндфл", "3 ндфл"),
    "sertifikat": ("сертификат", "сертификатом"),
}

TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    "price": ("цен", "стои", "сколько стоит", "прайс", "руб", "почем"),
    "discount": ("скид", "акци", "льгот", "процент", "суммир", "дешевле"),
    "trial": ("пробн", "пробное", "фрагмент", "попроб"),
    "camp": ("лагер", "лвш", "лш", "летняя школа", "летн", "смена", "менделеево", "проживан", "питан", "трансфер"),
    "schedule": (
        "распис",
        "во сколько",
        "по каким дням",
        "когда занят",
        "дни занят",
        "время занят",
        "раз в неделю",
        "суббот",
        "воскрес",
        "выходн",
    ),
    "format": ("формат", "онлайн или очно", "очно или онлайн"),
    "address": ("адрес", "где наход", "наход", "как добрат", "метро", "площадк"),
    "document": ("справк", "договор", "квитанц", "чек", "документ"),
    "identity": ("вы бот", "ты бот", "ты человек", "вы человек", "вы живой", "живой человек", "робот", "с кем я общаюсь", "это бот", "gpt", "нейросет"),
    "off_topic": ("айфон", "iphone", "погода", "биткоин", "сочинение про"),
}

DISCOUNT_SCOPE_MARKERS: dict[str, tuple[str, ...]] = {
    "discount_stacking": ("суммир", "складыв", "выбирается одна", "одна скидка", "наибольшая", "не суммируются"),
    "discount_second_subject": (
        "второй предмет",
        "вторым предметом",
        "второго предмета",
        "2-й предмет",
        "последующий предмет",
        "еще предмет",
        "ещё предмет",
        "второй онлайн-предмет",
        "второй онлайн предмет",
    ),
    "discount_multichild": ("второй ребенок", "второй ребёнок", "двое детей", "два ребенка", "два ребёнка", "многодет"),
    "discount_referral": ("приведи друга", "приглашенный друг", "приглашённый друг", "друг оплатит", "кэшбэк"),
}

CAMP_SCOPE_MARKERS: dict[str, tuple[str, ...]] = {
    "city_day_camp": ("городск", "дневн", "без прожив", "без проживания", "без ночев", "не выезд"),
    "residential_lvsh": ("лвш", "менделеево", "выездн", "с прожив", "проживан", "трансфер"),
}

ONLINE_TRACK_MARKERS: dict[str, tuple[str, ...]] = {
    "regular_online": ("не олимпиад", "обычн", "регулярн", "онлайн-курс", "онлайн курс"),
    "olympiad_online": ("олимпиад", "физтех", "рсош", "перечнев"),
}

SCHEDULE_SCOPE_MARKERS: dict[str, tuple[str, ...]] = {
    "office_hours": ("до скольки работает", "как работает офис", "график офиса", "часы работы", "телефон", "контакт"),
    "class_schedule": ("распис", "во сколько", "по каким дням", "дни занятий", "занятия", "уроки"),
}

ENROLL_NEIGHBORS = ("курс", "программ", "обучен", "группу", "занятия с", "к вам", "на физик", "на математ", "на информат")
RECORDING_NEIGHBORS = ("урок", "заняти", "лекци", "вебинар", "пропущ", "пропуст", "пересмотр", "запис уроков")
STRONG_RECORDING_MARKERS = ("пропущ", "пропуст", "пересмотр", "запис уроков")

TRANSFER_GROUP_NEIGHBORS = ("групп", "класс", "на другой курс", "на курс", "сильнее", "послабее", "посильнее", "уровень")
TRANSFER_MONEY_NEIGHBORS = ("деньги", "оплат", "платеж", "платёж", "банковск", "на счет", "на счёт", "реквизит")
TRANSFER_MANAGER_NEIGHBORS = ("менеджер", "человек", "специалист", "оператор", "живой", "сотрудник")

REFUND_MENTION = (
    "возврат",
    "вернут",
    "вернете",
    "вернёте",
    "вернуть деньги",
    "вернуть оплат",
    "возврат оплат",
    "деньги назад",
    "расторг",
    "отказ от обучен",
    "забрать деньги",
)
REFUND_DEMAND = (
    "верните",
    "требую",
    "хочу возврат",
    "хочу вернуть",
    "хочу деньги назад",
    "верните деньги",
    "расторгнуть договор",
    "немедленно верн",
)
REFUND_POST_PAYMENT = (
    "оплатил",
    "оплатила",
    "уже оплат",
    "после оплаты",
    "списали",
    "списал",
    "заключили договор",
    "мы платили",
    "я платил",
    "за наш",
    "с меня сняли",
)
REFUND_PRESALE_FRAME = (
    "если",
    "вдруг",
    "до начала",
    "до оплаты",
    "перед оплатой",
    "заранее",
    "передумаю",
    "передумаем",
    "не понравит",
    "не подойд",
    "какие условия",
    "какие правила",
    "условия возврата",
    "правила возврата",
    "можно ли вернуть",
    "вернут ли",
    "а если",
    "у знаком",
    "у друз",
    "у подруг",
    "у других",
    "слышал",
    "как с такими ситуац",
)
REFUND_POLICY_PROCESS_MARKERS = (
    "по заявлен",
    "заявление на возврат",
    "нужно писать заявлен",
    "писать заявлен",
    "порядок возврат",
    "порядок возврата",
    "как оформить возврат",
    "оформить возврат",
    "оформляется возврат",
    "какая процедура возврата",
    "процедура возврата",
)
REFUND_BENIGN_NON_REFUND = (
    "возврат к теме",
    "вернуться к теме",
    "вернуться к вопросу",
    "возврат к расписанию",
    "вернусь позже",
    "вернемся позже",
    "вернёмся позже",
)
REFUND_BENIGN_OBJECTS = (
    "верните меня",
    "вернуть меня",
    "верни меня",
    "верните в список",
    "верните меня в список",
    "верните меня в рассыл",
    "вернуть в рассыл",
)


@dataclass(frozen=True)
class MessageRoles:
    training_format: str = ""
    training_formats: tuple[str, ...] = ()
    enrollment_vs_recording: str = ""
    transfer_sense: str = ""
    payment_method: str = ""
    payment_source: str = ""
    asks_place: bool = False
    refund_frame: str = "none"
    discount_scope: str = ""
    camp_scope: str = ""
    online_track: str = ""
    schedule_scope: str = ""
    topics: tuple[str, ...] = ()
    evidence: Mapping[str, str] = field(default_factory=dict)

    def to_prompt_view(self) -> Mapping[str, object]:
        return {
            "schema_version": SEMANTIC_ROLES_SCHEMA_VERSION,
            "training_format": self.training_format,
            "training_formats": list(self.training_formats),
            "enrollment_vs_recording": self.enrollment_vs_recording,
            "transfer_sense": self.transfer_sense,
            "payment_method": self.payment_method,
            "payment_source": self.payment_source,
            "asks_place": self.asks_place,
            "refund_frame": self.refund_frame,
            "discount_scope": self.discount_scope,
            "camp_scope": self.camp_scope,
            "online_track": self.online_track,
            "schedule_scope": self.schedule_scope,
            "topics": list(self.topics),
            "evidence": dict(self.evidence),
        }


def tag_message_roles(text: object, *, context: Mapping[str, object] | None = None) -> MessageRoles:
    value = str(text or "")
    evidence: dict[str, str] = {}

    training_format, fmt_ev = _single_axis_value(value, FORMAT_MARKERS)
    training_formats = _multi_axis_values(value, FORMAT_MARKERS)
    if intent_state_repair_enabled() and _explicit_multi_format_request(value, training_formats):
        training_format = ""
        fmt_ev = "multi:" + "/".join(training_formats)
    if fmt_ev:
        evidence["training_format"] = fmt_ev

    payment_method, pm_ev = _payment_method_value(value, context=context)
    if pm_ev:
        evidence["payment_method"] = pm_ev

    payment_source, ps_ev = _first_axis_value(value, PAYMENT_SOURCE_MARKERS)
    if ps_ev:
        evidence["payment_source"] = ps_ev

    enrollment_vs_recording = _enrollment_vs_recording(value, context=context)
    transfer_sense = _transfer_sense(value, context=context)
    asks_place = has_any_marker(value, TOPIC_MARKERS["address"])
    refund_frame, refund_ev = _refund_frame(value, context=context)
    if refund_ev:
        evidence["refund_frame"] = refund_ev
    discount_scope, discount_ev = _first_axis_value(value, DISCOUNT_SCOPE_MARKERS)
    if discount_ev:
        evidence["discount_scope"] = discount_ev
    camp_scope, camp_ev = _single_axis_value(value, CAMP_SCOPE_MARKERS)
    if camp_ev:
        evidence["camp_scope"] = camp_ev
    online_track, track_ev = _first_axis_value(value, ONLINE_TRACK_MARKERS)
    if track_ev:
        evidence["online_track"] = track_ev
    schedule_scope, schedule_ev = _first_axis_value(value, SCHEDULE_SCOPE_MARKERS)
    if schedule_ev:
        evidence["schedule_scope"] = schedule_ev

    topics: list[str] = []
    for topic, markers in TOPIC_MARKERS.items():
        if has_any_marker(value, markers) and not _topic_markers_negated(value, markers):
            topics.append(topic)
    if payment_method in {"rassrochka", "dolyami"} and transfer_sense != "money":
        topics.append("installment")
    if payment_method == "invoice_monthly":
        topics.append("payment_method")
    if transfer_sense == "money":
        topics.append("payment_method")
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

    return MessageRoles(
        training_format=training_format,
        training_formats=tuple(training_formats if fmt_ev.startswith("multi:") else ((training_format,) if training_format else ())),
        enrollment_vs_recording=enrollment_vs_recording,
        transfer_sense=transfer_sense,
        payment_method=payment_method,
        payment_source=payment_source,
        asks_place=asks_place,
        refund_frame=refund_frame,
        discount_scope=discount_scope,
        camp_scope=camp_scope,
        online_track=online_track,
        schedule_scope=schedule_scope,
        topics=tuple(dict.fromkeys(topics)),
        evidence=evidence,
    )


def _first_axis_value(text: str, table: Mapping[str, tuple[str, ...]]) -> tuple[str, str]:
    for value, markers in table.items():
        for marker in markers:
            if has_marker(text, marker):
                if intent_state_repair_enabled() and _marker_negated(text, marker):
                    continue
                return value, marker
    return "", ""


def _payment_method_value(text: str, *, context: Mapping[str, object] | None = None) -> tuple[str, str]:
    if intent_state_repair_enabled() and _is_invoice_monthly_payment(text, context=context):
        return "invoice_monthly", "invoice_monthly"
    return _first_axis_value(text, PAYMENT_METHOD_MARKERS)


def _single_axis_value(text: str, table: Mapping[str, tuple[str, ...]]) -> tuple[str, str]:
    """Return one axis value unless the client is comparing alternatives.

    Example: "онлайн или очно?" is a question about available formats, not a
    confirmed choice. In that case we must not latch either format into memory.
    """

    matched: list[tuple[str, str]] = []
    negated_values: list[str] = []
    for value, markers in table.items():
        for marker in markers:
            if has_marker(text, marker):
                if _marker_negated(text, marker):
                    negated_values.append(value)
                    continue
                matched.append((value, marker))
                break
    if not matched:
        if negated_values:
            return "", "negated:" + "/".join(dict.fromkeys(negated_values))
        return "", ""
    if len(matched) > 1 and has_marker(text, "или"):
        return "", "ambiguous_question:" + "/".join(value for value, _ in matched)
    return matched[0]


def _multi_axis_values(text: str, table: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    values: list[str] = []
    for value, markers in table.items():
        if any(has_marker(text, marker) and not _marker_negated(text, marker) for marker in markers):
            values.append(value)
    return tuple(dict.fromkeys(values))


def _explicit_multi_format_request(text: str, formats: Sequence[str]) -> bool:
    if len(set(formats)) < 2:
        return False
    value = " ".join(str(text or "").casefold().replace("ё", "е").split())
    if has_marker(value, "или") and not has_any_marker(value, ("оба", "оба формата", "оба варианта", "и то и другое")):
        return False
    return bool(
        has_any_marker(value, ("оба", "оба формата", "оба варианта", "и то и другое", "и тот и другой", "пусть оба", "просила оба"))
        or re.search(r"\bи\s+очно\b.*\bи\s+онлайн\b|\bи\s+онлайн\b.*\bи\s+очно\b", value)
        or re.search(r"\bочно\b.*\bи\s+онлайн\b|\bонлайн\b.*\bи\s+очно\b", value)
    )


def _is_invoice_monthly_payment(text: str, *, context: Mapping[str, object] | None = None) -> bool:
    value = " ".join(str(text or "").casefold().replace("ё", "е").split())
    monthly = has_any_marker(value, INVOICE_MONTHLY_MARKERS)
    invoice_or_transfer = has_any_marker(value, INVOICE_TRANSFER_MARKERS)
    negates_installment = has_any_marker(value, NOT_INSTALLMENT_MARKERS)
    if monthly and (invoice_or_transfer or negates_installment):
        return True
    if monthly and context:
        active_topics_raw = context.get("active_topics") or ()
        active_topics = {
            str(item)
            for item in active_topics_raw
            if str(item).strip()
        } if isinstance(active_topics_raw, Sequence) and not isinstance(active_topics_raw, (str, bytes, bytearray)) else set()
        return str(context.get("last_transfer_sense") or "") == "money" or "payment_method" in active_topics
    return False


def _marker_negated(text: str, marker: str) -> bool:
    normalized = " ".join(str(text or "").casefold().replace("ё", "е").split())
    needle = " ".join(str(marker or "").casefold().replace("ё", "е").split())
    if not needle:
        return False
    return bool(
        has_marker(normalized, marker)
        and (
            f"не {needle}" in normalized
            or f"не {needle}," in normalized
            or f"не {needle}." in normalized
            or f"не про {needle}" in normalized
            or f"не о {needle}" in normalized
            or f"это не {needle}" in normalized
            or f"это не про {needle}" in normalized
            or f"только не {needle}" in normalized
            or f"не в {needle}" in normalized
        )
    )


def intent_state_repair_enabled() -> bool:
    raw = os.getenv(INTENT_STATE_REPAIR_ENV)
    if raw is None:
        return True
    return str(raw).strip().casefold() not in {"0", "false", "no", "off"}


def _enrollment_vs_recording(text: str, *, context: Mapping[str, object] | None = None) -> str:
    has_zapis = has_marker(text, "запис")
    enroll_verb = has_any_marker(text, ("записаться", "записать", "оформ"))
    near_recording = has_any_marker(text, RECORDING_NEIGHBORS)
    near_enroll = has_any_marker(text, ENROLL_NEIGHBORS)
    strong_recording = has_any_marker(text, STRONG_RECORDING_MARKERS)
    if _recording_followup_from_context(text, context=context) and not enroll_verb and not near_enroll:
        return "recording"
    if not (has_zapis or enroll_verb or strong_recording):
        return ""
    if enroll_verb or near_enroll:
        return "enroll"
    if near_recording:
        return "recording"
    return "enroll"


def _recording_followup_from_context(text: str, *, context: Mapping[str, object] | None = None) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    if not context:
        return False
    active_scope = str(context.get("active_fact_scope") or "")
    active_topics_raw = context.get("active_topics") or ()
    active_topics = {str(item) for item in active_topics_raw} if isinstance(active_topics_raw, Sequence) and not isinstance(active_topics_raw, (str, bytes, bytearray)) else set()
    recording_context = active_scope in {"online_recordings", "offline_recordings"} or "recording" in active_topics
    if not recording_context:
        return False
    asks_location_of_previous_material = bool(
        (
            has_any_marker(value, ("где", "куда", "как"))
            and has_any_marker(value, ("ее", "её", "это", "запис", "смотреть", "посмотреть", "открыть", "найти", "доступ"))
        )
        or (
            has_any_marker(value, ("запис", "материал", "урок", "вебинар"))
            and has_any_marker(value, ("смотреть", "посмотреть", "личн кабинет", "личном кабинете", "кабинет", "ссылк", "пришл", "доступ"))
        )
    )
    return asks_location_of_previous_material


def _transfer_sense(text: str, *, context: Mapping[str, object] | None = None) -> str:
    if not has_any_marker(text, ("перевод", "перевест", "перевед", "переведите", "переключите")):
        return ""
    if has_any_marker(text, TRANSFER_MANAGER_NEIGHBORS):
        return "manager"
    if has_any_marker(text, TRANSFER_MONEY_NEIGHBORS):
        return "money"
    if has_any_marker(text, TRANSFER_GROUP_NEIGHBORS):
        return "group"
    if context:
        last = str(context.get("last_transfer_sense") or "")
        if last:
            return last
        if context.get("group_topic_active"):
            return "group"
    return ""


def _refund_frame(text: str, *, context: Mapping[str, object] | None = None) -> tuple[str, str]:
    if has_any_marker(text, REFUND_BENIGN_NON_REFUND) or has_any_marker(text, REFUND_BENIGN_OBJECTS):
        return "none", ""
    demand_hit = next((m for m in REFUND_DEMAND if has_marker(text, m)), "")
    paid_hit = next((m for m in REFUND_POST_PAYMENT if has_marker(text, m)), "")
    if _is_tax_deduction_return_question(text) and not demand_hit and not paid_hit:
        return "none", "tax_deduction_return"
    if intent_state_repair_enabled() and not demand_hit and not paid_hit and is_negated_refund_topic(text):
        return "none", "negated_refund_topic"
    process_hit = next((m for m in REFUND_POLICY_PROCESS_MARKERS if has_marker(text, m)), "")
    context_has_presale_refund = _context_has_presale_refund(context)
    mentions_refund = has_any_marker(text, REFUND_MENTION) or bool(demand_hit) or (
        bool(process_hit) and context_has_presale_refund
    )
    if not mentions_refund:
        return "none", ""
    if demand_hit:
        return "dispute", f"demand:{demand_hit}"
    if process_hit and not has_any_marker(text, ("уже оплат", "оплатил", "оплатила", "списали", "сняли")):
        return "presale_policy", f"presale_process:{process_hit}"
    frame_hit = next((m for m in REFUND_PRESALE_FRAME if has_marker(text, m)), "")
    if frame_hit and not has_any_marker(text, ("уже оплат", "оплатил", "оплатила", "списали", "сняли")):
        return "presale_policy", f"presale_frame:{frame_hit}"
    if paid_hit:
        return "dispute", f"post_payment:{paid_hit}"
    return "dispute", "bare_refund_mention"


def _context_has_presale_refund(context: Mapping[str, object] | None) -> bool:
    if not context:
        return False
    topics = context.get("active_topics")
    if isinstance(topics, Sequence) and not isinstance(topics, (str, bytes, bytearray)):
        if any(str(item) == "refund_presale" for item in topics):
            return True
    return str(context.get("active_fact_scope") or "") == "refund_policy"


def _is_tax_deduction_return_question(text: str) -> bool:
    value = str(text or "")
    if not has_any_marker(value, PAYMENT_SOURCE_MARKERS["tax_deduction"]):
        return False
    if has_any_marker(value, ("верните", "требую", "хочу деньги назад", "деньги назад", "отдайте", "забрать деньги")):
        return False
    if has_any_marker(value, ("возврат оплат", "возврат денег за курс", "вернуть оплат", "расторг", "уже оплат", "оплатил", "оплатила", "списали")):
        return False
    return has_any_marker(value, ("вернуть", "вернут", "вернё", "возврат"))


def is_negated_refund_topic(text: object) -> bool:
    value = " ".join(str(text or "").casefold().replace("ё", "е").split())
    if not has_marker(value, "возврат"):
        return False
    if not re.search(r"(?:\bне\s+(?:про|о)\s+|\bэто\s+не\s+(?:про\s+)?|\bне\s+)возврат\w*", value):
        return False
    return not has_any_marker(
        value,
        (
            "верните",
            "требую",
            "хочу вернуть",
            "хочу возврат",
            "хочу деньги назад",
            "вернуть деньги",
            "деньги назад",
            "отдайте",
            "забрать деньги",
            "расторгнуть",
            "уже оплат",
            "оплатил",
            "оплатила",
            "списали",
            "сняли",
        ),
    )


def _topic_markers_negated(text: str, markers: Sequence[str]) -> bool:
    found = [marker for marker in markers if has_marker(text, marker)]
    return bool(found) and all(_marker_negated(text, marker) for marker in found)
