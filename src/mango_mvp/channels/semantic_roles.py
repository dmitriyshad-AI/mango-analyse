from __future__ import annotations

"""Typed semantic roles for a client message.

This module keeps word-boundary signal matching in one place and exposes
orthogonal axes such as training format, payment method/source and refund frame.
It is intentionally structural: generation still answers from facts and context.
"""

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from mango_mvp.channels.text_signals import has_any_marker, has_marker


SEMANTIC_ROLES_SCHEMA_VERSION = "semantic_roles_v1_2026_05_25"

FORMAT_MARKERS: dict[str, tuple[str, ...]] = {
    "online": ("онлайн", "дистанц", "вебинар", "удаленно", "удалённо", "из дома", "мтс", "линк", "link"),
    "ochno": ("очно", "очный", "очных", "офлайн", "в классе", "в аудитор", "пацаева", "сретен", "красносель"),
    "vyezd_camp": ("лагер", "лвш", "смена", "смену", "выездн", "менделеево", "проживан", "с проживанием"),
}

PAYMENT_METHOD_MARKERS: dict[str, tuple[str, ...]] = {
    "dolyami": ("долями",),
    "rassrochka": ("рассроч", "частями", "помесяч", "в рассрочку", "по месяцам"),
    "edinorazovo": ("единоразово", "сразу всю", "сразу всю сумму", "целиком", "полностью оплат", "одним платеж"),
}

PAYMENT_SOURCE_MARKERS: dict[str, tuple[str, ...]] = {
    "matkap": ("маткап", "материнск", "материнским", "сфр"),
    "tax_deduction": ("налоговый вычет", "вычет", "фнс", "кнд", "3-ндфл", "3 ндфл"),
    "sertifikat": ("сертификат", "сертификатом"),
}

TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    "price": ("цен", "стои", "сколько стоит", "прайс", "руб", "почем"),
    "discount": ("скид", "акци", "льгот", "процент", "суммир", "дешевле"),
    "trial": ("пробн", "пробное", "фрагмент", "попроб"),
    "camp": ("лагер", "лвш", "лш", "летняя школа", "смена", "менделеево", "проживан", "питан", "трансфер"),
    "schedule": ("распис", "во сколько", "по каким дням", "когда занят", "дни занят", "время занят", "раз в неделю"),
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
    "olympiad_online": ("олимпиад", "физтех", "рсош", "перечнев"),
    "regular_online": ("обычн", "регулярн", "не олимпиад", "онлайн-курс", "онлайн курс"),
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
REFUND_BENIGN_NON_REFUND = ("возврат к теме", "вернусь позже", "вернемся позже", "вернёмся позже")


@dataclass(frozen=True)
class MessageRoles:
    training_format: str = ""
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
    if fmt_ev:
        evidence["training_format"] = fmt_ev

    payment_method, pm_ev = _first_axis_value(value, PAYMENT_METHOD_MARKERS)
    if pm_ev:
        evidence["payment_method"] = pm_ev

    payment_source, ps_ev = _first_axis_value(value, PAYMENT_SOURCE_MARKERS)
    if ps_ev:
        evidence["payment_source"] = ps_ev

    enrollment_vs_recording = _enrollment_vs_recording(value)
    transfer_sense = _transfer_sense(value, context=context)
    asks_place = has_any_marker(value, TOPIC_MARKERS["address"])
    refund_frame, refund_ev = _refund_frame(value)
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

    return MessageRoles(
        training_format=training_format,
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
                return value, marker
    return "", ""


def _single_axis_value(text: str, table: Mapping[str, tuple[str, ...]]) -> tuple[str, str]:
    """Return one axis value unless the client is comparing alternatives.

    Example: "онлайн или очно?" is a question about available formats, not a
    confirmed choice. In that case we must not latch either format into memory.
    """

    matched: list[tuple[str, str]] = []
    for value, markers in table.items():
        for marker in markers:
            if has_marker(text, marker):
                matched.append((value, marker))
                break
    if not matched:
        return "", ""
    if len(matched) > 1 and has_marker(text, "или"):
        return "", "ambiguous_question:" + "/".join(value for value, _ in matched)
    return matched[0]


def _enrollment_vs_recording(text: str) -> str:
    has_zapis = has_marker(text, "запис")
    enroll_verb = has_any_marker(text, ("записаться", "записать", "оформ"))
    near_recording = has_any_marker(text, RECORDING_NEIGHBORS)
    near_enroll = has_any_marker(text, ENROLL_NEIGHBORS)
    strong_recording = has_any_marker(text, STRONG_RECORDING_MARKERS)
    if not (has_zapis or enroll_verb or strong_recording):
        return ""
    if enroll_verb or near_enroll:
        return "enroll"
    if near_recording:
        return "recording"
    return "enroll"


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


def _refund_frame(text: str) -> tuple[str, str]:
    if has_any_marker(text, REFUND_BENIGN_NON_REFUND):
        return "none", ""
    demand_hit = next((m for m in REFUND_DEMAND if has_marker(text, m)), "")
    paid_hit = next((m for m in REFUND_POST_PAYMENT if has_marker(text, m)), "")
    mentions_refund = has_any_marker(text, REFUND_MENTION) or bool(demand_hit)
    if not mentions_refund:
        return "none", ""
    if demand_hit:
        return "dispute", f"demand:{demand_hit}"
    if paid_hit:
        return "dispute", f"post_payment:{paid_hit}"
    frame_hit = next((m for m in REFUND_PRESALE_FRAME if has_marker(text, m)), "")
    if frame_hit:
        return "presale_policy", f"presale_frame:{frame_hit}"
    return "dispute", "bare_refund_mention"
