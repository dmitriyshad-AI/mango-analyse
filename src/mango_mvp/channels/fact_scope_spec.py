from __future__ import annotations

from typing import Sequence

from mango_mvp.channels.text_signals import has_any_marker, normalize_signal_text


FACT_SCOPE_SPEC_SCHEMA_VERSION = "fact_scope_spec_v1_2026_05_24"

SCOPE_FAMILIES: dict[str, frozenset[str]] = {
    "matkap_process": frozenset({"matkap_process", "tax_deduction"}),
    "tax_deduction": frozenset({"matkap_process", "tax_deduction"}),
    "matkap_age_limit": frozenset({"matkap_process", "matkap_age_limit", "tax_deduction"}),
    "class_schedule": frozenset({"class_schedule", "office_hours", "discount_second_subject", "discount_multichild", "discount_stacking"}),
    "office_hours": frozenset({"class_schedule", "office_hours"}),
    "payment_methods": frozenset({"payment_methods", "office_hours", "camp_extra_facts"}),
    "city_day_camp": frozenset({"city_day_camp", "residential_lvsh"}),
    "residential_lvsh": frozenset({"city_day_camp", "residential_lvsh"}),
    "program_subjects": frozenset({"program_subjects", "class_schedule", "city_day_camp", "residential_lvsh"}),
    "discount_second_subject": frozenset({"discount_second_subject", "discount_multichild", "discount_stacking", "discount_referral", "installment_bank", "dolyami_parts"}),
    "discount_multichild": frozenset({"discount_second_subject", "discount_multichild", "discount_stacking", "discount_referral", "installment_bank", "dolyami_parts"}),
    "discount_stacking": frozenset({"discount_second_subject", "discount_multichild", "discount_stacking", "discount_referral"}),
    "discount_referral": frozenset({"discount_second_subject", "discount_multichild", "discount_stacking", "discount_referral"}),
    "trial_offline": frozenset({"trial_offline", "trial_online_fragment"}),
    "trial_online_fragment": frozenset({"trial_offline", "trial_online_fragment"}),
    "offline_recordings": frozenset({"offline_recordings", "camp_extra_facts", "online_recordings"}),
    "camp_extra_facts": frozenset({"offline_recordings", "camp_extra_facts", "online_recordings"}),
    "online_recordings": frozenset({"offline_recordings", "camp_extra_facts", "online_recordings"}),
    "dolyami_parts": frozenset({"dolyami_parts", "installment_bank", "program_subjects", "class_schedule"}),
    "installment_bank": frozenset({"dolyami_parts", "installment_bank", "program_subjects", "class_schedule"}),
    "regular_online": frozenset({"regular_online", "olympiad_online"}),
    "olympiad_online": frozenset({"regular_online", "olympiad_online"}),
    "refund_policy": frozenset({"refund_policy", "office_hours", "class_schedule", "payment_methods"}),
}

_SCOPE_MARKERS: dict[str, tuple[str, ...]] = {
    "matkap_process": ("маткап", "материн", "сфр", "сертификат материн"),
    "matkap_age_limit": ("до 25 лет", "25 лет", "исполнилось 3 года", "3 года", "возрастной лимит"),
    "tax_deduction": ("налог", "вычет", "фнс", "13%", "110 000", "14300", "14 300"),
    "payment_methods": ("способ оплаты", "по счету", "по счёту", "банковский перевод", "реквизит", "ссылка на оплату", "как оплатить"),
    "office_hours": ("телефон", "email", "почта", "график: пн", "пн-вс", "часы работы", "работаем с"),
    "class_schedule": ("расписание занятий", "занятия проходят", "уроки проходят", "группа проходит", "вебинар"),
    "city_day_camp": ("городск", "дневн", "без проживания", "без прожив", "без ночев", "долгопрудн", "красносельск", "пн-пт", "пн–пт"),
    "residential_lvsh": ("лвш", "lvsh", "менделеево", "mendeleevo", "выездн", "прожив", "5-раз", "питан", "трансфер"),
    "program_subjects": ("предметы:", "список предметов", "математика", "физика", "информатика", "русский язык", "химия", "английский"),
    "discount_second_subject": (
        "второй предмет",
        "вторым предметом",
        "второго предмета",
        "второй онлайн-предмет",
        "второй онлайн предмет",
        "последующий предмет",
        "2-й предмет",
        "тот же ребенок",
        "того же ребенка",
        "одного ребенка",
    ),
    "discount_multichild": ("многодет", "удостоверение", "детей из", "ребенок из многодетной", "семьи учится"),
    "discount_stacking": ("суммир", "складыв", "выбирается одна", "одна скидка", "наибольшая", "не суммируются"),
    "discount_referral": ("приведи друга", "приглашенный друг", "приглашённый друг", "друг оплатит", "кэшбэк"),
    "trial_offline": (
        "очное пробное",
        "очного пробного",
        "пробное очно",
        "очно пробное",
        "пробному очному",
        "бесплатному пробному",
        "free_trial_offline",
        "trial_offline",
        "пацаева и в мфти",
    ),
    "trial_online_fragment": ("онлайн-формату", "онлайн формату", "фрагмент занятия", "фрагмент урока", "фрагмент"),
    "offline_recordings": (
        "запись очных",
        "записи очных",
        "записи по очным",
        "запись урока",
        "запись занятия",
        "запись очного",
        "пропущенные материалы",
        "материалы можно запросить",
        "чат группы",
        "offline_recordings",
    ),
    "camp_extra_facts": ("кружк", "медсестр", "ноутбук", "лагер", "лвш", "смен", "менделеево", "прожив", "питан", "трансфер"),
    "online_recordings": ("запись каждого урока", "записи уроков", "записи занятий", "онлайн-занятия", "мтс линк"),
    "dolyami_parts": ("долями", "части без", "4 части", "четыре части"),
    "installment_bank": ("рассроч", "т-банк", "банк", "6, 10 или 12", "6/10/12", "месяцев"),
    "regular_online": ("обычный онлайн", "обычного онлайн", "регулярный онлайн", "не олимпиадный онлайн", "онлайн 5-11", "онлайн-курс"),
    "olympiad_online": (
        "олимпиадная подготовка",
        "олимпиадный онлайн",
        "физтех онлайн",
        "online_olympiad",
        "olympiad_phystech",
        "phystech",
        "перечнев",
        "рсош",
    ),
    "refund_policy": (
        "условия возврата",
        "правила возврата",
        "порядок возврата",
        "как оформить возврат",
        "заявление на возврат",
    ),
}

ANSWER_COMPATIBLE_NEIGHBOR_SCOPES: dict[str, frozenset[str]] = {
    "installment_bank": frozenset({"dolyami_parts"}),
    "dolyami_parts": frozenset({"installment_bank"}),
    "discount_stacking": frozenset({"discount_second_subject", "discount_multichild"}),
}

FACT_COMPATIBLE_NEIGHBOR_SCOPES: dict[str, frozenset[str]] = {
    "discount_stacking": frozenset({"discount_second_subject", "discount_multichild"}),
    "discount_second_subject": frozenset({"discount_stacking"}),
    "discount_multichild": frozenset({"discount_stacking"}),
    "trial_offline": frozenset({"trial_online_fragment"}),
}


def normalize_scope_text(value: object) -> str:
    return normalize_signal_text(value)


def blocked_neighbors_for(scope: str) -> tuple[str, ...]:
    cleaned = str(scope or "").strip()
    if not cleaned:
        return ()
    return tuple(sorted(item for item in SCOPE_FAMILIES.get(cleaned, frozenset({cleaned})) if item != cleaned))


def scope_family_for(scope: str) -> frozenset[str]:
    cleaned = str(scope or "").strip()
    return SCOPE_FAMILIES.get(cleaned, frozenset({cleaned}) if cleaned else frozenset())


def detect_fact_scopes(text: object, *, fact_types: Sequence[str] = ()) -> set[str]:
    value = normalize_scope_text(text)
    fact_type_set = {normalize_scope_text(item) for item in fact_types}
    scopes: set[str] = set()
    if "contact" in fact_type_set:
        scopes.add("office_hours")
    for scope, markers in _SCOPE_MARKERS.items():
        if has_any_marker(value, markers):
            scopes.add(scope)
    if "camp_lvsh" in fact_type_set:
        scopes.add("residential_lvsh")
    if "camp_city" in fact_type_set:
        scopes.add("city_day_camp")
    if "installment" in fact_type_set:
        scopes.add("installment_bank")
    if "discount" in fact_type_set:
        if has_any_marker(value, ("второй предмет", "second_subject")):
            scopes.add("discount_second_subject")
        if has_any_marker(value, ("многодет", "multichild")):
            scopes.add("discount_multichild")
    if "city_day_camp" in scopes and "residential_lvsh" in scopes:
        no_lodging = has_any_marker(value, ("без проживания", "без прожив", "без ночев", "дневной формат"))
        negated_residential = has_any_marker(
            value,
            (
                "не лвш",
                "не выезд",
                "не подставляю",
                "не смешиваю",
                "не про лвш",
                "не про выезд",
            ),
        )
        positive_residential = has_any_marker(value, ("менделеево", "mendeleevo", "трансфер")) or (
            has_any_marker(value, ("лвш", "lvsh", "выездн")) and not negated_residential
        )
        positive_lodging = has_any_marker(value, ("с прожив", "проживание включ", "проживание и питание")) and not no_lodging
        if negated_residential or not (positive_residential or positive_lodging):
            scopes.discard("residential_lvsh")
    return scopes


def fact_scopes_allowed(record_scopes: set[str], *, requested_scope: str = "", blocked_neighbor_scopes: Sequence[str] = ()) -> bool:
    requested = str(requested_scope or "").strip()
    blocked = {str(item) for item in blocked_neighbor_scopes if str(item).strip()}
    if requested and requested in record_scopes:
        return True
    if requested:
        compatible = set(FACT_COMPATIBLE_NEIGHBOR_SCOPES.get(requested, frozenset()))
        if compatible and record_scopes and record_scopes <= compatible:
            if record_scopes & blocked:
                return False
            return True
        blocked -= compatible
    if record_scopes & blocked:
        return False
    if requested and not record_scopes:
        return True
    if not requested:
        return True
    family = scope_family_for(requested)
    if family and record_scopes & family:
        return requested in record_scopes
    return False


def answer_scopes_allowed(answer_scopes: set[str], *, requested_scope: str = "", blocked_neighbor_scopes: Sequence[str] = ()) -> bool:
    requested = str(requested_scope or "").strip()
    blocked = {str(item) for item in blocked_neighbor_scopes if str(item).strip()}
    if requested and requested in answer_scopes:
        blocked -= set(ANSWER_COMPATIBLE_NEIGHBOR_SCOPES.get(requested, frozenset()))
    return not (answer_scopes & blocked)
