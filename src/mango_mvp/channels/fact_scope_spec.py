from __future__ import annotations

from typing import Sequence


FACT_SCOPE_SPEC_SCHEMA_VERSION = "fact_scope_spec_v1_2026_05_24"

SCOPE_FAMILIES: dict[str, frozenset[str]] = {
    "matkap_process": frozenset({"matkap_process", "tax_deduction"}),
    "tax_deduction": frozenset({"matkap_process", "tax_deduction"}),
    "class_schedule": frozenset({"class_schedule", "office_hours"}),
    "office_hours": frozenset({"class_schedule", "office_hours"}),
    "city_day_camp": frozenset({"city_day_camp", "residential_lvsh"}),
    "residential_lvsh": frozenset({"city_day_camp", "residential_lvsh"}),
    "discount_second_subject": frozenset({"discount_second_subject", "discount_multichild", "installment_bank", "dolyami_parts"}),
    "discount_multichild": frozenset({"discount_second_subject", "discount_multichild", "installment_bank", "dolyami_parts"}),
    "trial_offline": frozenset({"trial_offline", "trial_online_fragment"}),
    "trial_online_fragment": frozenset({"trial_offline", "trial_online_fragment"}),
    "offline_recordings": frozenset({"offline_recordings", "camp_extra_facts", "online_recordings"}),
    "camp_extra_facts": frozenset({"offline_recordings", "camp_extra_facts", "online_recordings"}),
    "online_recordings": frozenset({"offline_recordings", "camp_extra_facts", "online_recordings"}),
    "dolyami_parts": frozenset({"dolyami_parts", "installment_bank"}),
    "installment_bank": frozenset({"dolyami_parts", "installment_bank"}),
    "regular_online": frozenset({"regular_online", "olympiad_online"}),
    "olympiad_online": frozenset({"regular_online", "olympiad_online"}),
}

_SCOPE_MARKERS: dict[str, tuple[str, ...]] = {
    "matkap_process": ("маткап", "материн", "сфр", "сертификат материн"),
    "tax_deduction": ("налог", "вычет", "фнс", "13%", "110 000", "14300", "14 300"),
    "office_hours": ("телефон", "email", "почта", "график: пн", "пн-вс", "часы работы", "работаем с"),
    "class_schedule": ("расписание занятий", "занятия проходят", "уроки проходят", "группа проходит", "вебинар"),
    "city_day_camp": ("городск", "дневн", "без проживания", "без прожив", "без ночев", "долгопрудн", "красносельск", "пн-пт", "пн–пт"),
    "residential_lvsh": ("лвш", "lvsh", "менделеево", "mendeleevo", "выездн", "прожив", "5-раз", "питан", "трансфер"),
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
    "trial_offline": ("очное пробное", "очного пробного", "пробное очно", "очно пробное", "пацаева и в мфти"),
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
    "olympiad_online": ("олимпиадная подготовка", "олимпиадный онлайн", "физтех онлайн", "перечнев", "рсош"),
}

ANSWER_COMPATIBLE_NEIGHBOR_SCOPES: dict[str, frozenset[str]] = {
    "installment_bank": frozenset({"dolyami_parts"}),
    "dolyami_parts": frozenset({"installment_bank"}),
}


def normalize_scope_text(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").replace("\u00a0", " ").split())


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
        if any(marker in value for marker in markers):
            scopes.add(scope)
    if "camp_lvsh" in fact_type_set:
        scopes.add("residential_lvsh")
    if "camp_city" in fact_type_set:
        scopes.add("city_day_camp")
    if "installment" in fact_type_set:
        scopes.add("installment_bank")
    if "discount" in fact_type_set:
        if "второй предмет" in value or "second_subject" in value:
            scopes.add("discount_second_subject")
        if "многодет" in value or "multichild" in value:
            scopes.add("discount_multichild")
    return scopes


def fact_scopes_allowed(record_scopes: set[str], *, requested_scope: str = "", blocked_neighbor_scopes: Sequence[str] = ()) -> bool:
    requested = str(requested_scope or "").strip()
    blocked = {str(item) for item in blocked_neighbor_scopes if str(item).strip()}
    if requested and requested in record_scopes:
        return True
    if requested and not record_scopes:
        return False
    if record_scopes & blocked:
        return False
    if not requested:
        return True
    family = scope_family_for(requested)
    if family and record_scopes & family:
        return requested in record_scopes
    return True


def answer_scopes_allowed(answer_scopes: set[str], *, requested_scope: str = "", blocked_neighbor_scopes: Sequence[str] = ()) -> bool:
    requested = str(requested_scope or "").strip()
    blocked = {str(item) for item in blocked_neighbor_scopes if str(item).strip()}
    if requested and requested in answer_scopes:
        blocked -= set(ANSWER_COMPATIBLE_NEIGHBOR_SCOPES.get(requested, frozenset()))
    return not (answer_scopes & blocked)
