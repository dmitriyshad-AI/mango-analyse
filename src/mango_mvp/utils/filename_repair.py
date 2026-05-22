from __future__ import annotations

import unicodedata
from pathlib import Path

KNOWN_MANAGER_NAME_REPAIRS = {
    "ТаЃѓЃҐ ОЂ•£": "Тропов Олег",
    "Тово≠≠®™ АЂ•™б†≠§а": "Тютюнник Александр",
    "ТаЃѓ®≠† А≠≠†": "Тропина Анна",
    "КЃаиг≠ЃҐ† А≠†бв†б®п": "Коршунова Анастасия",
    "Л•Ѓ≠ЃҐ АЂ•™б•©": "Леонов Алексей",
    "КЃІЂЃҐ† Е™†в•а®≠†": "Козлова Екатерина",
    "КЂлз•Ґ† Д†амп": "Клычева Дарья",
    "ГЃЂЃҐз•≠™Ѓ К†а®≠†": "Головченко Карина",
    "А§ђ®≠®бва†вЃа ФЃвЃ≠": "Администратор Фотон",
    "Га†дЃҐ† ПЃЂ®≠†": "Графова Полина",
    "ХЃЂЃ§®ЂЃҐ† Д†амп": "Холодилова Дарья",
    "ПаЃеЃаЃҐ† Т†вмп≠†": "Прохорова Татьяна",
    "ОЂм£†": "Ольга",
    "Б†вга®≠ СҐпвЃбЂ†Ґ": "Батурин Святослав",
    "Дђ®ва®© Ф†°†а®бЃҐ": "Дмитрий Фабарисов",
    "КЃѓЃв•Ґ† ЕҐ†": "Копотева Ева",
    "Шђл£Ђ•Ґ† ПЃЂ®≠†": "Шмыглева Полина",
    "ХЂ•°≠®™ЃҐ† А≠†бв†б®п": "Хлебникова Анастасия",
    "Каг£ЂЃҐ М†™б®ђ": "Круглов Максим",
    "З•Ђ•≠о™ Д†а®≠† КЃ≠бв†≠в®≠ЃҐ≠†": "Зеленюк Дарина Константиновна",
    "Б†а™ЃҐ К®а®ЂЂ": "Барков Кирилл",
    "Ч•аҐп™ЃҐ РЃ§®Ѓ≠ АЂ•™б••Ґ®з": "Червяков Родион Алексеевич",
    "БЃз™†а•Ґ† М†и†": "Бочкарева Маша",
    "СЃІ®≠ЃҐ П†Ґ•Ђ": "Созинов Павел",
    "ИҐ†≠®ЂЃҐ М†™б®ђ": "Иванилов Максим",
    "Ад†≠†бм•Ґ Аавсђ": "Афанасьев Артём",
    "Х†аЂ†ђЃҐ М†™б®ђ": "Харламов Максим",
    "Бга≠†иЃҐ Дђ®ва®©": "Бурнашов Дмитрий",
    # Operational alias: the same manager appeared in source data under an old surname.
    "Леонова Анна": "Тропина Анна",
}


def _normalize_text(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


NORMALIZED_MANAGER_NAME_REPAIRS = {
    _normalize_text(bad): _normalize_text(good)
    for bad, good in KNOWN_MANAGER_NAME_REPAIRS.items()
}


def repair_manager_name(value: str | None) -> str | None:
    text = _normalize_text(value)
    if not text:
        return None
    return NORMALIZED_MANAGER_NAME_REPAIRS.get(text, text)


def repair_text_manager_names(value: str | None) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    repaired = text
    for bad, good in NORMALIZED_MANAGER_NAME_REPAIRS.items():
        repaired = repaired.replace(bad, good)
    return _normalize_text(repaired)


def repair_filename_display(value: str | None) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    path = Path(text)
    stem = path.stem
    suffix = path.suffix
    parts = stem.split("__")
    repaired_parts = [repair_manager_name(part) or part for part in parts]
    return _normalize_text("__".join(repaired_parts) + suffix)
