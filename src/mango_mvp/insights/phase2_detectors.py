from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


OBJECTION_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "price": (
        r"\bдорог\w*",
        r"дороговат\w*",
        r"серь[её]зн\w+\s+сумм",
        r"сумм\w+\s+серь[её]зн",
        r"не\s+по\s+карман",
        r"слишком\s+дорог",
        r"дешевле",
        r"цена\s+куса",
        r"кусается",
    ),
    "think": (
        r"\bподума\w*",
        r"надо\s+подумать",
        r"посоветуюсь",
        r"посовет\w*",
        r"не\s+гото[вв]\w*\s+решить",
        r"пока\s+думаю",
        r"верн[уё]сь\s+позже",
    ),
    "compare": (
        r"сравнива\w*",
        r"у\s+других",
        r"в\s+другом\s+центр",
        r"конкурент",
        r"чем\s+вы\s+лучше",
        r"в\s+другом\s+месте",
    ),
    "format": (
        r"не\s+увер\w+\s+в\s+формат",
        r"очно\s+или\s+онлайн",
        r"онлайн\s+или\s+очно",
        r"какой\s+формат\s+лучше",
        r"очно\s+лучше|лучше\s+очно",
        r"онлайн\s+норм",
    ),
    "distance": (
        r"\bдалеко\b",
        r"не\s+наезд",
        r"долго\s+добира",
        r"далековато",
        r"неудобно\s+ездить",
        r"сколько\s+ехать",
        r"как\s+добират",
    ),
}


ANXIETY_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "capability": (
        r"не\s+потян",
        r"не\s+справ",
        r"боюсь.{0,24}слож",
        r"слишком\s+слож",
        r"тяжело\s+будет",
        r"не\s+вытян",
        r"пережива\w*.{0,24}(слож|потян|справ)",
    ),
    "late_start": (
        r"поздно.{0,8}начин",
        r"поздно\s+уже",
        r"упустил\w*",
        r"поезд\s+уш[её]л",
        r"наверстать",
    ),
    "level_fit": (
        r"подойд[её]т\s+ли.{0,20}(реб[её]нк|дочк|сын|уровн|программ|курс|нам|ему|ей|нагрузк)",
        r"не\s+увер\w+\s+в\s+уровн",
        r"справится\s+ли\s+по\s+уровн",
        r"справится\s+ли.{0,20}(реб[её]нк|дочк|сын|он\b|она\b|ему|ей)",
        r"потянет\s+ли\s+уровень",
        r"хватит\s+ли.{0,10}знани",
    ),
}


POSITIVE_OBJECTION_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("это дороговато для нас", "price"),
    ("слишком дорого", "price"),
    ("не по карману честно", "price"),
    ("серьёзная сумма для семьи", "price"),
    ("я подумаю и вернусь", "think"),
    ("надо подумать, посоветуюсь с мужем", "think"),
    ("я сравниваю с другим центром", "compare"),
    ("а чем вы лучше других", "compare"),
    ("не уверена в формате, очно или онлайн", "format"),
    ("какой формат лучше для ОГЭ", "format"),
    ("нам далеко добираться", "distance"),
    ("долго добираться до вас", "distance"),
)


POSITIVE_ANXIETY_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("боюсь, дочка не потянет", "capability"),
    ("он не справится, слишком сложно", "capability"),
    ("не поздно ли начинать в 9 классе", "late_start"),
    ("мы кажется упустили время", "late_start"),
    ("подойдёт ли ей этот уровень", "level_fit"),
    ("не уверена, справится ли по уровню", "level_fit"),
    ("справится ли дочка", "level_fit"),
    ("хватит ли его знаний", "level_fit"),
)


NEGATIVE_EXAMPLES: tuple[str, ...] = (
    "сколько стоит годовой курс",
    "когда начинаются занятия",
    "какие документы нужны",
    "есть ли онлайн физика для 9 класса",
    "во сколько занятия по субботам",
    "как записаться на пробное",
    "какой адрес в москве",
    "можно оплатить маткапиталом",
    "у вас недорого, мне нравится",
    "расскажите про формат семинара",
    "подойдёт ли время по субботам",
    "какой уровень преподавания у вас",
    "сколько длится одно занятие",
)


def detect_objection(text: object) -> str | None:
    return _match(text, OBJECTION_PATTERNS)


def detect_anxiety(text: object) -> str | None:
    return _match(text, ANXIETY_PATTERNS)


def _match(text: object, table: Mapping[str, Sequence[str]]) -> str | None:
    value = str(text or "").casefold().replace("ё", "е")
    if not value:
        return None
    for label, patterns in table.items():
        if any(re.search(pattern, value, re.I) for pattern in patterns):
            return label
    return None


__all__ = [
    "ANXIETY_PATTERNS",
    "NEGATIVE_EXAMPLES",
    "OBJECTION_PATTERNS",
    "POSITIVE_ANXIETY_EXAMPLES",
    "POSITIVE_OBJECTION_EXAMPLES",
    "detect_anxiety",
    "detect_objection",
]
