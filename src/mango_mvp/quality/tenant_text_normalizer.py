from __future__ import annotations

import re
from typing import Any, Iterable


BRAND_ALIASES_RE = re.compile(
    r"\b(?:[А-ЯA-Z]?МПК|(?!УНПК)[А-ЯA-Z]?НПК|О\s*Н\s*П\s*К|Н\s*П\s*К|УНФК|УНП|ЛНПК)\s*"
    r"М\s*[ФШ]\s*[ТД]?\s*[ИI]\b",
    re.IGNORECASE,
)

SUMMER_NIGHT_SCHOOL_PATTERNS = (
    (re.compile(r"\bлетн(?:яя|ей|юю)\s+ночн(?:ая|ой|ую)\s+школ\w*", re.IGNORECASE), "летняя очная школа"),
    (re.compile(r"\bлетн(?:ие|их|ими)\s+ночн(?:ые|ых|ыми)\s+школ\w*", re.IGNORECASE), "летние очные школы"),
)

COUNT_SUFFIX_RE = re.compile(r"\s*\(\s*\d+\s+касани[йя]\s*\)\s*$", re.IGNORECASE)
COUNTED_LABEL_RE = re.compile(r"^(?P<label>.+?)\s*:\s*\d+\s*$")
DATE_PREFIX_RE = re.compile(r"^\s*\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\s*:\s*")


def normalize_manager_text(value: Any) -> str:
    """Normalize tenant-specific ASR/LLM artifacts in manager-facing CRM text."""
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    text = BRAND_ALIASES_RE.sub("УНПК МФТИ", text)
    for pattern, replacement in SUMMER_NIGHT_SCHOOL_PATTERNS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_product_label(value: Any) -> str:
    text = normalize_manager_text(_strip_count(value)).strip(" .;,")
    if not text:
        return ""
    key = text.casefold()
    if "летн" in key and "школ" in key and ("очн" in key or "ночн" in key):
        return "летняя очная школа"
    if key in {"летняя школа", "летние школы", "летняя летняя школа"}:
        return "летняя очная школа"
    if ("летн" in key and ("лагер" in key or "выездн" in key)) or key in {"лвш", "летняя выездная школа"}:
        return "летний лагерь"
    if "индивидуаль" in key:
        return "индивидуальные занятия"
    if "годов" in key:
        return "годовые курсы"
    return text


def format_product_list(values: Any | Iterable[Any], *, max_items: int | None = None) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for item in _iter_items(values):
        normalized = normalize_product_label(item)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if max_items is not None and len(result) >= max_items:
            break
    return " | ".join(result)


def normalize_objection_label(value: Any) -> str:
    text = normalize_manager_text(_strip_count(value)).strip(" .;,")
    if not text:
        return ""
    text = re.sub(r"^(?:Актуальные|Исторические)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = DATE_PREFIX_RE.sub("", text).strip()
    key = text.casefold()
    if "муж" in key and ("договор" in key or "обсуд" in key or "прочит" in key):
        return "нужно согласовать договор с мужем"
    return text


def objection_key(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_objection_label(value).casefold()).strip(" .;,")


def format_objection_list(values: Any | Iterable[Any], *, max_items: int | None = None) -> str:
    result: list[str] = []
    seen: set[str] = set()
    for item in _iter_items(values):
        normalized = normalize_objection_label(item)
        key = objection_key(normalized)
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if max_items is not None and len(result) >= max_items:
            break
    return " | ".join(result)


def _strip_count(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    text = COUNT_SUFFIX_RE.sub("", text)
    match = COUNTED_LABEL_RE.match(text)
    if match:
        text = match.group("label").strip()
    return text


def _iter_items(values: Any | Iterable[Any]) -> Iterable[str]:
    if values is None:
        return
    if isinstance(values, str):
        raw_values: Iterable[Any] = [values]
    else:
        raw_values = values
    for raw in raw_values:
        text = "" if raw is None else str(raw)
        for part in re.split(r"\s*\|\s*|;\s*", text.replace("\n", " | ")):
            value = part.strip(" .;,")
            if value:
                yield value
