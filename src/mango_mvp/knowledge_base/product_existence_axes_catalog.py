from __future__ import annotations

"""Product-existence axis catalog derived from KB facts.

The KB snapshot remains the source of truth. This module only derives explicit
axes for proof-like questions such as "does this course/format exist?". It is
not a live-availability checker and must not be used to claim free seats,
booking, enrollment, or payment status.
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence


PRODUCT_EXISTENCE_SCHEMA_VERSION = "product_existence_axes_catalog_v1_2026_07_02"
FOTON = "foton"
UNPK = "unpk"

EXISTENCE_FACT_TYPES = {
    "program",
    "course_parameter",
    "format",
    "camp_city",
    "camp_lvsh",
    "camp_zvsh",
    "intensive",
    "deadline",
}
EXCLUDED_FACT_TYPES = {
    "discount",
    "tax",
    "teacher",
    "documents",
    "refund",
    "payment",
    "installment",
    "matkap",
    "contact",
    "contacts",
    "promocode",
}

SUBJECT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "math": ("математ", "math"),
    "physics": ("физик", "physics"),
    "informatics": ("информат", "программирован", "informatics", "programming"),
    "russian": ("русск", "russian"),
    "ai": ("искусствен", "ai lab", "ai-lab", " ии", "нейросет"),
    "chemistry": ("хим", "chemistry"),
    "english": ("англий", "english"),
    "biology": ("биолог", "biology"),
}
FORMAT_ALIASES: Mapping[str, tuple[str, ...]] = {
    "online": ("онлайн", "online", "soholms", "soho", "дистанц"),
    "offline": ("очно", "очная", "очный", "offline", "москва", "сретенка", "долгопруд"),
}
PROGRAM_KIND_ALIASES: Mapping[str, tuple[str, ...]] = {
    "olympiad": ("олимпиад", "физтех", "рсош", "росатом", "курчатов"),
    "camp": ("лагер", "лвш", "выезд", "летн", "смен", "городская школа", "формула физтеха"),
    "regular": ("годов", "регуляр", "обычн", "курс", "группа"),
    "intensive": ("интенсив",),
}


@dataclass(frozen=True)
class ProductExistenceEntry:
    entry_id: str
    source_fact_id: str
    source_fact_key: str
    source_fact_type: str
    source_kind: str
    brand: str
    product_family: str
    program_kind: str
    format: str
    venue: str
    grade_values: tuple[int, ...]
    subjects: tuple[str, ...]
    existence_status: str
    client_safe_text: str
    valid_until: str
    structured_value: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["grade_values"] = list(self.grade_values)
        data["subjects"] = list(self.subjects)
        data["structured_value"] = dict(self.structured_value)
        return data


def build_product_existence_axes_catalog(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries: list[ProductExistenceEntry] = []
    issues: list[dict[str, Any]] = []
    for fact in facts:
        entry, issue = _entry_from_fact(fact)
        if issue:
            issues.append(issue)
        if entry is not None:
            entries.append(entry)
    return {
        "schema_version": PRODUCT_EXISTENCE_SCHEMA_VERSION,
        "rules": {
            "default_status": "unknown",
            "not_found_does_not_mean_not_offered": True,
            "live_availability_excluded": True,
            "required_for_exists": ["brand", "fresh_client_safe_fact", "scope_match"],
        },
        "entries": [entry.to_dict() for entry in _dedupe_entries(entries)],
        "issues": issues,
    }


def verify_product_format_exists(
    catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    brand: str,
    grade: int | str | None = None,
    subject: str = "",
    format: str = "",
    program_kind: str = "",
    product_family: str = "",
) -> dict[str, Any]:
    entries = _catalog_entries(catalog)
    parsed_grade = _grade_int(grade)
    axes = {
        "brand": normalize_brand(brand),
        "grade": parsed_grade,
        "subject": normalize_subject(subject),
        "format": normalize_format(format),
        "program_kind": normalize_program_kind(program_kind),
        "product_family": normalize_product_family(product_family),
    }
    missing = []
    if not axes["brand"]:
        missing.append("brand")
    if not any(axes[key] for key in ("subject", "program_kind", "product_family")):
        missing.append("subject_or_product")
    invalid = []
    if _text(grade) and parsed_grade is None:
        invalid.append("grade")
    if missing:
        return {
            "status": "needs_slot",
            "reason": "required_axis_missing",
            "missing_slots": missing,
            "invalid_slots": invalid,
            "query_axes": axes,
            "matches": [],
        }
    if invalid:
        return {
            "status": "needs_slot",
            "reason": "invalid_axis",
            "missing_slots": [],
            "invalid_slots": invalid,
            "query_axes": axes,
            "matches": [],
        }

    matching = [entry for entry in entries if _entry_matches_axes(entry, axes)]
    negative = [entry for entry in matching if entry.get("existence_status") == "not_offered"]
    positive = [entry for entry in matching if entry.get("existence_status") == "exists"]
    if positive:
        return {
            "status": "exists",
            "reason": "exact_product_existence_fact",
            "query_axes": axes,
            "entry": positive[0],
            "matches": positive,
        }
    if negative:
        return {
            "status": "not_offered",
            "reason": "explicit_not_offered_fact",
            "query_axes": axes,
            "entry": negative[0],
            "matches": negative,
        }
    return {
        "status": "unknown",
        "reason": "no_exact_product_existence_fact",
        "query_axes": axes,
        "matches": [],
    }


def normalize_brand(value: Any) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("unpk", "унпк", "мфти")):
        return UNPK
    if any(marker in text for marker in ("foton", "фотон", "цдпо")):
        return FOTON
    clean = str(value or "").strip().casefold()
    return clean if clean in {FOTON, UNPK} else ""


def normalize_subject(value: Any) -> str:
    text = _normalize_text(value)
    for subject, aliases in SUBJECT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return subject
    return ""


def normalize_format(value: Any) -> str:
    text = _normalize_text(value)
    for fmt, aliases in FORMAT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return fmt
    return ""


def normalize_program_kind(value: Any) -> str:
    text = _normalize_text(value)
    for program_kind, aliases in PROGRAM_KIND_ALIASES.items():
        if any(alias in text for alias in aliases):
            return program_kind
    return ""


def normalize_product_family(value: Any) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("лагер", "лвш", "выезд", "летн", "смен", "camp")):
        return "camp"
    if any(marker in text for marker in ("интенсив", "intensive")):
        return "intensive"
    if any(marker in text for marker in ("платформ", "soholms", "mts-link", "webinar")):
        return "platform"
    if any(marker in text for marker in ("курс", "группа", "регуляр", "годов", "олимпиад", "regular")):
        return "regular_course"
    return ""


def _entry_from_fact(fact: Mapping[str, Any]) -> tuple[ProductExistenceEntry | None, dict[str, Any] | None]:
    fact_type = _text(fact.get("fact_type"))
    fact_key = _text(fact.get("fact_key") or fact.get("fact_id") or fact.get("id"))
    fact_id = _text(fact.get("fact_id") or fact.get("id") or fact_key)
    brand = normalize_brand(fact.get("brand"))
    if not brand or not fact_key or not fact_id:
        return None, None
    if fact_type in EXCLUDED_FACT_TYPES:
        return None, None
    if fact_type not in EXISTENCE_FACT_TYPES and fact_type != "availability":
        return None, None
    if not _client_safe(fact):
        return None, None
    if not _valid_until_ok(fact.get("valid_until")):
        return None, {"issue": "stale_or_missing_valid_until", "fact_key": fact_key}

    structured = _mapping(fact.get("structured_value"))
    text = _fact_text(fact)
    haystack = _fact_haystack(fact)
    existence_status = "not_offered" if _negative_fact(fact, haystack) else "exists"
    if fact_type == "availability" and existence_status != "not_offered":
        return None, None
    if existence_status == "exists" and _positive_fact_is_operational_or_payment_like(haystack):
        return None, None

    product_family = normalize_product_family(f"{fact.get('product') or ''} {haystack}")
    program_kind = normalize_program_kind(haystack)
    if not program_kind and product_family == "camp":
        program_kind = "camp"
    if not program_kind and product_family == "regular_course":
        program_kind = "regular"
    fmt = normalize_format(structured.get("format") or haystack)
    subjects = _subjects_from_text(haystack)
    grade_values = _grade_values(fact)
    venue = _venue_from_text(haystack)

    if not (product_family or program_kind or subjects or grade_values or fmt):
        return None, {"issue": "no_product_axes", "fact_key": fact_key}
    return (
        ProductExistenceEntry(
            entry_id=_stable_entry_id("product_existence", fact_id, existence_status),
            source_fact_id=fact_id,
            source_fact_key=fact_key,
            source_fact_type=fact_type,
            source_kind="kb_fact",
            brand=brand,
            product_family=product_family,
            program_kind=program_kind,
            format=fmt,
            venue=venue,
            grade_values=tuple(grade_values),
            subjects=tuple(subjects),
            existence_status=existence_status,
            client_safe_text=text,
            valid_until=_text(fact.get("valid_until")),
            structured_value=dict(structured),
        ),
        None,
    )


def _entry_matches_axes(entry: Mapping[str, Any], axes: Mapping[str, Any]) -> bool:
    if entry.get("brand") != axes.get("brand"):
        return False
    grade = axes.get("grade")
    if grade is not None and grade not in {int(item) for item in (entry.get("grade_values") or ()) if _grade_int(item) is not None}:
        return False
    subject = str(axes.get("subject") or "")
    if subject and subject not in {str(item) for item in (entry.get("subjects") or ())}:
        return False
    fmt = str(axes.get("format") or "")
    if fmt and entry.get("format") and entry.get("format") != fmt:
        return False
    if fmt and not entry.get("format"):
        return False
    requested_program = str(axes.get("program_kind") or "")
    requested_family = str(axes.get("product_family") or "")
    if requested_program and entry.get("program_kind") and entry.get("program_kind") != requested_program:
        return False
    if requested_program and not entry.get("program_kind"):
        return False
    if requested_family and entry.get("product_family") and entry.get("product_family") != requested_family:
        if not (requested_family == "camp" and entry.get("program_kind") == "camp"):
            return False
    if requested_family and not entry.get("product_family") and entry.get("program_kind") != requested_family:
        return False
    return True


def _catalog_entries(catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(catalog, Mapping):
        return [entry for entry in (catalog.get("entries") or ()) if isinstance(entry, Mapping)]
    return [entry for entry in build_product_existence_axes_catalog(catalog).get("entries", ()) if isinstance(entry, Mapping)]


def _client_safe(fact: Mapping[str, Any]) -> bool:
    if fact.get("forbidden_for_client") is True or fact.get("internal_only") is True:
        return False
    if "allowed_for_client_answer" in fact:
        return fact.get("allowed_for_client_answer") is True
    return bool(_fact_text(fact))


def _valid_until_ok(value: Any) -> bool:
    raw = _text(value)
    if not raw:
        return False
    try:
        return date.fromisoformat(raw[:10]) >= date.today()
    except ValueError:
        return False


def _negative_fact(fact: Mapping[str, Any], haystack: str) -> bool:
    structured = _mapping(fact.get("structured_value"))
    if structured.get("negative_fact") is True:
        return True
    raw_value = structured.get("raw_value")
    if raw_value is False or _normalize_text(raw_value) in {"false", "no", "нет"}:
        return True
    return any(
        marker in haystack
        for marker in ("не запуска", "не проводится", "нет ни в каком", "группы нет", "смены нет", "отменена")
    )


def _positive_fact_is_operational_or_payment_like(haystack: str) -> bool:
    return any(
        marker in haystack
        for marker in (
            "оплат",
            "платеж",
            "предоплат",
            "реквизит",
            "записаться",
            "запишет",
            "регистрац",
            "набор",
            "зачислен",
            "заявк",
            "запись до",
            "записи до",
            "лист ожидания",
            "по предзаписи",
            "менеджер подбер",
            "менеджер свяж",
            "менеджер подскажет",
            "свяжется",
        )
    )


def _subjects_from_text(text: str) -> list[str]:
    found = [subject for subject, aliases in SUBJECT_ALIASES.items() if any(alias in text for alias in aliases)]
    return sorted(set(found))


def _grade_values(fact: Mapping[str, Any]) -> list[int]:
    structured = _mapping(fact.get("structured_value"))
    values: list[int] = []
    for key in ("grade_values", "classes", "classes_raw", "grade"):
        values.extend(_parse_grade_values(structured.get(key), allow_bare=True))
    if not values:
        values.extend(_parse_grade_values(_fact_haystack(fact), allow_bare=False))
    return sorted({value for value in values if 1 <= value <= 11})


def _parse_grade_values(value: Any, *, allow_bare: bool) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result: list[int] = []
        for item in value:
            result.extend(_parse_grade_values(item, allow_bare=allow_bare))
        return result
    raw_text = _text(value).replace("ё", "е").casefold()
    if not raw_text.strip():
        return []
    result: list[int] = []
    range_pattern = (
        r"\b([1-9]|1[01])\s*[-–—]\s*([1-9]|1[01])\b"
        if allow_bare
        else (
            r"\b([1-9]|1[01])\s*[-–—]\s*([1-9]|1[01])"
            r"(?:\s*[-–—]?\s*(?:й|го|ый|ого|ых|х|е|м))?\s*"
            r"(?:класс|классы|классов|класса|кл)\b"
        )
    )
    for start, end in re.findall(range_pattern, raw_text):
        a, b = int(start), int(end)
        if a > b:
            a, b = b, a
        result.extend(range(a, b + 1))
    text_without_ranges = _normalize_text(re.sub(range_pattern, " ", raw_text))
    if allow_bare:
        result.extend(int(item) for item in re.findall(r"\b([1-9]|1[01])\b", text_without_ranges))
    else:
        result.extend(
            int(item)
            for item in re.findall(
                r"\b([1-9]|1[01])(?:\s*[-–—]?\s*(?:й|го|ый|ого|ых|х|е|м))?\s*"
                r"(?:класс|классе|класса|классов|кл)\b",
                text_without_ranges,
            )
        )
        result.extend(
            int(item)
            for item in re.findall(
                r"(?:math|physics|informatics|russian|ai)_([1-9]|1[01])_(?:regular|olympiad|advanced|oge|ege|before)",
                text_without_ranges,
            )
        )
    return result


def _venue_from_text(text: str) -> str:
    if "долгопруд" in text:
        return "dolgoprudny"
    if "менделеево" in text or "лвш" in text:
        return "lvsh_mendeleevo"
    if "сретенк" in text or "красносельск" in text or "москва" in text:
        return "moscow_regular"
    if "онлайн" in text or "online" in text:
        return "online"
    return ""


def _fact_text(fact: Mapping[str, Any]) -> str:
    for key in ("client_safe_text", "fact_text", "manager_display_text", "text"):
        value = _text(fact.get(key))
        if value:
            return value
    structured = _mapping(fact.get("structured_value"))
    return _text(structured.get("raw_value"))


def _fact_haystack(fact: Mapping[str, Any]) -> str:
    structured = _mapping(fact.get("structured_value"))
    return _normalize_text(
        " ".join(
            _text(value)
            for value in (
                fact.get("fact_key"),
                fact.get("fact_id"),
                fact.get("fact_type"),
                fact.get("product"),
                fact.get("client_safe_text"),
                fact.get("fact_text"),
                structured.get("raw_value"),
                structured.get("classes"),
                structured.get("format"),
            )
        )
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text(value: Any) -> str:
    text = _text(value).replace("ё", "е").casefold()
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _grade_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if 1 <= value <= 11 else None
    raw = _text(value).casefold()
    if re.fullmatch(r"\s*([1-9]|1[01])\s*", raw):
        return int(raw.strip())
    match = re.search(
        r"\b([1-9]|1[01])(?:\s*[-–—]?\s*(?:й|го|ый|ого|ых|х|е|м))?\s*"
        r"(?:класс|классе|класса|классов|кл)\b",
        raw,
    )
    return int(match.group(1)) if match else None


def _stable_entry_id(*parts: str) -> str:
    value = "_".join(re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ]+", "_", part).strip("_") for part in parts if part)
    return value[:220]


def _dedupe_entries(entries: Sequence[ProductExistenceEntry]) -> list[ProductExistenceEntry]:
    seen: set[str] = set()
    result: list[ProductExistenceEntry] = []
    for entry in entries:
        if entry.entry_id in seen:
            continue
        seen.add(entry.entry_id)
        result.append(entry)
    return result
