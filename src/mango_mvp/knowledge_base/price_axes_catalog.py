from __future__ import annotations

"""Price-axis catalog for regular course price retrieval.

The source KB snapshot is still the source of facts. This module builds a
derived catalog with explicit axes that the original snapshot does not yet
store atomically: class, format, period, subject availability and tariff.
"""

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


PRICE_AXES_SELECTOR_ENV = "TELEGRAM_PRICE_AXES_SELECTOR"
PRICE_AXES_SCHEMA_VERSION = "price_axes_catalog_v1_2026_06_21"
KC_SOURCE_DOCUMENT_ID = "1bMhN0DtqNK8Z2XdwGMci2lAv0CtSYQ4QGb1Hr4dQ9Oo"
KC_SOURCE_TITLE = "База знаний КЦ"
KC_SOURCE_UPDATED_AT = "2026-06-15T11:23:26.941Z"

FOTON = "foton"
UNPK = "unpk"

REGULAR_SUBJECTS: tuple[str, ...] = ("math", "physics", "informatics", "russian", "ai")
UNPK_WEEKEND_SUBJECTS: tuple[str, ...] = ("math", "physics")
UNPK_WEEKDAY_SUBJECTS: tuple[str, ...] = ("math", "physics", "informatics")

SUBJECT_LABELS: dict[str, str] = {
    "math": "математика",
    "physics": "физика",
    "informatics": "информатика",
    "russian": "русский",
    "ai": "ИИ",
}
TARIFF_LABELS: dict[str, str] = {
    "base": "Основа",
    "standard": "Стандартный",
    "advanced": "Продвинутый",
    "full_immersion": "Полное погружение",
}
TARIFF_INCLUDES: dict[str, tuple[str, ...]] = {
    "base": ("записи вебинаров", "конспекты", "банк задач", "самостоятельная подготовка"),
    "standard": ("35 живых вебинаров", "общение с преподавателем в чате", "записи вебинаров"),
    "advanced": ("все из тарифа «Стандартный»", "35 практических вебинаров", "группы до 20 человек"),
    "full_immersion": ("все из тарифа «Продвинутый»", "индивидуальные занятия раз в 2 недели"),
}

UNPK_ONLINE_PRICE_OVERRIDES: tuple[dict[str, Any], ...] = (
    {
        "source_fact_key": "kb_v6_6_client_safe_facts_2026_06_08.annual_online_courses_math_physics_5_11_weekend_2026_27.client_safe_text",
        "schedule": "weekend",
        "classes": "5-11",
        "subjects": UNPK_WEEKEND_SUBJECTS,
        "prices": {"semester": 37000, "year": 59000},
        "description": "онлайн-курсы по математике и физике для 5-11 классов по выходным",
    },
    {
        "source_fact_key": "kb_v6_6_client_safe_facts_2026_06_08.annual_online_courses_math_physics_informatics_9_11_weekday_2026_27.client_safe_text",
        "schedule": "weekday",
        "classes": "9 и 11",
        "subjects": UNPK_WEEKDAY_SUBJECTS,
        "prices": {"semester": 41800, "year": 69900},
        "description": "онлайн-курсы для 9 и 11 классов по будням",
    },
)


@dataclass(frozen=True)
class PriceAxisEntry:
    entry_id: str
    source_fact_id: str
    source_fact_key: str
    source_kind: str
    brand: str
    product_code: str
    format: str
    period: str
    amount: int
    currency: str
    classes: str
    grade_min: int | None
    grade_max: int | None
    grade_values: tuple[int, ...]
    subjects: tuple[str, ...]
    client_safe_text: str
    tariff_id: str = ""
    tariff_title: str = ""
    tariff_includes: tuple[str, ...] = ()
    schedule: str = ""
    source_document_id: str = ""
    source_document_title: str = ""
    source_document_updated_at: str = ""
    structured_value: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("grade_values", "subjects", "tariff_includes"):
            data[key] = list(data[key])
        data["structured_value"] = dict(self.structured_value)
        return data


def price_axes_selector_enabled() -> bool:
    return _truthy(os.getenv(PRICE_AXES_SELECTOR_ENV))


def build_price_axes_catalog(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries: list[PriceAxisEntry] = []
    issues: list[dict[str, Any]] = []
    facts_by_key = {str(fact.get("fact_key") or ""): fact for fact in facts}

    for fact in facts:
        structured = _mapping(fact.get("structured_value"))
        fact_id = _text(fact.get("fact_id") or fact.get("id"))
        fact_key = _text(fact.get("fact_key") or fact_id)
        if not fact_id or not fact_key:
            continue
        if structured.get("amount_min") is not None or structured.get("amount_max") is not None:
            if _looks_like_regular_price_fact(fact):
                issues.append(
                    {
                        "issue": "range_not_final_price",
                        "fact_id": fact_id,
                        "fact_key": fact_key,
                        "amount_min": structured.get("amount_min"),
                        "amount_max": structured.get("amount_max"),
                    }
                )
            continue
        amount = _int_or_none(structured.get("amount"))
        if amount is None:
            continue
        if not _looks_like_regular_price_fact(fact):
            continue
        axes = _grade_axes_from_classes(structured.get("classes"))
        if not axes:
            issues.append({"issue": "classes_missing_or_unparsed", "fact_id": fact_id, "fact_key": fact_key})
            continue
        if not normalize_brand(_text(fact.get("brand"))) or not normalize_format(_text(structured.get("format"))) or not normalize_period(_text(structured.get("period"))):
            continue
        client_safe_text = _text(fact.get("client_safe_text") or fact.get("text"))
        if not client_safe_text:
            issues.append({"issue": "empty_client_safe_text_not_final_price", "fact_id": fact_id, "fact_key": fact_key})
            continue
        entry = _entry_from_regular_fact(fact, structured, axes, amount, client_safe_text)
        if entry is not None:
            entries.append(entry)

    for override in UNPK_ONLINE_PRICE_OVERRIDES:
        fact = facts_by_key.get(str(override["source_fact_key"]))
        if not fact:
            issues.append({"issue": "unpk_online_source_fact_missing", "fact_key": override["source_fact_key"]})
            continue
        source_text = _text(fact.get("client_safe_text") or fact.get("text"))
        if not source_text:
            issues.append({"issue": "unpk_online_source_fact_empty_client_safe_text", "fact_key": override["source_fact_key"]})
            continue
        entries.extend(_entries_from_unpk_online_override(fact, override))

    for fact in facts:
        fact_key = _text(fact.get("fact_key"))
        if fact_key.endswith(".m9_online_math_oge_tariffs") or fact_key.endswith(".m11_online_math_ege_tariffs"):
            entries.extend(_entries_from_m9_m11_tariff_fact(fact))

    return {
        "schema_version": PRICE_AXES_SCHEMA_VERSION,
        "source_snapshot": "kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json",
        "source_truth": {
            "title": KC_SOURCE_TITLE,
            "document_id": KC_SOURCE_DOCUMENT_ID,
            "updated_at": KC_SOURCE_UPDATED_AT,
            "note": "КЦ-база подтверждает УНПК-онлайн цены; исходные факты r4.1 содержат эти цены текстом.",
        },
        "rules": {
            "regular_course_price_depends_on": ["brand", "grade", "format", "period"],
            "regular_course_price_does_not_depend_on": ["subject"],
            "regular_subjects_fixed_list": list(REGULAR_SUBJECTS),
            "grade_axes_source": "structured_value.classes only",
            "range_facts_are_not_final_price": True,
        },
        "entries": [entry.to_dict() for entry in _dedupe_entries(entries)],
        "issues": issues,
    }


def select_price(
    catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    brand: str,
    grade: int | None,
    subject: str = "",
    format: str,
    period: str,
    product_code: str = "",
    tariff_id: str = "",
) -> dict[str, Any]:
    entries = _catalog_entries(catalog)
    normalized_brand = normalize_brand(brand)
    normalized_format = normalize_format(format)
    normalized_period = normalize_period(period)
    normalized_subject = normalize_subject(subject)
    normalized_product = normalize_product_code(product_code)
    normalized_tariff = normalize_tariff_id(tariff_id)

    missing = []
    if not normalized_brand:
        missing.append("brand")
    if grade is None:
        missing.append("grade")
    if not normalized_format:
        missing.append("format")
    if not normalized_period:
        missing.append("period")
    if missing:
        return {"status": "needs_slot", "missing_slots": missing, "reason": "required_axis_missing", "matches": []}

    matching = [
        entry
        for entry in entries
        if entry.get("brand") == normalized_brand
        and entry.get("format") == normalized_format
        and entry.get("period") == normalized_period
        and _entry_contains_grade(entry, grade)
        and _entry_matches_subject(entry, normalized_subject)
        and _entry_matches_product(entry, normalized_product)
        and _entry_matches_tariff(entry, normalized_tariff)
    ]

    if not normalized_product:
        matching = [entry for entry in matching if not _text(entry.get("product_code")).startswith(("m9", "m11"))]
    if not normalized_tariff:
        matching = [entry for entry in matching if not _text(entry.get("tariff_id"))]

    if not matching:
        return {"status": "not_found", "missing_slots": (), "reason": "no_exact_price_for_axes", "matches": []}

    amounts = {_int_or_none(entry.get("amount")) for entry in matching}
    amounts.discard(None)
    schedules = {_text(entry.get("schedule")) for entry in matching if _text(entry.get("schedule"))}
    tariffs = {_text(entry.get("tariff_id")) for entry in matching if _text(entry.get("tariff_id"))}
    if len(matching) > 1 and len(amounts) > 1:
        missing_slots: list[str] = []
        if schedules:
            missing_slots.append("schedule")
        if tariffs:
            missing_slots.append("tariff_id")
        return {
            "status": "needs_slot",
            "missing_slots": missing_slots or ["price_variant"],
            "reason": "multiple_prices_for_axes",
            "matches": matching,
        }

    exact = matching[0]
    return {"status": "exact", "entry": exact, "missing_slots": (), "reason": "exact_price_found", "matches": matching}


def select_price_fact_for_query(
    facts: Sequence[Mapping[str, Any]],
    *,
    active_brand: str,
    query: str,
) -> Mapping[str, Any] | None:
    if not _looks_like_price_query(query):
        return None
    axes = extract_price_query_axes(query, active_brand=active_brand)
    result = select_price(build_price_axes_catalog(facts), **axes)
    if result.get("status") != "exact":
        return None
    entry = result.get("entry")
    if not isinstance(entry, Mapping):
        return None
    return virtual_fact_from_price_entry(entry)


def extract_price_query_axes(query: str, *, active_brand: str = "") -> dict[str, Any]:
    text = _normalize_text(query)
    product_code = normalize_product_code(query)
    return {
        "brand": normalize_brand(active_brand or query),
        "grade": _extract_grade(text),
        "subject": normalize_subject(query),
        "format": normalize_format(query),
        "period": normalize_period(query),
        "product_code": product_code,
        "tariff_id": normalize_tariff_id(query),
    }


def virtual_fact_from_price_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    fact_id = f"fact:v3:price_axes_selector:{_text(entry.get('entry_id'))}"
    fact_key = f"price_axes_selector.{_text(entry.get('brand'))}.{_text(entry.get('product_code') or 'regular')}.{_text(entry.get('format'))}.{_text(entry.get('period'))}"
    if _text(entry.get("tariff_id")):
        fact_key += f".{_text(entry.get('tariff_id'))}"
    return {
        "fact_id": fact_id,
        "id": fact_id,
        "fact_key": fact_key,
        "brand": _text(entry.get("brand")),
        "fact_type": "price",
        "fact_types": ["price"],
        "title": "Точная цена из каталога осей",
        "client_safe_text": _text(entry.get("client_safe_text")),
        "manager_check_text": _text(entry.get("client_safe_text")),
        "freshness_status": "document_verified",
        "allowed_for_client_answer": True,
        "usable_for_precise_answer": True,
        "source_id": "f8_price_axes_catalog",
        "source_title": KC_SOURCE_TITLE,
        "structured_value": dict(entry.get("structured_value") or {}),
        "price_axes_entry": dict(entry),
    }


def normalize_brand(value: str) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("unpk", "унпк", "мфти")):
        return UNPK
    if any(marker in text for marker in ("foton", "фотон", "цдпо")):
        return FOTON
    return ""


def normalize_format(value: str) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("online", "онлайн", "дистанц")):
        return "online"
    if any(marker in text for marker in ("offline", "очно", "очная", "очный", "сретенка", "москва")):
        return "offline"
    return ""


def normalize_period(value: str) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("semester", "sem", "семестр", "полугод")):
        return "semester"
    if any(marker in text for marker in ("year", "год", "годовой", "годовая", "учебный год")):
        return "year"
    return ""


def normalize_subject(value: str) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("математ", "math", "егэ по мат", "огэ по мат")):
        return "math"
    if "физик" in text or "physics" in text:
        return "physics"
    if any(marker in text for marker in ("информат", "программ", "it", "айти")):
        return "informatics"
    if any(marker in text for marker in ("русск", "русский")):
        return "russian"
    if any(marker in text for marker in (" ии", "искусствен", "ai ", "ai-lab", "ai lab")):
        return "ai"
    return ""


def normalize_product_code(value: str) -> str:
    text = _normalize_text(value)
    if re.search(r"\bм\s*9\b|\bm\s*9\b|\bm9\b", text):
        return "m9"
    if re.search(r"\bм\s*11\b|\bm\s*11\b|\bm11\b", text):
        return "m11"
    return ""


def normalize_tariff_id(value: str) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("основа", "базов", "base")):
        return "base"
    if any(marker in text for marker in ("стандарт", "standard")):
        return "standard"
    if any(marker in text for marker in ("продвинут", "advanced")):
        return "advanced"
    if any(marker in text for marker in ("полн", "погруж", "premium", "full")):
        return "full_immersion"
    return ""


def _entry_from_regular_fact(
    fact: Mapping[str, Any],
    structured: Mapping[str, Any],
    axes: Mapping[str, Any],
    amount: int,
    client_safe_text: str,
) -> PriceAxisEntry | None:
    brand = normalize_brand(_text(fact.get("brand")))
    fmt = normalize_format(_text(structured.get("format")))
    period = normalize_period(_text(structured.get("period")))
    if not brand or not fmt or not period:
        return None
    fact_id = _text(fact.get("fact_id") or fact.get("id"))
    fact_key = _text(fact.get("fact_key") or fact_id)
    classes = _text(structured.get("classes"))
    structured_value = _entry_structured_value(
        amount=amount,
        currency=_text(structured.get("currency") or "RUB"),
        fmt=fmt,
        period=period,
        classes=classes,
        axes=axes,
        subjects=REGULAR_SUBJECTS,
        source_fact_id=fact_id,
        source_fact_key=fact_key,
        source_kind="regular_structured_price",
    )
    return PriceAxisEntry(
        entry_id=_stable_entry_id("regular", fact_id, fmt, period, classes),
        source_fact_id=fact_id,
        source_fact_key=fact_key,
        source_kind="regular_structured_price",
        brand=brand,
        product_code="regular_course",
        format=fmt,
        period=period,
        amount=amount,
        currency=_text(structured.get("currency") or "RUB"),
        classes=classes,
        grade_min=axes["grade_min"],
        grade_max=axes["grade_max"],
        grade_values=tuple(axes["grade_values"]),
        subjects=REGULAR_SUBJECTS,
        client_safe_text=client_safe_text,
        source_document_id=KC_SOURCE_DOCUMENT_ID,
        source_document_title=KC_SOURCE_TITLE,
        source_document_updated_at=KC_SOURCE_UPDATED_AT,
        structured_value=structured_value,
    )


def _entries_from_unpk_online_override(fact: Mapping[str, Any], override: Mapping[str, Any]) -> list[PriceAxisEntry]:
    axes = _grade_axes_from_classes(override.get("classes"))
    if not axes:
        return []
    fact_id = _text(fact.get("fact_id") or fact.get("id"))
    fact_key = _text(fact.get("fact_key") or override.get("source_fact_key"))
    classes = _text(override.get("classes"))
    subjects = tuple(str(item) for item in override.get("subjects") or ())
    schedule = _text(override.get("schedule"))
    description = _text(override.get("description"))
    entries: list[PriceAxisEntry] = []
    for period, amount in sorted((override.get("prices") or {}).items()):
        normalized_period = normalize_period(str(period))
        amount_int = _int_or_none(amount)
        if amount_int is None or not normalized_period:
            continue
        client_safe = (
            f"УНПК МФТИ: {description}, "
            f"{_period_label(normalized_period)} — {_money(amount_int)}."
        )
        structured_value = _entry_structured_value(
            amount=amount_int,
            currency="RUB",
            fmt="online",
            period=normalized_period,
            classes=classes,
            axes=axes,
            subjects=subjects,
            source_fact_id=fact_id,
            source_fact_key=fact_key,
            source_kind="unpk_online_kc_source_price",
            schedule=schedule,
        )
        entries.append(
            PriceAxisEntry(
                entry_id=_stable_entry_id("unpk_online", fact_id, schedule, normalized_period),
                source_fact_id=fact_id,
                source_fact_key=fact_key,
                source_kind="unpk_online_kc_source_price",
                brand=UNPK,
                product_code="regular_course",
                format="online",
                period=normalized_period,
                amount=amount_int,
                currency="RUB",
                classes=classes,
                grade_min=axes["grade_min"],
                grade_max=axes["grade_max"],
                grade_values=tuple(axes["grade_values"]),
                subjects=subjects,
                client_safe_text=client_safe,
                schedule=schedule,
                source_document_id=KC_SOURCE_DOCUMENT_ID,
                source_document_title=KC_SOURCE_TITLE,
                source_document_updated_at=KC_SOURCE_UPDATED_AT,
                structured_value=structured_value,
            )
        )
    return entries


def _entries_from_m9_m11_tariff_fact(fact: Mapping[str, Any]) -> list[PriceAxisEntry]:
    structured = _mapping(fact.get("structured_value"))
    prices = _mapping(structured.get("prices"))
    fact_key = _text(fact.get("fact_key"))
    fact_id = _text(fact.get("fact_id") or fact.get("id"))
    product_code = "m11" if ".m11_" in fact_key else "m9"
    grade = 11 if product_code == "m11" else 9
    exam = "ЕГЭ" if product_code == "m11" else "ОГЭ"
    entries: list[PriceAxisEntry] = []
    for tariff_id in ("base", "standard", "advanced", "full_immersion"):
        amount = _int_or_none(prices.get(tariff_id))
        if amount is None:
            continue
        title = TARIFF_LABELS[tariff_id]
        includes = TARIFF_INCLUDES[tariff_id]
        includes_text = "; ".join(includes)
        client_safe = f"Фотон: {product_code.upper()} по математике ({exam}), тариф «{title}» — {_money(amount)}. Входит: {includes_text}."
        axes = {"grade_min": grade, "grade_max": grade, "grade_values": (grade,)}
        structured_value = _entry_structured_value(
            amount=amount,
            currency="RUB",
            fmt="online",
            period="year",
            classes=str(grade),
            axes=axes,
            subjects=("math",),
            source_fact_id=fact_id,
            source_fact_key=fact_key,
            source_kind="foton_m9_m11_tariff_price",
            tariff_id=tariff_id,
            tariff_title=title,
            tariff_includes=includes,
            product_code=product_code,
        )
        entries.append(
            PriceAxisEntry(
                entry_id=_stable_entry_id("tariff", fact_id, tariff_id),
                source_fact_id=fact_id,
                source_fact_key=fact_key,
                source_kind="foton_m9_m11_tariff_price",
                brand=FOTON,
                product_code=product_code,
                format="online",
                period="year",
                amount=amount,
                currency="RUB",
                classes=str(grade),
                grade_min=grade,
                grade_max=grade,
                grade_values=(grade,),
                subjects=("math",),
                client_safe_text=client_safe,
                tariff_id=tariff_id,
                tariff_title=title,
                tariff_includes=includes,
                source_document_id=KC_SOURCE_DOCUMENT_ID,
                source_document_title=KC_SOURCE_TITLE,
                source_document_updated_at=KC_SOURCE_UPDATED_AT,
                structured_value=structured_value,
            )
        )
    return entries


def _entry_structured_value(
    *,
    amount: int,
    currency: str,
    fmt: str,
    period: str,
    classes: str,
    axes: Mapping[str, Any],
    subjects: Sequence[str],
    source_fact_id: str,
    source_fact_key: str,
    source_kind: str,
    **extra: Any,
) -> dict[str, Any]:
    value = {
        "amount": amount,
        "currency": currency or "RUB",
        "format": fmt,
        "period": period,
        "classes": classes,
        "grade_min": axes.get("grade_min"),
        "grade_max": axes.get("grade_max"),
        "grade_values": list(axes.get("grade_values") or ()),
        "subjects": list(subjects),
        "source_fact_id": source_fact_id,
        "source_fact_key": source_fact_key,
        "source_kind": source_kind,
        "source_truth": KC_SOURCE_TITLE,
    }
    value.update({key: value for key, value in extra.items() if value not in (None, "", (), [])})
    return value


def _grade_axes_from_classes(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        numbers: list[int] = []
        for item in value:
            parsed = _grade_axes_from_classes(item)
            if parsed:
                numbers.extend(parsed["grade_values"])
        unique = tuple(sorted(set(numbers)))
        if not unique:
            return None
        return {"grade_min": min(unique), "grade_max": max(unique), "grade_values": unique}
    text = _normalize_text(value)
    if not text:
        return None
    range_match = re.search(r"\b([1-9]|1[01])\s*[-–—]\s*([1-9]|1[01])\b", text)
    if range_match:
        start = int(range_match.group(1))
        end = int(range_match.group(2))
        if start > end:
            start, end = end, start
        values = tuple(range(start, end + 1))
        return {"grade_min": start, "grade_max": end, "grade_values": values}
    numbers = [int(item) for item in re.findall(r"\b(?:[1-9]|1[01])\b", text)]
    unique = tuple(sorted(set(numbers)))
    if not unique:
        return None
    return {"grade_min": min(unique), "grade_max": max(unique), "grade_values": unique}


def _catalog_entries(catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(catalog, Mapping):
        entries = catalog.get("entries") or ()
        return [entry for entry in entries if isinstance(entry, Mapping)]
    built = build_price_axes_catalog(catalog)
    return [entry for entry in built.get("entries", ()) if isinstance(entry, Mapping)]


def _entry_contains_grade(entry: Mapping[str, Any], grade: int | None) -> bool:
    if grade is None:
        return False
    values = entry.get("grade_values") or ()
    return int(grade) in {int(item) for item in values if _int_or_none(item) is not None}


def _entry_matches_subject(entry: Mapping[str, Any], subject: str) -> bool:
    if not subject:
        return True
    subjects = {str(item) for item in (entry.get("subjects") or ())}
    return subject in subjects


def _entry_matches_product(entry: Mapping[str, Any], product_code: str) -> bool:
    if not product_code:
        return True
    return _text(entry.get("product_code")) == product_code


def _entry_matches_tariff(entry: Mapping[str, Any], tariff_id: str) -> bool:
    if not tariff_id:
        return True
    return _text(entry.get("tariff_id")) == tariff_id


def _looks_like_regular_price_fact(fact: Mapping[str, Any]) -> bool:
    structured = _mapping(fact.get("structured_value"))
    fact_key = _text(fact.get("fact_key"))
    if structured.get("do_not_use_as_current_price"):
        return False
    if "early_booking" in fact_key:
        return True
    return "prices_regular_2026_27." in fact_key and not fact_key.endswith(".note_internal")


def _looks_like_price_query(value: str) -> bool:
    text = _normalize_text(value)
    return any(marker in text for marker in ("стоим", "цена", "сколько", "оплат", "прайс", "тариф"))


def _extract_grade(text: str) -> int | None:
    patterns = (
        r"\b([1-9]|1[01])\s*(?:класс|кл\.?|класса|классе)\b",
        r"\b(?:[mм]9|[mм]\s*9)\b",
        r"\b(?:[mм]11|[mм]\s*11)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        if "м" in match.group(0):
            return 11 if "11" in match.group(0) else 9
        return int(match.group(1))
    return None


def _stable_entry_id(*parts: str) -> str:
    safe = "_".join(re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ]+", "_", part).strip("_") for part in parts if part)
    return safe[:220]


def _dedupe_entries(entries: Sequence[PriceAxisEntry]) -> list[PriceAxisEntry]:
    seen: set[str] = set()
    result: list[PriceAxisEntry] = []
    for entry in entries:
        if entry.entry_id in seen:
            continue
        seen.add(entry.entry_id)
        result.append(entry)
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value).replace("ё", "е")).casefold()


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d]", "", value)
        if cleaned:
            return int(cleaned)
    return None


def _period_label(period: str) -> str:
    return "семестр" if period == "semester" else "год"


def _money(amount: int) -> str:
    return f"{amount:,}".replace(",", " ") + " ₽"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on", "да"}
