from __future__ import annotations

"""Price-axis catalog for regular course price retrieval.

The source KB snapshot is still the source of facts. This module builds a
derived catalog with explicit axes that the original snapshot does not yet
store atomically: class, format, period, subject availability and tariff.
"""

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from mango_mvp.knowledge_base.fact_registry import (
    evaluate_fact_freshness_sla,
    fact_runtime_time_ok,
    fact_valid_until_ok,
)


PRICE_AXES_SELECTOR_ENV = "TELEGRAM_PRICE_AXES_SELECTOR"
PRICE_AXES_CLEAN_DEFER_ENV = "TELEGRAM_PRICE_AXES_CLEAN_DEFER"
DIRECT_PATH_PILOT_CONFIG_ENV = "TELEGRAM_DIRECT_PATH_PILOT_CONFIG"
DIRECT_PATH_PILOT_CONFIG_VERSION = "pilot_gold_v1"
PRICE_AXES_SCHEMA_VERSION = "price_axes_catalog_v1_2026_06_21"
FOTON = "foton"
UNPK = "unpk"

REGULAR_SUBJECTS: tuple[str, ...] = ("math", "physics", "informatics", "russian", "ai")
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
    valid_from: str = ""
    valid_until: str = ""
    freshness_check_date: str = ""
    tariff_id: str = ""
    tariff_title: str = ""
    tariff_includes: tuple[str, ...] = ()
    schedule: str = ""
    structured_value: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("grade_values", "subjects", "tariff_includes"):
            data[key] = list(data[key])
        data["structured_value"] = dict(self.structured_value)
        return data


def price_axes_selector_enabled() -> bool:
    return _flag_enabled_with_pilot_profile(PRICE_AXES_SELECTOR_ENV)


def price_axes_clean_defer_enabled() -> bool:
    return _flag_enabled_with_pilot_profile(PRICE_AXES_CLEAN_DEFER_ENV)


def _flag_enabled_with_pilot_profile(env_name: str) -> bool:
    if env_name in os.environ:
        return _truthy(os.getenv(env_name))
    return str(os.getenv(DIRECT_PATH_PILOT_CONFIG_ENV) or "").strip() == DIRECT_PATH_PILOT_CONFIG_VERSION


def build_price_axes_catalog(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries: list[PriceAxisEntry] = []
    issues: list[dict[str, Any]] = []
    for fact in facts:
        if not _fact_is_client_usable(fact):
            continue
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

    for fact in facts:
        if not _fact_is_client_usable(fact):
            continue
        fact_key = _text(fact.get("fact_key"))
        if fact_key.endswith(".m9_online_math_oge_tariffs") or fact_key.endswith(".m11_online_math_ege_tariffs"):
            entries.extend(_entries_from_m9_m11_tariff_fact(fact))

    return {
        "schema_version": PRICE_AXES_SCHEMA_VERSION,
        "source_snapshot": "current_runtime_snapshot",
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


def _fact_is_client_usable(fact: Mapping[str, Any]) -> bool:
    if fact.get("allowed_for_client_answer") is not True or fact.get("usable_for_precise_answer") is False:
        return False
    if _text(fact.get("freshness_status")) in {"do_not_use", "expired", "superseded"}:
        return False
    evaluation_day = _evaluation_day()
    if evaluation_day is False:
        return False
    # Keep a confirmed future tariff in the derived catalog. select_price()
    # applies valid_from at read time, so no catalog rebuild is needed on day 1.
    return fact_valid_until_ok(fact.get("valid_until"), today=evaluation_day) and (
        not _has_freshness_check_date(fact)
        or evaluate_fact_freshness_sla(fact, today=evaluation_day).within_sla
    )


def select_price(
    catalog: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    brand: str,
    grade: int | None,
    subject: str = "",
    format: str,
    period: str,
    schedule: str = "",
    product_code: str = "",
    tariff_id: str = "",
) -> dict[str, Any]:
    entries = [entry for entry in _catalog_entries(catalog) if _entry_runtime_usable(entry)]
    normalized_brand = normalize_brand(brand)
    normalized_format = normalize_format(format)
    normalized_period = normalize_period(period)
    normalized_schedule = normalize_schedule(schedule)
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
    if normalized_subject and not _regular_subject_is_supported(
        brand=normalized_brand,
        grade=grade,
        subject=normalized_subject,
    ):
        return {"status": "not_found", "missing_slots": (), "reason": "subject_not_offered", "matches": []}

    matching = [
        entry
        for entry in entries
        if entry.get("brand") == normalized_brand
        and entry.get("format") == normalized_format
        and entry.get("period") == normalized_period
        and _entry_contains_grade(entry, grade)
        and (not normalized_schedule or _text(entry.get("schedule")) == normalized_schedule)
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


def _evaluation_day() -> date | None | bool:
    raw = _text(os.getenv("MANGO_EVALUATION_DATE"))
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return False


def _has_freshness_check_date(fact: Mapping[str, Any]) -> bool:
    structured = _mapping(fact.get("structured_value"))
    return bool(_text(fact.get("freshness_check_date") or fact.get("verified_at") or structured.get("freshness_check_date")))


def _entry_runtime_usable(entry: Mapping[str, Any]) -> bool:
    evaluation_day = _evaluation_day()
    if evaluation_day is False:
        return False
    return fact_runtime_time_ok({**entry, "fact_type": "price"}, today=evaluation_day)


def select_price_fact_for_query(
    facts: Sequence[Mapping[str, Any]],
    *,
    active_brand: str,
    query: str,
) -> Mapping[str, Any] | None:
    result = select_price_result_for_query(facts, active_brand=active_brand, query=query)
    if result.get("status") != "exact":
        return None
    entry = result.get("entry")
    if not isinstance(entry, Mapping):
        return None
    return virtual_fact_from_price_entry(entry)


def select_price_result_for_query(
    facts: Sequence[Mapping[str, Any]],
    *,
    active_brand: str,
    query: str,
) -> dict[str, Any]:
    if not _looks_like_price_query(query):
        return {"status": "not_price_query", "missing_slots": (), "reason": "query_not_about_price", "matches": []}
    axes = extract_price_query_axes(query, active_brand=active_brand)
    result = dict(select_price(build_price_axes_catalog(facts), **axes))
    result["query_axes"] = axes
    return result


def extract_price_query_axes(query: str, *, active_brand: str = "") -> dict[str, Any]:
    text = _normalize_text(query)
    product_code = normalize_product_code(query)
    return {
        "brand": normalize_brand(active_brand or query),
        "grade": _extract_grade(text),
        "subject": normalize_subject(query),
        "format": normalize_format(query),
        "period": normalize_period(query),
        "schedule": normalize_schedule(query),
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
        "source_title": "Текущий проверенный снимок базы знаний",
        "valid_from": _text(entry.get("valid_from")),
        "valid_until": _text(entry.get("valid_until")),
        "freshness_check_date": _text(entry.get("freshness_check_date")),
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


def normalize_schedule(value: str) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("будн", "weekday", "по будням")):
        return "weekday"
    if any(marker in text for marker in ("выходн", "weekend", "суббот", "воскрес")):
        return "weekend"
    return ""


def normalize_subject(value: str) -> str:
    text = _normalize_text(value)
    if any(marker in text for marker in ("математ", "math", "егэ по мат", "огэ по мат")):
        return "math"
    if "физик" in text or "physics" in text:
        return "physics"
    if any(marker in text for marker in ("информат", "программ", "it", "айти")):
        return "informatics"
    if any(marker in text for marker in ("русск", "русский", "russian")):
        return "russian"
    if any(marker in text for marker in (" ии", "искусствен", "ai ", "ai-lab", "ai lab")):
        return "ai"
    return ""


def _regular_subject_is_supported(*, brand: str, grade: int, subject: str) -> bool:
    if subject not in REGULAR_SUBJECTS:
        return False
    if subject != "russian":
        return brand in {FOTON, UNPK}
    return brand == FOTON and grade >= 9


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
        subjects=tuple(str(item) for item in structured.get("subjects") or REGULAR_SUBJECTS),
        source_fact_id=fact_id,
        source_fact_key=fact_key,
        source_kind="regular_structured_price",
        schedule=_text(structured.get("schedule")),
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
        subjects=tuple(str(item) for item in structured.get("subjects") or REGULAR_SUBJECTS),
        client_safe_text=client_safe_text,
        valid_from=_text(fact.get("valid_from")),
        valid_until=_text(fact.get("valid_until")),
        freshness_check_date=_text(
            fact.get("freshness_check_date") or structured.get("freshness_check_date")
        ),
        schedule=_text(structured.get("schedule")),
        structured_value=structured_value,
    )


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
                valid_from=_text(fact.get("valid_from")),
                valid_until=_text(fact.get("valid_until")),
                freshness_check_date=_text(
                    fact.get("freshness_check_date") or structured.get("freshness_check_date")
                ),
                tariff_id=tariff_id,
                tariff_title=title,
                tariff_includes=includes,
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
        "source_truth": "current_runtime_snapshot",
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
    if structured.get("selector_excluded"):
        return False
    if _text(structured.get("product_code")) == "regular_course":
        return True
    if "early_booking" in fact_key:
        return True
    return "prices_regular_2026_27." in fact_key and not fact_key.endswith(".note_internal")


def _looks_like_price_query(value: str) -> bool:
    text = _normalize_text(value)
    return any(marker in text for marker in ("стоим", "цена", "сколько", "оплат", "прайс", "тариф"))


def _extract_grade(text: str) -> int | None:
    # БЛОК 7 (2026-07-25) bugfix: patterns[1]/[2] match a Cyrillic OR Latin
    # "m" (product codes M9/M11 -- the bot itself renders them in Latin, see
    # `_entries_from_m9_m11_tariff_fact`'s `product_code.upper()`), but had no
    # capturing group. The old branch below tested `"м" in match.group(0)`
    # (Cyrillic-only) to pick the return path; for a Latin "m9"/"m11" match
    # that test was False, so it fell through to `match.group(1)`, which does
    # not exist on these patterns -> IndexError crashing every price lookup
    # for a client typing "M9"/"M11" in Latin. Branching on pattern index
    # instead of sniffing match text fixes this without touching the regex
    # patterns themselves (no understanding/regex migration, see
    # docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md -- same mechanism, just
    # correct for both alphabets).
    patterns = (
        r"\b([1-9]|1[01])\s*(?:класс|кл\.?|класса|классе)\b",
        r"\b(?:[mм]9|[mм]\s*9)\b",
        r"\b(?:[mм]11|[mм]\s*11)\b",
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if not match:
            continue
        if index == 0:
            return int(match.group(1))
        return 11 if index == 2 else 9
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
