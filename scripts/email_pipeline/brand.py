from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

try:
    from mango_mvp.customer_timeline.canonical_readonly_import import infer_offline_brand
except Exception:  # pragma: no cover - import is best-effort for audit metadata only.
    infer_offline_brand = None  # type: ignore[assignment]


Brand = str
BrandSource = str


@dataclass(frozen=True)
class BrandDecision:
    brand: Brand
    brand_source: BrandSource
    raw_infer_offline_brand: str
    signals: Mapping[str, tuple[str, ...]]


RE_FOTON_EXPLICIT = re.compile(r"\b(?:фотон|photon)\b|цдпо\s*фотон|црдо\s*фотон", re.I)
RE_UNPK_EXPLICIT = re.compile(r"\bунпк\b|\bмфти\b|формула\s+физтеха|физтех", re.I)
RE_URL = re.compile(r"(?:(?:https?://)?(?:www\.)?[a-z0-9.-]+\.[a-z]{2,})(?:/[^\s<>\"]*)?", re.I)
RE_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)


def infer_email_brand(subject: str | None, body: str | None) -> BrandDecision:
    """Strict content-only brand decision for e-mail pipeline.

    The old pipeline guessed brand from mailbox/folder/from/domain. That is
    deliberately absent here. Silence or conflicts return brand='none'.
    """
    text = _normalize_text(f"{subject or ''}\n{body or ''}")
    raw = _raw_infer_offline_brand(text)

    explicit_hits = _explicit_hits(text)
    explicit_decision = _single_brand(explicit_hits)
    if explicit_decision:
        return BrandDecision(explicit_decision, "explicit_word", raw, {"explicit_word": tuple(sorted(explicit_hits))})
    if len(explicit_hits) > 1:
        return BrandDecision("none", "none", raw, {"explicit_word": tuple(sorted(explicit_hits))})

    link_hits = _course_link_hits(text)
    link_decision = _single_brand(link_hits)
    if link_decision:
        return BrandDecision(link_decision, "course_link", raw, {"course_link": tuple(sorted(link_hits))})
    if len(link_hits) > 1:
        return BrandDecision("none", "none", raw, {"course_link": tuple(sorted(link_hits))})

    date_hits = _date_window_hits(text)
    date_decision = _single_brand(date_hits)
    if date_decision:
        return BrandDecision(date_decision, "dates", raw, {"dates": tuple(sorted(date_hits))})
    return BrandDecision("none", "none", raw, {"dates": tuple(sorted(date_hits))})


def _normalize_text(text: str) -> str:
    return (text or "").casefold().replace("ё", "е")


def _raw_infer_offline_brand(text: str) -> str:
    if infer_offline_brand is None:
        return "unavailable"
    try:
        value = str(infer_offline_brand({"text": text}) or "").strip().lower()
    except Exception:
        return "error"
    return value if value in {"foton", "unpk", "unknown"} else "unknown"


def _explicit_hits(text: str) -> set[str]:
    hits: set[str] = set()
    if RE_FOTON_EXPLICIT.search(text):
        hits.add("foton")
    if RE_UNPK_EXPLICIT.search(text):
        hits.add("unpk")
    return hits


def _course_link_hits(text: str) -> set[str]:
    hits: set[str] = set()
    email_spans = [match.span() for match in RE_EMAIL.finditer(text)]
    for match in RE_URL.finditer(text):
        if _inside_any(match.span(), email_spans):
            continue
        url = match.group(0).casefold()
        if "cdpofoton.ru" in url:
            hits.add("foton")
        if _is_kmipt_course_link(url):
            hits.add("unpk")
    return hits


def _inside_any(span: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(other_start <= start and end <= other_end for other_start, other_end in spans)


def _is_kmipt_course_link(url: str) -> bool:
    if "kmipt.ru" not in url:
        return False
    if "@" in url:
        return False
    after_domain = url.split("kmipt.ru", 1)[1]
    return after_domain.startswith("/") and len(after_domain.strip("/")) > 0


def _date_window_hits(text: str) -> set[str]:
    hits: set[str] = set()
    for day, month in _iter_dates(text):
        if month == 6 and 20 <= day <= 28:
            hits.add("foton")
        if month == 7 and 18 <= day <= 26:
            hits.add("foton")
        if month == 8 and 15 <= day <= 25:
            hits.add("unpk")
    for start, end, month in _iter_date_ranges(text):
        days = range(min(start, end), max(start, end) + 1)
        if month == 6 and any(20 <= day <= 28 for day in days):
            hits.add("foton")
        if month == 7 and any(18 <= day <= 26 for day in days):
            hits.add("foton")
        if month == 8 and any(15 <= day <= 25 for day in days):
            hits.add("unpk")
    return hits


MONTHS = {
    "июн": 6,
    "июня": 6,
    "июнь": 6,
    "июл": 7,
    "июля": 7,
    "июль": 7,
    "авг": 8,
    "августа": 8,
    "август": 8,
}


def _iter_dates(text: str):
    for match in re.finditer(r"\b([0-3]?\d)[./-](0?[6-8])(?:[./-]\d{2,4})?\b", text):
        day = int(match.group(1))
        month = int(match.group(2))
        if 1 <= day <= 31:
            yield day, month
    month_pattern = "|".join(re.escape(month) for month in MONTHS)
    for match in re.finditer(rf"\b([0-3]?\d)\s*(?:{month_pattern})\b", text):
        day = int(match.group(1))
        month_token = re.search(month_pattern, match.group(0))
        if month_token and 1 <= day <= 31:
            yield day, MONTHS[month_token.group(0)]


def _iter_date_ranges(text: str):
    month_pattern = "|".join(re.escape(month) for month in MONTHS)
    for match in re.finditer(rf"\b([0-3]?\d)\s*(?:-|–|—|по)\s*([0-3]?\d)\s*({month_pattern})\b", text):
        start = int(match.group(1))
        end = int(match.group(2))
        month = MONTHS[match.group(3)]
        if 1 <= start <= 31 and 1 <= end <= 31:
            yield start, end, month


def _single_brand(hits: set[str]) -> str | None:
    if len(hits) == 1:
        return next(iter(hits))
    return None

