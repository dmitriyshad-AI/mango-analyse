from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

PHONE_CANDIDATE_RE = re.compile(r"(?:\+?\d[\d\s()\-.]{8,}\d)")


def normalize_phone(value: Any) -> str | None:
    """Return a stable digit-only phone key suitable for grouping client chains.

    Russian numbers are normalized to 11 digits starting with ``7``. Other plausible
    international numbers are returned as digit-only strings when length is 10-15.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None

    digits = _digits_from_value(raw)
    if not digits:
        return None

    if len(digits) == 10:
        return "7" + digits
    if len(digits) == 11 and digits.startswith("8"):
        return "7" + digits[1:]
    if len(digits) == 11 and digits.startswith("7"):
        return digits
    if len(digits) == 12 and digits.startswith("77"):
        return digits[1:]
    if 10 <= len(digits) <= 15:
        return digits
    return None


def phones_from_text(value: Any) -> list[str]:
    """Extract normalized phones from arbitrary text while preserving order."""
    if value is None:
        return []
    text = str(value)
    found: list[str] = []
    seen: set[str] = set()
    for match in PHONE_CANDIDATE_RE.finditer(text):
        phone = normalize_phone(match.group(0))
        if phone and phone not in seen:
            seen.add(phone)
            found.append(phone)
    return found


def client_key_for_phone(phone_key: str | None) -> str | None:
    if not phone_key:
        return None
    return f"phone:{phone_key}"


def _digits_from_value(raw: str) -> str:
    lowered = raw.strip().lower().replace(",", ".")
    if "e" in lowered and re.fullmatch(r"[+\-]?\d+(?:\.\d+)?e[+\-]?\d+", lowered):
        try:
            return str(int(Decimal(lowered)))
        except (InvalidOperation, ValueError, OverflowError):
            pass
    return "".join(ch for ch in raw if ch.isdigit())
