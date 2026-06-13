from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional


def normalize_phone(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None
    explicit_plus = raw.startswith("+")
    digits = _digits_from_value(raw)
    if not digits:
        return None
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if len(digits) == 12 and digits.startswith("77"):
        digits = digits[1:]
    if len(digits) == 10 and not explicit_plus:
        digits = "7" + digits
    if not 10 <= len(digits) <= 15:
        return None
    return f"+{digits}"


def last10(value: Any) -> Optional[str]:
    normalized = normalize_phone(value)
    if not normalized:
        return None
    digits = normalized.lstrip("+")
    if len(digits) < 10:
        return digits
    return digits[-10:]


def _digits_from_value(raw: str) -> str:
    lowered = raw.strip().lower().replace(",", ".")
    if "e" in lowered and re.fullmatch(r"[+\-]?\d+(?:\.\d+)?e[+\-]?\d+", lowered):
        try:
            return str(int(Decimal(lowered)))
        except (InvalidOperation, ValueError, OverflowError):
            pass
    return re.sub(r"\D+", "", raw)
