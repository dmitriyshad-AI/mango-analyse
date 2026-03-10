from __future__ import annotations

import re
from typing import Optional


def normalize_phone(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    digits = re.sub(r"\D+", "", value)
    if not digits:
        return None
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if len(digits) == 10:
        digits = "7" + digits
    return f"+{digits}"


def last10(value: Optional[str]) -> Optional[str]:
    normalized = normalize_phone(value)
    if not normalized:
        return None
    digits = normalized.lstrip("+")
    if len(digits) < 10:
        return digits
    return digits[-10:]
