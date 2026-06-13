from __future__ import annotations

import re
from typing import Any

from mango_mvp.utils.phone import normalize_phone as canonical_normalize_phone

PHONE_CANDIDATE_RE = re.compile(r"(?:\+?\d[\d\s()\-.]{8,}\d)")


def normalize_phone(value: Any) -> str | None:
    """Return a stable digit-only phone key suitable for grouping client chains.

    Russian numbers are normalized to 11 digits starting with ``7``. Other plausible
    international numbers are returned as digit-only strings when length is 10-15.
    """
    normalized = canonical_normalize_phone(value)
    return normalized.lstrip("+") if normalized else None


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
