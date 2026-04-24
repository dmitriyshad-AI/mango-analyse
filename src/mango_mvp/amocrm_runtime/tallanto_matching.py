from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from mango_mvp.utils.phone import normalize_phone


def build_phone_candidates(value: str) -> list[str]:
    normalized_value = str(value or "").strip()
    digits = "".join(char for char in normalized_value if char.isdigit())
    variants = [
        normalized_value,
        normalized_value.replace(" ", ""),
        digits,
        normalize_phone(normalized_value) or "",
        normalize_phone(digits) or "",
        f"8{digits[1:]}" if len(digits) == 11 and digits.startswith("7") else "",
        f"7{digits}" if len(digits) == 10 else "",
        f"+7{digits}" if len(digits) == 10 else "",
    ]
    result: list[str] = []
    seen: set[str] = set()
    for item in variants:
        candidate = str(item or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def extract_contact_phone_values(contact: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field_name in ("phone_mobile", "phone_work", "phone_home", "phone_other", "phone"):
        raw_value = contact.get(field_name)
        if raw_value is None:
            continue
        if isinstance(raw_value, list):
            for item in raw_value:
                values.extend(build_phone_candidates(str(item)))
        else:
            values.extend(build_phone_candidates(str(raw_value)))
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


@dataclass(frozen=True)
class IdentityMatchResult:
    matched_contact: Optional[dict[str, Any]]
    confidence: float
    reason: str
    matched_phone: Optional[str]
    ambiguous: bool
    candidates_considered: int


def match_contact_by_phone(
    call_phone: str,
    contacts: Iterable[dict[str, Any]],
    *,
    expected_branch: Optional[str] = None,
    expected_manager_name: Optional[str] = None,
) -> IdentityMatchResult:
    target_candidates = set(build_phone_candidates(call_phone))
    if not target_candidates:
        return IdentityMatchResult(
            matched_contact=None,
            confidence=0.0,
            reason="call_phone_invalid",
            matched_phone=None,
            ambiguous=False,
            candidates_considered=0,
        )

    scored_candidates: list[tuple[float, dict[str, Any], str]] = []
    for contact in contacts:
        contact_candidates = extract_contact_phone_values(contact)
        overlap = next((candidate for candidate in contact_candidates if candidate in target_candidates), None)
        if overlap is None:
            continue
        score = 0.8
        reasons = ["phone_match"]
        if expected_branch and str(contact.get("filial") or "").strip() == expected_branch.strip():
            score += 0.1
            reasons.append("branch_match")
        if expected_manager_name and str(contact.get("assigned_user_name") or "").strip() == expected_manager_name.strip():
            score += 0.1
            reasons.append("manager_match")
        scored_candidates.append((min(score, 0.99), contact, ",".join(reasons) + f":{overlap}"))

    if not scored_candidates:
        return IdentityMatchResult(
            matched_contact=None,
            confidence=0.0,
            reason="phone_not_found",
            matched_phone=None,
            ambiguous=False,
            candidates_considered=0,
        )

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    top_score, top_contact, top_reason = scored_candidates[0]
    ambiguous = len(scored_candidates) > 1 and abs(scored_candidates[0][0] - scored_candidates[1][0]) < 0.05
    if ambiguous:
        return IdentityMatchResult(
            matched_contact=None,
            confidence=top_score,
            reason="ambiguous_phone_match",
            matched_phone=top_reason.split(":")[-1],
            ambiguous=True,
            candidates_considered=len(scored_candidates),
        )

    return IdentityMatchResult(
        matched_contact=top_contact,
        confidence=top_score,
        reason=top_reason.split(":")[0],
        matched_phone=top_reason.split(":")[-1],
        ambiguous=False,
        candidates_considered=len(scored_candidates),
    )


__all__ = [
    "IdentityMatchResult",
    "build_phone_candidates",
    "extract_contact_phone_values",
    "match_contact_by_phone",
]
