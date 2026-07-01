from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.email_pipeline.classification import domain_of


OWN_DOMAINS = {"kmipt.ru", "cdpofoton.ru", "foton.school", "amocrm.ru", "amocrm.com"}
OWN_EMAILS = {"edu@kmipt.ru"}
HOTLINE_PHONE_DIGITS = {
    "88000000000",
    "88005553535",
    "74951234567",
}
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", re.I)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+7|7|8)\s*(?:\(?\d{3,4}\)?[\s.-]*)\d{2,3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)"
)

Participant = tuple[str, str, str]


@dataclass(frozen=True)
class ContactResult:
    contact_email: str | None
    contact_phone: str | None
    contact_name: str | None
    contact_source: str | None
    contact_missing: bool
    contact_ambiguous: bool
    contact_reason: str
    external_recipient_count: int


def resolve_customer_contact(
    *,
    direction: str,
    from_participant: Participant | None,
    to_participants: Iterable[Participant],
    cc_participants: Iterable[Participant],
    raw_text: str = "",
) -> ContactResult:
    """Resolve the customer contact from envelope structure, not from LLM text understanding."""
    to_external = [_clean_participant(item) for item in to_participants if _is_external_participant(item)]
    cc_external = [_clean_participant(item) for item in cc_participants if _is_external_participant(item)]
    external_recipients = [item for item in [*to_external, *cc_external] if item[1]]
    quoted_emails = {email.casefold() for email in EMAIL_RE.findall(raw_text or "")}

    if direction == "inbound":
        from_clean = _clean_participant(from_participant)
        if from_clean and _is_external_participant(from_clean):
            phone = _single_external_phone(raw_text) if raw_text else None
            return _result(
                email=from_clean[1],
                phone=phone,
                name=from_clean[0],
                source="header_from",
                reason="inbound_external_from",
                external_count=len(external_recipients),
            )
        quoted = _quoted_match([*to_external, *cc_external], quoted_emails)
        if quoted:
            return _result(
                email=quoted[1],
                phone=None,
                name=quoted[0],
                source="quoted_header",
                reason="quoted_email_matches_envelope",
                external_count=len(external_recipients),
            )
        return _missing("inbound_no_external_from", len(external_recipients))

    if len(external_recipients) > 1:
        return ContactResult(
            contact_email=None,
            contact_phone=None,
            contact_name=None,
            contact_source=None,
            contact_missing=False,
            contact_ambiguous=True,
            contact_reason="multiple_external_recipients",
            external_recipient_count=len(external_recipients),
        )
    if len(external_recipients) == 1:
        participant = external_recipients[0]
        source = "header_to" if participant in to_external else "header_cc"
        return _result(
            email=participant[1],
            phone=None,
            name=participant[0],
            source=source,
            reason="outbound_single_external_recipient",
            external_count=1,
        )
    quoted = _quoted_match([_clean_participant(from_participant)], quoted_emails)
    if quoted:
        return _result(
            email=quoted[1],
            phone=None,
            name=quoted[0],
            source="quoted_header",
            reason="quoted_email_matches_envelope",
            external_count=0,
        )
    return _missing("outbound_no_external_recipient", 0)


def contact_to_dict(result: ContactResult) -> dict[str, object]:
    return {
        "contact_email": result.contact_email,
        "contact_phone": result.contact_phone,
        "contact_name": result.contact_name,
        "contact_source": result.contact_source,
        "contact_missing": result.contact_missing,
        "contact_ambiguous": result.contact_ambiguous,
        "contact_reason": result.contact_reason,
        "external_recipient_count": result.external_recipient_count,
    }


def read_raw_eml_text(path: Path | None, *, limit: int = 65536) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        try:
            return path.read_bytes()[:limit].decode("utf-8", errors="ignore")
        except Exception:
            return ""


def _clean_participant(value: Participant | None) -> Participant | None:
    if not value:
        return None
    name, email, dom = value
    clean_email = str(email or "").strip().lower()
    clean_domain = str(dom or domain_of(clean_email)).strip().lower()
    return (str(name or "").strip(), clean_email, clean_domain)


def _is_external_participant(value: Participant | None) -> bool:
    participant = _clean_participant(value)
    if not participant:
        return False
    _, email, dom = participant
    return bool(email) and email not in OWN_EMAILS and dom not in OWN_DOMAINS


def _quoted_match(candidates: Iterable[Participant | None], quoted_emails: set[str]) -> Participant | None:
    for candidate in candidates:
        participant = _clean_participant(candidate)
        if participant and participant[1].casefold() in quoted_emails:
            return participant
    return None


def _single_external_phone(text: str) -> str | None:
    phones = []
    seen = set()
    for raw in PHONE_RE.findall(text or ""):
        digits = re.sub(r"\D", "", raw)
        if digits.startswith("8"):
            digits = "7" + digits[1:]
        if digits.startswith("7") and len(digits) == 11 and digits not in HOTLINE_PHONE_DIGITS and digits not in seen:
            seen.add(digits)
            phones.append("+" + digits)
    return phones[0] if len(phones) == 1 else None


def _result(
    *,
    email: str | None,
    phone: str | None,
    name: str | None,
    source: str,
    reason: str,
    external_count: int,
) -> ContactResult:
    return ContactResult(
        contact_email=email or None,
        contact_phone=phone or None,
        contact_name=name or None,
        contact_source=source,
        contact_missing=False,
        contact_ambiguous=False,
        contact_reason=reason,
        external_recipient_count=external_count,
    )


def _missing(reason: str, external_count: int) -> ContactResult:
    return ContactResult(
        contact_email=None,
        contact_phone=None,
        contact_name=None,
        contact_source=None,
        contact_missing=True,
        contact_ambiguous=False,
        contact_reason=reason,
        external_recipient_count=external_count,
    )
