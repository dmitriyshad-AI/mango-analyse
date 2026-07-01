from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


MONEY_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[\s\u00a0]\d{3})+|\d+)\s*(?:руб\.?|₽|р\.)", re.I)
MESSAGE_ID_RE = re.compile(r"<[^>]+>")
QUOTE_MARKER_RE = re.compile(r"(?:^|\n).{0,80}(?:писал\(а\)|wrote:|написал|написала)\s*", re.I)
REPLY_PREFIX_RE = re.compile(r"^\s*(?:re|fwd|fw)\s*:", re.I)
NO_REPLY_LOCAL_RE = re.compile(r"^(?:no-?reply|noreply|newsletter|news|mailer-daemon|postmaster|notification|notifications|notify|robot|auto)$", re.I)

TRUSTED_DOMAINS = {"kmipt.ru", "cdpofoton.ru", "foton.school", "amocrm.ru", "amocrm.com"}
MONEY_LIMIT_RUB = 1_000_000
LOW_UTILITY_CHARS = 80
ATTACHMENT_ONLY_CHARS = 160


@dataclass(frozen=True)
class QualityResult:
    memory_status: str
    quality_flags: list[str]
    money_amounts_rub: list[int]
    amount_zero: bool
    amount_uncertain: bool
    requires_human_confirmation: bool
    safe_next_step_note: str | None
    thread_id: str
    thread_basis: str
    is_broadcast_envelope: bool
    quote_marker_present: bool


def evaluate_quality(row: dict[str, Any]) -> QualityResult:
    payload = row.get("summary_payload") or {}
    full_text = str(row.get("full_clean_text") or "")
    subject = str(row.get("subject") or row.get("subject_full") or "")
    combined = "\n".join([subject, full_text])
    source_amounts = extract_money_amounts(combined)
    model_amount = _as_int(payload.get("amount_rub"))
    amounts = _unique_ints([*source_amounts, *([model_amount] if model_amount is not None else [])])
    amount_zero = any(amount == 0 for amount in amounts) or model_amount == 0
    has_amount_candidate = model_amount is not None or bool(amounts)
    amount_uncertain = (bool(payload.get("amount_uncertain")) and has_amount_candidate) or any(
        amount > MONEY_LIMIT_RUB for amount in amounts
    )

    event_type = _enum_text(payload.get("event_type"), "other")
    money_direction = _enum_text(payload.get("money_direction"), "none")
    amount_kind = _enum_text(payload.get("amount_kind"), "")
    has_model_fact = any(
        payload.get(key)
        for key in ("student_name", "grade", "subject_area", "contract_no", "document_no", "deadline_date")
    )
    has_money_fact = bool(amounts) or money_direction != "none" or amount_kind in {"quote", "actual_payment", "refund"}
    has_structural_fact = event_type not in {"other", "broadcast"} or has_model_fact or has_money_fact
    short_text = len(full_text.strip()) < LOW_UTILITY_CHARS
    has_attachment = bool(row.get("has_attachment") or payload.get("has_attachment"))
    quote_marker = bool(QUOTE_MARKER_RE.search(full_text))
    reply_like = bool(REPLY_PREFIX_RE.match(subject))
    thread_id, thread_basis = build_thread_id(row)
    broadcast = is_broadcast_envelope(row)

    flags: list[str] = []
    if any(amount < 0 for amount in amounts):
        flags.append("negative_amount")
    if any(amount > MONEY_LIMIT_RUB for amount in amounts):
        flags.append("money_over_limit")
    if amount_uncertain:
        flags.append("amount_uncertain")
    if amount_zero:
        flags.append("amount_zero")
    if has_attachment:
        flags.append("has_attachment")
    if short_text:
        flags.append("short_clean_text")
    if quote_marker:
        flags.append("quote_marker")
    if broadcast:
        flags.append("broadcast_envelope")

    requires_human = (
        money_direction != "none"
        or event_type in {"payment", "refund", "contract", "application", "scheduling"}
        or has_money_fact
    )
    safe_next_step = _safe_next_step(payload.get("next_step"), requires_human=requires_human)

    if any(amount < 0 for amount in amounts) or any(amount > MONEY_LIMIT_RUB for amount in amounts) or amount_uncertain:
        status = "financial_unverified"
    elif broadcast:
        status = "broadcast_not_usable"
    elif has_attachment and not has_structural_fact and len(full_text.strip()) < ATTACHMENT_ONLY_CHARS:
        status = "attachment_only"
    elif quote_marker and not has_structural_fact:
        status = "needs_thread_context" if reply_like or thread_basis != "sha" else "quote_only"
    elif short_text and not has_structural_fact:
        status = "needs_thread_context" if reply_like and thread_basis != "sha" else "thin_ack"
    else:
        status = "usable_memory"

    return QualityResult(
        memory_status=status,
        quality_flags=flags,
        money_amounts_rub=amounts,
        amount_zero=amount_zero,
        amount_uncertain=amount_uncertain,
        requires_human_confirmation=requires_human,
        safe_next_step_note=safe_next_step,
        thread_id=thread_id,
        thread_basis=thread_basis,
        is_broadcast_envelope=broadcast,
        quote_marker_present=quote_marker,
    )


def extract_money_amounts(text: str) -> list[int]:
    amounts: list[int] = []
    for match in MONEY_RE.finditer(text or ""):
        digits = re.sub(r"\D", "", match.group(1))
        if digits:
            amounts.append(int(digits))
    return _unique_ints(amounts)


def build_thread_id(row: dict[str, Any]) -> tuple[str, str]:
    references = _message_ids(str(row.get("references") or ""))
    in_reply_to = _message_ids(str(row.get("in_reply_to") or ""))
    message_id = _message_ids(str(row.get("message_id") or ""))
    if references:
        basis = references[0]
        basis_kind = "references"
    elif in_reply_to:
        basis = in_reply_to[0]
        basis_kind = "in_reply_to"
    elif message_id:
        basis = message_id[0]
        basis_kind = "message_id"
    else:
        basis = str(row.get("message_sha256") or "")
        basis_kind = "sha"
    digest = hashlib.sha256(basis.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"email-thread-{digest}", basis_kind


def is_broadcast_envelope(row: dict[str, Any]) -> bool:
    from_email = str(row.get("from_email") or "")
    from_domain = str(row.get("from_domain") or "").lower()
    local = from_email.split("@", 1)[0].lower() if "@" in from_email else ""
    if from_domain in TRUSTED_DOMAINS:
        return False
    envelope_bulk = bool(row.get("list_unsubscribe")) or str(row.get("precedence") or "").lower() in {"bulk", "list", "junk"}
    no_reply = bool(NO_REPLY_LOCAL_RE.match(local))
    return bool(from_domain) and (envelope_bulk or no_reply)


def quality_to_dict(result: QualityResult) -> dict[str, Any]:
    return {
        "memory_status": result.memory_status,
        "quality_flags": result.quality_flags,
        "money_amounts_rub": result.money_amounts_rub,
        "amount_zero": result.amount_zero,
        "amount_uncertain": result.amount_uncertain,
        "requires_human_confirmation": result.requires_human_confirmation,
        "safe_next_step_note": result.safe_next_step_note,
        "thread_id": result.thread_id,
        "thread_basis": result.thread_basis,
        "is_broadcast_envelope": result.is_broadcast_envelope,
        "quote_marker_present": result.quote_marker_present,
    }


def _safe_next_step(value: object, *, requires_human: bool) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if requires_human:
        return f"Требует ручной проверки менеджером: {text}"
    return text


def _message_ids(text: str) -> list[str]:
    values = MESSAGE_ID_RE.findall(text or "")
    if values:
        return values
    stripped = text.strip()
    return [stripped] if stripped else []


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _enum_text(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _unique_ints(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
