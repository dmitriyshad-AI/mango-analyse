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
MASS_RECIPIENT_THRESHOLD = 2


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
    paid_amount_rub: int | None
    refund_amount_rub: int | None
    quoted_price_rub: int | None
    amount_missing: bool


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
    amount_is_total = bool(payload.get("amount_is_total"))
    has_model_fact = any(
        payload.get(key)
        for key in (
            "student_name",
            "payer_name",
            "contact_name",
            "grade",
            "subject_area",
            "contract_no",
            "document_no",
            "deadline_date",
        )
    )
    has_money_fact = bool(amounts) or money_direction != "none" or amount_kind in {"quote", "actual_payment", "refund"}
    has_structural_fact = event_type not in {"other", "broadcast"} or has_model_fact or has_money_fact
    short_text = len(full_text.strip()) < LOW_UTILITY_CHARS
    has_attachment = bool(row.get("has_attachment") or payload.get("has_attachment"))
    quote_marker = bool(QUOTE_MARKER_RE.search(full_text))
    quote_only = _is_quote_only(full_text)
    reply_like = bool(REPLY_PREFIX_RE.match(subject))
    thread_id, thread_basis = build_thread_id(row)
    own_template_broadcast = _is_own_template_broadcast(row, event_type=event_type, reply_like=reply_like)
    model_structural_broadcast = _is_model_broadcast_with_structure(
        row,
        event_type=event_type,
        reply_like=reply_like,
        has_attachment=has_attachment,
    )
    broadcast = is_broadcast_envelope(row) or own_template_broadcast or model_structural_broadcast
    is_plain_ack = bool(payload.get("is_plain_acknowledgement"))
    if len(amounts) > 1 and model_amount is not None and not amount_is_total:
        amount_uncertain = True
    paid_amount, refund_amount, quoted_amount = _mapped_amounts(
        amount_kind=amount_kind,
        model_amount=model_amount,
        amount_uncertain=amount_uncertain,
    )
    amount_missing = amount_kind in {"actual_payment", "refund"} and model_amount is None

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
    if bool(row.get("is_outbound_template")):
        flags.append("outbound_template")
    if bool(row.get("is_mass_recipient")):
        flags.append("mass_recipient")
    if model_structural_broadcast:
        flags.append("model_broadcast_structural")
    if quote_only:
        flags.append("quote_only")
    if amount_missing:
        flags.append("amount_missing")

    base_requires_human = (
        money_direction != "none"
        or event_type in {"payment", "refund", "contract", "application", "scheduling", "medical", "tax"}
        or has_money_fact
    )
    plain_ack_relaxation = (
        is_plain_ack
        and event_type == "payment"
        and money_direction != "out"
        and not amount_uncertain
        and not any(amount > MONEY_LIMIT_RUB for amount in amounts)
    )
    requires_human = base_requires_human and not plain_ack_relaxation
    safe_next_step = _safe_next_step(payload.get("next_step"), requires_human=requires_human)

    if any(amount < 0 for amount in amounts) or any(amount > MONEY_LIMIT_RUB for amount in amounts) or amount_uncertain:
        status = "financial_unverified"
    elif broadcast:
        status = "broadcast_not_usable"
    elif has_attachment and _attachment_needs_context(full_text, has_structural_fact=has_structural_fact, has_model_fact=has_model_fact, model_amount=model_amount):
        status = "attachment_only"
    elif quote_only:
        status = "quote_only"
    elif quote_marker and (short_text or not has_structural_fact):
        status = "needs_thread_context" if reply_like or thread_basis != "sha" else "quote_only"
    elif short_text and _short_business_reply_needs_context(
        event_type=event_type,
        money_direction=money_direction,
        amount_kind=amount_kind,
        model_amount=model_amount,
        reply_like=reply_like,
        is_plain_ack=is_plain_ack,
    ):
        status = "needs_thread_context"
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
        paid_amount_rub=paid_amount,
        refund_amount_rub=refund_amount,
        quoted_price_rub=quoted_amount,
        amount_missing=amount_missing,
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
    envelope_bulk = bool(row.get("list_unsubscribe")) or str(row.get("precedence") or "").lower() in {"bulk", "list", "junk"}
    no_reply = bool(NO_REPLY_LOCAL_RE.match(local))
    foreign_broadcast = bool(from_domain) and from_domain not in TRUSTED_DOMAINS and (envelope_bulk or no_reply)
    own_broadcast = str(row.get("direction") or "") == "outbound" and (
        bool(row.get("is_mass_recipient")) or int(row.get("external_recipient_count") or 0) >= MASS_RECIPIENT_THRESHOLD
    )
    return foreign_broadcast or own_broadcast


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
        "paid_amount_rub": result.paid_amount_rub,
        "refund_amount_rub": result.refund_amount_rub,
        "quoted_price_rub": result.quoted_price_rub,
        "amount_missing": result.amount_missing,
    }


def sanitize_summary_payload_for_quality(payload: dict[str, Any], result: QualityResult) -> dict[str, Any]:
    cleaned = dict(payload)
    if result.memory_status != "financial_unverified" and not result.amount_uncertain:
        return cleaned
    for key in ("summary", "topic", "next_step"):
        if cleaned.get(key):
            cleaned[key] = _scrub_unverified_money(str(cleaned[key]))
    if result.amount_uncertain or any(amount > MONEY_LIMIT_RUB for amount in result.money_amounts_rub):
        if _as_int(cleaned.get("amount_rub")) and _as_int(cleaned.get("amount_rub")) > MONEY_LIMIT_RUB:
            cleaned["amount_rub"] = None
        items = []
        for item in cleaned.get("amount_items") or []:
            if not isinstance(item, dict):
                continue
            next_item = dict(item)
            if _as_int(next_item.get("amount_rub")) and _as_int(next_item.get("amount_rub")) > MONEY_LIMIT_RUB:
                next_item["amount_rub"] = None
                if next_item.get("description"):
                    next_item["description"] = _scrub_unverified_money(str(next_item["description"]))
            items.append(next_item)
        cleaned["amount_items"] = items
        cleaned["amount_uncertain"] = True
    return cleaned


def _safe_next_step(value: object, *, requires_human: bool) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if requires_human:
        return f"Требует ручной проверки менеджером: {text}"
    return text


def _is_own_template_broadcast(row: dict[str, Any], *, event_type: str, reply_like: bool) -> bool:
    if str(row.get("direction") or "") != "outbound" or not bool(row.get("is_outbound_template")):
        return False
    return not reply_like or event_type == "broadcast"


def _is_model_broadcast_with_structure(
    row: dict[str, Any],
    *,
    event_type: str,
    reply_like: bool,
    has_attachment: bool,
) -> bool:
    if event_type != "broadcast":
        return False
    direction = str(row.get("direction") or "")
    from_domain = str(row.get("from_domain") or "").lower()
    if direction == "outbound" and from_domain in TRUSTED_DOMAINS:
        return not reply_like
    if direction == "inbound" and from_domain and from_domain not in TRUSTED_DOMAINS:
        return has_attachment or bool(row.get("list_unsubscribe")) or str(row.get("precedence") or "").lower() in {"bulk", "list", "junk"}
    return False


def _is_quote_only(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", str(text or "").strip())
    if not stripped:
        return False
    return len(stripped) < LOW_UTILITY_CHARS and bool(QUOTE_MARKER_RE.search(stripped))


def _attachment_needs_context(
    text: str,
    *,
    has_structural_fact: bool,
    has_model_fact: bool,
    model_amount: int | None,
) -> bool:
    if len(str(text or "").strip()) >= ATTACHMENT_ONLY_CHARS:
        return False
    return not has_model_fact and model_amount is None and not has_structural_fact


def _short_business_reply_needs_context(
    *,
    event_type: str,
    money_direction: str,
    amount_kind: str,
    model_amount: int | None,
    reply_like: bool,
    is_plain_ack: bool,
) -> bool:
    if model_amount is not None:
        return False
    business_event = event_type in {"payment", "refund", "application", "contract", "scheduling"}
    money_event = money_direction != "none" or amount_kind in {"actual_payment", "refund", "quote"}
    if not (business_event or money_event):
        return False
    if event_type == "payment" and is_plain_ack and not reply_like:
        return False
    return True


def _mapped_amounts(
    *,
    amount_kind: str,
    model_amount: int | None,
    amount_uncertain: bool,
) -> tuple[int | None, int | None, int | None]:
    if model_amount is None or amount_uncertain:
        return None, None, None
    if amount_kind == "actual_payment":
        return model_amount, None, None
    if amount_kind == "refund":
        return None, model_amount, None
    if amount_kind == "quote":
        return None, None, model_amount
    return None, None, None


def _scrub_unverified_money(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(1))
        if digits and int(digits) > MONEY_LIMIT_RUB:
            return "<сумма требует проверки>"
        return match.group(0)

    return MONEY_RE.sub(repl, text or "")


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
