from __future__ import annotations

from typing import Any, Mapping


def build_question_catalog_channel_context(
    approval_record: Mapping[str, Any],
    *,
    facts_fresh: bool = False,
) -> Mapping[str, Any]:
    """Build a read-only channel preview context from one ROP approval row."""
    record = dict(approval_record)
    approved_for_bot = record.get("approved_for_bot") is True or str(record.get("approved_for_bot")).lower() == "yes"
    final_answer = str(record.get("final_approved_answer") or "").strip()
    required_fact_keys = tuple(str(item).strip() for item in record.get("required_fact_keys") or () if str(item).strip())
    safe_to_use = approved_for_bot and bool(final_answer) and (not required_fact_keys or facts_fresh)
    return {
        "question_catalog_answer": record,
        "question_catalog_facts_fresh": facts_fresh,
        "question_catalog_safe_to_use": safe_to_use,
        "question_catalog_blocked_reason": None
        if safe_to_use
        else _blocked_reason(record, required_fact_keys=required_fact_keys, facts_fresh=facts_fresh),
    }


def _blocked_reason(record: Mapping[str, Any], *, required_fact_keys: tuple[str, ...], facts_fresh: bool) -> str:
    approved_for_bot = record.get("approved_for_bot") is True or str(record.get("approved_for_bot")).lower() == "yes"
    if not approved_for_bot:
        return "answer_not_approved_by_rop"
    if not str(record.get("final_approved_answer") or "").strip():
        return "final_answer_missing"
    if required_fact_keys and not facts_fresh:
        return "required_facts_not_confirmed_fresh"
    return "unknown_block"
