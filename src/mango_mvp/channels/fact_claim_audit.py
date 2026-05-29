from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


FACT_AUDIT_VERSION = "judge_v2_j1_fact_scope"


def audit_fact_claims(
    text: str,
    *,
    client_message: str,
    active_brand: str,
    retrieved_facts: Mapping[str, Any] | None,
    snapshot_path: Path,
) -> Mapping[str, Any]:
    """Classify support for business claims in a client-facing draft.

    This is shared by the dynamic judge and runtime gates so that measurement
    and live/shadow decisions use one source of truth.
    """
    retrieved = {
        str(key): str(value)
        for key, value in (retrieved_facts or {}).items()
        if str(key).strip() and str(value).strip()
    }
    registry = snapshot_fact_registry(snapshot_path)
    brand = str(active_brand or "").casefold()
    items: list[dict[str, Any]] = []
    seen_claims: set[tuple[str, str]] = set()

    for finding in _wrong_scope_fact_findings(text, client_message, active_brand=brand, registry=registry):
        signature = (str(finding.get("claim_type") or ""), str(finding.get("claim_text") or ""))
        if signature not in seen_claims:
            items.append(finding)
            seen_claims.add(signature)

    for claim in extract_semantic_fact_claims(text):
        signature = (str(claim.get("claim_type") or ""), str(claim.get("claim_text") or ""))
        if signature in seen_claims:
            continue
        retrieved_matches = [
            key
            for key, fact_text in retrieved.items()
            if fact_text_supports_terms(fact_text, claim.get("terms") or ())
        ]
        same_brand_matches = [
            fact["key"]
            for fact in registry.get(brand, ())
            if fact_text_supports_terms(str(fact.get("text") or ""), claim.get("terms") or ())
        ]
        other_brand_matches = [
            fact["key"]
            for item_brand, facts in registry.items()
            if item_brand != brand
            for fact in facts
            if fact_text_supports_terms(str(fact.get("text") or ""), claim.get("terms") or ())
        ]
        if retrieved_matches:
            level = "retrieved_match"
            matched_brand = brand
            matched_keys = retrieved_matches[:8]
        elif same_brand_matches:
            level = "same_brand_global_match"
            matched_brand = brand
            matched_keys = same_brand_matches[:8]
        elif other_brand_matches:
            level = "other_brand_match"
            matched_brand = "other"
            matched_keys = other_brand_matches[:8]
        else:
            level = "no_match"
            matched_brand = ""
            matched_keys = []
        items.append(
            {
                "claim_type": claim.get("claim_type"),
                "claim_text": claim.get("claim_text"),
                "level": level,
                "matched_brand": matched_brand,
                "matched_fact_keys": matched_keys,
                "reason": claim.get("reason") or "",
            }
        )
        seen_claims.add(signature)

    counts = Counter(str(item.get("level") or "") for item in items if str(item.get("level") or ""))
    return {
        "version": FACT_AUDIT_VERSION,
        "items": items,
        "counts_by_level": dict(counts),
        "has_wrong_scope": bool(counts.get("wrong_scope")),
        "has_unverified_claim": bool(counts.get("no_match") or counts.get("other_brand_match")),
    }


def extract_semantic_fact_claims(text: str) -> list[Mapping[str, Any]]:
    normalized = normalize_fact_text(text)
    if not normalized:
        return []
    patterns: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
        ("refund_unspent_balance", ("неистрачен", "остаток", "средств"), ("неистрачен",)),
        ("tbank_installment", ("т", "банк", "рассроч"), ("т", "банк", "рассроч")),
        ("foton_installment_terms", ("рассроч", "6", "10", "12", "месяц"), ("6", "10", "12", "рассроч")),
        ("dolyami", ("долями",), ("долями",)),
        ("annual_discount", ("14", "год", "скид"), ("14", "год", "скид")),
        ("semester_discount", ("10", "семестр", "скид"), ("10", "семестр", "скид")),
        ("online_frequency_2x90", ("2", "раз", "недел", "90"), ("2", "раз", "недел", "90")),
        ("address_sretenka", ("сретенк", "20"), ("сретенк",)),
        ("address_lyalina", ("лялин",), ("лялин",)),
        ("address_mendeleevo", ("менделеево",), ("менделеево",)),
        ("mts_link", ("мтс", "линк"), ("мтс", "линк")),
        ("office_hours", ("пн", "вс", "10", "18"), ("пн", "вс")),
        ("weekend_slots", ("выходн",), ("выходн",)),
        ("bank_transfer_invoice", ("перевод", "счет"), ("перевод", "счет")),
    )
    result: list[Mapping[str, Any]] = []
    for claim_type, terms, triggers in patterns:
        if claim_type == "annual_discount" and not _discount_term_near_number(text, percent="14", term="год"):
            continue
        if claim_type == "semester_discount" and not _discount_term_near_number(text, percent="10", term="семестр"):
            continue
        if not all(trigger in normalized for trigger in triggers):
            continue
        if claim_type == "bank_transfer_invoice" and re.search(r"(можно\s+ли|уточн|подтверд|передам)", normalized, re.I):
            continue
        result.append(
            {
                "claim_type": claim_type,
                "claim_text": shorten_claim_text(text, terms),
                "terms": terms,
            }
        )
    return result


def _discount_term_near_number(text: str, *, percent: str, term: str) -> bool:
    source = str(text or "").casefold().replace("ё", "е")
    pct = re.escape(str(percent))
    term_re = re.escape(str(term).casefold())
    return bool(
        re.search(rf"{pct}\s*%?[^.!?\n]{{0,30}}{term_re}", source, re.I)
        or re.search(rf"{term_re}[^.!?\n]{{0,30}}{pct}\s*%?", source, re.I)
    )


def fact_text_supports_terms(text: str, terms: Sequence[Any]) -> bool:
    normalized = normalize_fact_text(text)
    return bool(normalized) and all(str(term).casefold() in normalized for term in terms if str(term).strip())


def normalize_fact_text(value: object) -> str:
    return re.sub(r"[^a-zа-яё0-9%]+", " ", str(value or "").casefold().replace("ё", "е")).strip()


@lru_cache(maxsize=8)
def snapshot_fact_registry(snapshot_path: Path) -> Mapping[str, tuple[Mapping[str, str], ...]]:
    path = Path(snapshot_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    facts = payload.get("facts") if isinstance(payload, Mapping) else []
    if not isinstance(facts, Sequence) or isinstance(facts, (str, bytes, bytearray)):
        return {}
    registry: dict[str, list[Mapping[str, str]]] = {}
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        if not fact.get("allowed_for_client_answer"):
            continue
        brand = str(fact.get("brand") or "").casefold()
        key = str(fact.get("fact_key") or fact.get("fact_id") or "")
        if not brand or not key:
            continue
        text_parts = [
            str(fact.get(field) or "")
            for field in ("client_safe_text", "fact_text", "manager_check_text", "short_fact", "title")
        ]
        structured = fact.get("structured_value")
        if isinstance(structured, Mapping):
            text_parts.append(json.dumps(structured, ensure_ascii=False, sort_keys=True))
        registry.setdefault(brand, []).append({"key": key, "text": " ".join(part for part in text_parts if part)})
    return {brand: tuple(items) for brand, items in registry.items()}


def _wrong_scope_fact_findings(
    text: str,
    client_message: str,
    *,
    active_brand: str,
    registry: Mapping[str, Sequence[Mapping[str, str]]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    bot = normalize_fact_text(text)
    client = normalize_fact_text(client_message)
    if _asks_class_schedule_days(client) and _mentions_contact_hours(bot):
        findings.append(
            {
                "claim_type": "contact_hours_as_class_schedule",
                "claim_text": "контактные часы поданы как дни занятий",
                "level": "wrong_scope",
                "matched_brand": active_brand,
                "matched_fact_keys": _matching_fact_keys(registry, active_brand, ("пн", "вс", "10", "18"))[:8],
                "reason": "contact_hours_do_not_answer_class_schedule_days",
            }
        )
    if _mentions_address(bot) and not _asks_address(client):
        findings.append(
            {
                "claim_type": "address_on_non_address_question",
                "claim_text": "адрес подан на вопрос не про адрес",
                "level": "wrong_scope",
                "matched_brand": active_brand,
                "matched_fact_keys": _matching_fact_keys(registry, active_brand, ("сретенк",))[:8]
                or _matching_fact_keys(registry, active_brand, ("лялин",))[:8]
                or _matching_fact_keys(registry, active_brand, ("менделеево",))[:8],
                "reason": "address_fact_does_not_answer_current_question",
            }
        )
    if _asks_refund_policy(client) and _mentions_course_rules(bot):
        findings.append(
            {
                "claim_type": "course_rules_as_refund_policy",
                "claim_text": "правила курса поданы как ответ про возврат",
                "level": "wrong_scope",
                "matched_brand": active_brand,
                "matched_fact_keys": _matching_fact_keys(registry, active_brand, ("правил", "курс"))[:8],
                "reason": "course_rules_do_not_answer_refund_policy",
            }
        )
    return findings


def _asks_class_schedule_days(text: str) -> bool:
    return bool(re.search(r"(по\s+каким\s+дням|дни\s+занят|когда\s+занят|по\s+выходн|суббот|воскрес|будн)", text, re.I))


def _mentions_contact_hours(text: str) -> bool:
    return bool(
        re.search(
            r"(пн\s*[-–]?\s*вс|10[:\s]*00\s*[-–]?\s*18[:\s]*00|с\s*10\s*до\s*18|10\s*[-–]\s*18|"
            r"контакт|контактн\w*\s+центр|на\s+связи|ежедневно|часы\s+работы|работае[тм])",
            text,
            re.I,
        )
    )


def _mentions_address(text: str) -> bool:
    return bool(re.search(r"(сретенк|лялин|адрес)", text, re.I))


def _asks_address(text: str) -> bool:
    return bool(re.search(r"(адрес|где\s+(?:вы|находит|проходит)|куда\s+(?:ехать|приезжать)|локац|площадк|сретенк|лялин|менделеево)", text, re.I))


def _asks_refund_policy(text: str) -> bool:
    return bool(re.search(r"(возврат|верн[её]т|вернут|передума|деньги\s+назад|неистрачен)", text, re.I))


def _mentions_course_rules(text: str) -> bool:
    return bool(re.search(r"(правил\w*\s+(?:курс|поведен|занят)|цифров\w*\s+этикет|поведен)", text, re.I))


def _matching_fact_keys(
    registry: Mapping[str, Sequence[Mapping[str, str]]],
    brand: str,
    terms: Sequence[str],
) -> list[str]:
    return [
        str(fact.get("key") or "")
        for fact in registry.get(str(brand or "").casefold(), ())
        if fact_text_supports_terms(str(fact.get("text") or ""), terms)
    ]


def shorten_claim_text(text: str, terms: Sequence[str], *, max_chars: int = 140) -> str:
    source = " ".join(str(text or "").split())
    if len(source) <= max_chars:
        return source
    normalized = normalize_fact_text(source)
    positions = [normalized.find(str(term).casefold()) for term in terms if str(term).casefold() in normalized]
    if positions:
        start = max(0, min(positions) - 30)
        return ("…" if start else "") + source[start : start + max_chars].rstrip() + "…"
    return source[: max_chars - 1].rstrip() + "…"
