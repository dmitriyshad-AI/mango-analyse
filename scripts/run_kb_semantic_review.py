#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mango_mvp.knowledge_base.fact_registry import (  # noqa: E402
    FRESHNESS_SLA_STATUS_UNKNOWN,
    evaluate_fact_freshness_sla,
)


DEFAULT_RELEASE_DIR = Path("product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved")
SEMANTIC_REVIEW_SCHEMA_VERSION = "kb_semantic_review_v1"
SEMANTIC_REVIEW_RULESET_VERSION = "kb_semantic_rules_2026_07_28"

# A price window lives only in the fact_key (`...before_2026_07_01...`); nothing
# else in the fact says when the offer stops being true -- `valid_until` is set
# release-wide, not per offer. Without this the reviewer cannot tell a live price
# from one that expired weeks ago.
PRICE_WINDOW_RE = re.compile(r"before_(\d{4})_(\d{2})_(\d{2})")

MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
# `(?<!\d)` and the ascending check keep "интенсивы 2026 — 15 апреля" from being
# read as the range "26-15 апреля".
FINISHED_PERIOD_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s*[-\u2013\u2014]\s*(\d{1,2})\s*("
    + "|".join(MONTHS_RU)
    + r")(?:\s+(\d{4})(?:\s*г(?:ода)?)?)?"
)
HISTORICAL_PERIOD_RE = re.compile(r"\b(?:прошл\w*|заверш\w*|законч\w*|состоял\w*|был[аио]?)\b", re.IGNORECASE)


def price_window_deadline(fact_key: str) -> date | None:
    """Deadline encoded in the fact_key of a price fact, if any."""
    match = PRICE_WINDOW_RE.search(fact_key)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def finished_periods(text: str, *, today: date) -> list[str]:
    """Date ranges in client text whose last day is already in the past."""
    finished: list[str] = []
    for match in FINISHED_PERIOD_RE.finditer(text):
        context = text[max(0, match.start() - 80) : min(len(text), match.end() + 80)]
        if HISTORICAL_PERIOD_RE.search(context):
            continue
        first_day, last = int(match.group(1)), int(match.group(2))
        if not 1 <= first_day < last <= 31:
            continue
        try:
            last_day = date(int(match.group(4) or today.year), MONTHS_RU[match.group(3)], last)
        except ValueError:
            continue
        if last_day < today:
            finished.append(match.group(0))
    return finished


def file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def resolve_release_path(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve(strict=False) if expanded.is_absolute() else (PROJECT_ROOT / expanded).resolve(strict=False)

NON_MONEY_PATH_MARKERS = (
    "total_lessons",
    "weekly_lessons",
    "daily_hours",
    "semester_1_weeks",
    "semester_2_weeks",
    "daily_pairs",
    "pair_duration_minutes",
    "duration_weeks",
    "experience_years",
    "retroactive_years",
    "lead_time_days",
    "certificate_lead_time_days",
)
GLOBAL_FORBIDDEN_CLIENT_MARKERS = (
    "source_id",
    "fact_id",
    "freshness_status",
    "AMO",
    "Tallanto",
    "GPT",
    "Claude",
    "Codex",
    "ChatGPT",
    "я бот",
    "я ИИ",
    "нейросеть",
    "искусственный интеллект",
    "Это автоматический ответ",
    "автоматический ответ",
    "раньше сотрудничали",
    "были одно",
    "наш партнёр",
    "наш партнер",
    "q7 закрыт",
    "ответу бухгалтерии",
    "dynamic needs check",
)
FOTON_FORBIDDEN_CLIENT_MARKERS = (
    "УНПК",
    "АНО ДПО",
    "НОУ УНПК",
    "kmipt.ru",
    "@unpk_mipt",
    "@unpkmfti",
    "@unpk mipt",
    "unpkmfti",
    "Сретенка",
    "Сретенка, 20",
    "edu@kmipt.ru",
)
UNPK_FORBIDDEN_CLIENT_MARKERS = (
    "Фотон",
    "ЦДПО",
    "ЦРДО",
    "cdpofoton.ru",
    "edu@cdpofoton.ru",
    "Т-Банк",
    "Долями",
    "рассрочка через банк",
    "через банк",
)
STALE_CERTIFICATE_MARKERS = (
    "3 рабочих дня",
    "3 рабочих дней",
    "тип справки",
    "работа / налоговая / иное",
)
TECHNICAL_ENGLISH_CLIENT_RE = re.compile(
    r"\b(?:prices?|lesson|session|package|base|plus|one\s+block|one\s+subject|two\s+subjects|"
    r"after\s+20\d{2}|before\s+20\d{2}|moscow|dolgoprudny|location|start\s+date|"
    r"online\s+platform|free\s+morning\s+club|factultative|program|pair\s+duration|"
    r"duration\s+minutes|non\s+stacking|over\s+18|under\s+18|used\s+for|receipt\s+rules|"
    r"fiztech\s+xxi|veb\s+rf|former\s+student)\b",
    re.IGNORECASE,
)
GENERIC_ROP_QUESTIONS = {
    "Можно ли использовать этот факт в ответе клиенту текущего бренда?",
    "Подтверждаете эту цену и область применения для бота?",
}
MONEY_FACT_TYPES = {"price", "discount", "promocode", "installment", "tax", "matkap", "refund"}
NON_MONEY_RUB_FACT_TYPES = {"course_parameter", "deadline", "program", "documents", "contact", "location", "teacher", "policy"}
VERIFIED_FRESHNESS = {"verified", "document_verified", "fresh_verified", "current"}
TIME_SENSITIVE_FACT_TYPES = {
    "price",
    "discount",
    "promocode",
    "deadline",
    "program",
    "camp_lvsh",
    "camp_city",
    "intensive",
    "installment",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    check_id: str
    message: str
    item_id: str = ""
    evidence: str = ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run semantic sanity review for a KB release folder.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    report = run_kb_semantic_review(args.release_dir, out_dir=args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["semantic_pass"] else 1


def run_kb_semantic_review(
    release_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    release_root = resolve_release_path(Path(release_dir))
    snapshot = load_snapshot(release_root)
    facts = load_facts(release_root, snapshot)
    approval_queue = load_approval_queue(release_root)

    checked_on = today or datetime.now(timezone.utc).date()

    findings: list[Finding] = []
    findings.extend(review_facts(facts, today=checked_on))
    findings.extend(review_approval_queue(approval_queue))
    findings.extend(review_snapshot(snapshot, facts=facts))
    findings.extend(review_fact_mirror_consistency(snapshot, facts=facts))

    counts_by_severity = Counter(finding.severity for finding in findings)
    blocking = [finding for finding in findings if finding.severity in {"P0", "P1"}]
    report = {
        "schema_version": SEMANTIC_REVIEW_SCHEMA_VERSION,
        "reviewer_ruleset_version": SEMANTIC_REVIEW_RULESET_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_dir": portable_path(release_root),
        "snapshot_path": portable_path(release_root / "kb_release_v3_snapshot.json"),
        # Without these a review cannot be attributed to anything: the same
        # report file survives snapshot rewrites and reviewer upgrades alike.
        "snapshot_sha256": file_sha256(release_root / "kb_release_v3_snapshot.json"),
        "facts_registry_sha256": file_sha256(release_root / "facts_registry.jsonl"),
        "checked_on": checked_on.isoformat(),
        "reviewer_check_ids": sorted({finding.check_id for finding in findings}),
        "formal_quality_passed": bool((snapshot.get("quality_summary") or {}).get("quality_passed")),
        "semantic_pass": not blocking,
        "facts_total": len(facts),
        "approval_queue_items": len(approval_queue),
        "findings_total": len(findings),
        "blocking_findings": len(blocking),
        "findings_by_severity": dict(counts_by_severity),
        "findings": [asdict(finding) for finding in findings],
    }

    if out_dir is not None:
        out_root = guard_output_dir(Path(out_dir))
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "semantic_review.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (out_root / "semantic_review.md").write_text(render_markdown(report), encoding="utf-8")

    return report


def review_facts(facts: Sequence[Mapping[str, Any]], *, today: date | None = None) -> list[Finding]:
    checked_on = today or datetime.now(timezone.utc).date()
    findings: list[Finding] = []
    for fact in facts:
        fact_id = str(fact.get("fact_id") or fact.get("fact_key") or "")
        fact_type = str(fact.get("fact_type") or "")
        brand = str(fact.get("brand") or "")
        structured = as_mapping(fact.get("structured_value"))
        text = str(fact.get("client_safe_text") or "")
        fact_key = str(fact.get("fact_key") or "")
        path = str(structured.get("path") or fact_key)
        allowed_flag = is_true(fact.get("allowed_for_client_answer"))
        allowed = allowed_flag and bool(text.strip())
        freshness = str(fact.get("freshness_status") or fact.get("verification_status") or "")
        freshness_ok = freshness in VERIFIED_FRESHNESS
        requires_confirmation = is_true(fact.get("requires_manager_confirmation"))

        if allowed and re.search(r"(?<!не\s)скидки\s+суммируются", text.casefold().replace("ё", "е")):
            findings.append(
                Finding(
                    "P1",
                    "discount_stacking_contradiction",
                    "Клиентский факт утверждает, что скидки суммируются, хотя действующее правило запрещает суммирование.",
                    item_id=fact_id,
                    evidence=text[:220],
                )
            )
        if allowed and " , " in text:
            findings.append(
                Finding(
                    "P1",
                    "machine_joined_client_text",
                    "Клиентский текст содержит машинную склейку ключа данных.",
                    item_id=fact_id,
                    evidence=text[:220],
                )
            )

        if allowed_flag and not text.strip():
            findings.append(
                Finding(
                    "P0",
                    "allowed_fact_has_empty_client_text",
                    "Факт разрешён для клиентского ответа, но клиентский текст пустой.",
                    item_id=fact_id,
                    evidence=fact_key,
                )
            )

        if allowed_flag and requires_confirmation:
            findings.append(
                Finding(
                    "P0",
                    "allowed_fact_requires_manager_confirmation",
                    "Факт нельзя одновременно разрешать клиенту и требовать подтверждение менеджера.",
                    item_id=fact_id,
                    evidence=f"{freshness} | {text[:220]}",
                )
            )

        if requires_confirmation and freshness_ok:
            findings.append(
                Finding(
                    "P0",
                    "verified_fact_marked_requires_manager_confirmation",
                    "Подтвержденный документом факт ошибочно помечен как требующий ручного подтверждения.",
                    item_id=fact_id,
                    evidence=f"{freshness} | {fact_key}",
                )
            )

        if allowed and (fact_type == "promocode" or "promo" in fact_key.casefold() or "promo" in path.casefold()):
            findings.append(
                Finding(
                    "P0",
                    "promo_code_allowed_for_client",
                    "Промокоды исключены из клиентской базы бота и не должны быть client-safe.",
                    item_id=fact_id,
                    evidence=text[:220],
                )
            )

        if allowed and brand == "foton" and re.search(r"vk\.com/kmipt_edu|kmipt_edu", text, re.I):
            findings.append(
                Finding(
                    "P0",
                    "old_foton_vk_handle_allowed",
                    "В клиентском факте Фотона остался старый VK, который путал Фотон и УНПК.",
                    item_id=fact_id,
                    evidence=text[:220],
                )
            )

        if allowed and brand == "unpk" and re.search(r"лобн|лобненск", text, re.I):
            findings.append(
                Finding(
                    "P0",
                    "closed_unpk_lobnya_allowed",
                    "Лобня закрыта и не должна попадать в клиентские факты УНПК.",
                    item_id=fact_id,
                    evidence=text[:220],
                )
            )

        if allowed and brand == "unpk" and re.search(r"patsayeva_2x_week|2\s*раз[а]?\s+в\s+нед|ЕГЭ-?математика|ФТЛ\s*2", f"{fact_key} {text}", re.I):
            if "patsayeva_2x_week" in fact_key or "пацаев" in text.casefold():
                findings.append(
                    Finding(
                        "P0",
                        "removed_unpk_patsayeva_2x_week_allowed",
                        "Снятый блок Пацаева 2 раза в неделю не должен быть client-safe.",
                        item_id=fact_id,
                        evidence=f"{fact_key} | {text[:220]}",
                    )
                )

        if allowed and brand == "unpk" and re.search(r"11\s*900|56\s*500|94\s*000", text):
            findings.append(
                Finding(
                    "P0",
                    "removed_preschool_prices_allowed",
                    "Старые цены дошкольников не должны попадать в клиентские факты.",
                    item_id=fact_id,
                    evidence=text[:220],
                )
            )

        if allowed and re.search(r"after_2026_|после\s+1\s+(?:июля|августа)|после\s+повышени|цена\s+выраст", f"{fact_key} {text}", re.I):
            findings.append(
                Finding(
                    "P0",
                    "future_price_allowed_for_client",
                    "Будущая цена после повышения не должна быть client-safe.",
                    item_id=fact_id,
                    evidence=f"{fact_key} | {text[:220]}",
                )
            )

        # Mirror of future_price_allowed_for_client for the other end of the
        # window. Restricted to price facts on purpose: schedule deadlines carry
        # the same `before_...` shape and must not be flagged as stale offers.
        window_deadline = price_window_deadline(fact_key) if fact_type == "price" else None
        if allowed and window_deadline is not None and window_deadline <= checked_on:
            findings.append(
                Finding(
                    "P1",
                    "expired_price_window_allowed_for_client",
                    "Цена вне своего окна применимости остаётся client-safe: окно из fact_key уже закрылось.",
                    item_id=fact_id,
                    evidence=f"window_until={window_deadline.isoformat()} | {fact_key} | {text[:180]}",
                )
            )

        for period in finished_periods(text, today=checked_on) if allowed else ():
            findings.append(
                Finding(
                    "P1",
                    "finished_period_in_client_text",
                    "Клиентский текст называет период, который уже закончился, как доступный.",
                    item_id=fact_id,
                    evidence=f"period={period} | {text[:180]}",
                )
            )
            break

        confirmed_social_proof = (
            (brand == "unpk" and "results_social_proof.total_alumni" in fact_key)
            or (brand == "foton" and "results_social_proof.industry_rating_2025" in fact_key)
        )
        if allowed and not confirmed_social_proof and re.search(r"100\s*000\s+учен|лидер\s+(?:отрасли|2025)", text, re.I):
            findings.append(
                Finding(
                    "P1",
                    "unverified_social_proof_allowed",
                    "Сомнительное маркетинговое число или статус не должен быть client-safe без отдельного подтверждения.",
                    item_id=fact_id,
                    evidence=text[:220],
                )
            )

        if allowed and fact_type == "price":
            amount = structured.get("amount")
            if isinstance(amount, (int, float)) and amount < 3000:
                findings.append(
                    Finding(
                        "P0",
                        "implausible_low_client_price",
                        "Клиентская цена выглядит неправдоподобно низкой.",
                        item_id=fact_id,
                        evidence=f"{amount} RUB | {text[:220]}",
                    )
                )
            amount_min = structured.get("amount_min")
            amount_max = structured.get("amount_max")
            if isinstance(amount_min, (int, float)) and amount_min < 3000:
                findings.append(
                    Finding(
                        "P0",
                        "implausible_low_price_range",
                        "Нижняя граница клиентского диапазона цены выглядит неправдоподобно низкой.",
                        item_id=fact_id,
                        evidence=f"{amount_min}-{amount_max} RUB | {text[:220]}",
                    )
                )

        if allowed and fact_type in NON_MONEY_RUB_FACT_TYPES and structured.get("currency") == "RUB":
            findings.append(
                Finding(
                    "P0",
                    "non_money_fact_has_rub",
                    "Недежный факт получил валюту RUB.",
                    item_id=fact_id,
                    evidence=f"{fact_type} | {path} | {text[:220]}",
                )
            )

        if allowed and any(marker in path.casefold() for marker in NON_MONEY_PATH_MARKERS):
            if fact_type == "price" or structured.get("currency") == "RUB":
                findings.append(
                    Finding(
                        "P0",
                        "non_money_path_became_price",
                        "Учебный параметр или срок распознан как цена.",
                        item_id=fact_id,
                        evidence=f"{fact_type} | {path} | {text[:220]}",
                    )
                )

        if ".range.min" in fact_key or ".range.max" in fact_key:
            if not structured.get("do_not_use_as_current_price"):
                findings.append(
                    Finding(
                        "P0",
                        "split_current_price_range",
                        "Текущий диапазон цены разорван на отдельные min/max факты.",
                        item_id=fact_id,
                        evidence=f"{fact_key} | {text[:220]}",
                    )
                )

        if structured.get("amount_min") is not None or structured.get("amount_max") is not None:
            if structured.get("amount_min") is None or structured.get("amount_max") is None:
                findings.append(
                    Finding("P0", "incomplete_price_range", "Диапазон цены содержит только одну границу.", item_id=fact_id)
                )
            elif structured["amount_min"] > structured["amount_max"]:
                findings.append(
                    Finding(
                        "P0",
                        "reversed_price_range",
                        "Нижняя граница диапазона больше верхней.",
                        item_id=fact_id,
                        evidence=f"{structured['amount_min']} > {structured['amount_max']}",
                    )
                )

        if allowed:
            findings.extend(review_client_text(text, brand=brand, item_id=fact_id))
            findings.extend(review_fact_freshness(fact, fact_type=fact_type, item_id=fact_id))
            findings.extend(review_fact_freshness_sla(fact, item_id=fact_id))
            route_policy = str(fact.get("route_policy") or "")
            template_required = bool(fact.get("bot_template_required"))
            if route_policy == "bot_answer_self_for_pilot" and has_machine_short_tail(text) and not template_required:
                findings.append(
                    Finding(
                        "P1",
                        "pilot_client_text_has_machine_short_tail",
                        "Пилотный клиентский факт выглядит как машинный обрывок и не годится для самостоятельного ответа.",
                        item_id=fact_id,
                        evidence=text[:220],
                    )
                )
            if template_required and not text:
                findings.append(
                    Finding(
                        "P1",
                        "template_required_fact_has_empty_text",
                        "Факт требует шаблон, но не содержит текста-источника для шаблона.",
                        item_id=fact_id,
                    )
                )
            if fact_type == "discount" and route_policy == "bot_answer_self_for_pilot" and not discount_text_has_condition(text):
                findings.append(
                    Finding(
                        "P1",
                        "discount_without_conditions",
                        "Скидка в pilot-маршруте разрешена клиенту без понятного условия применения прямо в тексте.",
                        item_id=fact_id,
                        evidence=text[:220],
                    )
                )
            elif fact_type == "discount" and not discount_has_condition(text, fact_key, structured):
                findings.append(
                    Finding(
                        "P1",
                        "discount_without_conditions",
                        "Скидка разрешена клиенту без понятного условия применения.",
                        item_id=fact_id,
                        evidence=text[:220],
                    )
                )

        if allowed and "theme_12_certificate" in fact_key:
            for marker in ("ФИО плательщика", "ФИО ребёнка", "за какой период"):
                if marker.casefold() in text.casefold():
                    findings.append(
                        Finding(
                            "P0",
                            "certificate_unconfirmed_field_request",
                            "Тема справок просит поле, которое не подтверждено текущей политикой.",
                            item_id=fact_id,
                            evidence=marker,
                        )
                    )

        if "online_olympiad_phystech_9_and_11" in fact_key:
            product = str(fact.get("product") or "")
            if product != "online_olympiad_phystech_classes_9_11":
                findings.append(
                    Finding(
                        "P1",
                        "phystech_product_collapsed",
                        "Онлайн Физтех 9/11 схлопнулся с общей олимпиадной подготовкой.",
                        item_id=fact_id,
                        evidence=product,
                    )
                )

    return findings


def review_fact_freshness(fact: Mapping[str, Any], *, fact_type: str, item_id: str) -> list[Finding]:
    if fact_type not in TIME_SENSITIVE_FACT_TYPES:
        return []
    structured = as_mapping(fact.get("structured_value"))
    valid_until = str(fact.get("valid_until") or structured.get("valid_until") or "")
    check_date = str(fact.get("freshness_check_date") or structured.get("freshness_check_date") or "")
    if valid_until:
        return []
    if check_date:
        return [
            Finding(
                "P2",
                "time_sensitive_fact_has_check_date_only",
                "У чувствительного к дате факта нет срока действия, но есть дата последней проверки.",
                item_id=item_id,
                evidence=f"{fact_type} checked={check_date}",
            )
        ]
    return [
        Finding(
            "P1",
            "time_sensitive_fact_missing_freshness_marker",
            "У чувствительного к дате факта нет ни срока действия, ни даты проверки.",
            item_id=item_id,
            evidence=fact_type,
        )
    ]


def review_fact_freshness_sla(fact: Mapping[str, Any], *, item_id: str) -> list[Finding]:
    """БЛОК 7 (2026-07-25): real age-vs-SLA check, independent of `valid_until`.

    `review_fact_freshness` above stops as soon as `valid_until` is present,
    which is true for effectively every fact in the release and is why the
    old semantic_pass reports zero freshness findings. `valid_until` is a
    *business* expiry date (e.g. end of a semester); it says nothing about
    whether anyone re-checked the fact against reality recently. This check
    compares `freshness_check_date` age against the owner SLA by fact class
    (schedule/availability 24h, price/dates/conditions 7d, stable rules 90d)
    regardless of `valid_until`.
    """
    result = evaluate_fact_freshness_sla(fact)
    if result.status == FRESHNESS_SLA_STATUS_UNKNOWN:
        return [
            Finding(
                "P1",
                "fact_freshness_sla_check_date_unknown",
                "Клиентский факт не имеет читаемой даты проверки (freshness_check_date/verified_at) для проверки SLA свежести.",
                item_id=item_id,
                evidence=f"sla_class={result.sla_class}",
            )
        ]
    if not result.within_sla:
        return [
            Finding(
                "P2",
                "fact_freshness_sla_breach",
                "Клиентский факт старше SLA свежести своего класса (проверить и переподтвердить у владельца факта).",
                item_id=item_id,
                evidence=(
                    f"sla_class={result.sla_class} max_age_days={result.sla_max_age_days} "
                    f"age_days={result.age_days} checked_at={result.checked_at}"
                ),
            )
        ]
    return []


def discount_has_condition(text: str, fact_key: str, structured: Mapping[str, Any]) -> bool:
    payload = " ".join(
        [
            text,
            fact_key.replace("_", " "),
            json.dumps(dict(structured), ensure_ascii=False),
        ]
    ).casefold().replace("ё", "е")
    return any(
        marker in payload
        for marker in (
            "услов",
            "если",
            "при ",
            "для ",
            "после",
            "до ",
            "действ",
            "многодет",
            "сотрудник",
            "ранн",
            "друг",
            "оплат",
            "смен",
            "предмет",
            "кэшбэк",
            "кешбэк",
            "рекоменд",
            "помесяч",
            "мфти",
            "суммир",
            "не сумм",
            "лагер",
            "лвш",
            "городской",
            "очно",
            "онлайн",
            "класс",
        )
    )


def discount_text_has_condition(text: str) -> bool:
    lowered = text.casefold().replace("ё", "е")
    return any(
        marker in lowered
        for marker in (
            "если",
            "при ",
            "для ",
            "после",
            "услов",
            "многодет",
            "сотрудник",
            "второй предмет",
            "помесяч",
            "оплатив",
            "действующ",
            "применяется",
            "с учетом",
            "с учётом",
            "постоянн",
            "участник",
            "друга",
            "другу",
            "семей",
        )
    )


def has_machine_short_tail(text: str) -> bool:
    stripped = " ".join(text.split())
    match = re.search(r"—\s*([0-9]+(?:[.,][0-9]+)?%?|да|нет)\.$", stripped, re.IGNORECASE)
    return bool(match)


def review_client_text(text: str, *, brand: str, item_id: str) -> list[Finding]:
    findings: list[Finding] = []
    lowered = text.casefold().replace("ё", "е")

    for marker in GLOBAL_FORBIDDEN_CLIENT_MARKERS:
        if marker.casefold().replace("ё", "е") in lowered:
            findings.append(
                Finding(
                    "P0",
                    "forbidden_client_marker",
                    "Клиентский текст содержит запрещенный служебный или брендовый маркер.",
                    item_id=item_id,
                    evidence=marker,
                )
            )

    for marker in STALE_CERTIFICATE_MARKERS:
        if marker.casefold().replace("ё", "е") in lowered:
            findings.append(
                Finding(
                    "P0",
                    "stale_certificate_phrase",
                    "Клиентский текст содержит старую или запрещенную формулировку по справкам.",
                    item_id=item_id,
                    evidence=marker,
                )
            )

    text_without_handles = re.sub(r"@[A-Za-z0-9_]+", "", text)
    text_without_handles = re.sub(
        r"\b(?:https?://|www\.|[a-z0-9-]+\.(?:ru|com|org|net))/[A-Za-z0-9_./-]+",
        "",
        text_without_handles,
        flags=re.I,
    )
    if re.search(r"\b[a-z]+_[a-z0-9_]+\b", text_without_handles) or " / " in text:
        findings.append(
            Finding(
                "P1",
                "machine_text_in_client_fact",
                "Клиентский текст выглядит как технический артефакт.",
                item_id=item_id,
                evidence=text[:220],
            )
        )
    if TECHNICAL_ENGLISH_CLIENT_RE.search(text):
        findings.append(
            Finding(
                "P1",
                "technical_english_in_client_fact",
                "Клиентский текст содержит английский технический фрагмент из ключа данных.",
                item_id=item_id,
                evidence=text[:220],
            )
        )

    if brand == "foton":
        for marker in FOTON_FORBIDDEN_CLIENT_MARKERS:
            if marker.casefold().replace("ё", "е") in lowered:
                findings.append(
                    Finding(
                        "P0",
                        "cross_brand_foton_client_text",
                        "Клиентский текст Фотона содержит маркер УНПК.",
                        item_id=item_id,
                        evidence=marker,
                    )
                )
    if brand == "unpk":
        for marker in UNPK_FORBIDDEN_CLIENT_MARKERS:
            if marker.casefold().replace("ё", "е") in lowered:
                findings.append(
                    Finding(
                        "P0",
                        "cross_brand_unpk_client_text",
                        "Клиентский текст УНПК содержит маркер Фотона или условия Фотона.",
                        item_id=item_id,
                        evidence=marker,
                    )
                )

    return findings


def review_approval_queue(rows: Sequence[Mapping[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    questions = [str(row.get("rop_question") or "") for row in rows]
    unique_questions = len(set(questions))

    for index, row in enumerate(rows, start=2):
        item_id = str(row.get("approval_item_id") or f"row:{index}")
        priority = str(row.get("priority") or "")
        decision = str(row.get("suggested_decision") or "")
        question = str(row.get("rop_question") or "")
        if priority == "P0" and decision == "keep_internal_only":
            findings.append(
                Finding(
                    "P0",
                    "p0_keep_internal_only",
                    "Внутренний факт не должен быть P0 для РОПа.",
                    item_id=item_id,
                    evidence=question[:220],
                )
            )
        if question in GENERIC_ROP_QUESTIONS:
            findings.append(
                Finding(
                    "P1",
                    "generic_rop_question",
                    "Вопрос РОПу слишком общий и не проверяет конкретный факт.",
                    item_id=item_id,
                    evidence=question,
                )
            )
        if decision == "keep_internal_only" and not keep_internal_question_matches(question):
            findings.append(
                Finding(
                    "P1",
                    "rop_question_mismatch_keep_internal",
                    "Решение keep_internal_only не отражено в вопросе РОПу.",
                    item_id=item_id,
                    evidence=question[:220],
                )
            )

    if len(rows) >= 50 and unique_questions < min(100, max(20, len(rows) // 4)):
        findings.append(
            Finding(
                "P1",
                "low_rop_question_variety",
                "В очереди РОПа слишком мало уникальных вопросов, это похоже на шаблонный дамп.",
                evidence=f"unique={unique_questions}, total={len(rows)}",
            )
        )

    return findings


def keep_internal_question_matches(question: str) -> bool:
    lowered = question.casefold().replace("ё", "е")
    return any(
        marker in lowered
        for marker in (
            "внутрен",
            "только для менеджера",
            "оставляем только",
            "оставить только",
            "не говорит его клиенту",
            "не показывать клиенту",
        )
    )


def review_snapshot(snapshot: Mapping[str, Any], *, facts: Sequence[Mapping[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    q15_products = {
        str(fact.get("product") or "")
        for fact in facts
        if "online_olympiad_phystech_9_and_11" in str(fact.get("fact_key") or "")
    }
    general_products = {
        str(fact.get("product") or "")
        for fact in facts
        if str(fact.get("fact_key") or "").startswith("fiztech_olympiad.")
    }
    if q15_products and q15_products != {"online_olympiad_phystech_classes_9_11"}:
        findings.append(
            Finding(
                "P1",
                "q15_product_scope_wrong",
                "Q15 онлайн Физтех 9/11 имеет неверный product scope.",
                evidence=", ".join(sorted(q15_products)),
            )
        )
    if general_products and general_products != {"fiztech_olympiad_general"}:
        findings.append(
            Finding(
                "P1",
                "general_phystech_product_scope_wrong",
                "Общая олимпиадная подготовка Физтех имеет неверный product scope.",
                evidence=", ".join(sorted(general_products)),
            )
        )
    if (snapshot.get("quality_summary") or {}).get("quality_passed") is not True:
        findings.append(
            Finding(
                "P1",
                "formal_quality_not_passed",
                "Формальная quality_summary не пройдена; semantic review не может заменить обычные проверки.",
            )
        )
    return findings


def review_fact_mirror_consistency(
    snapshot: Mapping[str, Any], *, facts: Sequence[Mapping[str, Any]]
) -> list[Finding]:
    snapshot_facts = [item for item in snapshot.get("facts", []) if isinstance(item, Mapping)]
    canonical = lambda rows: json.dumps(  # noqa: E731 - tiny local normalizer
        sorted((dict(item) for item in rows), key=lambda item: str(item.get("fact_id") or "")),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if canonical(snapshot_facts) == canonical(facts):
        return []
    return [
        Finding(
            "P0",
            "snapshot_registry_mismatch",
            "Снимок и реестр фактов расходятся; нельзя доказать, какой набор использует бот.",
            evidence=f"snapshot={len(snapshot_facts)} registry={len(facts)}",
        )
    ]


def load_snapshot(release_root: Path) -> Mapping[str, Any]:
    path = release_root / "kb_release_v3_snapshot.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping snapshot at {path}")
    return payload


def load_facts(release_root: Path, snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = [release_root / "facts_registry.jsonl"]
    for path in candidates:
        if path.exists():
            return [
                item
                for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                if isinstance(item, Mapping)
            ]
    for key in ("facts", "facts_registry"):
        value = snapshot.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    raise FileNotFoundError(f"No facts registry found under {release_root}")


def load_approval_queue(release_root: Path) -> list[Mapping[str, Any]]:
    path = release_root / "approval_queue_for_rop_v3.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def render_markdown(report: Mapping[str, Any]) -> str:
    findings = [item for item in report.get("findings", []) if isinstance(item, Mapping)]
    lines = [
        "# KB Semantic Review",
        "",
        f"- created_at: `{report.get('created_at')}`",
        f"- release_dir: `{report.get('release_dir')}`",
        f"- formal_quality_passed: `{report.get('formal_quality_passed')}`",
        f"- semantic_pass: `{report.get('semantic_pass')}`",
        f"- facts_total: `{report.get('facts_total')}`",
        f"- approval_queue_items: `{report.get('approval_queue_items')}`",
        f"- findings_total: `{report.get('findings_total')}`",
        f"- blocking_findings: `{report.get('blocking_findings')}`",
        f"- findings_by_severity: `{report.get('findings_by_severity')}`",
        "",
    ]
    if not findings:
        lines.extend(["## Findings", "", "Нет блокеров или предупреждений по текущим смысловым правилам."])
        return "\n".join(lines) + "\n"

    lines.extend(["## Findings", ""])
    for item in findings:
        lines.append(
            f"- `{item.get('severity')}` `{item.get('check_id')}` "
            f"{item.get('message')} item=`{item.get('item_id')}` evidence=`{item.get('evidence')}`"
        )
    return "\n".join(lines) + "\n"


def guard_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("Semantic review output must not be inside stable_runtime")
    return resolved


def as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, Mapping) else {}
    return {}


def is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "y", "да"}
    return False


if __name__ == "__main__":
    raise SystemExit(main())
