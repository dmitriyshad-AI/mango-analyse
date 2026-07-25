from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from mango_mvp.knowledge_base.fact_registry import (
    FRESHNESS_SLA_COMMERCIAL_TERMS,
    FRESHNESS_SLA_LIVE_STATUS,
    FRESHNESS_SLA_STABLE_RULES,
    FRESHNESS_SLA_STATUS_OK,
    FRESHNESS_SLA_STATUS_STALE,
    FRESHNESS_SLA_STATUS_UNKNOWN,
    evaluate_fact_freshness_sla,
    fact_sla_class,
    fact_sla_max_age_days,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json"


def test_fact_sla_class_matches_owner_buckets() -> None:
    # расписание и наличие -- 24 часа
    assert fact_sla_class("schedule") == FRESHNESS_SLA_LIVE_STATUS
    assert fact_sla_class("availability") == FRESHNESS_SLA_LIVE_STATUS
    # цены, даты, условия -- 7 дней
    assert fact_sla_class("price") == FRESHNESS_SLA_COMMERCIAL_TERMS
    assert fact_sla_class("deadline") == FRESHNESS_SLA_COMMERCIAL_TERMS
    assert fact_sla_class("discount") == FRESHNESS_SLA_COMMERCIAL_TERMS
    assert fact_sla_class("matkap") == FRESHNESS_SLA_COMMERCIAL_TERMS
    # стабильные правила -- 90 дней
    assert fact_sla_class("program") == FRESHNESS_SLA_STABLE_RULES
    assert fact_sla_class("camp_lvsh") == FRESHNESS_SLA_STABLE_RULES
    assert fact_sla_class("location") == FRESHNESS_SLA_STABLE_RULES


def test_fact_sla_max_age_days_matches_owner_sla_numbers() -> None:
    assert fact_sla_max_age_days("schedule") == 0  # 24 часа -> должен быть проверен сегодня
    assert fact_sla_max_age_days("price") == 7
    assert fact_sla_max_age_days("program") == 90


def test_unmapped_fact_type_defaults_to_commercial_terms_bucket() -> None:
    # Неизвестный fact_type не должен молча попадать в самый мягкий (90д) SLA.
    assert fact_sla_class("some_new_fact_type_nobody_classified_yet") == FRESHNESS_SLA_COMMERCIAL_TERMS


def test_schedule_fact_checked_yesterday_breaches_24h_sla() -> None:
    fact = {"fact_id": "fact:schedule-1", "fact_type": "schedule", "freshness_check_date": "2026-07-24"}

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.sla_class == FRESHNESS_SLA_LIVE_STATUS
    assert result.sla_max_age_days == 0
    assert result.age_days == 1
    assert result.within_sla is False
    assert result.status == FRESHNESS_SLA_STATUS_STALE


def test_availability_fact_checked_today_is_within_24h_sla() -> None:
    fact = {"fact_id": "fact:avail-1", "fact_type": "availability", "freshness_check_date": "2026-07-25"}

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.within_sla is True
    assert result.status == FRESHNESS_SLA_STATUS_OK


def test_price_fact_exactly_at_seven_day_boundary_is_ok() -> None:
    fact = {"fact_id": "fact:price-1", "fact_type": "price", "freshness_check_date": "2026-07-18"}

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.age_days == 7
    assert result.within_sla is True


def test_price_fact_one_day_past_seven_day_boundary_is_stale() -> None:
    fact = {"fact_id": "fact:price-2", "fact_type": "price", "freshness_check_date": "2026-07-17"}

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.age_days == 8
    assert result.within_sla is False
    assert result.status == FRESHNESS_SLA_STATUS_STALE


def test_stable_rule_within_90_days_is_ok() -> None:
    fact = {"fact_id": "fact:program-1", "fact_type": "program", "freshness_check_date": "2026-04-27"}  # 89 days

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.age_days == 89
    assert result.within_sla is True


def test_stable_rule_past_90_days_is_stale() -> None:
    fact = {"fact_id": "fact:program-2", "fact_type": "program", "freshness_check_date": "2026-04-25"}  # 91 days

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.age_days == 91
    assert result.within_sla is False
    assert result.status == FRESHNESS_SLA_STATUS_STALE


def test_missing_check_date_is_unknown_not_silently_fresh() -> None:
    fact = {"fact_id": "fact:no-date", "fact_type": "price"}

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.status == FRESHNESS_SLA_STATUS_UNKNOWN
    assert result.within_sla is False
    assert result.age_days is None


def test_malformed_check_date_is_unknown_not_silently_fresh() -> None:
    fact = {"fact_id": "fact:bad-date", "fact_type": "price", "freshness_check_date": "not-a-date"}

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.status == FRESHNESS_SLA_STATUS_UNKNOWN
    assert result.within_sla is False


def test_verified_at_used_as_fallback_when_freshness_check_date_missing() -> None:
    fact = {"fact_id": "fact:verified-at", "fact_type": "price", "verified_at": "2026-07-25"}

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.checked_at == "2026-07-25"
    assert result.within_sla is True


def test_fact_types_plural_used_when_fact_type_singular_missing() -> None:
    fact = {"fact_id": "fact:plural", "fact_types": ["camp_lvsh"], "freshness_check_date": "2026-07-25"}

    result = evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25))

    assert result.sla_class == FRESHNESS_SLA_STABLE_RULES


def test_current_kb_release_has_measurable_sla_breaches_as_of_reference_date() -> None:
    """БЛОК 7 requirement 1, empirical proof: the pre-existing semantic_pass
    measurer reports zero freshness findings for the live release because
    every fact has a `valid_until`. `valid_until` is a business-expiry date,
    not a verification-recency signal, so it hides real staleness. This test
    proves the new age-based SLA check finds real breaches in the actual
    release the bot serves, as of the fixed reference date of this audit.
    """
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    facts = [
        fact
        for fact in snapshot.get("facts") or []
        if fact.get("allowed_for_client_answer") is True and str(fact.get("client_safe_text") or "").strip()
    ]
    assert facts

    results = [evaluate_fact_freshness_sla(fact, today=date(2026, 7, 25)) for fact in facts]
    stale = [item for item in results if item.status == FRESHNESS_SLA_STATUS_STALE]

    assert stale
    assert all(item.status != FRESHNESS_SLA_STATUS_UNKNOWN for item in results)  # KB always stamps a check date today
