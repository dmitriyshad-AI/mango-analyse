from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.run_kb_semantic_review import run_kb_semantic_review


# Real KB facts always carry `freshness_check_date` (100% of the current
# release, see БЛОК 7 audit). Fixtures below that predate the БЛОК 7 SLA
# check used minimal facts without it; `_FRESH_CHECK_DATE` keeps them
# representative of real facts without hard-coding a date that ages out.
_FRESH_CHECK_DATE = date.today().isoformat()


def test_semantic_review_blocks_implausible_client_price(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:bad-price",
                "fact_key": "academic_year.total_lessons",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: занятий за год — 35 ₽.",
                "structured_value": {"amount": 35, "currency": "RUB", "path": "academic_year.total_lessons"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "implausible_low_client_price" for item in report["findings"])
    assert any(item["check_id"] == "non_money_path_became_price" for item in report["findings"])


def test_semantic_review_blocks_cross_brand_client_text(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:cross-brand",
                "fact_key": "contacts.telegram",
                "fact_type": "contact",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: пишите в @unpk_mipt.",
                "structured_value": {"path": "contacts.telegram"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "cross_brand_foton_client_text" for item in report["findings"])


def test_semantic_review_blocks_technical_english_client_text(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:machine-english",
                "fact_key": "city.prices.base",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: prices , base — 34 300 ₽.",
                "structured_value": {"amount": 34300, "currency": "RUB", "path": "city.prices.base"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "technical_english_in_client_fact" for item in report["findings"])


def test_semantic_review_blocks_bad_rop_queue_priority(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:internal",
                "fact_key": "internal.price",
                "fact_type": "price",
                "brand": "internal",
                "allowed_for_client_answer": False,
                "client_safe_text": "",
                "structured_value": {"amount": 50000, "currency": "RUB", "path": "internal.price"},
            }
        ],
        approval_rows=[
            {
                "priority": "P0",
                "approval_item_id": "approve:internal",
                "suggested_decision": "keep_internal_only",
                "rop_question": "Можно ли использовать этот факт в ответе клиенту текущего бренда?",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "p0_keep_internal_only" for item in report["findings"])


def test_semantic_review_accepts_manager_only_wording_for_internal_items(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:internal",
                "fact_key": "internal.note",
                "fact_type": "policy",
                "brand": "internal",
                "allowed_for_client_answer": False,
                "client_safe_text": "",
                "structured_value": {"path": "internal.note"},
            }
        ],
        approval_rows=[
            {
                "priority": "P2",
                "approval_item_id": "approve:internal",
                "suggested_decision": "keep_internal_only",
                "rop_question": "Оставляем только для менеджера: подтвердите, что клиентская версия не нужна.",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert report["findings"] == []


def test_semantic_review_passes_minimal_good_release(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:good-price",
                "fact_key": "prices.offline_5_11.year",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: цена за год для 5-11 класса — 74 500 ₽.",
                "structured_value": {"amount": 74500, "currency": "RUB", "path": "prices.offline_5_11.year"},
                "valid_until": "2026-07-01",
                "freshness_check_date": _FRESH_CHECK_DATE,
            },
            {
                "fact_id": "fact:good-lessons",
                "fact_key": "academic_year.total_lessons",
                "fact_type": "course_parameter",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: за год проходит 35 занятий.",
                "structured_value": {"count": 35, "unit": "lessons", "path": "academic_year.total_lessons"},
                "freshness_check_date": _FRESH_CHECK_DATE,
            },
        ],
        approval_rows=[
            {
                "priority": "P0",
                "approval_item_id": "approve:price",
                "suggested_decision": "approve_for_client_answer_after_rop_review",
                "rop_question": "Подтверждаете цену 74 500 ₽ для Фотона, 5-11 класс, год?",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert report["findings"] == []


def test_semantic_review_blocks_client_promocode(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:promo",
                "fact_key": "promo_codes.teacher_promo.code",
                "fact_type": "promocode",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: промокод TEACHER500 даёт скидку.",
                "structured_value": {"path": "promo_codes.teacher_promo.code"},
                "freshness_check_date": "2026-05-18",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "promo_code_allowed_for_client" for item in report["findings"])


def test_semantic_review_blocks_allowed_fact_that_requires_confirmation(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:contradiction",
                "fact_key": "prices.current.year",
                "fact_type": "price",
                "brand": "foton",
                "freshness_status": "document_verified",
                "allowed_for_client_answer": True,
                "requires_manager_confirmation": True,
                "client_safe_text": "Фотон: год стоит 74 500 ₽.",
                "structured_value": {"amount": 74500, "currency": "RUB", "path": "prices.current.year"},
                "valid_until": "2026-07-01",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "allowed_fact_requires_manager_confirmation" for item in report["findings"])
    assert any(item["check_id"] == "verified_fact_marked_requires_manager_confirmation" for item in report["findings"])


def test_semantic_review_blocks_empty_allowed_client_text(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:empty",
                "fact_key": "contacts.telegram",
                "fact_type": "contact",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "",
                "structured_value": {"path": "contacts.telegram"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "allowed_fact_has_empty_client_text" for item in report["findings"])


def test_semantic_review_blocks_discount_without_conditions(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:discount",
                "fact_key": "discounts.generic.percent",
                "fact_type": "discount",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: скидка — 10%.",
                "structured_value": {"percentage": 10, "unit": "percent", "path": "discounts.generic.percent"},
                "valid_until": "2026-07-01",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "discount_without_conditions" for item in report["findings"])


def test_semantic_review_allows_discount_with_conditions(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:discount-ok",
                "fact_key": "discounts.second_subject.percent",
                "fact_type": "discount",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: скидка 10% действует для второго предмета.",
                "structured_value": {"percentage": 10, "unit": "percent", "path": "discounts.second_subject.percent"},
                "valid_until": "2026-07-01",
                "freshness_check_date": _FRESH_CHECK_DATE,
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert report["findings"] == []


def test_semantic_review_blocks_discount_stacking_contradiction(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:bad-stacking",
                "fact_key": "objection_responses.too_expensive_camp.2",
                "fact_type": "program",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "route_policy": "draft_for_manager",
                "client_safe_text": "Фотон: черновик для ситуации «возражение о стоимости лагеря»: Скидки суммируются.",
                "structured_value": {"path": "objection_responses.too_expensive_camp.2"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "discount_stacking_contradiction" for item in report["findings"])


def test_semantic_review_blocks_pilot_short_machine_tail(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:matkap-short",
                "fact_key": "matkap.child_age.sertificate_owner_min",
                "fact_type": "matkap",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "route_policy": "bot_answer_self_for_pilot",
                "client_safe_text": "Фотон: материнский капитал — 3.",
                "structured_value": {"number": 3, "path": "matkap.child_age.sertificate_owner_min"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is False
    assert any(item["check_id"] == "pilot_client_text_has_machine_short_tail" for item in report["findings"])


def test_semantic_review_allows_pilot_short_tail_when_template_required(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:lvsh-deposit",
                "fact_key": "lvsh_mendeleevo_2026.pricing_2026.deposit",
                "fact_type": "camp_lvsh",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "route_policy": "bot_answer_self_for_pilot",
                "bot_template_required": True,
                "client_safe_text": "Фотон: ЛВШ Менделеево — 15 000 ₽.",
                "valid_until": "2026-08-31",
                "structured_value": {"amount": 15000, "currency": "RUB", "path": "lvsh_mendeleevo_2026.pricing_2026.deposit"},
                "freshness_check_date": _FRESH_CHECK_DATE,
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert report["findings"] == []


def test_semantic_review_allows_public_telegram_handle_with_underscore(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:unpk-telegram",
                "fact_key": "contacts_unpk.telegram",
                "fact_type": "contact",
                "brand": "unpk",
                "allowed_for_client_answer": True,
                "route_policy": "draft_for_manager",
                "client_safe_text": "УНПК: контакты — @unpk_mipt.",
                "structured_value": {"raw_value": "@unpk_mipt", "path": "contacts_unpk.telegram"},
                "freshness_check_date": _FRESH_CHECK_DATE,
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert report["findings"] == []


def test_semantic_review_allows_contextual_pilot_number(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:matkap-contextual",
                "fact_key": "matkap.child_age.sertificate_owner_min",
                "fact_type": "matkap",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "route_policy": "bot_answer_self_for_pilot",
                "client_safe_text": "Фотон: материнский капитал можно использовать, если ребёнку, на которого оформлен сертификат, исполнилось 3 года.",
                "structured_value": {"number": 3, "path": "matkap.child_age.sertificate_owner_min"},
                "freshness_check_date": _FRESH_CHECK_DATE,
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert report["findings"] == []


def test_semantic_review_warns_on_time_sensitive_fact_with_check_date_only(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:checked-price",
                "fact_key": "prices.current.year",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: год стоит 74 500 ₽.",
                "structured_value": {"amount": 74500, "currency": "RUB", "path": "prices.current.year"},
                "freshness_check_date": "2026-05-18",
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert report["semantic_pass"] is True
    assert any(item["check_id"] == "time_sensitive_fact_has_check_date_only" for item in report["findings"])


def test_semantic_review_flags_stale_price_fact_by_sla_even_with_future_valid_until(tmp_path: Path) -> None:
    """БЛОК 7: `valid_until` (business expiry) must not hide a stale verification date.

    Before БЛОК 7, `review_fact_freshness` returned no finding as soon as
    `valid_until` was present, which is true of ~every fact in the real
    release and is why the old semantic_pass reported zero freshness
    findings. This proves the new SLA check catches it independently.
    """
    stale_check_date = (date.today() - timedelta(days=30)).isoformat()
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:stale-price",
                "fact_key": "prices_regular_2026_27.foton.online.year",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "client_safe_text": "Фотон: онлайн, год — 47 250 ₽.",
                "valid_until": "2027-08-31",
                "freshness_check_date": stale_check_date,
                "structured_value": {"amount": 47250, "currency": "RUB", "path": "prices_regular_2026_27.foton.online.year"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    breach = [item for item in report["findings"] if item["check_id"] == "fact_freshness_sla_breach"]
    assert breach
    assert breach[0]["item_id"] == "fact:stale-price"
    assert "sla_class=commercial_terms" in breach[0]["evidence"]
    # Advisory (P2): a stale-but-datestamped fact does not flip the formal gate.
    assert report["semantic_pass"] is True


def test_semantic_review_flags_schedule_fact_with_no_freshness_marker_via_sla_check(tmp_path: Path) -> None:
    """`schedule`/`availability` are not in TIME_SENSITIVE_FACT_TYPES, so the
    pre-existing check never looked at them at all. Only the БЛОК 7 SLA check
    catches a schedule fact with no verification date whatsoever.
    """
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:schedule-no-date",
                "fact_key": "unpk.online.weekend.schedule",
                "fact_type": "schedule",
                "brand": "unpk",
                "allowed_for_client_answer": True,
                "client_safe_text": "УНПК: занятия по выходным.",
                "structured_value": {"path": "unpk.online.weekend.schedule"},
            }
        ],
    )

    report = run_kb_semantic_review(release)

    assert not [item for item in report["findings"] if item["check_id"] == "time_sensitive_fact_missing_freshness_marker"]
    unknown = [item for item in report["findings"] if item["check_id"] == "fact_freshness_sla_check_date_unknown"]
    assert unknown
    assert unknown[0]["item_id"] == "fact:schedule-no-date"
    # Unlike a stale-but-datestamped fact, an unreadable check date blocks the gate (P1).
    assert report["semantic_pass"] is False


def _write_release(
    tmp_path: Path,
    *,
    facts: list[dict],
    approval_rows: list[dict] | None = None,
) -> Path:
    release = tmp_path / "release"
    release.mkdir()
    snapshot = {
        "quality_summary": {"quality_passed": True},
        "facts": facts,
        "summary": {"facts_total": len(facts)},
    }
    (release / "kb_release_v3_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    if approval_rows is not None:
        fieldnames = sorted({key for row in approval_rows for key in row})
        with (release / "approval_queue_for_rop_v3.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(approval_rows)
    return release


# --- окно применимости цены и завершившийся период (аудит 2026-07-27) ---------
#
# Оба дефекта существуют в текущем релизе и до этих правил были невидимы:
# `valid_until` у всех ценовых фактов выставлен релизом на 2026-12-31, а окно
# применимости живёт только в `fact_key`; период проведения смены живёт только
# в тексте для клиента. Ревьюер давал semantic_pass=true при 8 просроченных
# ценах и 3 фактах с завершившейся сменой.

_TODAY = date(2026, 7, 27)


def test_semantic_review_blocks_price_whose_window_already_closed(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:expired-window",
                "fact_key": "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.year",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "freshness_status": "document_verified",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2026-12-31",
                "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, очно, год — 74 500 ₽.",
                "structured_value": {"amount": 74500, "currency": "RUB"},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    assert report["semantic_pass"] is False
    finding = next(
        item for item in report["findings"] if item["check_id"] == "expired_price_window_allowed_for_client"
    )
    assert finding["severity"] == "P1"
    assert "window_until=2026-07-01" in finding["evidence"]


def test_semantic_review_allows_price_whose_window_is_still_open(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:open-window",
                "fact_key": "prices_regular_2026_27.online_5_11_class.before_2026_08_01.year",
                "fact_type": "price",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "freshness_status": "document_verified",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2026-12-31",
                "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, год — 47 250 ₽.",
                "structured_value": {"amount": 47250, "currency": "RUB"},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    assert not any(
        item["check_id"] == "expired_price_window_allowed_for_client" for item in report["findings"]
    )


def test_semantic_review_ignores_schedule_deadline_shaped_like_a_price_window(tmp_path: Path) -> None:
    """Расписание использует тот же `before_...` в ключе. 107 таких фактов не
    должны разом стать блокерами, когда учебный год закончится."""
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:schedule-deadline",
                "fact_key": "schedule_2026_27.groups.group_start_date.w4.before_2026_05_30.client_safe_text",
                "fact_type": "deadline",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "freshness_status": "document_verified",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2027-05-30",
                "client_safe_text": "Математика, 7 класс, продвинутая группа, очно: суббота 10:00-12:00.",
                "structured_value": {},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    assert not any(
        item["check_id"] == "expired_price_window_allowed_for_client" for item in report["findings"]
    )


def test_semantic_review_blocks_finished_shift_offered_to_client(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:finished-shift",
                "fact_key": "ls_city_2026.unpk.moscow.dates",
                "fact_type": "deadline",
                "brand": "unpk",
                "allowed_for_client_answer": True,
                "freshness_status": "document_verified",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2026-08-31",
                "client_safe_text": "УНПК: городской летний лагерь, Москва, даты — 6-17 июля; 20-31 июля.",
                "structured_value": {},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    assert report["semantic_pass"] is False
    finding = next(
        item for item in report["findings"] if item["check_id"] == "finished_period_in_client_text"
    )
    assert finding["severity"] == "P1"
    assert "period=6-17 июля" in finding["evidence"]


def test_semantic_review_does_not_read_a_year_as_a_date_range(tmp_path: Path) -> None:
    """«интенсивы 2026 — 15 апреля» не является периодом «26-15 апреля»."""
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:single-date",
                "fact_key": "intensives_2026.ege_unpk.start_dates.math",
                "fact_type": "deadline",
                "brand": "unpk",
                "allowed_for_client_answer": True,
                "freshness_status": "document_verified",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2026-12-31",
                "client_safe_text": "УНПК: интенсивы 2026 — 15 апреля 2026.",
                "structured_value": {},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    assert not any(
        item["check_id"] == "finished_period_in_client_text" for item in report["findings"]
    )


@pytest.mark.parametrize(
    ("year", "blocked"),
    [(2025, True), (2026, True), (2027, False)],
)
def test_semantic_review_respects_explicit_period_year(tmp_path: Path, year: int, blocked: bool) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": f"fact:period-{year}",
                "fact_key": "city_camp.period",
                "fact_type": "deadline",
                "brand": "unpk",
                "allowed_for_client_answer": True,
                "freshness_status": "document_verified",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2027-12-31",
                "client_safe_text": f"Смена 6-17 июля {year} года.",
                "structured_value": {},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    found = any(item["check_id"] == "finished_period_in_client_text" for item in report["findings"])
    assert found is blocked


def test_semantic_review_blocks_snapshot_registry_mismatch(tmp_path: Path) -> None:
    release = _write_release(tmp_path, facts=[])
    (release / "facts_registry.jsonl").write_text(
        json.dumps({"fact_id": "fact:registry-only", "fact_key": "registry.only"}) + "\n",
        encoding="utf-8",
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    assert any(item["check_id"] == "snapshot_registry_mismatch" for item in report["findings"])


def test_semantic_review_allows_explicitly_historical_period(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:historical-shift",
                "fact_key": "history.city_camp",
                "fact_type": "program",
                "brand": "unpk",
                "allowed_for_client_answer": True,
                "freshness_status": "document_verified",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2026-12-31",
                "client_safe_text": "Смена 6-17 июля уже завершилась; подберём ближайшую актуальную.",
                "structured_value": {},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    assert not any(item["check_id"] == "finished_period_in_client_text" for item in report["findings"])


def test_semantic_review_ignores_disallowed_past_period(tmp_path: Path) -> None:
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:internal-past-shift",
                "fact_key": "internal.city_camp",
                "fact_type": "program",
                "brand": "unpk",
                "allowed_for_client_answer": False,
                "freshness_status": "do_not_use",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2026-07-17",
                "client_safe_text": "Смена 6-17 июля.",
                "structured_value": {},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    assert not any(item["check_id"] == "finished_period_in_client_text" for item in report["findings"])


def test_semantic_review_binds_verdict_to_reviewed_bytes(tmp_path: Path) -> None:
    """Отчёт без хэша нельзя привязать ни к какому снапшоту: ровно из-за этого
    ревью от 24.07 невозможно было соотнести с текущим файлом."""
    release = _write_release(
        tmp_path,
        facts=[
            {
                "fact_id": "fact:ok",
                "fact_key": "academic_year_2026_27.start",
                "fact_type": "course_parameter",
                "brand": "foton",
                "allowed_for_client_answer": True,
                "freshness_status": "document_verified",
                "freshness_check_date": _TODAY.isoformat(),
                "valid_until": "2027-08-31",
                "client_safe_text": "Фотон: занятия стартуют 12-20 сентября.",
                "structured_value": {},
            }
        ],
    )

    report = run_kb_semantic_review(release, today=_TODAY)

    import hashlib

    expected = hashlib.sha256((release / "kb_release_v3_snapshot.json").read_bytes()).hexdigest()
    assert report["snapshot_sha256"] == expected
    assert report["checked_on"] == _TODAY.isoformat()
    # Реестра рядом нет — поле обязано присутствовать и быть пустым, а не молчать.
    assert report["facts_registry_sha256"] == ""
    assert "expired_price_window_allowed_for_client" not in report["reviewer_check_ids"]
