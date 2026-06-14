from __future__ import annotations

from dataclasses import replace

from mango_mvp.amocrm_runtime import tallanto_context as tallanto_context_module
from mango_mvp.amocrm_runtime.tallanto_context import (
    brand_scope_from_filial,
    build_live_tallanto_context,
    build_tallanto_live_card,
)


def test_live_card_builds_balance_schedule_and_enrollment_without_teacher_name() -> None:
    card = build_tallanto_live_card(
        [
            {
                "contact": {"id": "c1", "filial": "mfti"},
                "finances": [{"date_entered": "2026-06-12", "payment_summa": "10000", "payment_status": "paid"}],
                "abonements": [{"status": "active", "num_visit_left": "4", "date_finish": "2026-09-01"}],
                "classes": [
                    {
                        "id": "cl1",
                        "name": "Физика 8",
                        "status": "active",
                        "date_start": "2026-09-15",
                        "time_start": "18:00",
                        "time_finish": "19:30",
                        "auditory": "301",
                        "remaining_seats": "3",
                        "teacher_name": "Не должен попасть",
                    }
                ],
            }
        ],
        active_brand="unpk",
        matched_via="tallanto_id",
    )

    assert card["status"] == "ok"
    assert card["brand"] == "unpk"
    assert card["payments"][0]["status"] == "paid"
    assert card["balance"][0]["visits_left"] == "4"
    assert card["schedule"][0]["title"] == "Физика 8"
    assert card["enrollment"][0]["remaining_seats"] == "3"
    assert "teacher" not in str(card).casefold()
    assert "Не должен попасть" not in str(card)


def test_live_tallanto_context_mock_mode_is_disabled_before_brand_checks(monkeypatch) -> None:
    monkeypatch.setenv("CRM_LIVE_CARD_BRAND_FAILCLOSED", "1")
    monkeypatch.setattr(
        tallanto_context_module,
        "settings",
        replace(tallanto_context_module.settings, crm_tallanto_mode="mock"),
    )
    monkeypatch.setattr(
        tallanto_context_module,
        "TallantoApiClient",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Tallanto client must not be created")),
    )

    context = build_live_tallanto_context(phone="+79990001122", active_brand=None)

    assert context["enabled"] is False
    assert context["status"] == "disabled"
    assert context["reason"] == "crm_tallanto_mode=mock"


def test_live_card_multi_match_returns_no_card() -> None:
    card = build_tallanto_live_card([{"contact": {"id": "1"}}, {"contact": {"id": "2"}}])

    assert card["status"] == "no_card"
    assert card["reason"] == "multiple_contacts"


def test_live_card_shd_is_skipped_not_foton() -> None:
    card = build_tallanto_live_card([{"contact": {"filial": "shd"}, "classes": []}], active_brand="foton")

    assert card["status"] == "no_card"
    assert card["reason"] == "filial_shd"
    assert card["skipped"]["filial_shd"] == 1


def test_live_card_filters_inactive_classes_and_10000_seats() -> None:
    card = build_tallanto_live_card(
        [
            {
                "contact": {"filial": "onlajn"},
                "classes": [
                    {"id": "inactive", "name": "Старое", "status": "notactive", "remaining_seats": "2"},
                    {"id": "active", "name": "Онлайн", "status": "active", "remaining_seats": "10000"},
                ],
            }
        ],
        active_brand="unpk",
    )

    assert card["status"] == "ok"
    assert card["skipped"]["inactive_class"] == 1
    assert len(card["schedule"]) == 1
    assert card["schedule"][0]["class_id"] == "active"
    assert "remaining_seats" not in card["enrollment"][0]


def test_live_card_brand_mismatch_blocks_card() -> None:
    card = build_tallanto_live_card([{"contact": {"filial": "mfti"}, "classes": []}], active_brand="foton")

    assert card["status"] == "no_card"
    assert card["reason"] == "brand_mismatch"


def test_live_card_fail_closed_blocks_unverified_brand(monkeypatch) -> None:
    monkeypatch.delenv("CRM_LIVE_CARD_BRAND_FAILCLOSED", raising=False)

    card = build_tallanto_live_card([{"contact": {"filial": "mfti"}, "classes": []}], active_brand=None)

    assert card["status"] == "no_card"
    assert card["reason"] == "brand_unverified"
    assert card["brand_scope"] == "unpk"


def test_live_card_fail_closed_blocks_shared_without_brand(monkeypatch) -> None:
    monkeypatch.delenv("CRM_LIVE_CARD_BRAND_FAILCLOSED", raising=False)

    card = build_tallanto_live_card([{"contact": {"filial": "onlajn"}, "classes": []}], active_brand="")

    assert card["status"] == "no_card"
    assert card["reason"] == "brand_unverified"
    assert card["brand_scope"] == "shared"


def test_live_card_fail_closed_explicit_off_keeps_old_unverified_behavior(monkeypatch) -> None:
    monkeypatch.setenv("CRM_LIVE_CARD_BRAND_FAILCLOSED", "0")

    card = build_tallanto_live_card([{"contact": {"filial": "mfti"}, "classes": []}], active_brand=None)

    assert card["status"] == "ok"
    assert card["brand"] == "unpk"


def test_live_card_skip_shd_takes_priority_over_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("CRM_LIVE_CARD_BRAND_FAILCLOSED", "1")

    card = build_tallanto_live_card([{"contact": {"filial": "shd"}, "classes": []}], active_brand=None)

    assert card["status"] == "no_card"
    assert card["reason"] == "filial_shd"


def test_brand_scope_mapping() -> None:
    assert brand_scope_from_filial("МФТИ") == "unpk"
    assert brand_scope_from_filial("Фотон") == "foton"
    assert brand_scope_from_filial("foton online") == "foton"
    assert brand_scope_from_filial("Онлайн") == "shared"
    assert brand_scope_from_filial("ШД") == "skip_shd"
    assert brand_scope_from_filial("Красносельская") == "unknown"
    assert brand_scope_from_filial("Менделеево") == "unknown"
    assert brand_scope_from_filial("Сретенка") == "unknown"
    assert brand_scope_from_filial("") == "unknown"


def test_live_card_foton_filial_matches_active_foton_brand(monkeypatch) -> None:
    monkeypatch.delenv("CRM_LIVE_CARD_BRAND_FAILCLOSED", raising=False)

    card = build_tallanto_live_card([{"contact": {"filial": "Фотон"}, "classes": []}], active_brand="foton")

    assert card["status"] == "ok"
    assert card["brand"] == "foton"
    assert card["brand_scope"] == "foton"
