from __future__ import annotations

from mango_mvp.amocrm_runtime.tallanto_context import brand_scope_from_filial, build_tallanto_live_card


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


def test_brand_scope_mapping() -> None:
    assert brand_scope_from_filial("МФТИ") == "unpk"
    assert brand_scope_from_filial("Онлайн") == "shared"
    assert brand_scope_from_filial("ШД") == "skip_shd"
    assert brand_scope_from_filial("") == "unknown"
