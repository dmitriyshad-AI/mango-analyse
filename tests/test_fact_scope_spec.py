from __future__ import annotations

from mango_mvp.channels.fact_scope_spec import blocked_neighbors_for, detect_fact_scopes


def test_fact_scope_detection_uses_token_boundaries_for_ambiguous_words() -> None:
    assert "trial_offline" not in detect_fact_scopes("без точной даты повышения")
    assert "offline_recordings" not in detect_fact_scopes("нужно записаться на курс")
    assert "class_schedule" not in detect_fact_scopes("скидка для многодетной семьи")


def test_fact_scope_detection_keeps_near_neighbor_scopes_distinct() -> None:
    assert detect_fact_scopes("скидка на второй предмет онлайн") == {"discount_second_subject"}
    assert detect_fact_scopes("многодетная скидка по удостоверению") == {"discount_multichild"}
    assert detect_fact_scopes("скидки не суммируются, применяется наибольшая") == {"discount_stacking"}
    assert detect_fact_scopes("программа приведи друга и кэшбэк") == {"discount_referral"}
    assert detect_fact_scopes("записи очных занятий можно запросить") == {"offline_recordings"}
    assert detect_fact_scopes("фрагмент занятия для онлайн-формата") == {"trial_online_fragment"}


def test_fact_scope_detection_covers_rc2b_neighbor_classes() -> None:
    assert "payment_methods" in detect_fact_scopes("оплатить по счёту банковским переводом")
    assert "matkap_age_limit" in detect_fact_scopes("маткапитал можно использовать до 25 лет")
    assert "program_subjects" in detect_fact_scopes("предметы: математика и физика")
    assert "refund_policy" in detect_fact_scopes("порядок возврата по заявлению")
    assert "office_hours" in blocked_neighbors_for("refund_policy")
    assert "class_schedule" in blocked_neighbors_for("refund_policy")


def test_fact_scope_detection_covers_contact_hours_wording() -> None:
    assert "office_hours" in detect_fact_scopes("Фотон на связи ежедневно с 10:00 до 18:00")
    assert "office_hours" in detect_fact_scopes("контактный центр работает Пн–Вс")


def test_fact_scope_detection_does_not_treat_no_lodging_as_residential_lvsh() -> None:
    city_scopes = detect_fact_scopes("очная городская школа без проживания")
    assert "city_day_camp" in city_scopes
    assert "residential_lvsh" not in city_scopes

    day_scopes = detect_fact_scopes("дневной формат без ночёвки и без проживания")
    assert "city_day_camp" in day_scopes
    assert "residential_lvsh" not in day_scopes
