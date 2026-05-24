from __future__ import annotations

from mango_mvp.channels.fact_scope_spec import detect_fact_scopes


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
