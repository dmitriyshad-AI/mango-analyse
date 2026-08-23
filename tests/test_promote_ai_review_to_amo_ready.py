from scripts.promote_ai_review_to_amo_ready import promote_contacts


def test_promotion_keeps_commercial_review_blocked() -> None:
    commercial = {
        "Нужна ручная проверка": "Да",
        "Коммерческая проверка": "Да",
        "Готово к записи в AMO": "Нет",
        "Причина статуса AMO": "",
    }
    ordinary_ai_review = {
        "Нужна ручная проверка": "Да",
        "Коммерческая проверка": "Нет",
        "Готово к записи в AMO": "Нет",
        "Причина статуса AMO": "",
    }
    bypassed_commercial = {
        "Нужна ручная проверка": "Нет",
        "Коммерческая проверка": "Да",
        "Готово к записи в AMO": "Да",
        "Причина статуса AMO": "устаревший статус",
    }
    reviewed_commercial = {
        "Нужна ручная проверка": "Да",
        "Коммерческая проверка": "Да",
        "Коммерческая проверка подтверждена": "Да",
        "Готово к записи в AMO": "Нет",
        "Причина статуса AMO": "",
    }

    promoted = promote_contacts(
        [commercial, ordinary_ai_review, bypassed_commercial, reviewed_commercial]
    )

    assert promoted == 2
    assert commercial["Нужна ручная проверка"] == "Да"
    assert commercial["Готово к записи в AMO"] == "Нет"
    assert commercial["Причина статуса AMO"] == "требуется коммерческая проверка РОПа"
    assert ordinary_ai_review["Нужна ручная проверка"] == "Нет"
    assert ordinary_ai_review["Готово к записи в AMO"] == "Да"
    assert bypassed_commercial["Нужна ручная проверка"] == "Да"
    assert bypassed_commercial["Готово к записи в AMO"] == "Нет"
    assert reviewed_commercial["Нужна ручная проверка"] == "Нет"
    assert reviewed_commercial["Готово к записи в AMO"] == "Да"
    assert "подтверждения коммерческой проверки" in reviewed_commercial["Причина статуса AMO"]
