from scripts.email_pipeline.brand import infer_email_brand


def test_brand_uses_explicit_content_words() -> None:
    assert infer_email_brand("ЦДПО Фотон: вопрос", "").brand == "foton"
    assert infer_email_brand("УНПК МФТИ", "").brand == "unpk"


def test_brand_conflict_returns_none() -> None:
    result = infer_email_brand("Фотон и УНПК", "")
    assert result.brand == "none"
    assert result.brand_source == "none"


def test_kmipt_email_is_not_brand_signal() -> None:
    result = infer_email_brand("Вопрос", "Напишите на edu@kmipt.ru")
    assert result.brand == "none"
    assert result.brand_source == "none"


def test_course_links_are_content_signals() -> None:
    assert infer_email_brand("", "https://cdpofoton.ru/program").brand == "foton"
    assert infer_email_brand("", "https://kmipt.ru/courses/math").brand == "unpk"


def test_dates_are_last_resort_signal() -> None:
    assert infer_email_brand("", "смена 20-28 июня").brand == "foton"
    assert infer_email_brand("", "период 15-25 августа").brand == "unpk"

