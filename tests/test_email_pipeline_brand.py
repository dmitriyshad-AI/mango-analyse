from scripts.email_pipeline.brand import infer_email_brand
from scripts.email_pipeline.summary import mask_pii


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


def test_mask_pii_handles_html_phone_fragments() -> None:
    masked = mask_pii("8 (800)&nbsp;123-45-67 hello@example.com Иванов И.И.")
    assert "123-45-67" not in masked
    assert "hello@example.com" not in masked
    assert "Иванов И.И." not in masked
    assert "[phone]" in masked
    assert "[email]" in masked
    assert "[name]" in masked
