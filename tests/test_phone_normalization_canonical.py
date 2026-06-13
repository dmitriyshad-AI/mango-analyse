from __future__ import annotations

import pytest

from mango_mvp.channels.telegram_history import normalize_phone as telegram_normalize_phone
from mango_mvp.customer_timeline.context_provider import normalize_phone_for_match
from mango_mvp.insights.phone_identity import normalize_phone as insight_normalize_phone
from mango_mvp.productization.mail_archive import normalize_phone as mail_normalize_phone
from mango_mvp.utils.phone import normalize_phone


PHONE_CASES = [
    ("+7 (916) 149-24-92", "+79161492492"),
    ("8 916 149 24 92", "+79161492492"),
    ("9161492492", "+79161492492"),
    ("7.9161492492e10", "+79161492492"),
    ("+7 903 438-16-08", "+79034381608"),
    ("8-903-438-16-08", "+79034381608"),
    ("9034381608", "+79034381608"),
    ("+7 (999) 000-00-00", "+79990000000"),
    ("8 (999) 000-00-00", "+79990000000"),
    ("9990000000", "+79990000000"),
    ("+7 909 200 99 33", "+79092009933"),
    ("8 (909) 200-99-33", "+79092009933"),
    ("9092009933", "+79092009933"),
    ("+1 212 555 0100", "+12125550100"),
    ("12125550100", "+12125550100"),
    ("+44 20 7123 4567", "+442071234567"),
    ("442071234567", "+442071234567"),
    ("+49 30 123456", "+4930123456"),
    ("4930123456", "+74930123456"),
    ("+7 916 149 24 92 доб. 123", "+79161492492123"),
    ("79161492492", "+79161492492"),
    ("779161492492", "+79161492492"),
    ("  +7 916 149 24 92  ", "+79161492492"),
    ("nan", None),
    ("none", None),
    ("null", None),
    ("", None),
    (None, None),
    ("not a phone", None),
    ("123", None),
]


@pytest.mark.parametrize(("raw", "expected"), PHONE_CASES)
def test_canonical_phone_normalizer_characterization(raw: object, expected: str | None) -> None:
    assert normalize_phone(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), PHONE_CASES)
def test_phone_normalizer_wrappers_keep_existing_output_contracts(raw: object, expected: str | None) -> None:
    expected_digits = expected.lstrip("+") if expected else None

    assert insight_normalize_phone(raw) == expected_digits
    assert telegram_normalize_phone(raw) == expected
    assert normalize_phone_for_match(raw) == (expected or "")
    assert mail_normalize_phone(raw) == (expected or "")
