from __future__ import annotations

from collections import Counter

from mango_mvp.insights.phone_identity import client_key_for_phone, normalize_phone, phones_from_text
from mango_mvp.insights.readiness import CallCandidate, _sample_stratum, _touch_bucket, _utility_score


def test_normalize_phone_russian_variants() -> None:
    assert normalize_phone("+7 (916) 149-24-92") == "79161492492"
    assert normalize_phone("8 916 149 24 92") == "79161492492"
    assert normalize_phone("9161492492") == "79161492492"
    assert normalize_phone("7.9161492492e10") == "79161492492"
    assert normalize_phone("not a phone") is None


def test_phones_from_text_preserves_unique_order() -> None:
    assert phones_from_text("a +7 916 149-24-92 b 89161492492 c +7 903 438-16-08") == [
        "79161492492",
        "79034381608",
    ]
    assert client_key_for_phone("79161492492") == "phone:79161492492"


def _call(call_type: str, *, year: str = "2025", next_step: str = "") -> CallCandidate:
    return CallCandidate(
        source_filename=f"{year}-01-01__10-00-00__Manager__79161492492.mp3",
        source_db="db",
        source_db_id=1,
        source_file="file",
        started_at=None,
        month=f"{year}-01",
        year=year,
        phone_key="79161492492",
        manager_name="Manager",
        duration_sec=60.0,
        transcription_status="done",
        resolve_status="done",
        analysis_status="done",
        call_type=call_type,
        lead_priority="warm",
        follow_up_score=70,
        needs_review=False,
        products=[],
        subjects=[],
        formats=[],
        exam_targets=[],
        objections=[],
        next_step=next_step,
        history_summary="summary",
        transcript_chars=100,
        analysis_chars=100,
        amocrm_contact_id="",
        amocrm_lead_id="",
    )


def test_touch_bucket() -> None:
    assert _touch_bucket(1) == "1"
    assert _touch_bucket(3) == "2-3"
    assert _touch_bucket(7) == "4-7"
    assert _touch_bucket(15) == "8-15"
    assert _touch_bucket(16) == "16+"


def test_sample_stratum_prioritizes_tallanto_and_sales() -> None:
    items = [_call("sales_call", year="2026")]
    assert _sample_stratum(items, object(), None, Counter({"sales_call": 1})) == "tallanto_matched_contentful"
    assert _sample_stratum(items, None, object(), Counter({"sales_call": 1})) == "amo_linked_2026"
    assert _sample_stratum(items, None, None, Counter({"sales_call": 1})) == "sales_call"


def test_utility_penalizes_non_conversation_only() -> None:
    non_conversation = [_call("non_conversation")]
    sales = [_call("sales_call", next_step="Перезвонить")]
    assert _utility_score(sales, None, None, Counter({"sales_call": 1}), Counter({"sales_call": 1}), {"M"}) > _utility_score(
        non_conversation,
        None,
        None,
        Counter({"non_conversation": 1}),
        Counter(),
        {"M"},
    )
