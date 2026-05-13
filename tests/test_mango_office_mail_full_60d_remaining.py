from __future__ import annotations

from scripts.mango_office_mail_full_60d_remaining import is_batch_complete


def test_full_mail_batch_complete_allows_live_plan_growth_under_limit() -> None:
    report = {
        "errors": [],
        "messages_found_since": 4,
        "messages_attempted": 4,
        "messages_inserted_or_seen": 4,
        "messages_excluded_by_sha256": 0,
    }

    assert is_batch_complete(report, {"verification_pass": True}, max_messages=250) is True


def test_full_mail_batch_complete_blocks_when_live_count_exceeds_limit() -> None:
    report = {
        "errors": [],
        "messages_found_since": 251,
        "messages_attempted": 250,
        "messages_inserted_or_seen": 250,
        "messages_excluded_by_sha256": 0,
    }

    assert is_batch_complete(report, {"verification_pass": True}, max_messages=250) is False


def test_full_mail_batch_complete_requires_all_attempted_messages_accounted() -> None:
    report = {
        "errors": [],
        "messages_found_since": 4,
        "messages_attempted": 4,
        "messages_inserted_or_seen": 3,
        "messages_excluded_by_sha256": 0,
    }

    assert is_batch_complete(report, {"verification_pass": True}, max_messages=250) is False
