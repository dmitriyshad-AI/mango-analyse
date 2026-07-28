from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mango_mvp.customer_timeline.wappi_history_import import (
    _wappi_talk_bridge_warnings,
    WappiFetchLimits,
    WappiFetchStats,
    WappiProfileSpec,
    collect_wappi_widget_links,
    load_wappi_widget_links,
    summarize_wappi_widget_link_cache,
)
from tests.test_wappi_history_import_to_timeline import FakeWidgetWappiClient


PROFILE = WappiProfileSpec(profile_id="p-tg", brand="foton", channel="telegram")
RUNTIME = {("telegram", "p-tg"): {"uuid": "p-tg", "platform": "tg"}}
LIMITS = WappiFetchLimits(
    chat_limit_per_profile=20,
    messages_per_chat=0,
    message_limit_total=50,
    request_limit_total=50,
    sleep_seconds=0,
)


def _collect(client: FakeWidgetWappiClient, db_path: Path) -> Mapping[str, Any]:
    return collect_wappi_widget_links(
        client=client,
        profiles=(PROFILE,),
        runtime_profiles=RUNTIME,
        crm_id="crm-id",
        db_path=db_path,
        limits=LIMITS,
    )


def test_personal_chat_denominator_does_not_include_groups(tmp_path: Path) -> None:
    client = FakeWidgetWappiClient(
        {
            "p-tg": [
                {"id": "1", "type": "user"},
                {"id": "2", "type": "user"},
                {"id": "g", "type": "group"},
            ]
        },
        {},
        {
            ("telegram", "1"): {"contact": {"id": 11}, "leads": [{"id": 101}]},
            ("telegram", "2"): {"contact": None, "leads": []},
        },
    )

    report = _collect(client, tmp_path / "links.sqlite")

    assert report["personal_chats_seen"] == 2
    assert report["personal_chats_total"] == 2
    assert report["counts"]["non_personal"] == 1
    assert report["profiles"]["telegram:p-tg"]["unique_catalogued"] == 3


def test_incomplete_catalogue_has_seen_count_but_no_total(tmp_path: Path) -> None:
    client = FakeWidgetWappiClient(
        {
            "p-tg": [
                {"id": "1", "type": "user"},
                {"id": "2", "type": "user"},
                {"id": "3", "type": "user"},
            ]
        },
        {},
        {},
    )

    report = collect_wappi_widget_links(
        client=client,
        profiles=(PROFILE,),
        runtime_profiles=RUNTIME,
        crm_id="crm-id",
        db_path=tmp_path / "links.sqlite",
        limits=WappiFetchLimits(
            chat_limit_per_profile=1,
            messages_per_chat=0,
            message_limit_total=1,
            request_limit_total=10,
            sleep_seconds=0,
        ),
    )

    assert report["accounting_complete"] is False
    assert report["personal_chats_seen"] == 1
    assert report["personal_chats_total"] is None


def test_post_bridge_cache_summary_exposes_count_delta_and_scope(tmp_path: Path) -> None:
    client = FakeWidgetWappiClient(
        {
            "p-tg": [
                {"id": "1", "type": "user"},
                {"id": "2", "type": "user"},
            ]
        },
        {},
        {
            ("telegram", "1"): {"contact": {"id": 11}, "leads": [{"id": 101}]},
            ("telegram", "2"): {"contact": None, "leads": []},
        },
    )
    db_path = tmp_path / "links.sqlite"
    _collect(client, db_path)
    client.chats["p-tg"] = [{"id": "1", "type": "user"}]
    report = _collect(client, db_path)

    summary = summarize_wappi_widget_link_cache(
        load_wappi_widget_links(db_path),
        personal_chats_total=int(report["personal_chats_total"]),
    )

    assert summary["link_rows_total"] == 2
    assert summary["personal_chats_total"] == 1
    assert summary["cache_row_count_delta"] == 1
    assert summary["scope"] == "amo_entity_link_not_timeline_customer_attribution"
    assert summary["counts"] == {
        "resolved_one_lead": 1,
        "widget_no_contact": 1,
    }
    assert summary["amo_entity_links_by_authority"] == {"wappi_widget": 1}


def test_missing_denominator_is_not_invented_as_zero(tmp_path: Path) -> None:
    client = FakeWidgetWappiClient(
        {"p-tg": [{"id": "1", "type": "user"}]},
        {},
        {("telegram", "1"): {"contact": {"id": 11}, "leads": [{"id": 101}]}},
    )
    db_path = tmp_path / "links.sqlite"
    _collect(client, db_path)

    summary = summarize_wappi_widget_link_cache(
        load_wappi_widget_links(db_path),
        personal_chats_total=None,
    )

    assert summary["link_rows_total"] == 1
    assert summary["personal_chats_total"] is None
    assert summary["cache_row_count_delta"] is None


def test_cache_row_count_delta_is_named_as_a_delta_not_stale_rows() -> None:
    summary = summarize_wappi_widget_link_cache(
        {},
        personal_chats_total=2,
    )

    assert summary["cache_row_count_delta"] == -2
    assert "stale_link_rows" not in summary
    assert "missing_link_rows" not in summary


def test_amo_talk_has_own_counter_and_legacy_counter_stays_zero() -> None:
    stats = WappiFetchStats(linked_by_amo_widget=2, linked_by_amo_talk=3)

    payload = stats.to_json_dict()

    assert payload["linked_by_amo_widget"] == 2
    assert payload["linked_by_amo_talk"] == 3
    assert payload["linked_by_amo_event_sequence"] == 0


def test_pending_candidate_without_talk_report_is_unavailable() -> None:
    warnings = _wappi_talk_bridge_warnings(
        post_bridge_cache={"counts": {"candidate": 1}},
        talk_report={},
    )

    assert warnings == ("wappi_amo_talk:bridge_unavailable",)


def test_talk_lookup_errors_are_reported_as_degraded() -> None:
    warnings = _wappi_talk_bridge_warnings(
        post_bridge_cache={"counts": {"candidate": 1}},
        talk_report={"lookup_error": 1, "invalid_response": 1},
    )

    assert warnings == ("wappi_amo_talk:bridge_degraded",)
