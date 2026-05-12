from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mango_mvp.customer_timeline import (
    ArtifactType,
    BotContextChunk,
    CustomerIdentity,
    CustomerOpportunity,
    DerivedSignal,
    EventArtifact,
    ExtractionStatus,
    IdentityLink,
    IdentityMatchClass,
    IdentityStatus,
    OpportunityType,
    SignalSeverity,
    TimelineDirection,
    TimelineEvent,
    TimelineEventType,
    TimelineParticipant,
    assert_customer_timeline_safety_contract,
    customer_timeline_contract_inventory,
    customer_timeline_safety_contract,
    dedupe_timeline_events,
    guard_customer_timeline_output_path,
    normalize_email,
    normalize_identity_value,
    stable_digest,
    stable_event_id,
)


NOW = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(hours=1)
SHA = "a" * 64


def test_customer_identity_normalizes_copies_and_serializes() -> None:
    summary = {"touches": 3}
    identity = CustomerIdentity(
        tenant_id=" FOTON ",
        identity_status=IdentityStatus.STRONG,
        display_name="  Иванова Мария  ",
        primary_phone="8 (916) 123-45-67",
        primary_email=" MAILTO:CLIENT@EXAMPLE.COM ",
        first_seen_at=NOW,
        last_seen_at=LATER,
        touch_count=2,
        summary=summary,
        created_at=NOW,
        updated_at=LATER,
    )
    summary["touches"] = 99

    payload = identity.to_json_dict()
    assert identity.tenant_id == "foton"
    assert identity.display_name == "Иванова Мария"
    assert identity.primary_phone == "+79161234567"
    assert identity.primary_email == "client@example.com"
    assert identity.customer_id.startswith("customer:")
    assert identity.summary["touches"] == 3
    assert payload["schema_version"] == "customer_timeline_contracts_v1"
    assert payload["identity_status"] == "strong"
    assert payload["first_seen_at"] == NOW.isoformat()
    json.dumps(payload, ensure_ascii=False)


def test_customer_identity_does_not_use_display_name_as_stable_id_seed() -> None:
    with pytest.raises(ValueError, match="customer_id requires primary_phone"):
        CustomerIdentity(
            tenant_id="foton",
            identity_status="partial",
            display_name="Only Name",
            created_at=NOW,
            updated_at=NOW,
        )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"tenant_id": ""}, "tenant_id must not be empty"),
        ({"identity_status": "confident"}, "'confident' is not a valid IdentityStatus"),
        ({"touch_count": -1}, "touch_count must not be negative"),
        ({"created_at": datetime(2026, 5, 12, 12, 0)}, "created_at must be timezone-aware"),
        ({"updated_at": NOW - timedelta(seconds=1)}, "updated_at must be greater than or equal"),
        ({"first_seen_at": LATER, "last_seen_at": NOW}, "last_seen_at must be greater than or equal"),
        ({"primary_email": "not an email"}, "invalid email identity value"),
    ],
)
def test_customer_identity_validation(kwargs: dict, error: str) -> None:
    base = {
        "tenant_id": "foton",
        "identity_status": "strong",
        "primary_phone": "+79161234567",
        "created_at": NOW,
        "updated_at": NOW,
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error):
        CustomerIdentity(**base)


def test_identity_link_normalizes_builds_stable_id_and_serializes() -> None:
    evidence = {"source": "Tallanto export"}
    link = IdentityLink(
        tenant_id="FOTON",
        customer_id="customer:1",
        link_type="email",
        link_value="CLIENT@EXAMPLE.COM",
        source_system="tallanto_export",
        source_ref="Ученики.csv#row=10",
        match_class="strong_unique",
        confidence=0.94,
        evidence=evidence,
        first_seen_at=NOW,
        last_seen_at=LATER,
    )
    evidence["source"] = "mutated"

    payload = link.to_json_dict()
    assert link.tenant_id == "foton"
    assert link.link_value == "client@example.com"
    assert link.link_id.startswith("identity_link:")
    assert link.evidence["source"] == "Tallanto export"
    assert payload["match_class"] == "strong_unique"
    assert payload["confidence"] == 0.94
    json.dumps(payload, ensure_ascii=False)


def test_identity_link_allows_ambiguous_same_identifier_without_auto_merge() -> None:
    first = IdentityLink(
        tenant_id="foton",
        customer_id="customer:1",
        link_type="phone",
        link_value="8 916 123 45 67",
        source_system="tallanto_export",
        source_ref="row=1",
        match_class="ambiguous",
    )
    second = IdentityLink(
        tenant_id="foton",
        customer_id="customer:2",
        link_type="phone",
        link_value="+7 916 123 45 67",
        source_system="tallanto_export",
        source_ref="row=2",
        match_class="ambiguous",
    )

    assert first.link_value == second.link_value
    assert first.link_id != second.link_id
    assert first.customer_id != second.customer_id


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"link_value": ""}, "link_value must not be empty"),
        ({"link_type": "telegram"}, "'telegram' is not a valid IdentityLinkType"),
        ({"match_class": "certain"}, "'certain' is not a valid IdentityMatchClass"),
        ({"confidence": 1.2}, "confidence must be between 0 and 1"),
        ({"first_seen_at": datetime(2026, 5, 12, 12, 0)}, "first_seen_at must be timezone-aware"),
        ({"first_seen_at": LATER, "last_seen_at": NOW}, "last_seen_at must be greater than or equal"),
    ],
)
def test_identity_link_validation(kwargs: dict, error: str) -> None:
    base = {
        "tenant_id": "foton",
        "customer_id": "customer:1",
        "link_type": "email",
        "link_value": "client@example.com",
        "source_system": "tallanto_export",
        "source_ref": "row=1",
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error):
        IdentityLink(**base)


def test_opportunity_normalizes_copies_and_serializes() -> None:
    context = {"subject": "math"}
    opportunity = CustomerOpportunity(
        tenant_id=" FOTON ",
        customer_id="customer:1",
        opportunity_type=OpportunityType.AMO_DEAL,
        source_system="amocrm_snapshot",
        source_id="lead:100",
        title="  ЕГЭ математика  ",
        status=" open ",
        product_context=context,
        opened_at=NOW,
        confidence=0.8,
    )
    context["subject"] = "mutated"

    payload = opportunity.to_json_dict()
    assert opportunity.tenant_id == "foton"
    assert opportunity.opportunity_id.startswith("opportunity:")
    assert opportunity.title == "ЕГЭ математика"
    assert opportunity.status == "open"
    assert opportunity.product_context["subject"] == "math"
    assert payload["opportunity_type"] == "amo_deal"
    json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"customer_id": ""}, "customer_id must not be empty"),
        ({"opportunity_type": "deal"}, "'deal' is not a valid OpportunityType"),
        ({"source_system": ""}, "source_system must not be empty"),
        ({"source_id": ""}, "source_id must not be empty"),
        ({"confidence": -0.1}, "confidence must be between 0 and 1"),
        ({"opened_at": LATER, "closed_at": NOW}, "closed_at must be greater than or equal"),
    ],
)
def test_opportunity_validation(kwargs: dict, error: str) -> None:
    base = {
        "tenant_id": "foton",
        "customer_id": "customer:1",
        "opportunity_type": "amo_deal",
        "source_system": "amocrm_snapshot",
        "source_id": "lead:100",
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error):
        CustomerOpportunity(**base)


def test_timeline_event_builds_stable_event_id_from_source_identity_only() -> None:
    first = TimelineEvent(
        tenant_id="foton",
        customer_id="customer:1",
        event_type="email_message",
        event_at=NOW,
        source_system="mail_archive",
        source_id="message-sha",
        direction="inbound",
        summary="first summary",
        record={"a": 1},
    )
    repeat_with_different_summary = TimelineEvent(
        tenant_id="foton",
        customer_id="customer:1",
        event_type="email_message",
        event_at=NOW,
        source_system="mail_archive",
        source_id="message-sha",
        direction="inbound",
        summary="updated summary",
        record={"b": 2},
    )
    different_type = TimelineEvent(
        tenant_id="foton",
        customer_id="customer:1",
        event_type="email_attachment",
        event_at=NOW,
        source_system="mail_archive",
        source_id="message-sha",
        direction="inbound",
    )

    assert first.event_id == stable_event_id(
        tenant_id="foton",
        source_system="mail_archive",
        source_id="message-sha",
        event_type="email_message",
    )
    assert first.event_id == repeat_with_different_summary.event_id
    assert first.dedupe_key == repeat_with_different_summary.dedupe_key
    assert first.event_id != different_type.event_id


def test_timeline_event_serializes_structured_payload_without_text() -> None:
    participant = TimelineParticipant(role="client", ref="phone:+79161234567", name="Client")
    event = TimelineEvent(
        tenant_id="foton",
        customer_id=None,
        event_type=TimelineEventType.AMO_DEAL_STAGE,
        event_at=NOW,
        source_system="amocrm_snapshot",
        source_id="deal:1:stage:2",
        source_ref="amocrm:deal:1",
        source_refs=("amocrm:import:run-1",),
        direction=TimelineDirection.SYSTEM,
        participants=(participant,),
        stage_before="new",
        stage_after="offer_sent",
        match_status=IdentityMatchClass.UNMATCHED,
        record={"stage": "offer_sent"},
        created_at=NOW,
    )

    payload = event.to_json_dict()
    assert payload["event_type"] == "amo_deal_stage"
    assert payload["direction"] == "system"
    assert payload["customer_id"] is None
    assert payload["match_status"] == "unmatched"
    assert payload["source_refs"][0] == "amocrm:deal:1"
    assert payload["participants"][0]["role"] == "client"
    json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"tenant_id": ""}, "tenant_id must not be empty"),
        ({"event_type": "call"}, "'call' is not a valid TimelineEventType"),
        ({"direction": "external"}, "'external' is not a valid TimelineDirection"),
        ({"event_at": datetime(2026, 5, 12, 12, 0)}, "event_at must be timezone-aware"),
        ({"source_system": ""}, "source_system must not be empty"),
        ({"source_id": ""}, "source_id must not be empty"),
        ({"importance": -1}, "importance must not be negative"),
        ({"confidence": 9}, "confidence must be between 0 and 1"),
        ({"participants": ("bad",)}, "participants must contain TimelineParticipant"),
    ],
)
def test_timeline_event_validation(kwargs: dict, error: str) -> None:
    base = {
        "tenant_id": "foton",
        "event_type": "mango_call",
        "event_at": NOW,
        "source_system": "mango_office",
        "source_id": "call-1",
        "direction": "inbound",
    }
    base.update(kwargs)

    with pytest.raises((ValueError, TypeError), match=error):
        TimelineEvent(**base)


def test_dedupe_timeline_events_keeps_first_occurrence() -> None:
    first = TimelineEvent(
        tenant_id="foton",
        event_type="mango_call",
        event_at=NOW,
        source_system="mango_office",
        source_id="call-1",
        direction="inbound",
        summary="first",
    )
    duplicate = TimelineEvent(
        tenant_id="foton",
        event_type="mango_call",
        event_at=NOW,
        source_system="mango_office",
        source_id="call-1",
        direction="inbound",
        summary="duplicate",
    )
    second = TimelineEvent(
        tenant_id="foton",
        event_type="mango_call",
        event_at=NOW,
        source_system="mango_office",
        source_id="call-2",
        direction="inbound",
    )

    assert dedupe_timeline_events((first, duplicate, second)) == (first, second)
    with pytest.raises(TypeError, match="events must contain TimelineEvent"):
        dedupe_timeline_events((first, "bad"))  # type: ignore[arg-type]


def test_event_artifact_builds_stable_id_without_reading_file() -> None:
    artifact = EventArtifact(
        tenant_id="FOTON",
        event_id="timeline_event:1",
        artifact_type=ArtifactType.RAW_EMAIL_EML,
        path="/does/not/exist/raw.eml",
        sha256=SHA.upper(),
        size_bytes=123,
        mime_type="message/rfc822",
        source_system="mail_archive",
        source_ref="mail:1",
        extraction_status=ExtractionStatus.PENDING,
        created_at=NOW,
    )

    payload = artifact.to_json_dict()
    assert artifact.tenant_id == "foton"
    assert artifact.artifact_id.startswith("event_artifact:")
    assert artifact.sha256 == SHA
    assert payload["path"] == "/does/not/exist/raw.eml"
    assert payload["extraction_status"] == "pending"
    json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"size_bytes": -1}, "size_bytes must not be negative"),
        ({"sha256": "not-sha"}, "sha256 must be a 64-character hex digest"),
        ({"artifact_type": "email"}, "'email' is not a valid ArtifactType"),
        ({"created_at": datetime(2026, 5, 12, 12, 0)}, "created_at must be timezone-aware"),
    ],
)
def test_event_artifact_validation(kwargs: dict, error: str) -> None:
    base = {
        "tenant_id": "foton",
        "event_id": "timeline_event:1",
        "artifact_type": "raw_email_eml",
        "path": "raw.eml",
        "source_system": "mail_archive",
        "source_ref": "mail:1",
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error):
        EventArtifact(**base)


def test_derived_signal_supports_multiple_source_events_and_serializes() -> None:
    signal = DerivedSignal(
        tenant_id="FOTON",
        customer_id="customer:1",
        opportunity_id="opportunity:1",
        event_id="timeline_event:1",
        source_event_ids=("timeline_event:2",),
        signal_type="price_question",
        severity=SignalSeverity.MEDIUM,
        evidence_text="Клиент спросил про стоимость.",
        confidence=0.77,
        recommended_action="send_price_explanation",
        requires_manager_review=True,
        created_at=NOW,
    )

    payload = signal.to_json_dict()
    assert signal.tenant_id == "foton"
    assert signal.signal_id.startswith("derived_signal:")
    assert signal.source_event_ids == ("timeline_event:1", "timeline_event:2")
    assert payload["severity"] == "medium"
    assert payload["requires_manager_review"] is True
    json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"signal_type": ""}, "signal_type must not be empty"),
        ({"severity": "urgent"}, "'urgent' is not a valid SignalSeverity"),
        ({"evidence_text": ""}, "evidence_text must not be empty"),
        ({"confidence": -0.1}, "confidence must be between 0 and 1"),
        ({"created_at": datetime(2026, 5, 12, 12, 0)}, "created_at must be timezone-aware"),
    ],
)
def test_derived_signal_validation(kwargs: dict, error: str) -> None:
    base = {
        "tenant_id": "foton",
        "customer_id": "customer:1",
        "signal_type": "price_question",
        "severity": "medium",
        "evidence_text": "Клиент спросил про цену.",
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error):
        DerivedSignal(**base)


def test_bot_context_chunk_defaults_are_safe_and_serializable() -> None:
    chunk = BotContextChunk(
        tenant_id="FOTON",
        customer_id="customer:1",
        event_id="timeline_event:1",
        source_system="mail_archive",
        chunk_type="recent_event",
        text="Клиент спрашивал расписание.",
        summary="Вопрос про расписание",
        event_at=NOW,
        freshness_score=0.9,
        relevance_tags=("Schedule_Question", "Follow_Up"),
        created_at=NOW,
    )

    payload = chunk.to_json_dict()
    assert chunk.tenant_id == "foton"
    assert chunk.chunk_id.startswith("bot_context_chunk:")
    assert chunk.relevance_tags == ("schedule_question", "follow_up")
    assert payload["allowed_for_bot"] is True
    assert payload["event_at"] == NOW.isoformat()
    json.dumps(payload, ensure_ascii=False)


def test_bot_context_chunk_requiring_manager_review_is_not_allowed_for_bot() -> None:
    with pytest.raises(ValueError, match="must not be allowed_for_bot"):
        BotContextChunk(
            tenant_id="foton",
            customer_id="customer:1",
            event_id="timeline_event:1",
            chunk_type="recent_event",
            text="Нужна проверка менеджера.",
            requires_manager_review=True,
            allowed_for_bot=True,
        )


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"text": ""}, "text must not be empty"),
        ({"event_id": None, "source_ref": None}, "chunk_id requires event_id or source_ref"),
        ({"freshness_score": 1.1}, "freshness_score must be between 0 and 1"),
        ({"event_at": datetime(2026, 5, 12, 12, 0)}, "event_at must be timezone-aware"),
        ({"ordinal": -1}, "ordinal must not be negative"),
    ],
)
def test_bot_context_chunk_validation(kwargs: dict, error: str) -> None:
    base = {
        "tenant_id": "foton",
        "customer_id": "customer:1",
        "event_id": "timeline_event:1",
        "chunk_type": "recent_event",
        "text": "Контекст.",
    }
    base.update(kwargs)

    with pytest.raises(ValueError, match=error):
        BotContextChunk(**base)


def test_stable_digest_is_independent_of_mapping_key_order() -> None:
    assert stable_digest({"a": 1, "b": {"x": 2, "y": 3}}) == stable_digest(
        {"b": {"y": 3, "x": 2}, "a": 1}
    )


def test_normalizers_cover_email_and_phone_identity_values() -> None:
    assert normalize_email(" MAILTO:Student@Example.Org ") == "student@example.org"
    assert normalize_email("not an email") == ""
    assert normalize_identity_value("phone", "8 916 123-45-67") == "+79161234567"
    assert normalize_identity_value("email", "CLIENT@EXAMPLE.COM") == "client@example.com"


def test_customer_timeline_safety_contract_blocks_external_effects(tmp_path: Path) -> None:
    safety = customer_timeline_safety_contract()

    assert safety["read_only_source_systems"] is True
    for key in (
        "write_crm",
        "write_tallanto",
        "send_email",
        "send_messenger",
        "live_send",
        "run_asr",
        "run_ra",
        "write_runtime_db",
        "runtime_db_writes",
        "mutate_stable_runtime",
        "stable_runtime_writes",
        "delete_source_artifacts",
        "store_raw_files_in_sqlite",
    ):
        assert safety[key] is False
    assert_customer_timeline_safety_contract(safety)

    allowed = guard_customer_timeline_output_path(tmp_path / "product_data" / "timeline.sqlite", tmp_path)
    assert allowed.name == "timeline.sqlite"
    with pytest.raises(ValueError, match="stable_runtime"):
        guard_customer_timeline_output_path(tmp_path / "Stable_Runtime" / "timeline.sqlite", tmp_path)
    with pytest.raises(ValueError, match="allowed root"):
        guard_customer_timeline_output_path(tmp_path.parent / "outside.sqlite", tmp_path)


def test_contract_inventory_lists_core_types_and_safety() -> None:
    inventory = customer_timeline_contract_inventory()

    assert "TimelineEvent" in inventory["contracts"]
    assert "mango_call" in inventory["event_types"]
    assert "email" in inventory["identity_link_types"]
    assert inventory["safety"]["write_crm"] is False
