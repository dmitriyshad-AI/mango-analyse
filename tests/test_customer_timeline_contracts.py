from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

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
    SignalStatus,
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
from mango_mvp.customer_timeline.safety import (
    guard_customer_timeline_writable_path,
    guard_managed_customer_timeline_staging_write,
    managed_staging_writer_scope,
    managed_staging_writer_subprocess_pass_fds,
)
from mango_mvp.customer_timeline.store import customer_timeline_run_lock


NOW = datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(hours=1)
SHA = "a" * 64


def test_customer_timeline_package_import_is_lazy() -> None:
    script = """
import sys
import mango_mvp.customer_timeline
unexpected = [
    name for name in (
        'mango_mvp.customer_timeline.store',
        'mango_mvp.customer_timeline.ingestion',
        'mango_mvp.customer_timeline.preview_quality_audit',
    )
    if name in sys.modules
]
assert not unexpected, unexpected
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_customer_timeline_lazy_facade_resolves_every_public_name() -> None:
    import mango_mvp.customer_timeline as timeline

    assert isinstance(timeline.__all__, list)
    assert len(timeline.__all__) == 193
    assert all(getattr(timeline, name) is not None for name in timeline.__all__)


def test_customer_timeline_lazy_facade_is_thread_safe() -> None:
    script = """
from concurrent.futures import ThreadPoolExecutor
import mango_mvp.customer_timeline as timeline
with ThreadPoolExecutor(max_workers=64) as pool:
    values = list(pool.map(lambda _: timeline.CustomerTimelineSQLiteStore, range(256)))
assert all(value is values[0] for value in values)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


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
    assert payload["status"] == "active"
    assert payload["expires_at"] is None
    assert payload["requires_manager_review"] is True
    json.dumps(payload, ensure_ascii=False)


def test_derived_signal_status_and_expires_at_are_serialized_but_not_in_stable_id() -> None:
    base = DerivedSignal(
        tenant_id="foton",
        customer_id="customer:1",
        event_id="timeline_event:1",
        source_event_ids=("timeline_event:1",),
        signal_type="paid_no_access",
        severity=SignalSeverity.HIGH,
        evidence_text="Оплата есть, активного доступа нет.",
        status=SignalStatus.ACTIVE,
        expires_at=NOW + timedelta(days=14),
        created_at=NOW,
    )
    resolved = DerivedSignal(
        tenant_id="foton",
        customer_id="customer:1",
        event_id="timeline_event:1",
        source_event_ids=("timeline_event:1",),
        signal_type="paid_no_access",
        severity=SignalSeverity.HIGH,
        evidence_text="Оплата есть, активного доступа нет.",
        status=SignalStatus.RESOLVED,
        expires_at=NOW + timedelta(days=30),
        created_at=NOW,
    )

    assert base.signal_id == resolved.signal_id
    assert base.to_json_dict()["status"] == "active"
    assert resolved.to_json_dict()["status"] == "resolved"
    assert base.to_json_dict()["expires_at"] == "2026-05-26T12:00:00+00:00"


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"signal_type": ""}, "signal_type must not be empty"),
        ({"severity": "urgent"}, "'urgent' is not a valid SignalSeverity"),
        ({"status": "closed"}, "'closed' is not a valid SignalStatus"),
        ({"evidence_text": ""}, "evidence_text must not be empty"),
        ({"confidence": -0.1}, "confidence must be between 0 and 1"),
        ({"created_at": datetime(2026, 5, 12, 12, 0)}, "created_at must be timezone-aware"),
        ({"expires_at": datetime(2026, 5, 19, 12, 0)}, "expires_at must be timezone-aware"),
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


def test_customer_timeline_writable_path_rejects_hard_link(tmp_path: Path) -> None:
    original = tmp_path / "customer_timeline.sqlite"
    linked = tmp_path / "staging.sqlite"
    original.write_bytes(b"not a database")
    os.link(original, linked)

    with pytest.raises(ValueError, match="hard link"):
        guard_customer_timeline_writable_path(linked)

    assert guard_customer_timeline_writable_path(tmp_path / "new-staging.sqlite").name == "new-staging.sqlite"
    assert guard_customer_timeline_writable_path(tmp_path) == tmp_path


def test_managed_writer_scope_requires_held_lock_and_exact_db(tmp_path: Path) -> None:
    db_path = _managed_staging_db(tmp_path / "owned")
    other_db = _managed_staging_db(tmp_path / "other")
    alias = db_path.with_name("staging-alias.sqlite")
    os.link(db_path, alias)

    with pytest.raises(ValueError, match="unified nightly service"):
        guard_managed_customer_timeline_staging_write(db_path)
    with pytest.raises(ValueError, match="unified nightly service"):
        guard_managed_customer_timeline_staging_write(alias)
    with pytest.raises(ValueError, match="held nightly service lock"):
        with managed_staging_writer_scope(db_path):
            pass

    with customer_timeline_run_lock(db_path, timeout_seconds=1):
        with managed_staging_writer_scope():
            assert guard_managed_customer_timeline_staging_write(db_path) == db_path
            with pytest.raises(ValueError, match="unified nightly service"):
                guard_managed_customer_timeline_staging_write(other_db)

    with pytest.raises(ValueError, match="unified nightly service"):
        guard_managed_customer_timeline_staging_write(db_path)


def test_managed_writer_scope_inherits_to_subprocess_only_for_locked_exact_db(tmp_path: Path) -> None:
    db_path = _managed_staging_db(tmp_path / "owned")
    other_db = _managed_staging_db(tmp_path / "other")

    with customer_timeline_run_lock(db_path, timeout_seconds=1) as lock_info:
        fake_env = os.environ.copy()
        fake_env.update(
            {
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_DB": str(db_path),
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_LOCK": str(lock_info["path"]),
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PARENT_PID": str(os.getpid()),
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_LOCK": str(lock_info["path"]) + ".owner",
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_FD": "9999",
            }
        )
        denied_forged_scope = _run_managed_guard_child(db_path, env=fake_env)
        with managed_staging_writer_scope():
            inherited_env = os.environ.copy()
            inherited_fds = managed_staging_writer_subprocess_pass_fds()
            allowed = _run_managed_guard_child(db_path, env=inherited_env, pass_fds=inherited_fds)
            wrong_fd_env = dict(inherited_env)
            wrong_fd_env["MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_FD"] = "9999"
            denied_wrong_fd = _run_managed_guard_child(db_path, env=wrong_fd_env, pass_fds=inherited_fds)
            denied_other = _run_managed_guard_child(other_db, env=inherited_env, pass_fds=inherited_fds)
            with Path(str(db_path) + ".nightly_service.lock.owner").open("r") as separate_lock:
                forged_fd_env = dict(inherited_env)
                forged_fd_env["MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_FD"] = str(separate_lock.fileno())
                allowed_same_parent_separate_fd = _run_managed_guard_child(
                    db_path,
                    env=forged_fd_env,
                    pass_fds=(separate_lock.fileno(),),
                )

    denied_after_lock = _run_managed_guard_child(db_path, env=inherited_env)
    denied_without_scope = _run_managed_guard_child(db_path)

    assert allowed.returncode == 0, allowed.stderr
    assert allowed.stdout.strip() == "allowed"
    assert denied_forged_scope.returncode == 1
    assert denied_wrong_fd.returncode == 1
    assert denied_other.returncode == 1
    assert allowed_same_parent_separate_fd.returncode == 0
    assert "unified nightly service" in denied_other.stdout
    assert denied_after_lock.returncode == 1
    assert "unified nightly service" in denied_after_lock.stdout
    assert denied_without_scope.returncode == 1
    assert "unified nightly service" in denied_without_scope.stdout


def test_managed_writer_scope_rejects_inherited_but_unlocked_descriptor(tmp_path: Path) -> None:
    db_path = _managed_staging_db(tmp_path / "owned")
    lock_path = Path(str(db_path) + ".nightly_service.lock")
    proof_lock_path = Path(str(lock_path) + ".owner")
    lock_path.touch()
    proof_lock_path.touch()
    with proof_lock_path.open("r+") as unlocked_handle:
        fake_env = os.environ.copy()
        fake_env.update(
            {
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_DB": str(db_path),
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_LOCK": str(lock_path),
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PARENT_PID": str(os.getpid()),
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_LOCK": str(proof_lock_path),
                "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_FD": str(unlocked_handle.fileno()),
            }
        )
        denied = _run_managed_guard_child(
            db_path,
            env=fake_env,
            pass_fds=(unlocked_handle.fileno(),),
        )

    assert denied.returncode == 1
    assert "unified nightly service" in denied.stdout


def test_managed_writer_scope_rejects_parent_proof_without_run_lock(tmp_path: Path) -> None:
    db_path = _managed_staging_db(tmp_path / "owned")
    lock_path = Path(str(db_path) + ".nightly_service.lock")
    proof_lock_path = Path(str(lock_path) + ".owner")
    lock_path.touch()
    proof_lock_path.touch()
    with proof_lock_path.open("r+") as proof_handle:
        fcntl.lockf(proof_handle.fileno(), fcntl.LOCK_EX)
        try:
            fake_env = os.environ.copy()
            fake_env.update(
                {
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_DB": str(db_path),
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_LOCK": str(lock_path),
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PARENT_PID": str(os.getpid()),
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_LOCK": str(proof_lock_path),
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_FD": str(proof_handle.fileno()),
                }
            )
            denied = _run_managed_guard_child(
                db_path,
                env=fake_env,
                pass_fds=(proof_handle.fileno(),),
            )
        finally:
            fcntl.lockf(proof_handle.fileno(), fcntl.LOCK_UN)

    assert denied.returncode == 1
    assert "unified nightly service" in denied.stdout


def test_managed_writer_scope_rejects_run_lock_owned_by_unrelated_process(tmp_path: Path) -> None:
    db_path = _managed_staging_db(tmp_path / "owned")
    lock_path = Path(str(db_path) + ".nightly_service.lock")
    proof_lock_path = Path(str(lock_path) + ".owner")
    lock_path.touch()
    proof_lock_path.touch()
    holder_script = (
        "import fcntl,os,sys,time; "
        "run=open(sys.argv[1],'r+'); proof=open(sys.argv[2],'r+'); "
        "fcntl.flock(run.fileno(),fcntl.LOCK_EX); "
        "fcntl.lockf(proof.fileno(),fcntl.LOCK_EX); "
        "print('ready',flush=True); time.sleep(30)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", holder_script, str(lock_path), str(proof_lock_path)],
        text=True,
        stdout=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "ready"
        with proof_lock_path.open("r+") as unlocked_handle:
            fake_env = os.environ.copy()
            fake_env.update(
                {
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_DB": str(db_path),
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_LOCK": str(lock_path),
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PARENT_PID": str(os.getpid()),
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_LOCK": str(proof_lock_path),
                    "MANGO_CUSTOMER_TIMELINE_MANAGED_WRITER_PROOF_FD": str(unlocked_handle.fileno()),
                }
            )
            denied = _run_managed_guard_child(
                db_path,
                env=fake_env,
                pass_fds=(unlocked_handle.fileno(),),
            )
    finally:
        holder.terminate()
        holder.wait(timeout=5)

    assert denied.returncode == 1
    assert "unified nightly service" in denied.stdout


def test_direct_raw_sqlite_apply_restore_paths_reject_managed_staging_without_scope(tmp_path: Path) -> None:
    from scripts.backfill_customer_timeline_next_steps_from_summary import backfill_next_steps_from_summary
    from scripts.repair_mail_stage2_event_dates import repair_dates
    from scripts.retrofit_channel_brand_tags_in_timeline import (
        RetrofitChannelBrandConfig,
        run_retrofit_channel_brand_tags,
    )
    from mango_mvp.customer_timeline.family_graph import FamilyGraphConfig, build_family_graph
    from mango_mvp.customer_timeline.mail_stage2_ingest import (
        MailStage2IngestConfig,
        apply_stage2_mail_ingest,
        restore_timeline_backup,
    )

    db_path = _managed_staging_db(tmp_path / "owned")
    event_jsonl = tmp_path / "mail_stage2.jsonl"
    event_jsonl.write_text("", encoding="utf-8")
    mail_config = MailStage2IngestConfig(
        timeline_db_path=db_path,
        allowed_root=db_path.parent,
        identity_db_path=tmp_path / "identity.sqlite",
        event_jsonl_paths=(event_jsonl,),
        out_dir=tmp_path / "mail-out",
    )
    cases = (
        ("backfill", lambda: backfill_next_steps_from_summary(db_path, apply=True)),
        (
            "retrofit",
            lambda: run_retrofit_channel_brand_tags(
                RetrofitChannelBrandConfig(
                    timeline_db=db_path,
                    allowed_root=db_path.parent,
                    apply=True,
                )
            ),
        ),
        (
            "repair",
            lambda: repair_dates(
                db_path=db_path,
                event_paths=(event_jsonl,),
                archive_roots=(tmp_path,),
                dry_run=False,
            ),
        ),
        (
            "family_graph",
            lambda: build_family_graph(
                FamilyGraphConfig(
                    timeline_db=db_path,
                    allowed_root=db_path.parent,
                    apply=True,
                )
            ),
        ),
        ("mail_apply", lambda: apply_stage2_mail_ingest(mail_config, backup_manifest_path=tmp_path / "missing.json")),
        ("mail_restore", lambda: restore_timeline_backup(mail_config, backup_manifest_path=tmp_path / "missing.json")),
    )

    for _name, invoke in cases:
        with pytest.raises(ValueError, match="unified nightly service"):
            invoke()


def test_mail_date_repair_dry_run_opens_timeline_immutable_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.repair_mail_stage2_event_dates as repair_module

    db_path = tmp_path / "customer_timeline.sqlite"
    with sqlite3.connect(db_path) as con:
        con.execute(
            "CREATE TABLE timeline_events ("
            "event_id TEXT,source_id TEXT,source_ref TEXT,source_system TEXT,opportunity_id TEXT,"
            "event_at TEXT,record_json TEXT)"
        )
    observed: list[tuple[str, bool]] = []
    real_connect = sqlite3.connect

    def observed_connect(database, *args, **kwargs):
        observed.append((str(database), bool(kwargs.get("uri"))))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(repair_module.sqlite3, "connect", observed_connect)
    report = repair_module.repair_dates(
        db_path=db_path,
        event_paths=(),
        archive_roots=(),
        dry_run=True,
    )

    assert report["mode"] == "dry_run"
    assert observed == [(db_path.resolve().as_uri() + "?mode=ro&immutable=1", True)]


def _managed_staging_db(root: Path) -> Path:
    staging = root / ".codex_local" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    db_path = staging / "customer_timeline_staging.sqlite"
    db_path.write_bytes(b"")
    state = staging / "state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "WRITER_OWNERSHIP.json").write_text("{}\n", encoding="utf-8")
    return db_path.resolve(strict=False)


def _run_managed_guard_child(
    db_path: Path,
    *,
    env: Mapping[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[str]:
    script = """
import sys
from mango_mvp.customer_timeline.safety import guard_managed_customer_timeline_staging_write
try:
    guard_managed_customer_timeline_staging_write(sys.argv[1])
except Exception as exc:
    print(str(exc))
    raise SystemExit(1)
print("allowed")
"""
    child_env = dict(os.environ if env is None else env)
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    if child_env.get("PYTHONPATH"):
        child_env["PYTHONPATH"] = repo_src + os.pathsep + str(child_env["PYTHONPATH"])
    else:
        child_env["PYTHONPATH"] = repo_src
    return subprocess.run(
        [sys.executable, "-c", script, str(db_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        pass_fds=pass_fds,
        check=False,
    )


@pytest.mark.parametrize("bad_customer_id", ("None", " null ", "NaN", "undefined", "<NA>", "N/A"))
def test_all_customer_owner_contracts_reject_textual_null_ids(bad_customer_id: str) -> None:
    constructors = (
        lambda: CustomerIdentity(
            tenant_id="foton",
            customer_id=bad_customer_id,
            identity_status="strong",
            primary_phone="+79161234567",
            created_at=NOW,
            updated_at=NOW,
        ),
        lambda: IdentityLink(
            tenant_id="foton",
            customer_id=bad_customer_id,
            link_type="amo_contact_id",
            link_value="contact-1",
            source_system="amocrm_snapshot",
            source_ref="contact:1",
        ),
        lambda: CustomerOpportunity(
            tenant_id="foton",
            customer_id=bad_customer_id,
            opportunity_type="amo_deal",
            source_system="amocrm_snapshot",
            source_id="lead-1",
        ),
        lambda: TimelineEvent(
            tenant_id="foton",
            customer_id=bad_customer_id,
            event_type="system_note",
            event_at=NOW,
            source_system="amocrm_snapshot",
            source_id="event-1",
            direction="system",
        ),
        lambda: DerivedSignal(
            tenant_id="foton",
            customer_id=bad_customer_id,
            signal_type="follow_up",
            severity="low",
            evidence_text="Проверка",
            created_at=NOW,
        ),
        lambda: BotContextChunk(
            tenant_id="foton",
            customer_id=bad_customer_id,
            source_ref="test:1",
            chunk_type="summary",
            text="Проверка",
            created_at=NOW,
        ),
    )

    for constructor in constructors:
        with pytest.raises(ValueError, match="null placeholder"):
            constructor()


def test_contract_inventory_lists_core_types_and_safety() -> None:
    inventory = customer_timeline_contract_inventory()

    assert "TimelineEvent" in inventory["contracts"]
    assert "mango_call" in inventory["event_types"]
    assert "whatsapp_message" in inventory["event_types"]
    assert "email" in inventory["identity_link_types"]
    assert "whatsapp_phone" in inventory["identity_link_types"]
    assert inventory["safety"]["write_crm"] is False
