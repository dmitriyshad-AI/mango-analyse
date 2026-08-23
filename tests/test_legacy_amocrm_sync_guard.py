from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services import dialogue_contract as contract
from mango_mvp.services.sync_amocrm import (
    AmoCRMSyncService,
    LEGACY_AMOCRM_SYNC_DISABLED_MESSAGE,
    ensure_legacy_amocrm_sync_enabled,
)
from tests import mango_provider_fixture as fx
from tests.test_dialogue_format import make_settings


def test_legacy_amocrm_sync_is_disabled_by_default_at_service_level() -> None:
    settings = make_settings()

    with pytest.raises(RuntimeError, match="Legacy amoCRM contact sync is disabled"):
        ensure_legacy_amocrm_sync_enabled(settings)


def test_legacy_amocrm_sync_allows_explicit_maintenance_opt_in() -> None:
    settings = replace(make_settings(), legacy_amocrm_sync_enabled=True)

    ensure_legacy_amocrm_sync_enabled(settings)


def test_legacy_amocrm_sync_disabled_message_points_to_current_runtime() -> None:
    assert "amocrm_runtime" in LEGACY_AMOCRM_SYNC_DISABLED_MESSAGE
    assert "LEGACY_AMOCRM_SYNC_ENABLED=true" in LEGACY_AMOCRM_SYNC_DISABLED_MESSAGE


def _variants(*, trusted: bool, source_call_id: str) -> str:
    variants = fx.proven_variants()
    if trusted:
        variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(
            source_call_id=source_call_id
        )
    else:
        variants.pop(contract.PROVIDER_EVIDENCE_FIELD, None)
    return json.dumps(variants, ensure_ascii=False)


@pytest.mark.parametrize("sync_dry_run", [True, False])
def test_sync_blocks_untrusted_analysis_before_dry_or_live_use(
    tmp_path, sync_dry_run: bool
) -> None:
    settings = replace(
        make_settings(),
        database_url=f"sqlite:///{tmp_path / 'sync.db'}",
        legacy_amocrm_sync_enabled=True,
        sync_dry_run=sync_dry_run,
    )
    init_db(settings)
    factory = build_session_factory(settings)
    call = CallRecord(
        source_file=str(tmp_path / "call.mp3"),
        source_filename="call.mp3",
        source_call_id="mango-sync-1",
        phone="+79990000001",
        transcript_variants_json=_variants(trusted=False, source_call_id="mango-sync-1"),
        analysis_status="done",
        sync_status="pending",
        analysis_json=json.dumps({"history_summary": "Клиент согласился оплатить."}),
    )
    with factory() as session:
        session.add(call)
        session.commit()
        call_id = call.id

    with patch("mango_mvp.services.sync_amocrm.AmoCRMClient") as client_class:
        with factory() as session:
            result = AmoCRMSyncService(settings).run(session, limit=1)
            stored = session.get(CallRecord, call_id)

    assert result["success"] == 0
    assert result["failed"] == 0
    assert result["skipped"] == (0 if sync_dry_run else 1)
    assert result["dry_run"] == (1 if sync_dry_run else 0)
    assert stored.sync_status == ("pending" if sync_dry_run else "skipped")
    assert stored.sync_attempts == 0
    assert stored.next_retry_at is None
    assert stored.dead_letter_stage is None
    client_class.return_value.find_contact_by_phone.assert_not_called()

    if not sync_dry_run:
        with patch("mango_mvp.services.sync_amocrm.AmoCRMClient"):
            with factory() as session:
                repeated = AmoCRMSyncService(settings).run(session, limit=1)
        assert repeated["processed"] == 0


def test_sync_dry_run_does_not_mark_contract_invalid_analysis_done(tmp_path) -> None:
    settings = replace(
        make_settings(),
        database_url=f"sqlite:///{tmp_path / 'sync.db'}",
        legacy_amocrm_sync_enabled=True,
    )
    init_db(settings)
    factory = build_session_factory(settings)
    call = CallRecord(
        source_file=str(tmp_path / "call.mp3"),
        source_filename="call.mp3",
        source_call_id="mango-sync-2",
        transcript_variants_json=_variants(trusted=True, source_call_id="mango-sync-2"),
        analysis_status="done",
        sync_status="pending",
        analysis_json=json.dumps(
            {"review_reasons": ["analysis_contract_invalid"]}, ensure_ascii=False
        ),
    )
    with factory() as session:
        session.add(call)
        session.commit()
        call_id = call.id
        result = AmoCRMSyncService(settings).run(session, limit=1)
        stored = session.get(CallRecord, call_id)

    assert result["success"] == 0
    assert result["failed"] == 0
    assert result["skipped"] == 0
    assert result["dry_run"] == 1
    assert stored.sync_status == "pending"
    assert stored.sync_attempts == 0
    assert stored.next_retry_at is None
    assert stored.dead_letter_stage is None


def test_live_sync_rechecks_source_after_contact_lookup_before_any_amo_write(
    tmp_path,
) -> None:
    settings = replace(
        make_settings(),
        database_url=f"sqlite:///{tmp_path / 'sync.db'}",
        legacy_amocrm_sync_enabled=True,
        sync_dry_run=False,
    )
    init_db(settings)
    factory = build_session_factory(settings)
    with factory() as session:
        call = CallRecord(
            source_file=str(tmp_path / "call.mp3"),
            source_filename="call.mp3",
            source_call_id="mango-sync-race",
            phone="+79990000001",
            transcript_variants_json=_variants(
                trusted=True, source_call_id="mango-sync-race"
            ),
            analysis_status="done",
            sync_status="pending",
            analysis_json=json.dumps({"version": "before"}),
        )
        session.add(call)
        session.commit()
        call_id = call.id

    guarded = {
        "quality_flags": {},
        "structured_fields": {"next_step": {"action": "Перезвонить"}},
        "follow_up_score": 0,
    }

    def mutate_then_return_contact(_phone):
        with factory() as other:
            stored = other.get(CallRecord, call_id)
            assert stored is not None
            stored.analysis_json = json.dumps({"version": "after"})
            other.commit()
        return {"id": 123}

    with patch(
        "mango_mvp.services.sync_amocrm.guard_stored_analysis",
        return_value=guarded,
    ), patch("mango_mvp.services.sync_amocrm.AmoCRMClient") as client_class:
        client = client_class.return_value
        client.find_contact_by_phone.side_effect = mutate_then_return_contact
        with factory() as session:
            result = AmoCRMSyncService(settings).run(session, limit=1)

    with factory() as session:
        stored = session.get(CallRecord, call_id)
        assert stored is not None
        assert stored.sync_status == "pending"
        assert stored.sync_attempts == 0
    assert result == {
        "processed": 1,
        "success": 0,
        "failed": 0,
        "skipped": 1,
        "dry_run": 0,
    }
    client.add_contact_note.assert_not_called()
    client.update_contact_fields.assert_not_called()
    client.create_task.assert_not_called()
