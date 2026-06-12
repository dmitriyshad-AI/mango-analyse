from __future__ import annotations

import argparse
import json
from pathlib import Path

import scripts.run_amo_wappi_draft_loop as runner
from mango_mvp.integrations.amo_wappi_transport import TransportDenied
from mango_mvp.integrations.draft_loop import DraftLoopKey


def test_build_config_loads_profiles_pairs_and_keeps_state_outside_repo(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles.json"
    pairs = tmp_path / "pairs.json"
    local_dir = tmp_path / ".mango_local" / "draft_loop"
    profiles.write_text(
        json.dumps([{"profile_id": "profile-foton", "brand": "foton", "channel": "telegram"}]),
        encoding="utf-8",
    )
    pairs.write_text(
        json.dumps(
            [
                {
                    "profile_id": "profile-foton",
                    "chat_id": "chat-1",
                    "lead_id": "49832125",
                    "expected_brand": "foton",
                }
            ]
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        profiles_file=profiles,
        pairs_file=pairs,
        phase1_config=tmp_path / "missing.json",
        local_dir=local_dir,
        stop_file=tmp_path / "STOP",
        manager_outgoing_visible="unknown",
    )

    config = runner.build_config(args)

    key = DraftLoopKey("profile-foton", "chat-1")
    assert config.brand_for_profile("profile-foton") == "foton"
    assert config.pair_for(key).lead_id == "49832125"
    assert config.state_path == local_dir / "state.json"
    assert config.journal_path == local_dir / "journal.jsonl"
    assert config.heartbeat_path == local_dir / "heartbeat.json"
    assert config.allowed_test_lead_ids == frozenset({"49832125"})


def test_context_builder_marks_draft_loop_as_not_sending_clients(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"schema_version": "kc_knowledge_snapshot_v1", "run_id": "test", "facts": [], "chunks": []}), encoding="utf-8")
    build_context = runner.build_context_builder(snapshot)

    context = build_context(DraftLoopKey("profile-foton", "chat-1"), ("Клиент: 9 класс",), "Цена?", "foton")

    assert context["client_identity"]["channel"] == "wappi_telegram"
    assert context["public_pilot_mode"]["sends_client_replies"] is False
    assert context["public_pilot_mode"]["no_crm_tallanto_write"] is True

    max_context = build_context(DraftLoopKey("profile-foton-max", "chat-1"), (), "Цена?", "foton", channel="max")
    assert max_context["client_identity"]["channel"] == "wappi_max"


def test_safe_transport_blocks_unlisted_wappi_get() -> None:
    ai_office_config = runner.AiOfficeClientConfig(base_url="https://api.fotonai.online", api_key="key")
    wappi_config = runner.WappiClientConfig(base_url="https://wappi.pro", telegram_token="token")
    transport = runner.build_safe_transport(ai_office_config, wappi_config)

    try:
        transport(method="GET", url="https://wappi.pro/tapi/profile/queue/purge?profile_id=p")
    except TransportDenied:
        pass
    else:  # pragma: no cover
        raise AssertionError("queue purge must be denied")

    try:
        transport(method="POST", url="https://educent.amocrm.ru/api/v4/leads/49832125/notes")
    except TransportDenied:
        pass
    else:  # pragma: no cover
        raise AssertionError("direct amoCRM note writes must be denied")
