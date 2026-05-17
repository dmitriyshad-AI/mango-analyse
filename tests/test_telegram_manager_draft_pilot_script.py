from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from scripts.telegram_manager_draft_pilot import build_preview_service_from_env, run_dry_run, run_long_polling
from mango_mvp.channels.contracts import ChannelDirection, ChannelMessage
from mango_mvp.channels.subscription_llm import FakeDraftProvider


def test_telegram_manager_draft_pilot_dry_run_builds_manager_payload(capsys) -> None:
    exit_code = run_dry_run(manager_chat_id="700100")
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["mode"] == "dry_run"
    assert payload["inbound_result"]["status"] == "accepted"
    assert payload["manager_deliveries"][0]["status"] == "ready_for_manager_chat"
    assert payload["manager_deliveries"][0]["telegram_api_called"] is False
    assert payload["safety"]["bot_polling"]["client_send"] is False
    assert payload["safety"]["manager_inbox"]["client_send"] is False


def test_telegram_manager_draft_pilot_long_polling_requires_explicit_confirmation() -> None:
    with pytest.raises(SystemExit, match="Long polling не запущен"):
        run_long_polling("")


def test_preview_service_from_env_uses_llm_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_PILOT_LLM_ENABLED", "1")
    monkeypatch.setenv("TELEGRAM_PILOT_CODEX_REASONING_EFFORT", "xhigh")

    service = build_preview_service_from_env()

    assert service is not None
    assert service.draft_provider.reasoning_effort == "xhigh"


def test_telegram_manager_draft_pilot_can_use_kb_snapshot_when_env_set(tmp_path, monkeypatch) -> None:
    snapshot_path = tmp_path / "kb_release_v3_snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "schema_version": "kb_release_v3_snapshot_test",
                "run_id": "kb_release_v3_test",
                "sources": [
                    {
                        "source_id": "claude_layer_v3:facts_for_bot_UNPK",
                        "title": "УНПК цены",
                        "fact_types": ["price"],
                        "freshness_status": "fresh_verified",
                        "usable_for_precise_answer": True,
                    }
                ],
                "facts": [
                    {
                        "fact_id": "fact:unpk_offline_5_11_price",
                        "fact_type": "price",
                        "brand": "unpk",
                        "client_safe_text": "УНПК очно для 5-11 классов: 49 000 или 82 000 рублей.",
                        "source_id": "claude_layer_v3:facts_for_bot_UNPK",
                        "freshness_status": "fresh_verified",
                        "usable_for_precise_answer": True,
                        "requires_manager_confirmation": False,
                        "forbidden_for_client": False,
                    },
                    {
                        "fact_id": "fact:foton_price",
                        "fact_type": "price",
                        "brand": "foton",
                        "client_safe_text": "Фотон: 120 000 рублей.",
                        "source_id": "claude_layer_v3:facts_for_bot_FOTON",
                        "freshness_status": "fresh_verified",
                        "usable_for_precise_answer": True,
                        "requires_manager_confirmation": False,
                        "forbidden_for_client": False,
                    },
                ],
                "chunks": [
                    {
                        "chunk_id": "chunk:unpk_price",
                        "source_id": "claude_layer_v3:facts_for_bot_UNPK",
                        "title": "УНПК цены",
                        "text": "УНПК очно для 5-11 классов: 49 000 или 82 000 рублей.",
                        "fact_types": ["price"],
                        "freshness_status": "fresh_verified",
                        "brand": "unpk",
                    },
                    {
                        "chunk_id": "chunk:foton_price",
                        "source_id": "claude_layer_v3:facts_for_bot_FOTON",
                        "title": "Фотон цены",
                        "text": "Фотон: 120 000 рублей.",
                        "fact_types": ["price"],
                        "freshness_status": "fresh_verified",
                        "brand": "foton",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    provider = FakeDraftProvider(
        {
            "message_type": "question",
            "topic_id": "theme:001_pricing",
            "route": "manager_only",
            "draft_text": "Передам вопрос менеджеру, он вернется с проверенным ответом.",
            "safety_flags": ["manager_approval_required", "no_auto_send"],
        }
    )
    monkeypatch.setenv("TELEGRAM_PILOT_LLM_ENABLED", "1")
    monkeypatch.setenv("TELEGRAM_PILOT_KB_SNAPSHOT_PATH", str(snapshot_path))
    monkeypatch.setenv("TELEGRAM_PILOT_ACTIVE_BRAND", "unpk")
    monkeypatch.setattr("scripts.telegram_manager_draft_pilot.SubscriptionLlmDraftProvider", lambda **_kwargs: provider)

    service = build_preview_service_from_env()
    message = ChannelMessage(
        channel="telegram_bot",
        channel_message_id="msg-1",
        channel_thread_id="chat-1",
        channel_user_id="user-1",
        direction=ChannelDirection.INBOUND,
        text="Сколько стоит очное обучение УНПК?",
        received_at=datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc),
        metadata={"telegram_chat_type": "private"},
    )

    context = service.context_builder(message)
    preview = service.build_preview(message)

    assert context["active_brand"] == "unpk"
    assert context["knowledge_base_version"] == "kb_release_v3_test"
    assert context["facts_context"]["snapshot_found"] is True
    assert context["facts_context"]["active_brand"] == "unpk"
    assert list(context["confirmed_facts"]) == ["fact:unpk_offline_5_11_price"]
    assert "49 000" in " ".join(context["knowledge_snippets"])
    assert "Фотон" not in " ".join(context["knowledge_snippets"])
    assert preview.reply.requires_approval is True
    assert "live_send_disabled" in preview.blocked_reasons
    assert preview.safety["live_send"] is False
    assert preview.safety["write_crm"] is False
    assert "draft_only" in preview.reply.safety_flags
    assert provider.prompts and "kb_release_v3_test" in provider.prompts[0]
