from __future__ import annotations

from pathlib import Path

import pytest

from mango_mvp.channels.subscription_llm_parts.contracts import SubscriptionDraftResult
from mango_mvp.replay_exam.models import ReplayCase, ReplayMessage
from mango_mvp.replay_exam.provider_adapter import (
    RealReplayDraftProvider,
    assert_real_replay_output_path,
    assert_real_replay_cases_safe,
    build_replay_provider_context,
)


def _case(**kwargs) -> ReplayCase:  # type: ignore[no-untyped-def]
    return ReplayCase(
        dialog_id=kwargs.get("dialog_id", "wappi_replay_dialog"),
        profile_id=kwargs.get("profile_id", "[profile_id:id_aaaaaaaaaaaa]"),
        chat_id=kwargs.get("chat_id", "[chat_id:id_bbbbbbbbbbbb]"),
        turn_id=kwargs.get("turn_id", "turn-1"),
        brand=kwargs.get("brand", "foton"),
        client_message=kwargs.get("client_message", "Есть места?"),
        manager_reference=kwargs.get("manager_reference", "SECRET_MANAGER_REFERENCE"),
        prefix_messages=kwargs.get(
            "prefix_messages",
            (
                ReplayMessage(
                    profile_id="[profile_id:id_aaaaaaaaaaaa]",
                    chat_id="[chat_id:id_bbbbbbbbbbbb]",
                    message_id="[message_id:id_cccccccccccc]",
                    text="Здравствуйте",
                    timestamp=1,
                    from_me=False,
                ),
                ReplayMessage(
                    profile_id="[profile_id:id_aaaaaaaaaaaa]",
                    chat_id="[chat_id:id_bbbbbbbbbbbb]",
                    message_id="[message_id:id_dddddddddddd]",
                    text="Добрый день!",
                    timestamp=2,
                    from_me=True,
                ),
            ),
        ),
        segment=kwargs.get("segment", "chat_only"),
        expected_p0=kwargs.get("expected_p0", False),
        metadata=kwargs.get("metadata", {}),
    )


def test_build_replay_provider_context_is_read_only_and_hides_manager_reference(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    case = _case()

    context = build_replay_provider_context(case, {"older_summary": "client: ранний вопрос"}, snapshot_path=snapshot)

    assert context["active_brand"] == "foton"
    assert context["snapshot_path"] == str(snapshot)
    assert context["knowledge_snapshot_path"] == str(snapshot)
    assert context["TELEGRAM_DIRECT_PATH_PILOT_CONFIG"] == "pilot_gold_v1"
    assert context["TELEGRAM_DIRECT_PATH"] == "1"
    assert context["direct_path_enabled"] is True
    assert context["direct_path_pilot_config"] == "pilot_gold_v1"
    assert context["public_pilot_mode"]["sends_client_replies"] is False
    assert context["public_pilot_mode"]["no_crm_tallanto_write"] is True
    assert "read_only_customer_context" not in context
    assert "SECRET_MANAGER_REFERENCE" not in repr(context)
    assert context["replay_exam"]["manager_reference_passed_to_provider"] is False
    assert any("Клиент:" in item or "Ответ:" in item for item in context["recent_messages"])


def test_assert_real_replay_cases_safe_rejects_non_chat_and_pii() -> None:
    assert_real_replay_cases_safe([_case()])
    with pytest.raises(ValueError, match="chat_only"):
        assert_real_replay_cases_safe([_case(segment="external_context")])
    with pytest.raises(ValueError, match="PII signals"):
        assert_real_replay_cases_safe([_case(client_message="Позвоните 8 999 123-45-67")])
    with pytest.raises(ValueError, match="PII signals"):
        assert_real_replay_cases_safe([_case(manager_reference="Позвоните 8 999 123-45-67")])


def test_assert_real_replay_output_path_rejects_runtime_and_raw_roots() -> None:
    with pytest.raises(ValueError, match="stable_runtime"):
        assert_real_replay_output_path(Path("/tmp/stable_runtime/replay_out"))
    with pytest.raises(ValueError, match="raw dump root"):
        assert_real_replay_output_path(Path("~/.mango_local/replay_exam/raw/pilot10").expanduser())


def test_real_replay_provider_converts_subscription_result_without_live_write(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")

    class FakeDraftProvider:
        def __init__(self) -> None:
            self.context = None

        def build_draft(self, client_message, *, context=None):  # type: ignore[no-untyped-def]
            self.context = context
            assert client_message == "Есть места?"
            return SubscriptionDraftResult(
                route="bot_answer_self_for_pilot",
                draft_text="Места в регулярных группах есть.",
                safety_flags=("seats_default_open_regular_groups",),
                raw_response='{"route":"bot_answer_self_for_pilot"}',
                metadata={"direct_path": {"enabled": True}},
            )

    fake = FakeDraftProvider()
    provider = RealReplayDraftProvider(snapshot_path=snapshot, draft_provider=fake)  # type: ignore[arg-type]
    result = provider(_case(), {"older_summary": ""})

    assert result.route == "bot_answer_self_for_pilot"
    assert result.bot_text == "Места в регулярных группах есть."
    assert result.metadata["replay_raw_response"] == '{"route":"bot_answer_self_for_pilot"}'
    assert result.metadata["replay_provider"]["live_writes_allowed"] is False
    assert fake.context["public_pilot_mode"]["sends_client_replies"] is False


def test_replay_context_overrides_disabling_direct_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TELEGRAM_DIRECT_PATH", "0")

    context = build_replay_provider_context(_case(), {"older_summary": ""}, snapshot_path=snapshot)

    assert context["TELEGRAM_DIRECT_PATH"] == "1"
    assert context["direct_path_enabled"] is True
