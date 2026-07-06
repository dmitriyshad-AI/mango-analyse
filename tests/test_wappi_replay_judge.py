from __future__ import annotations

from mango_mvp.replay_exam.judge import build_replay_judge_payload, serialize_judge_payload
from mango_mvp.replay_exam.models import BotReplayResult, ReplayCase


def test_replay_judge_payload_keeps_hidden_key_outside_payload() -> None:
    case = ReplayCase(
        dialog_id="d",
        profile_id="p",
        chat_id="c",
        turn_id="d#1",
        brand="foton",
        client_message="Нужна физика",
        manager_reference="Добрый день",
    )
    payload = build_replay_judge_payload(
        case,
        BotReplayResult(route="draft_for_manager", bot_text="B"),
        BotReplayResult(route="draft_for_manager", bot_text="ON"),
        seed="s",
    )
    serialized = serialize_judge_payload(payload)
    assert payload.hidden_key["answer_a"] in {"baseline", "candidate"}
    assert "hidden_key" not in serialized
    assert "replay_judge_v1" in serialized
