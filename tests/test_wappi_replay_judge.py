from __future__ import annotations

from mango_mvp.replay_exam.judge import build_balanced_replay_judge_payloads, build_replay_judge_payload, serialize_judge_payload
from mango_mvp.replay_exam.models import BotReplayResult, ReplayCase, ReplayMessage


def test_replay_judge_payload_keeps_hidden_key_outside_payload() -> None:
    case = ReplayCase(
        dialog_id="d",
        profile_id="p",
        chat_id="c",
        turn_id="d#1",
        brand="foton",
        client_message="Нужна физика",
        manager_reference="Добрый день",
        turn_index=1,
        contour="foton",
        dialog_key_masked="dialog-mask",
        prefix_messages=(ReplayMessage("p", "c", "m0", "Здравствуйте", 1, False, ts_masked="masked+000000s"),),
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
    assert "manager_reference" not in serialized
    assert "SECRET" not in serialized
    assert "prefix_messages" in serialized
    assert "client_safe_facts_digest" in serialized
    assert "replay_judge_v1" in serialized


def test_replay_judge_balances_ab_exactly_for_even_set() -> None:
    rows = []
    for index in range(4):
        case = ReplayCase("d", "p", "c", f"d#{index}", "foton", "Вопрос", "Ответ")
        rows.append(
            (
                case,
                BotReplayResult(route="draft_for_manager", bot_text="B"),
                BotReplayResult(route="draft_for_manager", bot_text="ON"),
            )
        )

    payloads = build_balanced_replay_judge_payloads(rows, seed="s")

    assert [payload.hidden_key["answer_a"] for payload in payloads].count("baseline") == 2
    assert [payload.hidden_key["answer_a"] for payload in payloads].count("candidate") == 2
