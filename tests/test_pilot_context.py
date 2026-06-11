from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.channels.contracts import ChannelDirection, ChannelMessage
from mango_mvp.channels.pilot_context import (
    MEMORY_PROVENANCE_COMPACT_ENV,
    build_pilot_context,
    compact_dialogue_memory_view,
    pilot_context_safety_contract,
)
from mango_mvp.channels.draft_prompt_builder import build_prompt_context
from mango_mvp.channels.few_shot_reference import build_gold_answer_context, build_few_shot_reference


def _write_gold_answers_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "bot_gold_answers.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "gold_answers_test_v1",
                "global_rules": ["Не использовать gold как источник фактов."],
                "topics": {
                    "camps": {
                        "unpk": {
                            "gold_answer_example": (
                                "ЛВШ Менделеево в УНПК сейчас стоит 114 000 ₽, "
                                "полная стоимость — 120 000 ₽."
                            ),
                            "must_include": ["ЛВШ Менделеево"],
                            "must_not_include": ["Фотон"],
                        }
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_pilot_context_marks_family_phone_and_quality() -> None:
    message = ChannelMessage(
        channel="telegram_bot",
        channel_message_id="m1",
        channel_thread_id="chat-1",
        channel_user_id="u1",
        direction=ChannelDirection.INBOUND,
        text="Какая группа подойдет?",
        received_at=datetime(2026, 5, 17, 9, 0, tzinfo=timezone.utc),
    )

    context = build_pilot_context(
        message,
        recent_messages=("Здравствуйте", "У меня двое детей"),
        client_identity={"phone": "+79000000000"},
        amo_context={"family_phone": True, "deals_count": 2},
        tallanto_context={"students_count": 2},
        facts_context={"missing": True},
    )
    payload = context.to_prompt_context()

    assert payload["context_quality"]["phone_found"] is True
    assert payload["context_quality"]["family_phone"] is True
    assert payload["context_quality"]["multiple_students"] is True
    assert payload["context_quality"]["multiple_deals"] is True
    assert payload["context_quality"]["facts_missing"] is True
    assert "family_phone" in payload["context_warnings"]
    assert "multiple_students" in payload["risk_flags"]


def test_pilot_context_safety_contract_is_read_only() -> None:
    safety = pilot_context_safety_contract()

    assert safety["read_amo"] is True
    assert safety["read_tallanto"] is True
    assert safety["write_amo"] is False
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["send_client_message"] is False


def test_pilot_context_preserves_full_autonomy_topic_list() -> None:
    allowed = [f"theme:{index:03d}_test" for index in range(1, 25)]
    allowed.extend(["theme:014_format", "theme:015_address", "theme:026_camp_general"])

    context = build_pilot_context(
        "Можно заниматься онлайн?",
        active_brand="foton",
        rop_policy={
            "bot_permission": "bot_answer_self_for_pilot",
            "autonomy_policy": {
                "allow_autonomous": True,
                "allowed_topic_ids": allowed,
            },
        },
    )
    payload = context.to_prompt_context()

    preserved = payload["rop_policy"]["autonomy_policy"]["allowed_topic_ids"]
    assert "theme:014_format" in preserved
    assert "theme:015_address" in preserved
    assert "theme:026_camp_general" in preserved
    assert len(preserved) == len(allowed)


def test_pilot_context_compaction_preserves_held_state_and_focus() -> None:
    context = build_pilot_context(
        "А где её смотреть потом?",
        active_brand="foton",
        dialogue_memory_view={
            "known_slots": {"grade": "8", "format": "онлайн"},
            "held_state": {
                "active_fact_scope": "online_recordings",
                "active_topics": ["recording"],
                "required_fact_keys": ["online_recordings.current"],
            },
            "topic_focus": {"product_family": "regular_course", "product": "онлайн"},
            "safe_answered_parts": ["по онлайн-занятиям записи доступны"],
        },
    )

    payload = context.to_prompt_context()
    memory = payload["dialogue_memory_view"]
    assert memory["held_state"]["active_fact_scope"] == "online_recordings"
    assert memory["topic_focus"]["product_family"] == "regular_course"
    assert "по онлайн-занятиям записи доступны" in memory["safe_answered_parts"]


def test_pilot_context_memory_provenance_compact_flag_preserves_late_provenance(monkeypatch) -> None:
    source = {
        **{f"filler_{index}": f"value {index}" for index in range(24)},
        "known_slots": {"grade": "9", "subject": "физика", "format": "очно"},
        "slot_sources": {"grade": "memory_provenance", "subject": "memory_provenance", "format": "memory_provenance"},
        "client_confirmed_slots": {"grade": "9", "subject": "физика", "format": "очно"},
        "slot_provenance": {
            "grade": {
                "value": "9",
                "source": "memory_provenance",
                "quote": "9 класс, физика, очно",
                "turn_index": 1,
            }
        },
    }

    monkeypatch.setenv("TELEGRAM_DIRECT_PATH_PILOT_CONFIG", "pilot_gold_v1")
    monkeypatch.setenv(MEMORY_PROVENANCE_COMPACT_ENV, "0")
    off = compact_dialogue_memory_view(source)
    assert "known_slots" in off
    assert "slot_provenance" not in off
    assert "client_confirmed_slots" not in off

    monkeypatch.delenv(MEMORY_PROVENANCE_COMPACT_ENV, raising=False)
    on = compact_dialogue_memory_view(source)
    assert on["known_slots"]["grade"] == "9"
    assert on["slot_sources"]["grade"] == "memory_provenance"
    assert on["client_confirmed_slots"]["format"] == "очно"
    assert on["slot_provenance"]["grade"]["quote"] == "9 класс, физика, очно"


def test_gold_answer_context_is_brand_topic_filtered_and_not_fact_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TELEGRAM_DRAFT_GOLD_V3_CONTEXT", "1")
    gold_path = _write_gold_answers_fixture(tmp_path)

    context = build_gold_answer_context(
        message_text="Сколько стоит ЛВШ Менделеево?",
        active_brand="unpk",
        topic_id="theme:026_camp_general",
        confirmed_facts={"unpk_lvsh": "ЛВШ Менделеево в УНПК сейчас стоит 114 000 ₽, полная стоимость — 120 000 ₽."},
        gold_path=gold_path,
    )

    assert context["active_brand"] == "unpk"
    assert context["detected_topic"] == "camps"
    examples = context["examples"]
    assert examples
    assert examples[0]["brand"] == "unpk"
    assert "114 000 ₽" in examples[0]["gold_answer_example"]
    assert "tone_and_structure_only_not_fact_source" in context["purpose"]


def test_gold_answer_context_skips_precise_example_without_confirmed_fact(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TELEGRAM_DRAFT_GOLD_V3_CONTEXT", "1")
    gold_path = _write_gold_answers_fixture(tmp_path)

    context = build_gold_answer_context(
        message_text="Сколько стоит ЛВШ Менделеево?",
        active_brand="unpk",
        topic_id="theme:026_camp_general",
        confirmed_facts={},
        gold_path=gold_path,
    )

    assert context.get("examples") in (None, [])
    assert "Gold-ответы задают тон" in " ".join(context["injection_rules"])
    assert "нельзя додумывать из gold-примера" in " ".join(context["injection_rules"])


def test_gold_answer_context_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("TELEGRAM_DRAFT_GOLD_V3_CONTEXT", "0")

    context = build_gold_answer_context(
        message_text="Сколько стоит ЛВШ Менделеево?",
        active_brand="unpk",
        topic_id="theme:026_camp_general",
        confirmed_facts={"unpk_lvsh": "114 000 ₽"},
    )

    assert context == {}


def test_prompt_context_keeps_expanded_few_shot_limits() -> None:
    payload = build_prompt_context(
        {
            "few_shot_style_examples": [f"style {idx}" for idx in range(8)],
            "few_shot_correction_examples": [f"correction {idx}" for idx in range(6)],
        }
    )

    assert payload["few_shot_style_examples"] == [f"style {idx}" for idx in range(6)]
    assert payload["few_shot_correction_examples"] == [f"correction {idx}" for idx in range(4)]


def test_few_shot_reference_uses_expanded_limits(monkeypatch) -> None:
    monkeypatch.delenv("MANGO_TELEGRAM_FEW_SHOT_WARM_PATH", raising=False)
    monkeypatch.delenv("MANGO_TELEGRAM_FEW_SHOT_ADVANCED_PATH", raising=False)

    reference = build_few_shot_reference(
        message_text="Сколько стоит очно?",
        active_brand="foton",
        topic_id="theme:001_pricing",
        confirmed_facts={"price": "Стоимость очного курса подтверждена."},
    )

    assert len(reference.get("style_examples", ())) <= 6
    assert len(reference.get("correction_examples", ())) <= 4
