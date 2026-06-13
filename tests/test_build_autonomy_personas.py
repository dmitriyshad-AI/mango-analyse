from __future__ import annotations

import json
from pathlib import Path

from scripts import build_autonomy_personas as builder


def test_build_autonomy_personas_masks_raw_and_indirect_identifiers(tmp_path: Path) -> None:
    telegram_export = write_telegram_export(tmp_path)
    voice_dir = write_voice_transcripts(tmp_path)

    rows = builder.build_autonomy_persona_rows(
        telegram_export=telegram_export,
        voice_transcripts_dir=voice_dir,
        limit=2,
        seed=7,
    )

    assert rows[0]["type"] == "simulator_spec"
    assert rows[1]["type"] == "judge_spec"
    personas = rows[2:]
    assert len(personas) == 2
    assert all(persona["brand"] == "unpk" for persona in personas)
    assert all(persona["privacy"]["raw_text_included"] is False for persona in personas)
    assert all(persona["source_provenance"]["donor_count"] <= builder.DONOR_LIMIT_PER_PERSONA for persona in personas)
    for persona in personas:
        provenance = persona["source_provenance"]
        public_tokens = [provenance["telegram_donor_hash"], *provenance["voice_donor_hashes"]]
        for token in public_tokens:
            _, pseudonym = token.split(":", 1)
            assert pseudonym.isalpha()
            assert pseudonym == pseudonym.lower()
    assert all("expected_action" in persona for persona in personas)
    assert all(persona["expected_action"]["manual_label"] is False for persona in personas)

    output = json.dumps(personas, ensure_ascii=False, sort_keys=True)
    forbidden_fragments = [
        "Мария Иванова",
        "Иванова",
        "79001234567",
        "maria@example.com",
        "лицей НИУ ВШЭ",
        "НИУ ВШЭ",
        "Менделеево",
        "17-27 июля",
        "33 383",
        "Сухаревская",
        "Скорняжный",
        "преподаватель Петрова",
        "Добрый день, меня зовут",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in output
    assert builder.safety_violations(builder.build_safe_text_blob(personas)) == []


def test_expected_action_is_deterministic_from_deal_card() -> None:
    assert builder.expected_action_from_deal_card({"preconditions": {"p0_required": True}})["action"] == "handoff_manager"
    assert (
        builder.expected_action_from_deal_card(
            {
                "preconditions": {
                    "product_selected": True,
                    "price_confirmed": True,
                    "client_ready_to_pay": True,
                }
            }
        )["action"]
        == "send_payment_link"
    )
    assert (
        builder.expected_action_from_deal_card(
            {"preconditions": {"product_selected": True, "wants_trial": True}}
        )["action"]
        == "book_trial"
    )
    assert (
        builder.expected_action_from_deal_card(
            {"preconditions": {"lead_data_sufficient": True, "lead_captured": False}}
        )["action"]
        == "capture_lead"
    )
    assert builder.expected_action_from_deal_card({"preconditions": {}})["action"] == "answer_only"


def test_build_autonomy_personas_is_reproducible(tmp_path: Path) -> None:
    telegram_export = write_telegram_export(tmp_path)
    voice_dir = write_voice_transcripts(tmp_path)

    first = builder.build_autonomy_persona_rows(
        telegram_export=telegram_export,
        voice_transcripts_dir=voice_dir,
        limit=3,
        seed=11,
    )
    second = builder.build_autonomy_persona_rows(
        telegram_export=telegram_export,
        voice_transcripts_dir=voice_dir,
        limit=3,
        seed=11,
    )

    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(second, ensure_ascii=False, sort_keys=True)


def test_cli_writes_dynamic_sim_jsonl(tmp_path: Path) -> None:
    telegram_export = write_telegram_export(tmp_path)
    voice_dir = write_voice_transcripts(tmp_path)
    out = tmp_path / "autonomy_personas.jsonl"

    code = builder.main(
        [
            "--telegram-export",
            str(telegram_export),
            "--voice-transcripts-dir",
            str(voice_dir),
            "--out",
            str(out),
            "--limit",
            "2",
            "--seed",
            "3",
        ]
    )

    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows[:2]] == ["simulator_spec", "judge_spec"]
    assert all(row["type"] == "persona" for row in rows[2:])


def write_telegram_export(tmp_path: Path) -> Path:
    payload = {
        "chats": {
            "list": [
                {
                    "name": "Maria",
                    "type": "personal_chat",
                    "id": 1,
                    "messages": [
                        {
                            "id": 10,
                            "type": "message",
                            "from": "Мария Иванова",
                            "text": (
                                "Добрый день, меня зовут Мария Иванова, телефон 79001234567, "
                                "maria@example.com. Интересует летняя выездная школа в Менделеево "
                                "для 9 класса по физике, даты 17-27 июля, лицей НИУ ВШЭ."
                            ),
                        }
                    ],
                },
                {
                    "name": "Client",
                    "type": "personal_chat",
                    "id": 2,
                    "messages": [
                        {
                            "id": 20,
                            "type": "message",
                            "from": "Клиент",
                            "text": "Здравствуйте, можно ли пробное занятие для 8 класса по математике онлайн?",
                        }
                    ],
                },
                {
                    "name": "Dispute",
                    "type": "personal_chat",
                    "id": 3,
                    "messages": [
                        {
                            "id": 30,
                            "type": "message",
                            "from": "Клиент",
                            "text": "Здравствуйте, хочу возврат денег, ситуация спорная, нужен ответственный сотрудник.",
                        }
                    ],
                },
            ]
        }
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_voice_transcripts(tmp_path: Path) -> Path:
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    (voice_dir / "001__raw_phone__call.md").write_text(
        """
## Резюме
Клиент спрашивал про занятия на Сухаревской. Ограничения/возражения: цена 33 383 руб.,
лицей НИУ ВШЭ, преподаватель Петрова Мария, время в воскресенье не подходит.
## Следующий шаг
Отправить материалы
""",
        encoding="utf-8",
    )
    (voice_dir / "002__raw_phone__call.md").write_text(
        """
## Резюме
Клиент переживал, потянет ли ребёнок уровень, уточнял программу и кто ведёт занятия.
В разговоре встретился Скорняжный переулок, но его нельзя переносить.
## Следующий шаг
Проверить группу
""",
        encoding="utf-8",
    )
    return voice_dir
