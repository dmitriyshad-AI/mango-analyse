import yaml

from mango_mvp.channels.draft_prompt_builder import build_draft_prompt
from mango_mvp.channels.few_shot_reference import build_few_shot_reference
from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot


def _write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_few_shot_reference_uses_no_fact_example_when_price_fact_missing(tmp_path):
    warm = tmp_path / "warm.yaml"
    advanced = tmp_path / "advanced.yaml"
    _write_yaml(
        warm,
        {
            "unpk": {
                "01_pricing_with_validity": [
                    {"client": "сколько стоит онлайн", "answer": "Онлайн стоит 41 800 ₽."}
                ]
            }
        },
    )
    _write_yaml(
        advanced,
        {
            "no_fact_examples": [
                {
                    "brand": "unpk",
                    "topic_id": "01_pricing_with_validity (online)",
                    "teaches": "graceful_unknown",
                    "client": "почём онлайн?",
                    "answer": "Точную цену подтвердит менеджер, по формату занятия идут вживую и в записи.",
                }
            ],
            "bad_good_pairs": [
                {
                    "flag": "fabricated_specific",
                    "brand": "unpk",
                    "bad": "Онлайн стоит 41 800 ₽.",
                    "good": "Точную цену подтвердит менеджер.",
                    "why": "Не называем неподтверждённую цену.",
                }
            ],
        },
    )

    reference = build_few_shot_reference(
        message_text="сколько стоит онлайн 5 класс математика",
        active_brand="unpk",
        missing_facts=("prices.current",),
        warm_path=warm,
        advanced_path=advanced,
    )

    joined_style = "\n".join(reference["style_examples"])
    assert "Точную цену подтвердит менеджер" in joined_style
    assert "41 800" not in joined_style
    assert any("fabricated_specific" in item for item in reference["correction_examples"])


def test_few_shot_reference_uses_brand_warm_example_only_with_confirmed_fact(tmp_path):
    warm = tmp_path / "warm.yaml"
    advanced = tmp_path / "advanced.yaml"
    _write_yaml(
        warm,
        {
            "foton": {
                "01_pricing_with_validity": [
                    {"client": "сколько стоит очно", "answer": "Очно сейчас: семестр — 44 600 ₽, год — 74 500 ₽."}
                ]
            }
        },
    )
    _write_yaml(advanced, {"bad_good_pairs": []})

    reference = build_few_shot_reference(
        message_text="сколько стоит очно 7 класс",
        active_brand="foton",
        confirmed_facts={"fact:price": "Фотон очно 5-11: 44 600 ₽ / 74 500 ₽"},
        warm_path=warm,
        advanced_path=advanced,
    )

    assert "44 600" in "\n".join(reference["style_examples"])


def test_pilot_context_includes_few_shot_reference(monkeypatch, tmp_path):
    warm = tmp_path / "warm.yaml"
    advanced = tmp_path / "advanced.yaml"
    _write_yaml(
        warm,
        {
            "foton": {
                "01_pricing_with_validity": [
                    {"client": "сколько стоит очно", "answer": "Очно сейчас: семестр — 44 600 ₽, год — 74 500 ₽."}
                ]
            }
        },
    )
    _write_yaml(
        advanced,
        {
            "style_phrases_no_facts": {"warm_openers": ["Поняла вас, сейчас сориентирую."]},
            "bad_good_pairs": [
                {
                    "flag": "over_handoff",
                    "brand": "foton",
                    "bad": "Менеджер подскажет.",
                    "good": "Очно сейчас: семестр — 44 600 ₽, год — 74 500 ₽.",
                    "why": "Факт есть.",
                }
            ],
        },
    )
    monkeypatch.setenv("MANGO_TELEGRAM_FEW_SHOT_WARM_PATH", str(warm))
    monkeypatch.setenv("MANGO_TELEGRAM_FEW_SHOT_ADVANCED_PATH", str(advanced))

    snapshot = {
        "schema_version": "test",
        "facts": [
                {
                    "fact_id": "fact:price",
                    "brand": "foton",
                    "fact_types": ["price"],
                    "client_safe_text": "Фотон очно 5-11: семестр — 44 600 ₽, год — 74 500 ₽.",
                    "usable_for_precise_answer": True,
                    "allowed_for_client_answer": True,
                    "freshness_status": "fresh",
                }
        ],
        "chunks": [],
        "sources": [],
    }

    context = build_telegram_pilot_context_from_snapshot(
        "сколько стоит очно 7 класс?",
        kc_snapshot=snapshot,
        active_brand="foton",
    ).to_prompt_context()

    assert context["few_shot_style_examples"]
    assert "44 600" in "\n".join(context["few_shot_style_examples"])
    prompt = build_draft_prompt("сколько стоит очно 7 класс?", context=context)
    assert "few_shot_style_examples" in prompt
    assert "Few-shot примеры НЕ являются источником фактов" in prompt


def test_few_shot_reference_uses_known_slots_for_reasked_known_corrections(tmp_path):
    warm = tmp_path / "warm.yaml"
    advanced = tmp_path / "advanced.yaml"
    _write_yaml(warm, {"foton": {"01_pricing_with_validity": []}})
    _write_yaml(
        advanced,
        {
            "bad_good_pairs": [
                {
                    "flag": "reasked_known",
                    "brand": "foton",
                    "bad": "Напишите класс и предмет.",
                    "good": "Класс и предмет уже вижу, отвечу по физике для 8 класса.",
                    "why": "Не переспрашиваем известное.",
                }
            ],
        },
    )

    reference = build_few_shot_reference(
        message_text="это цена на сейчас?",
        active_brand="foton",
        topic_id="theme:001_pricing",
        confirmed_facts={"fact:price": "Фотон 8 класс физика онлайн: 74 500 ₽"},
        known_slots={"grade": "8", "subject": "физика"},
        warm_path=warm,
        advanced_path=advanced,
    )

    assert any("reasked_known" in item for item in reference["correction_examples"])
