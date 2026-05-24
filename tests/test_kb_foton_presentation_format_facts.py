from __future__ import annotations

from pathlib import Path

import yaml


SOURCE = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_sources/facts/facts_for_bot_FOTON.yaml")


def test_foton_presentation_format_facts_are_client_safe() -> None:
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    section = data["presentation_format_facts_2026_05_21"]
    client_facts = section["client_facts"]

    assert "lesson_load_by_age" in client_facts
    assert "subjects_by_class" in client_facts
    assert "online_technical_requirements" in client_facts
    assert "manager_only_facts" in section

    forbidden_client_fragments = (
        "УНПК",
        "edu@kmipt",
        "kmipt.tallanto",
        "Tallanto",
        "без возврата денег",
    )
    for fact in client_facts.values():
        text = str(fact.get("client_safe_text") or "")
        assert text
        for fragment in forbidden_client_fragments:
            assert fragment not in text


def test_foton_presentation_format_keeps_brand_domain_signal_internal() -> None:
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    internal = data["presentation_format_facts_2026_05_21"]["manager_only_facts"]

    assert internal["internal_only"] is True
    assert "edu@kmipt.ru" in internal["brand_domain_signal"]["client_safe_text"]
    assert "бота Фотон" in internal["brand_domain_signal"]["client_safe_text"]


def test_foton_regular_online_lesson_load_does_not_reuse_oge_intensive_cadence() -> None:
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    lesson_load = data["presentation_format_facts_2026_05_21"]["client_facts"]["lesson_load_by_age"]["client_safe_text"]

    assert "для 9 и 11 классов онлайн возможен формат" not in lesson_load
    assert "2 раза в неделю по 2 академических часа по будням" not in lesson_load
    assert "по будням" not in lesson_load
    assert data["intensives_2026"]["oge_foton"]["format"] == "онлайн, 2 занятия/нед после 18:00"
