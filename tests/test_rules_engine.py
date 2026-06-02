from __future__ import annotations

from mango_mvp.channels.rules_engine import MIGRATED, apply_rule, load_rules_registry, select_rule


def test_rules_registry_loads_approved_wave2b1_rules() -> None:
    registry = load_rules_registry()

    assert len(registry) == 16
    assert set(MIGRATED) == {"teacher", "recordings", "contact_address"}
    assert select_rule("teacher", registry).rule_id == "teacher"  # type: ignore[union-attr]
    assert select_rule("recording", registry).rule_id == "recordings"  # type: ignore[union-attr]
    assert select_rule("address", registry).rule_id == "contact_address"  # type: ignore[union-attr]
    assert select_rule("pricing", registry) is None


def test_rules_engine_teacher_uses_fact_and_does_not_invent_specific_name() -> None:
    registry = load_rules_registry()
    rule = registry["teacher"]
    facts = {
        "bot_policy.approved_phrases.theme_17_teachers.foton": (
            "Преподаватели — из МФТИ, МГУ, ВШЭ, МГТУ им. Баумана, МИФИ. Эксперты ЕГЭ и члены жюри олимпиад."
        )
    }

    outcome = apply_rule(
        rule,
        plan={"primary_intent": "teacher", "direct_question": "как зовут преподавателя физики в Лобне?", "active_brand": "foton"},
        facts=facts,
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert outcome.route == "draft_for_manager"
    assert "Менеджер уточнит" in outcome.text
    assert "МГУ" in outcome.text
    assert "Иван" not in outcome.text


def test_rules_engine_contact_address_uses_registry_foton_spelling() -> None:
    registry = load_rules_registry()
    rule = registry["contact_address"]

    outcome = apply_rule(
        rule,
        plan={"primary_intent": "address", "direct_question": "где очные занятия, адрес?", "active_brand": "foton"},
        facts={},
        context={"active_brand": "foton"},
    )

    assert outcome is not None
    assert outcome.route == "bot_answer_self_for_pilot"
    assert "Скорняжный" in outcome.text
    assert "УНПК" not in outcome.text
