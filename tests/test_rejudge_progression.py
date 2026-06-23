from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts import rejudge_progression


class StaticModel:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> Mapping[str, Any]:
        self.prompts.append(prompt)
        if not self.payloads:
            return {}
        return dict(self.payloads.pop(0))


def _persona(dialog_id: str = "d1", *, stage_start: str = "S1", stage_target: str = "S2", deal_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "persona",
        "dialog_id": dialog_id,
        "brand": "foton",
        "held_facts": {"grade": "7"},
        "progression_tags": {
            "stage_start": stage_start,
            "stage_target": stage_target,
            "deal_state": dict(deal_state or {}),
        },
    }


def _dialog(persona: Mapping[str, Any], *, bot_text: str = "Могу предложить группу, напишите ФИО для записи.") -> dict[str, Any]:
    return {
        "dialog_id": persona["dialog_id"],
        "brand": persona["brand"],
        "persona": dict(persona),
        "turns": [
            {
                "turn": 1,
                "client_message": "Здравствуйте",
                "bot_text": bot_text,
                "bot_route": "bot_answer_self_for_pilot",
            }
        ],
    }


def test_model_only_observes_code_sets_stage_and_ignores_extra_stage() -> None:
    persona = _persona(stage_start="S1", stage_target="S5")
    model = StaticModel(
        {
            "requested_enrollment_data": True,
            "stage_reached": "S8",
            "dialog_verdict": "false_push",
        }
    )

    row = rejudge_progression.assess_dialog(judge_model=model, dialog=_dialog(persona), persona_by_id={})

    assert row["stage_reached"] == "S5"
    assert row["dialog_verdict"] == "advanced"
    assert row["turn_observations"] == [
        {field: (field == "requested_enrollment_data") for field in rejudge_progression.OBSERVATION_FIELDS}
    ]
    assert "stage_reached" not in row["turn_observations"][0]
    assert "stage_target в персоне" in model.prompts[0]
    assert "не подгоняй наблюдения под ожидание" in model.prompts[0].casefold()


def test_p0_handoff_with_data_collection_is_mis_routed() -> None:
    persona = _persona(stage_start="P0", stage_target="route", deal_state={"p0": "complaint"})
    dialog = _dialog(persona, bot_text="Передам менеджеру, а пока уточните класс ребёнка.")
    dialog["turns"][0]["client_message"] = "Жалоба на преподавателя"
    model = StaticModel({"handed_off_to_manager": True, "asked_missing_key": True})

    row = rejudge_progression.assess_dialog(judge_model=model, dialog=dialog, persona_by_id={})

    assert row["stage_start"] == "P0"
    assert row["dialog_verdict"] == "mis_routed"
    assert row["business_errors"] == ["p0_mishandled_collect_first"]
    assert row["next_step_each_turn"] == [False]


def test_p0_detector_does_not_match_obsudim() -> None:
    persona = _persona(stage_start="S3", stage_target="S5")
    turn = {"client_message": "Хорошо, спасибо, тогда обсудим дома и я напишу данные чуть позже"}

    assert not rejudge_progression._is_p0_turn(persona, turn, stage_start="S3", stage_target="S5")


def test_route_target_success_is_held_ok() -> None:
    persona = _persona(stage_start="S8", stage_target="route", deal_state={"existing_customer": True})
    model = StaticModel({"handed_off_to_manager": True})

    row = rejudge_progression.assess_dialog(
        judge_model=model,
        dialog=_dialog(persona, bot_text="Передам запрос бухгалтерии, менеджер вернётся с ответом."),
        persona_by_id={},
    )

    assert row["dialog_verdict"] == "held_ok"
    assert row["next_step_each_turn"] == [True]
    assert row["business_errors"] == []


def test_unneeded_manager_handoff_is_mis_routed() -> None:
    persona = _persona(stage_start="S1", stage_target="S2")
    model = StaticModel({"handed_off_to_manager": True})

    row = rejudge_progression.assess_dialog(
        judge_model=model,
        dialog=_dialog(persona, bot_text="Передам менеджеру, он подскажет по группе."),
        persona_by_id={},
    )

    assert row["dialog_verdict"] == "mis_routed"
    assert row["business_errors"] == ["over_handoff_service"]


def test_contentful_draft_is_not_manager_handoff_even_when_route_is_draft() -> None:
    persona = _persona(stage_start="S1", stage_target="hold")
    dialog = _dialog(
        persona,
        bot_text=(
            "Очно физика для 9 класса стоит 44 600 ₽ за семестр. "
            "По физике 9 класс очно есть базовая и продвинутая группы; менеджер поможет подобрать расписание."
        ),
    )
    dialog["turns"][0]["bot_route"] = "draft_for_manager"

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=dialog,
        persona_by_id={},
        turn_observations=[{"handed_off_to_manager": True, "gave_conditions": True, "named_concrete_offer": True}],
    )

    assert row["dialog_verdict"] == "held_ok"
    assert row["business_errors"] == []
    assert row["turn_observations"][0]["handed_off_to_manager"] is False
    assert row["turn_observations"][0]["named_concrete_option"] is True


def test_route_target_with_content_answer_is_under_handoff() -> None:
    persona = _persona(stage_start="S8", stage_target="route", deal_state={"existing_customer": True})

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=_dialog(
            persona,
            bot_text=(
                "Закрывающие документы можно запросить через менеджера, а доступ к документам появится "
                "после подтверждения оплаты."
            ),
        ),
        persona_by_id={},
        turn_observations=[{"handed_off_to_manager": True, "confirmed_access_or_docs": True}],
    )

    assert row["dialog_verdict"] == "mis_routed"
    assert row["business_errors"] == ["under_handoff_service"]
    assert row["turn_observations"][0]["handed_off_to_manager"] is False


def test_route_target_followup_after_completed_handoff_is_not_under_handoff() -> None:
    persona = _persona(stage_start="S8", stage_target="route", deal_state={"existing_customer": True})
    dialog = _dialog(persona)
    dialog["turns"] = [
        {
            "turn": 1,
            "client_message": "Передайте бухгалтерии запрос по закрывающим документам.",
            "bot_text": "Передам запрос бухгалтерии, менеджер вернётся с ответом.",
            "bot_route": "draft_for_manager",
        },
        {
            "turn": 2,
            "client_message": "Номер платежки найду и пришлю следом.",
            "bot_text": "Хорошо, пришлите номер платёжного поручения сюда, когда найдёте.",
            "bot_route": "bot_answer_self_for_pilot",
        },
    ]

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=dialog,
        persona_by_id={},
        turn_observations=[{"handed_off_to_manager": True}, {"left_dated_followup": True}],
    )

    assert row["dialog_verdict"] == "held_ok"
    assert row["business_errors"] == []
    assert row["turn_verdicts"] == ["held_ok", "held_ok"]


def test_progression_prompt_does_not_show_service_route() -> None:
    persona = _persona()
    dialog = _dialog(persona)

    prompt = rejudge_progression.build_progression_prompt(persona=persona, turns=dialog["turns"], current_turn_index=0)

    assert "Финальный маршрут бота" not in prompt


def test_summary_counts_next_step_and_business_errors() -> None:
    rows = [
        {
            "dialog_verdict": "advanced",
            "next_step_each_turn": [True, False],
            "turn_move_quality": ["winning", "weak"],
            "business_errors": [],
        },
        {
            "dialog_verdict": "mis_routed",
            "next_step_each_turn": [False],
            "turn_move_quality": ["wrong"],
            "business_errors": ["over_handoff_service"],
        },
    ]

    summary = rejudge_progression.summarize_results(rows, llm_calls={"progression_judge_fake": 3})

    assert summary["advanced_or_held_ok_rate"] == 0.5
    assert summary["false_push_or_mis_routed_rate"] == 0.5
    assert summary["valid_next_step_turn_rate"] == 0.3333
    assert summary["winning_move_rate"] == 0.3333
    assert summary["advanced_or_held_with_winning_move_rate_lt_0_5"] == 0
    assert summary["business_errors"] == {"over_handoff_service": 1}
    assert summary["llm_calls"]["progression_judge_fake"] == 3


def test_dialog_business_errors_drops_stale_target_not_reached() -> None:
    assessment = rejudge_progression.TurnAssessment(
        observation={},
        stage_reached="S3",
        turn_verdict="stalled",
        move_quality="weak",
        move_criteria_hit=("substantive_but_incomplete",),
        next_step=False,
        business_errors=("target_not_reached", "no_valid_next_step"),
        note="",
    )

    errors = rejudge_progression._dialog_business_errors(
        stage_reached="S5",
        stage_target="S5",
        assessments=[assessment],
    )

    assert errors == ["no_valid_next_step"]


def test_assess_dialog_can_reuse_stored_observations_without_model() -> None:
    persona = _persona(stage_start="S1", stage_target="S5")

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=_dialog(persona),
        persona_by_id={},
        turn_observations=[{"requested_enrollment_data": True}],
    )

    assert row["stage_reached"] == "S5"
    assert row["dialog_verdict"] == "advanced"


def test_cli_fake_mode_writes_results(tmp_path: Path) -> None:
    scenarios = tmp_path / "seed.jsonl"
    transcripts = tmp_path / "dynamic_dialog_transcripts.jsonl"
    out = tmp_path / "progression_results.jsonl"
    summary = tmp_path / "progression_summary.json"

    persona = _persona()
    scenarios.write_text(
        "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in (
                {"type": "simulator_spec", "rules": []},
                {"type": "judge_spec", "output_schema": {}},
                persona,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    transcripts.write_text(json.dumps(_dialog(persona), ensure_ascii=False) + "\n", encoding="utf-8")

    rc = rejudge_progression.main(
        [
            "--transcripts",
            str(transcripts),
            "--scenarios",
            str(scenarios),
            "--judge-mode",
            "fake",
            "--out",
            str(out),
            "--summary-out",
            str(summary),
        ]
    )

    assert rc == 0
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    saved_summary = json.loads(summary.read_text(encoding="utf-8"))
    assert row["dialog_id"] == "d1"
    assert "turn_observations" in row
    assert saved_summary["dialogs"] == 1


def test_legacy_named_concrete_offer_aliases_to_option() -> None:
    observation = rejudge_progression.normalize_observations({"named_concrete_offer": True}, bot_text="")

    assert observation["named_concrete_option"] is True
    assert "named_concrete_offer" not in observation


def test_quality_only_observations_do_not_advance_stage() -> None:
    persona = _persona(stage_start="S1", stage_target="S2")

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=_dialog(persona, bot_text="Вы уже писали класс выше, у нас есть разные курсы."),
        persona_by_id={},
        turn_observations=[{"gave_fit_reason": True, "reasked_known": True, "dumped_catalog": True}],
    )

    assert row["stage_reached"] == "S1"
    assert row["dialog_verdict"] == "stalled"
    assert row["turn_move_quality"] == ["weak"]
    assert row["move_criteria_hit"] == [["reasked_known"]]


def test_fact_audit_fabrication_overrides_winning_observation() -> None:
    persona = _persona(stage_start="S2", stage_target="S3")
    dialog = _dialog(persona, bot_text="Цена 57 000 ₽, можно оформить рассрочку.")
    dialog["turns"][0]["number_audit"] = {
        "items": [{"level": "no_match", "number": "57000"}],
        "has_risky_number": True,
    }

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=dialog,
        persona_by_id={},
        turn_observations=[{"gave_conditions": True, "named_concrete_option": True}],
    )

    assert row["dialog_verdict"] == "wrong_move"
    assert row["turn_verdicts"] == ["wrong_move"]
    assert row["turn_move_quality"] == ["wrong"]
    assert row["business_errors"] == ["fabrication_in_move"]


def test_wrong_scope_fact_audit_marks_wrong_venue_or_fact() -> None:
    persona = _persona(stage_start="S2", stage_target="S3")
    dialog = _dialog(persona, bot_text="Очные занятия проходят в Менделеево.")
    dialog["turns"][0]["judge_fact_audit"] = {
        "items": [{"level": "wrong_scope", "claim": "Менделеево"}],
    }

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=dialog,
        persona_by_id={},
        turn_observations=[{"gave_conditions": True}],
    )

    assert row["dialog_verdict"] == "wrong_move"
    assert row["business_errors"] == ["fabrication_in_move", "wrong_venue_or_fact"]


def test_redrive_after_pay_triggers_even_with_service_answer() -> None:
    persona = _persona(stage_start="S7", stage_target="S8", deal_state={"paid_claimed": True})

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=_dialog(persona, bot_text="Доступ придёт после оплаты, можем сразу оформить следующий курс."),
        persona_by_id={},
        turn_observations=[{"confirmed_access_or_docs": True, "pushed_sale": True}],
    )

    assert row["dialog_verdict"] == "false_push"
    assert row["business_errors"] == ["redrive_after_pay"]
    assert row["turn_move_quality"] == ["wrong"]


def test_service_not_resell_is_computed_in_code_not_llm_observation() -> None:
    persona = _persona(stage_start="S7", stage_target="S8", deal_state={"paid_claimed": True})

    row = rejudge_progression.assess_dialog(
        judge_model=None,
        dialog=_dialog(persona, bot_text="После оплаты доступ и документы придут на почту."),
        persona_by_id={},
        turn_observations=[{"confirmed_access_or_docs": True, "service_not_resell": False}],
    )

    assert row["dialog_verdict"] == "advanced"
    assert row["turn_move_quality"] == ["winning"]
    assert row["move_criteria_hit"] == [["service_not_resell"]]
