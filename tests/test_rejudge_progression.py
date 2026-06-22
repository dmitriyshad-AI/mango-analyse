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

    row = rejudge_progression.assess_dialog(judge_model=model, dialog=_dialog(persona), persona_by_id={})

    assert row["dialog_verdict"] == "held_ok"
    assert row["next_step_each_turn"] == [True]
    assert row["business_errors"] == []


def test_unneeded_manager_handoff_is_mis_routed() -> None:
    persona = _persona(stage_start="S1", stage_target="S2")
    model = StaticModel({"handed_off_to_manager": True})

    row = rejudge_progression.assess_dialog(judge_model=model, dialog=_dialog(persona), persona_by_id={})

    assert row["dialog_verdict"] == "mis_routed"
    assert row["business_errors"] == ["over_handoff_service"]


def test_summary_counts_next_step_and_business_errors() -> None:
    rows = [
        {"dialog_verdict": "advanced", "next_step_each_turn": [True, False], "business_errors": []},
        {"dialog_verdict": "mis_routed", "next_step_each_turn": [False], "business_errors": ["over_handoff_service"]},
    ]

    summary = rejudge_progression.summarize_results(rows)

    assert summary["advanced_or_held_ok_rate"] == 0.5
    assert summary["false_push_or_mis_routed_rate"] == 0.5
    assert summary["valid_next_step_turn_rate"] == 0.3333
    assert summary["business_errors"] == {"over_handoff_service": 1}


def test_dialog_business_errors_drops_stale_target_not_reached() -> None:
    assessment = rejudge_progression.TurnAssessment(
        observation={},
        stage_reached="S3",
        turn_verdict="stalled",
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
