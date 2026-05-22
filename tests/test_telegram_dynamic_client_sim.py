import json
from pathlib import Path

import pytest

from scripts import run_telegram_dynamic_client_sim as sim


class _FakePilotContext:
    def to_prompt_context(self):
        return {
            "active_brand": "unpk",
            "recent_messages": [],
            "facts_context": {},
            "confirmed_facts": {"fact:platform": "УНПК: онлайн-платформа — МТС Линк."},
            "knowledge_snippets": ["online platform: УНПК: онлайн-платформа — МТС Линк."],
            "missing_facts": [],
            "context_warnings": [],
        }


def _write_scenarios(path: Path) -> None:
    rows = [
        {
            "type": "simulator_spec",
            "instructions": "Клиент отвечает коротко.",
        },
        {
            "type": "judge_spec",
            "output_schema": {
                "dialog_id": "string",
                "brand": "unpk",
                "verdict": "PASS|PASS_WITH_NOTES|FAIL",
            },
        },
        {
            "type": "persona",
            "dialog_id": "v7_unpk_test",
            "brand": "unpk",
            "persona": "занятый родитель",
            "goal": "проверить удержание контекста",
            "max_turns": 3,
        },
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def test_load_dynamic_sim_input_counts_rows(tmp_path):
    path = tmp_path / "v7.jsonl"
    _write_scenarios(path)

    loaded = sim.load_dynamic_sim_input(path)

    assert loaded.simulator_spec["type"] == "simulator_spec"
    assert loaded.judge_spec["type"] == "judge_spec"
    assert len(loaded.personas) == 1
    assert loaded.personas[0]["dialog_id"] == "v7_unpk_test"


def test_known_dialog_fields_use_only_client_messages():
    fields = sim.known_dialog_fields_from_client_messages(
        [
            "Ответ: Напишите класс и предмет: 11 класс, физика.",
            "Клиент: 9 класс, физика, онлайн",
            "Ответ: Подойдёт очный формат.",
        ],
        active_brand="unpk",
    )

    assert fields["grade"] == "9"
    assert fields["subject"] == "физика"
    assert fields["format"] == "онлайн"
    assert fields["active_brand"] == "unpk"


def test_judge_prompt_marks_metadata_as_internal():
    prompt = sim.build_judge_prompt(
        {"output_schema": {"verdict": "PASS|FAIL"}},
        {"dialog_id": "v7", "brand": "foton"},
        [
            {
                "turn": 1,
                "client_message": "Здравствуйте",
                "bot_text": "Здравствуйте! Помогу подобрать курс.",
                "bot_route": "draft_for_manager",
                "bot_topic_id": "theme:016_program",
                "bot_safety_flags": ["manager_approval_required"],
                "bot_manager_checklist": ["Проверить группу"],
                "bot_missing_facts": ["расписание"],
                "bot_confirmed_facts": ["fact:format: Фотон: онлайн — МТС Линк."],
                "bot_knowledge_snippets": ["format: Фотон: онлайн — МТС Линк."],
            }
        ],
    )

    assert "клиент их НЕ видел" in prompt
    assert "Не считай эти внутренние метаданные раскрытием" in prompt
    assert "Не ставь fabrication за факт" in prompt
    assert "fact:format" in prompt
    assert "Клиент видел ответ бота" in prompt


def test_fake_run_writes_full_transcripts_and_review_queue(tmp_path, monkeypatch):
    path = tmp_path / "v7.jsonl"
    out_dir = tmp_path / "out"
    _write_scenarios(path)
    monkeypatch.setattr(sim, "build_telegram_pilot_context_from_snapshot", lambda *args, **kwargs: _FakePilotContext())

    rc = sim.main(
        [
            "--scenarios",
            str(path),
            "--snapshot",
            str(tmp_path / "snapshot.json"),
            "--out-dir",
            str(out_dir),
            "--client-mode",
            "fake",
            "--judge-mode",
            "fake",
            "--bot-mode",
            "fake",
            "--limit",
            "1",
        ]
    )

    assert rc == 0
    assert (out_dir / "dynamic_dialog_transcripts.jsonl").exists()
    assert (out_dir / "dynamic_judge_results.jsonl").exists()
    assert (out_dir / "dynamic_turns.csv").exists()
    assert (out_dir / "human_review_queue.csv").exists()
    assert (out_dir / "full_transcripts.md").exists()
    assert (out_dir / "transcripts_md" / "v7_unpk_test.md").exists()

    transcript = (out_dir / "full_transcripts.md").read_text(encoding="utf-8")
    assert "**Клиент:**" in transcript
    assert "**Бот:**" in transcript
    assert "route: `draft_for_manager`" in transcript
    assert "safety_flags:" in transcript
    assert "Soft flags:" in transcript
    assert "confirmed_facts_for_judge:" in transcript
    assert "knowledge_snippets_for_judge:" in transcript

    queue = (out_dir / "human_review_queue.csv").read_text(encoding="utf-8")
    assert "manual_check_hint" in queue
    assert "v7_unpk_test" in queue
    assert "assumed_unstated_need" in queue


def test_refuses_stable_runtime_output(tmp_path):
    with pytest.raises(ValueError, match="stable_runtime"):
        sim.main(["--out-dir", str(tmp_path / "stable_runtime" / "dynamic_sim")])
