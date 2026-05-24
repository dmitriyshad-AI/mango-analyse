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
            "answer_quality_reference": {"purpose": "test"},
            "few_shot_style_examples": ["Клиент: тест\nОбразец: тест"],
            "few_shot_correction_examples": ["Флаг: ignored_question"],
            "conversation_intent_plan": {
                "schema_version": "conversation_intent_plan_v1_2026_05_23",
                "primary_intent": "pricing",
                "topic_id": "theme:001_pricing",
                "direct_question": "Сколько стоит онлайн?",
                "topic_switch_decision": "continue",
                "known_slots": {"grade": "9", "subject": "физика", "format": "онлайн"},
            },
            "answer_contract": {
                "primary_intent": "pricing",
                "direct_question": "Сколько стоит онлайн?",
                "must_answer_first": True,
                "p0_required": False,
            },
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
                "bot_confirmed_facts": [
                    "verified_safe_template: Фотон: старая безопасная подмена.",
                    "fact:format: Фотон: онлайн — МТС Линк.",
                ],
                "bot_knowledge_snippets": ["format: Фотон: онлайн — МТС Линк."],
            }
        ],
    )

    assert "клиент их НЕ видел" in prompt
    assert "Не считай эти внутренние метаданные раскрытием" in prompt
    assert "цифровой помощник Фотона/УНПК МФТИ/центра" in prompt
    assert "GPT/Claude/Codex/OpenAI" in prompt
    assert "Не ставь fabrication за факт" in prompt
    assert "fact:format" in prompt
    assert "verified_safe_template: Фотон" not in prompt
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


def test_codex_bot_mode_requires_existing_snapshot(tmp_path):
    path = tmp_path / "v7.jsonl"
    _write_scenarios(path)

    with pytest.raises(FileNotFoundError):
        sim.main(
            [
                "--scenarios",
                str(path),
                "--snapshot",
                str(tmp_path / "missing_snapshot.json"),
                "--out-dir",
                str(tmp_path / "out"),
                "--client-mode",
                "fake",
                "--judge-mode",
                "fake",
                "--bot-mode",
                "codex",
                "--limit",
                "1",
            ]
        )


def test_dynamic_context_parity_includes_known_slots_funnel_and_few_shot(monkeypatch, tmp_path):
    monkeypatch.setattr(sim, "build_telegram_pilot_context_from_snapshot", lambda *args, **kwargs: _FakePilotContext())

    context = sim.build_bot_prompt_context(
        "Это цена на сейчас?",
        persona={"dialog_id": "ctx", "brand": "unpk", "persona": "родитель"},
        recent_messages=["Клиент: 9 класс, физика, онлайн", "Ответ: Подберём вариант."],
        snapshot_path=tmp_path / "snapshot.json",
    )

    assert context["context_parity_checked"] is True
    assert "funnel_state" in context
    assert context["known_slots"]["grade"] == "9"
    assert context["known_slots"]["subject"] == "физика"
    assert context["known_slots"]["format"] in {"онлайн", "online"}
    assert "known_dialog_fields" in context
    assert "dialogue_memory_view" in context
    assert context["dialogue_memory_view"]["known_slots"]["grade"] == "9"
    assert context["dialogue_memory_view"]["open_question"]["kind"] == "price_fix"
    assert "answer_quality_reference" in context
    assert "conversation_intent_plan" in context
    assert context["conversation_intent_plan"]["primary_intent"] == "pricing"


def test_dynamic_summary_counts_answer_first_known_multitopic_and_price_fix_findings(tmp_path):
    transcripts = [
        {
            "dialog_id": "quality_cases",
            "brand": "foton",
            "turns": [
                {
                    "turn": 1,
                    "bot_answer_quality_findings": [
                        "ignored_direct_question",
                        "reasked_known_grade",
                        "reasked_known_subject",
                        "reasked_known_format",
                        "single_topic_answer_to_multitopic_question",
                    ],
                    "bot_answer_quality_rewritten": False,
                    "context_parity_checked": True,
                },
                {
                    "turn": 2,
                    "bot_answer_quality_findings": [],
                    "bot_answer_quality_rewritten": True,
                    "context_parity_checked": True,
                },
            ],
        }
    ]
    judge_results = [
        {
            "dialog_id": "quality_cases",
            "brand": "foton",
            "hard_gates_passed": True,
            "soft_flags_present": ["ignored_question", "asked_known_data_again"],
            "verdict": "PASS_WITH_NOTES",
            "human_tone_score_0_100": 72,
        }
    ]

    summary = sim.build_summary(
        transcripts,
        judge_results,
        scenario_path=tmp_path / "scenarios.jsonl",
        snapshot_path=tmp_path / "snapshot.json",
        parallel=1,
    )

    findings = summary["answer_quality"]["finding_counts"]
    assert findings["ignored_direct_question"] == 1
    assert findings["reasked_known_grade"] == 1
    assert findings["reasked_known_subject"] == 1
    assert findings["reasked_known_format"] == 1
    assert findings["single_topic_answer_to_multitopic_question"] == 1
    assert summary["answer_quality"]["rewritten_turns"] == 1
    assert summary["answer_quality"]["context_parity_checked"] is True
    assert summary["soft_flags"]["ignored_question"] == 1


def test_fake_parallel_run_writes_all_dialogs(tmp_path, monkeypatch):
    path = tmp_path / "v7.jsonl"
    out_dir = tmp_path / "out_parallel"
    rows = [
        {"type": "simulator_spec", "instructions": "Клиент отвечает коротко."},
        {"type": "judge_spec", "output_schema": {"verdict": "PASS|PASS_WITH_NOTES|FAIL"}},
        {"type": "persona", "dialog_id": "v7_unpk_a", "brand": "unpk", "persona": "родитель A", "goal": "подбор", "max_turns": 2},
        {"type": "persona", "dialog_id": "v7_unpk_b", "brand": "unpk", "persona": "родитель B", "goal": "подбор", "max_turns": 2},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
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
            "--parallel",
            "2",
        ]
    )

    assert rc == 0
    dialogs = [
        json.loads(line)
        for line in (out_dir / "dynamic_dialog_transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [dialog["dialog_id"] for dialog in dialogs] == ["v7_unpk_a", "v7_unpk_b"]
    summary = json.loads((out_dir / "dynamic_summary.json").read_text(encoding="utf-8"))
    assert summary["run_config"]["parallel"] == 2
    assert summary["totals"]["dialogs"] == 2


def test_resume_only_failed_reruns_failed_dialogs_and_keeps_completed(tmp_path, monkeypatch):
    path = tmp_path / "v7.jsonl"
    out_dir = tmp_path / "out_resume"
    rows = [
        {"type": "simulator_spec", "instructions": "Клиент отвечает коротко."},
        {"type": "judge_spec", "output_schema": {"verdict": "PASS|PASS_WITH_NOTES|FAIL"}},
        {"type": "persona", "dialog_id": "ok", "brand": "unpk", "persona": "родитель A", "goal": "подбор", "max_turns": 1},
        {"type": "persona", "dialog_id": "bad", "brand": "unpk", "persona": "родитель B", "goal": "подбор", "max_turns": 1},
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    out_dir.mkdir()
    existing = [
        {"dialog_id": "ok", "brand": "unpk", "turns": [{"turn": 1}], "run_status": "completed", "judge_result": {"verdict": "PASS"}},
        {"dialog_id": "bad", "brand": "unpk", "turns": [], "run_status": "infra_error", "judge_result": {"verdict": "FAIL"}},
    ]
    (out_dir / "dynamic_dialog_transcripts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in existing),
        encoding="utf-8",
    )
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
            "--resume",
            "--only-failed",
        ]
    )

    assert rc == 0
    dialogs = [
        json.loads(line)
        for line in (out_dir / "dynamic_dialog_transcripts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [dialog["dialog_id"] for dialog in dialogs] == ["ok", "bad"]
    assert dialogs[0]["judge_result"]["verdict"] == "PASS"
    assert dialogs[1]["run_status"] == "completed"


def test_build_infra_error_dialog_marks_timeout_for_resumable_runner() -> None:
    dialog = sim.build_infra_error_dialog(
        {"dialog_id": "timeout_case", "brand": "foton"},
        TimeoutError("client timeout"),
        elapsed_seconds=12.3,
    )

    assert dialog["run_status"] == "timeout"
    assert dialog["judge_result"]["verdict"] == "FAIL"
    assert dialog["judge_result"]["violated_gates"] == ["timeout"]


def test_refuses_stable_runtime_output(tmp_path):
    with pytest.raises(ValueError, match="stable_runtime"):
        sim.main(["--out-dir", str(tmp_path / "stable_runtime" / "dynamic_sim")])
