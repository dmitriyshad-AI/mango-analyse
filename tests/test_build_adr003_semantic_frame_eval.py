import json
from pathlib import Path

from scripts import build_adr003_semantic_frame_eval as builder
from scripts import run_telegram_dynamic_client_sim as dynamic_sim


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_build_eval_writes_runner_compatible_m1_scenarios(tmp_path):
    whatif = tmp_path / "whatif_results.jsonl"
    out_dir = tmp_path / "sets"
    out_dir.mkdir()
    whatif_rows = []
    for index in range(1, 26):
        whatif_rows.append(
            {
                "idx": index,
                "status": "drafted",
                "source_event": "pair_missing",
                "brand": "foton" if index % 2 else "unpk",
                "channel": "telegram",
                "client_text": f"текущий вопрос {index} [phone] Ирина",
                "history_lines": [
                    "Клиент: записывали Ярослава на летнюю школу",
                    "Клиент: Фирсов и Буличев",
                    "Ответ: почту второго ребенка: Буличева Сергея",
                    f"Клиент: текущий вопрос {index} [phone] Ирина",
                    "Ответ: старый ответ",
                    "Ответ: это ИИ. Ольга",
                    "Ответ: автоответ после текущей реплики",
                    "Клиент: [fio] tsvetkova.vera@gmail…",
                ],
                "bot_route": "bot_answer_self_for_pilot",
                "bot_draft_text": "Безопасный ответ.",
                "safety_flags": ["manager_approval_required", "no_auto_send"],
                "context_used": ["direct_path"],
            }
        )
    _write_jsonl(whatif, whatif_rows)
    _write_jsonl(
        out_dir / "p0_stability_set_20260617.jsonl",
        [
            {
                "type": "persona",
                "dialog_id": "p0_persona",
                "brand": "foton",
                "persona": "P0",
                "behaviors": ["хочу возврат"],
                "max_turns": 1,
            }
        ],
    )
    _write_jsonl(
        out_dir / "forward_payment_personas_20260630.jsonl",
        [
            {"type": "simulator_spec", "rules": ["test"]},
            {"type": "judge_spec", "output_schema": {"verdict": "PASS|FAIL"}},
            {
                "type": "persona",
                "dialog_id": "forward_payment",
                "brand": "unpk",
                "persona": "оплата",
                "brand_forbidden": ["@unpk_mipt", "edu@kmipt.ru"],
                "behaviors": ["куда платить?"],
                "max_turns": 1,
            },
        ],
    )

    result = builder.build_eval(whatif_path=whatif, out_dir=out_dir, version="test")

    scenario_path = Path(result["m1_scenarios_path"])
    loaded = dynamic_sim.load_dynamic_sim_input(scenario_path)
    assert len(loaded.personas) == 27
    assert loaded.simulator_spec["type"] == "simulator_spec"
    assert loaded.judge_spec["type"] == "judge_spec"
    wappi = loaded.personas[0]
    assert wappi["dialog_id"] == "wappi_pair_missing_72h_001"
    assert wappi["scripted_behaviors"] == ["текущий вопрос 1 [phone] [fio]"]
    assert wappi["initial_history_lines"] == [
        "Клиент: записывали [fio] на летнюю школу",
        "Клиент: [fio]",
        "Ответ: почту второго ребенка: [fio]",
        "Ответ: старый ответ",
        "Ответ: это ИИ. [fio]",
        "Ответ: автоответ после текущей реплики",
        "Клиент: [fio] [contact]",
    ]
    assert wappi["expected_frame_status"] == "missing_manual_gold"
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["m1_scenarios"]["persona_count"] == 27
    assert manifest["m1_scenarios"]["source_breakdown"]["adr003_semantic_frame_wappi_latest25"] == 25
    assert manifest["m1_scenarios"]["source_breakdown"]["p0_stability_set_20260617"] == 1
    assert manifest["m1_scenarios"]["source_breakdown"]["forward_payment_personas_20260630"] == 1
    scenario_text = scenario_path.read_text(encoding="utf-8")
    eval_text = Path(result["eval_path"]).read_text(encoding="utf-8")
    assert "@" not in scenario_text
    assert "@" not in eval_text
    for raw_marker in ("Фирсов", "Буличев", "Ярослав", "Сергей", "Ирина", "Ольга"):
        assert raw_marker not in scenario_text
        assert raw_marker not in eval_text
    wappi_cases = [json.loads(line) for line in eval_text.splitlines()]
    assert wappi_cases[0]["client_text"] == "текущий вопрос 1 [phone] [fio]"
    assert not any(
        wappi_cases[0]["client_text"] in line for line in wappi_cases[0]["history_lines"]
    )
