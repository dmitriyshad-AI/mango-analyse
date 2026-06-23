# Sales Exam Progression Judge v2 Report

Дата: 2026-06-23

Ветка: `codex/sales-exam-progression-v2`
База: `codex/progression-judge` (`639740c`)

## Что сделано

- Доработан `scripts/rejudge_progression.py` до v2:
  - атомарные наблюдения вместо склеенных;
  - `service_not_resell` считается кодом, не LLM-наблюдением;
  - `judge_fact_audit` / `number_audit` перебивают ход в `wrong_move` + `fabrication_in_move`;
  - добавлена отдельная ось качества хода `turn_move_quality`;
  - добавлены `move_criteria_hit`, `winning_move_rate`, `business_errors`, `llm_calls`;
  - P0-handoff не засчитывается как продажный next step.
- Обновлены тесты `tests/test_rejudge_progression.py`.
- Создан набор 16 персон:
  - `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-23_NABOR_ekzamen_prodazhi.jsonl`
  - sha256: `317824f4d88e81df1d62c2e821e0efd3ed32662d3621c3d062761947215a0772`
- Создан smoke-набор seed+16:
  - `/Users/dmitrijfabarisov/Claude Projects/Foton/2026-06-23_NABOR_ekzamen_prodazhi_seed_plus_seedv1.jsonl`
  - sha256: `f0456939deea7077f82fa12d59ebeb0fa9ffa4a7c2b4c53fc3f5c5f3f5502bcc`

## Проверка набора

- Персон: 16
- Фотон / УНПК: 8 / 8
- POS / NEG: 8 / 8
- won / lost: 8 / 8
- `stage_target=S7`: 0
- 8 пар, в каждой по 2 кейса.

## Smoke

Папка:

`/Users/dmitrijfabarisov/Projects/Mango_sales_exam_progression_v2/runs/progression_seed16_fake_20260623_201137`

Ключевые файлы:

- `dynamic_dialog_transcripts.jsonl`
- `progression_results.jsonl`
- `progression_summary.json`
- `progression_summary.md`

Итог smoke:

- dialogs: 27
- turns: 81
- `llm_calls.total`: 81
- `llm_calls.progression_judge_fake`: 81
- `winning_move_rate`: 0.1481
- `advanced_or_held_with_winning_move_rate_lt_0_5`: 6

Вердикт качества не выносился.

## Тесты

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_rejudge_progression.py tests/test_telegram_dynamic_client_sim.py::test_dynamic_summary_includes_llm_call_counts tests/test_telegram_dynamic_client_sim.py::test_summary_includes_judge_fact_audit_counts
21 passed in 0.86s
```

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q
3618 passed, 5 skipped, 1 warning in 86.53s
```

## Остаточный риск

- Smoke выполнен в fake-режиме, это проверка формата и детерминированной логики, а не качества живой модели.
- Заход 2 / M1 не запускался.
- Live/CRM/Tallanto/stable_runtime не трогались.
