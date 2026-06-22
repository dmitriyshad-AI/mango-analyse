# Progression Judge Report

Дата: 2026-06-23
Ветка: `codex/progression-judge`
База: `main a9f80ba`

## Результат

Сделан офлайн-судья продвижения сделки:

- модель размечает только 10 булевых наблюдений по ходу бота;
- стадия, вердикт и ошибки бизнеса считаются кодом;
- есть выход `progression_results.jsonl` и summary;
- есть режим повторного расчёта по сохранённым наблюдениям.

## Файлы

- `scripts/rejudge_progression.py`
- `tests/test_rejudge_progression.py`
- `audits/_inbox/progression_judge_20260623/`

Локальные результаты прогона:

- `runs/20260623_progression_seed/dynamic_dialog_transcripts.jsonl`
- `runs/20260623_progression_seed/progression_results.jsonl`
- `runs/20260623_progression_seed/progression_summary.json`

## Проверки

- `tests/test_rejudge_progression.py`: 9 passed.
- `py_compile`: passed.
- Seed-прогон симулятора: 11 диалогов, 42 хода, completed=11.
- Progression rejudge: 11 результатов.

## Сводка progression

- `advanced`: 1
- `held_ok`: 1
- `mis_routed`: 8
- `false_push`: 1

Главные бизнес-сигналы:

- `over_handoff_service`: 8
- `stage_carried_to_sibling`: 1
- `under_handoff_service`: 1

## Важно для регрейда Claude

Это не финальный вердикт качества бота. Нужно проверить сырьё:

- правильно ли LLM-наблюдения соответствуют тексту каждого хода;
- не слишком ли широкая карта `confirmed_access_or_docs -> S8`;
- корректно ли `over_handoff_service` отделяет настоящий лишний уход от безопасного черновика.
