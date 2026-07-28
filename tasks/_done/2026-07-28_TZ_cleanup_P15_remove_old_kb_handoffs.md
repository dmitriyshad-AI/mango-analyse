> DONE 2026-07-28 23:40 | ветка main | codex

> TAKE 2026-07-28 23:33 | ветка main | codex

Ветка: main
Зоны: product_data/knowledge_base/kb_release_20260602_v6_4_schedule_handoff_for_claude_and_team/, product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_handoff_for_claude_and_team/, product_data/knowledge_base/kb_release_20260608_v6_6_staging_handoff_for_claude_and_team/, product_data/knowledge_base/kb_release_20260610_v6_7_staging_handoff_for_claude_and_team/, product_data/knowledge_base/kb_release_20260610_v6_7_staging_r2_handoff_for_claude_and_team/, product_data/knowledge_base/kb_release_20260610_v6_7_staging_r3_handoff_for_claude_and_team/, product_data/knowledge_base/kb_release_20260611_v6_7_staging_r4_handoff_for_claude_and_team/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_kb_release_v3_compact_contract.py tests/test_kb_fact_freshness_sla.py tests/test_kb_r4_1_owner_gap_answers.py tests/test_kb_price_axes_catalog.py tests/test_fact_venue_scope.py
Семантический-аудит: да

# P15: удалить семь устаревших handoff-комплектов базы знаний

## Цель

Удалить семь комплектов передачи Claude для выпусков v6.4–v6.7 r4,
полностью заменённых каноническим r4.1.

## Доказательства

1. Ноль ссылок из `src/`, `scripts/`, `tests/`, `deploy/` и шести канонических документов.
2. 9–11 файлов каждого комплекта побайтово совпадают с обычным каталогом того же выпуска.
3. Текущий Wappi, README, ARCHITECTURE и RUNBOOK указывают на
   `kb_release_20260612_v6_7_staging_r4_1`, а не на handoff-папки.
4. Все уникальные исторические записки восстанавливаются из Git.

## Граница

- Не трогать канонический r4.1, sources, bot-pack и его текущий handoff.
- Текущий r4.1 handoff остаётся до перевыпуска смыслового отчёта с путём на канон.

## Приёмка

- Семь каталогов удалены напрямую без копий и `_attic`.
- SHA-256 канонического snapshot r4.1 не изменился.
- Тесты базы знаний, импорт провайдера и полный `pytest` зелёные.
- Смысловое содержание канонического r4.1 не изменено.

## СТОП

- Любая живая ссылка на удаляемый каталог.
- Изменение SHA канонического snapshot или красный тест.

## Результат

- Удалены 7 каталогов, 91 файл, 2 034 145 строк, 140,85 МиБ.
- Канонический snapshot r4.1 не изменён: SHA-256 `f99ea55c11b589f93976818e6918016a3ce86e7d1f7633aac616fe267df05bb3`.
- Точечные тесты: 55 passed; полный pytest: 5031 passed, 2 skipped.
- Добавлено строк нетестового кода: 0; удалено: 0 (удалены только данные старых handoff-комплектов).
- Новых файлов: 0; флагов: 0; зависимостей: 0.
- Более простой вариант с `_attic` отвергнут: он сохранил бы мусор в рабочем проекте.
