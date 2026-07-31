> FAIL 2026-07-31 11:51 | ветка codex/regex-to-understanding-map-20260731 | codex | причина: СТОП по ТЗ: actual marker-helper calls=172, budget ceiling=255; inline=215 в текущем snapshot-периметре, не 233; нужен канонический знаменатель и stable row_id до разметки.

> TAKE 2026-07-31 11:33 | ветка codex/regex-to-understanding-map-20260731 | codex

Ветка: codex/regex-to-understanding-map-20260731
Зоны: tests/test_adr003_regex_understanding_moratorium.py, tests/fixtures/, docs/adr003_understanding_map.yaml, docs/worktrees_registry.md, artifacts/, audits/_inbox/, tasks/
Тест-команда: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_regex_understanding_moratorium.py
Семантический-аудит: да

# Wrapper ТЗ D1: карта ADR-003, ведро 2 и ведро 3

Полное обязательное ТЗ:
`/Users/dmitrijfabarisov/Claude Projects/Foton/2026-07-31_TZ_200_regex_to_ponimanie_karta.md`.

Исполняется только половина Д1 из части 9: ведро 2 (верификация/пол безопасности)
и ведро 3 (формат/гигиена). Ведро 1 и ведро 4 принадлежат параллельному
исполнителю. На шагах 1-3 изменения `src/**` запрещены.

Занятые зоны из полного ТЗ не трогать. Обязательны отдельные результаты
субагентов `problems_breaker.md` и `review_notes.md`.

## Приёмка

- числа генератора независимо сверены: 197 и 832 записи, живой периметр 190;
- разметка ведра 2 и ведра 3 не оставляет принадлежащих Д1 записей без класса;
- ведро 2 содержит не менее 110 записей и подтверждено воспроизведением;
- генератор расширен, а бюджеты моратория не выросли;
- `src/**` не изменён;
- ломатель и ревьюер оставили независимые файлы.

## СТОП

- расхождение контрольных чисел до объяснения;
- любая правка защитного пола или `src/**`;
- пересечение с рабочими файлами параллельного исполнителя;
- вход в занятые зоны трека `memory_selection_quality_20260731`;
- рост бюджета моратория или пропуск P0 в воспроизведении.
